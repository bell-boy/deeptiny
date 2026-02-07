#include "deeptiny/tensor_slice_proxy.h"

#include <cassert>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "autograd_meta.h"
#include "dispatch/dispatch.h"
#include "broadcast.h"
#include "deeptiny/math.h"
#include "deeptiny/tensor.h"
#include "engine.h"
#include "tensor_impl.h"
#include "utils.h"

namespace deeptiny {

namespace {

std::shared_ptr<AutogradMeta> EnsureMeta(const Tensor& t) {
  auto meta = utils::TensorAccessor::GetAutogradMeta(t);
  if (meta) {
    return meta;
  }
  return std::make_shared<AutogradMeta>(nullptr, false);
}

class SliceReadBackward : public Function {
 public:
  SliceReadBackward(std::shared_ptr<AutogradMeta> parent, Shape original_shape,
                    std::vector<Slice> slices)
      : Function({std::move(parent)}),
        slices_(std::move(slices)),
        original_shape_(std::move(original_shape)) {}

  void operator()(const Tensor& grad) override {
    auto scattered = utils::SliceScatterToShape(grad, original_shape_, slices_);
    if (!scattered) {
      throw std::runtime_error(
          "SliceReadBackward failed to scatter gradient to source shape.");
    }
    const auto& parents = getParents();
    assert(parents.size() == 1 &&
           "SliceReadBackward must have exactly one parent");
    assert(parents[0] && "SliceReadBackward parent must not be null");
    parents[0]->updateGrad(*scattered);
  }

 private:
  std::vector<Slice> slices_;
  Shape original_shape_;
};

class SliceAssignBackward : public Function {
 public:
  SliceAssignBackward(std::shared_ptr<AutogradMeta> base_parent,
                      std::shared_ptr<AutogradMeta> rhs_parent,
                      Shape original_shape, std::vector<Slice> slices)
      : Function({std::move(base_parent), std::move(rhs_parent)}),
        original_shape_(std::move(original_shape)),
        slices_(std::move(slices)) {}

  void operator()(const Tensor& grad) override {
    auto grad_rhs_impl =
        utils::TensorAccessor::GetTensorImpl(grad)->View(slices_);
    Tensor grad_rhs =
        utils::TensorAccessor::MakeTensor(std::move(grad_rhs_impl), nullptr);

    auto scattered =
        utils::SliceScatterToShape(grad_rhs, original_shape_, slices_);
    if (!scattered) {
      throw std::runtime_error(
          "SliceAssignBackward failed to scatter RHS gradient.");
    }

    const auto& parents = getParents();
    assert(parents.size() == 2 &&
           "SliceAssignBackward must have exactly two parents");
    assert(parents[0] && "SliceAssignBackward base parent must not be null");
    assert(parents[1] && "SliceAssignBackward RHS parent must not be null");

    parents[1]->updateGrad(grad_rhs);
    auto grad_base_impl =
        dispatch::binary::OutOfPlace(dispatch::binary::Op::Sub, grad, *scattered);
    parents[0]->updateGrad(grad_base_impl);
  }

 private:
  Shape original_shape_;
  std::vector<Slice> slices_;
};

void CopyTensorImplToSlice(const std::shared_ptr<const TensorImpl>& src_impl,
                           const std::shared_ptr<TensorImpl>& dst_impl,
                           DType dtype) {
  const Shape dst_shape = dst_impl->shape();
  const Stride src_stride = src_impl->stride();
  const Stride dst_stride = dst_impl->stride();

  for (const auto stride : dst_stride) {
    if (stride == 0) {
      throw std::runtime_error(
          "Cannot assign to a slice with zero stride (broadcasted view).");
    }
  }

  std::vector<std::byte> buf(dtype.size());
  auto Transfer = [src_impl, dst_impl, &buf](uint64_t src_offset,
                                             uint64_t dst_offset) {
    src_impl->storage()->CopyToHost(src_offset, 1, buf.data());
    dst_impl->storage()->CopyFromHost(dst_offset, 1, buf.data());
  };

  if (dst_shape.empty()) {
    Transfer(src_impl->offset(), dst_impl->offset());
    return;
  }

  const size_t rank = dst_shape.size();
  auto AssignFastBlock = [&dst_shape, &src_stride, &dst_stride, &Transfer](
                             size_t dim, size_t remaining, uint64_t src_base,
                             uint64_t dst_base) {
    for (uint64_t i0 = 0; i0 < dst_shape[dim]; ++i0) {
      const uint64_t src_offset_0 = src_base + i0 * src_stride[dim];
      const uint64_t dst_offset_0 = dst_base + i0 * dst_stride[dim];
      if (remaining == 1) {
        Transfer(src_offset_0, dst_offset_0);
        continue;
      }
      for (uint64_t i1 = 0; i1 < dst_shape[dim + 1]; ++i1) {
        const uint64_t src_offset_1 = src_offset_0 + i1 * src_stride[dim + 1];
        const uint64_t dst_offset_1 = dst_offset_0 + i1 * dst_stride[dim + 1];
        if (remaining == 2) {
          Transfer(src_offset_1, dst_offset_1);
          continue;
        }
        for (uint64_t i2 = 0; i2 < dst_shape[dim + 2]; ++i2) {
          const uint64_t src_offset_2 = src_offset_1 + i2 * src_stride[dim + 2];
          const uint64_t dst_offset_2 = dst_offset_1 + i2 * dst_stride[dim + 2];
          if (remaining == 3) {
            Transfer(src_offset_2, dst_offset_2);
            continue;
          }
          for (uint64_t i3 = 0; i3 < dst_shape[dim + 3]; ++i3) {
            const uint64_t src_offset_3 =
                src_offset_2 + i3 * src_stride[dim + 3];
            const uint64_t dst_offset_3 =
                dst_offset_2 + i3 * dst_stride[dim + 3];
            if (remaining == 4) {
              Transfer(src_offset_3, dst_offset_3);
              continue;
            }
            for (uint64_t i4 = 0; i4 < dst_shape[dim + 4]; ++i4) {
              const uint64_t src_offset_4 =
                  src_offset_3 + i4 * src_stride[dim + 4];
              const uint64_t dst_offset_4 =
                  dst_offset_3 + i4 * dst_stride[dim + 4];
              Transfer(src_offset_4, dst_offset_4);
            }
          }
        }
      }
    }
  };

  auto AssignRecursive = [rank, &dst_shape, &src_stride, &dst_stride,
                          &AssignFastBlock](auto&& self, size_t dim,
                                            uint64_t src_base,
                                            uint64_t dst_base) -> void {
    const size_t remaining = rank - dim;
    if (remaining <= 5) {
      AssignFastBlock(dim, remaining, src_base, dst_base);
      return;
    }
    for (uint64_t i0 = 0; i0 < dst_shape[dim]; ++i0) {
      const uint64_t src_offset_0 = src_base + i0 * src_stride[dim];
      const uint64_t dst_offset_0 = dst_base + i0 * dst_stride[dim];
      for (uint64_t i1 = 0; i1 < dst_shape[dim + 1]; ++i1) {
        const uint64_t src_offset_1 = src_offset_0 + i1 * src_stride[dim + 1];
        const uint64_t dst_offset_1 = dst_offset_0 + i1 * dst_stride[dim + 1];
        for (uint64_t i2 = 0; i2 < dst_shape[dim + 2]; ++i2) {
          const uint64_t src_offset_2 = src_offset_1 + i2 * src_stride[dim + 2];
          const uint64_t dst_offset_2 = dst_offset_1 + i2 * dst_stride[dim + 2];
          for (uint64_t i3 = 0; i3 < dst_shape[dim + 3]; ++i3) {
            const uint64_t src_offset_3 =
                src_offset_2 + i3 * src_stride[dim + 3];
            const uint64_t dst_offset_3 =
                dst_offset_2 + i3 * dst_stride[dim + 3];
            for (uint64_t i4 = 0; i4 < dst_shape[dim + 4]; ++i4) {
              const uint64_t src_offset_4 =
                  src_offset_3 + i4 * src_stride[dim + 4];
              const uint64_t dst_offset_4 =
                  dst_offset_3 + i4 * dst_stride[dim + 4];
              self(self, dim + 5, src_offset_4, dst_offset_4);
            }
          }
        }
      }
    }
  };

  AssignRecursive(AssignRecursive, 0, src_impl->offset(), dst_impl->offset());
}

}  // namespace

TensorSliceProxy::TensorSliceProxy(Tensor* base, std::vector<Slice> slices)
    : mutable_base_(base), base_(base), slices_(std::move(slices)) {}

TensorSliceProxy::TensorSliceProxy(const Tensor* base,
                                   std::vector<Slice> slices)
    : mutable_base_(nullptr), base_(base), slices_(std::move(slices)) {}

TensorSliceProxy& TensorSliceProxy::operator=(const Tensor& rhs) {
  if (!mutable_base_) {
    throw std::runtime_error(
        "Cannot assign to a slice from a const TensorSliceProxy.");
  }

  auto slice_impl =
      utils::TensorAccessor::GetTensorImpl(*mutable_base_)->View(slices_);
  auto broadcasted_rhs = utils::BroadcastToShape(rhs, slice_impl->shape());
  if (!broadcasted_rhs) {
    std::stringstream err;
    err << "Couldn't broadcast tensor of shape " << FormatShape(rhs.shape())
        << " to tensor of shape " << FormatShape(slice_impl->shape());
    throw std::runtime_error(err.str());
  }

  Tensor rhs_broadcasted = *broadcasted_rhs;
  if (rhs_broadcasted.dtype() != mutable_base_->dtype()) {
    std::stringstream err;
    err << "Trying to copy data of type " << rhs_broadcasted.dtype().ToString()
        << " to slice of type " << mutable_base_->dtype().ToString();
    throw std::runtime_error(err.str());
  }

  auto base_parent = EnsureMeta(*mutable_base_);
  auto rhs_parent = EnsureMeta(rhs_broadcasted);

  auto rhs_impl = utils::TensorAccessor::GetTensorImpl(rhs_broadcasted);
  CopyTensorImplToSlice(rhs_impl, slice_impl, mutable_base_->dtype());

  auto backward = std::make_shared<SliceAssignBackward>(
      std::move(base_parent), std::move(rhs_parent), mutable_base_->shape(),
      slices_);
  auto grad_meta = std::make_shared<AutogradMeta>(backward);
  mutable_base_->autograd_meta_ = std::move(grad_meta);
  return *this;
}

TensorSliceProxy::operator Tensor() const {
  if (!base_) {
    throw std::runtime_error(
        "Cannot materialize a slice without a base tensor");
  }

  auto base_impl = utils::TensorAccessor::GetTensorImpl(*base_);
  auto slice_impl = base_impl->View(slices_);
  auto backward = std::make_shared<SliceReadBackward>(EnsureMeta(*base_),
                                                      base_->shape(), slices_);
  auto grad_meta = std::make_shared<AutogradMeta>(backward);
  return utils::TensorAccessor::MakeTensor(std::move(slice_impl),
                                           std::move(grad_meta));
}

}  // namespace deeptiny
