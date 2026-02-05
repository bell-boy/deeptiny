#include "deeptiny/tensor.h"

#include <sys/resource.h>

#include <iostream>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "autograd_meta.h"
#include "cpu/kernels.h"
#include "deeptiny/autograd.h"
#include "deeptiny/view.h"
#include "engine.h"
#include "tensor_impl.h"
#include "utils.h"
#include "view_backward.h"

namespace deeptiny {

namespace {
class SqueezeBackward : public Function {
 private:
  Shape original_shape_;
  std::vector<uint8_t> squeeze_mask_;

 public:
  SqueezeBackward(const Tensor& t, std::vector<uint8_t> squeeze_mask)
      : Function({utils::TensorAccessor::GetAutogradMeta(t)}),
        original_shape_(t.shape()),
        squeeze_mask_(std::move(squeeze_mask)) {}

  void operator()(const Tensor& grad, Engine& engine) override {
    auto parent = getParents()[0];
    if (!parent) {
      return;
    }

    const auto& grad_shape = grad.shape();
    const size_t out_rank = grad_shape.size();
    const size_t in_rank = original_shape_.size();
    size_t squeezed = 0;
    for (uint8_t v : squeeze_mask_) {
      if (v != 0) {
        squeezed += 1;
      }
    }
    if (out_rank + squeezed != in_rank) {
      throw std::runtime_error("SqueezeBackward received invalid grad shape");
    }

    auto grad_impl = utils::TensorAccessor::GetTensorImpl(grad);
    Stride new_stride;
    new_stride.reserve(in_rank);
    size_t out_dim = 0;
    for (size_t i = 0; i < in_rank; ++i) {
      if (squeeze_mask_[i]) {
        new_stride.push_back(0);
      } else {
        new_stride.push_back(grad_impl->stride()[out_dim]);
        out_dim += 1;
      }
    }

    auto view_impl = grad_impl->View(
        Shape(original_shape_), std::move(new_stride), grad_impl->offset());
    auto grad_in = utils::TensorAccessor::MakeTensor(view_impl, nullptr);
    parent->updateGrad(grad_in, engine);
  }
};
}  // namespace

Tensor::Tensor(std::shared_ptr<TensorImpl> tensor_impl,
               std::shared_ptr<AutogradMeta> autograd_meta)
    : tensor_impl_(tensor_impl), autograd_meta_(autograd_meta) {}

Tensor::Tensor(Shape shape, DType dtype, Device device, bool requires_grad)
    : tensor_impl_(std::make_shared<TensorImpl>(shape, dtype, device)),
      autograd_meta_(std::make_shared<AutogradMeta>(nullptr, requires_grad)) {}

Shape Tensor::shape() const { return tensor_impl_->shape(); }
DType Tensor::dtype() const { return tensor_impl_->dtype(); }
Device Tensor::device() const { return tensor_impl_->device(); }

Tensor Tensor::Clone() const {
  Tensor result(shape(), dtype(), device(), false);
  auto src_impl = utils::TensorAccessor::GetTensorImpl(*this);
  auto dst_impl = utils::TensorAccessor::GetTensorImpl(result);
  auto src_storage = src_impl->getContiguousStorage();
  const uint64_t numel = src_storage->numel();
  if (numel == 0) {
    return result;
  }
  std::vector<std::byte> host_buf(numel * dtype().size());
  src_storage->CopyToHost(0, numel, host_buf.data());
  dst_impl->storage()->CopyFromHost(0, numel, host_buf.data());
  return result;
}

Tensor Tensor::Squeeze(std::initializer_list<uint64_t> dims) {
  return Squeeze(std::vector<uint64_t>(dims));
}

Tensor Tensor::Squeeze(const std::vector<uint64_t>& dims) {
  const auto& shape = tensor_impl_->shape();
  const auto& stride = tensor_impl_->stride();
  const size_t rank = shape.size();
  if (dims.empty() || rank == 0) {
    return *this;
  }

  std::vector<uint8_t> squeeze_mask(rank, 0);
  for (uint64_t dim : dims) {
    if (dim >= rank) {
      throw std::runtime_error("Squeeze dim out of range");
    }
    if (squeeze_mask[dim] != 0) {
      throw std::runtime_error("Squeeze dims must be unique");
    }
    if (shape[dim] != 1) {
      throw std::runtime_error("Squeeze dim must be size 1");
    }
    squeeze_mask[dim] = 1;
  }

  Shape new_shape;
  Stride new_stride;
  new_shape.reserve(rank);
  new_stride.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    if (squeeze_mask[i] == 0) {
      new_shape.push_back(shape[i]);
      new_stride.push_back(stride[i]);
    }
  }

  if (new_shape.size() == rank) {
    return *this;
  }

  auto view_impl = tensor_impl_->View(Shape(new_shape), std::move(new_stride),
                                      tensor_impl_->offset());
  auto backward =
      std::make_shared<SqueezeBackward>(*this, std::move(squeeze_mask));
  auto grad_meta = std::make_shared<AutogradMeta>(backward);
  return utils::TensorAccessor::MakeTensor(std::move(view_impl), grad_meta);
}

bool Tensor::requires_grad() const {
  if (!autograd_meta_) {
    return false;
  }
  return autograd_meta_->requires_grad();
}

std::optional<Tensor> Tensor::grad() const {
  if (!autograd_meta_) {
    return std::nullopt;
  }
  return autograd_meta_->grad();
}

void Tensor::Backward(bool keep_graph) {
  if (!autograd_meta_) {
    throw std::runtime_error("Tensor has no autograd metadata");
  }
  if (!shape().empty()) {
    throw std::runtime_error("Backward requires a scalar (empty shape) tensor");
  }

  Engine engine(autograd_meta_, keep_graph);
  if (!autograd_meta_->requires_grad()) {
    throw std::runtime_error("Cannot call Backward on tensor without grad");
  }

  Tensor grad({}, dtype(), device(), false);
  auto grad_impl = utils::TensorAccessor::GetTensorImpl(grad);
  switch (dtype()) {
    case DType::Float32: {
      auto* grad_data = static_cast<float*>(grad_impl->data());
      grad_data[0] = 1.0f;
      break;
    }
    default:
      throw std::runtime_error("Backward only supports Float32 gradients");
  }

  if (!keep_graph) {
    GradMode guard(false);
    autograd_meta_->updateGrad(grad, engine);
    engine.Run();
    return;
  }

  autograd_meta_->updateGrad(grad, engine);
  engine.Run();
}

Tensor Tensor::FromBuffer(std::span<const std::byte> bytes, Shape shape,
                          DType dtype, Device device, bool requires_grad) {
  std::shared_ptr<TensorImpl> result;
  switch (device) {
    case Device::CPU:
      result = cpu::FromBuffer(dtype, bytes, shape);
      break;
    case Device::Metal:
      throw std::runtime_error(
          "FromBuffer doesn't support Metal at the moment");
      break;
  }
  return utils::TensorAccessor::MakeTensor(
      result, std::make_shared<AutogradMeta>(nullptr, requires_grad));
}

View Tensor::operator()(std::vector<Slice> slices) {
  auto view_impl = tensor_impl_->View(slices);
  auto backward = std::make_shared<SliceBackward>(*this, std::move(slices));
  auto grad_meta = std::make_shared<AutogradMeta>(backward);
  return utils::TensorAccessor::MakeView(std::move(view_impl), grad_meta);
}

const View Tensor::operator()(std::vector<Slice> slices) const {
  auto view_impl = tensor_impl_->View(slices);
  auto backward = std::make_shared<SliceBackward>(*this, std::move(slices));
  auto grad_meta = std::make_shared<AutogradMeta>(backward);
  return (const View)utils::TensorAccessor::MakeView(std::move(view_impl),
                                                     grad_meta);
}

};  // namespace deeptiny
