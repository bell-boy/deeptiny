#include "deeptiny/tensor.h"

#include <sys/resource.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "autograd_meta.h"
#include "cpu/kernels.h"
#include "deeptiny/autograd.h"
#include "engine.h"
#include "tensor_impl.h"
#include "utils.h"

namespace deeptiny {

namespace {
class ReshapeBackward : public Function {
 private:
  Shape original_shape_;

 public:
  explicit ReshapeBackward(const Tensor& t)
      : Function({utils::TensorAccessor::GetAutogradMeta(t)}),
        original_shape_(t.shape()) {}

  void operator()(const Tensor& grad) override {
    const auto& parents = getParents();
    assert(parents.size() == 1 && "ReshapeBackward must have exactly 1 parent");
    assert(parents[0] && "ReshapeBackward parent must not be null");

    const uint64_t expected_size = utils::GetTotalSize(original_shape_);
    const uint64_t grad_size = utils::GetTotalSize(grad.shape());
    assert(grad_size == expected_size &&
           "ReshapeBackward received invalid grad shape");

    Tensor grad_view = grad;
    parents[0]->updateGrad(grad_view.Reshape(original_shape_));
  }
};
std::random_device uniform_rd;
std::mt19937 uniform_gen(uniform_rd());
std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

class SqueezeBackward : public Function {
 private:
  Shape original_shape_;
  std::vector<uint8_t> squeeze_mask_;

 public:
  SqueezeBackward(const Tensor& t, std::vector<uint8_t> squeeze_mask)
      : Function({utils::TensorAccessor::GetAutogradMeta(t)}),
        original_shape_(t.shape()),
        squeeze_mask_(std::move(squeeze_mask)) {}

  void operator()(const Tensor& grad) override {
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
    parent->updateGrad(grad_in);
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

Tensor Tensor::Reshape(const Shape& shape) {
  const uint64_t current_size = utils::GetTotalSize(tensor_impl_->shape());
  const uint64_t target_size = utils::GetTotalSize(shape);
  if (current_size != target_size) {
    throw std::runtime_error(
        "Reshape requires tensors with the same number of elements");
  }
  if (!tensor_impl_->isContiguous()) {
    throw std::runtime_error("Reshape requires contiguous input tensor");
  }
  if (shape == tensor_impl_->shape()) {
    return *this;
  }

  auto reshaped_impl = tensor_impl_->View(
      Shape(shape), utils::GetContinguousStride(shape), tensor_impl_->offset());
  auto backward = std::make_shared<ReshapeBackward>(*this);
  auto grad_meta = std::make_shared<AutogradMeta>(backward);
  return utils::TensorAccessor::MakeTensor(reshaped_impl, grad_meta);
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
    autograd_meta_->updateGrad(grad);
    engine.Run();
    return;
  }

  autograd_meta_->updateGrad(grad);
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

Tensor Tensor::CreateUniform(Shape shape, Device device, DType dtype,
                             bool requires_grad) {
  switch (dtype) {
    case DType::Float32: {
      const size_t total_size = utils::GetTotalSize(shape);
      std::vector<float> values(total_size, 0.0f);
      for (size_t i = 0; i < total_size; ++i) {
        values[i] = static_cast<float>(uniform_dist(uniform_gen));
      }
      return Tensor::FromVector(values, std::move(shape), device,
                                requires_grad);
    }
    default:
      throw std::runtime_error("DType is not supported yet");
  };
}

Tensor Tensor::Zeros(Shape shape, Device device, DType dtype,
                     bool requires_grad) {
  switch (dtype) {
    case DType::Float32: {
      Tensor result(shape, DType::Float32, device, requires_grad);
      auto impl = utils::TensorAccessor::GetTensorImpl(result);
      const size_t total_size = utils::GetTotalSize(shape);
      auto* data = static_cast<float*>(impl->data());
      std::fill_n(data, total_size, 0.0f);
      return result;
    }
    default:
      throw std::runtime_error("DType is not supported yet");
  };
}

TensorSliceProxy Tensor::operator()(std::vector<Slice> slices) {
  return TensorSliceProxy(this, std::move(slices));
}

Tensor Tensor::operator()(std::vector<Slice> slices) const {
  return TensorSliceProxy(this, std::move(slices));
}

};  // namespace deeptiny
