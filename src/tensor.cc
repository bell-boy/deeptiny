#include "deeptiny/tensor.h"

#include <sys/resource.h>

#include <iostream>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "autograd_meta.h"
#include "cpu/kernels.h"
#include "deeptiny/view.h"
#include "engine.h"
#include "tensor_impl.h"
#include "utils.h"

namespace deeptiny {

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
  if (!requires_grad()) {
    throw std::runtime_error("Cannot call Backward on tensor without grad");
  }
  if (!shape().empty()) {
    throw std::runtime_error("Backward requires a scalar (empty shape) tensor");
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

  Engine engine(autograd_meta_, keep_graph);
  autograd_meta_->updateGrad(grad, engine);
  engine.Run();
}

Tensor Tensor::FromBuffer(std::span<std::byte> bytes, Shape shape, DType dtype,
                          Device device, bool requires_grad) {
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
  return Tensor(result, std::make_shared<AutogradMeta>(nullptr, requires_grad));
}

View Tensor::operator()(std::vector<Slice> slices) {
  auto view_impl = tensor_impl_->View(slices);
  auto backward = std::make_shared<SliceBackward>(*this, std::move(slices));
  auto grad_meta = std::make_shared<AutogradMeta>(backward);
  return View(std::move(view_impl), grad_meta);
}

const View Tensor::operator()(std::vector<Slice> slices) const {
  auto view_impl = tensor_impl_->View(slices);
  auto backward = std::make_shared<SliceBackward>(*this, std::move(slices));
  auto grad_meta = std::make_shared<AutogradMeta>(backward);
  return (const View)View(std::move(view_impl), grad_meta);
}

};  // namespace deeptiny
