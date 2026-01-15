#include "deeptiny/tensor.h"

#include <sys/resource.h>

#include <iostream>
#include <utility>

#include "autograd_meta.h"
#include "cpu/kernels.h"
#include "deeptiny/view.h"
#include "tensor_impl.h"

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
