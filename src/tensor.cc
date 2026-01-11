#include "deeptiny/tensor.h"

#include <iostream>
#include <sstream>

#include "cpu/kernels.h"
#include "tensorImpl.h"
#include "utils.h"

namespace deeptiny {

Tensor::Tensor(std::shared_ptr<TensorImpl> tensor_impl)
    : tensor_impl_(tensor_impl) {}

Tensor::Tensor(Shape shape, DType dtype, Device device, bool requires_grad)
    : tensor_impl_(std::make_shared<TensorImpl>(shape, dtype, device)),
      requires_grad_(requires_grad) {}

View::View(std::shared_ptr<TensorImpl> tensor_impl) : Tensor(tensor_impl) {}

View Tensor::operator()(std::initializer_list<Slice> slices) {
  return View(tensor_impl_->View(slices));
}

Tensor Tensor::FromBuffer(std::span<std::byte> bytes, Shape shape, DType dtype,
                          Device device, bool requires_grad) {
  std::shared_ptr<TensorImpl> result;
  switch (device) {
    case Device::CPU:
      result = cpu::FromBuffer(dtype, bytes, shape);
      break;
    case Device::Metal:
      // TODO: remove this, get's around warnings until we have autogradMeta
      std::cout << requires_grad << std::endl;
      std::runtime_error("FromBuffer doesn't support Metal at the moment");
      break;
  }
  return Tensor(result);
}

};  // namespace deeptiny
