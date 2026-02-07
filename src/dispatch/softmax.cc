#include "dispatch/softmax.h"

#include <sstream>
#include <stdexcept>

#include "cpu/kernels.h"

namespace deeptiny::dispatch::softmax {

std::shared_ptr<TensorImpl> OutOfPlace(const std::shared_ptr<TensorImpl>& x,
                                       uint64_t dim) {
  auto out = std::make_shared<TensorImpl>(x->shape(), x->dtype(), x->device());
  switch (out->device()) {
    case Device::CPU:
      cpu::Softmax(x, out, dim);
      return out;
    default: {
      std::stringstream err;
      err << "Softmax does not support " << out->device().ToString();
      throw std::runtime_error(err.str());
    }
  }
}

std::shared_ptr<TensorImpl> Backward(
    const std::shared_ptr<TensorImpl>& y,
    const std::shared_ptr<TensorImpl>& grad_out, uint64_t dim) {
  auto grad_x =
      std::make_shared<TensorImpl>(y->shape(), y->dtype(), y->device());
  switch (grad_x->device()) {
    case Device::CPU:
      cpu::SoftmaxBackward(y, grad_out, grad_x, dim);
      return grad_x;
    default: {
      std::stringstream err;
      err << "Softmax backward does not support "
          << grad_x->device().ToString();
      throw std::runtime_error(err.str());
    }
  }
}

}  // namespace deeptiny::dispatch::softmax
