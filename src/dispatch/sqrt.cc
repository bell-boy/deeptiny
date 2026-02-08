#include "dispatch/sqrt.h"

#include <sstream>
#include <stdexcept>

#include "cpu/kernels.h"

namespace deeptiny::dispatch::sqrt {

std::shared_ptr<TensorImpl> OutOfPlace(const std::shared_ptr<TensorImpl>& x) {
  auto out = std::make_shared<TensorImpl>(x->shape(), x->dtype(), x->device());
  switch (out->device()) {
    case Device::CPU:
      cpu::Sqrt(x, out);
      return out;
    default: {
      std::stringstream err;
      err << "Sqrt does not support " << out->device().ToString();
      throw std::runtime_error(err.str());
    }
  }
}

std::shared_ptr<TensorImpl> Backward(const std::shared_ptr<TensorImpl>& x,
                                     const std::shared_ptr<TensorImpl>& grad) {
  auto grad_x =
      std::make_shared<TensorImpl>(x->shape(), x->dtype(), x->device());
  switch (grad_x->device()) {
    case Device::CPU:
      cpu::SqrtBackward(x, grad, grad_x);
      return grad_x;
    default: {
      std::stringstream err;
      err << "Sqrt backward does not support " << grad_x->device().ToString();
      throw std::runtime_error(err.str());
    }
  }
}

}  // namespace deeptiny::dispatch::sqrt
