#include "dispatch/binary.h"

#include <sstream>
#include <stdexcept>

#include "cpu/kernels.h"

namespace deeptiny::dispatch::binary {

std::shared_ptr<TensorImpl> OutOfPlace(Op op,
                                       const std::shared_ptr<TensorImpl>& a,
                                       const std::shared_ptr<TensorImpl>& b) {
  auto out = std::make_shared<TensorImpl>(a->shape(), a->dtype(), a->device());
  switch (out->device()) {
    case Device::CPU:
      switch (op) {
        case Op::Add:
          cpu::Add(a, b, out);
          break;
        case Op::Sub:
          cpu::Sub(a, b, out);
          break;
        case Op::Mul:
          cpu::Mul(a, b, out);
          break;
        case Op::Div:
          cpu::Div(a, b, out);
          break;
      }
      return out;
    default: {
      std::stringstream err;
      err << "Operation does not support " << out->device().ToString();
      throw std::runtime_error(err.str());
    }
  }
}

void Inplace(Op op, const std::shared_ptr<TensorImpl>& self,
             const std::shared_ptr<TensorImpl>& other) {
  switch (self->device()) {
    case Device::CPU:
      switch (op) {
        case Op::Add:
          cpu::Add(self, other, self);
          break;
        case Op::Sub:
          cpu::Sub(self, other, self);
          break;
        case Op::Mul:
          cpu::Mul(self, other, self);
          break;
        case Op::Div:
          cpu::Div(self, other, self);
          break;
      }
      return;
    default: {
      std::stringstream err;
      err << "Operation does not support " << self->device().ToString();
      throw std::runtime_error(err.str());
    }
  }
}

}  // namespace deeptiny::dispatch::binary
