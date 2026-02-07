#include "cpu/kernels.h"

#include "cpu/kernels/common.h"

namespace deeptiny::cpu {

void Add(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b,
         std::shared_ptr<TensorImpl> out) {
  detail::ApplyElementwiseBinaryOp(a, b, out, "Add",
                                   [](float x, float y) { return x + y; });
}

void Sub(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b,
         std::shared_ptr<TensorImpl> out) {
  detail::ApplyElementwiseBinaryOp(a, b, out, "Sub",
                                   [](float x, float y) { return x - y; });
}

void Mul(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b,
         std::shared_ptr<TensorImpl> out) {
  detail::ApplyElementwiseBinaryOp(a, b, out, "Mul",
                                   [](float x, float y) { return x * y; });
}

void Div(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b,
         std::shared_ptr<TensorImpl> out) {
  detail::ApplyElementwiseBinaryOp(a, b, out, "Div",
                                   [](float x, float y) { return x / y; });
}

}  // namespace deeptiny::cpu
