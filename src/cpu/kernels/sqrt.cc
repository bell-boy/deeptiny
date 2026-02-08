#include <cmath>

#include "cpu/kernels.h"
#include "cpu/kernels/common.h"

namespace deeptiny::cpu {

void Sqrt(std::shared_ptr<TensorImpl> x, std::shared_ptr<TensorImpl> out) {
  detail::ApplyElementwiseUnaryOp(
      x, out, "Sqrt", [](float x_value) { return std::sqrt(x_value); });
}

void SqrtBackward(std::shared_ptr<TensorImpl> x,
                  std::shared_ptr<TensorImpl> grad_out,
                  std::shared_ptr<TensorImpl> grad_x) {
  detail::ApplyElementwiseUnaryGradOp(x, grad_out, grad_x, "SqrtBackward",
                                      [](float x_value, float grad_out_value) {
                                        return grad_out_value * 0.5f /
                                               std::sqrt(x_value);
                                      });
}

}  // namespace deeptiny::cpu
