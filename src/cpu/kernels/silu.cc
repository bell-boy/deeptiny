#include <cmath>

#include "cpu/kernels.h"
#include "cpu/kernels/common.h"

namespace deeptiny::cpu {

namespace {

inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

void SiLU(std::shared_ptr<TensorImpl> x, std::shared_ptr<TensorImpl> out) {
  detail::ApplyElementwiseUnaryOp(
      x, out, "SiLU", [](float x_value) { return x_value * Sigmoid(x_value); });
}

void SiLUBackward(std::shared_ptr<TensorImpl> x,
                  std::shared_ptr<TensorImpl> grad_out,
                  std::shared_ptr<TensorImpl> grad_x) {
  detail::ApplyElementwiseUnaryGradOp(
      x, grad_out, grad_x, "SiLUBackward",
      [](float x_value, float grad_out_value) {
        const float sigmoid = Sigmoid(x_value);
        return grad_out_value *
               (sigmoid + x_value * sigmoid * (1.0f - sigmoid));
      });
}

}  // namespace deeptiny::cpu
