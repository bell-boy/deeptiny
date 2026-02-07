#include "cpu/kernels.h"

#include "cpu/kernels/common.h"

namespace deeptiny::cpu {

void ReLU(std::shared_ptr<TensorImpl> x, std::shared_ptr<TensorImpl> out) {
  detail::ApplyElementwiseUnaryOp(x, out, "ReLU",
                                  [](float x_value) {
                                    return x_value > 0.0f ? x_value : 0.0f;
                                  });
}

void ReLUBackward(std::shared_ptr<TensorImpl> x,
                  std::shared_ptr<TensorImpl> grad_out,
                  std::shared_ptr<TensorImpl> grad_x) {
  detail::ApplyElementwiseUnaryGradOp(
      x, grad_out, grad_x, "ReLUBackward",
      [](float x_value, float grad_out_value) {
        return x_value > 0.0f ? grad_out_value : 0.0f;
      });
}

}  // namespace deeptiny::cpu
