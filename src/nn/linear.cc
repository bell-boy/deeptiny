#include "deeptiny/nn/linear.h"

#include <cstddef>
#include <stdexcept>

#include "deeptiny/math.h"
#include "nn/validation.h"

namespace deeptiny::nn {

Linear::Linear(uint64_t in_dim, uint64_t out_dim, bool bias, Device device)
    : in_dim_(detail::ValidateNonZeroDimension("Linear", "in_dim", in_dim)),
      out_dim_(detail::ValidateNonZeroDimension("Linear", "out_dim", out_dim)),
      weight_(Tensor::CreateUniform({1, in_dim_, out_dim_}, device,
                                    DType::Float32, true)) {
  RegisterParameter(weight_);
  if (bias) {
    bias_ =
        Tensor::CreateUniform({1, 1, out_dim_}, device, DType::Float32, true);
    RegisterParameter(*bias_);
  }
}

Tensor Linear::operator()(const Tensor& x) const {
  const auto& input_shape = x.shape();
  if (input_shape.size() < 2) {
    throw std::runtime_error("Linear expects input rank >= 2");
  }
  if (input_shape.back() != in_dim_) {
    throw std::runtime_error("Linear input feature dimension mismatch");
  }

  uint64_t leading_size = 1;
  for (size_t i = 0; i + 1 < input_shape.size(); ++i) {
    leading_size *= input_shape[i];
  }

  Tensor x_view = x;
  Tensor x_2d = x_view.Reshape({leading_size, in_dim_});
  Tensor x_3d = x_2d.Reshape({1, leading_size, in_dim_});

  Tensor out = math::BatchedMatMul(x_3d, weight_);
  if (bias_.has_value()) {
    out = out + *bias_;
  }

  Shape output_shape(input_shape.begin(), input_shape.end() - 1);
  output_shape.push_back(out_dim_);
  return out.Reshape(output_shape);
}

Tensor Linear::weight() const { return weight_; }

std::optional<Tensor> Linear::bias() const { return bias_; }

Tensor Linear::weight() { return weight_; }

std::optional<Tensor> Linear::bias() { return bias_; }

}  // namespace deeptiny::nn
