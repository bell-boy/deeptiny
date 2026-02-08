#include "deeptiny/nn/rms_norm.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "deeptiny/functional.h"
#include "deeptiny/math.h"
#include "utils.h"

namespace deeptiny::nn {
namespace {

uint64_t ValidateHiddenDim(uint64_t hidden_dim) {
  if (hidden_dim == 0) {
    throw std::runtime_error("RMSNorm hidden_dim must be > 0");
  }
  return hidden_dim;
}

float ValidateEpsilon(float epsilon) {
  if (epsilon <= 0.0f) {
    throw std::runtime_error("RMSNorm epsilon must be > 0");
  }
  return epsilon;
}

Tensor Full(const Shape& shape, float value, Device device) {
  return Tensor::FromVector(std::vector<float>(static_cast<size_t>(
                                utils::GetTotalSize(shape)),
                            value),
                            shape, device, false);
}

Tensor ReciprocalSqrtNewton(const Tensor& x) {
  const Shape shape = x.shape();
  const Device device = x.device();

  Tensor inv = Full(shape, 1.0f, device);
  Tensor half = Full(shape, 0.5f, device);
  Tensor three = Full(shape, 3.0f, device);
  for (int i = 0; i < 4; ++i) {
    inv = half * inv * (three - x * inv * inv);
  }
  return inv;
}

}  // namespace

RMSNorm::RMSNorm(uint64_t hidden_dim, float epsilon, Device device)
    : hidden_dim_(ValidateHiddenDim(hidden_dim)),
      epsilon_(ValidateEpsilon(epsilon)),
      weight_(Tensor::FromVector(std::vector<float>(hidden_dim_, 1.0f),
                                 Shape{hidden_dim_}, device, true)) {
  RegisterParameter(weight_);
}

Tensor RMSNorm::operator()(const Tensor& x) const {
  const Shape& input_shape = x.shape();
  if (input_shape.empty()) {
    throw std::runtime_error("RMSNorm expects input rank >= 1");
  }
  if (input_shape.back() != hidden_dim_) {
    throw std::runtime_error("RMSNorm input feature dimension mismatch");
  }

  const uint64_t reduce_dim = static_cast<uint64_t>(input_shape.size() - 1);
  Tensor sq = x * x;
  Tensor sum_sq = functional::Reduce(sq, {reduce_dim}, true);

  Shape stat_shape = sum_sq.shape();
  Tensor hidden = Full(stat_shape, static_cast<float>(hidden_dim_), x.device());
  Tensor eps = Full(stat_shape, epsilon_, x.device());

  Tensor mean_sq = sum_sq / hidden;
  Tensor inv_rms = ReciprocalSqrtNewton(mean_sq + eps);

  Shape scale_shape(input_shape.size(), 1);
  scale_shape.back() = hidden_dim_;
  Tensor scale = weight_;
  scale = scale.Reshape(scale_shape);

  return x * inv_rms * scale;
}

Tensor& RMSNorm::weight() { return weight_; }

const Tensor& RMSNorm::weight() const { return weight_; }

float RMSNorm::epsilon() const { return epsilon_; }

}  // namespace deeptiny::nn
