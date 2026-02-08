#include "deeptiny/nn/rms_norm.h"

#include <stdexcept>
#include <vector>

#include "deeptiny/functional.h"
#include "deeptiny/math.h"
#include "nn/validation.h"

namespace deeptiny::nn {
namespace {

float ValidateEpsilon(float eps) {
  if (eps < 0.0f) {
    throw std::runtime_error("RMSNorm eps must be >= 0");
  }
  return eps;
}

}  // namespace

RMSNorm::RMSNorm(uint64_t dim, float eps, Device device)
    : dim_(detail::ValidateNonZeroDimension("RMSNorm", "dim", dim)),
      eps_(ValidateEpsilon(eps)),
      weight_(Tensor::FromVector(std::vector<float>(dim_, 1.0f), {dim_}, device,
                                 true)) {
  RegisterParameter(weight_);
}

Tensor RMSNorm::operator()(const Tensor& x) const {
  const auto& input_shape = x.shape();
  if (input_shape.empty()) {
    throw std::runtime_error("RMSNorm expects input rank >= 1");
  }
  if (input_shape.back() != dim_) {
    throw std::runtime_error("RMSNorm input feature dimension mismatch");
  }

  const uint64_t last_dim = static_cast<uint64_t>(input_shape.size() - 1);
  Tensor dim_tensor = Tensor::FromVector(
      std::vector<float>{static_cast<float>(dim_)}, x.device(), false);
  Tensor mean_square = functional::Reduce(x * x, {last_dim}, true) / dim_tensor;
  Tensor eps_tensor =
      Tensor::FromVector(std::vector<float>{eps_}, x.device(), false);
  Tensor rms = functional::Sqrt(mean_square + eps_tensor);
  return (x / rms) * weight_;
}

Tensor& RMSNorm::weight() { return weight_; }

const Tensor& RMSNorm::weight() const { return weight_; }

float RMSNorm::eps() const { return eps_; }

}  // namespace deeptiny::nn
