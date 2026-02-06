#include "modules/linear.h"

#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "deeptiny/math.h"

namespace module {

namespace {

uint64_t Numel(const deeptiny::Shape& shape) {
  uint64_t total = 1;
  for (const uint64_t dim : shape) {
    total *= dim;
  }
  return total;
}

uint64_t ValidatePositiveDimension(uint64_t dim, const char* name) {
  if (dim == 0) {
    throw std::runtime_error(std::string("Linear ") + name + " must be > 0");
  }
  return dim;
}

deeptiny::Tensor MakeUniformParameter(const deeptiny::Shape& shape,
                                      deeptiny::Device device) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  const uint64_t total = Numel(shape);
  std::vector<float> values(static_cast<size_t>(total), 0.0f);
  for (uint64_t i = 0; i < total; ++i) {
    values[static_cast<size_t>(i)] = dist(gen);
  }
  return deeptiny::Tensor::FromVector(values, shape, device, true);
}

}  // namespace

Linear::Linear(uint64_t in_dim, uint64_t out_dim, bool bias,
               deeptiny::Device device)
    : in_dim_(ValidatePositiveDimension(in_dim, "in_dim")),
      out_dim_(ValidatePositiveDimension(out_dim, "out_dim")),
      weight_(MakeUniformParameter({1, in_dim_, out_dim_}, device)) {
  if (bias) {
    bias_ = MakeUniformParameter({1, 1, out_dim_}, device);
  }
}

deeptiny::Tensor Linear::operator()(const deeptiny::Tensor& x) const {
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

  deeptiny::Tensor x_view = x;
  deeptiny::Tensor x_2d = x_view.Reshape({leading_size, in_dim_});
  deeptiny::Tensor x_3d = x_2d.Reshape({1, leading_size, in_dim_});

  deeptiny::Tensor out = deeptiny::math::BatchedMatMul(x_3d, weight_);
  if (bias_.has_value()) {
    out = out + *bias_;
  }

  deeptiny::Shape output_shape(input_shape.begin(), input_shape.end() - 1);
  output_shape.push_back(out_dim_);
  return out.Reshape(output_shape);
}

deeptiny::Tensor& Linear::weight() { return weight_; }

const deeptiny::Tensor& Linear::weight() const { return weight_; }

std::optional<deeptiny::Tensor>& Linear::bias() { return bias_; }

const std::optional<deeptiny::Tensor>& Linear::bias() const { return bias_; }

}  // namespace module
