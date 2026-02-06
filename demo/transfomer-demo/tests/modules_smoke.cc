#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "deeptiny/functional.h"
#include "deeptiny/tensor.h"
#include "modules/linear.h"
#include "modules/llama_mlp.h"
#include "utils.h"

namespace {

uint64_t Numel(const deeptiny::Shape& shape) {
  uint64_t total = 1;
  for (uint64_t dim : shape) {
    total *= dim;
  }
  return total;
}

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void ExpectNear(float actual, float expected, float tol,
                const std::string& label) {
  if (std::fabs(actual - expected) > tol) {
    throw std::runtime_error(label + " expected " + std::to_string(expected) +
                             " got " + std::to_string(actual));
  }
}

deeptiny::Tensor MakeInput(const deeptiny::Shape& shape,
                           bool requires_grad = true) {
  const uint64_t total = Numel(shape);
  std::vector<float> values(static_cast<size_t>(total), 0.0f);
  for (uint64_t i = 0; i < total; ++i) {
    values[static_cast<size_t>(i)] = 0.01f * static_cast<float>(i + 1);
  }
  return deeptiny::Tensor::FromVector(values, shape, deeptiny::Device::CPU,
                                      requires_grad);
}

void ExpectShape(const deeptiny::Tensor& t, const deeptiny::Shape& expected,
                 const std::string& label) {
  if (t.shape() != expected) {
    throw std::runtime_error(label + " shape mismatch");
  }
}

std::vector<float> ToVector(const deeptiny::Tensor& t) {
  auto impl = deeptiny::utils::TensorAccessor::GetTensorImpl(t);
  auto contiguous = impl->getContiguousStorage();
  const uint64_t n = contiguous->numel();
  std::vector<float> out(static_cast<size_t>(n), 0.0f);
  contiguous->CopyToHost(0, n, out.data());
  return out;
}

void ExpectVectorNear(const deeptiny::Tensor& t, const std::vector<float>& exp,
                      float tol, const std::string& label) {
  const auto actual = ToVector(t);
  if (actual.size() != exp.size()) {
    throw std::runtime_error(label + " size mismatch");
  }
  for (size_t i = 0; i < exp.size(); ++i) {
    ExpectNear(actual[i], exp[i], tol, label + " at idx " + std::to_string(i));
  }
}

}  // namespace

int main() {
  try {
    {
      module::Linear linear(/*in_dim=*/4, /*out_dim=*/6);
      auto x = MakeInput({5, 4});
      auto y = linear(x);
      ExpectShape(y, {5, 6}, "Linear rank-2 forward");
    }

    {
      module::Linear linear(/*in_dim=*/4, /*out_dim=*/6);
      auto x = MakeInput({2, 3, 4});
      auto y = linear(x);
      ExpectShape(y, {2, 3, 6}, "Linear rank-3 forward");
    }

    {
      module::Linear linear(/*in_dim=*/2, /*out_dim=*/3, /*bias=*/false);
      linear.weight() = deeptiny::Tensor::FromVector(
          std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
          deeptiny::Shape{1, 2, 3}, deeptiny::Device::CPU, true);

      auto x = deeptiny::Tensor::FromVector(
          std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, deeptiny::Shape{2, 2},
          deeptiny::Device::CPU, true);
      auto y = linear(x);
      auto loss = deeptiny::functional::Reduce(y, {0, 1});
      loss.Backward();

      auto grad = x.grad();
      Expect(grad.has_value(), "Linear backward missing input grad");
      ExpectVectorNear(*grad, {6.0f, 15.0f, 6.0f, 15.0f}, 1e-5f,
                       "Linear backward input grad");
    }

    {
      module::LlamaMLP mlp(/*in_dim=*/4, /*hidden_dim=*/8, /*out_dim=*/4);
      auto x = MakeInput({5, 4});
      auto y = mlp(x);
      ExpectShape(y, {5, 4}, "LlamaMLP rank-2 forward");

      auto loss = deeptiny::functional::Reduce(y, {0, 1});
      loss.Backward();
      Expect(x.grad().has_value(),
             "LlamaMLP rank-2 backward missing input grad");
    }

    {
      module::LlamaMLP mlp(/*in_dim=*/4, /*hidden_dim=*/8, /*out_dim=*/4);
      auto x = MakeInput({2, 3, 4});
      auto y = mlp(x);
      ExpectShape(y, {2, 3, 4}, "LlamaMLP rank-3 forward");

      auto loss = deeptiny::functional::Reduce(y, {0, 1, 2});
      loss.Backward();
      Expect(x.grad().has_value(),
             "LlamaMLP rank-3 backward missing input grad");
    }

    {
      module::LlamaMLP mlp(/*in_dim=*/2, /*hidden_dim=*/2, /*out_dim=*/1,
                           /*bias=*/false);
      mlp.gate_proj().weight() = deeptiny::Tensor::FromVector(
          std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, deeptiny::Shape{1, 2, 2},
          deeptiny::Device::CPU, true);
      mlp.up_proj().weight() = deeptiny::Tensor::FromVector(
          std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f}, deeptiny::Shape{1, 2, 2},
          deeptiny::Device::CPU, true);
      mlp.down_proj().weight() = deeptiny::Tensor::FromVector(
          std::vector<float>{2.0f, 3.0f}, deeptiny::Shape{1, 2, 1},
          deeptiny::Device::CPU, true);

      auto x = deeptiny::Tensor::FromVector(std::vector<float>{1.0f, 1.0f},
                                            deeptiny::Shape{1, 2},
                                            deeptiny::Device::CPU, true);
      auto y = mlp(x);
      auto loss = deeptiny::functional::Reduce(y, {0, 1});
      loss.Backward();

      auto grad = x.grad();
      Expect(grad.has_value(), "LlamaMLP analytic backward missing input grad");
      ExpectVectorNear(*grad, {256.0f, 440.0f}, 1e-5f,
                       "LlamaMLP analytic backward input grad");
    }

    return 0;
  } catch (const std::exception& err) {
    std::cerr << "modules_smoke failed: " << err.what() << "\n";
    return 1;
  }
}
