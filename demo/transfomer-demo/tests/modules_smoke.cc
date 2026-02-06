#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "deeptiny/functional.h"
#include "deeptiny/tensor.h"
#include "modules/linear.h"
#include "modules/llama_mlp.h"

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

    return 0;
  } catch (const std::exception& err) {
    std::cerr << "modules_smoke failed: " << err.what() << "\n";
    return 1;
  }
}
