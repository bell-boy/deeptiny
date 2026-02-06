#include <cstddef>
#include <iostream>
#include <span>
#include <vector>

#include "deeptiny/functional.h"
#include "deeptiny/math.h"
#include "deeptiny/tensor.h"
#include "deeptiny/types.h"

namespace {

deeptiny::Tensor MakeTensor(const deeptiny::Shape& shape,
                            const std::vector<float>& values,
                            bool requires_grad = false) {
  return deeptiny::Tensor::FromBuffer(
      std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(values.data()),
          values.size() * sizeof(float)),
      shape, deeptiny::DType::Float32, deeptiny::Device::CPU, requires_grad);
}

}  // namespace

int main() {
  using deeptiny::FormatShape;

  auto query = MakeTensor({2, 3}, {1.0f, 2.0f, 3.0f, 1.0f, 0.5f, 2.0f}, true);
  auto key = MakeTensor({1, 3}, {0.1f, 0.2f, 0.3f}, true);

  auto scores = query * key;
  auto loss = deeptiny::functional::Reduce(scores, {0, 1});
  loss.Backward();

  std::cout << "transfomer-demo is wired to Deep Tiny\n";
  std::cout << "scores shape: " << FormatShape(scores.shape()) << "\n";
  std::cout << "loss shape: " << FormatShape(loss.shape()) << "\n";
  std::cout << "query.grad available: " << std::boolalpha
            << query.grad().has_value() << "\n";

  return 0;
}
