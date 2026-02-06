#include <iostream>
#include <vector>

#include "deeptiny/functional.h"
#include "deeptiny/math.h"
#include "deeptiny/tensor.h"
#include "deeptiny/types.h"

int main() {
  using deeptiny::FormatShape;

  auto query = deeptiny::Tensor::FromVector(
      std::vector<float>{1.0f, 2.0f, 3.0f, 1.0f, 0.5f, 2.0f}, {2, 3},
      deeptiny::Device::CPU, true);
  auto key = deeptiny::Tensor::FromVector(std::vector<float>{0.1f, 0.2f, 0.3f},
                                          {1, 3}, deeptiny::Device::CPU, true);

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
