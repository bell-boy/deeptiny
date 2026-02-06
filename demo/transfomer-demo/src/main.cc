#include <iostream>
#include <vector>

#include "deeptiny/functional.h"
#include "deeptiny/tensor.h"
#include "deeptiny/types.h"
#include "modules/embedding.h"

int main() {
  using deeptiny::FormatShape;
  using transfomer_demo::modules::Embedding;

  Embedding embedding(/*num_embeddings=*/16, /*embedding_dim=*/8);
  const std::vector<int64_t> token_ids{1, 4, 8, 1};
  const deeptiny::Shape token_shape{2, 2};

  auto embedded = embedding.Forward(token_ids, token_shape);
  auto loss = deeptiny::functional::Reduce(embedded, {0, 1, 2});
  loss.Backward();

  std::cout << "transfomer-demo embedding module is wired to Deep Tiny\n";
  std::cout << "token shape: " << FormatShape(token_shape) << "\n";
  std::cout << "embedding weight shape: "
            << FormatShape(embedding.weight().shape()) << "\n";
  std::cout << "embedded shape: " << FormatShape(embedded.shape()) << "\n";
  std::cout << "loss shape: " << FormatShape(loss.shape()) << "\n";
  std::cout << "embedding.weight().grad available: " << std::boolalpha
            << embedding.weight().grad().has_value() << "\n";

  return 0;
}
