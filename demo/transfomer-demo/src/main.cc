#include <iostream>
#include <vector>

#include "deeptiny/functional.h"
#include "deeptiny/tensor.h"
#include "deeptiny/types.h"
#include "transformer.h"
#ifdef TRANSFOMER_DEMO_HAS_TOKENIZERS_CPP
#include "tokenizers_cpp.h"
#endif

int main() {
  using deeptiny::FormatShape;

  constexpr uint64_t vocab_size = 32;
  constexpr uint64_t hidden_size = 8;
  constexpr uint64_t intermediate_size = 16;
  constexpr uint64_t num_blocks = 3;
  constexpr uint64_t num_attention_heads = 2;
  constexpr uint64_t num_key_value_heads = 2;

  const std::vector<std::vector<int64_t>> tokens{
      {1, 7, 3, 5},
      {4, 2, 9, 6},
  };
  transfomer_demo::Transformer transformer(
      vocab_size, hidden_size, intermediate_size, num_blocks,
      num_attention_heads, num_key_value_heads, deeptiny::Device::CPU);

  auto y = transformer(tokens);
  auto loss = deeptiny::functional::Reduce(y, {0, 1, 2});
  loss.Backward();

#ifdef TRANSFOMER_DEMO_HAS_TOKENIZERS_CPP
  // Compile and link smoke test for tokenizers-cpp API integration.
  const auto from_blob_json = &tokenizers::Tokenizer::FromBlobJSON;
  (void)from_blob_json;
#endif

  std::cout << "transfomer-demo Transformer demo\n";
  std::cout << "token batch shape: "
            << FormatShape({static_cast<uint64_t>(tokens.size()),
                            static_cast<uint64_t>(tokens.front().size())})
            << "\n";
  std::cout << "transformer blocks: " << transformer.num_blocks() << "\n";
  std::cout << "output shape: " << FormatShape(y.shape()) << "\n";
  std::cout << "loss shape: " << FormatShape(loss.shape()) << "\n";
#ifdef TRANSFOMER_DEMO_HAS_TOKENIZERS_CPP
  std::cout << "tokenizers-cpp linked: true\n";
#else
  std::cout << "tokenizers-cpp linked: false\n";
#endif
  std::cout << "embed.weight.grad available: " << std::boolalpha
            << transformer.embed().weight().grad().has_value() << "\n";
  std::cout << "norm.weight.grad available: "
            << transformer.norm().weight().grad().has_value() << "\n";

  return 0;
}
