#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "deeptiny/functional.h"
#include "deeptiny/nn/gated_relu.h"
#include "deeptiny/tensor.h"
#include "deeptiny/types.h"
#include "smollm2_135m_instruct_loader.h"
#include "transformer.h"
#ifdef TRANSFOMER_DEMO_HAS_TOKENIZERS_CPP
#include "tokenizers_cpp.h"
#endif

namespace {

void PrintUsage() {
  std::cout << "usage:\n"
            << "  ./build/transfomer_demo\n"
            << "  ./build/transfomer_demo <model_dir>\n";
}

}  // namespace

int main(int argc, char** argv) {
  const auto config = demo::smollm2::DefaultSmolLM2_135M_InstructConfig();

  if (argc > 1) {
    const std::string arg1 = argv[1];
    if (arg1 == "--help") {
      PrintUsage();
      return 0;
    }

    auto model = demo::smollm2::CreateSmolLM2_135M_InstructTransformer(
        std::filesystem::path(arg1), config);

    std::cout << "loaded smollm2-135m-instruct weights\n";
    std::cout << "model_dir: " << std::filesystem::path(arg1) << "\n";
    std::cout << "hidden_size: " << config.hidden_size << "\n";
    std::cout << "num_hidden_layers: " << config.num_hidden_layers << "\n";
    std::cout << "num_attention_heads: " << config.num_attention_heads << "\n";
    std::cout << "num_key_value_heads: " << config.num_key_value_heads << "\n";
    std::cout << "transformer blocks: " << model->num_blocks() << "\n";
    return 0;
  }

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

  const auto specs = demo::smollm2::BuildWeightSpecs(config);

  std::cout << "\nsmollm2-135m-instruct defaults\n";
  std::cout << "hidden_size: " << config.hidden_size << "\n";
  std::cout << "intermediate_size: " << config.intermediate_size << "\n";
  std::cout << "num_hidden_layers: " << config.num_hidden_layers << "\n";
  std::cout << "num_attention_heads: " << config.num_attention_heads << "\n";
  std::cout << "num_key_value_heads: " << config.num_key_value_heads << "\n";
  std::cout << "vocab_size: " << config.vocab_size << "\n";
  std::cout << "expected weight tensors: " << specs.size() << "\n";
  std::cout << "pass a model directory to load model.safetensors via mmap\n";

  return 0;
}
