#include <algorithm>
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
            << "  ./build/transfomer_demo <model_dir>\n"
            << "  ./build/transfomer_demo --download-smoke-tests\n"
            << "  ./build/transfomer_demo --download-and-plan\n";
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
    if (arg1 == "--download-smoke-tests") {
      demo::smollm2::RunSmolLM2DownloadSmokeTests(
          std::filesystem::current_path());
      const auto model_dir =
          demo::smollm2::ModelFilesDir(std::filesystem::current_path());
      std::cout << "smoke tests completed\n";
      std::cout << "config: " << (model_dir / "config.json") << "\n";
      std::cout << "tokenizer config: " << (model_dir / "tokenizer_config.json")
                << "\n";
      return 0;
    }
    if (arg1 == "--download-and-plan") {
      const auto safetensors_path =
          demo::smollm2::DownloadSmolLM2_135M_InstructSafetensors(
              std::filesystem::current_path());
      const auto header_json =
          demo::smollm2::ReadSafetensorsHeaderJson(safetensors_path);
      const auto placement_plan =
          demo::smollm2::BuildTensorPlacementPlanFromSafetensors(
              safetensors_path, config);

      std::cout << "safetensors: " << safetensors_path << "\n";
      std::cout << "header bytes: " << placement_plan.header_json_size << "\n";
      std::cout << "placement entries: " << placement_plan.placements.size()
                << "\n";
      std::cout << "missing required tensors: "
                << placement_plan.missing_required_tensors.size() << "\n";
      std::cout << "unexpected tensors: "
                << placement_plan.unexpected_tensors.size() << "\n";

      const size_t preview =
          std::min<size_t>(3, placement_plan.placements.size());
      for (size_t i = 0; i < preview; ++i) {
        const auto& entry = placement_plan.placements[i];
        std::cout << "plan[" << i << "]: " << entry.hf_name << " -> "
                  << entry.deeptiny_target << ", present=" << std::boolalpha
                  << entry.present_in_safetensors
                  << ", shape_match=" << entry.shape_matches_expected << "\n";
      }

      const size_t header_preview_len =
          std::min<size_t>(256, header_json.size());
      std::cout << "header preview: "
                << header_json.substr(0, header_preview_len)
                << (header_json.size() > header_preview_len ? "..." : "")
                << "\n";
      return 0;
    }
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

  if (argc > 1) {
    const auto load_plan =
        demo::smollm2::LoadSmolLM2_135M_InstructWeights(argv[1], config);
    std::cout << "weight file: " << load_plan.weights_path << "\n";
    std::cout << "sharded checkpoint: " << std::boolalpha
              << load_plan.is_sharded_checkpoint << "\n";
  } else {
    std::cout << "pass a model directory to validate local safetensors files\n";
    std::cout << "or use --download-smoke-tests / --download-and-plan\n";
  }

  return 0;
}
