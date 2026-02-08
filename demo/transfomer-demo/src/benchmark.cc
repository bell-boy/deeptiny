#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "smollm2_135m_instruct_loader.h"
#include "transformer.h"
#ifdef TRANSFOMER_DEMO_HAS_TOKENIZERS_CPP
#include <tokenizers_cpp.h>
#endif

namespace {

const std::string kEvalText =
    "It's a hot summer Tuesday, and he's standing in the plaza in front of "
    "the Centraal Station with hiseyeballs powered up and the sunlight "
    "jangling off the canal, motor scooters and kamikaze cyclistswhizzing past "
    "and tourists chattering on every side. The square smells of water and "
    "dirt and hot metaland the fart-laden exhaust fumes of cold catalytic "
    "converters; the bells of trams ding in the background,and birds flock "
    "overhead. He glances up and grabs a pigeon, crops the shot, and squirts "
    "it at his weblogto show he's arrived. The bandwidth is good here, he "
    "realizes; and it's not just the bandwidth, it's thewhole scene. Amsterdam "
    "is making him feel wanted already, even though he's fresh off the train "
    "fromSchiphol: He's infected with the dynamic optimism of another time "
    "zone, another city. If the mood holds,someone out there is going to "
    "become very rich indeed.";

void PrintUsage() {
  std::cout << "usage:\n";
  std::cout << "  ./build/transfomer_demo_benchmark [tokenizer_dir]\n";
}

std::string ReadAllBytes(const std::filesystem::path& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (!fs.good()) {
    throw std::runtime_error("Failed to open file: " + path.string());
  }
  fs.seekg(0, std::ios::end);
  const size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  std::string bytes(size, '\0');
  fs.read(bytes.data(), static_cast<std::streamsize>(size));
  if (!fs.good()) {
    throw std::runtime_error("Failed to read file: " + path.string());
  }
  return bytes;
}

std::filesystem::path ResolveTokenizerPath(int argc, char** argv) {
  std::filesystem::path tokenizer_dir =
      demo::smollm2::ModelFilesDir(std::filesystem::current_path());
  if (argc == 2) {
    tokenizer_dir = argv[1];
  }

  std::filesystem::path tokenizer_path = tokenizer_dir / "tokenizer.json";
  if (!std::filesystem::exists(tokenizer_path) ||
      !std::filesystem::is_regular_file(tokenizer_path)) {
    tokenizer_path = demo::smollm2::DownloadSmolLM2_135M_InstructTokenizerJson(
        std::filesystem::current_path());
  }

  return tokenizer_path;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc > 2) {
      PrintUsage();
      return 1;
    }

    if (argc == 2 && std::string(argv[1]) == "--help") {
      PrintUsage();
      return 0;
    }

    const auto config = demo::smollm2::DefaultSmolLM2_135M_InstructConfig();
    transfomer_demo::Transformer model(
        config.vocab_size, config.hidden_size, config.intermediate_size,
        config.num_hidden_layers, config.num_attention_heads,
        config.num_key_value_heads, deeptiny::Device::CPU);
    const std::filesystem::path tokenizer_path =
        ResolveTokenizerPath(argc, argv);

#ifndef TRANSFOMER_DEMO_HAS_TOKENIZERS_CPP
    (void)tokenizer_path;
    throw std::runtime_error(
        "This benchmark requires tokenizers-cpp. Configure with "
        "TRANSFOMER_DEMO_ENABLE_TOKENIZERS_CPP=ON.");
#else
    const std::string tokenizer_blob = ReadAllBytes(tokenizer_path);
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(tokenizer_blob);
    if (!tokenizer) {
      throw std::runtime_error("Failed to initialize tokenizer from " +
                               tokenizer_path.string());
    }

    const std::vector<int32_t> token_ids = tokenizer->Encode(kEvalText);
    if (token_ids.empty()) {
      throw std::runtime_error("Tokenizer returned zero tokens for eval text");
    }
    std::vector<int64_t> tokens(token_ids.begin(), token_ids.end());

    const deeptiny::Tensor output = model({tokens});

    std::cout << "model_init: random (safetensors not loaded)\n";
    std::cout << "tokenizer: " << tokenizer_path << "\n";
    std::cout << "input_text: " << kEvalText << "\n";
    std::cout << "input_token_count: " << token_ids.size() << "\n";
    std::cout << "output_numel: " << output.numel() << "\n";
#endif
  } catch (const std::exception& err) {
    std::cerr << "error: " << err.what() << "\n";
    return 1;
  }
  return 0;
}
