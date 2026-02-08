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

void PrintUsage() {
  std::cout << "usage:\n"
            << "  ./build/transfomer_demo_benchmark <model_dir> [iterations]\n";
}

uint64_t ParseUint64Arg(const std::string& value, const char* name) {
  size_t parsed = 0;
  const uint64_t result = std::stoull(value, &parsed);
  if (parsed != value.size()) {
    throw std::runtime_error(std::string("Invalid ") + name + ": " + value);
  }
  return result;
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

void PrintShape(const deeptiny::Tensor& tensor) {
  std::cout << "output_shape: [";
  const auto shape = tensor.shape();
  for (size_t i = 0; i < shape.size(); ++i) {
    std::cout << shape[i];
    if (i + 1 < shape.size()) {
      std::cout << ", ";
    }
  }
  std::cout << "]\n";
}

void PrintTokenIds(const std::vector<int32_t>& token_ids) {
  std::cout << "input_token_ids:";
  for (const int32_t token_id : token_ids) {
    std::cout << " " << token_id;
  }
  std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 3) {
      PrintUsage();
      return 1;
    }

    const std::string arg1 = argv[1];
    if (arg1 == "--help") {
      PrintUsage();
      return 0;
    }

    uint64_t iterations = 1;
    if (argc == 3) {
      iterations = ParseUint64Arg(argv[2], "iterations");
      if (iterations == 0) {
        throw std::runtime_error("iterations must be non-zero");
      }
    }

    const auto config = demo::smollm2::DefaultSmolLM2_135M_InstructConfig();
    const std::filesystem::path model_dir = arg1;
    auto model = demo::smollm2::CreateSmolLM2_135M_InstructTransformer(
        model_dir, config);

    const std::filesystem::path downloaded_tokenizer =
        demo::smollm2::DownloadSmolLM2_135M_InstructTokenizerJson(
            std::filesystem::current_path());
    std::filesystem::path tokenizer_path = model_dir / "tokenizer.json";
    if (!std::filesystem::exists(tokenizer_path) ||
        !std::filesystem::is_regular_file(tokenizer_path)) {
      tokenizer_path = downloaded_tokenizer;
    }

#ifndef TRANSFOMER_DEMO_HAS_TOKENIZERS_CPP
    (void)model;
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

    const std::vector<int32_t> token_ids = tokenizer->Encode("hello world");
    if (token_ids.empty()) {
      throw std::runtime_error(
          "Tokenizer returned zero tokens for 'hello world'");
    }
    std::vector<int64_t> tokens(token_ids.begin(), token_ids.end());

    deeptiny::Tensor output = (*model)({tokens});
    for (uint64_t i = 1; i < iterations; ++i) {
      output = (*model)({tokens});
    }

    std::cout << "model_dir: " << model_dir << "\n";
    std::cout << "tokenizer: " << tokenizer_path << "\n";
    std::cout << "input_text: hello world\n";
    PrintTokenIds(token_ids);
    std::cout << "iterations: " << iterations << "\n";
    PrintShape(output);
#endif
  } catch (const std::exception& err) {
    std::cerr << "error: " << err.what() << "\n";
    return 1;
  }
  return 0;
}
