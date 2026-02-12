#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "smollm2_135m_instruct_loader.h"
#include "transformer.h"
#ifdef TRANSFOMER_DEMO_HAS_TOKENIZERS_CPP
#include <tokenizers_cpp.h>
#endif

namespace {

using GenerationOptions = transfomer_demo::Transformer::GenerationOptions;

void PrintUsage() {
  std::cout << "usage:\n"
            << "  ./build/transfomer_demo <model_dir> [max_new_tokens] "
               "[temperature]\n";
}

uint64_t ParseUint64Arg(const std::string& value, const char* name) {
  size_t parsed = 0;
  const uint64_t result = std::stoull(value, &parsed);
  if (parsed != value.size()) {
    throw std::runtime_error(std::string("Invalid ") + name + ": " + value);
  }
  return result;
}

float ParseFloatArg(const std::string& value, const char* name) {
  size_t parsed = 0;
  const float result = std::stof(value, &parsed);
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

std::vector<int32_t> ToInt32Tokens(const std::vector<int64_t>& tokens,
                                   const char* name) {
  std::vector<int32_t> out;
  out.reserve(tokens.size());
  for (const int64_t token : tokens) {
    if (token < std::numeric_limits<int32_t>::min() ||
        token > std::numeric_limits<int32_t>::max()) {
      throw std::runtime_error(std::string("Token id out of int32 range in ") +
                               name);
    }
    out.push_back(static_cast<int32_t>(token));
  }
  return out;
}

void PrintTokenIds(const char* label, const std::vector<int64_t>& ids) {
  std::cout << label << ":";
  for (const int64_t id : ids) {
    std::cout << " " << id;
  }
  std::cout << "\n";
}

#ifdef TRANSFOMER_DEMO_HAS_TOKENIZERS_CPP
void RunChatLoop(transfomer_demo::Transformer* model,
                 tokenizers::Tokenizer* tokenizer,
                 const GenerationOptions& options) {
  std::string line;
  std::cout << "chat ready (type 'exit' to quit)\n";
  while (true) {
    std::cout << "in> " << std::flush;
    if (!std::getline(std::cin, line)) {
      break;
    }
    if (line == "exit" || line == "quit") {
      break;
    }
    if (line.empty()) {
      continue;
    }

    const std::vector<int32_t> encoded_prompt_i32 = tokenizer->Encode(line);
    std::vector<int64_t> encoded_prompt(encoded_prompt_i32.begin(),
                                        encoded_prompt_i32.end());
    if (encoded_prompt.empty()) {
      std::cout << "out_ids:\n";
      std::cout << "out_text:\n";
      continue;
    }

    PrintTokenIds("in_ids", encoded_prompt);

    std::vector<int64_t> generated;
    generated.reserve(static_cast<size_t>(options.max_new_tokens));

    auto token_stream = model->GenerateAsync(encoded_prompt, options);
    std::cout << "out_ids:" << std::flush;
    int64_t token = 0;
    while (token_stream.WaitNext(&token)) {
      generated.push_back(token);
      std::cout << " " << token << std::flush;
    }
    token_stream.Join();
    std::cout << "\n";

    const std::vector<int32_t> generated_i32 =
        ToInt32Tokens(generated, "generated");

    std::cout << "out_text: " << tokenizer->Decode(generated_i32) << "\n";
  }
}
#endif

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      PrintUsage();
      return 1;
    }

    const std::string arg1 = argv[1];
    if (arg1 == "--help") {
      PrintUsage();
      return 0;
    }
    if (argc > 4) {
      PrintUsage();
      return 1;
    }

    const auto config = demo::smollm2::DefaultSmolLM2_135M_InstructConfig();
    GenerationOptions options;
    if (argc >= 3) {
      options.max_new_tokens = ParseUint64Arg(argv[2], "max_new_tokens");
      if (options.max_new_tokens == 0) {
        throw std::runtime_error("max_new_tokens must be non-zero");
      }
    }
    if (argc >= 4) {
      options.temperature = ParseFloatArg(argv[3], "temperature");
      if (options.temperature < 0.0f) {
        throw std::runtime_error("temperature must be >= 0");
      }
    }
    options.eos_token_id = config.eos_token_id;

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
        "This demo requires tokenizers-cpp. Configure with "
        "TRANSFOMER_DEMO_ENABLE_TOKENIZERS_CPP=ON.");
#else
    const std::string tokenizer_blob = ReadAllBytes(tokenizer_path);
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(tokenizer_blob);
    if (!tokenizer) {
      throw std::runtime_error("Failed to initialize tokenizer from " +
                               tokenizer_path.string());
    }

    std::cout << "model_dir: " << model_dir << "\n";
    std::cout << "tokenizer: " << tokenizer_path << "\n";
    std::cout << "max_new_tokens: " << options.max_new_tokens << "\n";
    std::cout << "temperature: " << options.temperature << "\n";
    RunChatLoop(model.get(), tokenizer.get(), options);
#endif
  } catch (const std::exception& err) {
    std::cerr << "error: " << err.what() << "\n";
    return 1;
  }
  return 0;
}
