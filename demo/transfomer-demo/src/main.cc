#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "smollm2_135m_instruct_loader.h"
#include "transformer.h"
#ifdef TRANSFOMER_DEMO_HAS_TOKENIZERS_CPP
#include <tokenizers_cpp.h>
#endif

namespace {

using GenerationOptions = transfomer_demo::Transformer::GenerationOptions;

struct ChatMessage {
  std::string role;
  std::string content;
};

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

void PrintDecodedIncrement(const std::string& decoded, std::string* emitted) {
  if (emitted == nullptr) {
    throw std::runtime_error("PrintDecodedIncrement requires output state.");
  }
  if (decoded.size() >= emitted->size() &&
      decoded.compare(0, emitted->size(), *emitted) == 0) {
    std::cout << decoded.substr(emitted->size()) << std::flush;
  } else {
    std::cout << decoded << std::flush;
  }
  *emitted = decoded;
}

std::string ApplyChatTemplate(const std::vector<ChatMessage>& messages,
                              bool add_generation_prompt) {
  constexpr std::string_view kImStart = "<|im_start|>";
  constexpr std::string_view kImEnd = "<|im_end|>";
  constexpr std::string_view kDefaultSystemPrompt =
      "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
  std::string result;

  if (!messages.empty() && messages.front().role != "system") {
    result += kImStart;
    result += "system\n";
    result += kDefaultSystemPrompt;
    result += kImEnd;
    result += "\n";
  }

  for (const ChatMessage& message : messages) {
    result += kImStart;
    result += message.role;
    result += "\n";
    result += message.content;
    result += kImEnd;
    result += "\n";
  }

  if (add_generation_prompt) {
    result += kImStart;
    result += "assistant\n";
  }

  return result;
}

std::string StripAssistantControlTokens(const std::string& decoded) {
  constexpr std::string_view kImEnd = "<|im_end|>";
  const size_t im_end_pos = decoded.find(kImEnd);
  if (im_end_pos == std::string::npos) {
    return decoded;
  }
  return decoded.substr(0, im_end_pos);
}

#ifdef TRANSFOMER_DEMO_HAS_TOKENIZERS_CPP
void RunChatLoop(transfomer_demo::Transformer* model,
                 tokenizers::Tokenizer* tokenizer,
                 const GenerationOptions& options) {
  std::string line;
  std::vector<ChatMessage> messages;
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

    messages.push_back(ChatMessage{.role = "user", .content = line});
    const std::string prompt =
        ApplyChatTemplate(messages, /*add_generation_prompt=*/true);
    const std::vector<int32_t> encoded_prompt_i32 = tokenizer->Encode(prompt);
    std::vector<int64_t> encoded_prompt(encoded_prompt_i32.begin(),
                                        encoded_prompt_i32.end());
    if (encoded_prompt.empty()) {
      messages.pop_back();
      std::cout << "out_text:\n";
      continue;
    }

    auto token_stream = model->GenerateAsync(encoded_prompt, options);
    std::cout << "out_text: " << std::flush;
    std::vector<int32_t> generated_i32;
    generated_i32.reserve(static_cast<size_t>(options.max_new_tokens));
    std::string emitted;

    int64_t token = 0;
    while (token_stream.WaitNext(&token)) {
      if (token < std::numeric_limits<int32_t>::min() ||
          token > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("Token id out of int32 range in generated");
      }
      generated_i32.push_back(static_cast<int32_t>(token));
      const std::string decoded = tokenizer->Decode(generated_i32);
      const std::string decoded_no_control =
          StripAssistantControlTokens(decoded);
      PrintDecodedIncrement(decoded_no_control, &emitted);
      if (decoded_no_control.size() != decoded.size()) {
        break;
      }
    }
    token_stream.Join();
    messages.push_back(ChatMessage{.role = "assistant", .content = emitted});
    std::cout << "\n";
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
