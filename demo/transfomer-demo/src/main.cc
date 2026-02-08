#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "deeptiny/math.h"
#include "deeptiny/tensor.h"
#include "deeptiny/types.h"
#include "smollm2_135m_instruct_loader.h"
#include "transformer.h"
#ifdef TRANSFOMER_DEMO_HAS_TOKENIZERS_CPP
#include <tokenizers_cpp.h>
#endif

namespace {

struct GenerationOptions {
  uint64_t max_new_tokens = 64;
  float temperature = 0.8f;
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

std::vector<float> TensorToFloatVector(const deeptiny::Tensor& tensor) {
  if (tensor.dtype() != deeptiny::DType::Float32) {
    throw std::runtime_error("TensorToFloatVector expects Float32 tensor");
  }
  std::vector<float> values(static_cast<size_t>(tensor.numel()), 0.0f);
  tensor.CopyToBuffer(
      std::as_writable_bytes(std::span<float>(values.data(), values.size())),
      tensor.shape(), deeptiny::DType::Float32);
  return values;
}

uint64_t ArgmaxIndex(const std::vector<float>& logits) {
  if (logits.empty()) {
    throw std::runtime_error("Cannot sample from empty logits");
  }
  size_t best_idx = 0;
  float best_value = logits[0];
  for (size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > best_value) {
      best_value = logits[i];
      best_idx = i;
    }
  }
  return static_cast<uint64_t>(best_idx);
}

uint64_t SampleFromLogits(const std::vector<float>& logits, float temperature,
                          std::mt19937* rng) {
  if (!(temperature > 0.0f)) {
    return ArgmaxIndex(logits);
  }

  const double inv_temp = 1.0 / static_cast<double>(temperature);
  double max_scaled = -std::numeric_limits<double>::infinity();
  for (const float value : logits) {
    max_scaled = std::max(max_scaled, static_cast<double>(value) * inv_temp);
  }

  std::vector<double> probs(logits.size(), 0.0);
  double total = 0.0;
  for (size_t i = 0; i < logits.size(); ++i) {
    const double scaled = static_cast<double>(logits[i]) * inv_temp;
    const double prob = std::exp(scaled - max_scaled);
    if (std::isfinite(prob) && prob > 0.0) {
      probs[i] = prob;
      total += prob;
    }
  }

  if (!(total > 0.0) || !std::isfinite(total)) {
    return ArgmaxIndex(logits);
  }

  std::discrete_distribution<size_t> distribution(probs.begin(), probs.end());
  return static_cast<uint64_t>(distribution(*rng));
}

deeptiny::Tensor ComputeNextTokenLogits(transfomer_demo::Transformer* model,
                                        const std::vector<int64_t>& tokens,
                                        const demo::smollm2::Config& config) {
  if (tokens.empty()) {
    throw std::runtime_error(
        "ComputeNextTokenLogits requires non-empty tokens");
  }
  const deeptiny::Tensor hidden_states = (*model)({tokens});
  const int64_t last_token_index =
      static_cast<int64_t>(tokens.size()) - static_cast<int64_t>(1);
  const int64_t hidden_size = static_cast<int64_t>(config.hidden_size);
  deeptiny::Tensor last_hidden =
      hidden_states({deeptiny::Slice(0, 1),
                     deeptiny::Slice(last_token_index, last_token_index + 1),
                     deeptiny::Slice(0, hidden_size)});

  deeptiny::Tensor query = last_hidden.Reshape({1, 1, config.hidden_size});
  deeptiny::Tensor tied_embedding = model->embed().weight().Reshape(
      {1, config.vocab_size, config.hidden_size});
  deeptiny::Tensor logits =
      deeptiny::math::BatchedMatMul(query, tied_embedding, false, true);
  return logits.Reshape({config.vocab_size});
}

std::vector<int64_t> Generate(transfomer_demo::Transformer* model,
                              const std::vector<int64_t>& prompt_tokens,
                              const demo::smollm2::Config& config,
                              const GenerationOptions& options,
                              std::mt19937* rng) {
  if (prompt_tokens.empty()) {
    throw std::runtime_error("Generate requires at least one prompt token");
  }

  std::vector<int64_t> context = prompt_tokens;
  std::vector<int64_t> generated;
  generated.reserve(static_cast<size_t>(options.max_new_tokens));

  for (uint64_t step = 0; step < options.max_new_tokens; ++step) {
    if (context.size() > config.max_position_embeddings) {
      const size_t drop = context.size() - config.max_position_embeddings;
      context.erase(context.begin(),
                    context.begin() + static_cast<int64_t>(drop));
    }

    const deeptiny::Tensor logits =
        ComputeNextTokenLogits(model, context, config);
    const std::vector<float> logits_values = TensorToFloatVector(logits);
    const uint64_t next_token =
        SampleFromLogits(logits_values, options.temperature, rng);

    context.push_back(static_cast<int64_t>(next_token));
    generated.push_back(static_cast<int64_t>(next_token));
    if (next_token == config.eos_token_id) {
      break;
    }
  }

  return generated;
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
                 const demo::smollm2::Config& config,
                 const GenerationOptions& options) {
  std::mt19937 rng(std::random_device{}());
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

    const std::vector<int64_t> generated =
        Generate(model, encoded_prompt, config, options, &rng);
    const std::vector<int32_t> generated_i32 =
        ToInt32Tokens(generated, "generated");

    PrintTokenIds("in_ids", encoded_prompt);
    PrintTokenIds("out_ids", generated);
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
    RunChatLoop(model.get(), tokenizer.get(), config, options);
#endif
  } catch (const std::exception& err) {
    std::cerr << "error: " << err.what() << "\n";
    return 1;
  }
  return 0;
}
