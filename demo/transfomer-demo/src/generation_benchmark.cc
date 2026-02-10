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

struct GenerationOptions {
  uint64_t max_new_tokens = 64;
  float temperature = 0.8f;
};

void PrintUsage() {
  std::cout << "usage:\n";
  std::cout
      << "  ./build/transfomer_demo_benchmark_generation [tokenizer_dir]\n";
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

    const std::vector<int32_t> prompt_i32 = tokenizer->Encode(kEvalText);
    if (prompt_i32.empty()) {
      throw std::runtime_error("Tokenizer returned zero tokens for eval text");
    }
    std::vector<int64_t> prompt(prompt_i32.begin(), prompt_i32.end());

    const GenerationOptions options;
    std::mt19937 rng(std::random_device{}());
    const std::vector<int64_t> generated =
        Generate(&model, prompt, config, options, &rng);

    std::cout << "model_init: random (safetensors not loaded)\n";
    std::cout << "tokenizer: " << tokenizer_path << "\n";
    std::cout << "input_token_count: " << prompt.size() << "\n";
    std::cout << "generated_token_count: " << generated.size() << "\n";
    std::cout << "max_new_tokens: " << options.max_new_tokens << "\n";
    std::cout << "temperature: " << options.temperature << "\n";
#endif
  } catch (const std::exception& err) {
    std::cerr << "error: " << err.what() << "\n";
    return 1;
  }
  return 0;
}
