#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "deeptiny/types.h"

namespace transfomer_demo {
class Transformer;
}

namespace demo::smollm2 {

struct Config {
  bool attention_bias = false;
  float attention_dropout = 0.0f;
  uint64_t bos_token_id = 1;
  uint64_t eos_token_id = 2;
  std::string hidden_act = "silu";
  uint64_t hidden_size = 576;
  double initializer_range = 0.041666666666666664;
  uint64_t intermediate_size = 1536;
  bool is_llama_config = true;
  uint64_t max_position_embeddings = 8192;
  bool mlp_bias = false;
  std::string model_type = "llama";
  uint64_t num_attention_heads = 9;
  uint64_t num_hidden_layers = 30;
  uint64_t num_key_value_heads = 3;
  uint64_t pad_token_id = 2;
  uint64_t pretraining_tp = 1;
  float rms_norm_eps = 1e-5f;
  bool rope_interleaved = false;
  float rope_theta = 100000.0f;
  bool tie_word_embeddings = true;
  deeptiny::DType torch_dtype = deeptiny::DType::BFloat16;
  bool use_cache = true;
  uint64_t vocab_size = 49152;
};

struct WeightSpec {
  enum class Transform {
    kNone,
    kTransposeLastTwo,
    kQkvOutInToHeadInDim,
  };

  std::string hf_name;
  std::string deeptiny_target;
  deeptiny::Shape hf_shape;
  deeptiny::Shape deeptiny_shape;
  Transform transform = Transform::kNone;
  bool required = true;
};

Config DefaultSmolLM2_135M_InstructConfig();

std::filesystem::path ModelFilesDir(
    const std::filesystem::path& cwd = std::filesystem::current_path());

// Downloads model.safetensors into cwd/model_files/model.safetensors.
std::filesystem::path DownloadSmolLM2_135M_InstructSafetensors(
    const std::filesystem::path& cwd = std::filesystem::current_path());

// Downloads tokenizer.json into cwd/model_files/tokenizer.json.
std::filesystem::path DownloadSmolLM2_135M_InstructTokenizerJson(
    const std::filesystem::path& cwd = std::filesystem::current_path());

std::vector<WeightSpec> BuildWeightSpecs(const Config& config);

// Builds the SmolLM2 demo Transformer with initialized module parameters only.
// This does not read model files or load safetensors weights.
std::unique_ptr<transfomer_demo::Transformer>
CreateSmolLM2_135M_InstructTransformerUninitialized(
    const Config& config = DefaultSmolLM2_135M_InstructConfig());

std::unique_ptr<transfomer_demo::Transformer>
CreateSmolLM2_135M_InstructTransformer(
    const std::filesystem::path& model_dir,
    const Config& config = DefaultSmolLM2_135M_InstructConfig());

}  // namespace demo::smollm2
