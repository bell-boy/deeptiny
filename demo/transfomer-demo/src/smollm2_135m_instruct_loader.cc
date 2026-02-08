#include "smollm2_135m_instruct_loader.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace demo::smollm2 {
namespace {

uint64_t HeadDim(const Config& config) {
  return config.hidden_size / config.num_attention_heads;
}

void ValidateConfig(const Config& config) {
  if (config.hidden_size == 0) {
    throw std::runtime_error("SmolLM2 config hidden_size must be non-zero");
  }
  if (config.intermediate_size == 0) {
    throw std::runtime_error(
        "SmolLM2 config intermediate_size must be non-zero");
  }
  if (config.num_attention_heads == 0) {
    throw std::runtime_error(
        "SmolLM2 config num_attention_heads must be non-zero");
  }
  if (config.num_key_value_heads == 0) {
    throw std::runtime_error(
        "SmolLM2 config num_key_value_heads must be non-zero");
  }
  if (config.num_hidden_layers == 0) {
    throw std::runtime_error(
        "SmolLM2 config num_hidden_layers must be non-zero");
  }
  if (config.vocab_size == 0) {
    throw std::runtime_error("SmolLM2 config vocab_size must be non-zero");
  }
  if (config.max_position_embeddings == 0) {
    throw std::runtime_error(
        "SmolLM2 config max_position_embeddings must be non-zero");
  }
  if (config.hidden_size % config.num_attention_heads != 0) {
    throw std::runtime_error(
        "SmolLM2 config hidden_size must be divisible by "
        "num_attention_heads");
  }
  if (config.num_attention_heads % config.num_key_value_heads != 0) {
    throw std::runtime_error(
        "SmolLM2 config num_attention_heads must be divisible by "
        "num_key_value_heads");
  }
  if (HeadDim(config) % 2 != 0) {
    throw std::runtime_error(
        "SmolLM2 config requires even head_dim for RoPE support");
  }
  if (!(config.rope_theta > 0.0f)) {
    throw std::runtime_error("SmolLM2 config rope_theta must be > 0");
  }
}

void AddWeight(std::vector<WeightSpec>* specs, std::string hf_name,
               deeptiny::Shape hf_shape, deeptiny::Shape deeptiny_shape,
               bool transpose_last_two, bool required = true) {
  specs->push_back(WeightSpec{std::move(hf_name), std::move(hf_shape),
                              std::move(deeptiny_shape), transpose_last_two,
                              required});
}

}  // namespace

Config DefaultSmolLM2_135M_InstructConfig() { return Config{}; }

std::vector<WeightSpec> BuildWeightSpecs(const Config& config) {
  ValidateConfig(config);

  const uint64_t head_dim = HeadDim(config);
  const uint64_t kv_hidden_size = config.num_key_value_heads * head_dim;

  std::vector<WeightSpec> specs;
  specs.reserve(static_cast<size_t>(config.num_hidden_layers * 9 + 3));

  AddWeight(&specs, "model.embed_tokens.weight",
            {config.vocab_size, config.hidden_size},
            {config.vocab_size, config.hidden_size}, false);

  for (uint64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    const std::string prefix = "model.layers." + std::to_string(layer);

    AddWeight(&specs, prefix + ".input_layernorm.weight", {config.hidden_size},
              {config.hidden_size}, false);

    AddWeight(&specs, prefix + ".self_attn.q_proj.weight",
              {config.hidden_size, config.hidden_size},
              {1, config.num_attention_heads, config.hidden_size, head_dim},
              true);
    AddWeight(&specs, prefix + ".self_attn.k_proj.weight",
              {kv_hidden_size, config.hidden_size},
              {1, config.num_key_value_heads, config.hidden_size, head_dim},
              true);
    AddWeight(&specs, prefix + ".self_attn.v_proj.weight",
              {kv_hidden_size, config.hidden_size},
              {1, config.num_key_value_heads, config.hidden_size, head_dim},
              true);
    AddWeight(&specs, prefix + ".self_attn.o_proj.weight",
              {config.hidden_size, config.hidden_size},
              {1, config.num_attention_heads, head_dim, config.hidden_size},
              true);

    AddWeight(&specs, prefix + ".post_attention_layernorm.weight",
              {config.hidden_size}, {config.hidden_size}, false);

    AddWeight(&specs, prefix + ".mlp.gate_proj.weight",
              {config.intermediate_size, config.hidden_size},
              {1, config.hidden_size, config.intermediate_size}, true);
    AddWeight(&specs, prefix + ".mlp.up_proj.weight",
              {config.intermediate_size, config.hidden_size},
              {1, config.hidden_size, config.intermediate_size}, true);
    AddWeight(&specs, prefix + ".mlp.down_proj.weight",
              {config.hidden_size, config.intermediate_size},
              {1, config.intermediate_size, config.hidden_size}, true);

    if (config.attention_bias) {
      AddWeight(&specs, prefix + ".self_attn.q_proj.bias", {config.hidden_size},
                {1, config.num_attention_heads, 1, head_dim}, false);
      AddWeight(&specs, prefix + ".self_attn.k_proj.bias", {kv_hidden_size},
                {1, config.num_key_value_heads, 1, head_dim}, false);
      AddWeight(&specs, prefix + ".self_attn.v_proj.bias", {kv_hidden_size},
                {1, config.num_key_value_heads, 1, head_dim}, false);
      AddWeight(&specs, prefix + ".self_attn.o_proj.bias", {config.hidden_size},
                {1, 1, config.hidden_size}, false);
    }

    if (config.mlp_bias) {
      AddWeight(&specs, prefix + ".mlp.gate_proj.bias",
                {config.intermediate_size}, {1, 1, config.intermediate_size},
                false);
      AddWeight(&specs, prefix + ".mlp.up_proj.bias",
                {config.intermediate_size}, {1, 1, config.intermediate_size},
                false);
      AddWeight(&specs, prefix + ".mlp.down_proj.bias", {config.hidden_size},
                {1, 1, config.hidden_size}, false);
    }
  }

  AddWeight(&specs, "model.norm.weight", {config.hidden_size},
            {config.hidden_size}, false);

  AddWeight(&specs, "lm_head.weight", {config.vocab_size, config.hidden_size},
            {config.vocab_size, config.hidden_size}, false,
            !config.tie_word_embeddings);

  return specs;
}

WeightLoadPlan LoadSmolLM2_135M_InstructWeights(
    const std::filesystem::path& model_dir, const Config& config) {
  ValidateConfig(config);

  if (model_dir.empty()) {
    throw std::runtime_error("SmolLM2 loader requires a non-empty model_dir");
  }

  if (!std::filesystem::exists(model_dir) ||
      !std::filesystem::is_directory(model_dir)) {
    throw std::runtime_error("SmolLM2 loader could not find model directory: " +
                             model_dir.string());
  }

  const auto single_file = model_dir / "model.safetensors";
  const auto sharded_index = model_dir / "model.safetensors.index.json";

  WeightLoadPlan plan;
  plan.model_dir = model_dir;
  plan.weights = BuildWeightSpecs(config);

  if (std::filesystem::exists(single_file)) {
    plan.weights_path = single_file;
    plan.is_sharded_checkpoint = false;
    return plan;
  }

  if (std::filesystem::exists(sharded_index)) {
    plan.weights_path = sharded_index;
    plan.is_sharded_checkpoint = true;
    return plan;
  }

  throw std::runtime_error(
      "SmolLM2 loader expected model.safetensors or "
      "model.safetensors.index.json in " +
      model_dir.string());
}

}  // namespace demo::smollm2
