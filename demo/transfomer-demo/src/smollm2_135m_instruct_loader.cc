#include "smollm2_135m_instruct_loader.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace demo::smollm2 {
namespace {

constexpr char kConfigJsonUrl[] =
    "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/"
    "config.json?download=true";
constexpr char kTokenizerConfigJsonUrl[] =
    "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/"
    "tokenizer_config.json?download=true";
constexpr char kModelSafetensorsUrl[] =
    "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/"
    "model.safetensors?download=true";

struct SafetensorsEntry {
  std::string dtype;
  deeptiny::Shape shape;
  uint64_t data_start = 0;
  uint64_t data_end = 0;
};

struct ParsedSafetensorsHeader {
  std::string raw_json;
  nlohmann::json parsed_json;
  std::unordered_map<std::string, SafetensorsEntry> tensors;
};

uint64_t HeadDim(const Config& config) {
  return config.hidden_size / config.num_attention_heads;
}

std::string ShellQuote(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 2);
  out.push_back('\'');
  for (const char c : value) {
    if (c == '\'') {
      out += "'\\''";
      continue;
    }
    out.push_back(c);
  }
  out.push_back('\'');
  return out;
}

bool FileExistsAndNonEmpty(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path) ||
      !std::filesystem::is_regular_file(path)) {
    return false;
  }
  return std::filesystem::file_size(path) > 0;
}

void DownloadUrlToPath(const std::string& url,
                       const std::filesystem::path& destination) {
  if (FileExistsAndNonEmpty(destination)) {
    return;
  }

  std::filesystem::create_directories(destination.parent_path());

  const std::filesystem::path tmp = destination.string() + ".tmp";
  std::error_code remove_err;
  std::filesystem::remove(tmp, remove_err);

  const std::string cmd =
      "curl -L --fail --silent --show-error --retry 3 --retry-delay 1 "
      "--output " +
      ShellQuote(tmp.string()) + " " + ShellQuote(url);

  const int status = std::system(cmd.c_str());
  if (status != 0) {
    std::filesystem::remove(tmp, remove_err);
    throw std::runtime_error("curl download failed for URL: " + url);
  }

  if (!FileExistsAndNonEmpty(tmp)) {
    std::filesystem::remove(tmp, remove_err);
    throw std::runtime_error("Downloaded file is empty: " + tmp.string());
  }

  std::filesystem::remove(destination, remove_err);
  std::error_code rename_err;
  std::filesystem::rename(tmp, destination, rename_err);
  if (rename_err) {
    std::filesystem::remove(tmp, remove_err);
    throw std::runtime_error("Failed to move downloaded file into place: " +
                             destination.string());
  }
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
               std::string deeptiny_target, deeptiny::Shape hf_shape,
               deeptiny::Shape deeptiny_shape, bool transpose_last_two,
               bool required = true) {
  specs->push_back(WeightSpec{std::move(hf_name), std::move(deeptiny_target),
                              std::move(hf_shape), std::move(deeptiny_shape),
                              transpose_last_two, required});
}

uint64_t ParseJsonUint64(const nlohmann::json& value,
                         const std::string& context) {
  if (!(value.is_number_unsigned() || value.is_number_integer())) {
    throw std::runtime_error("Expected integer value for " + context);
  }
  const int64_t as_signed = value.get<int64_t>();
  if (as_signed < 0) {
    throw std::runtime_error("Expected non-negative integer for " + context);
  }
  return static_cast<uint64_t>(as_signed);
}

deeptiny::Shape ParseShape(const nlohmann::json& shape_json,
                           const std::string& tensor_name) {
  if (!shape_json.is_array()) {
    throw std::runtime_error("Safetensors tensor shape is not an array: " +
                             tensor_name);
  }

  deeptiny::Shape shape;
  shape.reserve(shape_json.size());
  for (size_t i = 0; i < shape_json.size(); ++i) {
    shape.push_back(ParseJsonUint64(
        shape_json[i], tensor_name + ".shape[" + std::to_string(i) + "]"));
  }
  return shape;
}

uint64_t ParseHeaderLength(std::ifstream* stream) {
  std::array<unsigned char, 8> bytes{};
  stream->read(reinterpret_cast<char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!*stream) {
    throw std::runtime_error("Failed to read safetensors header length");
  }

  uint64_t value = 0;
  for (size_t i = 0; i < bytes.size(); ++i) {
    value |= static_cast<uint64_t>(bytes[i]) << (8U * i);
  }
  return value;
}

ParsedSafetensorsHeader ParseSafetensorsHeader(
    const std::filesystem::path& safetensors_path) {
  if (!std::filesystem::exists(safetensors_path) ||
      !std::filesystem::is_regular_file(safetensors_path)) {
    throw std::runtime_error("Safetensors file does not exist: " +
                             safetensors_path.string());
  }

  std::ifstream stream(safetensors_path, std::ios::binary);
  if (!stream.is_open()) {
    throw std::runtime_error("Failed to open safetensors file: " +
                             safetensors_path.string());
  }

  const uint64_t header_size = ParseHeaderLength(&stream);
  if (header_size == 0) {
    throw std::runtime_error("Safetensors header length must be non-zero");
  }

  const uint64_t total_size = std::filesystem::file_size(safetensors_path);
  if (total_size < 8) {
    throw std::runtime_error("Safetensors file is too small to contain header");
  }
  if (header_size > total_size - 8) {
    throw std::runtime_error("Safetensors header length exceeds file size");
  }

  if (header_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error("Safetensors header length is too large to read");
  }

  std::string header_json(static_cast<size_t>(header_size), '\0');
  stream.read(header_json.data(), static_cast<std::streamsize>(header_size));
  if (!stream) {
    throw std::runtime_error("Failed to read safetensors header JSON bytes");
  }

  ParsedSafetensorsHeader parsed;
  parsed.raw_json = std::move(header_json);

  try {
    parsed.parsed_json = nlohmann::json::parse(parsed.raw_json);
  } catch (const nlohmann::json::parse_error& err) {
    throw std::runtime_error("Failed to parse safetensors header JSON: " +
                             std::string(err.what()));
  }

  if (!parsed.parsed_json.is_object()) {
    throw std::runtime_error("Safetensors header JSON root must be an object");
  }

  for (const auto& [name, value] : parsed.parsed_json.items()) {
    if (name == "__metadata__") {
      continue;
    }
    if (!value.is_object()) {
      throw std::runtime_error("Safetensors entry is not an object: " + name);
    }

    const auto dtype_it = value.find("dtype");
    const auto shape_it = value.find("shape");
    const auto offsets_it = value.find("data_offsets");
    if (dtype_it == value.end() || shape_it == value.end() ||
        offsets_it == value.end()) {
      throw std::runtime_error("Safetensors entry missing required fields: " +
                               name);
    }
    if (!dtype_it->is_string()) {
      throw std::runtime_error("Safetensors dtype must be a string: " + name);
    }
    if (!offsets_it->is_array() || offsets_it->size() != 2) {
      throw std::runtime_error(
          "Safetensors data_offsets must be [start, end]: " + name);
    }

    const uint64_t start =
        ParseJsonUint64((*offsets_it)[0], name + ".data_offsets[0]");
    const uint64_t end =
        ParseJsonUint64((*offsets_it)[1], name + ".data_offsets[1]");
    if (end < start) {
      throw std::runtime_error("Safetensors data_offsets are invalid: " + name);
    }

    parsed.tensors.emplace(
        name, SafetensorsEntry{dtype_it->get<std::string>(),
                               ParseShape(*shape_it, name), start, end});
  }

  return parsed;
}

}  // namespace

Config DefaultSmolLM2_135M_InstructConfig() { return Config{}; }

std::filesystem::path ModelFilesDir(const std::filesystem::path& cwd) {
  return cwd / "model_files";
}

void RunSmolLM2DownloadSmokeTests(const std::filesystem::path& cwd) {
  const auto model_dir = ModelFilesDir(cwd);
  DownloadUrlToPath(kConfigJsonUrl, model_dir / "config.json");
  DownloadUrlToPath(kTokenizerConfigJsonUrl,
                    model_dir / "tokenizer_config.json");
}

std::filesystem::path DownloadSmolLM2_135M_InstructSafetensors(
    const std::filesystem::path& cwd) {
  RunSmolLM2DownloadSmokeTests(cwd);

  const auto model_dir = ModelFilesDir(cwd);
  const auto safetensors_path = model_dir / "model.safetensors";
  DownloadUrlToPath(kModelSafetensorsUrl, safetensors_path);
  return safetensors_path;
}

std::vector<WeightSpec> BuildWeightSpecs(const Config& config) {
  ValidateConfig(config);

  const uint64_t head_dim = HeadDim(config);
  const uint64_t kv_hidden_size = config.num_key_value_heads * head_dim;

  std::vector<WeightSpec> specs;
  specs.reserve(static_cast<size_t>(config.num_hidden_layers * 9 + 3));

  AddWeight(&specs, "model.embed_tokens.weight", "transformer.embed.weight",
            {config.vocab_size, config.hidden_size},
            {config.vocab_size, config.hidden_size}, false);

  for (uint64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    const std::string prefix = "model.layers." + std::to_string(layer);
    const std::string block_prefix =
        "transformer.blocks." + std::to_string(layer);

    AddWeight(&specs, prefix + ".input_layernorm.weight",
              block_prefix + ".attention_norm.weight", {config.hidden_size},
              {config.hidden_size}, false);

    AddWeight(&specs, prefix + ".self_attn.q_proj.weight",
              block_prefix + ".self_attention.q_weight",
              {config.hidden_size, config.hidden_size},
              {1, config.num_attention_heads, config.hidden_size, head_dim},
              true);
    AddWeight(&specs, prefix + ".self_attn.k_proj.weight",
              block_prefix + ".self_attention.k_weight",
              {kv_hidden_size, config.hidden_size},
              {1, config.num_key_value_heads, config.hidden_size, head_dim},
              true);
    AddWeight(&specs, prefix + ".self_attn.v_proj.weight",
              block_prefix + ".self_attention.v_weight",
              {kv_hidden_size, config.hidden_size},
              {1, config.num_key_value_heads, config.hidden_size, head_dim},
              true);
    AddWeight(&specs, prefix + ".self_attn.o_proj.weight",
              block_prefix + ".self_attention.o_weight",
              {config.hidden_size, config.hidden_size},
              {1, config.num_attention_heads, head_dim, config.hidden_size},
              true);

    AddWeight(&specs, prefix + ".post_attention_layernorm.weight",
              block_prefix + ".ffn_norm.weight", {config.hidden_size},
              {config.hidden_size}, false);

    AddWeight(&specs, prefix + ".mlp.gate_proj.weight",
              block_prefix + ".ffn.gate_proj.weight",
              {config.intermediate_size, config.hidden_size},
              {1, config.hidden_size, config.intermediate_size}, true);
    AddWeight(&specs, prefix + ".mlp.up_proj.weight",
              block_prefix + ".ffn.up_proj.weight",
              {config.intermediate_size, config.hidden_size},
              {1, config.hidden_size, config.intermediate_size}, true);
    AddWeight(&specs, prefix + ".mlp.down_proj.weight",
              block_prefix + ".ffn.down_proj.weight",
              {config.hidden_size, config.intermediate_size},
              {1, config.intermediate_size, config.hidden_size}, true);

    if (config.attention_bias) {
      AddWeight(&specs, prefix + ".self_attn.q_proj.bias",
                block_prefix + ".self_attention.q_bias", {config.hidden_size},
                {1, config.num_attention_heads, 1, head_dim}, false);
      AddWeight(&specs, prefix + ".self_attn.k_proj.bias",
                block_prefix + ".self_attention.k_bias", {kv_hidden_size},
                {1, config.num_key_value_heads, 1, head_dim}, false);
      AddWeight(&specs, prefix + ".self_attn.v_proj.bias",
                block_prefix + ".self_attention.v_bias", {kv_hidden_size},
                {1, config.num_key_value_heads, 1, head_dim}, false);
      AddWeight(&specs, prefix + ".self_attn.o_proj.bias",
                block_prefix + ".self_attention.o_bias", {config.hidden_size},
                {1, 1, config.hidden_size}, false);
    }

    if (config.mlp_bias) {
      AddWeight(&specs, prefix + ".mlp.gate_proj.bias",
                block_prefix + ".ffn.gate_proj.bias",
                {config.intermediate_size}, {1, 1, config.intermediate_size},
                false);
      AddWeight(&specs, prefix + ".mlp.up_proj.bias",
                block_prefix + ".ffn.up_proj.bias", {config.intermediate_size},
                {1, 1, config.intermediate_size}, false);
      AddWeight(&specs, prefix + ".mlp.down_proj.bias",
                block_prefix + ".ffn.down_proj.bias", {config.hidden_size},
                {1, 1, config.hidden_size}, false);
    }
  }

  AddWeight(&specs, "model.norm.weight", "transformer.norm.weight",
            {config.hidden_size}, {config.hidden_size}, false);

  AddWeight(&specs, "lm_head.weight", "lm_head.weight",
            {config.vocab_size, config.hidden_size},
            {config.vocab_size, config.hidden_size}, false,
            !config.tie_word_embeddings);

  return specs;
}

std::string ReadSafetensorsHeaderJson(
    const std::filesystem::path& safetensors_path) {
  return ParseSafetensorsHeader(safetensors_path).raw_json;
}

TensorPlacementPlan BuildTensorPlacementPlanFromSafetensors(
    const std::filesystem::path& safetensors_path, const Config& config) {
  ValidateConfig(config);

  const auto parsed = ParseSafetensorsHeader(safetensors_path);
  const auto specs = BuildWeightSpecs(config);

  TensorPlacementPlan plan;
  plan.safetensors_path = safetensors_path;
  plan.header_json_size = static_cast<uint64_t>(parsed.raw_json.size());
  plan.placements.reserve(specs.size());

  std::unordered_set<std::string> expected_names;
  expected_names.reserve(specs.size());

  for (const auto& spec : specs) {
    expected_names.insert(spec.hf_name);

    TensorPlacement placement;
    placement.hf_name = spec.hf_name;
    placement.deeptiny_target = spec.deeptiny_target;
    placement.expected_hf_shape = spec.hf_shape;
    placement.deeptiny_shape = spec.deeptiny_shape;
    placement.transpose_last_two = spec.transpose_last_two;
    placement.required = spec.required;

    const auto entry_it = parsed.tensors.find(spec.hf_name);
    if (entry_it != parsed.tensors.end()) {
      placement.present_in_safetensors = true;
      placement.dtype = entry_it->second.dtype;
      placement.safetensors_shape = entry_it->second.shape;
      placement.data_start = entry_it->second.data_start;
      placement.data_end = entry_it->second.data_end;
      placement.shape_matches_expected =
          (entry_it->second.shape == spec.hf_shape);
    } else {
      placement.present_in_safetensors = false;
      placement.shape_matches_expected = false;
      if (spec.required) {
        plan.missing_required_tensors.push_back(spec.hf_name);
      }
    }

    plan.placements.push_back(std::move(placement));
  }

  for (const auto& [name, _entry] : parsed.tensors) {
    if (!expected_names.contains(name)) {
      plan.unexpected_tensors.push_back(name);
    }
  }

  std::sort(plan.missing_required_tensors.begin(),
            plan.missing_required_tensors.end());
  std::sort(plan.unexpected_tensors.begin(), plan.unexpected_tensors.end());

  return plan;
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
