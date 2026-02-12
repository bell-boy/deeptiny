#include "smollm2_135m_instruct_loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include "deeptiny/tensor.h"
#include "transformer.h"

namespace demo::smollm2 {
namespace {

constexpr char kModelSafetensorsUrl[] =
    "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/"
    "model.safetensors?download=true";
constexpr char kTokenizerJsonUrl[] =
    "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/"
    "tokenizer.json?download=true";

struct SafetensorsEntry {
  std::string dtype;
  deeptiny::Shape shape;
  uint64_t data_start = 0;
  uint64_t data_end = 0;
};

struct ParsedSafetensorsHeader {
  uint64_t header_size = 0;
  std::unordered_map<std::string, SafetensorsEntry> tensors;
};

class MappedFile {
 public:
  explicit MappedFile(const std::filesystem::path& path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
      throw std::runtime_error(
          "Failed to open file for mmap: " + path.string() + " (" +
          std::generic_category().message(errno) + ")");
    }

    struct stat st{};
    if (fstat(fd_, &st) != 0) {
      const std::string err = std::generic_category().message(errno);
      close(fd_);
      fd_ = -1;
      throw std::runtime_error(
          "Failed to stat file for mmap: " + path.string() + " (" + err + ")");
    }

    if (st.st_size <= 0) {
      close(fd_);
      fd_ = -1;
      throw std::runtime_error("Cannot mmap empty file: " + path.string());
    }

    size_ = static_cast<uint64_t>(st.st_size);
    void* mapped = mmap(nullptr, static_cast<size_t>(size_), PROT_READ,
                        MAP_PRIVATE, fd_, 0);
    if (mapped == MAP_FAILED) {
      const std::string err = std::generic_category().message(errno);
      close(fd_);
      fd_ = -1;
      throw std::runtime_error("Failed to mmap file: " + path.string() + " (" +
                               err + ")");
    }

    data_ = static_cast<const std::byte*>(mapped);
  }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  ~MappedFile() {
    if (data_ != nullptr) {
      munmap(const_cast<std::byte*>(data_), static_cast<size_t>(size_));
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  const std::byte* data() const { return data_; }
  uint64_t size() const { return size_; }

 private:
  int fd_ = -1;
  const std::byte* data_ = nullptr;
  uint64_t size_ = 0;
};

uint64_t HeadDim(const Config& config) {
  return config.hidden_size / config.num_attention_heads;
}

uint64_t Numel(const deeptiny::Shape& shape) {
  uint64_t total = 1;
  for (const uint64_t dim : shape) {
    total *= dim;
  }
  return total;
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

deeptiny::nn::GatedMLP::HiddenAct ParseHiddenAct(
    const std::string& hidden_act) {
  if (hidden_act == "relu") {
    return deeptiny::nn::GatedMLP::HiddenAct::ReLU;
  }
  if (hidden_act == "silu") {
    return deeptiny::nn::GatedMLP::HiddenAct::SiLU;
  }
  throw std::runtime_error(
      "SmolLM2 config hidden_act must be one of: relu, silu");
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
  ParseHiddenAct(config.hidden_act);
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

uint64_t ParseHeaderLength(const std::byte* data, uint64_t size) {
  if (size < 8) {
    throw std::runtime_error("Safetensors file is too small to contain header");
  }
  uint64_t value = 0;
  for (size_t i = 0; i < 8; ++i) {
    value |= static_cast<uint64_t>(std::to_integer<unsigned char>(data[i]))
             << (8U * i);
  }
  return value;
}

uint64_t BytesPerElement(std::string_view dtype) {
  if (dtype == "F32") {
    return 4;
  }
  if (dtype == "BF16") {
    return 2;
  }
  return 0;
}

ParsedSafetensorsHeader ParseSafetensorsHeader(const MappedFile& mapped) {
  ParsedSafetensorsHeader parsed;
  parsed.header_size = ParseHeaderLength(mapped.data(), mapped.size());

  if (parsed.header_size == 0) {
    throw std::runtime_error("Safetensors header length must be non-zero");
  }
  if (parsed.header_size > mapped.size() - 8) {
    throw std::runtime_error("Safetensors header length exceeds file size");
  }

  const auto* header_ptr = reinterpret_cast<const char*>(mapped.data() + 8);
  std::string header_json(header_ptr, header_ptr + parsed.header_size);

  nlohmann::json parsed_json;
  try {
    parsed_json = nlohmann::json::parse(header_json);
  } catch (const nlohmann::json::parse_error& err) {
    throw std::runtime_error("Failed to parse safetensors header JSON: " +
                             std::string(err.what()));
  }

  if (!parsed_json.is_object()) {
    throw std::runtime_error("Safetensors header JSON root must be an object");
  }

  const uint64_t payload_size = mapped.size() - 8 - parsed.header_size;

  for (const auto& [name, value] : parsed_json.items()) {
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

    const auto shape = ParseShape(*shape_it, name);
    const uint64_t start =
        ParseJsonUint64((*offsets_it)[0], name + ".data_offsets[0]");
    const uint64_t end =
        ParseJsonUint64((*offsets_it)[1], name + ".data_offsets[1]");
    if (end < start) {
      throw std::runtime_error("Safetensors data_offsets are invalid: " + name);
    }
    if (end > payload_size) {
      throw std::runtime_error(
          "Safetensors data_offsets exceed payload bounds: " + name);
    }

    const std::string dtype = dtype_it->get<std::string>();
    const uint64_t elem_size = BytesPerElement(dtype);
    if (elem_size != 0) {
      const uint64_t expected_bytes = Numel(shape) * elem_size;
      if ((end - start) != expected_bytes) {
        throw std::runtime_error("Safetensors tensor byte-size mismatch for " +
                                 name);
      }
    }

    parsed.tensors.emplace(
        name, SafetensorsEntry{dtype, std::move(shape), start, end});
  }

  return parsed;
}

std::filesystem::path ResolveSafetensorsPath(
    const std::filesystem::path& model_dir) {
  if (model_dir.empty()) {
    throw std::runtime_error("SmolLM2 loader requires a non-empty model_dir");
  }
  if (!std::filesystem::exists(model_dir) ||
      !std::filesystem::is_directory(model_dir)) {
    throw std::runtime_error("SmolLM2 loader could not find model directory: " +
                             model_dir.string());
  }

  const auto single_file = model_dir / "model.safetensors";
  if (std::filesystem::exists(single_file) &&
      std::filesystem::is_regular_file(single_file)) {
    return single_file;
  }

  const auto sharded_index = model_dir / "model.safetensors.index.json";
  if (std::filesystem::exists(sharded_index) &&
      std::filesystem::is_regular_file(sharded_index)) {
    throw std::runtime_error(
        "Sharded safetensors checkpoints are not supported yet: " +
        sharded_index.string());
  }

  throw std::runtime_error("SmolLM2 loader expected model.safetensors in " +
                           model_dir.string());
}

float ReadFloat32At(const std::byte* src, uint64_t index) {
  float value = 0.0f;
  std::memcpy(&value, src + index * 4, sizeof(float));
  return value;
}

uint16_t ReadUint16LeAt(const std::byte* src, uint64_t index) {
  const auto* bytes = reinterpret_cast<const unsigned char*>(src + index * 2);
  return static_cast<uint16_t>(bytes[0]) |
         (static_cast<uint16_t>(bytes[1]) << 8U);
}

float ReadBFloat16AsFloat32At(const std::byte* src, uint64_t index) {
  const uint16_t bf16_bits = ReadUint16LeAt(src, index);
  const uint32_t f32_bits = static_cast<uint32_t>(bf16_bits) << 16U;
  return std::bit_cast<float>(f32_bits);
}

float ReadSourceAsFloat32(const std::byte* src, uint64_t index,
                          std::string_view dtype) {
  if (dtype == "F32") {
    return ReadFloat32At(src, index);
  }
  if (dtype == "BF16") {
    return ReadBFloat16AsFloat32At(src, index);
  }
  throw std::runtime_error("Unsupported source dtype: " + std::string(dtype));
}

std::vector<float> ConvertToFloat32(const std::byte* src, uint64_t numel,
                                    std::string_view dtype) {
  std::vector<float> converted(numel, 0.0f);
  for (uint64_t i = 0; i < numel; ++i) {
    converted[static_cast<size_t>(i)] = ReadSourceAsFloat32(src, i, dtype);
  }
  return converted;
}

std::vector<float> TransposeLastTwoToFloat32(const std::byte* src,
                                             const deeptiny::Shape& src_shape,
                                             std::string_view dtype) {
  if (src_shape.size() < 2) {
    throw std::runtime_error("Transpose requires tensor rank >= 2");
  }

  const uint64_t numel = Numel(src_shape);
  std::vector<float> transposed(numel, 0.0f);

  uint64_t prefix = 1;
  for (size_t i = 0; i + 2 < src_shape.size(); ++i) {
    prefix *= src_shape[i];
  }

  const uint64_t dim_m = src_shape[src_shape.size() - 2];
  const uint64_t dim_n = src_shape[src_shape.size() - 1];
  const uint64_t block = dim_m * dim_n;

  for (uint64_t p = 0; p < prefix; ++p) {
    const uint64_t src_base = p * block;
    const uint64_t dst_base = p * block;
    for (uint64_t n = 0; n < dim_n; ++n) {
      for (uint64_t m = 0; m < dim_m; ++m) {
        const uint64_t src_index = src_base + m * dim_n + n;
        const uint64_t dst_index = dst_base + n * dim_m + m;
        transposed[static_cast<size_t>(dst_index)] =
            ReadSourceAsFloat32(src, src_index, dtype);
      }
    }
  }

  return transposed;
}

std::unordered_map<std::string, deeptiny::Tensor*> BuildTargetTensorMap(
    transfomer_demo::Transformer& model, const Config& config) {
  if (model.num_blocks() != config.num_hidden_layers) {
    throw std::runtime_error(
        "Transformer block count does not match SmolLM2 config");
  }

  std::unordered_map<std::string, deeptiny::Tensor*> tensors;
  tensors.reserve(static_cast<size_t>(config.num_hidden_layers * 9 + 2));

  tensors.emplace("model.embed_tokens.weight", &model.embed().weight());

  for (uint64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    auto& block = model.block(layer);
    const std::string prefix = "model.layers." + std::to_string(layer);

    tensors.emplace(prefix + ".input_layernorm.weight",
                    &block.attention_norm().weight());
    tensors.emplace(prefix + ".self_attn.q_proj.weight",
                    &block.self_attention().q_weight());
    tensors.emplace(prefix + ".self_attn.k_proj.weight",
                    &block.self_attention().k_weight());
    tensors.emplace(prefix + ".self_attn.v_proj.weight",
                    &block.self_attention().v_weight());
    tensors.emplace(prefix + ".self_attn.o_proj.weight",
                    &block.self_attention().o_weight());
    tensors.emplace(prefix + ".post_attention_layernorm.weight",
                    &block.ffn_norm().weight());
    tensors.emplace(prefix + ".mlp.gate_proj.weight",
                    &block.ffn().gate_proj().weight());
    tensors.emplace(prefix + ".mlp.up_proj.weight",
                    &block.ffn().up_proj().weight());
    tensors.emplace(prefix + ".mlp.down_proj.weight",
                    &block.ffn().down_proj().weight());

    if (config.attention_bias) {
      auto& attn = block.self_attention();
      if (!attn.q_bias().has_value() || !attn.k_bias().has_value() ||
          !attn.v_bias().has_value() || !attn.o_bias().has_value()) {
        throw std::runtime_error(
            "SmolLM2 config requests attention bias, but transformer "
            "does not expose required bias tensors");
      }
      tensors.emplace(prefix + ".self_attn.q_proj.bias", &*attn.q_bias());
      tensors.emplace(prefix + ".self_attn.k_proj.bias", &*attn.k_bias());
      tensors.emplace(prefix + ".self_attn.v_proj.bias", &*attn.v_bias());
      tensors.emplace(prefix + ".self_attn.o_proj.bias", &*attn.o_bias());
    }

    if (config.mlp_bias) {
      auto& ffn = block.ffn();
      if (!ffn.gate_proj().bias().has_value() ||
          !ffn.up_proj().bias().has_value() ||
          !ffn.down_proj().bias().has_value()) {
        throw std::runtime_error(
            "SmolLM2 config requests MLP bias, but transformer "
            "does not expose required bias tensors");
      }
      tensors.emplace(prefix + ".mlp.gate_proj.bias", &*ffn.gate_proj().bias());
      tensors.emplace(prefix + ".mlp.up_proj.bias", &*ffn.up_proj().bias());
      tensors.emplace(prefix + ".mlp.down_proj.bias", &*ffn.down_proj().bias());
    }
  }

  tensors.emplace("model.norm.weight", &model.norm().weight());
  return tensors;
}

void ValidateShape(const deeptiny::Shape& actual,
                   const deeptiny::Shape& expected, const std::string& name,
                   const char* context) {
  if (actual != expected) {
    std::stringstream err;
    err << context << " shape mismatch for " << name << ": expected "
        << deeptiny::FormatShape(expected) << " but got "
        << deeptiny::FormatShape(actual);
    throw std::runtime_error(err.str());
  }
}

void LoadOneTensor(const WeightSpec& spec, const SafetensorsEntry& entry,
                   const std::byte* payload_base, deeptiny::Tensor* target) {
  const uint64_t src_numel = Numel(entry.shape);
  const uint64_t dst_numel = target->numel();
  if (src_numel != dst_numel) {
    throw std::runtime_error("Element-count mismatch while loading tensor " +
                             spec.hf_name);
  }

  const std::byte* src = payload_base + entry.data_start;
  const uint64_t src_bytes = entry.data_end - entry.data_start;

  if (!spec.transpose_last_two && entry.dtype == "F32") {
    target->CopyFromBuffer(
        std::span<const std::byte>(src, static_cast<size_t>(src_bytes)),
        target->shape(), deeptiny::DType::Float32);
    return;
  }

  std::vector<float> f32_values;
  if (spec.transpose_last_two) {
    f32_values = TransposeLastTwoToFloat32(src, entry.shape, entry.dtype);
  } else {
    f32_values = ConvertToFloat32(src, src_numel, entry.dtype);
  }

  target->CopyFromBuffer(std::as_bytes(std::span<const float>(
                             f32_values.data(), f32_values.size())),
                         target->shape(), deeptiny::DType::Float32);
}

}  // namespace

Config DefaultSmolLM2_135M_InstructConfig() { return Config{}; }

std::filesystem::path ModelFilesDir(const std::filesystem::path& cwd) {
  return cwd / "model_files";
}

std::filesystem::path DownloadSmolLM2_135M_InstructSafetensors(
    const std::filesystem::path& cwd) {
  const auto model_dir = ModelFilesDir(cwd);
  const auto safetensors_path = model_dir / "model.safetensors";
  DownloadUrlToPath(kModelSafetensorsUrl, safetensors_path);
  return safetensors_path;
}

std::filesystem::path DownloadSmolLM2_135M_InstructTokenizerJson(
    const std::filesystem::path& cwd) {
  const auto model_dir = ModelFilesDir(cwd);
  const auto tokenizer_path = model_dir / "tokenizer.json";
  DownloadUrlToPath(kTokenizerJsonUrl, tokenizer_path);
  return tokenizer_path;
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

std::unique_ptr<transfomer_demo::Transformer>
CreateSmolLM2_135M_InstructTransformerUninitialized(const Config& config) {
  ValidateConfig(config);
  const deeptiny::nn::GatedMLP::HiddenAct hidden_act =
      ParseHiddenAct(config.hidden_act);
  return std::make_unique<transfomer_demo::Transformer>(
      config.vocab_size, config.hidden_size, config.intermediate_size,
      config.num_hidden_layers, config.num_attention_heads,
      config.num_key_value_heads, deeptiny::Device::CPU, hidden_act);
}

std::unique_ptr<transfomer_demo::Transformer>
CreateSmolLM2_135M_InstructTransformer(const std::filesystem::path& model_dir,
                                       const Config& config) {
  auto model = CreateSmolLM2_135M_InstructTransformerUninitialized(config);

  const auto safetensors_path = ResolveSafetensorsPath(model_dir);
  const MappedFile mapped(safetensors_path);
  const auto parsed = ParseSafetensorsHeader(mapped);

  const auto specs = BuildWeightSpecs(config);
  auto target_tensors = BuildTargetTensorMap(*model, config);

  const std::byte* payload_base = mapped.data() + 8 + parsed.header_size;

  for (const auto& spec : specs) {
    const auto entry_it = parsed.tensors.find(spec.hf_name);
    if (entry_it == parsed.tensors.end()) {
      if (spec.required) {
        throw std::runtime_error("Missing required tensor in safetensors: " +
                                 spec.hf_name);
      }
      continue;
    }

    const auto target_it = target_tensors.find(spec.hf_name);
    if (target_it == target_tensors.end()) {
      if (spec.required) {
        throw std::runtime_error("Missing required target tensor in model: " +
                                 spec.hf_name);
      }
      continue;
    }

    ValidateShape(entry_it->second.shape, spec.hf_shape, spec.hf_name,
                  "Safetensors");
    if (entry_it->second.dtype != "F32" && entry_it->second.dtype != "BF16") {
      throw std::runtime_error("Unsupported dtype for tensor " + spec.hf_name +
                               ": " + entry_it->second.dtype);
    }
    ValidateShape(target_it->second->shape(), spec.deeptiny_shape, spec.hf_name,
                  "Target tensor");
    if (target_it->second->dtype() != deeptiny::DType::Float32) {
      throw std::runtime_error("Target tensor must be Float32 for tensor " +
                               spec.hf_name);
    }

    LoadOneTensor(spec, entry_it->second, payload_base, target_it->second);
  }

  return model;
}

}  // namespace demo::smollm2
