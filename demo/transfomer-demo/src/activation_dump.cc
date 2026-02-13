#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "deeptiny/autograd.h"
#include "deeptiny/math.h"
#include "deeptiny/tensor.h"
#include "smollm2_135m_instruct_loader.h"
#include "transformer.h"

namespace {

std::vector<int64_t> ParseTokensCsv(const std::string& csv) {
  std::vector<int64_t> tokens;
  size_t start = 0;
  while (start < csv.size()) {
    size_t end = csv.find(',', start);
    if (end == std::string::npos) {
      end = csv.size();
    }
    std::string token = csv.substr(start, end - start);
    const size_t first = token.find_first_not_of(" \t\r\n");
    const size_t last = token.find_last_not_of(" \t\r\n");
    if (first == std::string::npos || last == std::string::npos) {
      throw std::runtime_error("Token CSV contains an empty token segment.");
    }
    token = token.substr(first, last - first + 1);

    size_t parsed = 0;
    const int64_t value = std::stoll(token, &parsed, 10);
    if (parsed != token.size()) {
      throw std::runtime_error("Failed to parse token id: " + token);
    }
    tokens.push_back(value);
    start = end + 1;
  }

  if (tokens.empty()) {
    throw std::runtime_error("Token CSV must contain at least one token.");
  }
  return tokens;
}

std::string SanitizeFileStem(std::string_view name) {
  std::string out;
  out.reserve(name.size());
  for (const unsigned char c : name) {
    if (std::isalnum(c) != 0 || c == '_' || c == '-') {
      out.push_back(static_cast<char>(c));
    } else {
      out.push_back('_');
    }
  }
  return out;
}

std::vector<float> TensorToFloatVector(const deeptiny::Tensor& tensor) {
  if (tensor.dtype() != deeptiny::DType::Float32) {
    throw std::runtime_error("Activation dump expects Float32 tensors.");
  }

  std::vector<float> values(static_cast<size_t>(tensor.numel()), 0.0f);
  tensor.CopyToBuffer(
      std::as_writable_bytes(std::span<float>(values.data(), values.size())),
      tensor.shape(), deeptiny::DType::Float32);
  return values;
}

void WriteFloatBinary(const std::filesystem::path& path,
                      const std::vector<float>& values) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open output file: " + path.string());
  }
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!out.good()) {
    throw std::runtime_error("Failed to write output file: " + path.string());
  }
}

class ActivationDumper {
 public:
  explicit ActivationDumper(std::filesystem::path output_dir)
      : output_dir_(std::move(output_dir)) {
    std::filesystem::create_directories(output_dir_);
    manifest_["activations"] = nlohmann::ordered_json::object();
  }

  void SetTokens(const std::vector<int64_t>& tokens) {
    manifest_["tokens"] = tokens;
  }

  void Dump(std::string_view name, const deeptiny::Tensor& tensor) {
    const std::string key(name);
    const std::string file_name = SanitizeFileStem(name) + ".f32.bin";
    const std::filesystem::path file_path = output_dir_ / file_name;

    const std::vector<float> values = TensorToFloatVector(tensor);
    WriteFloatBinary(file_path, values);

    nlohmann::ordered_json entry;
    entry["dtype"] = "float32";
    entry["shape"] = tensor.shape();
    entry["numel"] = tensor.numel();
    entry["file"] = file_name;
    manifest_["activations"][key] = std::move(entry);
  }

  void WriteManifest() const {
    const std::filesystem::path manifest_path = output_dir_ / "manifest.json";
    std::ofstream out(manifest_path, std::ios::trunc);
    if (!out.is_open()) {
      throw std::runtime_error("Failed to write manifest: " +
                               manifest_path.string());
    }
    out << manifest_.dump(2) << "\n";
  }

 private:
  std::filesystem::path output_dir_;
  nlohmann::ordered_json manifest_;
};

deeptiny::Tensor ComputeLogits(const transfomer_demo::Transformer& model,
                               const deeptiny::Tensor& hidden_states) {
  const deeptiny::Shape& hidden_shape = hidden_states.shape();
  if (hidden_shape.size() != 3) {
    throw std::runtime_error(
        "Hidden states must have shape [batch, seq, hidden].");
  }

  deeptiny::Tensor embedding_weight = model.embed().weight();
  const deeptiny::Shape& embedding_shape = embedding_weight.shape();
  if (embedding_shape.size() != 2) {
    throw std::runtime_error(
        "Embedding weight must have shape [vocab, hidden].");
  }
  if (embedding_shape[1] != hidden_shape[2]) {
    throw std::runtime_error(
        "Embedding hidden size does not match hidden state size.");
  }

  deeptiny::Tensor tied =
      embedding_weight.Reshape({1, embedding_shape[0], embedding_shape[1]});
  return deeptiny::math::BatchedMatMul(hidden_states, tied, false, true);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      std::cerr << "Usage: " << argv[0]
                << " <model_dir> <output_dir> <tokens_csv>\n";
      return 1;
    }

    const std::filesystem::path model_dir(argv[1]);
    const std::filesystem::path output_dir(argv[2]);
    const std::vector<int64_t> tokens = ParseTokensCsv(argv[3]);

    auto model = demo::smollm2::CreateSmolLM2_135M_InstructTransformer(
        model_dir, demo::smollm2::DefaultSmolLM2_135M_InstructConfig());

    deeptiny::NoGrad no_grad;
    ActivationDumper dumper(output_dir);
    dumper.SetTokens(tokens);

    deeptiny::Tensor hidden_states = model->embed()(
        tokens, deeptiny::Shape{1, static_cast<uint64_t>(tokens.size())});
    dumper.Dump("model.embed_tokens", hidden_states);

    for (uint64_t layer = 0; layer < model->num_blocks(); ++layer) {
      const std::string prefix = "model.layers." + std::to_string(layer);
      const auto& block = model->block(layer);

      deeptiny::Tensor attn_input = block.attention_norm()(hidden_states);
      dumper.Dump(prefix + ".input_layernorm", attn_input);

      deeptiny::Tensor attn_output =
          block.self_attention()(attn_input, std::nullopt, 0, nullptr);
      dumper.Dump(prefix + ".self_attn", attn_output);

      deeptiny::Tensor post_attn = attn_output + hidden_states;
      dumper.Dump(prefix + ".post_attn_residual", post_attn);

      deeptiny::Tensor ffn_input = block.ffn_norm()(post_attn);
      dumper.Dump(prefix + ".post_attention_layernorm", ffn_input);

      deeptiny::Tensor mlp_output = block.ffn()(ffn_input);
      dumper.Dump(prefix + ".mlp", mlp_output);

      hidden_states = mlp_output + post_attn;
      dumper.Dump(prefix + ".output", hidden_states);
    }

    deeptiny::Tensor norm_output = model->norm()(hidden_states);
    dumper.Dump("model.norm", norm_output);
    dumper.Dump("last_hidden_state", norm_output);

    deeptiny::Tensor logits = ComputeLogits(*model, norm_output);
    dumper.Dump("logits", logits);

    dumper.WriteManifest();
    std::cout << "Wrote local activations to " << output_dir << "\n";
  } catch (const std::exception& err) {
    std::cerr << "activation dump failed: " << err.what() << "\n";
    return 1;
  }

  return 0;
}
