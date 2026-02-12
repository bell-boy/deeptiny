#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "deeptiny/autograd.h"
#include "deeptiny/functional.h"
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
    throw std::runtime_error(
        "Self-attention debug dump expects Float32 tensors.");
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

class TensorDumper {
 public:
  explicit TensorDumper(std::filesystem::path output_dir)
      : output_dir_(std::move(output_dir)) {
    std::filesystem::create_directories(output_dir_);
    manifest_["tensors"] = nlohmann::ordered_json::object();
  }

  void SetTokens(const std::vector<int64_t>& tokens) {
    manifest_["tokens"] = tokens;
  }

  void SetLayer(uint64_t layer) { manifest_["layer"] = layer; }

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
    manifest_["tensors"][key] = std::move(entry);
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

deeptiny::Tensor Scalar(float value, deeptiny::Device device) {
  return deeptiny::Tensor::FromVector<float>(std::vector<float>{value}, {1},
                                             device, false);
}

deeptiny::Tensor MakeCausalMask(uint64_t query_len, uint64_t key_len,
                                uint64_t query_position_offset,
                                uint64_t key_position_offset,
                                deeptiny::Device device) {
  std::vector<float> values(static_cast<size_t>(query_len * key_len), 0.0f);
  constexpr float kBlockedValue = -1.0e9f;
  for (uint64_t q = 0; q < query_len; ++q) {
    const uint64_t query_abs_pos = query_position_offset + q;
    for (uint64_t k = 0; k < key_len; ++k) {
      const uint64_t key_abs_pos = key_position_offset + k;
      if (key_abs_pos > query_abs_pos) {
        values[static_cast<size_t>(q * key_len + k)] = kBlockedValue;
      }
    }
  }

  return deeptiny::Tensor::FromVector(
      values, deeptiny::Shape{1, 1, query_len, key_len}, device, false);
}

deeptiny::Tensor BuildRoPERotationMatrices(uint64_t seq_len, uint64_t half_dim,
                                           uint64_t position_offset,
                                           float rope_theta,
                                           deeptiny::Device device) {
  std::vector<float> rotation_values(
      static_cast<size_t>(seq_len * half_dim * 4), 0.0f);
  const double full_dim = static_cast<double>(half_dim * 2);

  for (uint64_t pos = 0; pos < seq_len; ++pos) {
    const double position = static_cast<double>(position_offset + pos);
    for (uint64_t i = 0; i < half_dim; ++i) {
      const double exponent = -2.0 * static_cast<double>(i) / full_dim;
      const double inv_freq =
          std::pow(static_cast<double>(rope_theta), exponent);
      const double angle = position * inv_freq;
      const float cos_value = static_cast<float>(std::cos(angle));
      const float sin_value = static_cast<float>(std::sin(angle));
      const size_t base = static_cast<size_t>((pos * half_dim + i) * 4);
      rotation_values[base] = cos_value;
      rotation_values[base + 1] = -sin_value;
      rotation_values[base + 2] = sin_value;
      rotation_values[base + 3] = cos_value;
    }
  }

  return deeptiny::Tensor::FromVector(
      rotation_values, deeptiny::Shape{1, 1, seq_len, half_dim, 2, 2}, device,
      false);
}

deeptiny::Tensor BuildIdentityMatrix(uint64_t dim, deeptiny::Device device) {
  std::vector<float> values(static_cast<size_t>(dim * dim), 0.0f);
  for (uint64_t i = 0; i < dim; ++i) {
    values[static_cast<size_t>(i * dim + i)] = 1.0f;
  }
  return deeptiny::Tensor::FromVector(
      values, deeptiny::Shape{1, 1, 1, dim, dim}, device, false);
}

deeptiny::Tensor ApplyRoPE(const deeptiny::Tensor& x,
                           const deeptiny::Tensor& rotations,
                           bool rope_interleaved,
                           const deeptiny::Tensor& transpose_identity_2,
                           const deeptiny::Tensor& transpose_identity_half) {
  const auto& shape = x.shape();
  if (shape.size() != 4) {
    throw std::runtime_error("ApplyRoPE expects a rank-4 tensor");
  }

  const uint64_t seq_len = shape[2];
  const uint64_t head_dim = shape[3];
  if (head_dim % 2 != 0) {
    throw std::runtime_error("ApplyRoPE requires even head_dim");
  }
  const uint64_t half_dim = head_dim / 2;
  const deeptiny::Shape expected_rotation_shape{1, 1, seq_len, half_dim, 2, 2};
  if (rotations.shape() != expected_rotation_shape) {
    throw std::runtime_error("ApplyRoPE rotation shape mismatch");
  }

  if (rope_interleaved) {
    const deeptiny::Shape shape_6d{shape[0], shape[1], shape[2],
                                   half_dim, 1,        2};
    deeptiny::Tensor x_view = x;
    deeptiny::Tensor x_6d = x_view.Reshape(shape_6d);
    deeptiny::Tensor rotated_6d =
        deeptiny::math::BatchedMatMul(x_6d, rotations);
    return rotated_6d.Reshape(shape);
  }

  deeptiny::Tensor x_view = x;
  deeptiny::Tensor split_non_interleaved =
      x_view.Reshape({shape[0], shape[1], shape[2], 2, half_dim});
  deeptiny::Tensor interleaved_pairs = deeptiny::math::BatchedMatMul(
      split_non_interleaved, transpose_identity_2, /*transpose_a=*/true,
      /*transpose_b=*/false);
  deeptiny::Tensor interleaved_pairs_6d =
      interleaved_pairs.Reshape({shape[0], shape[1], shape[2], half_dim, 1, 2});
  deeptiny::Tensor rotated_pairs_6d =
      deeptiny::math::BatchedMatMul(interleaved_pairs_6d, rotations);
  deeptiny::Tensor rotated_pairs =
      rotated_pairs_6d.Reshape({shape[0], shape[1], shape[2], half_dim, 2});
  deeptiny::Tensor split_non_interleaved_rotated =
      deeptiny::math::BatchedMatMul(rotated_pairs, transpose_identity_half,
                                    /*transpose_a=*/true,
                                    /*transpose_b=*/false);
  return split_non_interleaved_rotated.Reshape(shape);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      std::cerr << "Usage: " << argv[0]
                << " <model_dir> <output_dir> <tokens_csv> <layer_index>\n";
      return 1;
    }

    const std::filesystem::path model_dir(argv[1]);
    const std::filesystem::path output_dir(argv[2]);
    const std::vector<int64_t> tokens = ParseTokensCsv(argv[3]);
    const uint64_t layer_index = static_cast<uint64_t>(std::stoull(argv[4]));

    const auto config = demo::smollm2::DefaultSmolLM2_135M_InstructConfig();
    auto model = demo::smollm2::CreateSmolLM2_135M_InstructTransformer(
        model_dir, config);
    if (layer_index >= model->num_blocks()) {
      throw std::runtime_error(
          "Layer index out of range for model block count.");
    }

    deeptiny::NoGrad no_grad;
    TensorDumper dumper(output_dir);
    dumper.SetTokens(tokens);
    dumper.SetLayer(layer_index);

    deeptiny::Tensor hidden_states = model->embed()(
        tokens, deeptiny::Shape{1, static_cast<uint64_t>(tokens.size())});
    for (uint64_t i = 0; i < layer_index; ++i) {
      hidden_states = model->block(i)(hidden_states, std::nullopt, 0, nullptr);
    }
    dumper.Dump("hidden_in", hidden_states);

    const auto& block = model->block(layer_index);
    const auto& attn = block.self_attention();
    deeptiny::Tensor attn_input = block.attention_norm()(hidden_states);
    dumper.Dump("attn_input", attn_input);

    dumper.Dump("q_weight", attn.q_weight());
    dumper.Dump("k_weight", attn.k_weight());
    dumper.Dump("v_weight", attn.v_weight());
    dumper.Dump("o_weight", attn.o_weight());

    const auto& input_shape = attn_input.shape();
    const uint64_t batch_size = input_shape[0];
    const uint64_t query_len = input_shape[1];
    const uint64_t hidden_size = input_shape[2];
    const uint64_t head_dim = config.hidden_size / config.num_attention_heads;
    const uint64_t key_len = query_len;

    deeptiny::Tensor x_4d =
        attn_input.Reshape({batch_size, 1, query_len, hidden_size});
    deeptiny::Tensor q = deeptiny::math::BatchedMatMul(x_4d, attn.q_weight());
    deeptiny::Tensor k = deeptiny::math::BatchedMatMul(x_4d, attn.k_weight());
    deeptiny::Tensor v = deeptiny::math::BatchedMatMul(x_4d, attn.v_weight());

    if (attn.q_bias().has_value()) {
      q = q + *attn.q_bias();
      k = k + *attn.k_bias();
      v = v + *attn.v_bias();
    }

    dumper.Dump("q_linear", q);
    dumper.Dump("k_linear", k);
    dumper.Dump("v_linear", v);

    deeptiny::Tensor transpose_identity_2 =
        BuildIdentityMatrix(2, attn_input.device());
    deeptiny::Tensor transpose_identity_half =
        BuildIdentityMatrix(head_dim / 2, attn_input.device());
    deeptiny::Tensor rope_rotations = BuildRoPERotationMatrices(
        query_len, head_dim / 2, 0, config.rope_theta, attn_input.device());
    dumper.Dump("rope_rotations", rope_rotations);

    deeptiny::Tensor q_rope =
        ApplyRoPE(q, rope_rotations, config.rope_interleaved,
                  transpose_identity_2, transpose_identity_half);
    deeptiny::Tensor k_rope =
        ApplyRoPE(k, rope_rotations, config.rope_interleaved,
                  transpose_identity_2, transpose_identity_half);
    dumper.Dump("q_rope", q_rope);
    dumper.Dump("k_rope", k_rope);

    deeptiny::Tensor q_grouped =
        q_rope.Reshape({batch_size, config.num_key_value_heads,
                        config.num_attention_heads / config.num_key_value_heads,
                        query_len, head_dim});
    deeptiny::Tensor k_grouped = k_rope.Reshape(
        {batch_size, config.num_key_value_heads, 1, key_len, head_dim});
    deeptiny::Tensor v_grouped = v.Reshape(
        {batch_size, config.num_key_value_heads, 1, key_len, head_dim});

    dumper.Dump("q_grouped", q_grouped);
    dumper.Dump("k_grouped", k_grouped);
    dumper.Dump("v_grouped", v_grouped);

    deeptiny::Tensor scores_grouped =
        deeptiny::math::BatchedMatMul(q_grouped, k_grouped, false, true);
    deeptiny::Tensor scores = scores_grouped.Reshape(
        {batch_size, config.num_attention_heads, query_len, key_len});
    dumper.Dump("scores_pre_scale", scores);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    deeptiny::Tensor scores_scaled =
        scores * Scalar(scale, attn_input.device());
    dumper.Dump("scores_scaled", scores_scaled);

    deeptiny::Tensor causal_mask =
        MakeCausalMask(query_len, key_len, 0, 0, attn_input.device());
    dumper.Dump("causal_mask", causal_mask);

    deeptiny::Tensor scores_masked = scores_scaled + causal_mask;
    dumper.Dump("scores_masked", scores_masked);

    deeptiny::Tensor probs = deeptiny::functional::Softmax(
        scores_masked, scores_masked.shape().size() - 1);
    dumper.Dump("probs", probs);

    deeptiny::Tensor probs_grouped =
        probs.Reshape({batch_size, config.num_key_value_heads,
                       config.num_attention_heads / config.num_key_value_heads,
                       query_len, key_len});
    deeptiny::Tensor context_grouped =
        deeptiny::math::BatchedMatMul(probs_grouped, v_grouped);
    deeptiny::Tensor context = context_grouped.Reshape(
        {batch_size, config.num_attention_heads, query_len, head_dim});
    dumper.Dump("context", context);

    deeptiny::Tensor projected =
        deeptiny::math::BatchedMatMul(context, attn.o_weight());
    dumper.Dump("projected", projected);

    deeptiny::Tensor self_attn_out =
        deeptiny::functional::Reduce(projected, {1}, true).Squeeze({1});
    if (attn.o_bias().has_value()) {
      self_attn_out = self_attn_out + *attn.o_bias();
    }
    dumper.Dump("self_attn_out_manual", self_attn_out);

    deeptiny::Tensor self_attn_out_direct =
        attn(attn_input, std::nullopt, 0, nullptr);
    dumper.Dump("self_attn_out_direct", self_attn_out_direct);

    dumper.WriteManifest();
    std::cout << "Wrote self-attention debug dump to " << output_dir << "\n";
  } catch (const std::exception& err) {
    std::cerr << "self-attention debug dump failed: " << err.what() << "\n";
    return 1;
  }

  return 0;
}
