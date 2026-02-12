#include "deeptiny/nn/multi_head_attention.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "deeptiny/functional.h"
#include "deeptiny/math.h"
#include "deeptiny/nn/kv_cache.h"
#include "nn/validation.h"
#include "utils.h"

namespace deeptiny::nn {
namespace {

void ValidateConstructorArgs(uint64_t hidden_size, uint64_t num_attention_heads,
                             uint64_t num_key_value_heads, float rope_theta) {
  if (hidden_size % num_attention_heads != 0) {
    throw std::runtime_error(
        "MultiHeadAttention hidden_size must be divisible by "
        "num_attention_heads");
  }
  if (num_attention_heads % num_key_value_heads != 0) {
    throw std::runtime_error(
        "MultiHeadAttention num_attention_heads must be divisible by "
        "num_key_value_heads");
  }
  const uint64_t head_dim = hidden_size / num_attention_heads;
  if (head_dim % 2 != 0) {
    throw std::runtime_error(
        "MultiHeadAttention requires even head_dim for RoPE");
  }
  if (!(rope_theta > 0.0f)) {
    throw std::runtime_error("MultiHeadAttention rope_theta must be > 0");
  }
}

Tensor Scalar(float value, Device device) {
  return Tensor::FromVector<float>(std::vector<float>{value}, Shape{1}, device,
                                   false);
}

Tensor MakeCausalMask(uint64_t query_len, uint64_t key_len,
                      uint64_t query_position_offset,
                      uint64_t key_position_offset, Device device,
                      DType dtype) {
  if (dtype != DType::Float32) {
    throw std::runtime_error(
        "MultiHeadAttention causal mask supports only Float32");
  }

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

  return Tensor::FromVector(values, Shape{1, 1, query_len, key_len}, device,
                            false);
}

Tensor BuildRoPERotationMatrices(uint64_t seq_len, uint64_t half_dim,
                                 uint64_t position_offset, float rope_theta,
                                 Device device) {
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

  return Tensor::FromVector(
      rotation_values, Shape{1, 1, seq_len, half_dim, 2, 2}, device, false);
}

Tensor ApplyRoPE(const Tensor& x, const Tensor& rotations) {
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
  const Shape expected_rotation_shape{1, 1, seq_len, half_dim, 2, 2};
  if (rotations.shape() != expected_rotation_shape) {
    throw std::runtime_error("ApplyRoPE rotation shape mismatch");
  }

  utils::CompatabilityCheck({x, rotations});

  const Shape shape_6d{shape[0], shape[1], shape[2], half_dim, 1, 2};
  Tensor x_view = x;
  Tensor x_6d = x_view.Reshape(shape_6d);
  Tensor rotated_6d = math::BatchedMatMul(x_6d, rotations);
  return rotated_6d.Reshape(shape);
}

Tensor MakeGroupedQueryView(const Tensor& q, uint64_t num_key_value_heads,
                            uint64_t num_key_value_groups) {
  const auto& shape = q.shape();
  if (shape.size() != 4) {
    throw std::runtime_error("Query tensor must be rank-4");
  }
  if (shape[1] != num_key_value_heads * num_key_value_groups) {
    throw std::runtime_error(
        "Query tensor head dimension does not match grouped attention layout");
  }
  auto q_impl = utils::TensorAccessor::GetTensorImpl(q);
  const auto& stride = q_impl->stride();
  Stride grouped_stride{
      stride[0], stride[1] * static_cast<int64_t>(num_key_value_groups),
      stride[1], stride[2],
      stride[3],
  };
  return utils::TensorAccessor::MakeTensor(
      q_impl->View(Shape{shape[0], num_key_value_heads, num_key_value_groups,
                         shape[2], shape[3]},
                   std::move(grouped_stride), q_impl->offset()),
      nullptr);
}

Tensor MakeGroupedKeyValueView(const Tensor& kv) {
  const auto& shape = kv.shape();
  if (shape.size() != 4) {
    throw std::runtime_error("Key/value tensor must be rank-4");
  }
  auto kv_impl = utils::TensorAccessor::GetTensorImpl(kv);
  const auto& stride = kv_impl->stride();
  Stride grouped_stride{stride[0], stride[1], 0, stride[2], stride[3]};
  return utils::TensorAccessor::MakeTensor(
      kv_impl->View(Shape{shape[0], shape[1], 1, shape[2], shape[3]},
                    std::move(grouped_stride), kv_impl->offset()),
      nullptr);
}

}  // namespace

MultiHeadAttention::MultiHeadAttention(uint64_t hidden_size,
                                       uint64_t num_attention_heads,
                                       uint64_t num_key_value_heads,
                                       bool attention_bias, bool is_causal,
                                       float rope_theta, Device device)
    : hidden_size_(detail::ValidateNonZeroDimension(
          "MultiHeadAttention", "hidden_size", hidden_size)),
      num_attention_heads_(detail::ValidateNonZeroDimension(
          "MultiHeadAttention", "num_attention_heads", num_attention_heads)),
      num_key_value_heads_(detail::ValidateNonZeroDimension(
          "MultiHeadAttention", "num_key_value_heads", num_key_value_heads)),
      num_key_value_groups_(num_attention_heads_ / num_key_value_heads_),
      head_dim_(hidden_size_ / num_attention_heads_),
      is_causal_(is_causal),
      rope_theta_(rope_theta),
      q_weight_(Tensor::CreateUniform(
          {1, num_attention_heads_, hidden_size_, head_dim_}, device,
          DType::Float32, true)),
      k_weight_(Tensor::CreateUniform(
          {1, num_key_value_heads_, hidden_size_, head_dim_}, device,
          DType::Float32, true)),
      v_weight_(Tensor::CreateUniform(
          {1, num_key_value_heads_, hidden_size_, head_dim_}, device,
          DType::Float32, true)),
      o_weight_(Tensor::CreateUniform(
          {1, num_attention_heads_, head_dim_, hidden_size_}, device,
          DType::Float32, true)) {
  ValidateConstructorArgs(hidden_size_, num_attention_heads_,
                          num_key_value_heads_, rope_theta_);

  RegisterParameter(q_weight_);
  RegisterParameter(k_weight_);
  RegisterParameter(v_weight_);
  RegisterParameter(o_weight_);

  if (attention_bias) {
    q_bias_ = Tensor::CreateUniform({1, num_attention_heads_, 1, head_dim_},
                                    device, DType::Float32, true);
    k_bias_ = Tensor::CreateUniform({1, num_key_value_heads_, 1, head_dim_},
                                    device, DType::Float32, true);
    v_bias_ = Tensor::CreateUniform({1, num_key_value_heads_, 1, head_dim_},
                                    device, DType::Float32, true);
    o_bias_ = Tensor::CreateUniform({1, 1, hidden_size_}, device,
                                    DType::Float32, true);

    RegisterParameter(*q_bias_);
    RegisterParameter(*k_bias_);
    RegisterParameter(*v_bias_);
    RegisterParameter(*o_bias_);
  }
}

Tensor MultiHeadAttention::operator()(const Tensor& hidden_states,
                                      std::optional<Tensor> attention_mask,
                                      uint64_t position_offset,
                                      KVCache* kv_cache) const {
  const auto& input_shape = hidden_states.shape();
  if (input_shape.size() != 3) {
    throw std::runtime_error("MultiHeadAttention expects input rank == 3");
  }
  if (input_shape[2] != hidden_size_) {
    throw std::runtime_error(
        "MultiHeadAttention input hidden dimension mismatch");
  }

  utils::CompatabilityCheck(
      {hidden_states, q_weight_, k_weight_, v_weight_, o_weight_});

  const uint64_t batch_size = input_shape[0];
  const uint64_t query_len = input_shape[1];

  Tensor hidden_view = hidden_states;
  Tensor x_4d = hidden_view.Reshape({batch_size, 1, query_len, hidden_size_});
  Tensor q = math::BatchedMatMul(x_4d, q_weight_);
  Tensor k = math::BatchedMatMul(x_4d, k_weight_);
  Tensor v = math::BatchedMatMul(x_4d, v_weight_);

  if (q_bias_.has_value()) {
    q = q + *q_bias_;
    k = k + *k_bias_;
    v = v + *v_bias_;
  }

  Tensor rope_rotations =
      BuildRoPERotationMatrices(query_len, head_dim_ / 2, position_offset,
                                rope_theta_, hidden_states.device());
  q = ApplyRoPE(q, rope_rotations);
  k = ApplyRoPE(k, rope_rotations);

  Tensor q_grouped;
  Tensor k_grouped;
  Tensor v_grouped;
  uint64_t key_len = query_len;
  uint64_t key_position_offset = position_offset;

  if (kv_cache == nullptr) {
    q_grouped = q.Reshape({batch_size, num_key_value_heads_,
                           num_key_value_groups_, query_len, head_dim_});
    k_grouped =
        k.Reshape({batch_size, num_key_value_heads_, 1, query_len, head_dim_});
    v_grouped =
        v.Reshape({batch_size, num_key_value_heads_, 1, query_len, head_dim_});
  } else {
    if (position_offset != kv_cache->seq_len()) {
      throw std::runtime_error(
          "MultiHeadAttention position_offset must match KV cache sequence "
          "length");
    }
    kv_cache->update(k, v);
    Tensor k_for_attention = kv_cache->keys();
    Tensor v_for_attention = kv_cache->values();
    key_position_offset = 0;
    const auto& key_shape = k_for_attention.shape();
    const auto& value_shape = v_for_attention.shape();
    if (key_shape.size() != 4 || value_shape.size() != 4) {
      throw std::runtime_error(
          "MultiHeadAttention expects rank-4 key/value tensors");
    }
    if (key_shape[0] != batch_size || key_shape[1] != num_key_value_heads_ ||
        key_shape[3] != head_dim_) {
      throw std::runtime_error("MultiHeadAttention key shape mismatch");
    }
    if (value_shape != key_shape) {
      throw std::runtime_error(
          "MultiHeadAttention expects key/value tensors to share shape");
    }
    key_len = key_shape[2];
    q_grouped =
        MakeGroupedQueryView(q, num_key_value_heads_, num_key_value_groups_);
    k_grouped = MakeGroupedKeyValueView(k_for_attention);
    v_grouped = MakeGroupedKeyValueView(v_for_attention);
  }

  Tensor scores_grouped =
      math::BatchedMatMul(q_grouped, k_grouped, false, true);
  Tensor scores = scores_grouped.Reshape(
      {batch_size, num_attention_heads_, query_len, key_len});

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
  scores = scores * Scalar(scale, hidden_states.device());

  if (is_causal_) {
    scores =
        scores + MakeCausalMask(query_len, key_len, position_offset,
                                key_position_offset, hidden_states.device(),
                                hidden_states.dtype());
  }

  if (attention_mask.has_value()) {
    utils::CompatabilityCheck({scores, *attention_mask});
    auto broadcasted_mask =
        utils::BroadcastToShape(*attention_mask, scores.shape());
    if (!broadcasted_mask) {
      throw std::runtime_error(
          "MultiHeadAttention attention_mask is not broadcastable to score "
          "shape");
    }
    scores = scores + *broadcasted_mask;
  }

  Tensor probs = functional::Softmax(scores, scores.shape().size() - 1);
  Tensor probs_grouped =
      probs.Reshape({batch_size, num_key_value_heads_, num_key_value_groups_,
                     query_len, key_len});
  Tensor context_grouped = math::BatchedMatMul(probs_grouped, v_grouped);
  Tensor context = context_grouped.Reshape(
      {batch_size, num_attention_heads_, query_len, head_dim_});

  Tensor projected = math::BatchedMatMul(context, o_weight_);
  Tensor out = functional::Reduce(projected, {1}, true).Squeeze({1});

  if (o_bias_.has_value()) {
    out = out + *o_bias_;
  }
  return out;
}

Tensor& MultiHeadAttention::q_weight() { return q_weight_; }

const Tensor& MultiHeadAttention::q_weight() const { return q_weight_; }

Tensor& MultiHeadAttention::k_weight() { return k_weight_; }

const Tensor& MultiHeadAttention::k_weight() const { return k_weight_; }

Tensor& MultiHeadAttention::v_weight() { return v_weight_; }

const Tensor& MultiHeadAttention::v_weight() const { return v_weight_; }

Tensor& MultiHeadAttention::o_weight() { return o_weight_; }

const Tensor& MultiHeadAttention::o_weight() const { return o_weight_; }

std::optional<Tensor>& MultiHeadAttention::q_bias() { return q_bias_; }

const std::optional<Tensor>& MultiHeadAttention::q_bias() const {
  return q_bias_;
}

std::optional<Tensor>& MultiHeadAttention::k_bias() { return k_bias_; }

const std::optional<Tensor>& MultiHeadAttention::k_bias() const {
  return k_bias_;
}

std::optional<Tensor>& MultiHeadAttention::v_bias() { return v_bias_; }

const std::optional<Tensor>& MultiHeadAttention::v_bias() const {
  return v_bias_;
}

std::optional<Tensor>& MultiHeadAttention::o_bias() { return o_bias_; }

const std::optional<Tensor>& MultiHeadAttention::o_bias() const {
  return o_bias_;
}

}  // namespace deeptiny::nn
