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
#include "utils.h"

namespace deeptiny::nn {
namespace {

uint64_t ValidateNonZero(uint64_t value, const char* name) {
  if (value == 0) {
    throw std::runtime_error(std::string("MultiHeadAttention ") + name +
                             " must be non-zero");
  }
  return value;
}

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

void CopyTensorData(const Tensor& src, const Tensor& dst, const char* label) {
  utils::CompatabilityCheck({src, dst});
  if (src.shape() != dst.shape()) {
    throw std::runtime_error(std::string("MultiHeadAttention ") + label +
                             " shape mismatch");
  }

  auto src_impl = utils::TensorAccessor::GetTensorImpl(src);
  auto dst_impl = utils::TensorAccessor::GetTensorImpl(dst);
  auto src_storage = src_impl->getContiguousStorage();
  const uint64_t numel = src_storage->numel();
  std::vector<std::byte> host_buffer(
      static_cast<size_t>(numel * src.dtype().size()));
  src_storage->CopyToHost(0, numel, host_buffer.data());
  dst_impl->storage()->CopyFromHost(0, numel, host_buffer.data());
}

Tensor Scalar(float value, Device device) {
  return Tensor::FromVector<float>(std::vector<float>{value}, Shape{1}, device,
                                   false);
}

Tensor MakeCausalMask(uint64_t seq_len, Device device, DType dtype) {
  if (dtype != DType::Float32) {
    throw std::runtime_error(
        "MultiHeadAttention causal mask supports only Float32");
  }

  std::vector<float> values(static_cast<size_t>(seq_len * seq_len), 0.0f);
  constexpr float kBlockedValue = -1.0e9f;
  for (uint64_t q = 0; q < seq_len; ++q) {
    for (uint64_t k = q + 1; k < seq_len; ++k) {
      values[static_cast<size_t>(q * seq_len + k)] = kBlockedValue;
    }
  }

  return Tensor::FromVector(values, Shape{1, 1, seq_len, seq_len}, device,
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

}  // namespace

MultiHeadAttention::MultiHeadAttention(uint64_t hidden_size,
                                       uint64_t num_attention_heads,
                                       uint64_t num_key_value_heads,
                                       bool attention_bias, bool is_causal,
                                       float rope_theta, Device device)
    : hidden_size_(ValidateNonZero(hidden_size, "hidden_size")),
      num_attention_heads_(
          ValidateNonZero(num_attention_heads, "num_attention_heads")),
      num_key_value_heads_(
          ValidateNonZero(num_key_value_heads, "num_key_value_heads")),
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
                                      uint64_t position_offset) const {
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
  const uint64_t seq_len = input_shape[1];

  Tensor hidden_view = hidden_states;
  Tensor x_4d = hidden_view.Reshape({batch_size, 1, seq_len, hidden_size_});
  Tensor q = math::BatchedMatMul(x_4d, q_weight_);
  Tensor k = math::BatchedMatMul(x_4d, k_weight_);
  Tensor v = math::BatchedMatMul(x_4d, v_weight_);

  if (q_bias_.has_value()) {
    q = q + *q_bias_;
    k = k + *k_bias_;
    v = v + *v_bias_;
  }

  Tensor rope_rotations =
      BuildRoPERotationMatrices(seq_len, head_dim_ / 2, position_offset,
                                rope_theta_, hidden_states.device());
  q = ApplyRoPE(q, rope_rotations);
  k = ApplyRoPE(k, rope_rotations);

  Tensor q_grouped = q.Reshape({batch_size, num_key_value_heads_,
                                num_key_value_groups_, seq_len, head_dim_});
  Tensor k_grouped =
      k.Reshape({batch_size, num_key_value_heads_, 1, seq_len, head_dim_});

  Tensor scores_grouped =
      math::BatchedMatMul(q_grouped, k_grouped, false, true);
  Tensor scores = scores_grouped.Reshape(
      {batch_size, num_attention_heads_, seq_len, seq_len});

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
  scores = scores * Scalar(scale, hidden_states.device());

  if (is_causal_) {
    scores = scores + MakeCausalMask(seq_len, hidden_states.device(),
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
                     seq_len, seq_len});
  Tensor v_grouped =
      v.Reshape({batch_size, num_key_value_heads_, 1, seq_len, head_dim_});
  Tensor context_grouped = math::BatchedMatMul(probs_grouped, v_grouped);
  Tensor context = context_grouped.Reshape(
      {batch_size, num_attention_heads_, seq_len, head_dim_});

  Tensor projected = math::BatchedMatMul(context, o_weight_);
  Tensor out = functional::Reduce(projected, {1}, true).Squeeze({1});

  if (o_bias_.has_value()) {
    out = out + *o_bias_;
  }
  return out;
}

Tensor MultiHeadAttention::q_weight() const { return q_weight_; }

Tensor MultiHeadAttention::k_weight() const { return k_weight_; }

Tensor MultiHeadAttention::v_weight() const { return v_weight_; }

Tensor MultiHeadAttention::o_weight() const { return o_weight_; }

std::optional<Tensor> MultiHeadAttention::q_bias() const { return q_bias_; }

std::optional<Tensor> MultiHeadAttention::k_bias() const { return k_bias_; }

std::optional<Tensor> MultiHeadAttention::v_bias() const { return v_bias_; }

std::optional<Tensor> MultiHeadAttention::o_bias() const { return o_bias_; }

void MultiHeadAttention::set_q_weight(const Tensor& weight) {
  CopyTensorData(weight, q_weight_, "q_weight");
}

void MultiHeadAttention::set_k_weight(const Tensor& weight) {
  CopyTensorData(weight, k_weight_, "k_weight");
}

void MultiHeadAttention::set_v_weight(const Tensor& weight) {
  CopyTensorData(weight, v_weight_, "v_weight");
}

void MultiHeadAttention::set_o_weight(const Tensor& weight) {
  CopyTensorData(weight, o_weight_, "o_weight");
}

void MultiHeadAttention::set_q_bias(const Tensor& bias) {
  if (!q_bias_.has_value()) {
    throw std::runtime_error(
        "MultiHeadAttention was constructed without attention bias");
  }
  CopyTensorData(bias, *q_bias_, "q_bias");
}

void MultiHeadAttention::set_k_bias(const Tensor& bias) {
  if (!k_bias_.has_value()) {
    throw std::runtime_error(
        "MultiHeadAttention was constructed without attention bias");
  }
  CopyTensorData(bias, *k_bias_, "k_bias");
}

void MultiHeadAttention::set_v_bias(const Tensor& bias) {
  if (!v_bias_.has_value()) {
    throw std::runtime_error(
        "MultiHeadAttention was constructed without attention bias");
  }
  CopyTensorData(bias, *v_bias_, "v_bias");
}

void MultiHeadAttention::set_o_bias(const Tensor& bias) {
  if (!o_bias_.has_value()) {
    throw std::runtime_error(
        "MultiHeadAttention was constructed without attention bias");
  }
  CopyTensorData(bias, *o_bias_, "o_bias");
}

}  // namespace deeptiny::nn
