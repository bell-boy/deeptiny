#include "deeptiny/nn/multi_head_attention.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "deeptiny/functional.h"
#include "deeptiny/math.h"
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
  for (uint64_t query = 0; query < query_len; ++query) {
    const uint64_t query_position = query_position_offset + query;
    for (uint64_t key = 0; key < key_len; ++key) {
      const uint64_t key_position = key_position_offset + key;
      if (key_position > query_position) {
        values[static_cast<size_t>(query * key_len + key)] = kBlockedValue;
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

uint64_t NextPowerOfTwoCapacity(uint64_t required) {
  if (required == 0) {
    return 1;
  }
  uint64_t capacity = 1;
  while (capacity < required) {
    if (capacity > std::numeric_limits<uint64_t>::max() / 2) {
      throw std::runtime_error("KVCache capacity overflow");
    }
    capacity *= 2;
  }
  return capacity;
}

void ValidateKVChunk(const Tensor& key, const Tensor& value) {
  if (key.shape().size() != 4 || value.shape().size() != 4) {
    throw std::runtime_error("KVCache expects rank-4 key/value tensors");
  }
  if (key.shape() != value.shape()) {
    throw std::runtime_error("KVCache key/value shape mismatch");
  }
  if (key.dtype() != value.dtype()) {
    throw std::runtime_error("KVCache key/value dtype mismatch");
  }
  if (key.device() != value.device()) {
    throw std::runtime_error("KVCache key/value device mismatch");
  }
  if (key.shape()[2] == 0) {
    throw std::runtime_error("KVCache append chunk must have non-zero length");
  }
}

void CopySequenceIntoCacheStorage(const Tensor& src, uint64_t src_seq_len,
                                  uint64_t src_capacity, uint64_t dst_offset,
                                  uint64_t dst_capacity, Tensor* dst) {
  const auto& src_shape = src.shape();
  if (src_shape.size() != 4) {
    throw std::runtime_error("KVCache copy expects rank-4 source tensor");
  }
  const auto& dst_shape = dst->shape();
  if (dst_shape.size() != 4) {
    throw std::runtime_error("KVCache copy expects rank-4 destination tensor");
  }
  if (src_shape[0] != dst_shape[0] || src_shape[1] != dst_shape[1] ||
      src_shape[3] != dst_shape[3]) {
    throw std::runtime_error(
        "KVCache copy mismatch on non-sequence dimensions");
  }
  if (src_seq_len > src_capacity) {
    throw std::runtime_error(
        "KVCache source capacity smaller than src_seq_len");
  }
  if (dst_offset + src_seq_len > dst_capacity) {
    throw std::runtime_error(
        "KVCache destination capacity smaller than required copy range");
  }

  auto src_impl = utils::TensorAccessor::GetTensorImpl(src);
  auto src_storage = src_impl->getContiguousStorage();
  auto dst_impl = utils::TensorAccessor::GetTensorImpl(*dst);
  auto dst_storage = dst_impl->storage();

  const uint64_t head_dim = src_shape[3];
  const uint64_t rows = src_shape[0] * src_shape[1];
  const uint64_t copy_numel = src_seq_len * head_dim;
  if (copy_numel == 0) {
    return;
  }

  const uint64_t src_row_stride = src_capacity * head_dim;
  const uint64_t dst_row_stride = dst_capacity * head_dim;
  const uint64_t dst_seq_offset = dst_offset * head_dim;
  std::vector<std::byte> row_buffer(
      static_cast<size_t>(copy_numel * src.dtype().size()));
  for (uint64_t row = 0; row < rows; ++row) {
    const uint64_t src_row_offset = row * src_row_stride;
    const uint64_t dst_row_offset = row * dst_row_stride + dst_seq_offset;
    src_storage->CopyToHost(src_row_offset, copy_numel, row_buffer.data());
    dst_storage->CopyFromHost(dst_row_offset, copy_numel, row_buffer.data());
  }
}

}  // namespace

void KVCache::Append(const Tensor& key, const Tensor& value) {
  ValidateKVChunk(key, value);

  const auto& chunk_shape = key.shape();
  const uint64_t batch_size = chunk_shape[0];
  const uint64_t num_heads = chunk_shape[1];
  const uint64_t chunk_len = chunk_shape[2];
  const uint64_t head_dim = chunk_shape[3];

  if (keys_storage_.has_value()) {
    const auto& cached_shape = keys_storage_->shape();
    if (key.dtype() != keys_storage_->dtype() ||
        value.dtype() != values_storage_->dtype()) {
      throw std::runtime_error("KVCache append dtype mismatch");
    }
    if (key.device() != keys_storage_->device() ||
        value.device() != values_storage_->device()) {
      throw std::runtime_error("KVCache append device mismatch");
    }
    if (batch_size != cached_shape[0] || num_heads != cached_shape[1] ||
        head_dim != cached_shape[3]) {
      throw std::runtime_error(
          "KVCache append shape mismatch on non-sequence dimensions");
    }
  }

  const uint64_t required = length_ + chunk_len;
  if (!keys_storage_.has_value()) {
    capacity_ = NextPowerOfTwoCapacity(required);
    keys_storage_ = Tensor::Zeros({batch_size, num_heads, capacity_, head_dim},
                                  key.device(), key.dtype(), false);
    values_storage_ =
        Tensor::Zeros({batch_size, num_heads, capacity_, head_dim},
                      value.device(), value.dtype(), false);
  } else if (required > capacity_) {
    const uint64_t new_capacity = NextPowerOfTwoCapacity(required);
    Tensor resized_keys =
        Tensor::Zeros({batch_size, num_heads, new_capacity, head_dim},
                      key.device(), key.dtype(), false);
    Tensor resized_values =
        Tensor::Zeros({batch_size, num_heads, new_capacity, head_dim},
                      value.device(), value.dtype(), false);
    CopySequenceIntoCacheStorage(*keys_storage_, length_, capacity_,
                                 /*dst_offset=*/0, new_capacity, &resized_keys);
    CopySequenceIntoCacheStorage(*values_storage_, length_, capacity_,
                                 /*dst_offset=*/0, new_capacity,
                                 &resized_values);
    keys_storage_ = std::move(resized_keys);
    values_storage_ = std::move(resized_values);
    capacity_ = new_capacity;
  }

  CopySequenceIntoCacheStorage(key, chunk_len, chunk_len, length_, capacity_,
                               &(*keys_storage_));
  CopySequenceIntoCacheStorage(value, chunk_len, chunk_len, length_, capacity_,
                               &(*values_storage_));
  length_ += chunk_len;
}

Tensor KVCache::keys() const {
  if (!keys_storage_.has_value()) {
    throw std::runtime_error("KVCache keys requested before any append");
  }
  using deeptiny::Slice;
  return (*keys_storage_)({Slice::All(), Slice::All(),
                           Slice(0, static_cast<int64_t>(length_)),
                           Slice::All()});
}

Tensor KVCache::values() const {
  if (!values_storage_.has_value()) {
    throw std::runtime_error("KVCache values requested before any append");
  }
  using deeptiny::Slice;
  return (*values_storage_)({Slice::All(), Slice::All(),
                             Slice(0, static_cast<int64_t>(length_)),
                             Slice::All()});
}

bool KVCache::empty() const { return length_ == 0; }

uint64_t KVCache::size() const { return length_; }

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
  if (q_bias_.has_value()) {
    q = q + *q_bias_;
  }

  Tensor rope_rotations =
      BuildRoPERotationMatrices(query_len, head_dim_ / 2, position_offset,
                                rope_theta_, hidden_states.device());
  q = ApplyRoPE(q, rope_rotations);

  std::optional<Tensor> k;
  std::optional<Tensor> v;
  uint64_t key_position_offset = position_offset;
  if (kv_cache != nullptr) {
    Tensor k_new = math::BatchedMatMul(x_4d, k_weight_);
    Tensor v_new = math::BatchedMatMul(x_4d, v_weight_);
    if (k_bias_.has_value()) {
      k_new = k_new + *k_bias_;
    }
    if (v_bias_.has_value()) {
      v_new = v_new + *v_bias_;
    }
    k_new = ApplyRoPE(k_new, rope_rotations);
    kv_cache->Append(k_new, v_new);
    k = kv_cache->keys();
    v = kv_cache->values();
    key_position_offset = 0;
  } else {
    k = math::BatchedMatMul(x_4d, k_weight_);
    v = math::BatchedMatMul(x_4d, v_weight_);
    if (k_bias_.has_value()) {
      *k = *k + *k_bias_;
    }
    if (v_bias_.has_value()) {
      *v = *v + *v_bias_;
    }
    *k = ApplyRoPE(*k, rope_rotations);
  }

  if (!k.has_value() || !v.has_value()) {
    throw std::runtime_error("MultiHeadAttention internal key/value unset");
  }
  if (!utils::TensorAccessor::GetTensorImpl(*k)->isContiguous()) {
    *k = k->Clone();
  }
  if (!utils::TensorAccessor::GetTensorImpl(*v)->isContiguous()) {
    *v = v->Clone();
  }

  const uint64_t key_len = k->shape()[2];

  Tensor q_grouped = q.Reshape({batch_size, num_key_value_heads_,
                                num_key_value_groups_, query_len, head_dim_});
  Tensor k_grouped =
      k->Reshape({batch_size, num_key_value_heads_, 1, key_len, head_dim_});

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
  Tensor v_grouped =
      v->Reshape({batch_size, num_key_value_heads_, 1, key_len, head_dim_});
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
