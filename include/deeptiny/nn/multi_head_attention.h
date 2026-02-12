#pragma once

#include <cstdint>
#include <optional>

#include "deeptiny/nn/module.h"
#include "deeptiny/tensor.h"
#include "deeptiny/types.h"

namespace deeptiny::nn {

class KVCache;

class MultiHeadAttention : public Module {
 public:
  MultiHeadAttention(uint64_t hidden_size, uint64_t num_attention_heads,
                     uint64_t num_key_value_heads, bool attention_bias = false,
                     bool is_causal = true, float rope_theta = 10000.0f,
                     Device device = Device::CPU);

  Tensor operator()(const Tensor& hidden_states,
                    std::optional<Tensor> attention_mask = std::nullopt,
                    uint64_t position_offset = 0,
                    KVCache* kv_cache = nullptr) const;

  Tensor& q_weight();
  const Tensor& q_weight() const;
  Tensor& k_weight();
  const Tensor& k_weight() const;
  Tensor& v_weight();
  const Tensor& v_weight() const;
  Tensor& o_weight();
  const Tensor& o_weight() const;
  std::optional<Tensor>& q_bias();
  const std::optional<Tensor>& q_bias() const;
  std::optional<Tensor>& k_bias();
  const std::optional<Tensor>& k_bias() const;
  std::optional<Tensor>& v_bias();
  const std::optional<Tensor>& v_bias() const;
  std::optional<Tensor>& o_bias();
  const std::optional<Tensor>& o_bias() const;

 private:
  uint64_t hidden_size_;
  uint64_t num_attention_heads_;
  uint64_t num_key_value_heads_;
  uint64_t num_key_value_groups_;
  uint64_t head_dim_;
  bool is_causal_;
  float rope_theta_;

  Tensor q_weight_;
  Tensor k_weight_;
  Tensor v_weight_;
  Tensor o_weight_;
  std::optional<Tensor> q_bias_;
  std::optional<Tensor> k_bias_;
  std::optional<Tensor> v_bias_;
  std::optional<Tensor> o_bias_;
};

}  // namespace deeptiny::nn
