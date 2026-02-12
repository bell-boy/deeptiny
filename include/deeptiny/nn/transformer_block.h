#pragma once

#include <cstdint>
#include <optional>

#include "deeptiny/nn/gated_relu.h"
#include "deeptiny/nn/module.h"
#include "deeptiny/nn/multi_head_attention.h"
#include "deeptiny/nn/rms_norm.h"
#include "deeptiny/tensor.h"
#include "deeptiny/types.h"

namespace deeptiny::nn {

class KVCache;

class TransformerBlock : public Module {
 public:
  TransformerBlock(uint64_t hidden_size, uint64_t mlp_hidden_dim,
                   uint64_t num_attention_heads, uint64_t num_key_value_heads,
                   bool attention_bias = false, bool mlp_bias = false,
                   bool is_causal = true, float rope_theta = 10000.0f,
                   float norm_eps = 1e-5f, Device device = Device::CPU);

  Tensor operator()(const Tensor& hidden_states,
                    std::optional<Tensor> attention_mask = std::nullopt,
                    uint64_t position_offset = 0,
                    KVCache* kv_cache = nullptr) const;

  RMSNorm& attention_norm();
  const RMSNorm& attention_norm() const;
  MultiHeadAttention& self_attention();
  const MultiHeadAttention& self_attention() const;
  RMSNorm& ffn_norm();
  const RMSNorm& ffn_norm() const;
  GatedReLU& ffn();
  const GatedReLU& ffn() const;

 private:
  uint64_t hidden_size_;
  RMSNorm attention_norm_;
  MultiHeadAttention self_attention_;
  RMSNorm ffn_norm_;
  GatedReLU ffn_;
};

}  // namespace deeptiny::nn
