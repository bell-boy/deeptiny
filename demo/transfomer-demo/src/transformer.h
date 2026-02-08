#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "deeptiny/nn/embedding.h"
#include "deeptiny/nn/gated_relu.h"
#include "deeptiny/nn/module.h"
#include "deeptiny/nn/multi_head_attention.h"
#include "deeptiny/nn/rms_norm.h"
#include "deeptiny/tensor.h"
#include "deeptiny/types.h"

namespace transfomer_demo {

class HiddenState : public deeptiny::nn::Module {
 public:
  HiddenState(uint64_t hidden_size, uint64_t intermediate_size,
              uint64_t num_attention_heads, uint64_t num_key_value_heads,
              deeptiny::Device device = deeptiny::Device::CPU);

  deeptiny::Tensor operator()(const deeptiny::Tensor& hidden_states) const;

 private:
  deeptiny::nn::RMSNorm attention_norm_;
  deeptiny::nn::MultiHeadAttention attention_;
  deeptiny::nn::RMSNorm mlp_norm_;
  deeptiny::nn::GatedReLU mlp_;
};

class Transformer : public deeptiny::nn::Module {
 public:
  Transformer(uint64_t vocab_size, uint64_t hidden_size,
              uint64_t intermediate_size, uint64_t num_hidden_states,
              uint64_t num_attention_heads, uint64_t num_key_value_heads,
              deeptiny::Device device = deeptiny::Device::CPU);

  deeptiny::Tensor operator()(
      const std::vector<std::vector<int64_t>>& tokens) const;

  uint64_t num_hidden_states() const;

  deeptiny::nn::Embedding& embed();
  const deeptiny::nn::Embedding& embed() const;
  deeptiny::nn::RMSNorm& norm();
  const deeptiny::nn::RMSNorm& norm() const;

 private:
  deeptiny::nn::Embedding embed_;
  std::vector<std::unique_ptr<HiddenState>> hidden_states_;
  deeptiny::nn::RMSNorm norm_;
};

}  // namespace transfomer_demo
