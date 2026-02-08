#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "deeptiny/nn/embedding.h"
#include "deeptiny/nn/module.h"
#include "deeptiny/nn/rms_norm.h"
#include "deeptiny/nn/transformer_block.h"
#include "deeptiny/tensor.h"
#include "deeptiny/types.h"

namespace transfomer_demo {

class Transformer : public deeptiny::nn::Module {
 public:
  Transformer(uint64_t vocab_size, uint64_t hidden_size,
              uint64_t intermediate_size, uint64_t num_blocks,
              uint64_t num_attention_heads, uint64_t num_key_value_heads,
              deeptiny::Device device = deeptiny::Device::CPU);

  deeptiny::Tensor operator()(
      const std::vector<std::vector<int64_t>>& tokens) const;

  uint64_t num_blocks() const;

  deeptiny::nn::Embedding& embed();
  const deeptiny::nn::Embedding& embed() const;
  deeptiny::nn::TransformerBlock& block(uint64_t index);
  const deeptiny::nn::TransformerBlock& block(uint64_t index) const;
  deeptiny::nn::RMSNorm& norm();
  const deeptiny::nn::RMSNorm& norm() const;

 private:
  deeptiny::nn::Embedding embed_;
  std::vector<std::unique_ptr<deeptiny::nn::TransformerBlock>> blocks_;
  deeptiny::nn::RMSNorm norm_;
};

}  // namespace transfomer_demo
