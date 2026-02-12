#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "deeptiny/nn/embedding.h"
#include "deeptiny/nn/kv_cache.h"
#include "deeptiny/nn/module.h"
#include "deeptiny/nn/rms_norm.h"
#include "deeptiny/nn/transformer_block.h"
#include "deeptiny/tensor.h"
#include "deeptiny/types.h"

namespace transfomer_demo {

class Transformer : public deeptiny::nn::Module {
 public:
  struct GenerationOptions {
    explicit GenerationOptions(
        uint64_t max_new_tokens = 64, float temperature = 0.8f,
        std::optional<uint64_t> eos_token_id = std::nullopt)
        : max_new_tokens(max_new_tokens),
          temperature(temperature),
          eos_token_id(eos_token_id) {}

    uint64_t max_new_tokens;
    float temperature;
    std::optional<uint64_t> eos_token_id;
  };

  Transformer(uint64_t vocab_size, uint64_t hidden_size,
              uint64_t intermediate_size, uint64_t num_blocks,
              uint64_t num_attention_heads, uint64_t num_key_value_heads,
              deeptiny::Device device = deeptiny::Device::CPU);

  deeptiny::Tensor operator()(
      const std::vector<std::vector<int64_t>>& tokens) const;
  std::vector<int64_t> Generate(
      const std::vector<int64_t>& prompt_tokens,
      const GenerationOptions& options = GenerationOptions(),
      std::mt19937* rng = nullptr) const;

  uint64_t num_blocks() const;

  deeptiny::nn::Embedding& embed();
  const deeptiny::nn::Embedding& embed() const;
  deeptiny::nn::TransformerBlock& block(uint64_t index);
  const deeptiny::nn::TransformerBlock& block(uint64_t index) const;
  deeptiny::nn::RMSNorm& norm();
  const deeptiny::nn::RMSNorm& norm() const;

 private:
  void ResetKVCache() const;
  deeptiny::Tensor ForwardChunkWithCache(
      const std::vector<int64_t>& flat_tokens,
      const deeptiny::Shape& token_shape, uint64_t position_offset) const;
  deeptiny::Tensor ComputeNextTokenLogitsFromHidden(
      const deeptiny::Tensor& hidden_states) const;

  uint64_t head_dim_;
  deeptiny::nn::Embedding embed_;
  std::vector<std::unique_ptr<deeptiny::nn::TransformerBlock>> blocks_;
  mutable std::vector<std::unique_ptr<deeptiny::nn::KVCache>> kv_caches_;
  deeptiny::nn::RMSNorm norm_;
};

}  // namespace transfomer_demo
