#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <thread>
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
  class TokenStream {
   public:
    TokenStream(const TokenStream&) = delete;
    TokenStream& operator=(const TokenStream&) = delete;
    TokenStream(TokenStream&& other) noexcept;
    TokenStream& operator=(TokenStream&& other) noexcept;
    ~TokenStream();

    bool WaitNext(int64_t* token);
    void Join();

   private:
    struct SharedState;

    TokenStream(std::thread worker, std::shared_ptr<SharedState> shared_state);

    std::thread worker_;
    std::shared_ptr<SharedState> shared_state_;

    friend class Transformer;
  };

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
              deeptiny::Device device = deeptiny::Device::CPU,
              deeptiny::nn::GatedMLP::HiddenAct mlp_hidden_act =
                  deeptiny::nn::GatedMLP::HiddenAct::ReLU);

  deeptiny::Tensor operator()(
      const std::vector<std::vector<int64_t>>& tokens) const;
  std::vector<int64_t> Generate(
      const std::vector<int64_t>& prompt_tokens,
      const GenerationOptions& options = GenerationOptions(),
      std::mt19937* rng = nullptr) const;
  TokenStream GenerateAsync(
      const std::vector<int64_t>& prompt_tokens,
      const GenerationOptions& options = GenerationOptions(),
      std::optional<uint64_t> seed = std::nullopt) const;

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
  void GenerateWithCallback(const std::vector<int64_t>& prompt_tokens,
                            const GenerationOptions& options, std::mt19937* rng,
                            const std::function<void(int64_t)>& on_token) const;

  uint64_t head_dim_;
  deeptiny::nn::Embedding embed_;
  std::vector<std::unique_ptr<deeptiny::nn::TransformerBlock>> blocks_;
  mutable std::vector<std::unique_ptr<deeptiny::nn::KVCache>> kv_caches_;
  mutable std::mutex generation_mutex_;
  deeptiny::nn::RMSNorm norm_;
};

}  // namespace transfomer_demo
