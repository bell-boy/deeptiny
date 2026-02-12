#include "deeptiny/nn/transformer_block.h"

#include <stdexcept>
#include <utility>

#include "nn/validation.h"

namespace deeptiny::nn {

TransformerBlock::TransformerBlock(
    uint64_t hidden_size, uint64_t mlp_hidden_dim, uint64_t num_attention_heads,
    uint64_t num_key_value_heads, bool attention_bias, bool mlp_bias,
    bool is_causal, float rope_theta, float norm_eps, Device device,
    HiddenAct mlp_hidden_act)
    : hidden_size_(detail::ValidateNonZeroDimension(
          "TransformerBlock", "hidden_size", hidden_size)),
      attention_norm_(hidden_size_, norm_eps, device),
      self_attention_(hidden_size_, num_attention_heads, num_key_value_heads,
                      attention_bias, is_causal, rope_theta, device),
      ffn_norm_(hidden_size_, norm_eps, device),
      ffn_(hidden_size_,
           detail::ValidateNonZeroDimension("TransformerBlock",
                                            "mlp_hidden_dim", mlp_hidden_dim),
           hidden_size_, mlp_bias, device, mlp_hidden_act) {
  RegisterSubmodule(attention_norm_);
  RegisterSubmodule(self_attention_);
  RegisterSubmodule(ffn_norm_);
  RegisterSubmodule(ffn_);
}

Tensor TransformerBlock::operator()(const Tensor& hidden_states,
                                    std::optional<Tensor> attention_mask,
                                    uint64_t position_offset) const {
  const auto& input_shape = hidden_states.shape();
  if (input_shape.size() != 3) {
    throw std::runtime_error("TransformerBlock expects input rank == 3");
  }
  if (input_shape[2] != hidden_size_) {
    throw std::runtime_error(
        "TransformerBlock input hidden dimension mismatch");
  }

  Tensor attention_input = attention_norm_(hidden_states);
  Tensor attention_output = self_attention_(
      attention_input, std::move(attention_mask), position_offset);
  Tensor hidden = attention_output;
  hidden += hidden_states;

  Tensor ffn_input = ffn_norm_(hidden);
  Tensor ffn_output = ffn_(ffn_input);
  ffn_output += hidden;
  return ffn_output;
}

RMSNorm& TransformerBlock::attention_norm() { return attention_norm_; }

const RMSNorm& TransformerBlock::attention_norm() const {
  return attention_norm_;
}

MultiHeadAttention& TransformerBlock::self_attention() {
  return self_attention_;
}

const MultiHeadAttention& TransformerBlock::self_attention() const {
  return self_attention_;
}

RMSNorm& TransformerBlock::ffn_norm() { return ffn_norm_; }

const RMSNorm& TransformerBlock::ffn_norm() const { return ffn_norm_; }

GatedReLU& TransformerBlock::ffn() { return ffn_; }

const GatedReLU& TransformerBlock::ffn() const { return ffn_; }

}  // namespace deeptiny::nn
