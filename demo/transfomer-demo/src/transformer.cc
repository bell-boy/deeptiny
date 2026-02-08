#include "transformer.h"

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "deeptiny/math.h"

namespace transfomer_demo {
namespace {

uint64_t ValidateNonZero(const char* field, uint64_t value) {
  if (value == 0) {
    std::stringstream err;
    err << "Transformer " << field << " must be non-zero.";
    throw std::runtime_error(err.str());
  }
  return value;
}

std::pair<std::vector<int64_t>, deeptiny::Shape> FlattenTokenBatch(
    const std::vector<std::vector<int64_t>>& tokens) {
  if (tokens.empty()) {
    throw std::runtime_error("Transformer token batch must be non-empty.");
  }

  const uint64_t seq_len = static_cast<uint64_t>(tokens.front().size());
  if (seq_len == 0) {
    throw std::runtime_error(
        "Transformer token batch sequences must be non-empty.");
  }

  std::vector<int64_t> flat_tokens;
  flat_tokens.reserve(tokens.size() * static_cast<size_t>(seq_len));

  for (size_t batch_index = 0; batch_index < tokens.size(); ++batch_index) {
    const auto& sequence = tokens[batch_index];
    if (sequence.size() != seq_len) {
      std::stringstream err;
      err << "Transformer expected equal sequence lengths, but batch "
          << batch_index << " has length " << sequence.size() << " (expected "
          << seq_len << ").";
      throw std::runtime_error(err.str());
    }
    flat_tokens.insert(flat_tokens.end(), sequence.begin(), sequence.end());
  }

  const uint64_t batch_size = static_cast<uint64_t>(tokens.size());
  return {std::move(flat_tokens), deeptiny::Shape{batch_size, seq_len}};
}

}  // namespace

HiddenState::HiddenState(uint64_t hidden_size, uint64_t intermediate_size,
                         uint64_t num_attention_heads,
                         uint64_t num_key_value_heads, deeptiny::Device device)
    : attention_norm_(hidden_size, 1.0e-5f, device),
      attention_(hidden_size, num_attention_heads, num_key_value_heads, false,
                 true, 10000.0f, device),
      mlp_norm_(hidden_size, 1.0e-5f, device),
      mlp_(hidden_size, intermediate_size, hidden_size, true, device) {
  RegisterSubmodule(attention_norm_);
  RegisterSubmodule(attention_);
  RegisterSubmodule(mlp_norm_);
  RegisterSubmodule(mlp_);
}

deeptiny::Tensor HiddenState::operator()(
    const deeptiny::Tensor& hidden_states) const {
  deeptiny::Tensor attention_input = attention_norm_(hidden_states);
  deeptiny::Tensor attended = hidden_states + attention_(attention_input);
  deeptiny::Tensor mlp_input = mlp_norm_(attended);
  return attended + mlp_(mlp_input);
}

Transformer::Transformer(uint64_t vocab_size, uint64_t hidden_size,
                         uint64_t intermediate_size, uint64_t num_hidden_states,
                         uint64_t num_attention_heads,
                         uint64_t num_key_value_heads, deeptiny::Device device)
    : embed_(ValidateNonZero("vocab_size", vocab_size),
             ValidateNonZero("hidden_size", hidden_size),
             deeptiny::DType::Float32, device, true),
      norm_(hidden_size, 1.0e-5f, device) {
  ValidateNonZero("intermediate_size", intermediate_size);
  ValidateNonZero("num_hidden_states", num_hidden_states);
  ValidateNonZero("num_attention_heads", num_attention_heads);
  ValidateNonZero("num_key_value_heads", num_key_value_heads);

  RegisterSubmodule(embed_);

  hidden_states_.reserve(static_cast<size_t>(num_hidden_states));
  for (uint64_t i = 0; i < num_hidden_states; ++i) {
    hidden_states_.push_back(std::make_unique<HiddenState>(
        hidden_size, intermediate_size, num_attention_heads,
        num_key_value_heads, device));
    RegisterSubmodule(*hidden_states_.back());
  }

  RegisterSubmodule(norm_);
}

deeptiny::Tensor Transformer::operator()(
    const std::vector<std::vector<int64_t>>& tokens) const {
  auto [flat_tokens, token_shape] = FlattenTokenBatch(tokens);
  deeptiny::Tensor hidden_states = embed_(flat_tokens, token_shape);
  for (const auto& hidden_state : hidden_states_) {
    hidden_states = (*hidden_state)(hidden_states);
  }
  return norm_(hidden_states);
}

uint64_t Transformer::num_hidden_states() const {
  return static_cast<uint64_t>(hidden_states_.size());
}

deeptiny::nn::Embedding& Transformer::embed() { return embed_; }

const deeptiny::nn::Embedding& Transformer::embed() const { return embed_; }

deeptiny::nn::RMSNorm& Transformer::norm() { return norm_; }

const deeptiny::nn::RMSNorm& Transformer::norm() const { return norm_; }

}  // namespace transfomer_demo
