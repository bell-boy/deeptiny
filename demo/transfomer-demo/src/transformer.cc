#include "transformer.h"

#include <cstdint>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

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

Transformer::Transformer(uint64_t vocab_size, uint64_t hidden_size,
                         uint64_t intermediate_size, uint64_t num_blocks,
                         uint64_t num_attention_heads,
                         uint64_t num_key_value_heads, deeptiny::Device device)
    : embed_(ValidateNonZero("vocab_size", vocab_size),
             ValidateNonZero("hidden_size", hidden_size),
             deeptiny::DType::Float32, device, true),
      norm_(hidden_size, 1.0e-5f, device) {
  ValidateNonZero("intermediate_size", intermediate_size);
  ValidateNonZero("num_blocks", num_blocks);
  ValidateNonZero("num_attention_heads", num_attention_heads);
  ValidateNonZero("num_key_value_heads", num_key_value_heads);

  RegisterSubmodule(embed_);

  blocks_.reserve(static_cast<size_t>(num_blocks));
  for (uint64_t i = 0; i < num_blocks; ++i) {
    blocks_.push_back(std::make_unique<deeptiny::nn::TransformerBlock>(
        hidden_size, intermediate_size, num_attention_heads,
        num_key_value_heads, false, true, true, 10000.0f, 1.0e-5f, device));
    RegisterSubmodule(*blocks_.back());
  }

  RegisterSubmodule(norm_);
}

deeptiny::Tensor Transformer::operator()(
    const std::vector<std::vector<int64_t>>& tokens) const {
  auto [flat_tokens, token_shape] = FlattenTokenBatch(tokens);
  deeptiny::Tensor hidden_states = embed_(flat_tokens, token_shape);
  for (const auto& block : blocks_) {
    hidden_states = (*block)(hidden_states, std::nullopt, 0);
  }
  return norm_(hidden_states);
}

uint64_t Transformer::num_blocks() const {
  return static_cast<uint64_t>(blocks_.size());
}

deeptiny::nn::Embedding& Transformer::embed() { return embed_; }

const deeptiny::nn::Embedding& Transformer::embed() const { return embed_; }

deeptiny::nn::TransformerBlock& Transformer::block(uint64_t index) {
  if (index >= blocks_.size()) {
    throw std::runtime_error("Transformer block index out of range.");
  }
  return *blocks_[index];
}

const deeptiny::nn::TransformerBlock& Transformer::block(uint64_t index) const {
  if (index >= blocks_.size()) {
    throw std::runtime_error("Transformer block index out of range.");
  }
  return *blocks_[index];
}

deeptiny::nn::RMSNorm& Transformer::norm() { return norm_; }

const deeptiny::nn::RMSNorm& Transformer::norm() const { return norm_; }

}  // namespace transfomer_demo
