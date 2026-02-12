#include "transformer.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "deeptiny/autograd.h"
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

std::vector<float> TensorToFloatVector(const deeptiny::Tensor& tensor) {
  if (tensor.dtype() != deeptiny::DType::Float32) {
    throw std::runtime_error("TensorToFloatVector expects Float32 tensor.");
  }

  std::vector<float> values(static_cast<size_t>(tensor.numel()), 0.0f);
  tensor.CopyToBuffer(
      std::as_writable_bytes(std::span<float>(values.data(), values.size())),
      tensor.shape(), deeptiny::DType::Float32);
  return values;
}

uint64_t ArgmaxIndex(const std::vector<float>& logits) {
  if (logits.empty()) {
    throw std::runtime_error("Cannot sample from empty logits.");
  }

  size_t best_idx = 0;
  float best_value = logits[0];
  for (size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > best_value) {
      best_idx = i;
      best_value = logits[i];
    }
  }
  return static_cast<uint64_t>(best_idx);
}

uint64_t SampleFromLogits(const std::vector<float>& logits, float temperature,
                          std::mt19937* rng) {
  if (!(temperature > 0.0f)) {
    return ArgmaxIndex(logits);
  }

  const double inv_temp = 1.0 / static_cast<double>(temperature);
  double max_scaled = -std::numeric_limits<double>::infinity();
  for (const float value : logits) {
    max_scaled = std::max(max_scaled, static_cast<double>(value) * inv_temp);
  }

  std::vector<double> probs(logits.size(), 0.0);
  double total = 0.0;
  for (size_t i = 0; i < logits.size(); ++i) {
    const double scaled = static_cast<double>(logits[i]) * inv_temp;
    const double prob = std::exp(scaled - max_scaled);
    if (std::isfinite(prob) && prob > 0.0) {
      probs[i] = prob;
      total += prob;
    }
  }

  if (!(total > 0.0) || !std::isfinite(total)) {
    return ArgmaxIndex(logits);
  }

  std::discrete_distribution<size_t> distribution(probs.begin(), probs.end());
  return static_cast<uint64_t>(distribution(*rng));
}

}  // namespace

struct Transformer::TokenStream::SharedState {
  std::mutex mutex;
  std::condition_variable cv;
  std::deque<int64_t> tokens;
  bool done = false;
  std::exception_ptr error = nullptr;
};

Transformer::TokenStream::TokenStream(std::thread worker,
                                      std::shared_ptr<SharedState> shared_state)
    : worker_(std::move(worker)), shared_state_(std::move(shared_state)) {}

Transformer::TokenStream::TokenStream(TokenStream&& other) noexcept
    : worker_(std::move(other.worker_)),
      shared_state_(std::move(other.shared_state_)) {}

Transformer::TokenStream& Transformer::TokenStream::operator=(
    TokenStream&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  Join();
  worker_ = std::move(other.worker_);
  shared_state_ = std::move(other.shared_state_);
  return *this;
}

Transformer::TokenStream::~TokenStream() { Join(); }

bool Transformer::TokenStream::WaitNext(int64_t* token) {
  if (token == nullptr) {
    throw std::runtime_error("TokenStream::WaitNext requires non-null token.");
  }
  if (!shared_state_) {
    return false;
  }

  std::exception_ptr error = nullptr;
  {
    std::unique_lock lock(shared_state_->mutex);
    shared_state_->cv.wait(lock, [this] {
      return !shared_state_->tokens.empty() || shared_state_->done;
    });

    if (!shared_state_->tokens.empty()) {
      *token = shared_state_->tokens.front();
      shared_state_->tokens.pop_front();
      return true;
    }

    error = shared_state_->error;
  }

  if (error != nullptr) {
    std::rethrow_exception(error);
  }
  return false;
}

void Transformer::TokenStream::Join() {
  if (worker_.joinable()) {
    worker_.join();
  }
}

Transformer::Transformer(uint64_t vocab_size, uint64_t hidden_size,
                         uint64_t intermediate_size, uint64_t num_blocks,
                         uint64_t num_attention_heads,
                         uint64_t num_key_value_heads, deeptiny::Device device,
                         deeptiny::nn::GatedMLP::HiddenAct mlp_hidden_act)
    : head_dim_(ValidateNonZero("hidden_size", hidden_size) /
                ValidateNonZero("num_attention_heads", num_attention_heads)),
      embed_(ValidateNonZero("vocab_size", vocab_size),
             ValidateNonZero("hidden_size", hidden_size),
             deeptiny::DType::Float32, device, true),
      norm_(hidden_size, 1.0e-5f, device) {
  ValidateNonZero("intermediate_size", intermediate_size);
  ValidateNonZero("num_blocks", num_blocks);
  ValidateNonZero("num_key_value_heads", num_key_value_heads);

  RegisterSubmodule(embed_);

  blocks_.reserve(static_cast<size_t>(num_blocks));
  kv_caches_.reserve(static_cast<size_t>(num_blocks));
  for (uint64_t i = 0; i < num_blocks; ++i) {
    blocks_.push_back(std::make_unique<deeptiny::nn::TransformerBlock>(
        hidden_size, intermediate_size, num_attention_heads,
        num_key_value_heads, false, true, true, 10000.0f, 1.0e-5f, device,
        mlp_hidden_act));
    RegisterSubmodule(*blocks_.back());
    kv_caches_.push_back(std::make_unique<deeptiny::nn::KVCache>(
        /*batch_size=*/1, num_key_value_heads, head_dim_));
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

std::vector<int64_t> Transformer::Generate(
    const std::vector<int64_t>& prompt_tokens, const GenerationOptions& options,
    std::mt19937* rng) const {
  std::vector<int64_t> generated;
  generated.reserve(static_cast<size_t>(options.max_new_tokens));
  GenerateWithCallback(
      prompt_tokens, options, rng,
      [&generated](int64_t token) { generated.push_back(token); });
  return generated;
}

Transformer::TokenStream Transformer::GenerateAsync(
    const std::vector<int64_t>& prompt_tokens, const GenerationOptions& options,
    std::optional<uint64_t> seed) const {
  auto shared_state = std::make_shared<TokenStream::SharedState>();
  std::thread worker([this, prompt_tokens, options, seed, shared_state]() {
    std::mt19937 local_rng;
    if (seed.has_value()) {
      local_rng.seed(*seed);
    } else {
      local_rng.seed(std::random_device{}());
    }

    try {
      GenerateWithCallback(prompt_tokens, options, &local_rng,
                           [shared_state](int64_t token) {
                             {
                               std::lock_guard lock(shared_state->mutex);
                               shared_state->tokens.push_back(token);
                             }
                             shared_state->cv.notify_one();
                           });
    } catch (...) {
      std::lock_guard lock(shared_state->mutex);
      shared_state->error = std::current_exception();
    }

    {
      std::lock_guard lock(shared_state->mutex);
      shared_state->done = true;
    }
    shared_state->cv.notify_all();
  });

  return TokenStream(std::move(worker), std::move(shared_state));
}

void Transformer::GenerateWithCallback(
    const std::vector<int64_t>& prompt_tokens, const GenerationOptions& options,
    std::mt19937* rng, const std::function<void(int64_t)>& on_token) const {
  if (prompt_tokens.empty()) {
    throw std::runtime_error("Generate requires at least one prompt token.");
  }
  if (options.max_new_tokens == 0) {
    throw std::runtime_error("Generate max_new_tokens must be non-zero.");
  }
  if (options.temperature < 0.0f) {
    throw std::runtime_error("Generate temperature must be >= 0.");
  }
  if (!on_token) {
    throw std::runtime_error("Generate callback must be set.");
  }

  deeptiny::NoGrad no_grad_guard;
  std::lock_guard lock(generation_mutex_);

  std::mt19937 local_rng;
  if (rng == nullptr) {
    local_rng.seed(std::random_device{}());
    rng = &local_rng;
  }

  ResetKVCache();

  const deeptiny::Shape prompt_shape{
      1, static_cast<uint64_t>(prompt_tokens.size())};
  deeptiny::Tensor hidden_states =
      ForwardChunkWithCache(prompt_tokens, prompt_shape, /*position_offset=*/0);
  deeptiny::Tensor logits = ComputeNextTokenLogitsFromHidden(hidden_states);

  for (uint64_t step = 0; step < options.max_new_tokens; ++step) {
    const std::vector<float> logits_values = TensorToFloatVector(logits);
    const uint64_t next_token =
        SampleFromLogits(logits_values, options.temperature, rng);
    const int64_t next_token_i64 = static_cast<int64_t>(next_token);
    on_token(next_token_i64);

    if (options.eos_token_id.has_value() &&
        next_token == *options.eos_token_id) {
      break;
    }
    if (step + 1 == options.max_new_tokens) {
      break;
    }

    const uint64_t position_offset = kv_caches_.front()->seq_len();
    const std::vector<int64_t> step_tokens{next_token_i64};
    hidden_states = ForwardChunkWithCache(step_tokens, deeptiny::Shape{1, 1},
                                          position_offset);
    logits = ComputeNextTokenLogitsFromHidden(hidden_states);
  }
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

void Transformer::ResetKVCache() const {
  for (const auto& cache : kv_caches_) {
    cache->Clear();
  }
}

deeptiny::Tensor Transformer::ForwardChunkWithCache(
    const std::vector<int64_t>& flat_tokens, const deeptiny::Shape& token_shape,
    uint64_t position_offset) const {
  deeptiny::Tensor hidden_states = embed_(flat_tokens, token_shape);
  for (size_t i = 0; i < blocks_.size(); ++i) {
    hidden_states = (*blocks_[i])(hidden_states, std::nullopt, position_offset,
                                  kv_caches_[i].get());
  }
  return norm_(hidden_states);
}

deeptiny::Tensor Transformer::ComputeNextTokenLogitsFromHidden(
    const deeptiny::Tensor& hidden_states) const {
  const deeptiny::Shape& hidden_shape = hidden_states.shape();
  if (hidden_shape.size() != 3) {
    throw std::runtime_error(
        "Transformer forward output must have shape [batch, seq, hidden].");
  }
  if (hidden_shape[0] != 1) {
    throw std::runtime_error("Transformer generation expects batch size 1.");
  }

  const int64_t last_token_index =
      static_cast<int64_t>(hidden_shape[1]) - static_cast<int64_t>(1);
  const int64_t hidden_size = static_cast<int64_t>(hidden_shape[2]);
  deeptiny::Tensor last_hidden =
      hidden_states({deeptiny::Slice(0, 1),
                     deeptiny::Slice(last_token_index, last_token_index + 1),
                     deeptiny::Slice(0, hidden_size)});

  deeptiny::Tensor embedding_weight = embed_.weight();
  const deeptiny::Shape& embedding_weight_shape = embedding_weight.shape();
  if (embedding_weight_shape.size() != 2) {
    throw std::runtime_error(
        "Transformer embedding weights must have shape [vocab, hidden].");
  }
  const uint64_t embedding_vocab_size = embedding_weight_shape[0];
  const uint64_t embedding_hidden_size = embedding_weight_shape[1];
  if (embedding_hidden_size != hidden_shape[2]) {
    std::stringstream err;
    err << "Embedding hidden size (" << embedding_hidden_size
        << ") does not match Transformer hidden size (" << hidden_shape[2]
        << ").";
    throw std::runtime_error(err.str());
  }

  deeptiny::Tensor tied_embedding =
      embedding_weight.Reshape({1, embedding_vocab_size, hidden_shape[2]});
  deeptiny::Tensor logits =
      deeptiny::math::BatchedMatMul(last_hidden, tied_embedding, false, true);
  return logits.Reshape({embedding_vocab_size});
}

}  // namespace transfomer_demo
