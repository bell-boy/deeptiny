#include "deeptiny/nn/kv_cache.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "deeptiny/autograd.h"
#include "tensor_impl.h"
#include "utils.h"

namespace deeptiny::nn {
namespace {

uint64_t ValidateNonZero(const char* field, uint64_t value) {
  if (value == 0) {
    std::stringstream err;
    err << "KVCache " << field << " must be non-zero";
    throw std::runtime_error(err.str());
  }
  return value;
}

void ValidateUpdateTensor(const Tensor& t, const char* name,
                          uint64_t batch_size, uint64_t num_key_value_heads,
                          uint64_t head_dim) {
  if (GradState.grad_enabled) {
    throw std::runtime_error(
        "KVCache is inference-only and requires NoGrad mode");
  }
  if (t.dtype() != DType::Float32) {
    std::stringstream err;
    err << "KVCache " << name << " must use Float32 dtype";
    throw std::runtime_error(err.str());
  }
  if (t.device() != Device::CPU) {
    std::stringstream err;
    err << "KVCache " << name << " must be on CPU";
    throw std::runtime_error(err.str());
  }
  if (t.requires_grad()) {
    std::stringstream err;
    err << "KVCache is inference-only and does not accept " << name
        << " tensors requiring gradients";
    throw std::runtime_error(err.str());
  }

  const auto& shape = t.shape();
  if (shape.size() != 4) {
    std::stringstream err;
    err << "KVCache " << name
        << " must be rank-4 [batch, kv_head, seq, "
           "head_dim]";
    throw std::runtime_error(err.str());
  }
  if (shape[0] != batch_size || shape[1] != num_key_value_heads ||
      shape[3] != head_dim) {
    std::stringstream err;
    err << "KVCache " << name << " shape mismatch: expected "
        << FormatShape({batch_size, num_key_value_heads, shape[2], head_dim})
        << " but got " << FormatShape(shape);
    throw std::runtime_error(err.str());
  }
}

}  // namespace

KVCache::KVCache(uint64_t batch_size, uint64_t num_key_value_heads,
                 uint64_t head_dim, uint64_t preallocate_seq_len)
    : key_buf_(nullptr),
      value_buf_(nullptr),
      key_impl_(std::make_shared<TensorImpl>(
          Shape{ValidateNonZero("batch_size", batch_size),
                ValidateNonZero("num_key_value_heads", num_key_value_heads),
                ValidateNonZero("preallocate_seq_len", preallocate_seq_len),
                ValidateNonZero("head_dim", head_dim)},
          DType::Float32, Device::CPU)),
      value_impl_(std::make_shared<TensorImpl>(
          Shape{batch_size, num_key_value_heads, preallocate_seq_len, head_dim},
          DType::Float32, Device::CPU)),
      key_tensor_(),
      value_tensor_(),
      batch_size_(batch_size),
      num_key_value_heads_(num_key_value_heads),
      head_dim_(head_dim),
      seq_len_(0),
      capacity_seq_len_(preallocate_seq_len) {
  key_buf_ = static_cast<float*>(key_impl_->storage()->data(0));
  value_buf_ = static_cast<float*>(value_impl_->storage()->data(0));
  RefreshActiveViews();
}

void KVCache::GrowIfNeeded(uint64_t required_seq_len) {
  if (required_seq_len <= capacity_seq_len_) {
    return;
  }

  uint64_t new_capacity_seq_len = std::max<uint64_t>(1, capacity_seq_len_);
  while (new_capacity_seq_len < required_seq_len) {
    new_capacity_seq_len *= 2;
  }
  if (new_capacity_seq_len < required_seq_len) {
    new_capacity_seq_len = required_seq_len;
  }

  auto new_key_impl = std::make_shared<TensorImpl>(
      Shape{batch_size_, num_key_value_heads_, new_capacity_seq_len, head_dim_},
      DType::Float32, Device::CPU);
  auto new_value_impl = std::make_shared<TensorImpl>(
      Shape{batch_size_, num_key_value_heads_, new_capacity_seq_len, head_dim_},
      DType::Float32, Device::CPU);

  if (seq_len_ > 0) {
    const auto old_key_storage =
        std::shared_ptr<const Storage>(key_impl_->storage());
    const auto old_value_storage =
        std::shared_ptr<const Storage>(value_impl_->storage());
    const auto* old_key_data =
        static_cast<const float*>(old_key_storage->data(0));
    const auto* old_value_data =
        static_cast<const float*>(old_value_storage->data(0));
    auto* new_key_data = static_cast<float*>(new_key_impl->storage()->data(0));
    auto* new_value_data =
        static_cast<float*>(new_value_impl->storage()->data(0));

    const uint64_t old_block_stride = capacity_seq_len_ * head_dim_;
    const uint64_t new_block_stride = new_capacity_seq_len * head_dim_;
    const uint64_t used_block_size = seq_len_ * head_dim_;
    const size_t used_block_bytes =
        static_cast<size_t>(used_block_size * sizeof(float));

    for (uint64_t batch = 0; batch < batch_size_; ++batch) {
      for (uint64_t head = 0; head < num_key_value_heads_; ++head) {
        const uint64_t block_index = batch * num_key_value_heads_ + head;
        const uint64_t old_base = block_index * old_block_stride;
        const uint64_t new_base = block_index * new_block_stride;
        std::memcpy(new_key_data + new_base, old_key_data + old_base,
                    used_block_bytes);
        std::memcpy(new_value_data + new_base, old_value_data + old_base,
                    used_block_bytes);
      }
    }
  }

  key_impl_ = std::move(new_key_impl);
  value_impl_ = std::move(new_value_impl);
  capacity_seq_len_ = new_capacity_seq_len;
  key_buf_ = static_cast<float*>(key_impl_->storage()->data(0));
  value_buf_ = static_cast<float*>(value_impl_->storage()->data(0));
}

void KVCache::AppendTensor(const Tensor& src, bool is_key) {
  const auto src_impl = utils::TensorAccessor::GetTensorImpl(src);
  const auto src_storage =
      std::shared_ptr<const Storage>(src_impl->getContiguousStorage());
  const auto* src_data = static_cast<const float*>(src_storage->data(0));
  const auto& src_shape = src.shape();
  const uint64_t append_seq_len = src_shape[2];
  if (append_seq_len == 0) {
    return;
  }

  float* dst_data = nullptr;
  if (is_key) {
    key_buf_ = static_cast<float*>(key_impl_->storage()->data(0));
    dst_data = key_buf_;
  } else {
    value_buf_ = static_cast<float*>(value_impl_->storage()->data(0));
    dst_data = value_buf_;
  }

  const uint64_t src_block_size = append_seq_len * head_dim_;
  const uint64_t dst_block_stride = capacity_seq_len_ * head_dim_;
  const size_t src_block_bytes =
      static_cast<size_t>(src_block_size * sizeof(float));
  for (uint64_t batch = 0; batch < batch_size_; ++batch) {
    for (uint64_t head = 0; head < num_key_value_heads_; ++head) {
      const uint64_t block_index = batch * num_key_value_heads_ + head;
      const uint64_t src_base = block_index * src_block_size;
      const uint64_t dst_base =
          block_index * dst_block_stride + seq_len_ * head_dim_;
      std::memcpy(dst_data + dst_base, src_data + src_base, src_block_bytes);
    }
  }
}

void KVCache::RefreshActiveViews() {
  const Shape active_shape{batch_size_, num_key_value_heads_, seq_len_,
                           head_dim_};
  auto key_view =
      key_impl_->View(Shape(active_shape), Stride(key_impl_->stride()), 0);
  auto value_view =
      value_impl_->View(Shape(active_shape), Stride(value_impl_->stride()), 0);
  key_tensor_ = utils::TensorAccessor::MakeTensor(std::move(key_view), nullptr);
  value_tensor_ =
      utils::TensorAccessor::MakeTensor(std::move(value_view), nullptr);
}

void KVCache::update(const Tensor& keys, const Tensor& values) {
  ValidateUpdateTensor(keys, "keys", batch_size_, num_key_value_heads_,
                       head_dim_);
  ValidateUpdateTensor(values, "values", batch_size_, num_key_value_heads_,
                       head_dim_);

  if (keys.shape() != values.shape()) {
    throw std::runtime_error(
        "KVCache update requires keys and values to have matching shapes");
  }

  const uint64_t append_seq_len = keys.shape()[2];
  GrowIfNeeded(seq_len_ + append_seq_len);
  AppendTensor(keys, /*is_key=*/true);
  AppendTensor(values, /*is_key=*/false);
  seq_len_ += append_seq_len;
  RefreshActiveViews();
}

const Tensor& KVCache::keys() const { return key_tensor_; }

const Tensor& KVCache::values() const { return value_tensor_; }

uint64_t KVCache::seq_len() const { return seq_len_; }

void KVCache::Clear() {
  seq_len_ = 0;
  RefreshActiveViews();
}

}  // namespace deeptiny::nn
