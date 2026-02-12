#pragma once

#include <cstdint>
#include <memory>

#include "deeptiny/tensor.h"

namespace deeptiny {
class TensorImpl;
}

namespace deeptiny::nn {

class KVCache {
 public:
  KVCache(uint64_t batch_size, uint64_t num_key_value_heads, uint64_t head_dim,
          uint64_t preallocate_seq_len = 1);

  void update(const Tensor& keys, const Tensor& values);

  const Tensor& keys() const;
  const Tensor& values() const;

  uint64_t seq_len() const;
  void Clear();

 private:
  void GrowIfNeeded(uint64_t required_seq_len);
  void AppendTensor(const Tensor& src, bool is_key);
  void RefreshActiveViews();

  float* key_buf_;
  float* value_buf_;
  std::shared_ptr<TensorImpl> key_impl_;
  std::shared_ptr<TensorImpl> value_impl_;
  Tensor key_tensor_;
  Tensor value_tensor_;
  uint64_t batch_size_;
  uint64_t num_key_value_heads_;
  uint64_t head_dim_;
  uint64_t seq_len_;
  uint64_t capacity_seq_len_;
};

}  // namespace deeptiny::nn
