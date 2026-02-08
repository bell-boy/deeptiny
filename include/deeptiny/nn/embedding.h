#pragma once

#include <cstdint>
#include <vector>

#include "deeptiny/nn/module.h"
#include "deeptiny/tensor.h"
#include "deeptiny/types.h"

namespace deeptiny::nn {

class Embedding : public Module {
 public:
  Embedding(uint64_t num_embeddings, uint64_t embedding_dim,
            DType dtype = DType::Float32, Device device = Device::CPU,
            bool requires_grad = true);

  Tensor operator()(const std::vector<int64_t>& indices,
                    const Shape& shape) const;

  Tensor weight();
  Tensor weight() const;

 private:
  uint64_t num_embeddings_;
  uint64_t embedding_dim_;
  DType dtype_;
  Device device_;
  Tensor weight_;
};

}  // namespace deeptiny::nn
