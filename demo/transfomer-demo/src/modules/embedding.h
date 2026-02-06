#pragma once

#include <cstdint>
#include <vector>

#include "deeptiny/tensor.h"
#include "deeptiny/types.h"

namespace transfomer_demo::modules {

class Embedding {
 public:
  Embedding(uint64_t num_embeddings, uint64_t embedding_dim,
            deeptiny::DType dtype = deeptiny::DType::Float32,
            deeptiny::Device device = deeptiny::Device::CPU,
            bool requires_grad = true);

  deeptiny::Tensor operator()(const std::vector<int64_t>& indices,
                              const deeptiny::Shape& shape) const;

  const deeptiny::Tensor& weight() const { return weight_; }

 private:
  uint64_t num_embeddings_;
  uint64_t embedding_dim_;
  deeptiny::DType dtype_;
  deeptiny::Device device_;
  deeptiny::Tensor weight_;
};

}  // namespace transfomer_demo::modules
