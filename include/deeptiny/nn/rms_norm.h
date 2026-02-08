#pragma once

#include <cstdint>

#include "deeptiny/nn/module.h"
#include "deeptiny/tensor.h"
#include "deeptiny/types.h"

namespace deeptiny::nn {

class RMSNorm : public Module {
 public:
  RMSNorm(uint64_t hidden_dim, float epsilon = 1e-5f,
          Device device = Device::CPU);

  Tensor operator()(const Tensor& x) const;

  Tensor& weight();
  const Tensor& weight() const;
  float epsilon() const;

 private:
  uint64_t hidden_dim_;
  float epsilon_;
  Tensor weight_;
};

}  // namespace deeptiny::nn
