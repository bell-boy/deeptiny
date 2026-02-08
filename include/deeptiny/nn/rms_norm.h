#pragma once

#include <cstdint>

#include "deeptiny/nn/module.h"
#include "deeptiny/tensor.h"
#include "deeptiny/types.h"

namespace deeptiny::nn {

class RMSNorm : public Module {
 public:
  RMSNorm(uint64_t dim, float eps = 1e-5f, Device device = Device::CPU);

  Tensor operator()(const Tensor& x) const;

  Tensor& weight();
  const Tensor& weight() const;
  float eps() const;

 private:
  uint64_t dim_;
  float eps_;
  Tensor weight_;
};

}  // namespace deeptiny::nn
