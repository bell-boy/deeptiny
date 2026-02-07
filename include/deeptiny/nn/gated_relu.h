#pragma once

#include <cstdint>

#include "deeptiny/nn/linear.h"
#include "deeptiny/nn/module.h"
#include "deeptiny/tensor.h"
#include "deeptiny/types.h"

namespace deeptiny::nn {

class GatedReLU : public Module {
 public:
  GatedReLU(uint64_t in_dim, uint64_t hidden_dim, uint64_t out_dim,
            bool bias = true, Device device = Device::CPU);

  Tensor operator()(const Tensor& x) const;

  Linear& gate_proj();
  const Linear& gate_proj() const;
  Linear& up_proj();
  const Linear& up_proj() const;
  Linear& down_proj();
  const Linear& down_proj() const;

 private:
  Linear gate_proj_;
  Linear up_proj_;
  Linear down_proj_;
};

}  // namespace deeptiny::nn
