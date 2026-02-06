#pragma once

#include <cstdint>

#include "deeptiny/tensor.h"
#include "deeptiny/types.h"
#include "modules/linear.h"

namespace module {

class LlamaMLP {
 public:
  LlamaMLP(uint64_t in_dim, uint64_t hidden_dim, uint64_t out_dim,
           bool bias = true, deeptiny::Device device = deeptiny::Device::CPU);

  deeptiny::Tensor Forward(const deeptiny::Tensor& x) const;
  deeptiny::Tensor operator()(const deeptiny::Tensor& x) const;

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

}  // namespace module
