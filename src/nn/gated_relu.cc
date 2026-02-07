#include "deeptiny/nn/gated_relu.h"

#include "deeptiny/functional.h"
#include "deeptiny/math.h"

namespace deeptiny::nn {

GatedReLU::GatedReLU(uint64_t in_dim, uint64_t hidden_dim, uint64_t out_dim,
                     bool bias, Device device)
    : gate_proj_(in_dim, hidden_dim, bias, device),
      up_proj_(in_dim, hidden_dim, bias, device),
      down_proj_(hidden_dim, out_dim, bias, device) {
  RegisterSubmodule(gate_proj_);
  RegisterSubmodule(up_proj_);
  RegisterSubmodule(down_proj_);
}

Tensor GatedReLU::operator()(const Tensor& x) const {
  Tensor gated = functional::ReLU(gate_proj_(x));
  Tensor up = up_proj_(x);
  Tensor hidden = gated * up;
  return down_proj_(hidden);
}

Linear& GatedReLU::gate_proj() { return gate_proj_; }

const Linear& GatedReLU::gate_proj() const { return gate_proj_; }

Linear& GatedReLU::up_proj() { return up_proj_; }

const Linear& GatedReLU::up_proj() const { return up_proj_; }

Linear& GatedReLU::down_proj() { return down_proj_; }

const Linear& GatedReLU::down_proj() const { return down_proj_; }

}  // namespace deeptiny::nn
