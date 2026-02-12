#include "deeptiny/nn/gated_mlp.h"

#include "deeptiny/functional.h"
#include "deeptiny/math.h"

namespace deeptiny::nn {
namespace {

Tensor ApplyHiddenAct(const Tensor& x, GatedMLP::HiddenAct hidden_act) {
  switch (hidden_act) {
    case GatedMLP::HiddenAct::ReLU:
      return functional::ReLU(x);
    case GatedMLP::HiddenAct::SiLU:
      return functional::SiLU(x);
  }
}

}  // namespace

GatedMLP::GatedMLP(uint64_t in_dim, uint64_t hidden_dim, uint64_t out_dim,
                   bool bias, Device device, HiddenAct hidden_act)
    : hidden_act_(hidden_act),
      gate_proj_(in_dim, hidden_dim, bias, device),
      up_proj_(in_dim, hidden_dim, bias, device),
      down_proj_(hidden_dim, out_dim, bias, device) {
  RegisterSubmodule(gate_proj_);
  RegisterSubmodule(up_proj_);
  RegisterSubmodule(down_proj_);
}

Tensor GatedMLP::operator()(const Tensor& x) const {
  Tensor gated = ApplyHiddenAct(gate_proj_(x), hidden_act_);
  Tensor up = up_proj_(x);
  Tensor hidden = gated * up;
  return down_proj_(hidden);
}

Linear& GatedMLP::gate_proj() { return gate_proj_; }

const Linear& GatedMLP::gate_proj() const { return gate_proj_; }

Linear& GatedMLP::up_proj() { return up_proj_; }

const Linear& GatedMLP::up_proj() const { return up_proj_; }

Linear& GatedMLP::down_proj() { return down_proj_; }

const Linear& GatedMLP::down_proj() const { return down_proj_; }

}  // namespace deeptiny::nn
