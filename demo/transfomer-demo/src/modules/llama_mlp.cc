#include "modules/llama_mlp.h"

#include "deeptiny/functional.h"
#include "deeptiny/math.h"

namespace module {

LlamaMLP::LlamaMLP(uint64_t in_dim, uint64_t hidden_dim, uint64_t out_dim,
                   bool bias, deeptiny::Device device)
    : gate_proj_(in_dim, hidden_dim, bias, device),
      up_proj_(in_dim, hidden_dim, bias, device),
      down_proj_(hidden_dim, out_dim, bias, device) {}

deeptiny::Tensor LlamaMLP::Forward(const deeptiny::Tensor& x) const {
  deeptiny::Tensor gated = deeptiny::functional::ReLU(gate_proj_(x));
  deeptiny::Tensor up = up_proj_(x);
  deeptiny::Tensor hidden = gated * up;
  return down_proj_(hidden);
}

deeptiny::Tensor LlamaMLP::operator()(const deeptiny::Tensor& x) const {
  return Forward(x);
}

const Linear& LlamaMLP::gate_proj() const { return gate_proj_; }

const Linear& LlamaMLP::up_proj() const { return up_proj_; }

const Linear& LlamaMLP::down_proj() const { return down_proj_; }

}  // namespace module
