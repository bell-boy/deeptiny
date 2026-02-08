#pragma once

#include <cstdint>
#include <optional>

#include "deeptiny/nn/module.h"
#include "deeptiny/tensor.h"
#include "deeptiny/types.h"

namespace deeptiny::nn {

class Linear : public Module {
 public:
  Linear(uint64_t in_dim, uint64_t out_dim, bool bias = true,
         Device device = Device::CPU);

  Tensor operator()(const Tensor& x) const;

  Tensor weight();
  Tensor weight() const;
  std::optional<Tensor> bias();
  std::optional<Tensor> bias() const;

 private:
  uint64_t in_dim_;
  uint64_t out_dim_;
  Tensor weight_;
  std::optional<Tensor> bias_;
};

}  // namespace deeptiny::nn
