#pragma once

#include <cstdint>
#include <optional>

#include "deeptiny/tensor.h"
#include "deeptiny/types.h"

namespace module {

class Linear {
 public:
  Linear(uint64_t in_dim, uint64_t out_dim, bool bias = true,
         deeptiny::Device device = deeptiny::Device::CPU);

  deeptiny::Tensor Forward(const deeptiny::Tensor& x) const;
  deeptiny::Tensor operator()(const deeptiny::Tensor& x) const;

  deeptiny::Tensor& weight();
  const deeptiny::Tensor& weight() const;
  std::optional<deeptiny::Tensor>& bias();
  const std::optional<deeptiny::Tensor>& bias() const;

 private:
  uint64_t in_dim_;
  uint64_t out_dim_;
  deeptiny::Tensor weight_;
  std::optional<deeptiny::Tensor> bias_;
};

}  // namespace module
