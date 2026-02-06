#pragma once
#include <optional>
#include <vector>

#include "deeptiny/tensor.h"

namespace deeptiny {

class Engine;

class Function;

class AutogradMeta {
 private:
  std::shared_ptr<Tensor> grad_;
  uint64_t pending_;
  std::shared_ptr<Function> grad_fn_;
  bool requires_grad_;
  friend class Engine;

 public:
  AutogradMeta(std::shared_ptr<Function> grad_fn, bool requires_grad = false);
  // TODO: add support for scatter grads
  // Contract: updateGrad accumulates gradient values for this node.
  void updateGrad(const Tensor& grad);
  bool requires_grad() const { return requires_grad_; }
  std::optional<Tensor> grad() const {
    if (!grad_) {
      return std::nullopt;
    }
    return *grad_;
  }
};

};  // namespace deeptiny
