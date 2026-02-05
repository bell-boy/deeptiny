#pragma once
#include <optional>
#include <vector>

#include "deeptiny/tensor.h"

namespace deeptiny {

class Engine;

class Function;

class AutogradMeta : public std::enable_shared_from_this<AutogradMeta> {
 private:
  std::shared_ptr<Tensor> grad_;
  uint64_t pending_;
  std::shared_ptr<Function> grad_fn_;
  bool requires_grad_;
  friend class Engine;

 public:
  AutogradMeta(std::shared_ptr<Function> grad_fn, bool requires_grad = false);
  // TODO: add support for scatter grads
  void updateGrad(const Tensor& grad, Engine& engine);
  void incrementPending();
};

};  // namespace deeptiny
