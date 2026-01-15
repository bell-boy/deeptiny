#pragma once
#include <optional>
#include <vector>

#include "deeptiny/tensor.h"

class Engine;

class Function;

namespace deeptiny {

class AutogradMeta : public std::enable_shared_from_this<AutogradMeta> {
 private:
  std::shared_ptr<Tensor> grad_;
  uint64_t pending_;
  std::shared_ptr<Function> grad_fn_;
  bool requires_grad_;

 public:
  // TODO: add support for scatter grads
  void updateGrad(const Tensor& grad, Engine& engine,
                  std::optional<std::vector<Slice>> slices = std::nullopt);
  AutogradMeta(std::shared_ptr<Function> grad_fn, bool requires_grad);
  void incrementPending();
};

};  // namespace deeptiny
