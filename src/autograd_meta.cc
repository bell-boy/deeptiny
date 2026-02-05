#include "autograd_meta.h"

#include <utility>

#include "engine.h"

namespace deeptiny {

AutogradMeta::AutogradMeta(std::shared_ptr<Function> grad_fn,
                           bool requires_grad)
    : grad_(nullptr), pending_(0), grad_fn_(nullptr), requires_grad_(false) {
  if (!GradState.grad_enabled) {
    return;
  }
  grad_fn_ = grad_fn;
  requires_grad_ = requires_grad;
}

void AutogradMeta::updateGrad(const Tensor& grad, Engine& engine) {
  (void)engine;
  if (!requires_grad_ || !GradState.grad_enabled) {
    return;
  }
  if (!grad_) {
    grad_ = std::make_shared<Tensor>(grad);
    return;
  }
  *grad_ += grad;
}

void AutogradMeta::incrementPending() { pending_ += 1; }

}  // namespace deeptiny
