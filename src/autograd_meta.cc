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

}  // namespace deeptiny
