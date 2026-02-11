#include "autograd_meta.h"

#include <cassert>
#include <utility>

#include "deeptiny/autograd.h"
#include "dispatch/binary.h"

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

void AutogradMeta::updateGrad(const Tensor& grad) {
  if (!requires_grad_) {
    return;
  }
  if (!grad_) {
    grad_ = std::make_shared<Tensor>(grad.Clone());
    return;
  }
  assert(grad_->shape() == grad.shape());
  deeptiny::dispatch::binary::Inplace(deeptiny::dispatch::binary::Op::Add,
                                      *grad_, grad);
}

}  // namespace deeptiny
