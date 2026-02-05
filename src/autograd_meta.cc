#include "autograd_meta.h"

#include <utility>
#include <vector>

#include "engine.h"
#include "utils.h"

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
    Tensor grad_copy(grad.shape(), grad.dtype(), grad.device(), false);
    auto src_impl = utils::TensorAccessor::GetTensorImpl(grad);
    auto dst_impl = utils::TensorAccessor::GetTensorImpl(grad_copy);
    auto src_storage = src_impl->getContiguousStorage();
    const uint64_t numel = src_storage->numel();
    if (numel > 0) {
      std::vector<std::byte> host_buf(numel * grad.dtype().size());
      src_storage->CopyToHost(0, numel, host_buf.data());
      dst_impl->storage()->CopyFromHost(0, numel, host_buf.data());
    }
    grad_ = std::make_shared<Tensor>(std::move(grad_copy));
    return;
  }
  *grad_ += grad;
}

void AutogradMeta::incrementPending() { pending_ += 1; }

}  // namespace deeptiny
