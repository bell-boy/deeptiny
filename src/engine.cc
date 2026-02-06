#include "engine.h"

#include <cassert>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "autograd_meta.h"
#include "deeptiny/autograd.h"
#include "utils.h"

namespace deeptiny {

State GradState;

uint64_t Context::GetStorageVersion(const Tensor& tensor) const {
  auto impl = utils::TensorAccessor::GetTensorImpl(tensor);
  if (!impl) {
    throw std::runtime_error("Context cannot access null TensorImpl");
  }
  auto storage = impl->storage();
  if (!storage) {
    throw std::runtime_error("Context cannot access null tensor storage");
  }
  return storage->version();
}

void Context::Set(uint64_t id, const Tensor& tensor) {
  saved_tensors_.insert_or_assign(
      id, SavedTensor{tensor, GetStorageVersion(tensor)});
}

Tensor Context::Get(uint64_t id) const {
  auto it = saved_tensors_.find(id);
  if (it == saved_tensors_.end()) {
    std::stringstream err;
    err << "Context key not found: " << id;
    throw std::runtime_error(err.str());
  }
  const uint64_t current_version = GetStorageVersion(it->second.tensor);
  if (current_version != it->second.version) {
    std::stringstream err;
    err << "Context tensor version mismatch for key " << id
        << ": expected storage version " << it->second.version << " but got "
        << current_version
        << ". Tensor used in backward was modified in-place.";
    throw std::runtime_error(err.str());
  }
  return it->second.tensor;
}

Engine::Engine(std::shared_ptr<AutogradMeta> root, bool keep_graph) {
  (void)keep_graph;
  if (!root) {
    throw std::runtime_error("Engine root is null");
  }

  std::vector<std::shared_ptr<AutogradMeta>> nodes;
  std::unordered_set<AutogradMeta*> visited;
  std::vector<std::shared_ptr<AutogradMeta>> stack;
  stack.push_back(root);

  // build list of nodes
  while (!stack.empty()) {
    auto node = stack.back();
    stack.pop_back();
    assert(node);
    if (!visited.insert(node.get()).second) {
      continue;
    }
    nodes.push_back(node);
    if (node->grad_fn_) {
      for (const auto& parent : node->grad_fn_->getParents()) {
        assert(parent && "Function parent must not be null");
        stack.push_back(parent);
      }
    }
  }

  // update pending counts
  for (const auto& node : nodes) {
    node->pending_ = 0;
  }

  for (const auto& node : nodes) {
    if (!node->grad_fn_) {
      continue;
    }
    for (const auto& parent : node->grad_fn_->getParents()) {
      assert(parent && "Function parent must not be null");
      parent->pending_ += 1;
    }
  }

  std::unordered_set<AutogradMeta*> computed;
  std::function<bool(const std::shared_ptr<AutogradMeta>&)> compute_requires;
  compute_requires = [&](const std::shared_ptr<AutogradMeta>& node) -> bool {
    if (!node) {
      return false;
    }
    if (!node->grad_fn_) {
      return node->requires_grad_;
    }
    if (computed.count(node.get()) != 0) {
      return node->requires_grad_;
    }
    bool requires_grad = false;
    for (const auto& parent : node->grad_fn_->getParents()) {
      assert(parent && "Function parent must not be null");
      if (compute_requires(parent)) {
        requires_grad = true;
      }
    }
    node->requires_grad_ = requires_grad;
    computed.insert(node.get());
    return requires_grad;
  };
  compute_requires(root);

  ready_queue_.push_back(root);
}

void Engine::Run() {
  while (!ready_queue_.empty()) {
    auto node = ready_queue_.front();
    ready_queue_.pop_front();
    if (!node || !node->requires_grad_) {
      continue;
    }
    if (!node->grad_fn_) {
      continue;
    }
    if (!node->grad_) {
      throw std::runtime_error("Backward called with no gradient");
    }
    (*node->grad_fn_)(*node->grad_, *this);
    EnqueueBackward(node->grad_fn_);
  }
}

void Engine::EnqueueBackward(std::shared_ptr<Function> func) {
  if (!func) {
    return;
  }
  for (const auto& parent : func->getParents()) {
    assert(parent && "EnqueueBackward encountered null parent");
    parent->pending_ -= 1;
    if (parent->pending_ == 0 && parent->requires_grad_) {
      ready_queue_.push_back(parent);
    }
  }
}
}  // namespace deeptiny
