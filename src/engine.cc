#include "engine.h"

#include <cassert>
#include <functional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "autograd_meta.h"

namespace deeptiny {

State GradState;

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
        if (parent) {
          stack.push_back(parent);
        }
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
      if (parent) {
        parent->pending_ += 1;
      }
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
}  // namespace deeptiny
