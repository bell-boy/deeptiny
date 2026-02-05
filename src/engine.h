#pragma once
#include <deque>
#include <memory>
#include <vector>

namespace deeptiny {

class AutogradMeta;

class Engine;

class Tensor;

extern struct State {
  bool grad_enabled = true;
} GradState;

class Function {
 private:
  using ParentList = std::vector<std::shared_ptr<AutogradMeta>>;
  ParentList parents_;
  friend class Engine;

 protected:
  const ParentList& getParents() const { return parents_; }

 public:
  Function(ParentList parents) : parents_(std::move(parents)) {}
  virtual void operator()(const Tensor& grad, Engine& engine) = 0;
  virtual ~Function() = default;
};

class Engine {
 private:
  std::deque<std::shared_ptr<AutogradMeta>> ready_queue_;

 public:
  Engine(std::shared_ptr<AutogradMeta> root, bool keep_graph = false);

  void Run();

  void EnqueueBackward(std::shared_ptr<Function> func);
};

};  // namespace deeptiny
