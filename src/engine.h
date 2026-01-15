#pragma once
#include <deque>
#include <vector>

namespace deeptiny {

class AutogradMeta;

class Engine;

class Tensor;

class Function {
 protected:
  std::vector<std::shared_ptr<AutogradMeta>> parents;

 public:
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

extern struct State {
  bool grad_enabled = true;
} GradState;

};  // namespace deeptiny
