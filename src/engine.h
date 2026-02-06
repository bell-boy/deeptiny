#pragma once
#include <cstdint>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include "deeptiny/tensor.h"

namespace deeptiny {

class AutogradMeta;

class Engine;

class Context {
 private:
  struct SavedTensor {
    Tensor tensor;
    uint64_t version;
  };
  std::unordered_map<uint64_t, SavedTensor> saved_tensors_;
  uint64_t GetStorageVersion(const Tensor& tensor) const;

 public:
  void Set(uint64_t id, const Tensor& tensor);
  Tensor Get(uint64_t id) const;
};

class Function {
 private:
  using ParentList = std::vector<std::shared_ptr<AutogradMeta>>;
  ParentList parents_;
  Context ctx_;
  friend class Engine;

 protected:
  const ParentList& getParents() const { return parents_; }
  Context& context() { return ctx_; }
  const Context& context() const { return ctx_; }

 public:
  Function(ParentList parents) : parents_(std::move(parents)) {}
  virtual void operator()(const Tensor& grad) = 0;
  virtual ~Function() = default;
};

class Engine {
 private:
  std::deque<std::shared_ptr<AutogradMeta>> ready_queue_;

 public:
  // pending_ is owned by Engine.
  Engine(std::shared_ptr<AutogradMeta> root, bool keep_graph = false);

  void Run();

  void EnqueueBackward(std::shared_ptr<Function> func);
};

};  // namespace deeptiny
