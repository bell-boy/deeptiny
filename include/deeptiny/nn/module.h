#pragma once

#include <cstdint>
#include <vector>

#include "deeptiny/tensor.h"

namespace deeptiny::nn {

class Module {
 public:
  virtual ~Module() = 0;

  void Update(float learning_rate);
  uint64_t NumParametersRequiringGrad() const;

 protected:
  void RegisterParameter(Tensor parameter);
  void RegisterSubmodule(Module& submodule);

 private:
  std::vector<Tensor> parameters_;
  std::vector<Module*> submodules_;
};

}  // namespace deeptiny::nn
