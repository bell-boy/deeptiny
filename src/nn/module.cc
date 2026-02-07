#include "deeptiny/nn/module.h"

#include <stdexcept>
#include <vector>

#include "deeptiny/math.h"

namespace deeptiny::nn {

Module::~Module() = default;

void Module::Update(float learning_rate) {
  for (auto& parameter : parameters_) {
    auto grad = parameter.grad();
    if (!grad.has_value()) {
      continue;
    }

    auto lr_tensor = Tensor::FromVector<float>(std::vector<float>{learning_rate},
                                               parameter.device(), false);
    parameter -= lr_tensor * *grad;
  }

  for (auto* submodule : submodules_) {
    submodule->Update(learning_rate);
  }
}

uint64_t Module::NumParametersRequiringGrad() const {
  uint64_t count = 0;
  for (const auto& parameter : parameters_) {
    count += parameter.numel();
  }

  for (const auto* submodule : submodules_) {
    count += submodule->NumParametersRequiringGrad();
  }

  return count;
}

void Module::RegisterParameter(Tensor parameter) {
  if (!parameter.requires_grad()) {
    throw std::runtime_error(
        "Registered parameters must have requires_grad=true");
  }
  parameters_.push_back(std::move(parameter));
}

void Module::RegisterSubmodule(Module& submodule) { submodules_.push_back(&submodule); }

}  // namespace deeptiny::nn
