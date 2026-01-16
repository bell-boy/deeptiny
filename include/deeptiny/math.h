#pragma once
#include <string>

#include "deeptiny/tensor.h"
#include "engine.h"

namespace deeptiny {
namespace math {

class AddBackward : public Function {
 public:
  AddBackward(const Tensor& a, const Tensor& b);
  void operator()(const Tensor& grad, Engine& engine) override;
};

Tensor operator+(const Tensor& a, const Tensor& b);

};  // namespace math
};  // namespace deeptiny
