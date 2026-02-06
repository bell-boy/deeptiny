#pragma once

#include <vector>

#include "deeptiny/tensor.h"
#include "engine.h"

namespace deeptiny {

class SliceBackward : public Function {
 private:
  std::vector<Slice> slices_;
  Shape original_shape_;

 public:
  SliceBackward(const Tensor& t, std::vector<Slice> slices);
  void operator()(const Tensor& grad) override;
};

class ViewAssignBackward : public Function {
 public:
  explicit ViewAssignBackward(const Tensor& src);
  void operator()(const Tensor& grad) override;
};

}  // namespace deeptiny
