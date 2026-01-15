#pragma once

#include <memory>
#include <vector>

#include "deeptiny/tensor.h"
#include "engine.h"
#include "tensor_impl.h"

namespace deeptiny {

class View : public Tensor {
 public:
  void operator=(const Tensor& other);
  View(std::shared_ptr<TensorImpl> tensor_impl,
       std::shared_ptr<AutogradMeta> autograd_meta);
};

class SliceBackward : public Function {
 private:
  std::vector<Slice> slices_;
  Shape original_shape_;

 public:
  SliceBackward(const Tensor& t, std::vector<Slice> slices);
  void operator()(const Tensor& grad, Engine& engine) override;
};

}  // namespace deeptiny
