#pragma once

#include <memory>
#include <vector>

#include "deeptiny/tensor.h"

namespace deeptiny {

class View : public Tensor {
 public:
  void operator=(const Tensor& other);

 private:
  View(std::shared_ptr<TensorImpl> tensor_impl,
       std::shared_ptr<AutogradMeta> autograd_meta);

  friend struct utils::TensorAccessor;
};

}  // namespace deeptiny
