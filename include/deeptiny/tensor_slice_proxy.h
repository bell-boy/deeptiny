#pragma once

#include <vector>

#include "deeptiny/types.h"

namespace deeptiny {

class Tensor;

class TensorSliceProxy {
 public:
  TensorSliceProxy(Tensor* base, std::vector<Slice> slices);
  TensorSliceProxy(const Tensor* base, std::vector<Slice> slices);

  TensorSliceProxy& operator=(const Tensor& rhs);
  operator Tensor() const;

 private:
  Tensor* mutable_base_;
  const Tensor* base_;
  std::vector<Slice> slices_;
};

}  // namespace deeptiny
