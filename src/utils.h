// Lib level utils, hidden from users
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "deeptiny/tensor.h"
#include "tensorImpl.h"

namespace deeptiny {

namespace utils {

struct TensorAccessor {
  static std::shared_ptr<TensorImpl> GetTensorImpl(Tensor& t);
};

uint64_t GetTotalSize(Shape shape);

Stride GetContinguousStride(Shape shape);

};  // namespace utils

};  // namespace deeptiny
