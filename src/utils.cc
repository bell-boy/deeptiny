#include "utils.h"

namespace deeptiny {

namespace utils {

std::shared_ptr<TensorImpl> TensorAccessor::GetTensorImpl(Tensor& t) {
  return t.tensor_impl_;
}

uint64_t GetTotalSize(Shape shape) {
  uint64_t total = 1;
  for (const auto& x : shape) {
    total *= x;
  }
  return total;
}

Stride GetContinguousStride(Shape shape) {
  Stride stride(shape.size());
  int64_t running = 1;
  for (size_t i = shape.size(); i-- > 0;) {
    stride[i] = running;
    running *= static_cast<int64_t>(shape[i]);
  }
  return stride;
}

};  // namespace utils

};  // namespace deeptiny
