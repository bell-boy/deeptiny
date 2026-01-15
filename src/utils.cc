#include "utils.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "deeptiny/functional.h"
#include "deeptiny/tensor.h"
#include "deeptiny/view.h"

namespace deeptiny {

namespace utils {

std::shared_ptr<TensorImpl> TensorAccessor::GetTensorImpl(const Tensor& t) {
  return t.tensor_impl_;
}

std::shared_ptr<AutogradMeta> TensorAccessor::GetAutogradMeta(const Tensor& t) {
  return t.autograd_meta_;
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

std::optional<Tensor> SliceScatterToShape(const Tensor& a, const Shape& shape,
                                          const std::vector<Slice>& slices) {
  Tensor res = functional::Zeros(shape, a.device(), a.dtype());
  try {
    res(slices) = a;
  } catch (...) {
    return std::nullopt;
  }
  return res;
}

};  // namespace utils

};  // namespace deeptiny
