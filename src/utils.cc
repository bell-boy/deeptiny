#include "utils.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

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

Tensor TensorAccessor::MakeTensor(std::shared_ptr<TensorImpl> tensor_impl,
                                  std::shared_ptr<AutogradMeta> autograd_meta) {
  return Tensor(std::move(tensor_impl), std::move(autograd_meta));
}

View TensorAccessor::MakeView(std::shared_ptr<TensorImpl> tensor_impl,
                              std::shared_ptr<AutogradMeta> autograd_meta) {
  return View(std::move(tensor_impl), std::move(autograd_meta));
}

void CompatabilityCheck(std::initializer_list<Tensor> tensors) {
  if (tensors.size() == 0) return;

  DType dtype = tensors.begin()->dtype();
  Device device = tensors.begin()->device();
  for (const auto& t : tensors) {
    if (t.dtype() != dtype) {
      std::stringstream err;
      err << "Tensor dtype mismatch: one tensor of type " << dtype.ToString()
          << " but other of type " << t.dtype().ToString();
      throw std::runtime_error(err.str());
    }
    if (t.device() != device) {
      std::stringstream err;
      err << "Tensor device mismatch: one tensor on device "
          << device.ToString() << " but other tensor on device " << t.device();
      throw std::runtime_error(err.str());
    }
  }
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
