// Lib level utils, hidden from users
#pragma once
#include <stddef.h>
#include <stdint.h>

#include <initializer_list>
#include <optional>
#include <vector>

#include "autograd_meta.h"
#include "broadcast.h"
#include "deeptiny/tensor.h"
#include "tensor_impl.h"

namespace deeptiny {

namespace utils {

struct TensorAccessor {
  static std::shared_ptr<TensorImpl> GetTensorImpl(const Tensor& t);
  static std::shared_ptr<AutogradMeta> GetAutogradMeta(const Tensor& t);
  static Tensor MakeTensor(std::shared_ptr<TensorImpl> tensor_impl,
                           std::shared_ptr<AutogradMeta> autograd_meta);
};

/**
 * Check that all tensors are on the same device and have the same dtype
 */
void CompatabilityCheck(std::initializer_list<Tensor> tensors);

uint64_t GetTotalSize(Shape shape);

Stride GetContinguousStride(Shape shape);

std::optional<Tensor> SliceScatterToShape(const Tensor& a, const Shape& shape,
                                          const std::vector<Slice>& slices);

};  // namespace utils

};  // namespace deeptiny
