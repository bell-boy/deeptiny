// Lib level utils, hidden from users
#pragma once
#include <stddef.h>
#include <stdint.h>

#include <optional>

#include "deeptiny/tensor.h"
#include "tensor_impl.h"

namespace deeptiny {

namespace utils {

struct TensorAccessor {
  static std::shared_ptr<TensorImpl> GetTensorImpl(const Tensor& t);
};

uint64_t GetTotalSize(Shape shape);

Stride GetContinguousStride(Shape shape);

std::optional<Shape> GetBroadcastShape(const Tensor& a, const Tensor& b);

std::optional<Tensor> BroadcastToShape(const Tensor& a, const Shape& shape);

std::pair<Tensor, Tensor> Broadcast(const Tensor& a, const Tensor& b);

};  // namespace utils

};  // namespace deeptiny
