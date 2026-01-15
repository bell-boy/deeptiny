// Broadcasting utils, hidden from users
#pragma once
#include <optional>
#include <utility>

#include "deeptiny/tensor.h"

namespace deeptiny {

namespace utils {

std::optional<Shape> GetBroadcastShape(const Tensor& a, const Tensor& b);

std::optional<Tensor> BroadcastToShape(const Tensor& a, const Shape& shape);

std::pair<Tensor, Tensor> Broadcast(const Tensor& a, const Tensor& b);

}  // namespace utils

}  // namespace deeptiny
