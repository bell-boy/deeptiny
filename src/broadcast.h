// Broadcasting utils, hidden from users
#pragma once
#include <optional>
#include <utility>

#include "deeptiny/tensor.h"
#include "engine.h"

namespace deeptiny {

namespace utils {

class BroadcastBackward : public Function {
 private:
  Shape original_shape_;

 public:
  explicit BroadcastBackward(const Tensor& t);
  void operator()(const Tensor& grad) override;
};

std::optional<Shape> GetBroadcastShape(const Tensor& a, const Tensor& b);

std::optional<Tensor> BroadcastToShape(const Tensor& a, const Shape& shape);

std::pair<Tensor, Tensor> Broadcast(const Tensor& a, const Tensor& b);

}  // namespace utils

}  // namespace deeptiny
