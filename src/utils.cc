#include "utils.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace deeptiny {

namespace utils {

std::shared_ptr<TensorImpl> TensorAccessor::GetTensorImpl(const Tensor& t) {
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

std::optional<Shape> GetBroadcastShape(const Tensor& a, const Tensor& b) {
  Shape a_shape = a.shape();
  Shape b_shape = b.shape();
  uint64_t a_rank = a_shape.size();
  uint64_t b_rank = b_shape.size();
  uint64_t new_shape_rank = std::max(a_rank, b_rank);
  Shape new_shape(new_shape_rank, 0);
  for (uint64_t i = 0; i < new_shape_rank; ++i) {
    if (i < a_rank && i < b_rank) {
      auto a_dim = a_shape[a_rank - i - 1];
      auto b_dim = b_shape[b_rank - i - 1];
      if (a_dim != b_dim && !(a_dim == 1 || b_dim == 1)) {
        return std::nullopt;
      } else {
        new_shape[new_shape_rank - i - 1] = std::max(a_dim, b_dim);
      }
    } else if (i < a_rank) {
      auto a_dim = a_shape[a_rank - i - 1];
      new_shape[new_shape_rank - i - 1] = a_dim;
    } else {
      auto b_dim = b_shape[b_rank - i - 1];
      new_shape[new_shape_rank - i - 1] = b_dim;
    }
  }
  return new_shape;
}

std::optional<Tensor> BroadcastToShape(const Tensor& a, const Shape& shape) {
  // check that the shape works
  Shape a_shape = a.shape();
  auto a_rank = a_shape.size();
  auto shape_rank = shape.size();
  if (shape_rank < a_rank) {
    return std::nullopt;
  }
  for (uint64_t i = 0; i < a_rank; ++i) {
    if (a_shape[a_rank - i - 1] != shape[shape_rank - i - 1] &&
        a_shape[a_rank - i - 1] != 1) {
      return std::nullopt;
    }
  }
  if (a_shape == shape) {
    return a;
  }
  // calculate the strides needed
  Stride new_strides(shape_rank, 0);
  auto a_impl = TensorAccessor::GetTensorImpl(a);
  const auto& a_stride = a_impl->stride();
  for (uint64_t i = 0; i < shape_rank; ++i) {
    const uint64_t out_index = shape_rank - i - 1;
    if (i >= a_rank) {
      new_strides[out_index] = 0;
      continue;
    }
    const uint64_t a_dim = a_shape[a_rank - i - 1];
    const uint64_t out_dim = shape[out_index];
    if (a_dim == out_dim) {
      new_strides[out_index] = a_stride[a_rank - i - 1];
    } else {
      new_strides[out_index] = 0;
    }
  }
  auto new_tensor_impl =
      a_impl->View(Shape(shape), std::move(new_strides), a_impl->offset());
  return Tensor(new_tensor_impl);
}

std::pair<Tensor, Tensor> Broadcast(const Tensor& a, const Tensor& b) {
  auto target_shape = GetBroadcastShape(a, b);
  if (!target_shape) {
    auto format_shape = [](const Shape& shape) {
      std::ostringstream oss;
      oss << "{ ";
      for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (i + 1 < shape.size()) {
          oss << ", ";
        }
      }
      oss << " }";
      return oss.str();
    };
    std::ostringstream err;
    err << "Cannot broadcast tensors with shapes " << format_shape(a.shape())
        << " and " << format_shape(b.shape()) << ".";
    throw std::runtime_error(err.str());
  }

  auto a_broadcast = BroadcastToShape(a, *target_shape);
  auto b_broadcast = BroadcastToShape(b, *target_shape);
  if (!a_broadcast || !b_broadcast) {
    std::ostringstream err;
    err << "Failed to broadcast tensors to shape { ";
    for (size_t i = 0; i < target_shape->size(); ++i) {
      err << (*target_shape)[i];
      if (i + 1 < target_shape->size()) {
        err << ", ";
      }
    }
    err << " }.";
    throw std::runtime_error(err.str());
  }
  return {*a_broadcast, *b_broadcast};
}

};  // namespace utils

};  // namespace deeptiny
