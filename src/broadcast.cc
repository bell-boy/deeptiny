#include "broadcast.h"

#include <algorithm>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "deeptiny/functional.h"
#include "utils.h"

namespace deeptiny {

namespace utils {

BroadcastBackward::BroadcastBackward(const Tensor& t)
    : Function({utils::TensorAccessor::GetAutogradMeta(t)}),
      original_shape_(t.shape()) {}

void BroadcastBackward::operator()(const Tensor& grad, Engine& engine) {
  const auto& out_shape = grad.shape();
  const auto out_rank = out_shape.size();
  const auto in_rank = original_shape_.size();

  if (out_rank < in_rank) {
    throw std::runtime_error("BroadcastBackward received invalid grad shape");
  }

  const size_t rank_diff = out_rank - in_rank;

  std::vector<uint64_t> reduce_dims;
  reduce_dims.reserve(out_rank);
  for (uint64_t i = 0; i < out_rank; ++i) {
    if (i < rank_diff) {
      reduce_dims.push_back(i);
      continue;
    }
    const uint64_t in_i = i - rank_diff;
    if (original_shape_[in_i] == 1 && out_shape[i] != 1) {
      reduce_dims.push_back(i);
    }
  }

  Tensor reduced =
      reduce_dims.empty() ? grad : functional::Reduce(grad, reduce_dims, true);

  Tensor grad_in = reduced;
  if (rank_diff > 0) {
    std::vector<uint64_t> squeeze_dims;
    squeeze_dims.reserve(rank_diff);
    for (uint64_t i = 0; i < rank_diff; ++i) {
      squeeze_dims.push_back(i);
    }
    grad_in = reduced.Squeeze(squeeze_dims);
  }

  auto parent = getParents()[0];
  if (!parent) {
    return;
  }
  parent->updateGrad(grad_in, engine);
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
  auto backward = std::make_shared<BroadcastBackward>(a);
  auto parent_meta = TensorAccessor::GetAutogradMeta(a);
  const bool requires_grad = parent_meta && parent_meta->requires_grad();
  auto grad_meta = std::make_shared<AutogradMeta>(backward, requires_grad);
  return TensorAccessor::MakeTensor(new_tensor_impl, grad_meta);
}

std::pair<Tensor, Tensor> Broadcast(const Tensor& a, const Tensor& b) {
  auto target_shape = GetBroadcastShape(a, b);
  if (!target_shape) {
    std::ostringstream err;
    err << "Cannot broadcast tensors with shapes " << FormatShape(a.shape())
        << " and " << FormatShape(b.shape()) << ".";
    throw std::runtime_error(err.str());
  }

  auto a_broadcast = BroadcastToShape(a, *target_shape);
  auto b_broadcast = BroadcastToShape(b, *target_shape);
  if (!a_broadcast || !b_broadcast) {
    std::ostringstream err;
    err << "Failed to broadcast tensors to shape " << FormatShape(*target_shape)
        << ".";
    throw std::runtime_error(err.str());
  }
  return {*a_broadcast, *b_broadcast};
}

}  // namespace utils

}  // namespace deeptiny
