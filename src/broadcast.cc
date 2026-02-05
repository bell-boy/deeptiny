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

namespace {
template <typename Fn>
void ForEachIndex(const Shape& shape, Fn&& fn) {
  if (shape.empty()) {
    fn(std::vector<uint64_t>{});
    return;
  }
  for (auto dim : shape) {
    if (dim == 0) {
      return;
    }
  }
  std::vector<uint64_t> index(shape.size(), 0);
  while (true) {
    fn(index);
    size_t dim = shape.size();
    while (dim > 0) {
      --dim;
      index[dim] += 1;
      if (index[dim] < shape[dim]) {
        break;
      }
      index[dim] = 0;
      if (dim == 0) {
        return;
      }
    }
  }
}

int64_t OffsetForIndex(const Stride& stride,
                       const std::vector<uint64_t>& index) {
  int64_t offset = 0;
  for (size_t i = 0; i < stride.size(); ++i) {
    offset += static_cast<int64_t>(index[i]) * stride[i];
  }
  return offset;
}
}  // namespace

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

  Tensor grad_in =
      functional::Zeros(original_shape_, grad.device(), grad.dtype());
  auto grad_in_impl = TensorAccessor::GetTensorImpl(grad_in);
  auto grad_out_impl = TensorAccessor::GetTensorImpl(grad);

  if (grad.dtype() != DType::Float32 || grad.device() != Device::CPU) {
    throw std::runtime_error("BroadcastBackward only supports CPU Float32");
  }

  const auto* out_data = static_cast<const float*>(grad_out_impl->data());
  auto* in_data = static_cast<float*>(grad_in_impl->data());

  const auto& out_stride = grad_out_impl->stride();
  const auto& in_stride = grad_in_impl->stride();
  const size_t rank_diff = out_rank - in_rank;

  ForEachIndex(out_shape, [&](const std::vector<uint64_t>& out_index) {
    std::vector<uint64_t> in_index(in_rank, 0);
    for (size_t i = 0; i < in_rank; ++i) {
      const uint64_t out_i = out_index[rank_diff + i];
      in_index[i] = (original_shape_[i] == 1) ? 0 : out_i;
    }
    const int64_t out_offset = OffsetForIndex(out_stride, out_index);
    const int64_t in_offset = OffsetForIndex(in_stride, in_index);
    in_data[static_cast<size_t>(in_offset)] +=
        out_data[static_cast<size_t>(out_offset)];
  });

  auto parent = getParents()[0];
  if (!parent) {
    return;
  }
  if (parent->pending() == 0) {
    parent->incrementPending();
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

// TODO: handle backward
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
  return Tensor(new_tensor_impl, grad_meta);
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
