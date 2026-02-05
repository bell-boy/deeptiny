#include "deeptiny/functional.h"

#include <algorithm>
#include <initializer_list>
#include <optional>
#include <random>
#include <unordered_set>

#include "deeptiny/view.h"
#include "utils.h"

namespace deeptiny {

namespace functional {

namespace detail {
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> uniform_double(0.0, 1.0);
};  // namespace detail

template <typename T>
Tensor _CreateUniform(Shape shape, DType dtype, Device device) {
  size_t total_size = utils::GetTotalSize(shape);
  T* buf = new T[total_size];
  for (size_t i = 0; i < total_size; ++i) {
    buf[i] = static_cast<T>(detail::uniform_double(detail::gen));
  }
  return Tensor::FromBuffer(
      std::span<std::byte>{(std::byte*)buf, total_size * sizeof(T)}, shape,
      dtype, device);
}

/**
 * Creates a uniform random tensor on the requested device.
 */
Tensor CreateUniform(Shape shape, Device device, DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return _CreateUniform<float>(shape, DType::Float32, device);
    default:
      throw std::runtime_error("DType is not supported yet");
  };
}

Tensor Zeros(Shape shape, Device device, DType dtype) {
  switch (dtype) {
    case DType::Float32: {
      Tensor result(shape, DType::Float32, device, false);
      auto impl = utils::TensorAccessor::GetTensorImpl(result);
      const size_t total_size = utils::GetTotalSize(shape);
      auto* data = static_cast<float*>(impl->data());
      std::fill_n(data, total_size, 0.0f);
      return result;
    }
    default:
      throw std::runtime_error("DType is not supported yet");
  };
}

// TODO: make this not suck
// TODO: support for inplace add
Tensor Reduce(const Tensor& x, std::initializer_list<uint64_t> dims) {
  std::unordered_set<uint64_t> dims_set(dims.begin(), dims.end());
  const auto& x_shape = x.shape();
  const uint64_t rank = x_shape.size();

  Shape reduced_shape;
  reduced_shape.reserve(rank);
  for (uint64_t i = 0; i < rank; ++i) {
    if (!dims_set.contains(i)) {
      reduced_shape.push_back(x_shape[i]);
    }
  }

  Tensor res = Zeros(reduced_shape, x.device(), x.dtype());

  if (x.dtype() != DType::Float32 || x.device() != Device::CPU) {
    throw std::runtime_error("Reduce only supports CPU Float32 for now");
  }

  auto x_impl = utils::TensorAccessor::GetTensorImpl(x);
  auto res_impl = utils::TensorAccessor::GetTensorImpl(res);
  const auto& x_stride = x_impl->stride();
  const auto& res_stride = res_impl->stride();

  const auto* x_data = static_cast<const float*>(x_impl->data());
  auto* res_data = static_cast<float*>(res_impl->data());

  if (rank == 0) {
    res_data[0] += x_data[0];
    return res;
  }

  std::vector<uint64_t> index(rank, 0);
  const uint64_t out_rank = reduced_shape.size();
  std::vector<uint64_t> out_index(out_rank, 0);

  auto compute_offsets = [&](const std::vector<uint64_t>& idx) {
    int64_t x_offset = static_cast<int64_t>(x_impl->offset());
    for (uint64_t i = 0; i < rank; ++i) {
      x_offset += static_cast<int64_t>(idx[i]) * x_stride[i];
    }
    return x_offset;
  };

  auto compute_out_offset = [&](const std::vector<uint64_t>& out_idx) {
    int64_t out_offset = static_cast<int64_t>(res_impl->offset());
    for (uint64_t i = 0; i < out_rank; ++i) {
      out_offset += static_cast<int64_t>(out_idx[i]) * res_stride[i];
    }
    return out_offset;
  };

  while (true) {
    uint64_t out_dim = 0;
    for (uint64_t i = 0; i < rank; ++i) {
      if (!dims_set.contains(i)) {
        out_index[out_dim++] = index[i];
      }
    }

    const int64_t x_offset = compute_offsets(index);
    const int64_t out_offset =
        out_rank == 0 ? 0 : compute_out_offset(out_index);

    res_data[static_cast<size_t>(out_offset)] +=
        x_data[static_cast<size_t>(x_offset)];

    uint64_t dim = rank;
    while (dim > 0) {
      --dim;
      index[dim] += 1;
      if (index[dim] < x_shape[dim]) {
        break;
      }
      index[dim] = 0;
      if (dim == 0) {
        return res;
      }
    }
  }
}

};  // namespace functional

};  // namespace deeptiny
