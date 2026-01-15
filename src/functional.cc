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
  Shape reduced_shape;
  for (const auto& dim : x.shape()) {
    if (!dims_set.contains(dim)) {
      reduced_shape.push_back(dim);
    }
  }
  Tensor res = Zeros(reduced_shape, x.device(), x.dtype());
  auto recursive_reduce = [&res, &x, &dims_set](auto&& self, uint64_t dim_idx,
                                                std::vector<Slice>& slices) {
    if (slices.size() == x.shape().size()) {
      // res += x(slices);
      return;
    }
    if (dims_set.contains(dim_idx)) {
      for (uint64_t i = 0; i < x.shape()[dim_idx]; ++i) {
        slices.push_back(Slice(i));
        self(self, dim_idx + 1, slices);
        slices.pop_back();
      }
    } else {
      slices.push_back(Slice(std::nullopt, std::nullopt));
      self(self, dim_idx + 1, slices);
      slices.pop_back();
    }
  };
  std::vector<Slice> slices;
  recursive_reduce(recursive_reduce, 0, slices);
  return res;
}

};  // namespace functional

};  // namespace deeptiny
