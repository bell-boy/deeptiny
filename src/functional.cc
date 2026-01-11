#include "deeptiny/functional.h"

#include <random>

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
      std::runtime_error("DType is not supported yet");
      exit(-1);
  };
}

};  // namespace functional

};  // namespace deeptiny
