#include "deeptiny/functional.h"

#include <random>

#include "utils.h"

namespace deeptiny {

namespace functional {

namespace detail {
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> uniform_float32(0.0f, 1.0f);
};  // namespace detail

// TODO: Make these device agnostic

Tensor _CreateUniformFloat32(Shape shape) {
  size_t total_size = utils::GetTotalSize(shape);
  float* buf = new float[total_size];
  for (size_t i = 0; i < total_size; ++i) {
    buf[i] = detail::uniform_float32(detail::gen);
  }
  return Tensor::FromBuffer(
      DType::Float32,
      std::span<std::byte>{(std::byte*)buf, total_size * sizeof(float)}, shape);
}

/**
 * Creates a uniform random tensor on the cpu
 */
Tensor CreateUniform(Shape shape, DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return _CreateUniformFloat32(shape);
    default:
      std::runtime_error("DType is not supported yet");
      exit(-1);
  };
}

};  // namespace functional

};  // namespace deeptiny