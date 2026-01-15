#pragma once
// Common tensor functions
#include <initializer_list>

#include "deeptiny/tensor.h"

namespace deeptiny {

namespace functional {
/**
 * Generates a tensor with entries sampled i.i.d from a Uniform(0, 1)
 */
Tensor CreateUniform(Shape shape, Device device = Device::CPU,
                     DType dtype = DType::Float32);

Tensor Zeros(Shape shape, Device device = Device::CPU,
             DType dtype = DType::Float32);
}  // namespace functional

/**
 * Reduce a tensor along the given dimensions
 */
Tensor Reduce(const Tensor& x, std::initializer_list<uint64_t> dims);

};  // namespace deeptiny
