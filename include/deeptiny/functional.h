#pragma once
// Common tensor functions
#include "deeptiny/tensor.h"

namespace deeptiny {

namespace functional {
/**
 * Generates a tensor with entries sampled i.i.d from a Uniform(0, 1)
 */
Tensor CreateUniform(Shape shape, Device device = Device::CPU,
                     DType dtype = DType::Float32);
}  // namespace functional

};  // namespace deeptiny
