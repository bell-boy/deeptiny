#pragma once
// Common tensor functions
#include "deeptiny/tensor.h"

namespace deeptiny {

namespace functional {
/**
 * Generates a tensor with entries sampled i.i.d from a Uniform(0, 1)
 */
Tensor CreateUniform(Shape shape, DType dtype);
}  // namespace functional

};  // namespace deeptiny
