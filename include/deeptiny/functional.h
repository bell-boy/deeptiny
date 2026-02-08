#pragma once
// Common tensor functions
#include <vector>

#include "deeptiny/tensor.h"

namespace deeptiny {

namespace functional {
/**
 * Reduce a tensor along the given dimensions
 */
Tensor Reduce(const Tensor& x, const std::vector<uint64_t>& dims,
              bool keep_dims = false);
/**
 * Element-wise ReLU activation.
 */
Tensor ReLU(const Tensor& x);
/**
 * Softmax along a single dimension.
 */
Tensor Softmax(const Tensor& x, uint64_t dim);
/**
 * Element-wise square root.
 */
Tensor Sqrt(const Tensor& x);
}  // namespace functional

};  // namespace deeptiny
