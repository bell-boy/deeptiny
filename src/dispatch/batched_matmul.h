#pragma once

#include <memory>

#include "tensor_impl.h"

namespace deeptiny::dispatch::batched_matmul {

std::shared_ptr<TensorImpl> OutOfPlace(const std::shared_ptr<TensorImpl>& a,
                                       const std::shared_ptr<TensorImpl>& b,
                                       bool transpose_a, bool transpose_b);

}  // namespace deeptiny::dispatch::batched_matmul
