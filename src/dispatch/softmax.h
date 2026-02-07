#pragma once

#include <cstdint>
#include <memory>

#include "tensor_impl.h"

namespace deeptiny::dispatch::softmax {

std::shared_ptr<TensorImpl> OutOfPlace(const std::shared_ptr<TensorImpl>& x,
                                       uint64_t dim);

std::shared_ptr<TensorImpl> Backward(
    const std::shared_ptr<TensorImpl>& y,
    const std::shared_ptr<TensorImpl>& grad_out, uint64_t dim);

}  // namespace deeptiny::dispatch::softmax
