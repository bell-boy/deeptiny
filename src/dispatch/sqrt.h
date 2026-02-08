#pragma once

#include <memory>

#include "tensor_impl.h"

namespace deeptiny::dispatch::sqrt {

std::shared_ptr<TensorImpl> OutOfPlace(const std::shared_ptr<TensorImpl>& x);

std::shared_ptr<TensorImpl> Backward(const std::shared_ptr<TensorImpl>& x,
                                     const std::shared_ptr<TensorImpl>& grad);

}  // namespace deeptiny::dispatch::sqrt
