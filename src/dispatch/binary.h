#pragma once

#include <memory>

#include "tensor_impl.h"

namespace deeptiny::dispatch::binary {

enum class Op { Add, Sub, Mul, Div };

std::shared_ptr<TensorImpl> OutOfPlace(Op op,
                                       const std::shared_ptr<TensorImpl>& a,
                                       const std::shared_ptr<TensorImpl>& b);

void Inplace(Op op, const std::shared_ptr<TensorImpl>& self,
             const std::shared_ptr<TensorImpl>& other);

}  // namespace deeptiny::dispatch::binary
