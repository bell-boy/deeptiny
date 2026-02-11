#pragma once

#include <memory>

#include "tensor_impl.h"

namespace deeptiny::dispatch::reduce {

std::shared_ptr<TensorImpl> OutOfPlace(const std::shared_ptr<TensorImpl>& a,
                                       const std::vector<uint64_t>& dims,
                                       bool keep_dims);

}  // namespace deeptiny::dispatch::reduce
