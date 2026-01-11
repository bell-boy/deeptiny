#pragma once
#include <memory>
#include <span>

#include "tensorImpl.h"

namespace deeptiny {

namespace cpu {

/**
 * Copy the data from the buffer into a TensorImpl on CPU
 */
std::shared_ptr<TensorImpl> FromBuffer(DType dtype, std::span<std::byte> buffer,
                                       Shape shape);

};  // namespace cpu

};  // namespace deeptiny
