#pragma once
#include <memory>
#include <span>

#include "tensor_impl.h"

namespace deeptiny {

namespace cpu {

/**
 * Copy the data from the buffer into a TensorImpl on CPU
 */
std::shared_ptr<TensorImpl> FromBuffer(DType dtype,
                                       std::span<const std::byte> buffer,
                                       Shape shape);

/**
 * Add the two TensorImpls element-wise and store the result in the output
 * TensorImpl
 */
void Add(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b,
         std::shared_ptr<TensorImpl> out);
void Sub(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b,
         std::shared_ptr<TensorImpl> out);
void Mul(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b,
         std::shared_ptr<TensorImpl> out);
void Div(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b,
         std::shared_ptr<TensorImpl> out);

};  // namespace cpu

};  // namespace deeptiny
