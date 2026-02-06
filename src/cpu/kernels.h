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

/**
 * Batched matrix multiply.
 * Expects matching leading batch dimensions and:
 *   a: (..., n, k), b: (..., k, m), out: (..., n, m)
 */
void BatchedMatMul(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b,
                   std::shared_ptr<TensorImpl> out);

/**
 * Gradient w.r.t. a for batched matrix multiply.
 * Expects:
 *   grad_out: (..., n, m), b: (..., k, m), grad_a: (..., n, k)
 */
void BatchedMatMulGradA(std::shared_ptr<TensorImpl> grad_out,
                        std::shared_ptr<TensorImpl> b,
                        std::shared_ptr<TensorImpl> grad_a);

/**
 * Gradient w.r.t. b for batched matrix multiply.
 * Expects:
 *   a: (..., n, k), grad_out: (..., n, m), grad_b: (..., k, m)
 */
void BatchedMatMulGradB(std::shared_ptr<TensorImpl> a,
                        std::shared_ptr<TensorImpl> grad_out,
                        std::shared_ptr<TensorImpl> grad_b);

};  // namespace cpu

};  // namespace deeptiny
