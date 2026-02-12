#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

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
void ReLU(std::shared_ptr<TensorImpl> x, std::shared_ptr<TensorImpl> out);
void ReLUBackward(std::shared_ptr<TensorImpl> x,
                  std::shared_ptr<TensorImpl> grad_out,
                  std::shared_ptr<TensorImpl> grad_x);
void SiLU(std::shared_ptr<TensorImpl> x, std::shared_ptr<TensorImpl> out);
void SiLUBackward(std::shared_ptr<TensorImpl> x,
                  std::shared_ptr<TensorImpl> grad_out,
                  std::shared_ptr<TensorImpl> grad_x);
void Softmax(std::shared_ptr<TensorImpl> x, std::shared_ptr<TensorImpl> out,
             uint64_t dim);
void SoftmaxBackward(std::shared_ptr<TensorImpl> y,
                     std::shared_ptr<TensorImpl> grad_out,
                     std::shared_ptr<TensorImpl> grad_x, uint64_t dim);
void Sqrt(std::shared_ptr<TensorImpl> x, std::shared_ptr<TensorImpl> out);
void SqrtBackward(std::shared_ptr<TensorImpl> x,
                  std::shared_ptr<TensorImpl> grad_out,
                  std::shared_ptr<TensorImpl> grad_x);

/**
 * Batched matrix multiply.
 * Expects matching leading batch dimensions and:
 *   a: (..., n, k), b: (..., k, m), out: (..., n, m)
 * Transposition flags are applied per-batch matrix before multiplication.
 */
void BatchedMatMul(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b,
                   std::shared_ptr<TensorImpl> out, bool transpose_a = false,
                   bool transpose_b = false);

/**
 * Reduce (sum).
 * Expects out to be contiguous
 */

void Reduce(std::shared_ptr<const TensorImpl> a,
            std::shared_ptr<TensorImpl> out, const std::vector<uint64_t>& dims,
            bool keep_dims);

};  // namespace cpu

};  // namespace deeptiny
