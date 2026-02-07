#include "dispatch/batched_matmul.h"

#include <sstream>
#include <stdexcept>

#include "cpu/kernels.h"

namespace deeptiny::dispatch::batched_matmul {

std::shared_ptr<TensorImpl> OutOfPlace(const std::shared_ptr<TensorImpl>& a,
                                       const std::shared_ptr<TensorImpl>& b,
                                       bool transpose_a, bool transpose_b) {
  const auto& a_shape = a->shape();
  const auto& b_shape = b->shape();
  const uint64_t a_rows = a_shape[a_shape.size() - 2];
  const uint64_t a_cols = a_shape[a_shape.size() - 1];
  const uint64_t b_rows = b_shape[b_shape.size() - 2];
  const uint64_t b_cols = b_shape[b_shape.size() - 1];
  const uint64_t lhs_rows = transpose_a ? a_cols : a_rows;
  const uint64_t rhs_cols = transpose_b ? b_rows : b_cols;

  Shape out_shape(a_shape.begin(), a_shape.end() - 2);
  out_shape.push_back(lhs_rows);
  out_shape.push_back(rhs_cols);

  auto out = std::make_shared<TensorImpl>(out_shape, a->dtype(), a->device());
  switch (out->device()) {
    case Device::CPU:
      cpu::BatchedMatMul(a, b, out, transpose_a, transpose_b);
      return out;
    default: {
      std::stringstream err;
      err << "Operation does not support " << out->device().ToString();
      throw std::runtime_error(err.str());
    }
  }
}

}  // namespace deeptiny::dispatch::batched_matmul
