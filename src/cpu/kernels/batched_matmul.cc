#include "cpu/kernels.h"

#include <cblas.h>

#include <memory>
#include <sstream>
#include <stdexcept>

#include "cpu/kernels/common.h"
#include "utils.h"

namespace deeptiny::cpu {

namespace {
void ValidateBatchedMatMulInputs(const std::shared_ptr<TensorImpl>& a,
                                 const std::shared_ptr<TensorImpl>& b,
                                 const std::shared_ptr<TensorImpl>& out,
                                 bool transpose_a, bool transpose_b) {
  if (!a || !b || !out) {
    throw std::runtime_error("BatchedMatMul received null TensorImpl.");
  }
  if (a->dtype() != b->dtype() || a->dtype() != out->dtype()) {
    throw std::runtime_error("BatchedMatMul requires matching tensor dtypes.");
  }
  if (a->dtype() != DType::Float32) {
    throw std::runtime_error("Only Float32 dtype is supported in BatchedMatMul.");
  }

  const auto& a_shape = a->shape();
  const auto& b_shape = b->shape();
  const auto& out_shape = out->shape();

  if (a_shape.size() < 3 || b_shape.size() < 3 || out_shape.size() < 3) {
    throw std::runtime_error("BatchedMatMul expects rank >= 3 tensors.");
  }
  if (a_shape.size() != b_shape.size() || a_shape.size() != out_shape.size()) {
    throw std::runtime_error("BatchedMatMul requires equal tensor ranks.");
  }

  const size_t rank = a_shape.size();
  for (size_t dim = 0; dim + 2 < rank; ++dim) {
    if (a_shape[dim] != b_shape[dim] || a_shape[dim] != out_shape[dim]) {
      std::stringstream err;
      err << "BatchedMatMul requires matching batch dims, got a="
          << FormatShape(a_shape) << ", b=" << FormatShape(b_shape)
          << ", out=" << FormatShape(out_shape);
      throw std::runtime_error(err.str());
    }
  }

  const uint64_t a_src_rows = a_shape[rank - 2];
  const uint64_t a_src_cols = a_shape[rank - 1];
  const uint64_t b_src_rows = b_shape[rank - 2];
  const uint64_t b_src_cols = b_shape[rank - 1];

  const uint64_t a_rows = transpose_a ? a_src_cols : a_src_rows;
  const uint64_t a_cols = transpose_a ? a_src_rows : a_src_cols;
  const uint64_t b_rows = transpose_b ? b_src_cols : b_src_rows;
  const uint64_t b_cols = transpose_b ? b_src_rows : b_src_cols;

  assert(a_cols == b_rows &&
         "BatchedMatMul kernel requires matching inner dimensions");
  assert(out_shape[rank - 2] == a_rows &&
         "BatchedMatMul kernel output rows mismatch");
  assert(out_shape[rank - 1] == b_cols &&
         "BatchedMatMul kernel output cols mismatch");

  assert(out->isContiguous() &&
         "BatchedMatMul kernel requires contiguous output tensor");
  assert(out->offset() == 0 &&
         "BatchedMatMul kernel requires output tensor offset == 0");
  assert(out->storage()->numel() == utils::GetTotalSize(out_shape) &&
         "BatchedMatMul kernel requires output storage size to match shape");
}
}  // namespace

void BatchedMatMul(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b,
                   std::shared_ptr<TensorImpl> out, bool transpose_a,
                   bool transpose_b) {
  ValidateBatchedMatMulInputs(a, b, out, transpose_a, transpose_b);

  const auto& out_shape = out->shape();
  if (utils::GetTotalSize(out_shape) == 0) {
    return;
  }

  std::shared_ptr<const Storage> a_storage = a->getContiguousStorage();
  std::shared_ptr<const Storage> b_storage = b->getContiguousStorage();
  auto out_storage = out->storage();

  const auto* a_data = static_cast<const float*>(a_storage->data(0));
  const auto* b_data = static_cast<const float*>(b_storage->data(0));
  auto* out_data = static_cast<float*>(out_storage->data(0));

  const auto& a_shape = a->shape();
  const auto& b_shape = b->shape();
  const size_t rank = out_shape.size();
  const size_t batch_rank = rank - 2;

  const uint64_t a_src_rows = a_shape[rank - 2];
  const uint64_t a_src_cols = a_shape[rank - 1];
  const uint64_t b_src_rows = b_shape[rank - 2];
  const uint64_t b_src_cols = b_shape[rank - 1];

  const uint64_t rows = transpose_a ? a_src_cols : a_src_rows;
  const uint64_t inner = transpose_a ? a_src_rows : a_src_cols;
  const uint64_t cols = transpose_b ? b_src_rows : b_src_cols;

  const int cblas_rows = detail::ToCblasInt(rows, "rows");
  const int cblas_cols = detail::ToCblasInt(cols, "cols");
  const int cblas_inner = detail::ToCblasInt(inner, "inner dimension");
  const uint64_t a_matrix_size = a_src_rows * a_src_cols;
  const uint64_t b_matrix_size = b_src_rows * b_src_cols;
  const uint64_t out_matrix_size = out_shape[rank - 2] * out_shape[rank - 1];

  uint64_t batch_size = 1;
  for (size_t dim = 0; dim < batch_rank; ++dim) {
    batch_size *= out_shape[dim];
  }

  const CBLAS_TRANSPOSE cblas_transpose_a =
      transpose_a ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE cblas_transpose_b =
      transpose_b ? CblasTrans : CblasNoTrans;
  const int lda = transpose_a ? cblas_rows : cblas_inner;
  const int ldb = transpose_b ? cblas_inner : cblas_cols;
  const int ldc = cblas_cols;

  for (uint64_t batch = 0; batch < batch_size; ++batch) {
    const float* a_batch = a_data + static_cast<size_t>(batch * a_matrix_size);
    const float* b_batch = b_data + static_cast<size_t>(batch * b_matrix_size);
    float* out_batch = out_data + static_cast<size_t>(batch * out_matrix_size);
    cblas_sgemm(CblasRowMajor, cblas_transpose_a, cblas_transpose_b,
                cblas_rows, cblas_cols, cblas_inner, 1.0f, a_batch, lda,
                b_batch, ldb, 0.0f, out_batch, ldc);
  }
}

}  // namespace deeptiny::cpu
