#include "cpu/kernels.h"

#include <cblas.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "utils.h"

namespace deeptiny {

namespace cpu {

namespace {
void ValidateElementwiseBinaryOpInputs(const std::shared_ptr<TensorImpl>& a,
                                       const std::shared_ptr<TensorImpl>& b,
                                       const std::shared_ptr<TensorImpl>& out) {
  if (!a || !b || !out) {
    throw std::runtime_error("Elementwise op received null TensorImpl");
  }
  assert(a->shape() == b->shape());
  assert(a->shape() == out->shape());
  assert(a->dtype() == b->dtype());
  assert(a->dtype() == out->dtype());

  if (a->shape() != b->shape() || a->shape() != out->shape()) {
    throw std::runtime_error("Elementwise op requires matching tensor shapes");
  }
  if (a->dtype() != b->dtype() || a->dtype() != out->dtype()) {
    throw std::runtime_error("Elementwise op requires matching tensor dtypes");
  }
}

template <typename Op>
void ApplyElementwiseBinaryOp(const std::shared_ptr<TensorImpl>& a,
                              const std::shared_ptr<TensorImpl>& b,
                              const std::shared_ptr<TensorImpl>& out,
                              const char* op_name, Op&& op) {
  ValidateElementwiseBinaryOpInputs(a, b, out);

  if (a->dtype() != DType::Float32) {
    std::stringstream err;
    err << "Only Float32 dtype is supported in " << op_name;
    throw std::runtime_error(err.str());
  }

  const auto a_storage = static_cast<const TensorImpl&>(*a).storage();
  const auto b_storage = static_cast<const TensorImpl&>(*b).storage();
  auto out_storage = out->storage();

  const auto* a_data = static_cast<const float*>(a_storage->data(0));
  const auto* b_data = static_cast<const float*>(b_storage->data(0));
  auto* out_data = static_cast<float*>(out_storage->data(0));

  const int64_t a_base = static_cast<int64_t>(a->offset());
  const int64_t b_base = static_cast<int64_t>(b->offset());
  const int64_t out_base = static_cast<int64_t>(out->offset());

  const auto& shape = out->shape();
  const auto& a_stride = a->stride();
  const auto& b_stride = b->stride();
  const auto& out_stride = out->stride();

  if (shape.empty()) {
    out_data[static_cast<size_t>(out_base)] =
        op(a_data[static_cast<size_t>(a_base)],
           b_data[static_cast<size_t>(b_base)]);
    return;
  }

  std::vector<uint64_t> index(shape.size(), 0);
  while (true) {
    int64_t a_offset = a_base;
    int64_t b_offset = b_base;
    int64_t out_offset = out_base;

    for (size_t dim = 0; dim < shape.size(); ++dim) {
      const int64_t i = static_cast<int64_t>(index[dim]);
      a_offset += i * a_stride[dim];
      b_offset += i * b_stride[dim];
      out_offset += i * out_stride[dim];
    }

    out_data[static_cast<size_t>(out_offset)] =
        op(a_data[static_cast<size_t>(a_offset)],
           b_data[static_cast<size_t>(b_offset)]);

    size_t dim = shape.size();
    while (dim > 0) {
      --dim;
      index[dim] += 1;
      if (index[dim] < shape[dim]) {
        break;
      }
      index[dim] = 0;
      if (dim == 0) {
        return;
      }
    }
  }
}

int ToCblasInt(uint64_t value, const char* value_name) {
  if (value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    std::stringstream err;
    err << "BatchedMatMul does not support " << value_name
        << " larger than INT_MAX.";
    throw std::runtime_error(err.str());
  }
  return static_cast<int>(value);
}

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
    throw std::runtime_error(
        "Only Float32 dtype is supported in BatchedMatMul.");
  }

  if (a->device() != b->device() || a->device() != out->device()) {
    throw std::runtime_error("BatchedMatMul requires matching tensor devices.");
  }
  if (a->device() != Device::CPU) {
    throw std::runtime_error("BatchedMatMul only supports CPU.");
  }

  const auto& a_shape = a->shape();
  const auto& b_shape = b->shape();
  const auto& out_shape = out->shape();
  const size_t rank = a_shape.size();
  if (rank < 2) {
    throw std::runtime_error("BatchedMatMul requires rank >= 2 tensors.");
  }
  if (b_shape.size() != rank || out_shape.size() != rank) {
    throw std::runtime_error("BatchedMatMul requires matching tensor ranks.");
  }

  for (size_t dim = 0; dim + 2 < rank; ++dim) {
    if (a_shape[dim] != b_shape[dim] || a_shape[dim] != out_shape[dim]) {
      throw std::runtime_error(
          "BatchedMatMul requires matching leading batch dimensions.");
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

  if (a_cols != b_rows) {
    throw std::runtime_error(
        "BatchedMatMul requires matching inner dimensions.");
  }
  if (out_shape[rank - 2] != a_rows || out_shape[rank - 1] != b_cols) {
    throw std::runtime_error("BatchedMatMul output shape mismatch.");
  }

  if (!out->isContiguous() || out->offset() != 0 ||
      out->storage()->numel() != utils::GetTotalSize(out->shape())) {
    throw std::runtime_error(
        "BatchedMatMul requires contiguous output tensor storage.");
  }
}

void ApplyBatchedMatMul(const std::shared_ptr<TensorImpl>& a,
                        const std::shared_ptr<TensorImpl>& b,
                        const std::shared_ptr<TensorImpl>& out,
                        bool transpose_a, bool transpose_b) {
  ValidateBatchedMatMulInputs(a, b, out, transpose_a, transpose_b);

  const auto& out_shape = out->shape();
  if (utils::GetTotalSize(out_shape) == 0) {
    return;
  }

  auto a_storage = a->getContiguousStorage();
  auto b_storage = b->getContiguousStorage();
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

  const int cblas_rows = ToCblasInt(rows, "rows");
  const int cblas_cols = ToCblasInt(cols, "cols");
  const int cblas_inner = ToCblasInt(inner, "inner dimension");
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
    cblas_sgemm(CblasRowMajor, cblas_transpose_a, cblas_transpose_b, cblas_rows,
                cblas_cols, cblas_inner, 1.0f, a_batch, lda, b_batch, ldb, 0.0f,
                out_batch, ldc);
  }
}
}  // namespace

std::shared_ptr<TensorImpl> FromBuffer(DType dtype,
                                       std::span<const std::byte> buffer,
                                       Shape shape) {
  uint64_t total_size = utils::GetTotalSize(shape);
  std::shared_ptr<TensorImpl> result;
  switch (dtype) {
    case DType::Float32:
      if (buffer.size() != total_size * 4) {
        std::stringstream err;
        err << "Failed to create tensor with shape " << FormatShape(shape)
            << " with dtype float32 on CPU. Expected " << total_size
            << " bytes in buffer but only found " << buffer.size();
        throw std::runtime_error(err.str());
      }
      result = std::make_shared<TensorImpl>(shape, DType::Float32, Device::CPU);
      result->storage()->CopyFromHost(0, total_size, buffer.data());
      break;

    default:
      throw std::runtime_error("DType unsupported");
      break;
  }
  return result;
}

void Add(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b,
         std::shared_ptr<TensorImpl> out) {
  ApplyElementwiseBinaryOp(a, b, out, "Add",
                           [](float x, float y) { return x + y; });
}

void Sub(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b,
         std::shared_ptr<TensorImpl> out) {
  ApplyElementwiseBinaryOp(a, b, out, "Sub",
                           [](float x, float y) { return x - y; });
}

void Mul(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b,
         std::shared_ptr<TensorImpl> out) {
  ApplyElementwiseBinaryOp(a, b, out, "Mul",
                           [](float x, float y) { return x * y; });
}

void Div(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b,
         std::shared_ptr<TensorImpl> out) {
  ApplyElementwiseBinaryOp(a, b, out, "Div",
                           [](float x, float y) { return x / y; });
}

void BatchedMatMul(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b,
                   std::shared_ptr<TensorImpl> out, bool transpose_a,
                   bool transpose_b) {
  ApplyBatchedMatMul(a, b, out, transpose_a, transpose_b);
}

};  // namespace cpu

};  // namespace deeptiny
