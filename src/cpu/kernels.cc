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
enum class MatMulVariant { Forward, GradA, GradB };

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

struct MatMulSpec {
  bool transpose_a;
  bool transpose_b;
  const char* op_name;
};

MatMulSpec GetMatMulSpec(MatMulVariant variant) {
  switch (variant) {
    case MatMulVariant::Forward:
      return MatMulSpec{false, false, "BatchedMatMul"};
    case MatMulVariant::GradA:
      return MatMulSpec{false, true, "BatchedMatMulGradA"};
    case MatMulVariant::GradB:
      return MatMulSpec{true, false, "BatchedMatMulGradB"};
  }
  throw std::runtime_error("Unknown batched matmul variant");
}

int ToCblasInt(uint64_t value, const char* value_name, const char* op_name) {
  if (value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    std::stringstream err;
    err << op_name << " does not support " << value_name
        << " larger than INT_MAX";
    throw std::runtime_error(err.str());
  }
  return static_cast<int>(value);
}

void PackMatrix(const float* src_data, int64_t base_offset,
                const Stride& src_stride, uint64_t src_rows, uint64_t src_cols,
                bool transpose, std::vector<float>& packed) {
  const uint64_t rows = transpose ? src_cols : src_rows;
  const uint64_t cols = transpose ? src_rows : src_cols;
  packed.resize(static_cast<size_t>(rows * cols));

  const int64_t stride_row = src_stride[src_stride.size() - 2];
  const int64_t stride_col = src_stride[src_stride.size() - 1];
  for (uint64_t row = 0; row < rows; ++row) {
    for (uint64_t col = 0; col < cols; ++col) {
      const uint64_t src_row = transpose ? col : row;
      const uint64_t src_col = transpose ? row : col;
      const int64_t src_offset = base_offset +
                                 static_cast<int64_t>(src_row) * stride_row +
                                 static_cast<int64_t>(src_col) * stride_col;
      packed[static_cast<size_t>(row * cols + col)] =
          src_data[static_cast<size_t>(src_offset)];
    }
  }
}

void ScatterMatrix(const std::vector<float>& packed, float* dst_data,
                   int64_t base_offset, const Stride& dst_stride, uint64_t rows,
                   uint64_t cols) {
  const int64_t stride_row = dst_stride[dst_stride.size() - 2];
  const int64_t stride_col = dst_stride[dst_stride.size() - 1];
  for (uint64_t row = 0; row < rows; ++row) {
    for (uint64_t col = 0; col < cols; ++col) {
      const int64_t dst_offset = base_offset +
                                 static_cast<int64_t>(row) * stride_row +
                                 static_cast<int64_t>(col) * stride_col;
      dst_data[static_cast<size_t>(dst_offset)] =
          packed[static_cast<size_t>(row * cols + col)];
    }
  }
}

void ValidateBatchedMatMulInputs(const std::shared_ptr<TensorImpl>& a,
                                 const std::shared_ptr<TensorImpl>& b,
                                 const std::shared_ptr<TensorImpl>& out,
                                 const MatMulSpec& spec) {
  if (!a || !b || !out) {
    std::stringstream err;
    err << spec.op_name << " received null TensorImpl";
    throw std::runtime_error(err.str());
  }

  if (a->dtype() != b->dtype() || a->dtype() != out->dtype()) {
    std::stringstream err;
    err << spec.op_name << " requires matching tensor dtypes";
    throw std::runtime_error(err.str());
  }
  if (a->dtype() != DType::Float32) {
    std::stringstream err;
    err << "Only Float32 dtype is supported in " << spec.op_name;
    throw std::runtime_error(err.str());
  }

  if (a->device() != b->device() || a->device() != out->device()) {
    std::stringstream err;
    err << spec.op_name << " requires matching tensor devices";
    throw std::runtime_error(err.str());
  }
  if (a->device() != Device::CPU) {
    std::stringstream err;
    err << spec.op_name << " only supports CPU";
    throw std::runtime_error(err.str());
  }

  const auto& a_shape = a->shape();
  const auto& b_shape = b->shape();
  const auto& out_shape = out->shape();
  const size_t rank = a_shape.size();
  if (rank < 2) {
    std::stringstream err;
    err << spec.op_name << " requires rank >= 2 tensors";
    throw std::runtime_error(err.str());
  }
  if (b_shape.size() != rank || out_shape.size() != rank) {
    std::stringstream err;
    err << spec.op_name << " requires matching tensor ranks";
    throw std::runtime_error(err.str());
  }

  for (size_t dim = 0; dim + 2 < rank; ++dim) {
    if (a_shape[dim] != b_shape[dim] || a_shape[dim] != out_shape[dim]) {
      std::stringstream err;
      err << spec.op_name << " requires matching leading batch dimensions";
      throw std::runtime_error(err.str());
    }
  }

  const uint64_t a_src_rows = a_shape[rank - 2];
  const uint64_t a_src_cols = a_shape[rank - 1];
  const uint64_t b_src_rows = b_shape[rank - 2];
  const uint64_t b_src_cols = b_shape[rank - 1];

  const uint64_t a_rows = spec.transpose_a ? a_src_cols : a_src_rows;
  const uint64_t a_cols = spec.transpose_a ? a_src_rows : a_src_cols;
  const uint64_t b_rows = spec.transpose_b ? b_src_cols : b_src_rows;
  const uint64_t b_cols = spec.transpose_b ? b_src_rows : b_src_cols;

  if (a_cols != b_rows) {
    std::stringstream err;
    err << spec.op_name << " requires matching inner dimensions";
    throw std::runtime_error(err.str());
  }
  if (out_shape[rank - 2] != a_rows || out_shape[rank - 1] != b_cols) {
    std::stringstream err;
    err << spec.op_name << " output shape mismatch";
    throw std::runtime_error(err.str());
  }
}

void ApplyBatchedMatMul(const std::shared_ptr<TensorImpl>& a,
                        const std::shared_ptr<TensorImpl>& b,
                        const std::shared_ptr<TensorImpl>& out,
                        MatMulVariant variant) {
  const MatMulSpec spec = GetMatMulSpec(variant);
  ValidateBatchedMatMulInputs(a, b, out, spec);

  const auto& out_shape = out->shape();
  if (utils::GetTotalSize(out_shape) == 0) {
    return;
  }

  const auto a_storage = static_cast<const TensorImpl&>(*a).storage();
  const auto b_storage = static_cast<const TensorImpl&>(*b).storage();
  auto out_storage = out->storage();

  const auto* a_data = static_cast<const float*>(a_storage->data(0));
  const auto* b_data = static_cast<const float*>(b_storage->data(0));
  auto* out_data = static_cast<float*>(out_storage->data(0));

  const auto& a_shape = a->shape();
  const auto& b_shape = b->shape();
  const auto& a_stride = a->stride();
  const auto& b_stride = b->stride();
  const auto& out_stride = out->stride();
  const size_t rank = out_shape.size();
  const size_t batch_rank = rank - 2;

  const uint64_t a_src_rows = a_shape[rank - 2];
  const uint64_t a_src_cols = a_shape[rank - 1];
  const uint64_t b_src_rows = b_shape[rank - 2];
  const uint64_t b_src_cols = b_shape[rank - 1];

  const uint64_t rows = spec.transpose_a ? a_src_cols : a_src_rows;
  const uint64_t inner = spec.transpose_a ? a_src_rows : a_src_cols;
  const uint64_t cols = spec.transpose_b ? b_src_rows : b_src_cols;

  const int cblas_rows = ToCblasInt(rows, "rows", spec.op_name);
  const int cblas_cols = ToCblasInt(cols, "cols", spec.op_name);
  const int cblas_inner = ToCblasInt(inner, "inner dimension", spec.op_name);

  std::vector<float> packed_a;
  std::vector<float> packed_b;
  std::vector<float> packed_out(static_cast<size_t>(rows * cols), 0.0f);

  auto run_batch = [&](const std::vector<uint64_t>& batch_index) {
    int64_t a_base = static_cast<int64_t>(a->offset());
    int64_t b_base = static_cast<int64_t>(b->offset());
    int64_t out_base = static_cast<int64_t>(out->offset());
    for (size_t dim = 0; dim < batch_rank; ++dim) {
      const int64_t idx = static_cast<int64_t>(batch_index[dim]);
      a_base += idx * a_stride[dim];
      b_base += idx * b_stride[dim];
      out_base += idx * out_stride[dim];
    }

    PackMatrix(a_data, a_base, a_stride, a_src_rows, a_src_cols,
               spec.transpose_a, packed_a);
    PackMatrix(b_data, b_base, b_stride, b_src_rows, b_src_cols,
               spec.transpose_b, packed_b);
    packed_out.assign(static_cast<size_t>(rows * cols), 0.0f);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, cblas_rows,
                cblas_cols, cblas_inner, 1.0f, packed_a.data(), cblas_inner,
                packed_b.data(), cblas_cols, 0.0f, packed_out.data(),
                cblas_cols);
    ScatterMatrix(packed_out, out_data, out_base, out_stride, rows, cols);
  };

  if (batch_rank == 0) {
    run_batch({});
    return;
  }

  std::vector<uint64_t> batch_index(batch_rank, 0);
  while (true) {
    run_batch(batch_index);

    size_t dim = batch_rank;
    while (dim > 0) {
      --dim;
      batch_index[dim] += 1;
      if (batch_index[dim] < out_shape[dim]) {
        break;
      }
      batch_index[dim] = 0;
      if (dim == 0) {
        return;
      }
    }
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
                   std::shared_ptr<TensorImpl> out) {
  ApplyBatchedMatMul(a, b, out, MatMulVariant::Forward);
}

void BatchedMatMulGradA(std::shared_ptr<TensorImpl> grad_out,
                        std::shared_ptr<TensorImpl> b,
                        std::shared_ptr<TensorImpl> grad_a) {
  ApplyBatchedMatMul(grad_out, b, grad_a, MatMulVariant::GradA);
}

void BatchedMatMulGradB(std::shared_ptr<TensorImpl> a,
                        std::shared_ptr<TensorImpl> grad_out,
                        std::shared_ptr<TensorImpl> grad_b) {
  ApplyBatchedMatMul(a, grad_out, grad_b, MatMulVariant::GradB);
}

};  // namespace cpu

};  // namespace deeptiny
