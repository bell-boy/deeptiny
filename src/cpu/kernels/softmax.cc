#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "cpu/kernels.h"

namespace deeptiny::cpu {
namespace {

void ValidateSoftmaxInputs(const std::shared_ptr<TensorImpl>& x,
                           const std::shared_ptr<TensorImpl>& out,
                           uint64_t dim) {
  if (!x || !out) {
    throw std::runtime_error("Softmax received null TensorImpl.");
  }
  if (x->shape() != out->shape()) {
    throw std::runtime_error("Softmax expects matching input/output shapes.");
  }
  if (x->dtype() != out->dtype()) {
    throw std::runtime_error("Softmax expects matching input/output dtypes.");
  }
  if (x->shape().empty()) {
    throw std::runtime_error("Softmax requires rank >= 1 tensor.");
  }
  if (dim >= x->shape().size()) {
    throw std::runtime_error("Softmax dim out of range.");
  }
  if (x->dtype() != DType::Float32) {
    throw std::runtime_error("Only Float32 dtype is supported in Softmax.");
  }
}

void ValidateSoftmaxBackwardInputs(const std::shared_ptr<TensorImpl>& y,
                                   const std::shared_ptr<TensorImpl>& grad_out,
                                   const std::shared_ptr<TensorImpl>& grad_x,
                                   uint64_t dim) {
  if (!y || !grad_out || !grad_x) {
    throw std::runtime_error("SoftmaxBackward received null TensorImpl.");
  }
  if (y->shape() != grad_out->shape() || y->shape() != grad_x->shape()) {
    throw std::runtime_error(
        "SoftmaxBackward expects matching y/grad_out/grad_x shapes.");
  }
  if (y->dtype() != grad_out->dtype() || y->dtype() != grad_x->dtype()) {
    throw std::runtime_error(
        "SoftmaxBackward expects matching y/grad_out/grad_x dtypes.");
  }
  if (y->shape().empty()) {
    throw std::runtime_error("SoftmaxBackward requires rank >= 1 tensor.");
  }
  if (dim >= y->shape().size()) {
    throw std::runtime_error("SoftmaxBackward dim out of range.");
  }
  if (y->dtype() != DType::Float32) {
    throw std::runtime_error(
        "Only Float32 dtype is supported in SoftmaxBackward.");
  }
}

bool HasZeroDim(const Shape& shape) {
  for (const uint64_t dim_size : shape) {
    if (dim_size == 0) {
      return true;
    }
  }
  return false;
}

int64_t ComputeBaseOffset(const std::vector<uint64_t>& outer_dims,
                          const std::vector<uint64_t>& outer_index,
                          const Stride& stride, int64_t base) {
  int64_t offset = base;
  for (size_t i = 0; i < outer_dims.size(); ++i) {
    offset += static_cast<int64_t>(outer_index[i]) * stride[outer_dims[i]];
  }
  return offset;
}

template <typename RowFn>
void ForEachOuterRow(const Shape& shape, uint64_t dim, RowFn&& row_fn) {
  std::vector<uint64_t> outer_dims;
  outer_dims.reserve(shape.size() - 1);
  for (uint64_t d = 0; d < shape.size(); ++d) {
    if (d != dim) {
      outer_dims.push_back(d);
    }
  }

  if (outer_dims.empty()) {
    const std::vector<uint64_t> empty_index;
    row_fn(outer_dims, empty_index);
    return;
  }

  std::vector<uint64_t> outer_index(outer_dims.size(), 0);
  while (true) {
    row_fn(outer_dims, outer_index);

    size_t i = outer_index.size();
    while (i > 0) {
      --i;
      outer_index[i] += 1;
      if (outer_index[i] < shape[outer_dims[i]]) {
        break;
      }
      outer_index[i] = 0;
      if (i == 0) {
        return;
      }
    }
  }
}

}  // namespace

void Softmax(std::shared_ptr<TensorImpl> x, std::shared_ptr<TensorImpl> out,
             uint64_t dim) {
  ValidateSoftmaxInputs(x, out, dim);
  const auto& shape = x->shape();
  if (HasZeroDim(shape)) {
    return;
  }

  const auto x_storage = static_cast<const TensorImpl&>(*x).storage();
  auto out_storage = out->storage();
  const auto* x_data = static_cast<const float*>(x_storage->data(0));
  auto* out_data = static_cast<float*>(out_storage->data(0));

  const auto& x_stride = x->stride();
  const auto& out_stride = out->stride();
  const int64_t x_base = static_cast<int64_t>(x->offset());
  const int64_t out_base = static_cast<int64_t>(out->offset());
  const int64_t x_dim_stride = x_stride[dim];
  const int64_t out_dim_stride = out_stride[dim];
  const uint64_t dim_size = shape[dim];

  ForEachOuterRow(
      shape, dim,
      [&](const std::vector<uint64_t>& outer_dims,
          const std::vector<uint64_t>& outer_index) {
        const int64_t x_row_base =
            ComputeBaseOffset(outer_dims, outer_index, x_stride, x_base);
        const int64_t out_row_base =
            ComputeBaseOffset(outer_dims, outer_index, out_stride, out_base);

        float max_value = -std::numeric_limits<float>::infinity();
        for (uint64_t i = 0; i < dim_size; ++i) {
          const int64_t x_offset =
              x_row_base + static_cast<int64_t>(i) * x_dim_stride;
          const float value = x_data[static_cast<size_t>(x_offset)];
          if (value > max_value) {
            max_value = value;
          }
        }

        float sum_exp = 0.0f;
        for (uint64_t i = 0; i < dim_size; ++i) {
          const int64_t x_offset =
              x_row_base + static_cast<int64_t>(i) * x_dim_stride;
          const int64_t out_offset =
              out_row_base + static_cast<int64_t>(i) * out_dim_stride;
          const float exp_value =
              std::exp(x_data[static_cast<size_t>(x_offset)] - max_value);
          out_data[static_cast<size_t>(out_offset)] = exp_value;
          sum_exp += exp_value;
        }

        for (uint64_t i = 0; i < dim_size; ++i) {
          const int64_t out_offset =
              out_row_base + static_cast<int64_t>(i) * out_dim_stride;
          out_data[static_cast<size_t>(out_offset)] /= sum_exp;
        }
      });
}

void SoftmaxBackward(std::shared_ptr<TensorImpl> y,
                     std::shared_ptr<TensorImpl> grad_out,
                     std::shared_ptr<TensorImpl> grad_x, uint64_t dim) {
  ValidateSoftmaxBackwardInputs(y, grad_out, grad_x, dim);
  const auto& shape = y->shape();
  if (HasZeroDim(shape)) {
    return;
  }

  const auto y_storage = static_cast<const TensorImpl&>(*y).storage();
  const auto grad_out_storage =
      static_cast<const TensorImpl&>(*grad_out).storage();
  auto grad_x_storage = grad_x->storage();

  const auto* y_data = static_cast<const float*>(y_storage->data(0));
  const auto* grad_out_data =
      static_cast<const float*>(grad_out_storage->data(0));
  auto* grad_x_data = static_cast<float*>(grad_x_storage->data(0));

  const auto& y_stride = y->stride();
  const auto& grad_out_stride = grad_out->stride();
  const auto& grad_x_stride = grad_x->stride();

  const int64_t y_base = static_cast<int64_t>(y->offset());
  const int64_t grad_out_base = static_cast<int64_t>(grad_out->offset());
  const int64_t grad_x_base = static_cast<int64_t>(grad_x->offset());

  const int64_t y_dim_stride = y_stride[dim];
  const int64_t grad_out_dim_stride = grad_out_stride[dim];
  const int64_t grad_x_dim_stride = grad_x_stride[dim];
  const uint64_t dim_size = shape[dim];

  ForEachOuterRow(
      shape, dim,
      [&](const std::vector<uint64_t>& outer_dims,
          const std::vector<uint64_t>& outer_index) {
        const int64_t y_row_base =
            ComputeBaseOffset(outer_dims, outer_index, y_stride, y_base);
        const int64_t grad_out_row_base = ComputeBaseOffset(
            outer_dims, outer_index, grad_out_stride, grad_out_base);
        const int64_t grad_x_row_base = ComputeBaseOffset(
            outer_dims, outer_index, grad_x_stride, grad_x_base);

        float dot = 0.0f;
        for (uint64_t i = 0; i < dim_size; ++i) {
          const int64_t y_offset =
              y_row_base + static_cast<int64_t>(i) * y_dim_stride;
          const int64_t grad_out_offset =
              grad_out_row_base + static_cast<int64_t>(i) * grad_out_dim_stride;
          dot += y_data[static_cast<size_t>(y_offset)] *
                 grad_out_data[static_cast<size_t>(grad_out_offset)];
        }

        for (uint64_t i = 0; i < dim_size; ++i) {
          const int64_t y_offset =
              y_row_base + static_cast<int64_t>(i) * y_dim_stride;
          const int64_t grad_out_offset =
              grad_out_row_base + static_cast<int64_t>(i) * grad_out_dim_stride;
          const int64_t grad_x_offset =
              grad_x_row_base + static_cast<int64_t>(i) * grad_x_dim_stride;
          const float y_value = y_data[static_cast<size_t>(y_offset)];
          const float grad_out_value =
              grad_out_data[static_cast<size_t>(grad_out_offset)];
          grad_x_data[static_cast<size_t>(grad_x_offset)] =
              y_value * (grad_out_value - dot);
        }
      });
}

}  // namespace deeptiny::cpu
