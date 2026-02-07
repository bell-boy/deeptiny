#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "tensor_impl.h"

namespace deeptiny::cpu::detail {

inline int ToCblasInt(uint64_t value, const char* value_name) {
  if (value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    std::stringstream err;
    err << "BatchedMatMul does not support " << value_name
        << " larger than INT_MAX.";
    throw std::runtime_error(err.str());
  }
  return static_cast<int>(value);
}


inline void ValidateElementwiseBinaryOpInputs(
    const std::shared_ptr<TensorImpl>& a, const std::shared_ptr<TensorImpl>& b,
    const std::shared_ptr<TensorImpl>& out) {
  if (!a || !b || !out) {
    throw std::runtime_error("Elementwise op received null TensorImpl");
  }
  assert(a->shape() == b->shape());
  assert(a->shape() == out->shape());
  assert(a->dtype() == b->dtype());
  assert(a->dtype() == out->dtype());
}

inline void ValidateElementwiseUnaryOpInputs(
    const std::shared_ptr<TensorImpl>& x,
    const std::shared_ptr<TensorImpl>& out) {
  if (!x || !out) {
    throw std::runtime_error("Elementwise op received null TensorImpl");
  }
  assert(x->shape() == out->shape());
  assert(x->dtype() == out->dtype());
}

template <typename Op>
inline void ApplyElementwiseBinaryOp(const std::shared_ptr<TensorImpl>& a,
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

template <typename Op>
inline void ApplyElementwiseUnaryOp(const std::shared_ptr<TensorImpl>& x,
                                    const std::shared_ptr<TensorImpl>& out,
                                    const char* op_name, Op&& op) {
  ValidateElementwiseUnaryOpInputs(x, out);
  if (x->dtype() != DType::Float32) {
    std::stringstream err;
    err << "Only Float32 dtype is supported in " << op_name;
    throw std::runtime_error(err.str());
  }

  const auto x_storage = static_cast<const TensorImpl&>(*x).storage();
  auto out_storage = out->storage();

  const auto* x_data = static_cast<const float*>(x_storage->data(0));
  auto* out_data = static_cast<float*>(out_storage->data(0));

  const int64_t x_base = static_cast<int64_t>(x->offset());
  const int64_t out_base = static_cast<int64_t>(out->offset());

  const auto& shape = out->shape();
  const auto& x_stride = x->stride();
  const auto& out_stride = out->stride();

  if (shape.empty()) {
    out_data[static_cast<size_t>(out_base)] =
        op(x_data[static_cast<size_t>(x_base)]);
    return;
  }

  std::vector<uint64_t> index(shape.size(), 0);
  while (true) {
    int64_t x_offset = x_base;
    int64_t out_offset = out_base;

    for (size_t dim = 0; dim < shape.size(); ++dim) {
      const int64_t i = static_cast<int64_t>(index[dim]);
      x_offset += i * x_stride[dim];
      out_offset += i * out_stride[dim];
    }

    out_data[static_cast<size_t>(out_offset)] =
        op(x_data[static_cast<size_t>(x_offset)]);

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

inline void ValidateElementwiseUnaryGradOpInputs(
    const std::shared_ptr<TensorImpl>& x,
    const std::shared_ptr<TensorImpl>& grad_out,
    const std::shared_ptr<TensorImpl>& grad_x) {
  if (!x || !grad_out || !grad_x) {
    throw std::runtime_error("Elementwise backward op received null TensorImpl");
  }
  assert(x->shape() == grad_out->shape());
  assert(x->shape() == grad_x->shape());
  assert(x->dtype() == grad_out->dtype());
  assert(x->dtype() == grad_x->dtype());
}

template <typename Op>
inline void ApplyElementwiseUnaryGradOp(
    const std::shared_ptr<TensorImpl>& x,
    const std::shared_ptr<TensorImpl>& grad_out,
    const std::shared_ptr<TensorImpl>& grad_x, const char* op_name, Op&& op) {
  ValidateElementwiseUnaryGradOpInputs(x, grad_out, grad_x);
  if (x->dtype() != DType::Float32) {
    std::stringstream err;
    err << "Only Float32 dtype is supported in " << op_name;
    throw std::runtime_error(err.str());
  }

  const auto x_storage = static_cast<const TensorImpl&>(*x).storage();
  const auto grad_out_storage =
      static_cast<const TensorImpl&>(*grad_out).storage();
  auto grad_x_storage = grad_x->storage();

  const auto* x_data = static_cast<const float*>(x_storage->data(0));
  const auto* grad_out_data =
      static_cast<const float*>(grad_out_storage->data(0));
  auto* grad_x_data = static_cast<float*>(grad_x_storage->data(0));

  const int64_t x_base = static_cast<int64_t>(x->offset());
  const int64_t grad_out_base = static_cast<int64_t>(grad_out->offset());
  const int64_t grad_x_base = static_cast<int64_t>(grad_x->offset());

  const auto& shape = grad_x->shape();
  const auto& x_stride = x->stride();
  const auto& grad_out_stride = grad_out->stride();
  const auto& grad_x_stride = grad_x->stride();

  if (shape.empty()) {
    grad_x_data[static_cast<size_t>(grad_x_base)] =
        op(x_data[static_cast<size_t>(x_base)],
           grad_out_data[static_cast<size_t>(grad_out_base)]);
    return;
  }

  std::vector<uint64_t> index(shape.size(), 0);
  while (true) {
    int64_t x_offset = x_base;
    int64_t grad_out_offset = grad_out_base;
    int64_t grad_x_offset = grad_x_base;

    for (size_t dim = 0; dim < shape.size(); ++dim) {
      const int64_t i = static_cast<int64_t>(index[dim]);
      x_offset += i * x_stride[dim];
      grad_out_offset += i * grad_out_stride[dim];
      grad_x_offset += i * grad_x_stride[dim];
    }

    grad_x_data[static_cast<size_t>(grad_x_offset)] =
        op(x_data[static_cast<size_t>(x_offset)],
           grad_out_data[static_cast<size_t>(grad_out_offset)]);

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

}  // namespace deeptiny::cpu::detail
