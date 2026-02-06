#include "cpu/kernels.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
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

};  // namespace cpu

};  // namespace deeptiny
