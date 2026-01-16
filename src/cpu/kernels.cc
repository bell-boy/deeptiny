#include "cpu/kernels.h"

#include <cblas.h>

#include <cassert>
#include <span>
#include <sstream>

#include "utils.h"

namespace deeptiny {

namespace cpu {

std::shared_ptr<TensorImpl> FromBuffer(DType dtype, std::span<std::byte> buffer,
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
  // TODO: fast path for contiguous tensors
  // TODO: fast path for when a XOR b is equal to out
  // TODO: fast path for when a AND b is equal to out
  assert(a->shape() == b->shape());
  assert(a->shape() == out->shape());
  assert(a->dtype() == b->dtype());
  assert(a->dtype() == out->dtype());

  if (a->dtype() != DType::Float32) {
    throw std::runtime_error("Only Float32 dtype is supported in Add");
  }

  auto a_buffer = a->getContiguousStorage();
  auto b_buffer = b->getContiguousStorage();

  cblas_saxpy(a_buffer->numel(), 1.0, (float*)a_buffer->data(0), 1,
              (float*)b_buffer->data(0), 1);

  out->storage()->CopyFromHost(0, b_buffer->numel(), b_buffer->data(0));
}

};  // namespace cpu

};  // namespace deeptiny
