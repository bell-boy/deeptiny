#include "deeptiny/tensor.h"

#include <sstream>

#include "utils.h"

namespace deeptiny {

View Tensor::operator()(std::initializer_list<Slice> slices) {
  return View(tensor_impl_->View(slices));
}
// TODO: handle requires grad
Tensor _FromBufferFloat32(std::span<float> bytes, Shape shape) {
  Tensor res{shape, DType::Float32, Device::CPU, false};
  size_t total_size = utils::GetTotalSize(shape);
  float* data = (float*)utils::TensorAccessor::GetTensorImpl(res)->data();
  for (size_t i = 0; i < total_size; ++i) {
    data[i] = bytes[i];
  }
  return res;
}

Tensor Tensor::FromBuffer(DType dtype, std::span<std::byte> bytes,
                          Shape shape) {
  switch (dtype) {
    case Float32:
      // cast buffer to float
      if (bytes.size() % sizeof(float) != 0 ||
          bytes.size() / sizeof(float) != utils::GetTotalSize(shape)) {
        std::stringstream err;
        err << "Failed to create Float32 Tensor from buffer at memory address "
            << bytes.data() << std::endl;
        if (bytes.size() % sizeof(float) != 0)
          err << ". Size of buffer is not divisible by size of float.\n";
        else
          err << ". Not enough floats in buffer.\n";
        std::runtime_error(err.str());
      }
      return _FromBufferFloat32(
          std::span<float>{(float*)bytes.data(), bytes.size() / sizeof(float)},
          shape);
    default:
      std::runtime_error("Dtype not supported");
      return _FromBufferFloat32(
          std::span<float>{(float*)bytes.data(), bytes.size() / sizeof(float)},
          shape);
  }
}

};  // namespace deeptiny
