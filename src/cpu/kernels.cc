#include "cpu/kernels.h"

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

};  // namespace cpu

};  // namespace deeptiny
