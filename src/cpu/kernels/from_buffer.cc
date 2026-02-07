#include "cpu/kernels.h"

#include <sstream>
#include <stdexcept>

#include "utils.h"

namespace deeptiny::cpu {

std::shared_ptr<TensorImpl> FromBuffer(DType dtype,
                                       std::span<const std::byte> buffer,
                                       Shape shape) {
  uint64_t total_size = utils::GetTotalSize(shape);
  std::shared_ptr<TensorImpl> result;
  switch (dtype) {
    case DType::Float32:
      {
        const uint64_t expected_bytes = total_size * 4;
        const uint64_t actual_bytes = buffer.size();
        if (actual_bytes != expected_bytes) {
          std::stringstream err;
          err << "Failed to create tensor with shape " << FormatShape(shape)
              << " with dtype float32 on CPU. Expected " << expected_bytes
              << " bytes in buffer but found " << actual_bytes << " bytes";
          throw std::runtime_error(err.str());
        }
      }
      result = std::make_shared<TensorImpl>(shape, DType::Float32, Device::CPU);
      result->storage()->CopyFromHost(0, total_size, buffer.data());
      break;
    default:
      throw std::runtime_error("DType unsupported");
  }
  return result;
}

}  // namespace deeptiny::cpu
