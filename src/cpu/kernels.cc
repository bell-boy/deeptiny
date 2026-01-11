#include "cpu/kernels.h"

#include <span>
#include <sstream>

#include "utils.h"

namespace deeptiny {

namespace cpu {

std::shared_ptr<TensorImpl> FromBuffer(DType dtype, std::span<std::byte> buffer,
                                       Shape shape) {
  uint64_t total_size = utils::GetTotalSize(shape);
  std::shared_ptr<Storage> result;
  switch (dtype) {
    case DType::Float32:
      if (buffer.size() != total_size * 4) {
        std::stringstream err;
        err << "Failed to create tensor with shape { ";
        for (const auto& dim : shape) {
          err << dim << ", ";
        }
        err << "} with dtype float32 on CPU. Expected " << total_size
            << " bytes in buffer but only found " << buffer.size();
        std::runtime_error(err.str());
      }
      result = Storage::MakeStorage(total_size, DType::Float32, Device::CPU);
      memcpy((void*)buffer.data(), result->data(0), total_size * 4);
      break;

    default:
      std::runtime_error("DType unsupported");
      break;
  }
  return std::make_shared<TensorImpl>(shape, utils::GetContinguousStride(shape),
                                      0, result);
}

};  // namespace cpu

};  // namespace deeptiny
