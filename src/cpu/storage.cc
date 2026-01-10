#include "cpu/storage.h"

namespace deeptiny {

namespace cpu {

CPUStorage::CPUStorage(uint64_t numel, DType dtype)
    : Storage(numel, dtype, Device::CPU) {
  switch (dtype_) {
    case DType::Float32:
      buffer = (void*)(new float[numel]);
      break;
    default:
      std::runtime_error("Couldn't handle dtype on CPU");
      exit(-1);
      break;
  }
}

void* CPUStorage::get_(uint64_t offset) const {
  switch (dtype_) {
    case DType::Float32:
      return (void*)((sizeof(float) * offset) + (uintptr_t)buffer);
    default:
      std::runtime_error("Couldn't handle dtype on CPU");
      exit(-1);
      break;
  }
}

};  // namespace cpu

};  // namespace deeptiny
