#include "cpu/storage.h"

#include <cstring>

namespace deeptiny {

namespace cpu {

CPUStorage::CPUStorage(uint64_t numel, DType dtype)
    : Storage(numel, dtype, Device::CPU) {
  switch (dtype_) {
    case DType::Float32:
      buffer = (void*)(new float[numel]);
      break;
    default:
      throw std::runtime_error("Couldn't handle dtype on CPU");
      break;
  }
}

void* CPUStorage::get_(uint64_t offset) const {
  return (void*)((dtype_.size() * offset) + (uintptr_t)buffer);
}

void CPUStorage::CopyFromHost_(uint64_t offset, uint64_t numel,
                               const void* buffer) {
  memcpy(get_(offset), buffer, numel * dtype_.size());
}

void CPUStorage::CopyToHost(uint64_t offset, uint64_t numel,
                            void* buffer) const {
  memcpy(buffer, get_(offset), numel * dtype_.size());
}

};  // namespace cpu

};  // namespace deeptiny
