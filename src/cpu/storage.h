#pragma once
#include "tensor_impl.h"

namespace deeptiny {

namespace cpu {

class CPUStorage : public Storage {
  void* buffer = nullptr;

  void* get_(uint64_t offset) const override;
  void CopyFromHost_(uint64_t offset, uint64_t numel,
                     const void* buffer) override;

 public:
  CPUStorage(uint64_t numel, DType dtype);
  void CopyToHost(uint64_t offset, uint64_t numel, void* buffer) const override;

  friend Storage;
};

};  // namespace cpu

};  // namespace deeptiny
