#pragma once
#include "tensorImpl.h"

namespace deeptiny {

namespace cpu {

class CPUStorage : public Storage {
  void* buffer = nullptr;

 public:
  CPUStorage(uint64_t numel, DType dtype);

  void* get_(uint64_t offset) const override;

  friend Storage;
};

};  // namespace cpu

};  // namespace deeptiny
