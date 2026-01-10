#pragma once
#include "deeptiny/tensorImpl.h"

namespace deeptiny {

namespace cpu {

class CPUStorage : public detail::Storage {
  void* buffer = nullptr;

 public:
  CPUStorage(uint64_t numel, DType dtype);

  void* get_(uint64_t offset) const override;

  friend detail::Storage;
};

};  // namespace cpu

};  // namespace deeptiny