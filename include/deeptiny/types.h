#pragma once
#include <cstdint>
#include <optional>
#include <vector>

namespace deeptiny {
using Shape = std::vector<uint64_t>;
using Stride = std::vector<int64_t>;

enum DType { Float32, BFloat16 };

enum Device { CPU, Metal };

struct Slice {
  using Index = std::optional<int64_t>;
  Index start;
  Index end;
  Index stride;

  Slice(int64_t x) : start(x), end(x + 1), stride(1) {}
  Slice(int64_t start, int64_t end) : start(start), end(end), stride(1) {}
  Slice(int64_t start, int64_t end, int64_t stride)
      : start(start), end(end), stride(stride) {}
};
};  // namespace deeptiny
