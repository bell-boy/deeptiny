#pragma once
#include <cstdint>
#include <optional>
#include <vector>

namespace deeptiny {
using Shape = std::vector<uint64_t>;
using Stride = std::vector<int64_t>;

class DType {
 public:
  enum Enum { Float32, BFloat16 };
  constexpr DType(Enum e) : e_(e) {}
  constexpr operator Enum() const { return e_; }

  /**
   * Size in bytes of the data type
   */
  uint64_t size() {
    switch (e_) {
      case Float32:
        return 4;
      case BFloat16:
        return 2;
    }
  }

  std::string ToString() const {
    switch (e_) {
      case Float32:
        return "Float32";
      case BFloat16:
        return "Bfloat16";
    }
  }

 private:
  Enum e_;
};

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
