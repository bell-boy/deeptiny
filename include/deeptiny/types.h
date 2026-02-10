#pragma once
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace deeptiny {
using Shape = std::vector<uint64_t>;
using Stride = std::vector<int64_t>;

// Formats a shape as "{ d0, d1, ... }".
std::string FormatShape(const Shape& shape);

class DType {
 public:
  enum Enum { Float32, BFloat16 };
  constexpr DType(Enum e) : e_(e) {}
  constexpr operator Enum() const { return e_; }

  /**
   * Size in bytes of the data type
   */
  uint64_t size() const {
    switch (e_) {
      case Float32:
        return 4;
      case BFloat16:
        return 2;
    }
    throw std::logic_error("Unknown DType enum");
  }

  std::string ToString() const {
    switch (e_) {
      case Float32:
        return "Float32";
      case BFloat16:
        return "Bfloat16";
    }
    throw std::logic_error("Unknown DType enum");
  }

 private:
  Enum e_;
};

class Device {
 public:
  enum Enum { CPU, Metal };
  constexpr Device(Enum e) : e_(e) {}
  constexpr operator Enum() const { return e_; }

  std::string ToString() const {
    switch (e_) {
      case CPU:
        return "CPU";
      case Metal:
        return "Metal";
    }
    throw std::logic_error("Unknown Device enum");
  }

 private:
  Enum e_;
};

struct Slice {
  using Index = std::optional<int64_t>;
  Index start;
  Index end;
  int64_t stride;

  Slice(int64_t x) : start(x), end(std::nullopt), stride(1) {
    if (x != -1) {
      end = x + 1;
    }
  }
  Slice(Index start, Index end) : start(start), end(end), stride(1) {}
  Slice(Index start, Index end, int64_t stride)
      : start(start), end(end), stride(stride) {}

  static Slice All() { return Slice(std::nullopt, std::nullopt); }
};
};  // namespace deeptiny
