#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace deeptiny::nn::detail {

inline uint64_t ValidateNonZeroDimension(const char* module_name,
                                         const char* parameter_name,
                                         uint64_t value) {
  if (value == 0) {
    throw std::runtime_error(std::string(module_name) + " " + parameter_name +
                             " must be non-zero");
  }
  return value;
}

}  // namespace deeptiny::nn::detail
