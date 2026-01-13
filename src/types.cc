#include "deeptiny/types.h"

#include <sstream>

namespace deeptiny {

std::string FormatShape(const Shape& shape) {
  std::ostringstream oss;
  oss << "{ ";
  for (size_t i = 0; i < shape.size(); ++i) {
    oss << shape[i];
    if (i + 1 < shape.size()) {
      oss << ", ";
    }
  }
  oss << " }";
  return oss.str();
}

}  // namespace deeptiny
