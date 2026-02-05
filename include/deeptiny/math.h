#pragma once
#include <string>

#include "deeptiny/tensor.h"

namespace deeptiny {
namespace math {

Tensor operator+(const Tensor& a, const Tensor& b);

};  // namespace math
};  // namespace deeptiny
