#pragma once
#include <string>

#include "deeptiny/tensor.h"

namespace deeptiny {
namespace math {

Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);
Tensor BatchedMatMul(const Tensor& a, const Tensor& b);

};  // namespace math

using math::operator+;
using math::operator-;
using math::operator*;
using math::operator/;
using math::BatchedMatMul;
};  // namespace deeptiny
