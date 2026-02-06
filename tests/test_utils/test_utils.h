#pragma once

#include <vector>

#include "deeptiny/tensor.h"
#include "doctest.h"

namespace deeptiny::test_utils {
constexpr double kDefaultEpsilon = 1e-5;

inline doctest::Approx Approx(float expected,
                              double epsilon = kDefaultEpsilon) {
  return doctest::Approx(static_cast<double>(expected)).epsilon(epsilon);
}

inline doctest::Approx Approx(double expected,
                              double epsilon = kDefaultEpsilon) {
  return doctest::Approx(expected).epsilon(epsilon);
}

Tensor MakeTensor(const Shape& shape, const std::vector<float>& values,
                  bool requires_grad = false);
std::vector<float> ToVector(const Tensor& t);
void CheckTensorData(const Tensor& t, const std::vector<float>& expected);
}  // namespace deeptiny::test_utils
