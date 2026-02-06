#pragma once

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
}  // namespace deeptiny::test_utils
