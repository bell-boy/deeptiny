#include "utils.h"

#include <stdexcept>
#include <vector>

#include "test_utils.h"

namespace deeptiny::test_utils {

Tensor MakeTensor(const Shape& shape, const std::vector<float>& values,
                  bool requires_grad) {
  if (values.size() != utils::GetTotalSize(shape)) {
    throw std::runtime_error(
        "MakeTensor received mismatched values/shape size");
  }
  return Tensor::FromVector(values, shape, Device::CPU, requires_grad);
}

std::vector<float> ToVector(const Tensor& t) {
  auto impl = utils::TensorAccessor::GetTensorImpl(t);
  auto contiguous = impl->getContiguousStorage();
  const auto n = contiguous->numel();
  std::vector<float> out(static_cast<size_t>(n), 0.0f);
  contiguous->CopyToHost(0, n, out.data());
  return out;
}

void CheckTensorData(const Tensor& t, const std::vector<float>& expected) {
  const auto actual = ToVector(t);
  REQUIRE(actual.size() == expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    CHECK(actual[i] == Approx(expected[i]));
  }
}

}  // namespace deeptiny::test_utils
