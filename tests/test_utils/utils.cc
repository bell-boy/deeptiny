#include "utils.h"

#include <cstddef>
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

void CopyTensorData(const Tensor& src, const Tensor& dst) {
  utils::CompatabilityCheck({src, dst});
  if (src.shape() != dst.shape()) {
    throw std::runtime_error("CopyTensorData shape mismatch");
  }

  auto src_impl = utils::TensorAccessor::GetTensorImpl(src);
  auto dst_impl = utils::TensorAccessor::GetTensorImpl(dst);
  auto src_storage = src_impl->getContiguousStorage();
  const uint64_t numel = src_storage->numel();
  std::vector<std::byte> host_buffer(
      static_cast<size_t>(numel * src.dtype().size()));
  src_storage->CopyToHost(0, numel, host_buffer.data());
  dst_impl->storage()->CopyFromHost(0, numel, host_buffer.data());
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
