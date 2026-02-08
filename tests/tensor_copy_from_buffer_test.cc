#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <span>
#include <vector>

#include "deeptiny/tensor.h"
#include "doctest/doctest.h"
#include "test_utils.h"

using deeptiny::test_utils::CheckTensorData;

TEST_CASE("Tensor::CopyFromBuffer copies data into existing storage") {
  auto tensor = deeptiny::Tensor::Zeros({2, 2}, deeptiny::Device::CPU,
                                        deeptiny::DType::Float32, true);

  const std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
  tensor.CopyFromBuffer(
      std::as_bytes(std::span<const float>(values.data(), values.size())),
      {2, 2}, deeptiny::DType::Float32);

  CheckTensorData(tensor, values);
}

TEST_CASE("Tensor::CopyFromBuffer validates shape") {
  auto tensor = deeptiny::Tensor::Zeros({2, 2}, deeptiny::Device::CPU,
                                        deeptiny::DType::Float32, true);

  const std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
  CHECK_THROWS_WITH(tensor.CopyFromBuffer(std::as_bytes(std::span<const float>(
                                              values.data(), values.size())),
                                          {4}, deeptiny::DType::Float32),
                    doctest::Contains("shape mismatch"));
}

TEST_CASE("Tensor::CopyFromBuffer validates dtype") {
  auto tensor = deeptiny::Tensor::Zeros({2, 2}, deeptiny::Device::CPU,
                                        deeptiny::DType::Float32, true);

  const std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
  CHECK_THROWS_WITH(tensor.CopyFromBuffer(std::as_bytes(std::span<const float>(
                                              values.data(), values.size())),
                                          {2, 2}, deeptiny::DType::BFloat16),
                    doctest::Contains("dtype mismatch"));
}

TEST_CASE("Tensor::CopyFromBuffer validates byte size") {
  auto tensor = deeptiny::Tensor::Zeros({2, 2}, deeptiny::Device::CPU,
                                        deeptiny::DType::Float32, true);

  const std::vector<float> values{1.0f, 2.0f, 3.0f};
  CHECK_THROWS_WITH(tensor.CopyFromBuffer(std::as_bytes(std::span<const float>(
                                              values.data(), values.size())),
                                          {2, 2}, deeptiny::DType::Float32),
                    doctest::Contains("byte-size mismatch"));
}

TEST_CASE("Tensor::CopyToBuffer copies tensor bytes out") {
  auto tensor = deeptiny::Tensor::Zeros({2, 2}, deeptiny::Device::CPU,
                                        deeptiny::DType::Float32, true);
  const std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
  tensor.CopyFromBuffer(
      std::as_bytes(std::span<const float>(values.data(), values.size())),
      {2, 2}, deeptiny::DType::Float32);

  std::vector<float> out(values.size(), 0.0f);
  tensor.CopyToBuffer(
      std::as_writable_bytes(std::span<float>(out.data(), out.size())), {2, 2},
      deeptiny::DType::Float32);
  CHECK(out == values);
}

TEST_CASE("Tensor::CopyToBuffer validates shape") {
  auto tensor = deeptiny::Tensor::Zeros({2, 2}, deeptiny::Device::CPU,
                                        deeptiny::DType::Float32, true);
  std::vector<float> out(4, 0.0f);
  CHECK_THROWS_WITH(tensor.CopyToBuffer(std::as_writable_bytes(std::span<float>(
                                            out.data(), out.size())),
                                        {4}, deeptiny::DType::Float32),
                    doctest::Contains("shape mismatch"));
}

TEST_CASE("Tensor::CopyToBuffer validates dtype") {
  auto tensor = deeptiny::Tensor::Zeros({2, 2}, deeptiny::Device::CPU,
                                        deeptiny::DType::Float32, true);
  std::vector<float> out(4, 0.0f);
  CHECK_THROWS_WITH(tensor.CopyToBuffer(std::as_writable_bytes(std::span<float>(
                                            out.data(), out.size())),
                                        {2, 2}, deeptiny::DType::BFloat16),
                    doctest::Contains("dtype mismatch"));
}

TEST_CASE("Tensor::CopyToBuffer validates byte size") {
  auto tensor = deeptiny::Tensor::Zeros({2, 2}, deeptiny::Device::CPU,
                                        deeptiny::DType::Float32, true);
  std::vector<float> out(3, 0.0f);
  CHECK_THROWS_WITH(tensor.CopyToBuffer(std::as_writable_bytes(std::span<float>(
                                            out.data(), out.size())),
                                        {2, 2}, deeptiny::DType::Float32),
                    doctest::Contains("byte-size mismatch"));
}
