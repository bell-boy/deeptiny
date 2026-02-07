#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <vector>

#include "deeptiny/tensor.h"
#include "doctest/doctest.h"
#include "test_utils.h"

using deeptiny::test_utils::CheckTensorData;
using deeptiny::test_utils::ToVector;

TEST_CASE("Tensor::Zeros initializes tensor values to zero") {
  auto t = deeptiny::Tensor::Zeros({2, 3}, deeptiny::Device::CPU,
                                   deeptiny::DType::Float32);
  CHECK(t.shape() == deeptiny::Shape{2, 3});
  CheckTensorData(t, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  CHECK(!t.requires_grad());
}

TEST_CASE("Tensor::Zeros supports requires_grad") {
  auto t = deeptiny::Tensor::Zeros({2, 3}, deeptiny::Device::CPU,
                                   deeptiny::DType::Float32, true);
  CHECK(t.shape() == deeptiny::Shape{2, 3});
  CHECK(t.requires_grad());
  CheckTensorData(t, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
}

TEST_CASE("Tensor::CreateUniform creates Float32 values in [0, 1]") {
  auto t = deeptiny::Tensor::CreateUniform({4, 5}, deeptiny::Device::CPU,
                                           deeptiny::DType::Float32);
  CHECK(t.shape() == deeptiny::Shape{4, 5});
  const auto values = ToVector(t);
  CHECK(values.size() == 20);
  for (const auto value : values) {
    CHECK(value >= 0.0f);
    CHECK(value <= 1.0f);
  }
  CHECK(!t.requires_grad());
}

TEST_CASE("Tensor::CreateUniform supports requires_grad") {
  auto t = deeptiny::Tensor::CreateUniform({4, 5}, deeptiny::Device::CPU,
                                           deeptiny::DType::Float32, true);
  CHECK(t.shape() == deeptiny::Shape{4, 5});
  CHECK(t.requires_grad());
}

TEST_CASE("Tensor::numel returns product of shape dimensions") {
  SUBCASE("Rank-2 tensor") {
    auto t = deeptiny::Tensor::Zeros({4, 5}, deeptiny::Device::CPU,
                                     deeptiny::DType::Float32);
    CHECK(t.numel() == 20);
  }

  SUBCASE("Scalar tensor has one element") {
    auto t = deeptiny::Tensor::Zeros({}, deeptiny::Device::CPU,
                                     deeptiny::DType::Float32);
    CHECK(t.numel() == 1);
  }

  SUBCASE("Any zero dimension yields zero elements") {
    auto t = deeptiny::Tensor::Zeros({2, 0, 3}, deeptiny::Device::CPU,
                                     deeptiny::DType::Float32);
    CHECK(t.numel() == 0);
  }
}

TEST_CASE("Tensor::FromVector creates tensors and validates shape") {
  SUBCASE("Round trip with requires_grad") {
    auto t =
        deeptiny::Tensor::FromVector(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
                                     {2, 2}, deeptiny::Device::CPU, true);
    CHECK(t.shape() == deeptiny::Shape{2, 2});
    CHECK(t.requires_grad());
    CheckTensorData(t, {1.0f, 2.0f, 3.0f, 4.0f});
  }

  SUBCASE("Throws on shape/value mismatch") {
    CHECK_THROWS_WITH(
        deeptiny::Tensor::FromVector(std::vector<float>{1.0f, 2.0f, 3.0f},
                                     {2, 2}, deeptiny::Device::CPU, false),
        doctest::Contains("mismatched values/shape size"));
  }
}
