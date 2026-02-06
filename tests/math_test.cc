#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "deeptiny/math.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "autograd_meta.h"
#include "deeptiny/functional.h"
#include "doctest/doctest.h"
#include "test_utils.h"
#include "utils.h"

namespace {
deeptiny::Tensor MakeTensor(const deeptiny::Shape& shape,
                            const std::vector<float>& values,
                            bool requires_grad = false) {
  if (values.size() != deeptiny::utils::GetTotalSize(shape)) {
    throw std::runtime_error(
        "MakeTensor received mismatched values/shape size");
  }
  return deeptiny::Tensor::FromBuffer(
      std::span<const std::byte>(
          reinterpret_cast<const std::byte*>(values.data()),
          values.size() * sizeof(float)),
      shape, deeptiny::DType::Float32, deeptiny::Device::CPU, requires_grad);
}

std::vector<float> ToVector(const deeptiny::Tensor& t) {
  auto impl = deeptiny::utils::TensorAccessor::GetTensorImpl(t);
  auto contiguous = impl->getContiguousStorage();
  const auto n = contiguous->numel();
  std::vector<float> out(static_cast<size_t>(n), 0.0f);
  contiguous->CopyToHost(0, n, out.data());
  return out;
}

void CheckTensorData(const deeptiny::Tensor& t,
                     const std::vector<float>& expected) {
  const auto actual = ToVector(t);
  REQUIRE(actual.size() == expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    CHECK(actual[i] == deeptiny::test_utils::Approx(expected[i]));
  }
}
}  // namespace

TEST_CASE("Elementary out-of-place forward") {
  deeptiny::Tensor a = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6});
  deeptiny::Tensor b = MakeTensor({1, 3}, {1, 2, 4});

  SUBCASE("Add") {
    auto out = a + b;
    CheckTensorData(out, {2, 4, 7, 5, 7, 10});
  }

  SUBCASE("Sub") {
    auto out = a - b;
    CheckTensorData(out, {0, 0, -1, 3, 3, 2});
  }

  SUBCASE("Mul") {
    auto out = a * b;
    CheckTensorData(out, {1, 4, 12, 4, 10, 24});
  }

  SUBCASE("Div") {
    auto out = a / b;
    CheckTensorData(out, {1.0f, 1.0f, 0.75f, 4.0f, 2.5f, 1.5f});
  }
}

TEST_CASE("Elementary out-of-place backward") {
  SUBCASE("Add") {
    deeptiny::Tensor a = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6}, true);
    deeptiny::Tensor b = MakeTensor({1, 3}, {1, 2, 4}, true);
    auto loss = deeptiny::functional::Reduce(a + b, {0, 1});
    loss.Backward();

    auto a_grad = a.grad();
    auto b_grad = b.grad();
    REQUIRE(a_grad.has_value());
    REQUIRE(b_grad.has_value());
    CheckTensorData(*a_grad, {1, 1, 1, 1, 1, 1});
    CheckTensorData(*b_grad, {2, 2, 2});
  }

  SUBCASE("Sub") {
    deeptiny::Tensor a = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6}, true);
    deeptiny::Tensor b = MakeTensor({1, 3}, {1, 2, 4}, true);
    auto loss = deeptiny::functional::Reduce(a - b, {0, 1});
    loss.Backward();

    auto a_grad = a.grad();
    auto b_grad = b.grad();
    REQUIRE(a_grad.has_value());
    REQUIRE(b_grad.has_value());
    CheckTensorData(*a_grad, {1, 1, 1, 1, 1, 1});
    CheckTensorData(*b_grad, {-2, -2, -2});
  }

  SUBCASE("Mul") {
    deeptiny::Tensor a = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6}, true);
    deeptiny::Tensor b = MakeTensor({1, 3}, {1, 2, 4}, true);
    auto loss = deeptiny::functional::Reduce(a * b, {0, 1});
    loss.Backward();

    auto a_grad = a.grad();
    auto b_grad = b.grad();
    REQUIRE(a_grad.has_value());
    REQUIRE(b_grad.has_value());
    CheckTensorData(*a_grad, {1, 2, 4, 1, 2, 4});
    CheckTensorData(*b_grad, {5, 7, 9});
  }

  SUBCASE("Div") {
    deeptiny::Tensor a = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6}, true);
    deeptiny::Tensor b = MakeTensor({1, 3}, {1, 2, 4}, true);
    auto loss = deeptiny::functional::Reduce(a / b, {0, 1});
    loss.Backward();

    auto a_grad = a.grad();
    auto b_grad = b.grad();
    REQUIRE(a_grad.has_value());
    REQUIRE(b_grad.has_value());
    CheckTensorData(*a_grad, {1.0f, 0.5f, 0.25f, 1.0f, 0.5f, 0.25f});
    CheckTensorData(*b_grad, {-5.0f, -1.75f, -0.5625f});
  }
}

TEST_CASE("Elementary in-place forward") {
  SUBCASE("Add") {
    deeptiny::Tensor a = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6});
    deeptiny::Tensor b = MakeTensor({1, 3}, {1, 2, 4});
    a += b;
    CheckTensorData(a, {2, 4, 7, 5, 7, 10});
  }

  SUBCASE("Sub") {
    deeptiny::Tensor a = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6});
    deeptiny::Tensor b = MakeTensor({1, 3}, {1, 2, 4});
    a -= b;
    CheckTensorData(a, {0, 0, -1, 3, 3, 2});
  }

  SUBCASE("Mul") {
    deeptiny::Tensor a = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6});
    deeptiny::Tensor b = MakeTensor({1, 3}, {1, 2, 4});
    a *= b;
    CheckTensorData(a, {1, 4, 12, 4, 10, 24});
  }

  SUBCASE("Div") {
    deeptiny::Tensor a = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6});
    deeptiny::Tensor b = MakeTensor({1, 3}, {1, 2, 4});
    a /= b;
    CheckTensorData(a, {1.0f, 1.0f, 0.75f, 4.0f, 2.5f, 1.5f});
  }
}

TEST_CASE("Elementary in-place backward") {
  SUBCASE("Add") {
    deeptiny::Tensor a = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6}, true);
    deeptiny::Tensor b = MakeTensor({1, 3}, {1, 2, 4}, true);
    auto a_leaf_meta = deeptiny::utils::TensorAccessor::GetAutogradMeta(a);
    REQUIRE(a_leaf_meta != nullptr);

    a += b;
    auto loss = deeptiny::functional::Reduce(a, {0, 1});
    loss.Backward();

    auto a_grad = a_leaf_meta->grad();
    auto b_grad = b.grad();
    REQUIRE(a_grad.has_value());
    REQUIRE(b_grad.has_value());
    CheckTensorData(*a_grad, {1, 1, 1, 1, 1, 1});
    CheckTensorData(*b_grad, {2, 2, 2});
  }

  SUBCASE("Sub") {
    deeptiny::Tensor a = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6}, true);
    deeptiny::Tensor b = MakeTensor({1, 3}, {1, 2, 4}, true);
    auto a_leaf_meta = deeptiny::utils::TensorAccessor::GetAutogradMeta(a);
    REQUIRE(a_leaf_meta != nullptr);

    a -= b;
    auto loss = deeptiny::functional::Reduce(a, {0, 1});
    loss.Backward();

    auto a_grad = a_leaf_meta->grad();
    auto b_grad = b.grad();
    REQUIRE(a_grad.has_value());
    REQUIRE(b_grad.has_value());
    CheckTensorData(*a_grad, {1, 1, 1, 1, 1, 1});
    CheckTensorData(*b_grad, {-2, -2, -2});
  }

  SUBCASE("Mul") {
    deeptiny::Tensor a = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6}, true);
    deeptiny::Tensor b = MakeTensor({1, 3}, {1, 2, 4}, true);
    auto a_leaf_meta = deeptiny::utils::TensorAccessor::GetAutogradMeta(a);
    REQUIRE(a_leaf_meta != nullptr);

    a *= b;
    auto loss = deeptiny::functional::Reduce(a, {0, 1});
    loss.Backward();

    auto a_grad = a_leaf_meta->grad();
    auto b_grad = b.grad();
    REQUIRE(a_grad.has_value());
    REQUIRE(b_grad.has_value());
    CheckTensorData(*a_grad, {1, 2, 4, 1, 2, 4});
    CheckTensorData(*b_grad, {5, 7, 9});
  }

  SUBCASE("Div") {
    deeptiny::Tensor a = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6}, true);
    deeptiny::Tensor b = MakeTensor({1, 3}, {1, 2, 4}, true);
    auto a_leaf_meta = deeptiny::utils::TensorAccessor::GetAutogradMeta(a);
    REQUIRE(a_leaf_meta != nullptr);

    a /= b;
    auto loss = deeptiny::functional::Reduce(a, {0, 1});
    loss.Backward();

    auto a_grad = a_leaf_meta->grad();
    auto b_grad = b.grad();
    REQUIRE(a_grad.has_value());
    REQUIRE(b_grad.has_value());
    CheckTensorData(*a_grad, {1.0f, 0.5f, 0.25f, 1.0f, 0.5f, 0.25f});
    CheckTensorData(*b_grad, {-5.0f, -1.75f, -0.5625f});
  }
}

TEST_CASE("Elementary in-place ops reject zero-stride destination") {
  SUBCASE("Add") {
    deeptiny::Tensor base = MakeTensor({2, 1}, {1, 2});
    auto dst = deeptiny::utils::BroadcastToShape(base, {2, 3});
    REQUIRE(dst.has_value());
    deeptiny::Tensor rhs = MakeTensor({2, 3}, {1, 1, 1, 1, 1, 1});
    CHECK_THROWS_WITH((*dst) += rhs, doctest::Contains("zero stride"));
  }

  SUBCASE("Sub") {
    deeptiny::Tensor base = MakeTensor({2, 1}, {1, 2});
    auto dst = deeptiny::utils::BroadcastToShape(base, {2, 3});
    REQUIRE(dst.has_value());
    deeptiny::Tensor rhs = MakeTensor({2, 3}, {1, 1, 1, 1, 1, 1});
    CHECK_THROWS_WITH((*dst) -= rhs, doctest::Contains("zero stride"));
  }

  SUBCASE("Mul") {
    deeptiny::Tensor base = MakeTensor({2, 1}, {1, 2});
    auto dst = deeptiny::utils::BroadcastToShape(base, {2, 3});
    REQUIRE(dst.has_value());
    deeptiny::Tensor rhs = MakeTensor({2, 3}, {1, 1, 1, 1, 1, 1});
    CHECK_THROWS_WITH((*dst) *= rhs, doctest::Contains("zero stride"));
  }

  SUBCASE("Div") {
    deeptiny::Tensor base = MakeTensor({2, 1}, {1, 2});
    auto dst = deeptiny::utils::BroadcastToShape(base, {2, 3});
    REQUIRE(dst.has_value());
    deeptiny::Tensor rhs = MakeTensor({2, 3}, {1, 1, 1, 1, 1, 1});
    CHECK_THROWS_WITH((*dst) /= rhs, doctest::Contains("zero stride"));
  }
}
