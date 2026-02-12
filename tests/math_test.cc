#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "deeptiny/math.h"

#include <cstddef>
#include <cstdint>

#include "autograd_meta.h"
#include "deeptiny/functional.h"
#include "doctest/doctest.h"
#include "engine.h"
#include "test_utils.h"
#include "utils.h"

using deeptiny::test_utils::CheckTensorData;
using deeptiny::test_utils::MakeTensor;

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

TEST_CASE("Functional ReLU forward/backward") {
  SUBCASE("Forward") {
    deeptiny::Tensor x =
        MakeTensor({2, 3}, {-1.0f, 0.0f, 2.0f, 3.0f, -4.0f, 5.0f});
    auto out = deeptiny::functional::ReLU(x);
    CheckTensorData(out, {0.0f, 0.0f, 2.0f, 3.0f, 0.0f, 5.0f});
  }

  SUBCASE("Backward") {
    deeptiny::Tensor x =
        MakeTensor({2, 3}, {-1.0f, 0.0f, 2.0f, 3.0f, -4.0f, 5.0f}, true);
    auto loss =
        deeptiny::functional::Reduce(deeptiny::functional::ReLU(x), {0, 1});
    loss.Backward();

    auto grad = x.grad();
    REQUIRE(grad.has_value());
    CheckTensorData(*grad, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f});
  }
}

TEST_CASE("Functional SiLU forward/backward") {
  SUBCASE("Forward") {
    deeptiny::Tensor x = MakeTensor({1, 5}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
    auto out = deeptiny::functional::SiLU(x);
    CheckTensorData(out,
                    {-0.23840584f, -0.26894143f, 0.0f, 0.7310586f, 1.7615942f});
  }

  SUBCASE("Backward") {
    deeptiny::Tensor x =
        MakeTensor({1, 5}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, true);
    auto loss =
        deeptiny::functional::Reduce(deeptiny::functional::SiLU(x), {0, 1});
    loss.Backward();

    auto grad = x.grad();
    REQUIRE(grad.has_value());
    CheckTensorData(*grad,
                    {-0.09078425f, 0.07232949f, 0.5f, 0.92767054f, 1.0907843f});
  }
}

TEST_CASE("Functional Sqrt forward/backward") {
  SUBCASE("Forward") {
    deeptiny::Tensor x = MakeTensor({2, 2}, {1.0f, 4.0f, 9.0f, 16.0f});
    auto out = deeptiny::functional::Sqrt(x);
    CheckTensorData(out, {1.0f, 2.0f, 3.0f, 4.0f});
  }

  SUBCASE("Backward") {
    deeptiny::Tensor x = MakeTensor({2, 2}, {1.0f, 4.0f, 9.0f, 16.0f}, true);
    auto loss =
        deeptiny::functional::Reduce(deeptiny::functional::Sqrt(x), {0, 1});
    loss.Backward();

    auto grad = x.grad();
    REQUIRE(grad.has_value());
    CheckTensorData(*grad, {0.5f, 0.25f, 0.16666667f, 0.125f});
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

TEST_CASE("BatchedMatMul forward") {
  SUBCASE("No broadcast") {
    deeptiny::Tensor a =
        MakeTensor({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    deeptiny::Tensor b =
        MakeTensor({2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto out = deeptiny::math::BatchedMatMul(a, b);

    const deeptiny::Shape expected_shape{2, 2, 2};
    CHECK(out.shape() == expected_shape);
    CheckTensorData(out, {22, 28, 49, 64, 220, 244, 301, 334});
  }

  SUBCASE("Leading batch dims broadcast") {
    deeptiny::Tensor a = MakeTensor({1, 2, 2, 2}, {1, 0, 0, 1, 2, 0, 0, 2});
    deeptiny::Tensor b =
        MakeTensor({3, 1, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto out = deeptiny::math::BatchedMatMul(a, b);

    const deeptiny::Shape expected_shape{3, 2, 2, 2};
    CHECK(out.shape() == expected_shape);
    CheckTensorData(out, {1,  2,  3,  4,  2, 4,  6,  8,  5,  6,  7,  8,
                          10, 12, 14, 16, 9, 10, 11, 12, 18, 20, 22, 24});
  }

  SUBCASE("Supports non-contiguous batch views with contiguous inner dims") {
    deeptiny::Tensor a_base =
        MakeTensor({4, 2, 3}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    deeptiny::Tensor b_base =
        MakeTensor({4, 3, 2}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

    deeptiny::Tensor a =
        a_base({deeptiny::Slice(1, 4, 2), deeptiny::Slice::All(),
                deeptiny::Slice::All()});
    deeptiny::Tensor b =
        b_base({deeptiny::Slice(1, 4, 2), deeptiny::Slice::All(),
                deeptiny::Slice::All()});
    auto out = deeptiny::math::BatchedMatMul(a, b);

    const deeptiny::Shape expected_shape{2, 2, 2};
    CHECK(out.shape() == expected_shape);
    CheckTensorData(out, {220, 244, 301, 334, 1264, 1324, 1453, 1522});
  }

  SUBCASE("Supports non-contiguous inner dims via contiguous fallback") {
    deeptiny::Tensor a_base = MakeTensor({1, 2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    deeptiny::Tensor b = MakeTensor({1, 2, 3}, {1, 2, 3, 4, 5, 6});

    deeptiny::Tensor a = a_base({deeptiny::Slice::All(), deeptiny::Slice::All(),
                                 deeptiny::Slice(0, 4, 2)});
    auto out = deeptiny::math::BatchedMatMul(a, b);

    const deeptiny::Shape expected_shape{1, 2, 3};
    CHECK(out.shape() == expected_shape);
    CheckTensorData(out, {13, 17, 21, 33, 45, 57});
  }

  SUBCASE("Rejects rank smaller than 3") {
    deeptiny::Tensor a = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6});
    deeptiny::Tensor b = MakeTensor({3, 2}, {1, 2, 3, 4, 5, 6});
    CHECK_THROWS_WITH(deeptiny::math::BatchedMatMul(a, b),
                      doctest::Contains("rank >= 3"));
  }

  SUBCASE("Rejects inner-dim mismatch") {
    deeptiny::Tensor a = MakeTensor({1, 2, 3}, {1, 2, 3, 4, 5, 6});
    deeptiny::Tensor b = MakeTensor({1, 4, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    CHECK_THROWS_WITH(deeptiny::math::BatchedMatMul(a, b),
                      doctest::Contains("inner dimensions"));
  }

  SUBCASE("Rejects non-broadcastable leading dims") {
    deeptiny::Tensor a = MakeTensor({2, 2, 2, 3}, std::vector<float>(24, 1.0f));
    deeptiny::Tensor b = MakeTensor({3, 2, 3, 4}, std::vector<float>(72, 1.0f));
    CHECK_THROWS_WITH(deeptiny::math::BatchedMatMul(a, b),
                      doctest::Contains("broadcast batch dimensions"));
  }

  SUBCASE("Supports transpose flags") {
    deeptiny::Tensor a = MakeTensor({1, 3, 2}, {1, 2, 3, 4, 5, 6});
    deeptiny::Tensor b =
        MakeTensor({1, 3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto out = deeptiny::math::BatchedMatMul(a, b, true, false);

    const deeptiny::Shape expected_shape{1, 2, 4};
    CHECK(out.shape() == expected_shape);
    CheckTensorData(out, {61, 70, 79, 88, 76, 88, 100, 112});
  }
}

TEST_CASE("BatchedMatMul backward") {
  SUBCASE("No broadcast") {
    deeptiny::Tensor a = MakeTensor({1, 2, 3}, {1, 2, 3, 4, 5, 6}, true);
    deeptiny::Tensor b = MakeTensor({1, 3, 2}, {7, 8, 9, 10, 11, 12}, true);
    auto loss = deeptiny::functional::Reduce(
        deeptiny::math::BatchedMatMul(a, b), {0, 1, 2});
    loss.Backward();

    auto a_grad = a.grad();
    auto b_grad = b.grad();
    REQUIRE(a_grad.has_value());
    REQUIRE(b_grad.has_value());
    CheckTensorData(*a_grad, {15, 19, 23, 15, 19, 23});
    CheckTensorData(*b_grad, {5, 5, 7, 7, 9, 9});
  }

  SUBCASE("Broadcasted batch dims reduce gradients") {
    deeptiny::Tensor a =
        MakeTensor({1, 2, 2, 2}, {1, 0, 0, 1, 2, 0, 0, 2}, true);
    deeptiny::Tensor b =
        MakeTensor({3, 1, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, true);
    auto loss = deeptiny::functional::Reduce(
        deeptiny::math::BatchedMatMul(a, b), {0, 1, 2, 3});
    loss.Backward();

    auto a_grad = a.grad();
    auto b_grad = b.grad();
    REQUIRE(a_grad.has_value());
    REQUIRE(b_grad.has_value());
    CheckTensorData(*a_grad, {33, 45, 33, 45, 33, 45, 33, 45});
    CheckTensorData(*b_grad, {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3});
  }

  SUBCASE("Transpose flags propagate through backward") {
    deeptiny::Tensor a = MakeTensor({1, 3, 2}, {1, 2, 3, 4, 5, 6}, true);
    deeptiny::Tensor b =
        MakeTensor({1, 3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, true);
    auto loss = deeptiny::functional::Reduce(
        deeptiny::math::BatchedMatMul(a, b, true, false), {0, 1, 2});
    loss.Backward();

    auto a_grad = a.grad();
    auto b_grad = b.grad();
    REQUIRE(a_grad.has_value());
    REQUIRE(b_grad.has_value());
    CheckTensorData(*a_grad, {10, 10, 26, 26, 42, 42});
    CheckTensorData(*b_grad, {3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11});
  }

  SUBCASE("Shared inputs across multiple matmuls do not trip version checks") {
    deeptiny::Tensor x = MakeTensor({1, 2, 3}, {1, 2, 3, 4, 5, 6}, true);
    deeptiny::Tensor w1 = MakeTensor({1, 3, 2}, {1, 1, 1, 1, 1, 1}, true);
    deeptiny::Tensor w2 = MakeTensor({1, 3, 2}, {2, 2, 2, 2, 2, 2}, true);

    auto y1 = deeptiny::math::BatchedMatMul(x, w1);
    auto y2 = deeptiny::math::BatchedMatMul(x, w2);
    auto loss = deeptiny::functional::Reduce(y1 + y2, {0, 1, 2});

    CHECK_NOTHROW(loss.Backward());

    auto x_grad = x.grad();
    REQUIRE(x_grad.has_value());
    CheckTensorData(*x_grad, {6, 6, 6, 6, 6, 6});
  }
}

TEST_CASE("Reshape forward/backward and guards") {
  SUBCASE("Reshape preserves row-major order for contiguous tensor") {
    deeptiny::Tensor t = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6});
    auto reshaped = t.Reshape({3, 2});
    const deeptiny::Shape expected_shape{3, 2};
    CHECK(reshaped.shape() == expected_shape);
    CheckTensorData(reshaped, {1, 2, 3, 4, 5, 6});
  }

  SUBCASE("Reshape backward maps gradient to original shape") {
    deeptiny::Tensor t = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6}, true);
    auto loss = deeptiny::functional::Reduce(t.Reshape({3, 2}), {0, 1});
    loss.Backward();

    auto grad = t.grad();
    REQUIRE(grad.has_value());
    CheckTensorData(*grad, {1, 1, 1, 1, 1, 1});
  }

  SUBCASE("Reshape rejects non-contiguous input") {
    deeptiny::Tensor t = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6});
    auto slice = t({deeptiny::Slice(0, 2), deeptiny::Slice(0, 3, 2)});
    deeptiny::Tensor view_tensor = slice;
    CHECK_THROWS_WITH(view_tensor.Reshape({4}),
                      doctest::Contains("contiguous"));
  }

  SUBCASE("Reshape rejects element-count mismatch") {
    deeptiny::Tensor t = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6});
    CHECK_THROWS_WITH(t.Reshape({7}),
                      doctest::Contains("same number of elements"));
  }
}

TEST_CASE("Context set/get and version checks") {
  SUBCASE("Set and Get round trip") {
    deeptiny::Context ctx;
    deeptiny::Tensor t = MakeTensor({2}, {1, 2});
    ctx.Set(11, t);
    CheckTensorData(ctx.Get(11), {1, 2});
  }

  SUBCASE("Get throws when key is missing") {
    deeptiny::Context ctx;
    CHECK_THROWS_WITH(ctx.Get(99), doctest::Contains("Context key not found"));
  }

  SUBCASE("Get throws when storage version changed") {
    deeptiny::Context ctx;
    deeptiny::Tensor t = MakeTensor({2}, {1, 2});
    ctx.Set(7, t);
    auto impl = deeptiny::utils::TensorAccessor::GetTensorImpl(t);
    auto* data = static_cast<float*>(impl->data());
    data[0] += 1.0f;
    CHECK_THROWS_WITH(ctx.Get(7), doctest::Contains("version mismatch"));
  }
}

TEST_CASE("Backward fails when saved tensors are modified after forward") {
  SUBCASE("Mul backward detects saved tensor mutation") {
    deeptiny::Tensor a = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6}, true);
    deeptiny::Tensor b = MakeTensor({1, 3}, {1, 2, 4}, true);
    auto loss = deeptiny::functional::Reduce(a * b, {0, 1});
    auto a_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(a);
    auto* a_data = static_cast<float*>(a_impl->data());
    a_data[0] += 1.0f;
    CHECK_THROWS_WITH(loss.Backward(), doctest::Contains("modified in-place"));
  }

  SUBCASE("Div backward detects saved tensor mutation") {
    deeptiny::Tensor a = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6}, true);
    deeptiny::Tensor b = MakeTensor({1, 3}, {1, 2, 4}, true);
    auto loss = deeptiny::functional::Reduce(a / b, {0, 1});
    auto b_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(b);
    auto* b_data = static_cast<float*>(b_impl->data());
    b_data[0] += 1.0f;
    CHECK_THROWS_WITH(loss.Backward(), doctest::Contains("modified in-place"));
  }

  SUBCASE("BatchedMatMul backward detects saved tensor mutation") {
    deeptiny::Tensor a = MakeTensor({1, 2, 3}, {1, 2, 3, 4, 5, 6}, true);
    deeptiny::Tensor b = MakeTensor({1, 3, 2}, {1, 2, 3, 4, 5, 6}, true);
    auto loss = deeptiny::functional::Reduce(
        deeptiny::math::BatchedMatMul(a, b), {0, 1, 2});
    auto a_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(a);
    auto* a_data = static_cast<float*>(a_impl->data());
    a_data[0] += 1.0f;
    CHECK_THROWS_WITH(loss.Backward(), doctest::Contains("modified in-place"));
  }
}
