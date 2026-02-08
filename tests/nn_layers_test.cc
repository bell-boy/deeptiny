#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <stdexcept>
#include <vector>

#include "deeptiny/functional.h"
#include "deeptiny/nn/embedding.h"
#include "deeptiny/nn/gated_relu.h"
#include "deeptiny/nn/linear.h"
#include "deeptiny/nn/rms_norm.h"
#include "doctest/doctest.h"
#include "test_utils.h"

using deeptiny::test_utils::CheckTensorData;
using deeptiny::test_utils::MakeTensor;
using deeptiny::test_utils::ToVector;

TEST_CASE("nn::Linear forward preserves leading dimensions") {
  deeptiny::nn::Linear linear(/*in_dim=*/4, /*out_dim=*/6);

  SUBCASE("rank-2") {
    auto out = linear(MakeTensor({5, 4}, std::vector<float>(20, 1.0f), true));
    CHECK(out.shape() == deeptiny::Shape({5, 6}));
  }

  SUBCASE("rank-3") {
    auto out =
        linear(MakeTensor({2, 3, 4}, std::vector<float>(24, 1.0f), true));
    CHECK(out.shape() == deeptiny::Shape({2, 3, 6}));
  }
}

TEST_CASE("nn::Linear backward has expected analytic input gradient") {
  deeptiny::nn::Linear linear(/*in_dim=*/2, /*out_dim=*/3, /*bias=*/false);
  linear.weight() = deeptiny::Tensor::FromVector(
      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      deeptiny::Shape{1, 2, 3}, deeptiny::Device::CPU, true);

  auto x = deeptiny::Tensor::FromVector(
      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, deeptiny::Shape{2, 2},
      deeptiny::Device::CPU, true);
  auto y = linear(x);
  auto loss = deeptiny::functional::Reduce(y, {0, 1});
  loss.Backward();

  auto grad = x.grad();
  REQUIRE(grad.has_value());
  CheckTensorData(*grad, {6.0f, 15.0f, 6.0f, 15.0f});
}

TEST_CASE("nn::Embedding contract and backward accumulation") {
  deeptiny::nn::Embedding embedding(/*num_embeddings=*/5, /*embedding_dim=*/3);

  SUBCASE("shape and lookup") {
    const std::vector<int64_t> indices{1, 3, 1, 0};
    auto out = embedding(indices, {2, 2});
    CHECK(out.shape() == deeptiny::Shape({2, 2, 3}));

    const auto out_data = ToVector(out);
    const auto weight_data = ToVector(embedding.weight());
    for (uint64_t i = 0; i < indices.size(); ++i) {
      for (uint64_t c = 0; c < 3; ++c) {
        CHECK(out_data[i * 3 + c] ==
              deeptiny::test_utils::Approx(
                  weight_data[static_cast<uint64_t>(indices[i]) * 3 + c]));
      }
    }
  }

  SUBCASE("backward accumulation") {
    auto out = embedding({1, 3, 1, 1}, {2, 2});
    auto loss = deeptiny::functional::Reduce(out, {0, 1, 2});
    loss.Backward();

    auto grad = embedding.weight().grad();
    REQUIRE(grad.has_value());
    CHECK(grad->shape() == deeptiny::Shape({5, 3}));
    CheckTensorData(*grad, {
                               0.0f,
                               0.0f,
                               0.0f,
                               3.0f,
                               3.0f,
                               3.0f,
                               0.0f,
                               0.0f,
                               0.0f,
                               1.0f,
                               1.0f,
                               1.0f,
                               0.0f,
                               0.0f,
                               0.0f,
                           });
  }

  SUBCASE("guards") {
    CHECK_THROWS(embedding({0, 5}, {2}));
    CHECK_THROWS(embedding({-1, 2}, {2}));
    CHECK_THROWS(embedding({1, 2, 3}, {2, 2}));
  }
}

TEST_CASE("nn::GatedReLU forward/backward smoke") {
  deeptiny::nn::GatedReLU mlp(/*in_dim=*/4, /*hidden_dim=*/8, /*out_dim=*/4);
  auto x = MakeTensor({2, 3, 4}, std::vector<float>(24, 0.25f), true);
  auto y = mlp(x);
  CHECK(y.shape() == deeptiny::Shape({2, 3, 4}));

  auto loss = deeptiny::functional::Reduce(y, {0, 1, 2});
  loss.Backward();
  CHECK(x.grad().has_value());
}

TEST_CASE("nn::RMSNorm forward/backward") {
  deeptiny::nn::RMSNorm norm(/*dim=*/2, /*eps=*/0.0f);
  auto x = MakeTensor({2, 2}, {3.0f, 4.0f, 6.0f, 8.0f}, true);
  auto y = norm(x);
  CHECK(y.shape() == deeptiny::Shape({2, 2}));
  CheckTensorData(y, {0.84852815f, 1.1313709f, 0.84852815f, 1.1313709f});

  auto loss = deeptiny::functional::Reduce(y, {0, 1});
  loss.Backward();

  auto x_grad = x.grad();
  REQUIRE(x_grad.has_value());
  CheckTensorData(*x_grad,
                  {0.04525483f, -0.03394112f, 0.02262741f, -0.01697056f});

  auto w_grad = norm.weight().grad();
  REQUIRE(w_grad.has_value());
  CheckTensorData(*w_grad, {1.6970563f, 2.2627418f});
}

TEST_CASE("nn::RMSNorm guards") {
  CHECK_THROWS(deeptiny::nn::RMSNorm(/*dim=*/0));
  CHECK_THROWS(deeptiny::nn::RMSNorm(/*dim=*/4, /*eps=*/-1.0f));

  deeptiny::nn::RMSNorm norm(/*dim=*/4);
  CHECK_THROWS(norm(MakeTensor({2, 3}, std::vector<float>(6, 1.0f), true)));
  CHECK_THROWS(norm(MakeTensor({}, std::vector<float>{1.0f}, true)));
}
