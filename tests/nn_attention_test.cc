#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <vector>

#include "deeptiny/functional.h"
#include "deeptiny/nn/multi_head_attention.h"
#include "doctest/doctest.h"
#include "test_utils.h"

using deeptiny::test_utils::MakeTensor;
using deeptiny::test_utils::ToVector;

namespace {

void InstallRopeProbeWeights(deeptiny::nn::MultiHeadAttention& attn) {
  const deeptiny::Shape weight_shape{1, 1, 4, 4};

  // q = [x0, 0, 0, 0]
  attn.set_q_weight(deeptiny::Tensor::FromVector(
      std::vector<float>{
          1.0f,
          0.0f,
          0.0f,
          0.0f,  //
          0.0f,
          0.0f,
          0.0f,
          0.0f,  //
          0.0f,
          0.0f,
          0.0f,
          0.0f,  //
          0.0f,
          0.0f,
          0.0f,
          0.0f,  //
      },
      weight_shape, deeptiny::Device::CPU, true));

  // k = [x0, 0, 0, 0]
  attn.set_k_weight(deeptiny::Tensor::FromVector(
      std::vector<float>{
          1.0f,
          0.0f,
          0.0f,
          0.0f,  //
          0.0f,
          0.0f,
          0.0f,
          0.0f,  //
          0.0f,
          0.0f,
          0.0f,
          0.0f,  //
          0.0f,
          0.0f,
          0.0f,
          0.0f,  //
      },
      weight_shape, deeptiny::Device::CPU, true));

  // v = [x2, 0, 0, 0]
  attn.set_v_weight(deeptiny::Tensor::FromVector(
      std::vector<float>{
          0.0f,
          0.0f,
          0.0f,
          0.0f,  //
          0.0f,
          0.0f,
          0.0f,
          0.0f,  //
          1.0f,
          0.0f,
          0.0f,
          0.0f,  //
          0.0f,
          0.0f,
          0.0f,
          0.0f,  //
      },
      weight_shape, deeptiny::Device::CPU, true));

  // output = [context0, 0, 0, 0]
  attn.set_o_weight(deeptiny::Tensor::FromVector(
      std::vector<float>{
          1.0f,
          0.0f,
          0.0f,
          0.0f,  //
          0.0f,
          0.0f,
          0.0f,
          0.0f,  //
          0.0f,
          0.0f,
          0.0f,
          0.0f,  //
          0.0f,
          0.0f,
          0.0f,
          0.0f,  //
      },
      weight_shape, deeptiny::Device::CPU, true));
}

}  // namespace

TEST_CASE("nn::MultiHeadAttention constructor and shape guards") {
  SUBCASE("Invalid constructor parameters") {
    CHECK_THROWS_WITH(deeptiny::nn::MultiHeadAttention(6, 4, 2),
                      doctest::Contains("divisible"));
    CHECK_THROWS_WITH(deeptiny::nn::MultiHeadAttention(8, 4, 3),
                      doctest::Contains("divisible"));
    CHECK_THROWS_WITH(deeptiny::nn::MultiHeadAttention(6, 2, 1),
                      doctest::Contains("even head_dim"));
  }

  SUBCASE("Input rank guard") {
    deeptiny::nn::MultiHeadAttention attn(4, 2, 1);
    CHECK_THROWS_WITH(attn(MakeTensor({2, 4}, std::vector<float>(8, 1.0f))),
                      doctest::Contains("rank == 3"));
  }
}

TEST_CASE("nn::MultiHeadAttention forward shape and parameter assignment") {
  SUBCASE("MHA shape") {
    deeptiny::nn::MultiHeadAttention attn(4, 2, 2);
    auto y = attn(MakeTensor({3, 5, 4}, std::vector<float>(60, 0.25f)));
    CHECK(y.shape() == deeptiny::Shape({3, 5, 4}));
  }

  SUBCASE("GQA shape") {
    deeptiny::nn::MultiHeadAttention attn(8, 4, 2);
    auto y = attn(MakeTensor({2, 3, 8}, std::vector<float>(48, 0.5f)));
    CHECK(y.shape() == deeptiny::Shape({2, 3, 8}));
  }

  SUBCASE("Direct weight assignment remains usable") {
    deeptiny::nn::MultiHeadAttention attn(4, 1, 1);
    InstallRopeProbeWeights(attn);
    auto y = attn(MakeTensor({1, 2, 4},
                             {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f}));
    CHECK(y.shape() == deeptiny::Shape({1, 2, 4}));
  }
}

TEST_CASE("nn::MultiHeadAttention RoPE and masking behavior") {
  SUBCASE(
      "RoPE changes row behavior when q/k content is identical across tokens") {
    deeptiny::nn::MultiHeadAttention attn(4, 1, 1, false, false);
    InstallRopeProbeWeights(attn);

    auto x = MakeTensor({1, 2, 4}, {1.0f, 0.0f, 0.0f, 0.0f,  //
                                    1.0f, 0.0f, 1.0f, 0.0f});
    auto y = attn(x);
    const auto out = ToVector(y);

    const float token0 = out[0];
    const float token1 = out[4];
    CHECK(token0 != deeptiny::test_utils::Approx(token1, 1e-4));
  }

  SUBCASE("Causal mask blocks future token influence") {
    deeptiny::nn::MultiHeadAttention attn(4, 1, 1, false, true);
    InstallRopeProbeWeights(attn);

    auto x_a = MakeTensor({1, 2, 4}, {1.0f, 0.0f, 0.0f, 0.0f,  //
                                      1.0f, 0.0f, 1.0f, 0.0f});
    auto x_b = MakeTensor({1, 2, 4}, {1.0f, 0.0f, 0.0f, 0.0f,  //
                                      1.0f, 0.0f, 7.0f, 0.0f});

    auto y_a = attn(x_a);
    auto y_b = attn(x_b);
    const auto out_a = ToVector(y_a);
    const auto out_b = ToVector(y_b);

    CHECK(out_a[0] == deeptiny::test_utils::Approx(out_b[0]));
  }

  SUBCASE("External additive mask is applied") {
    deeptiny::nn::MultiHeadAttention attn(4, 1, 1, false, false);
    InstallRopeProbeWeights(attn);

    auto x = MakeTensor({1, 2, 4}, {1.0f, 0.0f, 0.0f, 0.0f,  //
                                    1.0f, 0.0f, 1.0f, 0.0f});
    auto unmasked = attn(x);

    auto mask = MakeTensor({1, 1, 2, 2}, {0.0f, -1.0e9f, 0.0f, -1.0e9f});
    auto masked = attn(x, mask);

    const auto unmasked_data = ToVector(unmasked);
    const auto masked_data = ToVector(masked);
    CHECK(unmasked_data[4] > 0.1f);
    CHECK(masked_data[4] == deeptiny::test_utils::Approx(0.0f, 1e-4));
  }
}

TEST_CASE("nn::MultiHeadAttention backward smoke") {
  deeptiny::nn::MultiHeadAttention attn(4, 2, 1, true, true);
  auto x = MakeTensor(
      {2, 3, 4},
      {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f,
       1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f, 2.3f},
      true);

  auto y = attn(x);
  auto loss = deeptiny::functional::Reduce(y, {0, 1, 2});
  loss.Backward();

  CHECK(x.grad().has_value());
  CHECK(attn.q_weight().grad().has_value());
  CHECK(attn.k_weight().grad().has_value());
  CHECK(attn.v_weight().grad().has_value());
  CHECK(attn.o_weight().grad().has_value());

  auto q_bias = attn.q_bias();
  auto k_bias = attn.k_bias();
  auto v_bias = attn.v_bias();
  auto o_bias = attn.o_bias();
  REQUIRE(q_bias.has_value());
  REQUIRE(k_bias.has_value());
  REQUIRE(v_bias.has_value());
  REQUIRE(o_bias.has_value());
  CHECK(q_bias->grad().has_value());
  CHECK(k_bias->grad().has_value());
  CHECK(v_bias->grad().has_value());
  CHECK(o_bias->grad().has_value());
}
