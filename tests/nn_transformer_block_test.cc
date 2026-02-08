#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <vector>

#include "deeptiny/functional.h"
#include "deeptiny/nn/transformer_block.h"
#include "doctest/doctest.h"
#include "test_utils.h"

using deeptiny::test_utils::MakeTensor;

TEST_CASE("nn::TransformerBlock constructor and input guards") {
  CHECK_THROWS_WITH(deeptiny::nn::TransformerBlock(0, 8, 2, 1),
                    doctest::Contains("hidden_size"));
  CHECK_THROWS_WITH(deeptiny::nn::TransformerBlock(4, 0, 2, 1),
                    doctest::Contains("mlp_hidden_dim"));
  CHECK_THROWS_WITH(deeptiny::nn::TransformerBlock(6, 8, 4, 2),
                    doctest::Contains("divisible"));

  deeptiny::nn::TransformerBlock block(4, 8, 2, 1);
  CHECK(block.NumParametersRequiringGrad() == 152);

  CHECK_THROWS_WITH(block(MakeTensor({2, 4}, std::vector<float>(8, 1.0f))),
                    doctest::Contains("rank == 3"));
  CHECK_THROWS_WITH(block(MakeTensor({2, 3, 2}, std::vector<float>(12, 1.0f))),
                    doctest::Contains("hidden dimension mismatch"));
}

TEST_CASE("nn::TransformerBlock forward shape and optional attention args") {
  deeptiny::nn::TransformerBlock block(
      /*hidden_size=*/4,
      /*mlp_hidden_dim=*/8,
      /*num_attention_heads=*/2,
      /*num_key_value_heads=*/1,
      /*attention_bias=*/false,
      /*mlp_bias=*/false,
      /*is_causal=*/false);

  auto x = MakeTensor({2, 3, 4}, std::vector<float>(24, 0.5f), true);
  auto y = block(x);
  CHECK(y.shape() == deeptiny::Shape({2, 3, 4}));

  auto mask = MakeTensor({1, 1, 3, 3}, {0.0f, -1.0e9f, 0.0f, 0.0f, 0.0f,
                                        -1.0e9f, -1.0e9f, 0.0f, 0.0f});
  auto y_masked = block(x, mask, /*position_offset=*/2);
  CHECK(y_masked.shape() == deeptiny::Shape({2, 3, 4}));
}

TEST_CASE("nn::TransformerBlock backward smoke") {
  deeptiny::nn::TransformerBlock block(
      /*hidden_size=*/4,
      /*mlp_hidden_dim=*/8,
      /*num_attention_heads=*/2,
      /*num_key_value_heads=*/1,
      /*attention_bias=*/true,
      /*mlp_bias=*/true,
      /*is_causal=*/true);

  auto x = MakeTensor(
      {2, 3, 4},
      {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f,
       1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f, 2.3f},
      true);

  auto y = block(x);
  auto loss = deeptiny::functional::Reduce(y, {0, 1, 2});
  loss.Backward();

  CHECK(x.grad().has_value());
  CHECK(block.attention_norm().weight().grad().has_value());
  CHECK(block.self_attention().q_weight().grad().has_value());
  CHECK(block.self_attention().k_weight().grad().has_value());
  CHECK(block.self_attention().v_weight().grad().has_value());
  CHECK(block.self_attention().o_weight().grad().has_value());
  CHECK(block.ffn_norm().weight().grad().has_value());
  CHECK(block.ffn().gate_proj().weight().grad().has_value());
  CHECK(block.ffn().up_proj().weight().grad().has_value());
  CHECK(block.ffn().down_proj().weight().grad().has_value());

  const auto& q_bias = block.self_attention().q_bias();
  const auto& k_bias = block.self_attention().k_bias();
  const auto& v_bias = block.self_attention().v_bias();
  const auto& o_bias = block.self_attention().o_bias();
  REQUIRE(q_bias.has_value());
  REQUIRE(k_bias.has_value());
  REQUIRE(v_bias.has_value());
  REQUIRE(o_bias.has_value());
  CHECK(q_bias->grad().has_value());
  CHECK(k_bias->grad().has_value());
  CHECK(v_bias->grad().has_value());
  CHECK(o_bias->grad().has_value());

  const auto& gate_bias = block.ffn().gate_proj().bias();
  const auto& up_bias = block.ffn().up_proj().bias();
  const auto& down_bias = block.ffn().down_proj().bias();
  REQUIRE(gate_bias.has_value());
  REQUIRE(up_bias.has_value());
  REQUIRE(down_bias.has_value());
  CHECK(gate_bias->grad().has_value());
  CHECK(up_bias->grad().has_value());
  CHECK(down_bias->grad().has_value());
}
