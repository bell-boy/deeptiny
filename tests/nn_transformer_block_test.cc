#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <vector>

#include "deeptiny/autograd.h"
#include "deeptiny/functional.h"
#include "deeptiny/nn/kv_cache.h"
#include "deeptiny/nn/transformer_block.h"
#include "doctest/doctest.h"
#include "test_utils.h"

using deeptiny::test_utils::CopyTensorData;
using deeptiny::test_utils::MakeTensor;
using deeptiny::test_utils::ToVector;

namespace {

void InstallAttentionProbeWeights(deeptiny::nn::TransformerBlock& block) {
  const deeptiny::Shape weight_shape{1, 1, 4, 4};

  // q = [x0, 0, 0, 0]
  CopyTensorData(deeptiny::Tensor::FromVector(
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
                     weight_shape, deeptiny::Device::CPU, true),
                 block.self_attention().q_weight());

  // k = [x0, 0, 0, 0]
  CopyTensorData(deeptiny::Tensor::FromVector(
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
                     weight_shape, deeptiny::Device::CPU, true),
                 block.self_attention().k_weight());

  // v = [x2, 0, 0, 0]
  CopyTensorData(deeptiny::Tensor::FromVector(
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
                     weight_shape, deeptiny::Device::CPU, true),
                 block.self_attention().v_weight());

  // output = [context0, 0, 0, 0]
  CopyTensorData(deeptiny::Tensor::FromVector(
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
                     weight_shape, deeptiny::Device::CPU, true),
                 block.self_attention().o_weight());
}

void ZeroTensor(deeptiny::Tensor& tensor) {
  CopyTensorData(
      deeptiny::Tensor::Zeros(tensor.shape(), tensor.device(), tensor.dtype()),
      tensor);
}

void ZeroFeedForward(deeptiny::nn::TransformerBlock& block) {
  auto& ffn = block.ffn();
  ZeroTensor(ffn.gate_proj().weight());
  ZeroTensor(ffn.up_proj().weight());
  ZeroTensor(ffn.down_proj().weight());
}

}  // namespace

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
      /*num_attention_heads=*/1,
      /*num_key_value_heads=*/1,
      /*attention_bias=*/false,
      /*mlp_bias=*/false,
      /*is_causal=*/false);
  InstallAttentionProbeWeights(block);
  ZeroFeedForward(block);

  auto x = MakeTensor({1, 2, 4},
                      {1.0f, 0.0f, 0.0f, 0.0f,  //
                       1.0f, 0.0f, 1.0f, 0.0f},
                      true);
  auto y_unmasked = block(x);
  CHECK(y_unmasked.shape() == deeptiny::Shape({1, 2, 4}));

  auto mask = MakeTensor({1, 1, 2, 2}, {0.0f, -1.0e9f, 0.0f, -1.0e9f});
  auto y_masked = block(x, mask, /*position_offset=*/2);
  CHECK(y_masked.shape() == deeptiny::Shape({1, 2, 4}));

  const auto unmasked_data = ToVector(y_unmasked);
  const auto masked_data = ToVector(y_masked);
  CHECK(unmasked_data[4] > masked_data[4]);
  CHECK(masked_data[4] == deeptiny::test_utils::Approx(1.0f, 1e-4));
}

TEST_CASE("nn::TransformerBlock KV cache behavior") {
  deeptiny::nn::TransformerBlock block(
      /*hidden_size=*/4,
      /*mlp_hidden_dim=*/8,
      /*num_attention_heads=*/1,
      /*num_key_value_heads=*/1,
      /*attention_bias=*/false,
      /*mlp_bias=*/false,
      /*is_causal=*/true);
  InstallAttentionProbeWeights(block);
  ZeroFeedForward(block);
  deeptiny::NoGrad no_grad_guard;

  auto x = MakeTensor({1, 4, 4}, {1.0f, 0.0f, 0.0f, 0.0f,  //
                                  1.0f, 0.0f, 1.0f, 0.0f,  //
                                  1.0f, 0.0f, 2.0f, 0.0f,  //
                                  1.0f, 0.0f, 3.0f, 0.0f});
  auto y_full = block(x);

  deeptiny::nn::KVCache cache(/*batch_size=*/1, /*num_key_value_heads=*/1,
                              /*head_dim=*/4);
  auto x0 =
      x({deeptiny::Slice(0, 1), deeptiny::Slice(0, 2), deeptiny::Slice(0, 4)});
  auto x1 =
      x({deeptiny::Slice(0, 1), deeptiny::Slice(2, 4), deeptiny::Slice(0, 4)});
  auto y0 = block(x0, std::nullopt, /*position_offset=*/0, &cache);
  auto y1 = block(x1, std::nullopt, /*position_offset=*/2, &cache);

  const auto full = ToVector(y_full);
  const auto out0 = ToVector(y0);
  const auto out1 = ToVector(y1);
  REQUIRE(full.size() == 16);
  REQUIRE(out0.size() == 8);
  REQUIRE(out1.size() == 8);

  for (size_t i = 0; i < out0.size(); ++i) {
    CHECK(out0[i] == deeptiny::test_utils::Approx(full[i], 1e-4));
  }
  for (size_t i = 0; i < out1.size(); ++i) {
    CHECK(out1[i] == deeptiny::test_utils::Approx(full[out0.size() + i], 1e-4));
  }

  CHECK_THROWS_WITH(block(x1, std::nullopt, /*position_offset=*/1, &cache),
                    doctest::Contains("position_offset"));
}
TEST_CASE("nn::TransformerBlock supports SiLU MLP hidden activation") {
  deeptiny::nn::TransformerBlock block(
      /*hidden_size=*/4,
      /*mlp_hidden_dim=*/8,
      /*num_attention_heads=*/1,
      /*num_key_value_heads=*/1,
      /*attention_bias=*/false,
      /*mlp_bias=*/false,
      /*is_causal=*/false,
      /*rope_theta=*/10000.0f,
      /*norm_eps=*/1.0e-5f, deeptiny::Device::CPU,
      deeptiny::nn::GatedMLP::HiddenAct::SiLU);
  auto x = MakeTensor({1, 2, 4},
                      {0.1f, -0.2f, 0.3f, -0.4f,  //
                       0.5f, -0.6f, 0.7f, -0.8f},
                      true);

  auto y = block(x);
  CHECK(y.shape() == deeptiny::Shape({1, 2, 4}));
  auto loss = deeptiny::functional::Reduce(y, {0, 1, 2});
  loss.Backward();
  CHECK(x.grad().has_value());
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
