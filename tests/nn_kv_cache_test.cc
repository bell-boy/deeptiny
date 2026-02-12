#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <vector>

#include "deeptiny/autograd.h"
#include "deeptiny/nn/kv_cache.h"
#include "doctest/doctest.h"
#include "test_utils.h"

using deeptiny::test_utils::MakeTensor;
using deeptiny::test_utils::ToVector;

TEST_CASE("nn::KVCache append, growth, and clear") {
  deeptiny::NoGrad no_grad_guard;
  deeptiny::nn::KVCache cache(/*batch_size=*/1, /*num_key_value_heads=*/2,
                              /*head_dim=*/2, /*preallocate_seq_len=*/1);

  auto k1 = MakeTensor({1, 2, 1, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  auto v1 = MakeTensor({1, 2, 1, 2}, {101.0f, 102.0f, 103.0f, 104.0f});
  cache.update(k1, v1);
  CHECK(cache.seq_len() == 1);
  CHECK(cache.keys().shape() == deeptiny::Shape({1, 2, 1, 2}));
  CHECK(ToVector(cache.keys()) == std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
  CHECK(ToVector(cache.values()) ==
        std::vector<float>{101.0f, 102.0f, 103.0f, 104.0f});

  auto k2 = MakeTensor({1, 2, 2, 2},
                       {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  auto v2 = MakeTensor({1, 2, 2, 2}, {105.0f, 106.0f, 107.0f, 108.0f, 109.0f,
                                      110.0f, 111.0f, 112.0f});
  cache.update(k2, v2);
  CHECK(cache.seq_len() == 3);
  CHECK(cache.keys().shape() == deeptiny::Shape({1, 2, 3, 2}));
  CHECK(ToVector(cache.keys()) ==
        std::vector<float>{1.0f, 2.0f, 5.0f, 6.0f, 7.0f, 8.0f, 3.0f, 4.0f, 9.0f,
                           10.0f, 11.0f, 12.0f});
  CHECK(ToVector(cache.values()) ==
        std::vector<float>{101.0f, 102.0f, 105.0f, 106.0f, 107.0f, 108.0f,
                           103.0f, 104.0f, 109.0f, 110.0f, 111.0f, 112.0f});

  cache.Clear();
  CHECK(cache.seq_len() == 0);
  CHECK(cache.keys().shape() == deeptiny::Shape({1, 2, 0, 2}));
  CHECK(cache.values().shape() == deeptiny::Shape({1, 2, 0, 2}));
}

TEST_CASE("nn::KVCache update guards") {
  deeptiny::nn::KVCache cache(/*batch_size=*/1, /*num_key_value_heads=*/2,
                              /*head_dim=*/2);

  auto valid = MakeTensor({1, 2, 1, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  auto bad_rank = MakeTensor({1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  auto bad_heads = MakeTensor({1, 1, 1, 2}, {1.0f, 2.0f});
  auto bad_dim = MakeTensor({1, 2, 1, 1}, {1.0f, 2.0f});
  auto requires_grad = MakeTensor({1, 2, 1, 2}, {1.0f, 2.0f, 3.0f, 4.0f},
                                  /*requires_grad=*/true);

  CHECK_THROWS_WITH(cache.update(valid, valid),
                    doctest::Contains("inference-only"));

  deeptiny::NoGrad no_grad_guard;
  CHECK_THROWS_WITH(cache.update(bad_rank, valid), doctest::Contains("rank-4"));
  CHECK_THROWS_WITH(cache.update(bad_heads, bad_heads),
                    doctest::Contains("shape mismatch"));
  CHECK_THROWS_WITH(cache.update(bad_dim, bad_dim),
                    doctest::Contains("shape mismatch"));
  CHECK_THROWS_WITH(cache.update(valid, bad_dim),
                    doctest::Contains("shape mismatch"));
  CHECK_THROWS_WITH(cache.update(requires_grad, valid),
                    doctest::Contains("inference-only"));
}
