#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <vector>

#include "deeptiny/functional.h"
#include "deeptiny/math.h"
#include "doctest/doctest.h"
#include "test_utils.h"
#include "utils.h"

using deeptiny::test_utils::CheckTensorData;
using deeptiny::test_utils::MakeTensor;

TEST_CASE("functional::Softmax forward") {
  SUBCASE("Last dimension") {
    deeptiny::Tensor x =
        MakeTensor({2, 3}, {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f});
    auto y = deeptiny::functional::Softmax(x, 1);
    CheckTensorData(y, {0.09003057f, 0.24472848f, 0.66524094f, 0.09003057f,
                        0.24472848f, 0.66524094f});
  }

  SUBCASE("Non-last dimension") {
    deeptiny::Tensor x = MakeTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto y = deeptiny::functional::Softmax(x, 0);
    CheckTensorData(y, {0.11920292f, 0.11920292f, 0.88079709f, 0.88079709f});
  }

  SUBCASE("Stable with large logits") {
    deeptiny::Tensor x = MakeTensor({1, 2}, {1000.0f, 1001.0f});
    auto y = deeptiny::functional::Softmax(x, 1);
    CheckTensorData(y, {0.26894143f, 0.73105860f});
  }
}

TEST_CASE("functional::Softmax backward") {
  deeptiny::Tensor x = MakeTensor({3}, {1.0f, 2.0f, 3.0f}, true);
  deeptiny::Tensor w = MakeTensor({3}, {1.0f, 2.0f, 3.0f});
  auto loss = deeptiny::functional::Reduce(
      deeptiny::functional::Softmax(x, 0) * w, {0});
  loss.Backward();

  auto grad = x.grad();
  REQUIRE(grad.has_value());
  CheckTensorData(*grad, {-0.14181709f, -0.14077036f, 0.28258744f});
}

TEST_CASE("functional::Softmax guards") {
  SUBCASE("Scalar input is rejected in frontend") {
    deeptiny::Tensor x = MakeTensor({}, {1.0f});
    CHECK_THROWS_WITH(deeptiny::functional::Softmax(x, 0),
                      doctest::Contains("does not support scalar input"));
  }

  SUBCASE("Dim out of range") {
    deeptiny::Tensor x = MakeTensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    CHECK_THROWS_WITH(deeptiny::functional::Softmax(x, 2),
                      doctest::Contains("dim out of range"));
  }

  SUBCASE("Backward detects modified saved output") {
    deeptiny::Tensor x = MakeTensor({1, 3}, {1.0f, 2.0f, 3.0f}, true);
    auto y = deeptiny::functional::Softmax(x, 1);
    auto loss = deeptiny::functional::Reduce(y, {0, 1});

    auto y_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(y);
    auto* y_data = static_cast<float*>(y_impl->data());
    y_data[0] += 1.0f;

    CHECK_THROWS_WITH(loss.Backward(), doctest::Contains("version mismatch"));
  }
}
