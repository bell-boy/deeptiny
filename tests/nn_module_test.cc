#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "deeptiny/nn/module.h"

#include <vector>

#include "deeptiny/functional.h"
#include "deeptiny/math.h"
#include "doctest/doctest.h"
#include "test_utils.h"

using deeptiny::test_utils::CheckTensorData;
using deeptiny::test_utils::MakeTensor;

namespace {

class LeafModule final : public deeptiny::nn::Module {
 public:
  explicit LeafModule(deeptiny::Tensor parameter) {
    RegisterParameter(std::move(parameter));
  }
};

class ParentModule final : public deeptiny::nn::Module {
 public:
  ParentModule(std::vector<deeptiny::Tensor> parameters,
               deeptiny::nn::Module& child) {
    for (auto& parameter : parameters) {
      RegisterParameter(std::move(parameter));
    }
    RegisterSubmodule(child);
  }
};

class InvalidModule final : public deeptiny::nn::Module {
 public:
  explicit InvalidModule(deeptiny::Tensor parameter) {
    RegisterParameter(std::move(parameter));
  }
};

}  // namespace

TEST_CASE("nn::Module tracks requires-grad parameter elements recursively") {
  LeafModule child(MakeTensor({2}, std::vector<float>{1.0f, 2.0f}, true));

  ParentModule model({MakeTensor({2}, std::vector<float>{3.0f, 4.0f}, true)},
                     child);

  CHECK(model.NumParametersRequiringGrad() == 4);
}

TEST_CASE("nn::Module update applies gradients recursively") {
  auto child_weight = MakeTensor({2}, std::vector<float>{3.0f, 4.0f}, true);
  auto root_weight = MakeTensor({2}, std::vector<float>{1.0f, 2.0f}, true);

  auto loss = deeptiny::functional::Reduce(root_weight * root_weight +
                                               child_weight * child_weight,
                                           {0});
  loss.Backward();

  LeafModule child(child_weight);
  ParentModule model({root_weight}, child);

  model.Update(0.1f);

  CheckTensorData(root_weight, {0.8f, 1.6f});
  CheckTensorData(child_weight, {2.4f, 3.2f});
}

TEST_CASE("nn::Module rejects non-trainable parameter registration") {
  CHECK_THROWS_WITH(InvalidModule(MakeTensor({2}, {1.0f, 2.0f}, false)),
                    doctest::Contains("requires_grad=true"));
}
