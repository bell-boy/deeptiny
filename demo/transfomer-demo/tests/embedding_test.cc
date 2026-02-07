#include "modules/embedding.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "deeptiny/functional.h"
#include "deeptiny/tensor.h"
#include "deeptiny/types.h"
#include "utils.h"

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool NearlyEqual(float a, float b, float eps = 1e-5f) {
  return std::fabs(a - b) <= eps;
}

std::vector<float> ToVector(const deeptiny::Tensor& t) {
  auto impl = deeptiny::utils::TensorAccessor::GetTensorImpl(t);
  auto contiguous = impl->getContiguousStorage();
  const uint64_t numel = contiguous->numel();
  std::vector<float> out(static_cast<size_t>(numel), 0.0f);
  contiguous->CopyToHost(0, numel, out.data());
  return out;
}

void TestForwardShapeAndLookup() {
  transfomer_demo::modules::Embedding embedding(/*num_embeddings=*/6,
                                                /*embedding_dim=*/4);
  const std::vector<int64_t> indices{2, 1, 5, 0};
  const deeptiny::Shape shape{2, 2};

  deeptiny::Tensor output = embedding(indices, shape);
  Require(output.shape() == deeptiny::Shape({2, 2, 4}),
          "Forward output shape should be shape + {D}");
  Require(output.dtype() == deeptiny::DType::Float32,
          "Forward output dtype should match module dtype");

  const std::vector<float> out_data = ToVector(output);
  const std::vector<float> weight_data = ToVector(embedding.weight());

  constexpr uint64_t kEmbeddingDim = 4;
  for (uint64_t row = 0; row < indices.size(); ++row) {
    const uint64_t token =
        static_cast<uint64_t>(indices[static_cast<size_t>(row)]);
    for (uint64_t col = 0; col < kEmbeddingDim; ++col) {
      const float actual =
          out_data[static_cast<size_t>(row * kEmbeddingDim + col)];
      const float expected =
          weight_data[static_cast<size_t>(token * kEmbeddingDim + col)];
      if (!NearlyEqual(actual, expected)) {
        std::stringstream err;
        err << "Forward lookup mismatch at row " << row << ", col " << col
            << " (actual=" << actual << ", expected=" << expected << ")";
        throw std::runtime_error(err.str());
      }
    }
  }
}

void TestBackwardAccumulation() {
  transfomer_demo::modules::Embedding embedding(/*num_embeddings=*/5,
                                                /*embedding_dim=*/3);
  const std::vector<int64_t> indices{1, 3, 1, 1};
  const deeptiny::Shape shape{2, 2};

  auto output = embedding(indices, shape);
  auto loss = deeptiny::functional::Reduce(output, {0, 1, 2});
  loss.Backward();

  auto grad_opt = embedding.weight().grad();
  Require(grad_opt.has_value(),
          "weight.grad() should be present after backward");
  Require(grad_opt->shape() == deeptiny::Shape({5, 3}),
          "weight.grad() shape should match weight shape");

  const std::vector<float> grad_data = ToVector(*grad_opt);
  const std::vector<float> expected_row_counts{0.0f, 3.0f, 0.0f, 1.0f, 0.0f};

  constexpr uint64_t kEmbeddingDim = 3;
  for (uint64_t row = 0; row < expected_row_counts.size(); ++row) {
    for (uint64_t col = 0; col < kEmbeddingDim; ++col) {
      const float actual =
          grad_data[static_cast<size_t>(row * kEmbeddingDim + col)];
      const float expected = expected_row_counts[static_cast<size_t>(row)];
      if (!NearlyEqual(actual, expected)) {
        std::stringstream err;
        err << "Backward accumulation mismatch at row " << row << ", col "
            << col;
        throw std::runtime_error(err.str());
      }
    }
  }
}

void TestInvalidIndexGuard() {
  transfomer_demo::modules::Embedding embedding(/*num_embeddings=*/4,
                                                /*embedding_dim=*/2);

  bool threw = false;
  try {
    (void)embedding({0, 4}, {2});
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Require(threw, "Forward should throw for index >= num_embeddings");

  threw = false;
  try {
    (void)embedding({-1, 2}, {2});
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Require(threw, "Forward should throw for negative indices");
}

void TestShapeContractGuard() {
  transfomer_demo::modules::Embedding embedding(/*num_embeddings=*/4,
                                                /*embedding_dim=*/2);

  bool threw = false;
  try {
    (void)embedding({1, 2, 3}, {2, 2});
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Require(threw, "Forward should throw when indices size != product(shape)");
}

}  // namespace

int main() {
  try {
    TestForwardShapeAndLookup();
    TestBackwardAccumulation();
    TestInvalidIndexGuard();
    TestShapeContractGuard();
  } catch (const std::exception& err) {
    std::cerr << "transfomer_demo_embedding_test failed: " << err.what()
              << "\n";
    return 1;
  }

  std::cout << "transfomer_demo_embedding_test passed\n";
  return 0;
}
