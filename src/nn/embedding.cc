#include "deeptiny/nn/embedding.h"

#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "nn/validation.h"

namespace deeptiny::nn {
namespace {

int64_t ToSliceIndex(uint64_t value, const char* label) {
  if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    std::stringstream err;
    err << label << " is too large for Slice indexing.";
    throw std::runtime_error(err.str());
  }
  return static_cast<int64_t>(value);
}

Tensor MakeWeight(uint64_t num_embeddings, uint64_t embedding_dim, DType dtype,
                  Device device, bool requires_grad) {
  detail::ValidateNonZeroDimension("Embedding", "num_embeddings",
                                   num_embeddings);
  detail::ValidateNonZeroDimension("Embedding", "embedding_dim", embedding_dim);
  if (dtype != DType::Float32) {
    throw std::runtime_error(
        "Embedding currently supports only Float32 dtype.");
  }
  if (device != Device::CPU) {
    throw std::runtime_error("Embedding currently supports only CPU.");
  }

  return Tensor::CreateUniform({num_embeddings, embedding_dim}, device, dtype,
                               requires_grad);
}

}  // namespace

Embedding::Embedding(uint64_t num_embeddings, uint64_t embedding_dim,
                     DType dtype, Device device, bool requires_grad)
    : num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim),
      dtype_(dtype),
      device_(device),
      weight_(MakeWeight(num_embeddings, embedding_dim, dtype, device,
                         requires_grad)) {
  RegisterParameter(weight_);
}

Tensor Embedding::operator()(const std::vector<int64_t>& indices,
                             const Shape& shape) const {
  const uint64_t index_count = static_cast<uint64_t>(indices.size());
  Shape output_shape = shape;
  output_shape.push_back(embedding_dim_);
  Tensor flat_output =
      Tensor::Zeros({index_count, embedding_dim_}, device_, dtype_);

  try {
    (void)flat_output.Reshape(output_shape);
  } catch (const std::runtime_error&) {
    throw std::runtime_error(
        "Embedding::operator() expected indices.size() == product(shape).");
  }

  const int64_t end_col = ToSliceIndex(embedding_dim_, "embedding_dim");
  for (uint64_t i = 0; i < index_count; ++i) {
    const int64_t token = indices[static_cast<size_t>(i)];
    if (token < 0 || static_cast<uint64_t>(token) >= num_embeddings_) {
      std::stringstream err;
      err << "Embedding::operator() received out-of-range index at position "
          << i << ": " << token;
      throw std::runtime_error(err.str());
    }

    const int64_t row = ToSliceIndex(i, "row index");
    Tensor gathered_row = weight_({Slice(token), Slice(0, end_col)});
    flat_output({Slice(row), Slice(0, end_col)}) = gathered_row;
  }

  return flat_output.Reshape(output_shape);
}

Tensor& Embedding::weight() { return weight_; }

const Tensor& Embedding::weight() const { return weight_; }

}  // namespace deeptiny::nn
