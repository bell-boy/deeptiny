#include "modules/embedding.h"

#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace transfomer_demo::modules {
namespace {

uint64_t Product(const deeptiny::Shape& shape) {
  uint64_t total = 1;
  for (const uint64_t dim : shape) {
    if (dim == 0) {
      return 0;
    }
    if (total > std::numeric_limits<uint64_t>::max() / dim) {
      throw std::runtime_error("Shape product overflowed uint64_t.");
    }
    total *= dim;
  }
  return total;
}

int64_t ToSliceIndex(uint64_t value, const char* label) {
  if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    std::stringstream err;
    err << label << " is too large for Slice indexing.";
    throw std::runtime_error(err.str());
  }
  return static_cast<int64_t>(value);
}

template <typename TensorType>
auto CreateUniformWithRequiresGrad(const deeptiny::Shape& shape,
                                   deeptiny::Device device,
                                   deeptiny::DType dtype, bool requires_grad,
                                   int)
    -> decltype(TensorType::CreateUniform(shape, device, dtype,
                                          requires_grad)) {
  return TensorType::CreateUniform(shape, device, dtype, requires_grad);
}

template <typename TensorType>
deeptiny::Tensor CreateUniformWithRequiresGrad(const deeptiny::Shape& shape,
                                               deeptiny::Device device,
                                               deeptiny::DType dtype,
                                               bool requires_grad, long) {
  deeptiny::Tensor sampled = TensorType::CreateUniform(shape, device, dtype);
  if (!requires_grad) {
    return sampled;
  }

  deeptiny::Tensor tracked(shape, dtype, device, true);
  std::vector<deeptiny::Slice> slices;
  slices.reserve(shape.size());
  for (const uint64_t dim : shape) {
    slices.emplace_back(0, ToSliceIndex(dim, "shape dimension"));
  }
  tracked(slices) = sampled;
  return tracked;
}

deeptiny::Tensor MakeWeight(uint64_t num_embeddings, uint64_t embedding_dim,
                            deeptiny::DType dtype, deeptiny::Device device,
                            bool requires_grad) {
  if (num_embeddings == 0 || embedding_dim == 0) {
    throw std::runtime_error("Embedding dimensions must be positive.");
  }
  if (dtype != deeptiny::DType::Float32) {
    throw std::runtime_error(
        "Demo Embedding currently supports only Float32 dtype.");
  }
  if (device != deeptiny::Device::CPU) {
    throw std::runtime_error("Demo Embedding currently supports only CPU.");
  }

  const deeptiny::Shape weight_shape{num_embeddings, embedding_dim};
  return CreateUniformWithRequiresGrad<deeptiny::Tensor>(
      weight_shape, device, dtype, requires_grad, 0);
}

}  // namespace

Embedding::Embedding(uint64_t num_embeddings, uint64_t embedding_dim,
                     deeptiny::DType dtype, deeptiny::Device device,
                     bool requires_grad)
    : num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim),
      dtype_(dtype),
      device_(device),
      weight_(MakeWeight(num_embeddings, embedding_dim, dtype, device,
                         requires_grad)) {}

deeptiny::Tensor Embedding::operator()(const std::vector<int64_t>& indices,
                                       const deeptiny::Shape& shape) const {
  const uint64_t expected_count = Product(shape);
  if (expected_count != static_cast<uint64_t>(indices.size())) {
    std::stringstream err;
    err << "Embedding::operator() expected indices.size() == product(shape), "
           "got "
        << indices.size() << " and " << expected_count;
    throw std::runtime_error(err.str());
  }

  deeptiny::Shape output_shape = shape;
  output_shape.push_back(embedding_dim_);
  deeptiny::Tensor output =
      deeptiny::Tensor::Zeros(output_shape, device_, dtype_);
  deeptiny::Tensor flat_output =
      output.Reshape({expected_count, embedding_dim_});

  const int64_t end_col = ToSliceIndex(embedding_dim_, "embedding_dim");
  for (uint64_t i = 0; i < expected_count; ++i) {
    const int64_t token = indices[static_cast<size_t>(i)];
    if (token < 0 || static_cast<uint64_t>(token) >= num_embeddings_) {
      std::stringstream err;
      err << "Embedding::operator() received out-of-range index at position "
          << i << ": " << token;
      throw std::runtime_error(err.str());
    }

    const int64_t row = ToSliceIndex(i, "row index");
    deeptiny::Tensor gathered_row =
        weight_({deeptiny::Slice(token), deeptiny::Slice(0, end_col)});
    flat_output({deeptiny::Slice(row), deeptiny::Slice(0, end_col)}) =
        gathered_row;
  }

  return flat_output.Reshape(output_shape);
}

}  // namespace transfomer_demo::modules
