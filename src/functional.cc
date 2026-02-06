#include "deeptiny/functional.h"

#include <optional>
#include <unordered_set>

#include "utils.h"

namespace deeptiny {

namespace functional {

namespace {
template <typename DimContainer>
Tensor ReduceImpl(const Tensor& x, const DimContainer& dims, bool keep_dims) {
  std::unordered_set<uint64_t> dims_set(dims.begin(), dims.end());
  const auto& x_shape = x.shape();
  const uint64_t rank = x_shape.size();

  Shape keep_shape;
  keep_shape.reserve(rank);
  for (uint64_t i = 0; i < rank; ++i) {
    keep_shape.push_back(dims_set.contains(i) ? 1 : x_shape[i]);
  }

  Shape reduced_shape;
  reduced_shape.reserve(rank);
  for (uint64_t i = 0; i < rank; ++i) {
    if (!dims_set.contains(i)) {
      reduced_shape.push_back(x_shape[i]);
    }
  }

  Tensor res = Tensor::Zeros(keep_shape, x.device(), x.dtype());
  auto recursive_reduce = [&res, &x, &dims_set](auto&& self, uint64_t dim_idx,
                                                std::vector<Slice>& slices) {
    if (slices.size() == x.shape().size()) {
      res += x(slices);
      return;
    }
    if (dims_set.contains(dim_idx)) {
      for (uint64_t i = 0; i < x.shape()[dim_idx]; ++i) {
        slices.push_back(Slice(i));
        self(self, dim_idx + 1, slices);
        slices.pop_back();
      }
    } else {
      slices.push_back(Slice(std::nullopt, std::nullopt));
      self(self, dim_idx + 1, slices);
      slices.pop_back();
    }
  };
  std::vector<Slice> slices;
  recursive_reduce(recursive_reduce, 0, slices);

  if (keep_dims || reduced_shape == keep_shape) {
    return res;
  }

  auto res_impl = utils::TensorAccessor::GetTensorImpl(res);
  Stride squeezed_stride;
  squeezed_stride.reserve(reduced_shape.size());
  for (uint64_t i = 0; i < rank; ++i) {
    if (!dims_set.contains(i)) {
      squeezed_stride.push_back(res_impl->stride()[i]);
    }
  }

  auto squeezed_impl = res_impl->View(
      Shape(reduced_shape), std::move(squeezed_stride), res_impl->offset());
  return utils::TensorAccessor::MakeTensor(
      squeezed_impl, utils::TensorAccessor::GetAutogradMeta(res));
}
}  // namespace

Tensor Reduce(const Tensor& x, const std::vector<uint64_t>& dims,
              bool keep_dims) {
  return ReduceImpl(x, dims, keep_dims);
}
}  // namespace functional

};  // namespace deeptiny
