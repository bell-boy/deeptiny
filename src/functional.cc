#include "deeptiny/functional.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "autograd_meta.h"
#include "dispatch/dispatch.h"
#include "engine.h"
#include "utils.h"

namespace deeptiny {

namespace functional {

namespace {
class ReLUBackward : public Function {
 public:
  enum struct ContextObjects : uint64_t {
    SAVED_X = 0,
  };

  explicit ReLUBackward(const Tensor& parent_x, const Tensor& saved_x)
      : Function({utils::TensorAccessor::GetAutogradMeta(parent_x)}) {
    context().Set(static_cast<uint64_t>(ContextObjects::SAVED_X), saved_x);
  }

  void operator()(const Tensor& grad) override {
    Tensor saved_x =
        context().Get(static_cast<uint64_t>(ContextObjects::SAVED_X));
    const auto& parents = getParents();
    assert(parents.size() == 1 && "ReLUBackward must have exactly 1 parent");
    assert(parents[0] && "ReLUBackward parent must not be null");

    if (grad.shape() != saved_x.shape()) {
      throw std::runtime_error("ReLUBackward received invalid grad shape");
    }

    auto grad_x_impl = dispatch::relu::Backward(saved_x, grad);
    parents[0]->updateGrad(grad_x_impl);
  }
};

class SqrtBackward : public Function {
 public:
  enum struct ContextObjects : uint64_t {
    SAVED_X = 0,
  };

  explicit SqrtBackward(const Tensor& parent_x, const Tensor& saved_x)
      : Function({utils::TensorAccessor::GetAutogradMeta(parent_x)}) {
    context().Set(static_cast<uint64_t>(ContextObjects::SAVED_X), saved_x);
  }

  void operator()(const Tensor& grad) override {
    Tensor saved_x =
        context().Get(static_cast<uint64_t>(ContextObjects::SAVED_X));
    const auto& parents = getParents();
    assert(parents.size() == 1 && "SqrtBackward must have exactly 1 parent");
    assert(parents[0] && "SqrtBackward parent must not be null");

    if (grad.shape() != saved_x.shape()) {
      throw std::runtime_error("SqrtBackward received invalid grad shape");
    }

    auto grad_x_impl = dispatch::sqrt::Backward(saved_x, grad);
    parents[0]->updateGrad(grad_x_impl);
  }
};

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

Tensor ReLU(const Tensor& x) {
  auto out_impl = dispatch::relu::OutOfPlace(x);

  auto backward = std::make_shared<ReLUBackward>(x, x);
  auto out_meta = std::make_shared<AutogradMeta>(backward);
  return utils::TensorAccessor::MakeTensor(out_impl, out_meta);
}

Tensor Sqrt(const Tensor& x) {
  auto out_impl = dispatch::sqrt::OutOfPlace(x);

  auto backward = std::make_shared<SqrtBackward>(x, x);
  auto out_meta = std::make_shared<AutogradMeta>(backward);
  return utils::TensorAccessor::MakeTensor(out_impl, out_meta);
}
}  // namespace functional

};  // namespace deeptiny
