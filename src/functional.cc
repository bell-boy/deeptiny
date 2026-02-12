#include "deeptiny/functional.h"

#include <cassert>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>

#include "autograd_meta.h"
#include "dispatch/dispatch.h"
#include "dispatch/reduce.h"
#include "engine.h"
#include "tensor_impl.h"
#include "utils.h"

namespace deeptiny {

namespace functional {

namespace {
using ReduceDimsLookup = utils::UInt64IdentityMap<bool>;

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

class SiLUBackward : public Function {
 public:
  enum struct ContextObjects : uint64_t {
    SAVED_X = 0,
  };

  explicit SiLUBackward(const Tensor& parent_x, const Tensor& saved_x)
      : Function({utils::TensorAccessor::GetAutogradMeta(parent_x)}) {
    context().Set(static_cast<uint64_t>(ContextObjects::SAVED_X), saved_x);
  }

  void operator()(const Tensor& grad) override {
    Tensor saved_x =
        context().Get(static_cast<uint64_t>(ContextObjects::SAVED_X));
    const auto& parents = getParents();
    assert(parents.size() == 1 && "SiLUBackward must have exactly 1 parent");
    assert(parents[0] && "SiLUBackward parent must not be null");

    if (grad.shape() != saved_x.shape()) {
      throw std::runtime_error("SiLUBackward received invalid grad shape");
    }

    auto grad_x_impl = dispatch::silu::Backward(saved_x, grad);
    parents[0]->updateGrad(grad_x_impl);
  }
};

class SoftmaxBackward : public Function {
 public:
  enum struct ContextObjects : uint64_t {
    SAVED_Y = 0,
  };

  SoftmaxBackward(const Tensor& parent_x, const Tensor& saved_y, uint64_t dim)
      : Function({utils::TensorAccessor::GetAutogradMeta(parent_x)}),
        dim_(dim) {
    context().Set(static_cast<uint64_t>(ContextObjects::SAVED_Y), saved_y);
  }

  void operator()(const Tensor& grad) override {
    Tensor saved_y =
        context().Get(static_cast<uint64_t>(ContextObjects::SAVED_Y));
    const auto& parents = getParents();
    assert(parents.size() == 1 && "SoftmaxBackward must have exactly 1 parent");
    assert(parents[0] && "SoftmaxBackward parent must not be null");

    if (grad.shape() != saved_y.shape()) {
      throw std::runtime_error("SoftmaxBackward received invalid grad shape");
    }

    auto grad_x_impl = dispatch::softmax::Backward(saved_y, grad, dim_);
    parents[0]->updateGrad(grad_x_impl);
  }

 private:
  uint64_t dim_;
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

class ReduceBackward : public Function {
  ReduceDimsLookup dims_;
  Shape original_shape_;
  bool keep_dims_;

 public:
  ReduceBackward(const Tensor& parent, ReduceDimsLookup dims, bool keep_dims)
      : Function({utils::TensorAccessor::GetAutogradMeta(parent)}),
        dims_(std::move(dims)),
        original_shape_(parent.shape()),
        keep_dims_(keep_dims) {}
  void operator()(const Tensor& grad) override {
    Shape unsqueezed_shape;
    Stride unsqueezed_stride;
    uint64_t next_stride = 0;
    auto old_impl = utils::TensorAccessor::GetTensorImpl(grad);
    for (uint64_t i = 0; i < original_shape_.size(); ++i) {
      unsqueezed_shape.push_back(original_shape_[i]);
      if (dims_.contains(i)) {
        unsqueezed_stride.push_back(0);
        if (keep_dims_) next_stride++;
      } else {
        assert(grad.shape()[next_stride] == original_shape_[i]);
        unsqueezed_stride.push_back(old_impl->stride()[next_stride]);
        next_stride++;
      }
    }
    auto new_impl =
        std::make_shared<TensorImpl>(unsqueezed_shape, unsqueezed_stride,
                                     old_impl->offset(), old_impl->storage());
    getParents()[0]->updateGrad(new_impl);
  }
};

}  // namespace

Tensor Reduce(const Tensor& x, const std::vector<uint64_t>& dims,
              bool keep_dims) {
  ReduceDimsLookup dims_lookup;
  dims_lookup.reserve(dims.size());
  for (const auto dim : dims) {
    dims_lookup.emplace(dim, true);
  }
  auto reduce_impl = dispatch::reduce::OutOfPlace(x, dims, keep_dims);
  auto reduce_meta = std::make_shared<AutogradMeta>(
      std::make_shared<ReduceBackward>(x, std::move(dims_lookup), keep_dims),
      true);
  return utils::TensorAccessor::MakeTensor(reduce_impl, reduce_meta);
}

Tensor ReLU(const Tensor& x) {
  auto out_impl = dispatch::relu::OutOfPlace(x);

  auto backward = std::make_shared<ReLUBackward>(x, x);
  auto out_meta = std::make_shared<AutogradMeta>(backward);
  return utils::TensorAccessor::MakeTensor(out_impl, out_meta);
}

Tensor SiLU(const Tensor& x) {
  auto out_impl = dispatch::silu::OutOfPlace(x);

  auto backward = std::make_shared<SiLUBackward>(x, x);
  auto out_meta = std::make_shared<AutogradMeta>(backward);
  return utils::TensorAccessor::MakeTensor(out_impl, out_meta);
}

Tensor Softmax(const Tensor& x, uint64_t dim) {
  const auto& shape = x.shape();
  if (shape.empty()) {
    throw std::runtime_error("Softmax does not support scalar input");
  }
  if (dim >= shape.size()) {
    throw std::runtime_error("Softmax dim out of range");
  }

  auto out_impl = dispatch::softmax::OutOfPlace(x, dim);
  Tensor out_saved = utils::TensorAccessor::MakeTensor(out_impl, nullptr);
  auto backward = std::make_shared<SoftmaxBackward>(x, out_saved, dim);
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
