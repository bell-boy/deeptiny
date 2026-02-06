#include "deeptiny/math.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "autograd_meta.h"
#include "cpu/kernels.h"
#include "deeptiny/functional.h"
#include "deeptiny/types.h"
#include "engine.h"
#include "utils.h"

namespace deeptiny {
namespace math {

namespace {
enum class BinaryOp { Add, Sub, Mul, Div };

std::string BinaryOpName(BinaryOp op) {
  switch (op) {
    case BinaryOp::Add:
      return "addition";
    case BinaryOp::Sub:
      return "subtraction";
    case BinaryOp::Mul:
      return "multiplication";
    case BinaryOp::Div:
      return "division";
  }
  throw std::runtime_error("Unknown binary op");
}

void DispatchBinaryOpKernel(BinaryOp op, const Tensor& a, const Tensor& b,
                            Tensor& out) {
  switch (out.device()) {
    case Device::CPU: {
      auto a_impl = utils::TensorAccessor::GetTensorImpl(a);
      auto b_impl = utils::TensorAccessor::GetTensorImpl(b);
      auto out_impl = utils::TensorAccessor::GetTensorImpl(out);
      switch (op) {
        case BinaryOp::Add:
          cpu::Add(a_impl, b_impl, out_impl);
          break;
        case BinaryOp::Sub:
          cpu::Sub(a_impl, b_impl, out_impl);
          break;
        case BinaryOp::Mul:
          cpu::Mul(a_impl, b_impl, out_impl);
          break;
        case BinaryOp::Div:
          cpu::Div(a_impl, b_impl, out_impl);
          break;
      }
      return;
    }
    default:
      std::stringstream err;
      err << "Operation does not support " << out.device().ToString();
      throw std::runtime_error(err.str());
  }
}

Tensor BinaryOut(BinaryOp op, const Tensor& a, const Tensor& b) {
  Tensor out(a.shape(), a.dtype(), a.device(), false);
  DispatchBinaryOpKernel(op, a, b, out);
  return out;
}

Tensor Negate(const Tensor& x) {
  Tensor zeros = functional::Zeros(x.shape(), x.device(), x.dtype());
  return BinaryOut(BinaryOp::Sub, zeros, x);
}

void ValidateInplaceDestination(const Tensor& dst, BinaryOp op) {
  const auto dst_impl = utils::TensorAccessor::GetTensorImpl(dst);
  for (const auto stride : dst_impl->stride()) {
    if (stride == 0) {
      std::stringstream err;
      err << "Cannot perform in-place " << BinaryOpName(op)
          << " on tensor with zero stride (broadcasted view).";
      throw std::runtime_error(err.str());
    }
  }
}

class AddBackward : public Function {
 public:
  AddBackward(const Tensor& a, const Tensor& b)
      : Function({utils::TensorAccessor::GetAutogradMeta(a),
                  utils::TensorAccessor::GetAutogradMeta(b)}) {}

  void operator()(const Tensor& grad) override {
    for (const auto& parent : getParents()) {
      assert(parent && "AddBackward parent must not be null");
      parent->updateGrad(grad);
    }
  }
};

class SubBackward : public Function {
 public:
  SubBackward(const Tensor& a, const Tensor& b)
      : Function({utils::TensorAccessor::GetAutogradMeta(a),
                  utils::TensorAccessor::GetAutogradMeta(b)}) {}

  void operator()(const Tensor& grad) override {
    const auto& parents = getParents();
    assert(parents.size() == 2 && "SubBackward must have exactly 2 parents");
    assert(parents[0] && "SubBackward first parent must not be null");
    assert(parents[1] && "SubBackward second parent must not be null");
    parents[0]->updateGrad(grad);
    parents[1]->updateGrad(Negate(grad));
  }
};

class MulBackward : public Function {
 public:
  enum struct ContextObjects : uint64_t {
    SAVED_A = 0,
    SAVED_B = 1,
  };

  MulBackward(const Tensor& parent_a, const Tensor& parent_b, Tensor saved_a,
              Tensor saved_b)
      : Function({utils::TensorAccessor::GetAutogradMeta(parent_a),
                  utils::TensorAccessor::GetAutogradMeta(parent_b)}) {
    context().Set(static_cast<uint64_t>(ContextObjects::SAVED_A), saved_a);
    context().Set(static_cast<uint64_t>(ContextObjects::SAVED_B), saved_b);
  }

  void operator()(const Tensor& grad) override {
    Tensor saved_a =
        context().Get(static_cast<uint64_t>(ContextObjects::SAVED_A));
    Tensor saved_b =
        context().Get(static_cast<uint64_t>(ContextObjects::SAVED_B));
    const auto& parents = getParents();
    assert(parents.size() == 2 && "MulBackward must have exactly 2 parents");
    assert(parents[0] && "MulBackward first parent must not be null");
    assert(parents[1] && "MulBackward second parent must not be null");
    parents[0]->updateGrad(BinaryOut(BinaryOp::Mul, grad, saved_b));
    parents[1]->updateGrad(BinaryOut(BinaryOp::Mul, grad, saved_a));
  }
};

class DivBackward : public Function {
 public:
  enum struct ContextObjects : uint64_t {
    SAVED_A = 0,
    SAVED_B = 1,
  };

  DivBackward(const Tensor& parent_a, const Tensor& parent_b, Tensor saved_a,
              Tensor saved_b)
      : Function({utils::TensorAccessor::GetAutogradMeta(parent_a),
                  utils::TensorAccessor::GetAutogradMeta(parent_b)}) {
    context().Set(static_cast<uint64_t>(ContextObjects::SAVED_A), saved_a);
    context().Set(static_cast<uint64_t>(ContextObjects::SAVED_B), saved_b);
  }

  void operator()(const Tensor& grad) override {
    Tensor saved_a =
        context().Get(static_cast<uint64_t>(ContextObjects::SAVED_A));
    Tensor saved_b =
        context().Get(static_cast<uint64_t>(ContextObjects::SAVED_B));
    const auto& parents = getParents();
    assert(parents.size() == 2 && "DivBackward must have exactly 2 parents");
    assert(parents[0] && "DivBackward first parent must not be null");
    assert(parents[1] && "DivBackward second parent must not be null");
    parents[0]->updateGrad(BinaryOut(BinaryOp::Div, grad, saved_b));
    Tensor numerator = BinaryOut(BinaryOp::Mul, grad, saved_a);
    Tensor denominator = BinaryOut(BinaryOp::Mul, saved_b, saved_b);
    parents[1]->updateGrad(
        Negate(BinaryOut(BinaryOp::Div, numerator, denominator)));
  }
};

std::shared_ptr<Function> MakeBackward(BinaryOp op, const Tensor& parent_a,
                                       const Tensor& parent_b,
                                       const Tensor& saved_a,
                                       const Tensor& saved_b) {
  switch (op) {
    case BinaryOp::Add:
      return std::make_shared<AddBackward>(parent_a, parent_b);
    case BinaryOp::Sub:
      return std::make_shared<SubBackward>(parent_a, parent_b);
    case BinaryOp::Mul:
      return std::make_shared<MulBackward>(parent_a, parent_b, saved_a,
                                           saved_b);
    case BinaryOp::Div:
      return std::make_shared<DivBackward>(parent_a, parent_b, saved_a,
                                           saved_b);
  }
  throw std::runtime_error("Unknown binary op");
}

Tensor BinaryOutOfPlace(BinaryOp op, const Tensor& a, const Tensor& b) {
  utils::CompatabilityCheck({a, b});
  auto [a_b, b_b] = utils::Broadcast(a, b);

  Tensor out = BinaryOut(op, a_b, b_b);
  auto backward = MakeBackward(op, a_b, b_b, a_b, b_b);
  auto out_impl = utils::TensorAccessor::GetTensorImpl(out);
  auto out_meta = std::make_shared<AutogradMeta>(backward);
  return utils::TensorAccessor::MakeTensor(out_impl, out_meta);
}

std::shared_ptr<AutogradMeta> BinaryInplace(BinaryOp op, Tensor& self,
                                            const Tensor& other) {
  ValidateInplaceDestination(self, op);
  utils::CompatabilityCheck({self, other});

  auto broadcasted_other = utils::BroadcastToShape(other, self.shape());
  if (!broadcasted_other) {
    std::stringstream err;
    err << "Could not broadcast tensor of shape " << FormatShape(other.shape())
        << " to shape " << FormatShape(self.shape());
    throw std::runtime_error(err.str());
  }

  Tensor lhs_parent = self;
  Tensor rhs_parent = *broadcasted_other;
  Tensor lhs_saved = self;
  Tensor rhs_saved = *broadcasted_other;
  if (op == BinaryOp::Mul || op == BinaryOp::Div) {
    lhs_saved = self.Clone();
    rhs_saved = broadcasted_other->Clone();
  }

  DispatchBinaryOpKernel(op, self, *broadcasted_other, self);

  auto backward =
      MakeBackward(op, lhs_parent, rhs_parent, lhs_saved, rhs_saved);
  return std::make_shared<AutogradMeta>(backward);
}

std::optional<Shape> BroadcastLeadingShape(const Shape& a, const Shape& b) {
  const size_t a_rank = a.size();
  const size_t b_rank = b.size();
  const size_t rank = std::max(a_rank, b_rank);
  Shape out(rank, 0);

  for (size_t i = 0; i < rank; ++i) {
    const bool has_a = i < a_rank;
    const bool has_b = i < b_rank;
    if (has_a && has_b) {
      const uint64_t a_dim = a[a_rank - i - 1];
      const uint64_t b_dim = b[b_rank - i - 1];
      if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
        return std::nullopt;
      }
      out[rank - i - 1] = std::max(a_dim, b_dim);
      continue;
    }
    if (has_a) {
      out[rank - i - 1] = a[a_rank - i - 1];
      continue;
    }
    out[rank - i - 1] = b[b_rank - i - 1];
  }
  return out;
}

Tensor ReduceGradientToShape(const Tensor& grad, const Shape& target_shape) {
  const auto& grad_shape = grad.shape();
  if (grad_shape == target_shape) {
    return grad;
  }

  if (grad_shape.size() < target_shape.size()) {
    std::stringstream err;
    err << "Cannot reduce gradient of shape " << FormatShape(grad_shape)
        << " to shape " << FormatShape(target_shape);
    throw std::runtime_error(err.str());
  }

  const size_t rank_diff = grad_shape.size() - target_shape.size();
  std::vector<uint64_t> reduce_dims;
  reduce_dims.reserve(grad_shape.size());
  for (size_t i = 0; i < grad_shape.size(); ++i) {
    if (i < rank_diff) {
      reduce_dims.push_back(static_cast<uint64_t>(i));
      continue;
    }
    const uint64_t grad_dim = grad_shape[i];
    const uint64_t target_dim = target_shape[i - rank_diff];
    if (grad_dim == target_dim) {
      continue;
    }
    if (target_dim == 1) {
      reduce_dims.push_back(static_cast<uint64_t>(i));
      continue;
    }
    std::stringstream err;
    err << "Cannot reduce gradient of shape " << FormatShape(grad_shape)
        << " to shape " << FormatShape(target_shape);
    throw std::runtime_error(err.str());
  }

  Tensor reduced =
      reduce_dims.empty() ? grad : functional::Reduce(grad, reduce_dims, true);
  if (rank_diff == 0) {
    return reduced;
  }

  std::vector<uint64_t> squeeze_dims(rank_diff, 0);
  for (size_t i = 0; i < rank_diff; ++i) {
    squeeze_dims[i] = static_cast<uint64_t>(i);
  }
  return reduced.Squeeze(squeeze_dims);
}

void DispatchBatchedMatMulKernel(const Tensor& a, const Tensor& b,
                                 Tensor& out) {
  switch (out.device()) {
    case Device::CPU: {
      auto a_impl = utils::TensorAccessor::GetTensorImpl(a);
      auto b_impl = utils::TensorAccessor::GetTensorImpl(b);
      auto out_impl = utils::TensorAccessor::GetTensorImpl(out);
      cpu::BatchedMatMul(a_impl, b_impl, out_impl);
      return;
    }
    default: {
      std::stringstream err;
      err << "Operation does not support " << out.device().ToString();
      throw std::runtime_error(err.str());
    }
  }
}

void DispatchBatchedMatMulGradAKernel(const Tensor& grad_out, const Tensor& b,
                                      Tensor& grad_a) {
  switch (grad_a.device()) {
    case Device::CPU: {
      auto grad_out_impl = utils::TensorAccessor::GetTensorImpl(grad_out);
      auto b_impl = utils::TensorAccessor::GetTensorImpl(b);
      auto grad_a_impl = utils::TensorAccessor::GetTensorImpl(grad_a);
      cpu::BatchedMatMulGradA(grad_out_impl, b_impl, grad_a_impl);
      return;
    }
    default: {
      std::stringstream err;
      err << "Operation does not support " << grad_a.device().ToString();
      throw std::runtime_error(err.str());
    }
  }
}

void DispatchBatchedMatMulGradBKernel(const Tensor& a, const Tensor& grad_out,
                                      Tensor& grad_b) {
  switch (grad_b.device()) {
    case Device::CPU: {
      auto a_impl = utils::TensorAccessor::GetTensorImpl(a);
      auto grad_out_impl = utils::TensorAccessor::GetTensorImpl(grad_out);
      auto grad_b_impl = utils::TensorAccessor::GetTensorImpl(grad_b);
      cpu::BatchedMatMulGradB(a_impl, grad_out_impl, grad_b_impl);
      return;
    }
    default: {
      std::stringstream err;
      err << "Operation does not support " << grad_b.device().ToString();
      throw std::runtime_error(err.str());
    }
  }
}

class BatchedMatMulBackward : public Function {
 public:
  enum struct ContextObjects : uint64_t {
    SAVED_A = 0,
    SAVED_B = 1,
  };

  BatchedMatMulBackward(const Tensor& parent_a, const Tensor& parent_b,
                        Tensor saved_a, Tensor saved_b)
      : Function({utils::TensorAccessor::GetAutogradMeta(parent_a),
                  utils::TensorAccessor::GetAutogradMeta(parent_b)}),
        parent_a_shape_(parent_a.shape()),
        parent_b_shape_(parent_b.shape()) {
    context().Set(static_cast<uint64_t>(ContextObjects::SAVED_A), saved_a);
    context().Set(static_cast<uint64_t>(ContextObjects::SAVED_B), saved_b);
  }

  void operator()(const Tensor& grad) override {
    Tensor saved_a =
        context().Get(static_cast<uint64_t>(ContextObjects::SAVED_A));
    Tensor saved_b =
        context().Get(static_cast<uint64_t>(ContextObjects::SAVED_B));

    Tensor grad_a(saved_a.shape(), grad.dtype(), grad.device(), false);
    Tensor grad_b(saved_b.shape(), grad.dtype(), grad.device(), false);
    DispatchBatchedMatMulGradAKernel(grad, saved_b, grad_a);
    DispatchBatchedMatMulGradBKernel(saved_a, grad, grad_b);

    const auto& parents = getParents();
    assert(parents.size() == 2 &&
           "BatchedMatMulBackward must have exactly 2 parents");
    assert(parents[0] && "BatchedMatMulBackward first parent must not be null");
    assert(parents[1] &&
           "BatchedMatMulBackward second parent must not be null");
    parents[0]->updateGrad(ReduceGradientToShape(grad_a, parent_a_shape_));
    parents[1]->updateGrad(ReduceGradientToShape(grad_b, parent_b_shape_));
  }

 private:
  Shape parent_a_shape_;
  Shape parent_b_shape_;
};

Tensor BatchedMatMulOutOfPlace(const Tensor& a, const Tensor& b) {
  utils::CompatabilityCheck({a, b});
  if (a.shape().size() < 3 || b.shape().size() < 3) {
    throw std::runtime_error("BatchedMatMul requires rank >= 3 inputs");
  }

  const auto& a_shape = a.shape();
  const auto& b_shape = b.shape();
  const uint64_t n = a_shape[a_shape.size() - 2];
  const uint64_t k = a_shape[a_shape.size() - 1];
  const uint64_t rhs_k = b_shape[b_shape.size() - 2];
  const uint64_t m = b_shape[b_shape.size() - 1];
  if (k != rhs_k) {
    throw std::runtime_error(
        "BatchedMatMul requires matching inner dimensions");
  }

  Shape a_batch(a_shape.begin(), a_shape.end() - 2);
  Shape b_batch(b_shape.begin(), b_shape.end() - 2);
  auto out_batch = BroadcastLeadingShape(a_batch, b_batch);
  if (!out_batch) {
    std::stringstream err;
    err << "Cannot broadcast batch dimensions from shapes "
        << FormatShape(a_shape) << " and " << FormatShape(b_shape) << ".";
    throw std::runtime_error(err.str());
  }

  Shape a_target = *out_batch;
  a_target.push_back(n);
  a_target.push_back(k);
  Shape b_target = *out_batch;
  b_target.push_back(k);
  b_target.push_back(m);

  auto a_broadcast = utils::BroadcastToShape(a, a_target);
  auto b_broadcast = utils::BroadcastToShape(b, b_target);
  if (!a_broadcast || !b_broadcast) {
    std::stringstream err;
    err << "Failed to broadcast BatchedMatMul inputs to shapes "
        << FormatShape(a_target) << " and " << FormatShape(b_target);
    throw std::runtime_error(err.str());
  }

  Shape out_shape = *out_batch;
  out_shape.push_back(n);
  out_shape.push_back(m);
  Tensor out(out_shape, a.dtype(), a.device(), false);
  DispatchBatchedMatMulKernel(*a_broadcast, *b_broadcast, out);

  auto backward =
      std::make_shared<BatchedMatMulBackward>(a, b, *a_broadcast, *b_broadcast);
  auto out_impl = utils::TensorAccessor::GetTensorImpl(out);
  auto out_meta = std::make_shared<AutogradMeta>(backward);
  return utils::TensorAccessor::MakeTensor(out_impl, out_meta);
}
}  // namespace

Tensor operator+(const Tensor& a, const Tensor& b) {
  return BinaryOutOfPlace(BinaryOp::Add, a, b);
}

Tensor operator-(const Tensor& a, const Tensor& b) {
  return BinaryOutOfPlace(BinaryOp::Sub, a, b);
}

Tensor operator*(const Tensor& a, const Tensor& b) {
  return BinaryOutOfPlace(BinaryOp::Mul, a, b);
}

Tensor operator/(const Tensor& a, const Tensor& b) {
  return BinaryOutOfPlace(BinaryOp::Div, a, b);
}

Tensor BatchedMatMul(const Tensor& a, const Tensor& b) {
  return BatchedMatMulOutOfPlace(a, b);
}

std::shared_ptr<AutogradMeta> InplaceAdd(Tensor& self, const Tensor& other) {
  return BinaryInplace(BinaryOp::Add, self, other);
}

std::shared_ptr<AutogradMeta> InplaceSub(Tensor& self, const Tensor& other) {
  return BinaryInplace(BinaryOp::Sub, self, other);
}

std::shared_ptr<AutogradMeta> InplaceMul(Tensor& self, const Tensor& other) {
  return BinaryInplace(BinaryOp::Mul, self, other);
}

std::shared_ptr<AutogradMeta> InplaceDiv(Tensor& self, const Tensor& other) {
  return BinaryInplace(BinaryOp::Div, self, other);
}

}  // namespace math

void Tensor::operator+=(const Tensor& other) {
  autograd_meta_ = math::InplaceAdd(*this, other);
}

void Tensor::operator-=(const Tensor& other) {
  autograd_meta_ = math::InplaceSub(*this, other);
}

void Tensor::operator*=(const Tensor& other) {
  autograd_meta_ = math::InplaceMul(*this, other);
}

void Tensor::operator/=(const Tensor& other) {
  autograd_meta_ = math::InplaceDiv(*this, other);
}

}  // namespace deeptiny
