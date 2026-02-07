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
#include "dispatch/dispatch.h"
#include "deeptiny/types.h"
#include "engine.h"
#include "utils.h"

namespace deeptiny {
namespace math {

namespace {
std::string BinaryOpName(dispatch::binary::Op op) {
  switch (op) {
    case dispatch::binary::Op::Add:
      return "addition";
    case dispatch::binary::Op::Sub:
      return "subtraction";
    case dispatch::binary::Op::Mul:
      return "multiplication";
    case dispatch::binary::Op::Div:
      return "division";
  }
  throw std::runtime_error("Unknown binary op");
}

Tensor Negate(const Tensor& x) {
  Tensor zeros = Tensor::Zeros(x.shape(), x.device(), x.dtype());
  auto neg_impl = dispatch::binary::OutOfPlace(dispatch::binary::Op::Sub, zeros, x);
  return neg_impl;
}

void ValidateInplaceDestination(const Tensor& dst, dispatch::binary::Op op) {
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
    auto grad_a_impl =
        dispatch::binary::OutOfPlace(dispatch::binary::Op::Mul, grad, saved_b);
    parents[0]->updateGrad(grad_a_impl);
    auto grad_b_impl =
        dispatch::binary::OutOfPlace(dispatch::binary::Op::Mul, grad, saved_a);
    parents[1]->updateGrad(grad_b_impl);
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
    auto grad_a_impl =
        dispatch::binary::OutOfPlace(dispatch::binary::Op::Div, grad, saved_b);
    parents[0]->updateGrad(grad_a_impl);

    auto numerator_impl =
        dispatch::binary::OutOfPlace(dispatch::binary::Op::Mul, grad, saved_a);
    auto denominator_impl = dispatch::binary::OutOfPlace(
        dispatch::binary::Op::Mul, saved_b, saved_b);
    auto grad_b_impl =
        dispatch::binary::OutOfPlace(dispatch::binary::Op::Div, numerator_impl,
                                     denominator_impl);
    parents[1]->updateGrad(Negate(grad_b_impl));
  }
};

std::shared_ptr<Function> MakeBackward(dispatch::binary::Op op, const Tensor& parent_a,
                                       const Tensor& parent_b,
                                       const Tensor& saved_a,
                                       const Tensor& saved_b) {
  switch (op) {
    case dispatch::binary::Op::Add:
      return std::make_shared<AddBackward>(parent_a, parent_b);
    case dispatch::binary::Op::Sub:
      return std::make_shared<SubBackward>(parent_a, parent_b);
    case dispatch::binary::Op::Mul:
      return std::make_shared<MulBackward>(parent_a, parent_b, saved_a,
                                           saved_b);
    case dispatch::binary::Op::Div:
      return std::make_shared<DivBackward>(parent_a, parent_b, saved_a,
                                           saved_b);
  }
  throw std::runtime_error("Unknown binary op");
}

Tensor BinaryOutOfPlace(dispatch::binary::Op op, const Tensor& a, const Tensor& b) {
  utils::CompatabilityCheck({a, b});
  auto [a_b, b_b] = utils::Broadcast(a, b);

  auto out_impl = dispatch::binary::OutOfPlace(op, a_b, b_b);
  auto backward = MakeBackward(op, a_b, b_b, a_b, b_b);
  auto out_meta = std::make_shared<AutogradMeta>(backward);
  return utils::TensorAccessor::MakeTensor(out_impl, out_meta);
}

std::shared_ptr<AutogradMeta> BinaryInplace(dispatch::binary::Op op, Tensor& self,
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
  if (op == dispatch::binary::Op::Mul || op == dispatch::binary::Op::Div) {
    lhs_saved = self.Clone();
    rhs_saved = broadcasted_other->Clone();
  }

  dispatch::binary::Inplace(op, self, *broadcasted_other);

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

class BatchedMatMulBackward : public Function {
 public:
  enum struct ContextObjects : uint64_t {
    SAVED_A = 0,
    SAVED_B = 1,
  };

  BatchedMatMulBackward(const Tensor& saved_a, const Tensor& saved_b,
                        bool transpose_a, bool transpose_b)
      : Function({utils::TensorAccessor::GetAutogradMeta(saved_a),
                  utils::TensorAccessor::GetAutogradMeta(saved_b)}),
        transpose_a_(transpose_a),
        transpose_b_(transpose_b) {
    context().Set(static_cast<uint64_t>(ContextObjects::SAVED_A), saved_a);
    context().Set(static_cast<uint64_t>(ContextObjects::SAVED_B), saved_b);
  }

  void operator()(const Tensor& grad) override {
    Tensor saved_a =
        context().Get(static_cast<uint64_t>(ContextObjects::SAVED_A));
    Tensor saved_b =
        context().Get(static_cast<uint64_t>(ContextObjects::SAVED_B));

    const auto& parents = getParents();
    assert(parents.size() == 2 &&
           "BatchedMatMulBackward must have exactly 2 parents");
    assert(parents[0] && "BatchedMatMulBackward first parent must not be null");
    assert(parents[1] &&
           "BatchedMatMulBackward second parent must not be null");

    if (!transpose_a_) {
      parents[0]->updateGrad(
          dispatch::batched_matmul::OutOfPlace(grad, saved_b, false,
                                               !transpose_b_));
    } else {
      parents[0]->updateGrad(
          dispatch::batched_matmul::OutOfPlace(saved_b, grad, transpose_b_,
                                               true));
    }

    if (!transpose_b_) {
      parents[1]->updateGrad(
          dispatch::batched_matmul::OutOfPlace(saved_a, grad, !transpose_a_,
                                               false));
    } else {
      parents[1]->updateGrad(
          dispatch::batched_matmul::OutOfPlace(grad, saved_a, true,
                                               transpose_a_));
    }
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
};

}  // namespace

Tensor BatchedMatMul(const Tensor& a, const Tensor& b, bool transpose_a,
                     bool transpose_b) {
  utils::CompatabilityCheck({a, b});
  if (a.shape().size() < 3 || b.shape().size() < 3) {
    throw std::runtime_error("BatchedMatMul requires rank >= 3 inputs");
  }

  const auto& a_shape = a.shape();
  const auto& b_shape = b.shape();
  const uint64_t a_rows = a_shape[a_shape.size() - 2];
  const uint64_t a_cols = a_shape[a_shape.size() - 1];
  const uint64_t b_rows = b_shape[b_shape.size() - 2];
  const uint64_t b_cols = b_shape[b_shape.size() - 1];

  const uint64_t lhs_cols = transpose_a ? a_rows : a_cols;
  const uint64_t rhs_rows = transpose_b ? b_cols : b_rows;
  if (lhs_cols != rhs_rows) {
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
  a_target.push_back(a_rows);
  a_target.push_back(a_cols);
  Shape b_target = *out_batch;
  b_target.push_back(b_rows);
  b_target.push_back(b_cols);

  auto a_broadcast = utils::BroadcastToShape(a, a_target);
  auto b_broadcast = utils::BroadcastToShape(b, b_target);
  if (!a_broadcast || !b_broadcast) {
    std::stringstream err;
    err << "Failed to broadcast BatchedMatMul inputs to shapes "
        << FormatShape(a_target) << " and " << FormatShape(b_target);
    throw std::runtime_error(err.str());
  }

  auto out_impl = dispatch::batched_matmul::OutOfPlace(*a_broadcast,
                                                       *b_broadcast, transpose_a,
                                                       transpose_b);

  auto backward = std::make_shared<BatchedMatMulBackward>(
      *a_broadcast, *b_broadcast, transpose_a, transpose_b);
  auto out_meta = std::make_shared<AutogradMeta>(backward);
  return utils::TensorAccessor::MakeTensor(out_impl, out_meta);
}

Tensor operator+(const Tensor& a, const Tensor& b) {
  return BinaryOutOfPlace(dispatch::binary::Op::Add, a, b);
}

Tensor operator-(const Tensor& a, const Tensor& b) {
  return BinaryOutOfPlace(dispatch::binary::Op::Sub, a, b);
}

Tensor operator*(const Tensor& a, const Tensor& b) {
  return BinaryOutOfPlace(dispatch::binary::Op::Mul, a, b);
}

Tensor operator/(const Tensor& a, const Tensor& b) {
  return BinaryOutOfPlace(dispatch::binary::Op::Div, a, b);
}

std::shared_ptr<AutogradMeta> InplaceAdd(Tensor& self, const Tensor& other) {
  return BinaryInplace(dispatch::binary::Op::Add, self, other);
}

std::shared_ptr<AutogradMeta> InplaceSub(Tensor& self, const Tensor& other) {
  return BinaryInplace(dispatch::binary::Op::Sub, self, other);
}

std::shared_ptr<AutogradMeta> InplaceMul(Tensor& self, const Tensor& other) {
  return BinaryInplace(dispatch::binary::Op::Mul, self, other);
}

std::shared_ptr<AutogradMeta> InplaceDiv(Tensor& self, const Tensor& other) {
  return BinaryInplace(dispatch::binary::Op::Div, self, other);
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
