#include "deeptiny/math.h"

#include <cassert>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

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

  void operator()(const Tensor& grad, Engine& engine) override {
    for (const auto& parent : getParents()) {
      assert(parent && "AddBackward parent must not be null");
      parent->updateGrad(grad, engine);
    }
  }
};

class SubBackward : public Function {
 public:
  SubBackward(const Tensor& a, const Tensor& b)
      : Function({utils::TensorAccessor::GetAutogradMeta(a),
                  utils::TensorAccessor::GetAutogradMeta(b)}) {}

  void operator()(const Tensor& grad, Engine& engine) override {
    const auto& parents = getParents();
    assert(parents.size() == 2 && "SubBackward must have exactly 2 parents");
    assert(parents[0] && "SubBackward first parent must not be null");
    assert(parents[1] && "SubBackward second parent must not be null");
    parents[0]->updateGrad(grad, engine);
    parents[1]->updateGrad(Negate(grad), engine);
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

  void operator()(const Tensor& grad, Engine& engine) override {
    Tensor saved_a =
        context().Get(static_cast<uint64_t>(ContextObjects::SAVED_A));
    Tensor saved_b =
        context().Get(static_cast<uint64_t>(ContextObjects::SAVED_B));
    const auto& parents = getParents();
    assert(parents.size() == 2 && "MulBackward must have exactly 2 parents");
    assert(parents[0] && "MulBackward first parent must not be null");
    assert(parents[1] && "MulBackward second parent must not be null");
    parents[0]->updateGrad(BinaryOut(BinaryOp::Mul, grad, saved_b), engine);
    parents[1]->updateGrad(BinaryOut(BinaryOp::Mul, grad, saved_a), engine);
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

  void operator()(const Tensor& grad, Engine& engine) override {
    Tensor saved_a =
        context().Get(static_cast<uint64_t>(ContextObjects::SAVED_A));
    Tensor saved_b =
        context().Get(static_cast<uint64_t>(ContextObjects::SAVED_B));
    const auto& parents = getParents();
    assert(parents.size() == 2 && "DivBackward must have exactly 2 parents");
    assert(parents[0] && "DivBackward first parent must not be null");
    assert(parents[1] && "DivBackward second parent must not be null");
    parents[0]->updateGrad(BinaryOut(BinaryOp::Div, grad, saved_b), engine);
    Tensor numerator = BinaryOut(BinaryOp::Mul, grad, saved_a);
    Tensor denominator = BinaryOut(BinaryOp::Mul, saved_b, saved_b);
    parents[1]->updateGrad(
        Negate(BinaryOut(BinaryOp::Div, numerator, denominator)), engine);
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
