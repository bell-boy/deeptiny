#include "deeptiny/math.h"

#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "autograd_meta.h"
#include "cpu/kernels.h"
#include "deeptiny/types.h"
#include "engine.h"
#include "utils.h"

namespace deeptiny {
namespace math {

class AddBackward : public Function {
 public:
  AddBackward(const Tensor& a, const Tensor& b)
      : Function({utils::TensorAccessor::GetAutogradMeta(a),
                  utils::TensorAccessor::GetAutogradMeta(b)}) {}
  void operator()(const Tensor& grad, Engine& engine) override {
    for (const auto& t : getParents()) {
      t->updateGrad(grad, engine);
    }
  }
};

}  // namespace math

void Tensor::operator+=(const Tensor& other) {
  utils::CompatabilityCheck({*this, other});
  auto broadcasted_other = utils::BroadcastToShape(other, shape());
  if (broadcasted_other == std::nullopt) {
    std::stringstream err;
    err << "Could not broadcast tensor of shape " << FormatShape(other.shape())
        << " to shape " << FormatShape(shape());
    throw std::runtime_error(err.str());
  }
  switch (device()) {
    case Device::CPU:
      cpu::Add(utils::TensorAccessor::GetTensorImpl(*this),
               utils::TensorAccessor::GetTensorImpl(*broadcasted_other),
               utils::TensorAccessor::GetTensorImpl(*this));
      break;
    default:
      std::stringstream err;
      err << "Operation does not support " << device().ToString();
      throw std::runtime_error(err.str());
  }
  auto backward =
      std::make_shared<math::AddBackward>(*this, *broadcasted_other);
  auto new_autograd_meta = std::make_shared<AutogradMeta>(backward);
  autograd_meta_ = new_autograd_meta;
}
};  // namespace deeptiny
