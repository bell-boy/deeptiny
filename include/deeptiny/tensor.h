#pragma once
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "deeptiny/types.h"

namespace deeptiny {

namespace utils {
struct TensorAccessor;
};

class TensorImpl;
class AutogradMeta;
class View;

class Tensor {
 protected:
  std::shared_ptr<TensorImpl> tensor_impl_;
  std::shared_ptr<AutogradMeta> autograd_meta_;

 public:
  // Create a a contingous tensor with uninitialized data
  Tensor(Shape shape, DType dtype, Device device, bool requires_grad);
  Tensor(std::shared_ptr<TensorImpl> tensor_impl,
         std::shared_ptr<AutogradMeta> autograd_meta);

  View operator()(std::vector<Slice> slices);
  const View operator()(std::vector<Slice> slices) const;
  Shape shape() const;
  DType dtype() const;
  Device device() const;

  /**
   * Creates a tensor on the CPU with the expectation that the bytes are laid
   * out in row-major order
   */
  static Tensor FromBuffer(std::span<std::byte> bytes, Shape shape,
                           DType dtype = DType::Float32,
                           Device device = Device::CPU,
                           bool requires_grad = false);
  void operator+=(const Tensor& other);

  friend struct utils::TensorAccessor;
};

};  // namespace deeptiny
