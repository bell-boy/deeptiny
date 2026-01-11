#pragma once
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <span>

#include "deeptiny/types.h"

namespace deeptiny {

namespace utils {
struct TensorAccessor;
};

class TensorImpl;
class View;

class Tensor {
 protected:
  std::shared_ptr<TensorImpl> tensor_impl_;
  bool requires_grad_;

 public:
  Tensor(std::shared_ptr<TensorImpl> tensor_impl);
  /**
   * Create a a contingous tensor with uninitialized data
   */
  Tensor(Shape shape, DType dtype, Device device, bool requires_grad);

  View operator()(std::initializer_list<Slice> slices);
  Shape shape() const;
  DType dtype() const;

  /**
   * Creates a tensor on the CPU with the expectation that the bytes are laid
   * out in row-major order
   */
  static Tensor FromBuffer(std::span<std::byte> bytes, Shape shape,
                           DType dtype = DType::Float32,
                           Device device = Device::CPU,
                           bool requires_grad = false);

  friend struct utils::TensorAccessor;
};

class View : public Tensor {
 private:
  View(std::shared_ptr<TensorImpl> tensor_impl);

 public:
  void operator=(const Tensor& other);

  friend Tensor;
};

};  // namespace deeptiny
