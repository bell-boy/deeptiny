#pragma once
#include <cstddef>
#include <memory>
#include <span>

#include "deeptiny/tensorImpl.h"

namespace deeptiny {

namespace utils {
struct TensorAccessor;
};

class View;

class Tensor {
 protected:
  Tensor(std::shared_ptr<detail::TensorImpl> tensor_impl)
      : tensor_impl_(tensor_impl) {}
  std::shared_ptr<detail::TensorImpl> tensor_impl_;
  bool requires_grad_;

 public:
  /**
   * Create a a contingous tensor with uninitialized data
   */
  Tensor(Shape shape, DType dtype, Device device, bool requires_grad)
      : tensor_impl_(
            std::make_shared<detail::TensorImpl>(shape, dtype, device)),
        requires_grad_(requires_grad) {}

  View operator()(std::initializer_list<Slice> slices);

  /**
   * Creates a tensor on the CPU with the expectation that the bytes are laid
   * out in row-major order
   * TODO: Make device agnostic
   */
  static Tensor FromBuffer(DType dtype, std::span<std::byte> bytes,
                           Shape shape);

  friend struct utils::TensorAccessor;
};

class View : public Tensor {
 private:
  View(std::shared_ptr<detail::TensorImpl> tensor_impl) : Tensor(tensor_impl) {}

 public:
  void operator=(const Tensor& other);

  friend Tensor;
};

};  // namespace deeptiny
