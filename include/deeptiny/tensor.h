#pragma once
#include <cstddef>
#include <memory>
#include <optional>
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

  Tensor(std::shared_ptr<TensorImpl> tensor_impl,
         std::shared_ptr<AutogradMeta> autograd_meta);

 public:
  // Create a a contingous tensor with uninitialized data
  Tensor(Shape shape, DType dtype, Device device, bool requires_grad);

  View operator()(std::vector<Slice> slices);
  const View operator()(std::vector<Slice> slices) const;
  Shape shape() const;
  DType dtype() const;
  Device device() const;
  Tensor Clone() const;
  Tensor Squeeze(std::initializer_list<uint64_t> dims);
  Tensor Squeeze(const std::vector<uint64_t>& dims);
  bool requires_grad() const;
  std::optional<Tensor> grad() const;
  void Backward(bool keep_graph = false);

  /**
   * Creates a tensor on the CPU with the expectation that the bytes are laid
   * out in row-major order
   */
  static Tensor FromBuffer(std::span<const std::byte> bytes, Shape shape,
                           DType dtype = DType::Float32,
                           Device device = Device::CPU,
                           bool requires_grad = false);
  void operator+=(const Tensor& other);
  void operator-=(const Tensor& other);
  void operator*=(const Tensor& other);
  void operator/=(const Tensor& other);

  friend struct utils::TensorAccessor;
};

};  // namespace deeptiny
