#pragma once
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "deeptiny/tensor_slice_proxy.h"
#include "deeptiny/types.h"

namespace deeptiny {

namespace utils {
struct TensorAccessor;
};

class TensorImpl;
class AutogradMeta;
class TensorSliceProxy;

class Tensor {
 protected:
  std::shared_ptr<TensorImpl> tensor_impl_;
  std::shared_ptr<AutogradMeta> autograd_meta_;

  Tensor(std::shared_ptr<TensorImpl> tensor_impl,
         std::shared_ptr<AutogradMeta> autograd_meta);

 public:
  // Create a a contingous tensor with uninitialized data
  Tensor(Shape shape, DType dtype, Device device, bool requires_grad);

  TensorSliceProxy operator()(std::vector<Slice> slices);
  Tensor operator()(std::vector<Slice> slices) const;
  Shape shape() const;
  DType dtype() const;
  Device device() const;
  Tensor Clone() const;
  Tensor Reshape(const Shape& shape);
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
  static Tensor CreateUniform(Shape shape, Device device = Device::CPU,
                              DType dtype = DType::Float32,
                              bool requires_grad = false);
  static Tensor Zeros(Shape shape, Device device = Device::CPU,
                      DType dtype = DType::Float32, bool requires_grad = false);

  template <typename T>
  static Tensor FromVector(const std::vector<T>& values, Shape shape,
                           Device device = Device::CPU,
                           bool requires_grad = false) {
    static_assert(
        std::is_same_v<T, float>,
        "Tensor::FromVector currently only supports std::vector<float>");

    uint64_t total = 1;
    for (const auto dim : shape) {
      total *= dim;
    }
    if (values.size() != total) {
      throw std::runtime_error(
          "FromVector received mismatched values/shape size");
    }

    return FromBuffer(std::span<const std::byte>(
                          reinterpret_cast<const std::byte*>(values.data()),
                          values.size() * sizeof(T)),
                      std::move(shape), DType::Float32, device, requires_grad);
  }

  template <typename T>
  static Tensor FromVector(const std::vector<T>& values,
                           Device device = Device::CPU,
                           bool requires_grad = false) {
    return FromVector(values, Shape{static_cast<uint64_t>(values.size())},
                      device, requires_grad);
  }

  void operator+=(const Tensor& other);
  void operator-=(const Tensor& other);
  void operator*=(const Tensor& other);
  void operator/=(const Tensor& other);

  friend struct utils::TensorAccessor;
  friend class TensorSliceProxy;
};

};  // namespace deeptiny
