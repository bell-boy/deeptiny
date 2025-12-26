#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace deeptiny {

using Shape = std::vector<size_t>;
using Stride = std::vector<ptrdiff_t>;

enum DType {
  Float32,
};

namespace detail {

template <class T>
consteval DType DTypeOf() {
  if constexpr (std::is_same_v<T, float>)
    return DType::Floaf32;
  else
    static_assert(sizeof(T) == 0, "No dtype mapping for this T");
}

};  // namespace detail

class TensorImplBase {
 public:
  virtual DType dtype() const = 0;
  virtual Shape shape() const = 0;
  virtual Stride stride() const = 0;
  virtual size_t offset() const = 0;

  virtual const void* data() const = 0;
  virtual void* data() = 0;

  virtual ~TensorImplBase() = default;
};

template <typename T>
class Storage {
 private:
  size_t version_count_;
  size_t numel_;
  std::shared_ptr<T[]> data_;

 public:
  Storage(size_t numel)
      : version_count_(0), numel_(numel), data_(new T[numel]) {}

  void* data(size_t offset) {
    version_count_++;
    return (void*)(data_.get() + offset);
  }

  void* data(size_t offset) const { return (void*)(data_.get() + offset); }

  size_t numel() const { return numel_; }
};

template <typename T>
class TensorImpl : public TensorImplBase {
 private:
  Storage<T> storage_;

  Shape shape_;
  Stride stride_;
  size_t offset_;

 public:
  DType dtype() const override { return detail::DTypeOf<T>(); }

  Shape shape() const override { return shape_; }

  Stride stride() const override { return stride_; }

  const void* data() const override { return storage_.data(offset_); }
  void* data() override { return storage_.data(offset_); };
};

class Tensor {
 public:
  std::shared_ptr<TensorImplBase> tensor_impl;
};
};  // namespace deeptiny