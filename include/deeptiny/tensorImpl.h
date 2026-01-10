#pragma once
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <vector>

namespace deeptiny {

using Shape = std::vector<uint64_t>;
using Stride = std::vector<int64_t>;

enum DType { Float32, BFloat16 };

enum Device { CPU, Metal };

struct Slice {
  using Index = std::optional<int64_t>;
  Index start;
  Index end;
  Index stride;

  Slice(ptrdiff_t x) : start(x), end(x + 1), stride(1) {}
  Slice(int64_t start, int64_t end) : start(start), end(end), stride(1) {}
  Slice(int64_t start, int64_t end, int64_t stride)
      : start(start), end(end), stride(stride) {}
};

namespace detail {

/**
 * Abstract Base Class for Storage objects.
 *
 * Each implementation is required to implement get_, which is a pointer to the
 * data on the device. Each storage object owns it's underlying buffer. Once
 * it's destroyed, it's responsible for freeing the buffer.
 */
class Storage {
 protected:
  uint64_t version_count_;
  uint64_t numel_;
  DType dtype_;
  Device device_;

  virtual void* get_(uint64_t offset) const = 0;

 public:
  Storage(const Storage&) = delete;

  Storage(uint64_t numel, DType dtype, Device device)
      : version_count_(0), numel_(numel), dtype_(dtype), device_(device) {};

  virtual ~Storage() = default;

  void* data(uint64_t offset) {
    version_count_++;
    return get_(offset);
  }

  const void* data(uint64_t offset) const { return (const void*)get_(offset); }

  uint64_t numel() const { return numel_; }

  DType dtype() const { return dtype_; }

  Device device() const { return device_; }

  static std::shared_ptr<Storage> MakeStorage(uint64_t numel, DType dtype,
                                              Device device);
};

class TensorImpl {
 private:
  std::shared_ptr<Storage> storage_;

  Shape shape_;
  Stride stride_;
  uint64_t offset_;

 public:
  TensorImpl(Shape shape, Stride stride, uint64_t offset,
             std::shared_ptr<Storage> storage);
  /**
   * Create a contiguous uninitialized tensor
   */
  TensorImpl(Shape shape, DType dtype, Device device);

  std::shared_ptr<TensorImpl> View(std::initializer_list<Slice> slices);

  DType dtype() const { return storage_->dtype(); }

  Device device() const { return storage_->device(); }

  Shape shape() const { return shape_; }

  Stride stride() const { return stride_; }

  uint64_t offset() const { return offset_; }

  const void* data() const { return storage_->data(offset_); }

  void* data() { return storage_->data(offset_); }
};

}  // namespace detail

}  // namespace deeptiny
