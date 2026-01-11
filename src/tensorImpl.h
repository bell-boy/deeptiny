#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <vector>

#include "deeptiny/types.h"

namespace deeptiny {

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

  virtual void CopyToHost(uint64_t offset, uint64_t numel,
                          void* buffer) const = 0;
  // TODO: this is currently dumb, right now the implementer has to remember to
  // increment the version count, and I don't want that
  virtual void CopyFromHost(uint64_t offset, uint64_t numel,
                            const void* buffer) = 0;

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

  void CopyToHost(uint64_t offset, uint64_t numel, void* buffer) const {
    storage_->CopyToHost(offset, numel, buffer);
  }

  void CopyFromHost(uint64_t offset, uint64_t numel, const void* buffer) const {
    storage_->CopyFromHost(offset, numel, buffer);
  }

  DType dtype() const { return storage_->dtype(); }

  Device device() const { return storage_->device(); }

  Shape shape() const { return shape_; }

  Stride stride() const { return stride_; }

  uint64_t offset() const { return offset_; }

  const void* data() const { return storage_->data(offset_); }

  void* data() { return storage_->data(offset_); }
};

}  // namespace deeptiny
