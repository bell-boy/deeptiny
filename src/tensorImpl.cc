#include "deeptiny/tensorImpl.h"

#include <sstream>
#include <stdexcept>

#include "cpu/storage.h"
#include "utils.h"

namespace deeptiny {

namespace detail {

std::shared_ptr<Storage> Storage::MakeStorage(uint64_t numel, DType dtype,
                                              Device device) {
  switch (device) {
    case Device::CPU:
      return std::make_shared<cpu::CPUStorage>(numel, dtype);
    default:
      std::runtime_error("Device not yet supported");
      exit(-1);
      break;
  }
}

TensorImpl::TensorImpl(Shape shape, Stride stride, uint64_t offset,
                       std::shared_ptr<Storage> storage)
    : storage_(storage), shape_(shape), stride_(stride), offset_(offset) {}

TensorImpl::TensorImpl(Shape shape, DType dtype, Device device)
    : storage_(Storage::MakeStorage(utils::GetTotalSize(shape), dtype, device)),
      shape_(shape),
      stride_(utils::GetContinguousStride(shape)),
      offset_(0) {}

std::shared_ptr<TensorImpl> TensorImpl::View(
    std::initializer_list<Slice> slices) {
  const size_t rank = shape_.size();
  Shape new_shape;
  Stride new_stride;
  new_shape.reserve(rank);
  new_stride.reserve(rank);

  uint64_t new_offset = offset_;

  size_t dim_index = 0;
  for (const auto& slice : slices) {
    if (dim_index >= rank) {
      break;
    }

    const int64_t dim = static_cast<int64_t>(shape_[dim_index]);
    int64_t slice_stride = slice.stride.value_or(1);

    int64_t start = slice.start.value_or(0);
    int64_t end = slice.end.value_or(dim);

    if (start < 0) {
      start += dim;
    }
    if (end < 0) {
      end += dim;
    }

    auto format_slice_error = [&](const char* reason) {
      std::ostringstream oss;
      oss << reason << " slice(start=" << start << ", end=" << end
          << ", stride=" << slice_stride << "), dim=" << dim << ".";
      return oss.str();
    };

    if (slice_stride == 0) {
      throw std::runtime_error(format_slice_error("Slice stride cannot be 0."));
    }

    if (start < 0 || start > dim || end < 0 || end > dim) {
      throw std::runtime_error(format_slice_error("Slice is out of bounds."));
    }

    if (slice_stride > 0 && start > end) {
      throw std::runtime_error(format_slice_error(
          "Slice start is greater than end with non-negative stride."));
    }

    int64_t len = 0;
    if (slice_stride > 0) {
      if (start < end) {
        len = (end - start + slice_stride - 1) / slice_stride;
      }
    } else {
      if (start > end) {
        const int64_t stride_abs = -slice_stride;
        len = (start - end + stride_abs - 1) / stride_abs;
      }
    }

    new_shape.push_back(static_cast<uint64_t>(len));
    new_stride.push_back(stride_[dim_index] * slice_stride);
    new_offset += static_cast<uint64_t>(start * stride_[dim_index]);
    ++dim_index;
  }

  for (; dim_index < rank; ++dim_index) {
    new_shape.push_back(shape_[dim_index]);
    new_stride.push_back(stride_[dim_index]);
  }

  return std::make_shared<TensorImpl>(new_shape, new_stride, new_offset,
                                      storage_);
}

};  // namespace detail

};  // namespace deeptiny
