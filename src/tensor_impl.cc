#include "tensor_impl.h"

#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "cpu/storage.h"
#include "utils.h"

namespace deeptiny {

std::shared_ptr<Storage> Storage::MakeStorage(uint64_t numel, DType dtype,
                                              Device device) {
  switch (device) {
    case Device::CPU:
      return std::make_shared<cpu::CPUStorage>(numel, dtype);
    default:
      throw std::runtime_error("Device not yet supported");
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

void TensorImpl::ValidateView(const Shape& shape, const Stride& stride,
                              uint64_t offset) const {
  if (!storage_) {
    throw std::runtime_error("Cannot create view with null storage.");
  }
  if (shape.size() != stride.size()) {
    throw std::runtime_error(
        "Cannot create view with mismatched shape and stride ranks.");
  }

  const uint64_t storage_numel = storage_->numel();
  if (storage_numel >
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    throw std::runtime_error(
        "Cannot create view: storage too large for validation.");
  }

  bool has_zero_dim = false;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == 0) {
      has_zero_dim = true;
      break;
    }
  }

  if (has_zero_dim) {
    if (storage_numel == 0) {
      if (offset != 0) {
        throw std::runtime_error(
            "Cannot create view: offset out of bounds for empty storage.");
      }
    } else if (offset >= storage_numel) {
      throw std::runtime_error(
          "Cannot create view: offset out of bounds for empty tensor.");
    }
    return;
  }

  if (storage_numel == 0) {
    throw std::runtime_error(
        "Cannot create view with non-empty shape on empty storage.");
  }

  if (offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    throw std::runtime_error(
        "Cannot create view: offset too large for validation.");
  }

  int64_t min_delta = 0;
  int64_t max_delta = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == 0) {
      continue;
    }
    if (shape[i] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      throw std::runtime_error(
          "Cannot create view: shape too large for validation.");
    }
    const int64_t span = static_cast<int64_t>(shape[i] - 1);
    const int64_t stride_i = stride[i];

    if (span != 0) {
      if (stride_i == std::numeric_limits<int64_t>::min()) {
        throw std::runtime_error(
            "Cannot create view: stride too large for validation.");
      }
      const int64_t stride_abs = stride_i < 0 ? -stride_i : stride_i;
      if (stride_abs != 0 &&
          span > std::numeric_limits<int64_t>::max() / stride_abs) {
        throw std::runtime_error("Cannot create view: stride/shape overflow.");
      }
    }

    const int64_t delta = span * stride_i;
    if (delta >= 0) {
      if (max_delta > std::numeric_limits<int64_t>::max() - delta) {
        throw std::runtime_error("Cannot create view: stride/shape overflow.");
      }
      max_delta += delta;
    } else {
      if (min_delta < std::numeric_limits<int64_t>::min() - delta) {
        throw std::runtime_error("Cannot create view: stride/shape overflow.");
      }
      min_delta += delta;
    }
  }

  const int64_t base = static_cast<int64_t>(offset);
  const int64_t min_offset = base + min_delta;
  const int64_t max_offset = base + max_delta;

  if (min_offset < 0) {
    throw std::runtime_error(
        "Cannot create view: stride/offset underflows storage bounds.");
  }
  if (max_offset >= static_cast<int64_t>(storage_numel)) {
    throw std::runtime_error(
        "Cannot create view: stride/offset exceeds storage bounds.");
  }
}

std::shared_ptr<TensorImpl> TensorImpl::View(Shape shape, Stride stride,
                                             uint64_t offset) const {
  ValidateView(shape, stride, offset);
  return std::shared_ptr<TensorImpl>(
      new TensorImpl(std::move(shape), std::move(stride), offset, storage_));
}

std::shared_ptr<TensorImpl> TensorImpl::View(const std::vector<Slice>& slices) {
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
    int64_t slice_stride = slice.stride;

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

  return View(std::move(new_shape), std::move(new_stride), new_offset);
}

};  // namespace deeptiny
