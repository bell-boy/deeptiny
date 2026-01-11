#include "deeptiny/tensor.h"

#include <iostream>
#include <sstream>
#include <vector>

#include "cpu/kernels.h"
#include "tensorImpl.h"
#include "utils.h"

namespace deeptiny {

Tensor::Tensor(std::shared_ptr<TensorImpl> tensor_impl)
    : tensor_impl_(tensor_impl) {}

Tensor::Tensor(Shape shape, DType dtype, Device device, bool requires_grad)
    : tensor_impl_(std::make_shared<TensorImpl>(shape, dtype, device)),
      requires_grad_(requires_grad) {}

Shape Tensor::shape() const { return tensor_impl_->shape(); }
DType Tensor::dtype() const { return tensor_impl_->dtype(); }

View::View(std::shared_ptr<TensorImpl> tensor_impl) : Tensor(tensor_impl) {}

View Tensor::operator()(std::initializer_list<Slice> slices) {
  return View(tensor_impl_->View(slices));
}

void View::operator=(const Tensor& other) {
  // TODO: handle broadcasting
  // TODO: handle casting
  const Shape other_shape = other.shape();
  const Shape my_shape = shape();

  if (other_shape != my_shape) {
    std::stringstream err;
    err << "Trying to assign a tensor of shape { ";
    for (const auto& dim : other_shape) {
      err << dim << ", ";
    }
    err << "} to a View of shape {";
    for (const auto& dim : my_shape) {
      err << dim << ", ";
    }
    err << "}";
    throw std::runtime_error(err.str());
  }
  if (other.dtype() != dtype()) {
    std::stringstream err;
    err << "Trying to copy data of type " << other.dtype().ToString()
        << " to View of type " << dtype().ToString();
    throw std::runtime_error(err.str());
  }
  // TODO: fast path for contiguous tensors
  // TODO: fast path for Accelerator -> CPU transfers
  // TODO: fast path for small ranks
  // TODO: issue, currently i have to tell the tensor accessor to give me a
  // const TensorImpl, this isn't enforced by the tensor being const Need to
  // think about this
  std::shared_ptr<const TensorImpl> other_impl =
      utils::TensorAccessor::GetTensorImpl(other);
  auto my_impl = tensor_impl_;
  std::vector<std::byte> buf(dtype().size());
  auto Transfer = [my_impl, other_impl, &buf](uint64_t src_offset,
                                              uint64_t dest_offset) {
    other_impl->CopyToHost(src_offset, 1, buf.data());
    my_impl->CopyFromHost(dest_offset, 1, buf.data());
  };

  Stride other_stride = other_impl->stride();
  Stride my_stride = my_impl->stride();

  if (my_shape.empty()) {
    Transfer(other_impl->offset(), my_impl->offset());
    return;
  }

  const size_t rank = my_shape.size();
  auto AssignFastBlock = [&my_shape, &other_stride, &my_stride, &Transfer](
                             size_t dim, size_t remaining, uint64_t other_base,
                             uint64_t my_base) {
    for (uint64_t i0 = 0; i0 < my_shape[dim]; ++i0) {
      const uint64_t other_offset_0 = other_base + i0 * other_stride[dim];
      const uint64_t my_offset_0 = my_base + i0 * my_stride[dim];
      if (remaining == 1) {
        Transfer(other_offset_0, my_offset_0);
        continue;
      }
      for (uint64_t i1 = 0; i1 < my_shape[dim + 1]; ++i1) {
        const uint64_t other_offset_1 =
            other_offset_0 + i1 * other_stride[dim + 1];
        const uint64_t my_offset_1 = my_offset_0 + i1 * my_stride[dim + 1];
        if (remaining == 2) {
          Transfer(other_offset_1, my_offset_1);
          continue;
        }
        for (uint64_t i2 = 0; i2 < my_shape[dim + 2]; ++i2) {
          const uint64_t other_offset_2 =
              other_offset_1 + i2 * other_stride[dim + 2];
          const uint64_t my_offset_2 = my_offset_1 + i2 * my_stride[dim + 2];
          if (remaining == 3) {
            Transfer(other_offset_2, my_offset_2);
            continue;
          }
          for (uint64_t i3 = 0; i3 < my_shape[dim + 3]; ++i3) {
            const uint64_t other_offset_3 =
                other_offset_2 + i3 * other_stride[dim + 3];
            const uint64_t my_offset_3 = my_offset_2 + i3 * my_stride[dim + 3];
            if (remaining == 4) {
              Transfer(other_offset_3, my_offset_3);
              continue;
            }
            for (uint64_t i4 = 0; i4 < my_shape[dim + 4]; ++i4) {
              const uint64_t other_offset_4 =
                  other_offset_3 + i4 * other_stride[dim + 4];
              const uint64_t my_offset_4 =
                  my_offset_3 + i4 * my_stride[dim + 4];
              Transfer(other_offset_4, my_offset_4);
            }
          }
        }
      }
    }
  };

  auto AssignRecursive = [rank, &my_shape, &other_stride, &my_stride,
                          &AssignFastBlock](auto&& self, size_t dim,
                                            uint64_t other_base,
                                            uint64_t my_base) -> void {
    const size_t remaining = rank - dim;
    if (remaining <= 5) {
      AssignFastBlock(dim, remaining, other_base, my_base);
      return;
    }
    for (uint64_t i0 = 0; i0 < my_shape[dim]; ++i0) {
      const uint64_t other_offset_0 = other_base + i0 * other_stride[dim];
      const uint64_t my_offset_0 = my_base + i0 * my_stride[dim];
      for (uint64_t i1 = 0; i1 < my_shape[dim + 1]; ++i1) {
        const uint64_t other_offset_1 =
            other_offset_0 + i1 * other_stride[dim + 1];
        const uint64_t my_offset_1 = my_offset_0 + i1 * my_stride[dim + 1];
        for (uint64_t i2 = 0; i2 < my_shape[dim + 2]; ++i2) {
          const uint64_t other_offset_2 =
              other_offset_1 + i2 * other_stride[dim + 2];
          const uint64_t my_offset_2 = my_offset_1 + i2 * my_stride[dim + 2];
          for (uint64_t i3 = 0; i3 < my_shape[dim + 3]; ++i3) {
            const uint64_t other_offset_3 =
                other_offset_2 + i3 * other_stride[dim + 3];
            const uint64_t my_offset_3 = my_offset_2 + i3 * my_stride[dim + 3];
            for (uint64_t i4 = 0; i4 < my_shape[dim + 4]; ++i4) {
              const uint64_t other_offset_4 =
                  other_offset_3 + i4 * other_stride[dim + 4];
              const uint64_t my_offset_4 =
                  my_offset_3 + i4 * my_stride[dim + 4];
              self(self, dim + 5, other_offset_4, my_offset_4);
            }
          }
        }
      }
    }
  };

  AssignRecursive(AssignRecursive, 0, other_impl->offset(), my_impl->offset());
}

Tensor Tensor::FromBuffer(std::span<std::byte> bytes, Shape shape, DType dtype,
                          Device device, bool requires_grad) {
  std::shared_ptr<TensorImpl> result;
  switch (device) {
    case Device::CPU:
      result = cpu::FromBuffer(dtype, bytes, shape);
      break;
    case Device::Metal:
      // TODO: remove this, get's around warnings until we have autogradMeta
      std::cout << requires_grad << std::endl;
      std::runtime_error("FromBuffer doesn't support Metal at the moment");
      break;
  }
  return Tensor(result);
}

};  // namespace deeptiny
