#include <cassert>
#include <cstdint>
#include <memory>
#include <unordered_set>

#include "cpu/kernels.h"
#include "deeptiny/types.h"
#include "utils.h"

namespace deeptiny::cpu {

void Reduce(std::shared_ptr<const TensorImpl> a,
            std::shared_ptr<TensorImpl> out,
            const std::unordered_set<uint64_t>& dims, bool keep_dims) {
  assert(out->isContiguous());
  assert(out->dtype() == DType::Float32);
  auto recursive_zero_fill = [&out](auto&& self, uint64_t dim_idx,
                                    int64_t offset) -> void {
    float* outp = (float*)out->data();
    if (out->shape().size() == 0) {
      *outp = 0;
      return;
    }
    for (int64_t i = 0; i < (int64_t)out->shape()[dim_idx]; ++i) {
      int64_t new_offset = offset + i * out->stride()[dim_idx];
      if (dim_idx == out->shape().size() - 1) {
        *(outp + new_offset) = 0;
      } else {
        self(self, dim_idx + 1, new_offset);
      }
    }
  };
  recursive_zero_fill(recursive_zero_fill, 0, 0);
  if (keep_dims == false) {
    Shape unsqueezed_shape = a->shape();
    for (const auto& dim : dims) unsqueezed_shape[dim] = 1;
    out = std::make_shared<TensorImpl>(
        unsqueezed_shape, utils::GetContinguousStride(unsqueezed_shape), 0,
        out->storage());
  }
  auto recursive_reduce = [&dims, &a, &out](auto&& self, uint64_t dim_idx,
                                            int64_t a_offset,
                                            int64_t out_offset) -> void {
    for (int64_t i = 0; i < (int64_t)a->shape()[dim_idx]; ++i) {
      int64_t new_out_offset = out_offset;
      if (!dims.contains(dim_idx)) {
        new_out_offset += i * out->stride()[dim_idx];
      }
      int64_t new_a_offset = a_offset + i * a->stride()[dim_idx];
      if (dim_idx != a->shape().size() - 1) {
        self(self, dim_idx + 1, new_a_offset, new_out_offset);
      } else {
        float* out_ = ((float*)out->data()) + new_out_offset;
        float* a_ = ((float*)a->data()) + new_a_offset;
        *out_ += *a_;
      }
    }
  };
  recursive_reduce(recursive_reduce, 0, a->offset(), 0);
}
}  // namespace deeptiny::cpu
