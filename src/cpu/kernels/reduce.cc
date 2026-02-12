#include <cblas.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>

#include "cpu/kernels.h"
#include "deeptiny/types.h"
#include "tensor_impl.h"
#include "utils.h"

namespace deeptiny::cpu {
namespace {
// expects out to be zerod out
bool ReduceContiguous(std::shared_ptr<const TensorImpl> a,
                      std::shared_ptr<TensorImpl> out,
                      const utils::UInt64IdentitySet& dims) {
  assert(out->isContiguous());
  assert(out->dtype() == DType::Float32);
  if (!a->isContiguous()) {
    return false;
  }
  uint64_t latest_kept = 0;
  uint64_t earliest_reduced = UINT64_MAX;
  uint64_t step = 1;
  uint64_t numel = 1;
  for (uint64_t i = 0; i < a->shape().size(); ++i) {
    if (dims.contains(i)) {
      earliest_reduced = std::min(earliest_reduced, i);
      step *= a->shape()[i];
    } else {
      latest_kept = std::max(latest_kept, i);
      numel *= a->shape()[i];
    }
  }
  if (latest_kept > earliest_reduced) {
    return false;
  }

  float one = 1;
  float* ap = (float*)a->data();
  float* outp = (float*)out->data();
  for (uint64_t i = 0; i < numel; ++i) {
    *(outp + i) = cblas_sdot(step, (ap + (step * i)), 1, &one, 0);
  }
  return true;
}
}  // namespace

void Reduce(std::shared_ptr<const TensorImpl> a,
            std::shared_ptr<TensorImpl> out, const std::vector<uint64_t>& dims,
            bool keep_dims) {
  assert(out->isContiguous());
  assert(out->dtype() == DType::Float32);

  utils::UInt64IdentitySet dims_lookup;
  dims_lookup.reserve(dims.size());
  for (const auto dim : dims) {
    dims_lookup.insert(dim);
  }

  memset(out->data(), 0, out->storage()->numel() * out->dtype().size());
  if (ReduceContiguous(a, out, dims_lookup)) return;
  if (keep_dims == false) {
    Shape unsqueezed_shape = a->shape();
    for (const auto dim : dims_lookup) {
      unsqueezed_shape[dim] = 1;
    }
    out = std::make_shared<TensorImpl>(
        unsqueezed_shape, utils::GetContinguousStride(unsqueezed_shape), 0,
        out->storage());
  }
  Shape a_shape = a->shape();
  Stride a_stride = a->stride();
  Stride out_stride = out->stride();
  uint64_t a_rank = a_shape.size();
  float* ap = (float*)a->data();
  float* outp = (float*)out->data();
  auto recursive_reduce = [&dims_lookup, &ap, &a_shape, &a_stride, &a_rank,
                           &outp, &out_stride](auto&& self, uint64_t dim_idx,
                                               int64_t a_offset,
                                               int64_t out_offset) -> void {
    for (int64_t i = 0; i < (int64_t)a_shape[dim_idx]; ++i) {
      int64_t new_out_offset = out_offset;
      if (!dims_lookup.contains(dim_idx)) {
        new_out_offset += i * out_stride[dim_idx];
      }
      int64_t new_a_offset = a_offset + i * a_stride[dim_idx];
      if (dim_idx != a_rank - 1) {
        self(self, dim_idx + 1, new_a_offset, new_out_offset);
      } else {
        float* out_ = outp + new_out_offset;
        float* a_ = ap + new_a_offset;
        *out_ += *a_;
      }
    }
  };
  recursive_reduce(recursive_reduce, 0, a->offset(), 0);
}
}  // namespace deeptiny::cpu
