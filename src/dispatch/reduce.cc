#include "reduce.h"

#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "cpu/kernels.h"
#include "deeptiny/tensor.h"
#include "tensor_impl.h"
#include "utils.h"
namespace deeptiny::dispatch::reduce {

std::shared_ptr<TensorImpl> OutOfPlace(const std::shared_ptr<TensorImpl>& a,
                                       const std::vector<uint64_t>& dims,
                                       bool keep_dims) {
  utils::UInt64IdentitySet dims_lookup;
  dims_lookup.reserve(dims.size());
  for (const auto dim : dims) {
    if (dim >= a->shape().size()) {
      throw std::runtime_error("Reduce dim out of range");
    }
    dims_lookup.insert(dim);
  }

  Shape new_shape;
  for (uint64_t i = 0; i < a->shape().size(); ++i) {
    const bool is_reduced_dim = dims_lookup.contains(i);
    if (is_reduced_dim) {
      if (keep_dims) {
        new_shape.push_back(1);
      }
    } else {
      new_shape.push_back(a->shape()[i]);
    }
  }
  std::shared_ptr<TensorImpl> out =
      std::make_shared<TensorImpl>(new_shape, a->dtype(), a->device());
  switch (out->device()) {
    case Device::CPU:
      cpu::Reduce(a, out, dims, keep_dims);
      return out;
    default: {
      std::stringstream err;
      err << "Reduce does not support " << out->device().ToString();
      throw std::runtime_error(err.str());
    }
  }
  return out;
}

}  // namespace deeptiny::dispatch::reduce
