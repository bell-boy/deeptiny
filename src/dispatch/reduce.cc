#include "reduce.h"

#include <cstdint>
#include <memory>
#include <sstream>
#include <unordered_set>

#include "cpu/kernels.h"
#include "deeptiny/tensor.h"
#include "tensor_impl.h"
#include "utils.h"
namespace deeptiny::dispatch::reduce {

std::shared_ptr<TensorImpl> OutOfPlace(const std::shared_ptr<TensorImpl>& a,
                                       const std::vector<uint64_t>& dims,
                                       bool keep_dims) {
  std::unordered_set<uint64_t> kept_dims(dims.begin(), dims.end());
  Shape new_shape;
  for (uint64_t i = 0; i < a->shape().size(); ++i) {
    if (kept_dims.contains(i)) {
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
      cpu::Reduce(a, out, std::move(kept_dims), keep_dims);
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
