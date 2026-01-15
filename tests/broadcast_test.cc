#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "broadcast.h"

#include <array>
#include <random>
#include <vector>

#include "deeptiny/functional.h"
#include "deeptiny/tensor.h"
#include "doctest/doctest.h"
#include "utils.h"

namespace {
void FillSequential(deeptiny::Tensor& t, float base) {
  auto impl = deeptiny::utils::TensorAccessor::GetTensorImpl(t);
  const uint64_t total = deeptiny::utils::GetTotalSize(impl->shape());
  auto* data = static_cast<float*>(impl->data());
  for (uint64_t i = 0; i < total; ++i) {
    data[i] = base + static_cast<float>(i);
  }
}

int64_t OffsetForIndex(const deeptiny::Stride& stride,
                       const std::vector<uint64_t>& index) {
  int64_t offset = 0;
  for (size_t i = 0; i < stride.size(); ++i) {
    offset += static_cast<int64_t>(index[i]) * stride[i];
  }
  return offset;
}

template <typename Fn>
void ForEachIndex(const deeptiny::Shape& shape, Fn&& fn) {
  if (shape.empty()) {
    fn(std::vector<uint64_t>{});
    return;
  }
  for (auto dim : shape) {
    if (dim == 0) {
      return;
    }
  }
  std::vector<uint64_t> index(shape.size(), 0);
  while (true) {
    fn(index);
    size_t dim = shape.size();
    while (dim > 0) {
      --dim;
      index[dim] += 1;
      if (index[dim] < shape[dim]) {
        break;
      }
      index[dim] = 0;
      if (dim == 0) {
        return;
      }
    }
  }
}
}  // namespace

TEST_CASE("Broadcast Forward Test") {
  SUBCASE("Singleton expansion keeps data and uses zero strides") {
    deeptiny::Tensor a({2, 1, 3}, deeptiny::DType::Float32,
                       deeptiny::Device::CPU, false);
    FillSequential(a, 0.0f);

    auto out = deeptiny::utils::BroadcastToShape(a, {2, 4, 3});
    REQUIRE(out.has_value());

    auto a_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(a);
    auto out_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(*out);

    CHECK(out_impl->shape() == deeptiny::Shape{2, 4, 3});

    const auto& a_shape = a_impl->shape();
    const auto& a_stride = a_impl->stride();
    const auto& out_stride = out_impl->stride();
    const auto& out_shape = out_impl->shape();

    const size_t rank_diff = out_shape.size() - a_shape.size();
    for (size_t j = 0; j < out_shape.size(); ++j) {
      if (j < rank_diff) {
        CHECK(out_stride[j] == 0);
        continue;
      }
      const uint64_t a_dim = a_shape[j - rank_diff];
      const uint64_t out_dim = out_shape[j];
      if (a_dim == out_dim) {
        CHECK(out_stride[j] == a_stride[j - rank_diff]);
      } else {
        CHECK(out_stride[j] == 0);
      }
    }

    const auto* out_data = static_cast<const float*>(out_impl->data());
    const auto* a_data = static_cast<const float*>(a_impl->data());

    ForEachIndex(out_shape, [&](const std::vector<uint64_t>& out_index) {
      std::vector<uint64_t> a_index(a_shape.size(), 0);
      for (size_t i = 0; i < a_shape.size(); ++i) {
        const uint64_t out_i = out_index[rank_diff + i];
        a_index[i] = (a_shape[i] == 1) ? 0 : out_i;
      }
      const int64_t out_offset = OffsetForIndex(out_stride, out_index);
      const int64_t a_offset = OffsetForIndex(a_stride, a_index);
      CHECK(out_data[static_cast<size_t>(out_offset)] ==
            a_data[static_cast<size_t>(a_offset)]);
    });
  }

  SUBCASE("Rank expansion adds leading zero strides") {
    deeptiny::Tensor b({3}, deeptiny::DType::Float32, deeptiny::Device::CPU,
                       false);
    FillSequential(b, 10.0f);

    auto out = deeptiny::utils::BroadcastToShape(b, {2, 3});
    REQUIRE(out.has_value());

    auto b_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(b);
    auto out_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(*out);

    CHECK(out_impl->shape() == deeptiny::Shape{2, 3});
    CHECK(out_impl->stride()[0] == 0);
    CHECK(out_impl->stride()[1] == b_impl->stride()[0]);

    const auto& b_shape = b_impl->shape();
    const auto& b_stride = b_impl->stride();
    const auto& out_stride = out_impl->stride();
    const auto& out_shape = out_impl->shape();

    const size_t rank_diff = out_shape.size() - b_shape.size();
    const auto* out_data = static_cast<const float*>(out_impl->data());
    const auto* b_data = static_cast<const float*>(b_impl->data());

    ForEachIndex(out_shape, [&](const std::vector<uint64_t>& out_index) {
      std::vector<uint64_t> b_index(b_shape.size(), 0);
      for (size_t i = 0; i < b_shape.size(); ++i) {
        const uint64_t out_i = out_index[rank_diff + i];
        b_index[i] = (b_shape[i] == 1) ? 0 : out_i;
      }
      const int64_t out_offset = OffsetForIndex(out_stride, out_index);
      const int64_t b_offset = OffsetForIndex(b_stride, b_index);
      CHECK(out_data[static_cast<size_t>(out_offset)] ==
            b_data[static_cast<size_t>(b_offset)]);
    });
  }

  SUBCASE("No-op broadcast returns identical view") {
    deeptiny::Tensor a({2, 3}, deeptiny::DType::Float32, deeptiny::Device::CPU,
                       false);
    FillSequential(a, -5.0f);

    auto out = deeptiny::utils::BroadcastToShape(a, {2, 3});
    REQUIRE(out.has_value());

    auto a_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(a);
    auto out_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(*out);

    CHECK(out_impl->shape() == a_impl->shape());
    CHECK(out_impl->stride() == a_impl->stride());
    CHECK(out_impl->data() == a_impl->data());
  }

  SUBCASE("Invalid broadcast returns nullopt") {
    deeptiny::Tensor a({2, 3}, deeptiny::DType::Float32, deeptiny::Device::CPU,
                       false);
    auto out = deeptiny::utils::BroadcastToShape(a, {2, 4});
    CHECK(!out.has_value());
  }

  SUBCASE("Broadcast pair helper matches expected shape and mapping") {
    deeptiny::Tensor a({2, 1, 3}, deeptiny::DType::Float32,
                       deeptiny::Device::CPU, false);
    deeptiny::Tensor b({1, 4, 3}, deeptiny::DType::Float32,
                       deeptiny::Device::CPU, false);
    FillSequential(a, 0.0f);
    FillSequential(b, 100.0f);

    auto [a_out, b_out] = deeptiny::utils::Broadcast(a, b);

    auto a_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(a);
    auto b_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(b);
    auto a_out_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(a_out);
    auto b_out_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(b_out);

    CHECK(a_out_impl->shape() == deeptiny::Shape{2, 4, 3});
    CHECK(b_out_impl->shape() == deeptiny::Shape{2, 4, 3});

    const auto& out_shape = a_out_impl->shape();
    const auto* a_out_data = static_cast<const float*>(a_out_impl->data());
    const auto* b_out_data = static_cast<const float*>(b_out_impl->data());
    const auto* a_data = static_cast<const float*>(a_impl->data());
    const auto* b_data = static_cast<const float*>(b_impl->data());

    const auto& a_stride = a_impl->stride();
    const auto& b_stride = b_impl->stride();
    const auto& a_out_stride = a_out_impl->stride();
    const auto& b_out_stride = b_out_impl->stride();

    const size_t a_rank_diff = out_shape.size() - a_impl->shape().size();
    const size_t b_rank_diff = out_shape.size() - b_impl->shape().size();

    ForEachIndex(out_shape, [&](const std::vector<uint64_t>& out_index) {
      std::vector<uint64_t> a_index(a_impl->shape().size(), 0);
      for (size_t i = 0; i < a_impl->shape().size(); ++i) {
        const uint64_t out_i = out_index[a_rank_diff + i];
        a_index[i] = (a_impl->shape()[i] == 1) ? 0 : out_i;
      }
      std::vector<uint64_t> b_index(b_impl->shape().size(), 0);
      for (size_t i = 0; i < b_impl->shape().size(); ++i) {
        const uint64_t out_i = out_index[b_rank_diff + i];
        b_index[i] = (b_impl->shape()[i] == 1) ? 0 : out_i;
      }
      const int64_t a_out_offset = OffsetForIndex(a_out_stride, out_index);
      const int64_t b_out_offset = OffsetForIndex(b_out_stride, out_index);
      const int64_t a_offset = OffsetForIndex(a_stride, a_index);
      const int64_t b_offset = OffsetForIndex(b_stride, b_index);

      CHECK(a_out_data[static_cast<size_t>(a_out_offset)] ==
            a_data[static_cast<size_t>(a_offset)]);
      CHECK(b_out_data[static_cast<size_t>(b_out_offset)] ==
            b_data[static_cast<size_t>(b_offset)]);
    });
  }
}

TEST_CASE("Broadcast Backward Test") {}
