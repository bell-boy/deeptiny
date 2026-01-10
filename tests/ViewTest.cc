#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <array>
#include <random>

#include "deeptiny/functional.h"
#include "deeptiny/tensor.h"
#include "doctest/doctest.h"
#include "utils.h"

namespace {
constexpr int kRank = 5;
}  // namespace

TEST_CASE("Random view forward test.") {
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int64_t> dim_dist(1, 7);

  for (int trial = 0; trial < 10; ++trial) {
    deeptiny::Shape shape(kRank);
    for (int i = 0; i < kRank; ++i) {
      shape[i] = static_cast<uint64_t>(dim_dist(rng));
    }

    auto t =
        deeptiny::functional::CreateUniform(shape, deeptiny::DType::Float32);
    auto t_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(t);

    std::array<int64_t, kRank> starts{};
    std::array<int64_t, kRank> strides{};
    std::array<int64_t, kRank> lens{};
    std::array<deeptiny::Slice, kRank> slices{
        deeptiny::Slice(0), deeptiny::Slice(0), deeptiny::Slice(0),
        deeptiny::Slice(0), deeptiny::Slice(0),
    };

    for (int i = 0; i < kRank; ++i) {
      const int64_t dim = static_cast<int64_t>(shape[i]);
      std::uniform_int_distribution<int64_t> start_dist(0, dim - 1);
      const int64_t start = start_dist(rng);

      std::uniform_int_distribution<int64_t> stride_dist(1, dim);
      const int64_t stride = stride_dist(rng);

      const int64_t max_len = ((dim - 1 - start) / stride) + 1;
      std::uniform_int_distribution<int64_t> len_dist(1, max_len);
      const int64_t len = len_dist(rng);

      const int64_t end = start + (len - 1) * stride + 1;  // end is exclusive

      starts[i] = start;
      strides[i] = stride;
      lens[i] = len;
      slices[i] = deeptiny::Slice(start, end, stride);
    }

    auto view = t({slices[0], slices[1], slices[2], slices[3], slices[4]});
    auto view_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(view);

    deeptiny::Shape expected_shape(kRank);
    for (int i = 0; i < kRank; ++i) {
      expected_shape[i] = static_cast<uint64_t>(lens[i]);
    }

    CHECK(view_impl->shape() == expected_shape);

    const auto* view_data = static_cast<const float*>(view_impl->data());
    const auto& view_stride = view_impl->stride();
    const auto* src_data = static_cast<const float*>(t_impl->data());
    const auto& src_stride = t_impl->stride();

    for (int64_t i0 = 0; i0 < lens[0]; ++i0) {
      for (int64_t i1 = 0; i1 < lens[1]; ++i1) {
        for (int64_t i2 = 0; i2 < lens[2]; ++i2) {
          for (int64_t i3 = 0; i3 < lens[3]; ++i3) {
            for (int64_t i4 = 0; i4 < lens[4]; ++i4) {
              const int64_t view_offset =
                  i0 * view_stride[0] + i1 * view_stride[1] +
                  i2 * view_stride[2] + i3 * view_stride[3] +
                  i4 * view_stride[4];

              const int64_t src_i0 = starts[0] + i0 * strides[0];
              const int64_t src_i1 = starts[1] + i1 * strides[1];
              const int64_t src_i2 = starts[2] + i2 * strides[2];
              const int64_t src_i3 = starts[3] + i3 * strides[3];
              const int64_t src_i4 = starts[4] + i4 * strides[4];

              const int64_t src_offset =
                  src_i0 * src_stride[0] + src_i1 * src_stride[1] +
                  src_i2 * src_stride[2] + src_i3 * src_stride[3] +
                  src_i4 * src_stride[4];

              CHECK(view_data[static_cast<size_t>(view_offset)] ==
                    src_data[static_cast<size_t>(src_offset)]);
            }
          }
        }
      }
    }
  }
}

TEST_CASE("View assignment test") {
  // 1. Generate a random tensor
  // 2. Generate random view of that tensor
  // 3. Genrate a random tensor that looks like this view
  // 4. Use view assignment
  // 5. Ensure that the changes are reflected in the data buffer of a) the og
  // tensor b) the view
  CHECK(false);
}
