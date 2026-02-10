#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <array>
#include <random>
#include <vector>

#include "broadcast.h"
#include "deeptiny/functional.h"
#include "deeptiny/tensor.h"
#include "doctest/doctest.h"
#include "test_utils.h"
#include "utils.h"

namespace {
constexpr int kRank = 5;
}  // namespace

TEST_CASE("Slice convenience constructors") {
  SUBCASE("Slice(int64_t) creates a single-element range for positive index") {
    const int64_t index = 3;
    deeptiny::Slice slice(index);

    REQUIRE(slice.start.has_value());
    REQUIRE(slice.end.has_value());
    CHECK(*slice.start == 3);
    CHECK(*slice.end == 4);
    CHECK(slice.stride == 1);
  }

  SUBCASE("Slice(-1) selects the last element") {
    deeptiny::Tensor t = deeptiny::test_utils::MakeTensor(
        {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, false);

    deeptiny::Tensor row = t({deeptiny::Slice(-1), deeptiny::Slice::All()});
    CHECK(row.shape() == deeptiny::Shape{1, 3});
    deeptiny::test_utils::CheckTensorData(row, {4.0f, 5.0f, 6.0f});
  }

  SUBCASE("Slice::All() selects the full dimension") {
    deeptiny::Tensor t = deeptiny::test_utils::MakeTensor(
        {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, false);

    deeptiny::Tensor row = t({deeptiny::Slice(1), deeptiny::Slice::All()});
    CHECK(row.shape() == deeptiny::Shape{1, 3});
    deeptiny::test_utils::CheckTensorData(row, {4.0f, 5.0f, 6.0f});
  }
}

TEST_CASE("Random slice forward test") {
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int64_t> dim_dist(1, 7);

  for (int trial = 0; trial < 10; ++trial) {
    deeptiny::Shape shape(kRank);
    for (int i = 0; i < kRank; ++i) {
      shape[i] = static_cast<uint64_t>(dim_dist(rng));
    }

    auto t = deeptiny::Tensor::CreateUniform(shape, deeptiny::Device::CPU,
                                             deeptiny::DType::Float32);
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

      const int64_t end = start + (len - 1) * stride + 1;

      starts[i] = start;
      strides[i] = stride;
      lens[i] = len;
      slices[i] = deeptiny::Slice(start, end, stride);
    }

    std::vector<deeptiny::Slice> view_slices(slices.begin(), slices.end());
    deeptiny::Tensor slice_tensor = t(view_slices);
    auto slice_impl =
        deeptiny::utils::TensorAccessor::GetTensorImpl(slice_tensor);

    deeptiny::Shape expected_shape(kRank);
    for (int i = 0; i < kRank; ++i) {
      expected_shape[i] = static_cast<uint64_t>(lens[i]);
    }

    CHECK(slice_impl->shape() == expected_shape);

    const auto* slice_data = static_cast<const float*>(slice_impl->data());
    const auto& slice_stride = slice_impl->stride();
    const auto* src_data = static_cast<const float*>(t_impl->data());
    const auto& src_stride = t_impl->stride();

    for (int64_t i0 = 0; i0 < lens[0]; ++i0) {
      for (int64_t i1 = 0; i1 < lens[1]; ++i1) {
        for (int64_t i2 = 0; i2 < lens[2]; ++i2) {
          for (int64_t i3 = 0; i3 < lens[3]; ++i3) {
            for (int64_t i4 = 0; i4 < lens[4]; ++i4) {
              const int64_t slice_offset =
                  i0 * slice_stride[0] + i1 * slice_stride[1] +
                  i2 * slice_stride[2] + i3 * slice_stride[3] +
                  i4 * slice_stride[4];

              const int64_t src_i0 = starts[0] + i0 * strides[0];
              const int64_t src_i1 = starts[1] + i1 * strides[1];
              const int64_t src_i2 = starts[2] + i2 * strides[2];
              const int64_t src_i3 = starts[3] + i3 * strides[3];
              const int64_t src_i4 = starts[4] + i4 * strides[4];

              const int64_t src_offset =
                  src_i0 * src_stride[0] + src_i1 * src_stride[1] +
                  src_i2 * src_stride[2] + src_i3 * src_stride[3] +
                  src_i4 * src_stride[4];

              CHECK(slice_data[static_cast<size_t>(slice_offset)] ==
                    deeptiny::test_utils::Approx(
                        src_data[static_cast<size_t>(src_offset)]));
            }
          }
        }
      }
    }
  }
}

TEST_CASE("Slice assignment test") {
  std::mt19937 rng(54321);
  std::uniform_int_distribution<int64_t> dim_dist(1, 7);

  for (int trial = 0; trial < 10; ++trial) {
    deeptiny::Shape shape(kRank);
    for (int i = 0; i < kRank; ++i) {
      shape[i] = static_cast<uint64_t>(dim_dist(rng));
    }

    deeptiny::Tensor t(shape, deeptiny::DType::Float32, deeptiny::Device::CPU,
                       false);
    auto t_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(t);
    const uint64_t total_size = deeptiny::utils::GetTotalSize(shape);
    auto* t_data = static_cast<float*>(t_impl->data());
    for (uint64_t i = 0; i < total_size; ++i) {
      t_data[i] = static_cast<float>(i) + static_cast<float>(trial) * 0.01f;
    }
    std::vector<float> original_data(t_data, t_data + total_size);

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

      const int64_t end = start + (len - 1) * stride + 1;

      starts[i] = start;
      strides[i] = stride;
      lens[i] = len;
      slices[i] = deeptiny::Slice(start, end, stride);
    }

    std::vector<deeptiny::Slice> slice_specs(slices.begin(), slices.end());
    auto slice_proxy = t(slice_specs);
    deeptiny::Tensor slice_tensor = static_cast<deeptiny::Tensor>(slice_proxy);
    auto slice_impl =
        deeptiny::utils::TensorAccessor::GetTensorImpl(slice_tensor);

    deeptiny::Shape expected_shape(kRank);
    for (int i = 0; i < kRank; ++i) {
      expected_shape[i] = static_cast<uint64_t>(lens[i]);
    }

    CHECK(slice_impl->shape() == expected_shape);

    deeptiny::Tensor rhs(expected_shape, deeptiny::DType::Float32,
                         deeptiny::Device::CPU, false);
    auto rhs_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(rhs);
    const uint64_t rhs_total_size =
        deeptiny::utils::GetTotalSize(expected_shape);
    auto* rhs_data = static_cast<float*>(rhs_impl->data());
    for (uint64_t i = 0; i < rhs_total_size; ++i) {
      rhs_data[i] =
          1000.0f + static_cast<float>(i) + static_cast<float>(trial) * 0.001f;
    }

    std::vector<float> expected_data = original_data;
    const auto& src_stride = t_impl->stride();
    const auto& slice_stride = slice_impl->stride();
    const auto& rhs_stride = rhs_impl->stride();
    const auto* rhs_data_const = static_cast<const float*>(rhs_impl->data());

    for (int64_t i0 = 0; i0 < lens[0]; ++i0) {
      for (int64_t i1 = 0; i1 < lens[1]; ++i1) {
        for (int64_t i2 = 0; i2 < lens[2]; ++i2) {
          for (int64_t i3 = 0; i3 < lens[3]; ++i3) {
            for (int64_t i4 = 0; i4 < lens[4]; ++i4) {
              const int64_t src_i0 = starts[0] + i0 * strides[0];
              const int64_t src_i1 = starts[1] + i1 * strides[1];
              const int64_t src_i2 = starts[2] + i2 * strides[2];
              const int64_t src_i3 = starts[3] + i3 * strides[3];
              const int64_t src_i4 = starts[4] + i4 * strides[4];

              const int64_t src_offset =
                  src_i0 * src_stride[0] + src_i1 * src_stride[1] +
                  src_i2 * src_stride[2] + src_i3 * src_stride[3] +
                  src_i4 * src_stride[4];

              const int64_t rhs_offset =
                  i0 * rhs_stride[0] + i1 * rhs_stride[1] + i2 * rhs_stride[2] +
                  i3 * rhs_stride[3] + i4 * rhs_stride[4];

              expected_data[static_cast<size_t>(src_offset)] =
                  rhs_data_const[static_cast<size_t>(rhs_offset)];
            }
          }
        }
      }
    }

    slice_proxy = rhs;

    const auto* t_data_after = static_cast<const float*>(t_impl->data());
    for (uint64_t i = 0; i < total_size; ++i) {
      CHECK(t_data_after[i] == deeptiny::test_utils::Approx(expected_data[i]));
    }

    const auto* slice_data = static_cast<const float*>(slice_impl->data());
    for (int64_t i0 = 0; i0 < lens[0]; ++i0) {
      for (int64_t i1 = 0; i1 < lens[1]; ++i1) {
        for (int64_t i2 = 0; i2 < lens[2]; ++i2) {
          for (int64_t i3 = 0; i3 < lens[3]; ++i3) {
            for (int64_t i4 = 0; i4 < lens[4]; ++i4) {
              const int64_t slice_offset =
                  i0 * slice_stride[0] + i1 * slice_stride[1] +
                  i2 * slice_stride[2] + i3 * slice_stride[3] +
                  i4 * slice_stride[4];

              const int64_t rhs_offset =
                  i0 * rhs_stride[0] + i1 * rhs_stride[1] + i2 * rhs_stride[2] +
                  i3 * rhs_stride[3] + i4 * rhs_stride[4];

              CHECK(slice_data[static_cast<size_t>(slice_offset)] ==
                    deeptiny::test_utils::Approx(
                        rhs_data_const[static_cast<size_t>(rhs_offset)]));
            }
          }
        }
      }
    }
  }
}

TEST_CASE("Slice assignment guards and autograd metadata") {
  SUBCASE("Slice assignment installs new metadata on destination tensor") {
    deeptiny::Tensor t({2, 3}, deeptiny::DType::Float32, deeptiny::Device::CPU,
                       false);
    auto before = deeptiny::utils::TensorAccessor::GetAutogradMeta(t);
    REQUIRE(before != nullptr);

    deeptiny::Tensor rhs({2, 3}, deeptiny::DType::Float32,
                         deeptiny::Device::CPU, true);
    t({deeptiny::Slice(0, 2), deeptiny::Slice(0, 3)}) = rhs;

    auto after = deeptiny::utils::TensorAccessor::GetAutogradMeta(t);
    REQUIRE(after != nullptr);
    CHECK(after.get() != before.get());
  }

  SUBCASE("Slice assignment forbids zero-stride views") {
    deeptiny::Tensor base({2, 1}, deeptiny::DType::Float32,
                          deeptiny::Device::CPU, false);
    auto broadcasted = deeptiny::utils::BroadcastToShape(base, {2, 3});
    REQUIRE(broadcasted.has_value());

    deeptiny::Tensor rhs({2, 3}, deeptiny::DType::Float32,
                         deeptiny::Device::CPU, false);
    CHECK_THROWS_WITH(
        (*broadcasted)({deeptiny::Slice(0, 2), deeptiny::Slice(0, 3)}) = rhs,
        doctest::Contains("Cannot assign to a slice with zero stride"));
  }

  SUBCASE("Temporary assignment path backpropagates to RHS and prior chain") {
    deeptiny::Tensor t({2, 3}, deeptiny::DType::Float32, deeptiny::Device::CPU,
                       true);
    deeptiny::Tensor t_before = t;

    deeptiny::Tensor rhs({2, 2}, deeptiny::DType::Float32,
                         deeptiny::Device::CPU, true);
    t({deeptiny::Slice(0, 2), deeptiny::Slice(1, 3)}) = rhs;

    auto loss = deeptiny::functional::Reduce(t, {0, 1});
    loss.Backward();

    auto rhs_grad = rhs.grad();
    REQUIRE(rhs_grad.has_value());
    deeptiny::test_utils::CheckTensorData(*rhs_grad, {1.0f, 1.0f, 1.0f, 1.0f});

    auto t_before_grad = t_before.grad();
    REQUIRE(t_before_grad.has_value());
    deeptiny::test_utils::CheckTensorData(*t_before_grad,
                                          {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f});
  }

  SUBCASE("Broadcasted RHS assignment reduces gradients correctly") {
    deeptiny::Tensor t({2, 3}, deeptiny::DType::Float32, deeptiny::Device::CPU,
                       true);
    deeptiny::Tensor t_before = t;

    deeptiny::Tensor rhs({2, 1}, deeptiny::DType::Float32,
                         deeptiny::Device::CPU, true);
    t({deeptiny::Slice(0, 2), deeptiny::Slice(0, 3)}) = rhs;

    auto loss = deeptiny::functional::Reduce(t, {0, 1});
    loss.Backward();

    auto rhs_grad = rhs.grad();
    REQUIRE(rhs_grad.has_value());
    deeptiny::test_utils::CheckTensorData(*rhs_grad, {3.0f, 3.0f});

    auto t_before_grad = t_before.grad();
    REQUIRE(t_before_grad.has_value());
    deeptiny::test_utils::CheckTensorData(*t_before_grad,
                                          {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  }
}

TEST_CASE("Slice backward test") {
  SUBCASE("Strided slice scatter matches expected gradient") {
    deeptiny::Tensor t({4, 5}, deeptiny::DType::Float32, deeptiny::Device::CPU,
                       true);
    auto t_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(t);
    auto* t_data = static_cast<float*>(t_impl->data());
    const uint64_t total_size = deeptiny::utils::GetTotalSize(t.shape());
    for (uint64_t i = 0; i < total_size; ++i) {
      t_data[i] = static_cast<float>(i);
    }

    std::vector<deeptiny::Slice> slices{
        deeptiny::Slice(1, 4, 2),
        deeptiny::Slice(0, 5, 2),
    };
    deeptiny::Tensor slice_tensor = t(slices);
    auto loss = deeptiny::functional::Reduce(slice_tensor, {0, 1});
    loss.Backward();

    auto grad_opt = t.grad();
    REQUIRE(grad_opt.has_value());
    auto grad_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(*grad_opt);
    const auto* grad_data = static_cast<const float*>(grad_impl->data());

    std::vector<float> expected(total_size, 0.0f);
    const auto& stride = t_impl->stride();
    for (int64_t i0 = 0; i0 < 2; ++i0) {
      for (int64_t i1 = 0; i1 < 3; ++i1) {
        const int64_t src_i0 = 1 + i0 * 2;
        const int64_t src_i1 = 0 + i1 * 2;
        const int64_t src_offset = src_i0 * stride[0] + src_i1 * stride[1];
        expected[static_cast<size_t>(src_offset)] += 1.0f;
      }
    }

    for (uint64_t i = 0; i < total_size; ++i) {
      CHECK(grad_data[i] == deeptiny::test_utils::Approx(expected[i]));
    }
  }

  SUBCASE("Slice + broadcast chain accumulates gradients") {
    deeptiny::Tensor t({2, 3}, deeptiny::DType::Float32, deeptiny::Device::CPU,
                       true);
    std::vector<deeptiny::Slice> slices{
        deeptiny::Slice(0, 2, 1),
        deeptiny::Slice(0, 1, 1),
    };
    deeptiny::Tensor slice_tensor = t(slices);
    auto broadcasted = deeptiny::utils::BroadcastToShape(slice_tensor, {2, 4});
    REQUIRE(broadcasted.has_value());

    auto loss = deeptiny::functional::Reduce(*broadcasted, {0, 1});
    loss.Backward();

    auto grad_opt = t.grad();
    REQUIRE(grad_opt.has_value());
    auto grad_impl = deeptiny::utils::TensorAccessor::GetTensorImpl(*grad_opt);
    const auto* grad_data = static_cast<const float*>(grad_impl->data());

    const auto& stride = grad_impl->stride();
    std::vector<float> expected(6, 0.0f);
    for (int64_t i0 = 0; i0 < 2; ++i0) {
      const int64_t offset = i0 * stride[0] + 0 * stride[1];
      expected[static_cast<size_t>(offset)] = 4.0f;
    }

    for (size_t i = 0; i < expected.size(); ++i) {
      CHECK(grad_data[i] == deeptiny::test_utils::Approx(expected[i]));
    }
  }
}

TEST_CASE("Backward input validation") {
  SUBCASE("Backward fails on non-scalar") {
    deeptiny::Tensor t({2, 2}, deeptiny::DType::Float32, deeptiny::Device::CPU,
                       true);
    CHECK_THROWS(t.Backward());
  }

  SUBCASE("Backward fails when requires_grad is false") {
    deeptiny::Tensor t({}, deeptiny::DType::Float32, deeptiny::Device::CPU,
                       false);
    CHECK_THROWS(t.Backward());
  }
}
