# deeptiny

`deeptiny` is a small C++23 tensor/autograd project focused on core mechanics:
views, broadcasting, elementwise math, and reverse-mode backpropagation.

## Current capabilities

- Tensor creation on CPU (`Tensor` constructor, `Tensor::FromBuffer`)
- Tensor views and slicing via `deeptiny::Slice`
- Broadcasting utilities used by math kernels
- Elementwise math: `+`, `-`, `*`, `/` (out-of-place and in-place)
- Batched matrix multiply: `math::BatchedMatMul`
- Reverse-mode autograd for elementwise ops, views, and reduction
- Reduction via `functional::Reduce`
- View-only `Tensor::Reshape` and `Tensor::Squeeze`

## Requirements

- CMake 3.23+
- C++23-compatible compiler
- OpenBLAS discoverable by CMake (`find_package(OpenBLAS REQUIRED)`)

## Build with presets

### Shared preset

Use this if OpenBLAS is already discoverable on your machine:

```bash
cmake --preset dev
cmake --build --preset dev -j
```

### Shared release preset (without tests)

Use this for optimized builds where test targets should not be generated:

```bash
cmake --preset release
cmake --build --preset release -j
```

### Local OpenBLAS path (macOS/Homebrew example)

Keep machine-specific paths in `CMakeUserPresets.json`:

```bash
cp CMakeUserPresets.json.example CMakeUserPresets.json
# edit CMAKE_PREFIX_PATH if needed
cmake --preset dev-local-openblas
cmake --build --preset dev-local-openblas -j
```

Release build with local OpenBLAS path:

```bash
cmake --preset release-local-openblas
cmake --build --preset release-local-openblas -j
```

## Run tests

Use a dev preset for tests. The `release` preset sets
`DEEPTINY_BUILD_TESTS=OFF`.

```bash
ctest --test-dir build --output-on-failure
```

Run a specific suite:

```bash
ctest --test-dir build -R math_test --output-on-failure
```

## Use deeptiny in your own CMake project

### FetchContent (recommended)

```cmake
include(FetchContent)

FetchContent_Declare(
  deeptiny
  GIT_REPOSITORY https://github.com/bell-boy/deeptiny.git
  GIT_TAG <pin-a-commit-sha-or-release-tag>
)
FetchContent_MakeAvailable(deeptiny)

add_executable(my_app main.cc)
target_link_libraries(my_app PRIVATE deeptiny::deeptiny)
```

Configure the consumer project with OpenBLAS discoverable:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/openblas
cmake --build build -j
```

If your OpenBLAS package provides `OpenBLASConfig.cmake`, you can also set
`-DOpenBLAS_DIR=/path/to/openblas/cmake`.

When used as a subproject (including FetchContent), deeptiny defaults to:

- `DEEPTINY_BUILD_TESTS=OFF`
- `DEEPTINY_ENABLE_WERROR=OFF`

You can override either from the parent configure command:

```bash
cmake -S . -B build -DDEEPTINY_BUILD_TESTS=ON -DDEEPTINY_ENABLE_WERROR=ON
```

If enabled from a FetchContent consumer build, run deeptiny tests with:

```bash
ctest --test-dir build/_deps/deeptiny-build --output-on-failure
```

`deeptiny` is intentionally FetchContent-first and does not provide
install/export package metadata for `find_package`.

## Minimal autograd example

```cpp
#include <array>
#include <span>

#include "deeptiny/functional.h"
#include "deeptiny/math.h"
#include "deeptiny/tensor.h"

int main() {
  using namespace deeptiny;

  std::array<float, 6> a_data{1, 2, 3, 4, 5, 6};
  std::array<float, 3> b_data{1, 2, 4};

  Tensor a = Tensor::FromBuffer(std::as_bytes(std::span<const float>(a_data)),
                                {2, 3}, DType::Float32, Device::CPU, true);
  Tensor b = Tensor::FromBuffer(std::as_bytes(std::span<const float>(b_data)),
                                {1, 3}, DType::Float32, Device::CPU, true);

  Tensor loss = functional::Reduce(a * b, {0, 1});
  loss.Backward();

  auto a_grad = a.grad();
  auto b_grad = b.grad();
}
```

## Notes and constraints

- `Tensor::Backward()` requires a scalar tensor and `requires_grad=true`.
- In-place math on zero-stride (broadcasted) destinations is rejected.
- `math::BatchedMatMul` broadcasts leading batch dims only; matrix inner dims must match.
- `Tensor::Reshape` requires contiguous input and matching element count.
- Current compute kernels are CPU-first and Float32-focused.
