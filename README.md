# deeptiny

`deeptiny` is a small C++23 tensor/autograd project focused on core mechanics:
views, broadcasting, elementwise math, and reverse-mode backpropagation.

## Current capabilities

- Tensor creation on CPU (`Tensor` constructor, `Tensor::FromBuffer`)
- Tensor views and slicing via `deeptiny::Slice`
- Broadcasting utilities used by math kernels
- Elementwise math: `+`, `-`, `*`, `/` (out-of-place and in-place)
- Reverse-mode autograd for elementwise ops, views, and reduction
- Reduction via `functional::Reduce`

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

Use a dev preset for tests. The `release` preset sets `BUILD_TESTING=OFF`.

```bash
ctest --test-dir build --output-on-failure
```

Run a specific suite:

```bash
ctest --test-dir build -R math_test --output-on-failure
```

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
- Current compute kernels are CPU-first and Float32-focused.
