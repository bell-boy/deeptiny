# transfomer-demo

Standalone CMake demo that consumes Deep Tiny via `FetchContent`.

This demo now includes a simple PyTorch-style embedding module under
`src/modules/embedding.{h,cc}`:

- `Embedding::Forward(indices, shape)` expects
  `indices.size() == product(shape)`
- Output shape is `shape + {embedding_dim}`
- Invalid indices (`< 0` or `>= num_embeddings`) throw
- Forward is implemented with zero-init + slice assignment, and supports
  backward through Deep Tiny autograd

## Configure

```bash
cmake --preset dev
```

If OpenBLAS is not discoverable by default, create a local user preset:

```bash
cp CMakeUserPresets.json.example CMakeUserPresets.json
# edit CMAKE_PREFIX_PATH if needed
cmake --preset dev-local-openblas
```

## Build

```bash
cmake --build --preset dev
```

If you configured with `dev-local-openblas`, build with:

```bash
cmake --build --preset dev-local-openblas
```

Build demo embedding tests:

```bash
cmake --build --preset dev --target transfomer_demo_embedding_test
```

## Run

```bash
./build/transfomer_demo
```

Run embedding tests:

```bash
ctest --test-dir build --output-on-failure
```

Override the pinned Deep Tiny commit:

```bash
cmake --preset dev -DDEEPTINY_GIT_TAG=<commit-ish>
```

Point `FetchContent` at a local Deep Tiny checkout:

```bash
cmake --preset dev -DFETCHCONTENT_SOURCE_DIR_DEEPTINY=/path/to/deeptiny
```
