# transfomer-demo

Standalone CMake demo that consumes Deep Tiny via `FetchContent`.

This demo includes transformer modules under `src/modules`:

- `Linear` and `LlamaMLP` demo modules
- `Embedding` with PyTorch-style lookup via `Embedding::operator()(indices, shape)`
- `Embedding::operator()(indices, shape)` expects
  `indices.size() == product(shape)`
- Embedding output shape is `shape + {embedding_dim}`
- Invalid embedding indices (`< 0` or `>= num_embeddings`) throw

## Configure

```bash
cmake --preset dev
```

When this demo lives inside a Deep Tiny checkout, it auto-uses the local
source tree via `FETCHCONTENT_SOURCE_DIR_DEEPTINY`.

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

Build demo test targets:

```bash
cmake --build build --target modules_smoke
cmake --build build --target transfomer_demo_embedding_test
```

## Run

```bash
./build/transfomer_demo
```

Run demo tests:

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
