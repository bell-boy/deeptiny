# transfomer-demo

Standalone CMake demo that consumes Deep Tiny via `FetchContent`.
The demo also uses the `tokenizers-cpp` git submodule at
`third_party/tokenizers-cpp`.

This demo uses `deeptiny::nn` modules from the main library:

- `Linear` and `GatedReLU`
- `Embedding` with PyTorch-style lookup via `Embedding::operator()(indices, shape)`
- `Embedding::operator()(indices, shape)` expects
  `indices.size() == product(shape)`
- Embedding output shape is `shape + {embedding_dim}`
- Invalid embedding indices (`< 0` or `>= num_embeddings`) throw

## Configure

```bash
cmake --preset dev
```

`tokenizers-cpp` builds through Cargo, so Rust is required when
`TRANSFOMER_DEMO_ENABLE_TOKENIZERS_CPP=ON` (default).
Initialize submodules (including nested ones) before configuring:

```bash
git submodule update --init --recursive
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

## Run

```bash
./build/transfomer_demo
```

Override the pinned Deep Tiny commit:

```bash
cmake --preset dev -DDEEPTINY_GIT_TAG=<commit-ish>
```

Point `FetchContent` at a local Deep Tiny checkout:

```bash
cmake --preset dev -DFETCHCONTENT_SOURCE_DIR_DEEPTINY=/path/to/deeptiny
```

Disable tokenizer integration if needed:

```bash
cmake --preset dev -DTRANSFOMER_DEMO_ENABLE_TOKENIZERS_CPP=OFF
```
