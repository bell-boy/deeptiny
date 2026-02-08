# transfomer-demo

Standalone CMake demo that consumes Deep Tiny via `FetchContent`.

This demo uses `deeptiny::nn` modules from the main library:

- `Embedding`, `MultiHeadAttention`, `GatedReLU`, and `RMSNorm`
- `nn::TransformerBlock` for repeated hidden-state updates
- demo-local `Transformer` module with pipeline: `embed -> transformer_block * N -> norm`
- `Embedding` with PyTorch-style lookup via `Embedding::operator()(indices, shape)`
- `Embedding::operator()(indices, shape)` expects
  `indices.size() == product(shape)`
- Embedding output shape is `shape + {embedding_dim}`
- Invalid embedding indices (`< 0` or `>= num_embeddings`) throw
- `src/smollm2_135m_instruct_loader.{h,cc}` provides a phase-1 SmolLM2 loader
  scaffold (default config + HF-to-DeepTiny tensor mapping + local artifact
  validation for `model.safetensors` / `model.safetensors.index.json`)

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

## Run

```bash
./build/transfomer_demo
```

Validate a local SmolLM2 checkpoint directory:

```bash
./build/transfomer_demo /path/to/SmolLM2-135M-Instruct
```

Override the pinned Deep Tiny commit:

```bash
cmake --preset dev -DDEEPTINY_GIT_TAG=<commit-ish>
```

Point `FetchContent` at a local Deep Tiny checkout:

```bash
cmake --preset dev -DFETCHCONTENT_SOURCE_DIR_DEEPTINY=/path/to/deeptiny
```
