# transfomer-demo

Standalone CMake demo that consumes Deep Tiny via `FetchContent`.
The demo also uses the `tokenizers-cpp` git submodule at
`third_party/tokenizers-cpp`.

This demo uses `deeptiny::nn` modules from the main library:

- `Embedding`, `MultiHeadAttention`, `GatedMLP`, and `RMSNorm`
- `nn::TransformerBlock` for repeated hidden-state updates
- demo-local `Transformer` module with pipeline: `embed -> transformer_block * N -> norm`
- demo-local `Transformer::Generate(...)` for autoregressive completion
- `Embedding` with PyTorch-style lookup via `Embedding::operator()(indices, shape)`
- `Embedding::operator()(indices, shape)` expects
  `indices.size() == product(shape)`
- Embedding output shape is `shape + {embedding_dim}`
- Invalid embedding indices (`< 0` or `>= num_embeddings`) throw
- `src/smollm2_135m_instruct_loader.{h,cc}` exposes a factory that builds a
  demo `Transformer` from SmolLM2 safetensors using mmap-backed reads, `F32`
  direct copy, and `BF16 -> F32` conversion for runtime compatibility
  - also exposes an uninitialized factory that creates the SmolLM2 model
    structure without loading weights from disk

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
./build/transfomer_demo /path/to/SmolLM2-135M-Instruct
```

Optional generation args:

```bash
./build/transfomer_demo /path/to/SmolLM2-135M-Instruct 64 0.8
```

The demo runs a simple CLI chat loop:

- input line is tokenized
- `Transformer::Generate(...)` runs autoregressive token generation
- generated token IDs are decoded and printed

If `tokenizer.json` is missing in your provided model directory, the demo
downloads `tokenizer.json` into `model_files/` under the current working
directory and uses it.

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

## Generation Benchmark (No Weight Load)

The benchmark target constructs the SmolLM2-135M model in memory without
reading `model.safetensors`, starts from exactly one prompt token
(`bos_token_id`), and generates exactly 128 new tokens.

Configure and build with profiling flags (`-pg`):

```bash
cmake --preset bench-pg
cmake --build --preset bench-pg
```

Run:

```bash
./build/transfomer_generation_benchmark
```

Optional: pass `max_new_tokens` for a shorter smoke run (default is `128`):

```bash
./build/transfomer_generation_benchmark 4
```
