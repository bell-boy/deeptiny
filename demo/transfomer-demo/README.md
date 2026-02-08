# transfomer-demo

Standalone CMake demo that consumes Deep Tiny via `FetchContent`.
The demo also uses the `tokenizers-cpp` git submodule at
`third_party/tokenizers-cpp`.

This demo uses `deeptiny::nn` modules from the main library:

- `Embedding`, `MultiHeadAttention`, `GatedReLU`, and `RMSNorm`
- `nn::TransformerBlock` for repeated hidden-state updates
- demo-local `Transformer` module with pipeline: `embed -> transformer_block * N -> norm`
- `Embedding` with PyTorch-style lookup via `Embedding::operator()(indices, shape)`
- `Embedding::operator()(indices, shape)` expects
  `indices.size() == product(shape)`
- Embedding output shape is `shape + {embedding_dim}`
- Invalid embedding indices (`< 0` or `>= num_embeddings`) throw
- `src/smollm2_135m_instruct_loader.{h,cc}` exposes a factory that builds a
  demo `Transformer` from SmolLM2 safetensors using mmap-backed reads, `F32`
  direct copy, and `BF16 -> F32` conversion for runtime compatibility

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
- tokens are fed through the model
- next-token IDs are sampled from logits
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

## Benchmark (gperftools)

Build the benchmark executable:

```bash
cmake --build --preset benchmark
```

Run a forward-only benchmark on fixed input text `"hello world"`:

```bash
./build/transfomer_demo_benchmark /path/to/SmolLM2-135M-Instruct
```

If `model.safetensors` is missing in the provided `model_dir` (or if no
`model_dir` is provided), the benchmark downloads weights into
`./model_files/model.safetensors` and uses that directory automatically.

Run with default local model cache location (`./model_files`):

```bash
./build/transfomer_demo_benchmark
```

Generate a CPU profile (defaults to
`./transfomer_demo_benchmark.prof`):

```bash
./build/transfomer_demo_benchmark /path/to/SmolLM2-135M-Instruct
```

Override profile output path:

```bash
TRANSFOMER_DEMO_PROFILE_PATH=benchmark.prof ./build/transfomer_demo_benchmark /path/to/SmolLM2-135M-Instruct
```

Inspect the profile with `pprof` (from Go toolchain or another `pprof`
installation):

```bash
go tool pprof -text ./build/transfomer_demo_benchmark benchmark.prof
```
