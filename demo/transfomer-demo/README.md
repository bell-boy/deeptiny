# transfomer-demo

Standalone CMake demo that consumes Deep Tiny from source.

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

## Run

```bash
./build/transfomer_demo
```

If Deep Tiny is not located at `../..` relative to this folder, override it:

```bash
cmake --preset dev -DDEEPTINY_SOURCE_DIR=/path/to/deeptiny
```
