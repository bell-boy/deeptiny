# transfomer-demo

Standalone CMake demo that consumes Deep Tiny from source.

## Configure

```bash
cmake --preset dev
```

## Build

```bash
cmake --build --preset dev
```

## Run

```bash
./build/transfomer_demo
```

If Deep Tiny is not located at `../..` relative to this folder, override it:

```bash
cmake --preset dev -DDEEPTINY_SOURCE_DIR=/path/to/deeptiny
```
