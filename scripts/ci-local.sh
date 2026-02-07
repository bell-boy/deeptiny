#!/usr/bin/env bash
set -euo pipefail

CONFIGURE_PRESET="${1:-dev}"
BUILD_PRESET="${2:-${CONFIGURE_PRESET}}"
CTEST_DIR="${DEEPTINY_CTEST_DIR:-build}"

if command -v nproc >/dev/null 2>&1; then
  JOBS="$(nproc)"
elif command -v getconf >/dev/null 2>&1; then
  JOBS="$(getconf _NPROCESSORS_ONLN)"
elif command -v sysctl >/dev/null 2>&1; then
  JOBS="$(sysctl -n hw.ncpu)"
else
  JOBS=4
fi

cmake --preset "${CONFIGURE_PRESET}"
cmake --build --preset "${BUILD_PRESET}" -j"${JOBS}"

if [[ "${CONFIGURE_PRESET}" == release* || "${BUILD_PRESET}" == release* ]]; then
  echo "Skipping tests for release preset."
  exit 0
fi

ctest --test-dir "${CTEST_DIR}" --output-on-failure
