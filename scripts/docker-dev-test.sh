#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-deeptiny-dev:local}"
USERNAME="${USERNAME:-dev}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"

if [[ $# -eq 0 ]]; then
  set -- dev
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running. Start Docker and retry." >&2
  exit 1
fi

docker build \
  --build-arg USERNAME="${USERNAME}" \
  --build-arg UID="${HOST_UID}" \
  --build-arg GID="${HOST_GID}" \
  -t "${IMAGE_TAG}" \
  .

docker run --rm -t \
  -v "$(pwd):/workspace" \
  -w /workspace \
  "${IMAGE_TAG}" \
  ./scripts/ci-local.sh "$@"
