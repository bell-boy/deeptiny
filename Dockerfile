FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=dev
ARG UID=1000
ARG GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git vim nano less \
    build-essential pkg-config cmake ninja-build \
    python3 python3-venv python3-pip \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    if getent group "${GID}" >/dev/null 2>&1; then \
      GROUP_NAME="$(getent group "${GID}" | cut -d: -f1)"; \
    else \
      if ! getent group "${USERNAME}" >/dev/null 2>&1; then \
        groupadd -g "${GID}" "${USERNAME}" || groupadd "${USERNAME}"; \
      fi; \
      GROUP_NAME="${USERNAME}"; \
    fi; \
    if id -u "${USERNAME}" >/dev/null 2>&1; then \
      usermod -g "${GROUP_NAME}" "${USERNAME}" || true; \
    else \
      useradd -m -u "${UID}" -g "${GROUP_NAME}" -s /bin/bash "${USERNAME}" || \
      useradd -m -g "${GROUP_NAME}" -s /bin/bash "${USERNAME}"; \
    fi

WORKDIR /workspace
COPY . /workspace
USER ${USERNAME}

CMD ["bash"]
