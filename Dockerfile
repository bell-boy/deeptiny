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

RUN groupadd -g ${GID} ${USERNAME} \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

WORKDIR /workspace
COPY . /workspace
USER ${USERNAME}

CMD ["bash"]
