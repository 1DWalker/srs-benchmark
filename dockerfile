# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

ARG PYTHON_VERSION=3.12
ARG UV_INSTALLER_URL=https://astral.sh/uv/install.sh

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv

# Avoid accidentally overriding JAX's pip-installed CUDA libraries.
ENV LD_LIBRARY_PATH=""

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf "${UV_INSTALLER_URL}" | env UV_INSTALL_DIR=/usr/local/bin sh

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /app

COPY --link pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --python "python${PYTHON_VERSION}"

# Add JAX CUDA 12 support.
# If you add "jax[cuda12]" to pyproject.toml later, remove this separate line.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "${VIRTUAL_ENV}/bin/python" -U "jax[cuda12]"

CMD ["bash"]
