# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=11.8.0
ARG UBUNTU_VERSION=22.04
ARG CUDNN_FLAVOR=cudnn8
ARG LLVM_VERSION=18
ARG ENZYME_REF=v0.0.261

FROM nvidia/cuda:${CUDA_VERSION}-${CUDNN_FLAVOR}-devel-ubuntu${UBUNTU_VERSION} AS enzyme-builder

ARG LLVM_VERSION
ARG ENZYME_REF

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    wget \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/llvm-snapshot.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm-snapshot.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-${LLVM_VERSION} main" \
        > /etc/apt/sources.list.d/llvm.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        clang-${LLVM_VERSION} \
        cmake \
        git \
        libclang-${LLVM_VERSION}-dev \
        libzstd-dev \
        lld-${LLVM_VERSION} \
        llvm-${LLVM_VERSION} \
        llvm-${LLVM_VERSION}-dev \
        llvm-${LLVM_VERSION}-tools \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/enzyme-src \
    && curl -LsSf "https://github.com/EnzymeAD/Enzyme/archive/refs/tags/${ENZYME_REF}.tar.gz" \
        | tar -xz --strip-components=1 -C /tmp/enzyme-src \
    && cmake -S /tmp/enzyme-src/enzyme -B /tmp/enzyme-src/enzyme/build -G Ninja \
        -DLLVM_DIR="/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/llvm" \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/enzyme-src/enzyme/build \
    && mkdir -p /opt/enzyme/lib \
    && find /tmp/enzyme-src/enzyme/build -type f -name '*Enzyme*.so' -exec cp -v {} /opt/enzyme/lib/ \; \
    && test -n "$(find /opt/enzyme/lib -type f -name '*Enzyme*.so' -print -quit)"

FROM nvidia/cuda:${CUDA_VERSION}-${CUDNN_FLAVOR}-devel-ubuntu${UBUNTU_VERSION}

ARG PYTHON_VERSION=3.12
ARG UV_INSTALLER_URL=https://astral.sh/uv/install.sh
ARG LLVM_VERSION

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    wget \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/llvm-snapshot.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm-snapshot.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-${LLVM_VERSION} main" \
        > /etc/apt/sources.list.d/llvm.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        clang-${LLVM_VERSION} \
        lld-${LLVM_VERSION} \
        llvm-${LLVM_VERSION} \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s "/usr/bin/opt-${LLVM_VERSION}" /usr/local/bin/opt \
    && ln -s "/usr/bin/llc-${LLVM_VERSION}" /usr/local/bin/llc \
    && ln -s "/usr/bin/llvm-as-${LLVM_VERSION}" /usr/local/bin/llvm-as \
    && ln -s "/usr/bin/llvm-dis-${LLVM_VERSION}" /usr/local/bin/llvm-dis \
    && ln -s "/usr/bin/llvm-link-${LLVM_VERSION}" /usr/local/bin/llvm-link \
    && ln -s "/usr/bin/clang-${LLVM_VERSION}" /usr/local/bin/clang \
    && ln -s "/usr/bin/clang++-${LLVM_VERSION}" /usr/local/bin/clang++

COPY --from=enzyme-builder /opt/enzyme/lib /opt/enzyme/lib

ENV ENZYME_DIR=/opt/enzyme \
    ENZYME_LIB_DIR=/opt/enzyme/lib

RUN curl -LsSf "${UV_INSTALLER_URL}" | env UV_INSTALL_DIR=/usr/local/bin sh

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /app

COPY --link pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-install-package torch --python "${PYTHON_VERSION}" \
    && uv pip install --python "${VIRTUAL_ENV}" \
        --index-url https://download.pytorch.org/whl/cu118 \
        "torch==2.7.1+cu118"

ENTRYPOINT []
CMD ["bash"]
