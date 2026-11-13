# syntax=docker/dockerfile:1.7
# Kimi-K3.1 production-verification serving image, built from this checkout with the same
# lineage as an official SGLang release rather than a published nightly base.
#
# Stage layout:
#   1. base .. framework_final: the stages of docker/Dockerfile (the build the release
#      workflow publishes as lmsysorg/sglang:v*) with BRANCH_TYPE locked to local, so the
#      dependency floor (flashinfer 0.7.3, sgl-deep-gemm, sgl-flash-mla, mooncake, ...) is
#      pinned by THIS checkout's python/pyproject.toml instead of drifting a release or
#      nightly tag. Deliberately skipped, mirroring what the official runtime image never
#      shipped: the devtools stage (gdb/vim/tmux/nsight) and the docker/configs dotfiles.
#   2. runtime: the official production runtime stage, ported verbatim from the
#      v0.5.23 docker/Dockerfile (upstream removed it in #48983; v0.5.23 is the last
#      release that published *-runtime images). It exists for exactly the problem this
#      image had on a dev base: JIT compiles at server start link against libraries a
#      framework_final image does not register system-wide. Keep it in sync manually when
#      docker/Dockerfile moves; the file carries CUDA compiler paths on PATH, apt
#      libnccl2/libnccl-dev for pynccl_allocator's `-lnccl` link, the RDMA runtime libs,
#      and a closing ldconfig.
#   3. k31: the K3.1 patch chain on top of runtime. The compiled parts the K3.1
#      optimizations rely on (the server starts and answers correctly without them, only
#      slower -- silent fallback):
#        1. Rust radix TreeCore   built by the editable install in framework_final
#        2. FlashInfer DynBlock   TRT-LLM MoE routing jit-cache modules rebuilt for 1024
#                                 experts (K3.1 has 896); needs INSTALL_FLASHINFER_JIT_CACHE=1
#        3. FlashInfer Prims-TS   main 784bfaf5's prims_ts/ over 0.7.3, for --dsa-*-backend
#                                 prims_ts
#        4. Dynamo 1.6.0          ai-dynamo-runtime rebuilt with the session-affinity patches
#                                 and the K3.1 edits to dynamo.sglang
#      FlashMLA and DeepGEMM come from the sgl-flash-mla / sgl-deep-gemm versions the
#      checkout's python/pyproject.toml pins. verify_image.py fails the build if any
#      compiled part fell back.
#
# Build from the repo root (keep .git; setuptools-scm derives the version from it), on amd64
# or arm64 with BuildKit; no GPU is needed:
#   docker buildx build -f docker/k31.Dockerfile -t <tag> --load .
# Throttled network: the official mirror args (UBUNTU_MIRROR, PIP_DEFAULT_INDEX,
# GITHUB_ARTIFACTORY) work unchanged; the k31 chain adds its own:
#   --build-arg PIP_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
#   --build-arg K31_RUST_MIRROR=rsproxy.cn .
# The last step runs verify_image.py; it needs no GPU, so a built image can be rechecked with:
#   docker run --rm <tag> python3 /opt/k31-image/verify_image.py

ARG CUDA_VERSION=13.0.3
ARG UV_VERSION=latest
ARG PYTHON_VERSION=3.12
ARG ENABLE_SGL_NVIDIA_MEGA=1

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS base

ARG TARGETARCH
ARG BUILD_TYPE=all
ARG SGL_KERNEL_VERSION=0.4.9
ARG GDRCOPY_VERSION=2.5.1
ARG SGL_NCCL_VERSION=2.30.7
ARG PIP_DEFAULT_INDEX
ARG UBUNTU_MIRROR
ARG GITHUB_ARTIFACTORY=github.com
ARG INSTALL_FLASHINFER_JIT_CACHE=1
ARG FLASHINFER_VERSION=0.7.3
ARG MOONCAKE_VERSION=0.3.14
ARG MSCCLPP_VERSION=sglang-v0.10.1

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-${GDRCOPY_VERSION}/ \
    FLASHINFER_VERSION=${FLASHINFER_VERSION}

# Add GKE default lib and bin locations
ENV PATH="${PATH}:/usr/local/nvidia/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# Replace Ubuntu sources if specified
RUN if [ -n "$UBUNTU_MIRROR" ]; then \
    sed -i "s|http://.*archive.ubuntu.com|$UBUNTU_MIRROR|g" /etc/apt/sources.list && \
    sed -i "s|http://.*security.ubuntu.com|$UBUNTU_MIRROR|g" /etc/apt/sources.list; \
fi

# create virtual env for sglang to avoid conflict with system python packages
ARG PYTHON_VERSION
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/sglang
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "${PIP_DEFAULT_INDEX}" ]; then \
        mkdir -p /etc/uv \
        && printf '[[index]]\nurl = "%s"\ndefault = true\n' "${PIP_DEFAULT_INDEX}" > /etc/uv/uv.toml; \
    fi \
    && if [ "${GITHUB_ARTIFACTORY}" != "github.com" ]; then \
        export UV_PYTHON_INSTALL_MIRROR="https://${GITHUB_ARTIFACTORY}/astral-sh/python-build-standalone/releases/download"; \
    fi \
    && uv --version \
    && uv python install ${PYTHON_VERSION} \
    && uv venv --seed --python ${PYTHON_VERSION} /opt/sglang \
    && uv pip install --upgrade pip \
    && /opt/sglang/bin/python -VV
ENV PATH="/opt/sglang/bin:${PATH}"

# Install system dependencies (organized by category for better caching)
RUN --mount=type=cache,target=/var/cache/apt,id=base-apt \
    apt-get update && apt-get install -y --no-install-recommends \
    # Core system utilities
    ca-certificates \
    netcat-openbsd \
    kmod \
    unzip \
    openssh-server \
    curl \
    wget \
    lsof \
    locales \
    # Build essentials (needed for framework stage)
    build-essential \
    cmake \
    perl \
    patchelf \
    ccache \
    git-lfs \
    # MPI and NUMA
    libopenmpi-dev \
    libnuma1 \
    libnuma-dev \
    numactl \
    # transformers multimodal VLM
    ffmpeg \
    # InfiniBand/RDMA
    libibverbs-dev \
    libibverbs1 \
    libibumad3 \
    librdmacm1 \
    libnl-3-200 \
    libnl-route-3-200 \
    libnl-route-3-dev \
    libnl-3-dev \
    ibverbs-providers \
    infiniband-diags \
    perftest \
    # Development libraries
    libgoogle-glog-dev \
    libgtest-dev \
    libjsoncpp-dev \
    libunwind-dev \
    libboost-all-dev \
    libssl-dev \
    libgrpc-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler \
    protobuf-compiler-grpc \
    pybind11-dev \
    libhiredis-dev \
    libcurl4-openssl-dev \
    libczmq4 \
    libczmq-dev \
    libfabric-dev \
    linux-libc-dev \
    # Package building tools
    devscripts \
    debhelper \
    fakeroot \
    dkms \
    check \
    libsubunit0 \
    libsubunit-dev \
    && rm -f /usr/bin/python \
    && printf '#!/bin/sh\nexec /opt/sglang/bin/python "$@"\n' > /usr/bin/python \
    && chmod +x /usr/bin/python \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Replace pip global cache if specified
RUN if [ -n "${PIP_DEFAULT_INDEX}" ]; then \
    python3 -m pip config set global.index-url ${PIP_DEFAULT_INDEX}; \
fi

# GDRCopy installation
RUN mkdir -p /tmp/gdrcopy && cd /tmp \
    && curl --retry 3 --retry-delay 2 -fsSL -o v${GDRCOPY_VERSION}.tar.gz \
        https://${GITHUB_ARTIFACTORY}/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz \
    && tar -xzf v${GDRCOPY_VERSION}.tar.gz && rm v${GDRCOPY_VERSION}.tar.gz \
    && cd gdrcopy-${GDRCOPY_VERSION}/packages \
    && CUDA=/usr/local/cuda ./build-deb-packages.sh \
    && dpkg -i gdrdrv-dkms_*.deb libgdrapi_*.deb gdrcopy-tests_*.deb gdrcopy_*.deb \
    && cd / && rm -rf /tmp/gdrcopy

# Fix DeepEP IBGDA symlink
RUN ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

########################################################
########## PARALLEL BUILDER STAGES ####################
########################################################
#
# These stages run IN PARALLEL via BuildKit:
#
#   base
#     |
#     +-- torch_deps ------> flashinfer_cache
#     |                  \-> hpc_ops_builder (cmake-only build)
#     |                  \-> halops_wheel (prebuilt wheel download)
#     |
#     +-- gateway_builder  (independent, only needs gateway source)
#     |
#     v
#   framework (combines all artifacts)
#
# docker/Dockerfile's devtools_builder is absent: its tools never reached the official
# runtime image either, so this production image skips them. Use k31-dev.Dockerfile for
# a debugging shell.

########################################################
# PARALLEL STAGE 1: Torch/Deps Builder (starts from base)
########################################################
FROM base AS torch_deps

ARG CUDA_VERSION
ARG BUILD_TYPE
ARG SGL_KERNEL_VERSION
ARG GITHUB_ARTIFACTORY

WORKDIR /sgl-workspace

# Rust toolchain for setuptools-rust extensions (e.g. sglang-grpc).
# Requires >= 1.85 (edition 2024). Inherited by framework via FROM torch_deps.
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal \
    && rustc --version && cargo --version

# Install sgl-kernel (from pre-built wheel)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --upgrade pip setuptools wheel html5lib six \
    && case "$CUDA_VERSION" in \
        13.0.3) ;; \
        *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
    # --no-deps prevents uv from pulling torch from default PyPI
    && uv pip install sglang-kernel==${SGL_KERNEL_VERSION} --reinstall --no-deps

# Copy the dependency spec, Rust extension sources, and shared API types.
# Split from the pip install so changes to these build inputs invalidate the
# dependency layer, while unrelated Python source changes do not.
COPY python/pyproject.toml /tmp/sglang_deps/python/pyproject.toml
COPY rust/sglang-grpc      /tmp/sglang_deps/rust/sglang-grpc
COPY rust/sglang-mm        /tmp/sglang_deps/rust/sglang-mm
COPY rust/sglang-runtime-v1-types /tmp/sglang_deps/rust/sglang-runtime-v1-types

# Install sglang dependencies (torch, transformers, etc.).
# Generate constraints.txt to prevent reinstalling these deps in later stages.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cargo/registry \
    case "$CUDA_VERSION" in \
        13.0.3) CUINDEX=130 ;; \
        *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
    && cd /tmp/sglang_deps/python \
    && mkdir -p sglang \
    && touch sglang/__init__.py \
    && echo '__version__ = "0.0.0"' > sglang/version.py \
    && touch README.md \
    && touch LICENSE \
    && uv pip install --extra-index-url https://download.pytorch.org/whl/cu${CUINDEX} \
        --index-strategy unsafe-best-match ".[${BUILD_TYPE}]" \
    && cd /sgl-workspace \
    && rm -rf /tmp/sglang_deps \
    && uv pip freeze | grep -vE "^sglang( |==|@)" > /sgl-workspace/constraints.txt

# Cosmos3 guardrails default ON: sglang enables them whenever `cosmos_guardrail`
# is importable (Cosmos3Pipeline.create_pipeline_stages); opt out at runtime with
# SGLANG_DISABLE_COSMOS3_GUARDRAILS=1. Installed with --no-deps because the
# package metadata conflicts with our pins in ways that don't matter at runtime
# (imageio>=2.37 vs our ==2.36) or would be harmful (opencv-python — also pulled
# by retinaface-py — would shadow our opencv-python-headless cv2). The deps not
# already in the image are installed explicitly; the find_spec check mirrors
# sglang's availability probe plus cosmos_guardrail's own imports.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps \
        cosmos-guardrail==0.3.1 \
        retinaface-py==0.0.2 \
    && uv pip install \
        "better-profanity==0.7.0" \
        "nltk==3.9.1" \
        "peft==0.18.1" \
    && python3 -c "import importlib.util as u; missing = [m for m in ('cosmos_guardrail', 'retinaface', 'better_profanity', 'nltk', 'cv2', 'peft', 'sentencepiece') if u.find_spec(m) is None]; assert not missing, f'guardrail deps missing: {missing}'"

########################################################
# PARALLEL STAGE 2: HPC-Ops Builder (needs torch_deps)
########################################################
FROM torch_deps AS hpc_ops_builder

# HPC-Ops (https://github.com/Tencent/hpc-ops, MIT): fused attention / MoE /
# RoPE kernels from the Tencent Hunyuan AI Infra team, consumed by the opt-in
# hpc_ops attention and MoE runner backends.
ARG HPC_OPS_COMMIT=ab1a402724635507037426068f6cddc3d30dc0a8

WORKDIR /build

# The kernels target Hopper (sm90a) only, so skip non-x86_64 images.
# setup.py derives the version from `git rev-parse`, so keep the .git dir
# (a source zip archive would not build).
RUN mkdir -p /wheels && \
    if [ "$(uname -m)" = "x86_64" ]; then \
        git clone https://github.com/Tencent/hpc-ops.git && \
        cd hpc-ops && \
        git checkout ${HPC_OPS_COMMIT} && \
        python3 setup.py bdist_wheel -d /wheels; \
    fi

########################################################
# PARALLEL STAGE 2b: HALOps Wheel (needs torch_deps)
########################################################
FROM torch_deps AS halops_wheel

# HALOps (https://github.com/PaddlePaddle/HALOps): MXFP8 w8a8 dense GEMM kernels
# for the opt-in --fp8-gemm-backend=halops. Only prebuilt wheels are published,
# so fetch the one matching this image's Python and platform. A missing or
# mismatched wheel leaves /wheels empty and the install step skips it.
ARG HALOPS_VERSION=0.1.0
ARG HALOPS_INDEX_URL=https://www.paddlepaddle.org.cn/packages/stable/cu132/halops/

WORKDIR /build

RUN mkdir -p /wheels && \
    (python3 -m pip download "halops==${HALOPS_VERSION}" \
        --no-deps --only-binary=:all: --no-index \
        -f "${HALOPS_INDEX_URL}" -d /wheels \
    || echo "WARNING: no halops ${HALOPS_VERSION} wheel for this platform; skipping")

########################################################
# PARALLEL STAGE 3: FlashInfer Cache (needs torch_deps)
########################################################
FROM torch_deps AS flashinfer_cache

ARG CUDA_VERSION
ARG INSTALL_FLASHINFER_JIT_CACHE
ARG FLASHINFER_VERSION

# Stage jit-cache/cubin artifacts into /flashinfer_jit_output for clean COPY later
RUN --mount=type=cache,target=/root/.cache/uv \
    case "$CUDA_VERSION" in \
        13.0.3) CUINDEX=130 ;; \
        *) echo "Unsupported CUDA version: $CUDA_VERSION" && exit 1 ;; \
    esac \
    && mkdir -p /flashinfer_jit_output \
    # flashinfer-cubin is CUDA-version-agnostic, unlike jit-cache, so its index-url has no cu${CUINDEX} suffix
    && uv pip install flashinfer-cubin==${FLASHINFER_VERSION} --index-url https://flashinfer.ai/whl \
    && cp -r /opt/sglang/lib/python3.12/site-packages/flashinfer_cubin /flashinfer_jit_output/ \
    && cp -r /opt/sglang/lib/python3.12/site-packages/flashinfer_cubin-*.dist-info /flashinfer_jit_output/ \
    && if [ "$INSTALL_FLASHINFER_JIT_CACHE" = "1" ]; then \
        uv pip install flashinfer-jit-cache==${FLASHINFER_VERSION} --index-url https://flashinfer.ai/whl/cu${CUINDEX} \
        && cp -r /opt/sglang/lib/python3.12/site-packages/flashinfer_jit_cache /flashinfer_jit_output/ \
        && cp -r /opt/sglang/lib/python3.12/site-packages/flashinfer_jit_cache*.dist-info /flashinfer_jit_output/ ; \
    fi

########################################################
# PARALLEL STAGE 5: Gateway Builder (starts from base)
########################################################
# Builds sgl-model-gateway in isolation so Python-only changes
# don't trigger a full Rust recompilation.
FROM base AS gateway_builder

ARG GITHUB_ARTIFACTORY

WORKDIR /build

# Copy ONLY the gateway source (not the full repo)
COPY sgl-model-gateway /build/sgl-model-gateway

# Install Rust, build gateway binary and Python bindings, then clean up Rust toolchain
RUN --mount=type=cache,target=/root/.cache/uv \
    curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && export PATH="/root/.cargo/bin:${PATH}" \
    && uv pip install maturin \
    && cd /build/sgl-model-gateway/bindings/python \
    && ulimit -n 65536 && maturin build --release --features vendored-openssl --out /build/gateway_wheels \
    && cd /build/sgl-model-gateway \
    && cargo build --release --bin sgl-model-gateway --features vendored-openssl \
    && cp target/release/sgl-model-gateway /build/sgl-model-gateway-bin \
    && rm -rf /root/.cargo /root/.rustup /build/sgl-model-gateway/target /build/sgl-model-gateway/bindings/python/target

########################################################
########## Final Framework Image ######################
########################################################
#
# Combines all artifacts from parallel builder stages. BRANCH_TYPE is locked to local:
# this image always ships the checkout it was built from, never a cloned tag.

FROM torch_deps AS framework

ARG BUILD_TYPE
ARG CUDA_VERSION
ARG GITHUB_ARTIFACTORY
ARG MOONCAKE_VERSION
ARG MSCCLPP_VERSION
# Renamed from NCCL_VERSION: the nvidia/cuda base image sets ENV NCCL_VERSION
# (Debian apt format, e.g. 2.28.3-1), which shadows a same-named ARG inside RUN.
ARG SGL_NCCL_VERSION

WORKDIR /sgl-workspace

# =============================================================================
# Copy artifacts from parallel builders
# =============================================================================

# Copy HPC-Ops wheel and install (empty on non-x86_64; kernels are sm90a-only)
COPY --from=hpc_ops_builder /wheels /tmp/wheels/hpc-ops
RUN --mount=type=cache,target=/root/.cache/uv \
    if ls /tmp/wheels/hpc-ops/*.whl >/dev/null 2>&1; then \
        uv pip install --no-deps /tmp/wheels/hpc-ops/*.whl; \
    fi && rm -rf /tmp/wheels/hpc-ops

# Copy HALOps wheel and install if it was staged; a platform-mismatched or
# otherwise unusable wheel must not fail the image build.
COPY --from=halops_wheel /wheels /tmp/wheels/halops
RUN --mount=type=cache,target=/root/.cache/uv \
    if ls /tmp/wheels/halops/*.whl >/dev/null 2>&1; then \
        uv pip install --no-deps /tmp/wheels/halops/*.whl \
        || (uv pip uninstall halops >/dev/null 2>&1 || true; \
            echo "WARNING: halops wheel install failed; building without it"); \
    fi; \
    rm -rf /tmp/wheels/halops

# Copy flashinfer cubin (always) and jit-cache (if installed) packages
COPY --from=flashinfer_cache /flashinfer_jit_output/ /opt/sglang/lib/python3.12/site-packages/

# =============================================================================
# Python packages and tools (before source copy for better caching)
# =============================================================================

# Install Mooncake
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install mooncake-transfer-engine-cuda13==${MOONCAKE_VERSION}

# Install MSCCL++ Python dependencies and package (builds extension via CMake through uv)
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone --depth=1 --branch ${MSCCLPP_VERSION} https://${GITHUB_ARTIFACTORY}/microsoft/mscclpp.git /tmp/mscclpp \
    && case "${CUDA_VERSION}" in \
           13.*) \
               CMAKE_ARGS="-DMSCCLPP_BYPASS_GPU_CHECK=ON -DMSCCLPP_USE_CUDA=ON -DMSCCLPP_GPU_ARCHS=80,90,100,100a,103,103a" \
               uv pip install "/tmp/mscclpp[cuda13]"; \
               ;; \
           *) \
               echo "Unsupported CUDA version for MSCCL++: ${CUDA_VERSION}" && exit 1; \
               ;; \
       esac \
    && rm -rf /tmp/mscclpp

# Install essential Python packages (use constraints to prevent conflicts)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -c /sgl-workspace/constraints.txt \
    datamodel_code_generator \
    pre-commit \
    pytest \
    black \
    isort \
    icdiff \
    wheel \
    scikit-build-core \
    py-spy \
    cubloaty \
    google-cloud-storage \
    pandas \
    matplotlib \
    tabulate \
    termplotlib \
    "runai-model-streamer[s3,gcs,azure]>=0.15.7"

# DeepEP needs SGL_NCCL_VERSION at runtime even if torch pins another version;
# install it after the constrained dependency solve so the solve cannot revert it.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --reinstall --no-deps \
        "nvidia-nccl-cu13==${SGL_NCCL_VERSION}"

# The `nixl` stub package is needed (it owns the `nixl` import path) but
# unconditionally requires nixl-cu12, so install it with --no-deps and pair it
# with the nixl-cu13 binary to avoid shipping wrong-CUDA libs.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install nixl nixl-cu13 --no-deps \
    && uv pip install "cuda-python>=13,<14"

# Install the latest available AI Dynamo prerelease from NVIDIA's package index.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --prerelease=allow --index-strategy unsafe-best-match \
        --extra-index-url https://pypi.nvidia.com/ ai-dynamo

# Fix Trivy-reported pip CVEs (urllib3 CVE-2025-43859, pillow CVE-2026-25990).
# The framework stage's apt-side CVE upgrades of docker/Dockerfile are skipped:
# those patch framework-final system libraries, and only site-packages and the
# workspace copied from this stage reach the runtime image below.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --upgrade "urllib3>=2.6.3" "pillow>=12.1.1"

# =============================================================================
# Copy sglang source and do editable install (LAST for better caching)
# =============================================================================

FROM scratch AS local_src
COPY . /src

# A Git ADD resolves main before cache lookup, unlike RUN git clone. Select the
# empty stage when disabled so those builds do not need access to the Mega repo.
FROM scratch AS mega_source_0
WORKDIR /src

FROM scratch AS mega_source_1
ARG SGL_NVIDIA_MEGA_REPO="https://github.com/sgl-project/sgl-nvidia-mega.git"
# Pinned like a release build (mega-release-ref.txt); keep in sync with that file.
ARG SGL_NVIDIA_MEGA_REF=77360f1fd74dc46010056ebdac0ed5e0b87d3978
ADD --keep-git-dir=true ${SGL_NVIDIA_MEGA_REPO}#${SGL_NVIDIA_MEGA_REF} /src/

FROM mega_source_${ENABLE_SGL_NVIDIA_MEGA} AS mega_source

FROM framework AS framework_final

ARG BUILD_TYPE
ARG CUDA_VERSION

WORKDIR /sgl-workspace

COPY --from=local_src /src /sgl-workspace/sglang

# Editable install (fast - dependencies already installed via constraints)
# Clean up __pycache__/tests/pyc in same RUN to avoid writing ~28k files to layer
RUN --mount=type=cache,target=/root/.cache/uv \
    cd /sgl-workspace/sglang \
    && uv pip install --no-deps -e "python[${BUILD_TYPE}]" \
    && mkdir -p /root/.cache/huggingface /root/.cache/sglang \
    && ( find /opt/sglang/lib/python3.12/site-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true )

# The Rust TreeCore extension must ship with the image: never let import compile it.
# cargo is still on PATH in this stage, so the editable install above builds it.
ENV SGLANG_RUST_BUILD_MODE=never

# Install pre-built gateway artifacts from parallel builder
COPY --from=gateway_builder /build/sgl-model-gateway-bin /opt/sglang/bin/sgl-model-gateway
COPY --from=gateway_builder /build/gateway_wheels /tmp/gateway_wheels
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --reinstall /tmp/gateway_wheels/*.whl \
    && rm -rf /tmp/gateway_wheels

# sgl-nvidia-mega: the NVIDIA megakernel plugin, inert unless a server passes
# --enable-nvidia-megakernel. Installed with CUDA >= 12.9 only (its kernels are
# SM103, compiled at runtime); disable with --build-arg ENABLE_SGL_NVIDIA_MEGA=0.
ARG ENABLE_SGL_NVIDIA_MEGA
RUN --mount=type=bind,from=mega_source,source=/src,target=/tmp/mega-src \
    --mount=type=cache,target=/root/.cache/uv \
    if [ "${ENABLE_SGL_NVIDIA_MEGA}" != "1" ]; then \
        echo "[sgl-nvidia-mega] skip (ENABLE_SGL_NVIDIA_MEGA=${ENABLE_SGL_NVIDIA_MEGA})"; \
    elif ! printf '%s\n' 12.9 "${CUDA_VERSION}" | sort -V -C; then \
        echo "[sgl-nvidia-mega] skip (CUDA_VERSION=${CUDA_VERSION} < 12.9)"; \
    else \
        mkdir -p /sgl-workspace/sgl-nvidia-mega \
        && cp -a /tmp/mega-src/. /sgl-workspace/sgl-nvidia-mega/ \
        && cd /sgl-workspace/sgl-nvidia-mega \
        && git rev-parse HEAD > /sgl-workspace/sgl-nvidia-mega.commit \
        && echo "[sgl-nvidia-mega] revision=$(cat /sgl-workspace/sgl-nvidia-mega.commit)" \
        && uv pip install --no-deps -e . \
        && cd / \
        && python3 -c "from importlib.metadata import entry_points; assert any(e.name == 'nvidia_megakernel' for e in entry_points(group='sglang.srt.plugins')), 'sgl-nvidia-mega plugin not registered'"; \
    fi

########################################################
########## Runtime Image ##############################
########################################################
#
# Official production stage, ported from v0.5.23's docker/Dockerfile (removed upstream in
# #48983): a production-ready image with JIT support -- full CUDA toolchain for DeepGEMM,
# Triton and FlashInfer JIT compiles, ~50% smaller than framework_final. The Ubuntu/pip
# mirror hooks and the ARG-expanded GDRCopy path are the only additions beyond the port.

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS runtime

ARG CUDA_VERSION
ARG TARGETARCH
ARG GDRCOPY_VERSION=2.5.1
ARG PYTHON_VERSION
ARG PIP_DEFAULT_INDEX
ARG UBUNTU_MIRROR

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    GDRCOPY_HOME=/usr/src/gdrdrv-${GDRCOPY_VERSION}/ \
    SGLANG_RUST_BUILD_MODE=never

# Add GKE default lib and bin locations + CUDA compiler paths for FlashInfer JIT
ENV PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

# See the base stage; a fresh FROM resets /etc/apt.
RUN if [ -n "$UBUNTU_MIRROR" ]; then \
    sed -i "s|http://.*archive.ubuntu.com|$UBUNTU_MIRROR|g" /etc/apt/sources.list && \
    sed -i "s|http://.*security.ubuntu.com|$UBUNTU_MIRROR|g" /etc/apt/sources.list; \
fi

# Install runtime dependencies (devel base provides gcc/g++/build tools)
RUN --mount=type=cache,target=/var/cache/apt,id=runtime-apt \
    apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
    wget \
    # Core system utilities
    ca-certificates \
    netcat-openbsd \
    curl \
    ffmpeg \
    git \
    # Runtime libraries
    libopenmpi3 \
    libnuma1 \
    numactl \
    libibverbs1 \
    libibumad3 \
    librdmacm1 \
    libnl-3-200 \
    libnl-route-3-200 \
    ibverbs-providers \
    libgoogle-glog0v6t64 \
    libunwind8 \
    libboost-system1.83.0 \
    libboost-thread1.83.0 \
    libboost-filesystem1.83.0 \
    libgrpc++1.51t64 \
    libprotobuf32t64 \
    libhiredis1.1.0 \
    libcurl4 \
    libczmq4 \
    libfabric1 \
    libssl-dev \
    # RDMA runtime
    rdma-core \
    infiniband-diags \
    perftest \
    # Build tools for JIT compilation
    ninja-build \
    # NCCL packages needed for pynccl_allocator JIT compilation (-lnccl)
    libnccl2 \
    libnccl-dev \
    # GPG key verification
    gnupg2 \
    linux-libc-dev \
    && for py in python python3; do \
        rm -f /usr/bin/$py \
        && printf '#!/bin/sh\nexec /opt/sglang/bin/python "$@"\n' > /usr/bin/$py \
        && chmod +x /usr/bin/$py; \
    done \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# create virtual env for sglang to avoid conflict with system python packages
COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=base /opt/python /opt/python
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/sglang
RUN UV_PYTHON_DOWNLOADS=never uv venv --python ${PYTHON_VERSION} /opt/sglang \
    && /opt/sglang/bin/python -VV
ENV PATH="/opt/sglang/bin:${PATH}"

# Set up locale
RUN apt-get update && apt-get install -y --no-install-recommends locales \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Fix Trivy-reported CVEs (see the framework stage of docker/Dockerfile for the list)
RUN --mount=type=cache,target=/var/cache/apt,id=runtime-apt \
    apt-get update && apt-get install -y --only-upgrade \
    binutils binutils-common binutils-x86-64-linux-gnu libbinutils \
    libctf0 libctf-nobfd0 libgprofng0 libsframe1 \
    libgnutls30t64 \
    libpam-modules libpam-modules-bin libpam-runtime libpam0g \
    libsqlite3-0 libtasn1-6 \
    dpkg dpkg-dev libdpkg-perl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python site-packages from framework (already cleaned of __pycache__/tests/pyc files)
COPY --from=framework_final /opt/sglang/lib/python3.12/site-packages /opt/sglang/lib/python3.12/site-packages
COPY --from=framework_final /opt/sglang/bin/pip* /opt/sglang/bin/

# Copy SGLang workspace
COPY --from=framework_final /sgl-workspace /sgl-workspace

# Copy sgl-model-gateway binary
COPY --from=framework_final /opt/sglang/bin/sgl-model-gateway /opt/sglang/bin/sgl-model-gateway

# Copy sglang binary
COPY --from=framework_final /opt/sglang/bin/sglang /opt/sglang/bin/sglang

# Copy py-spy binary
COPY --from=framework_final /opt/sglang/bin/py-spy /opt/sglang/bin/py-spy

# Copy cache for kernels from kernels community
COPY --from=framework_final /root/.cache/huggingface /root/.cache/huggingface
COPY --from=framework_final /root/.cache/sglang /root/.cache/sglang

# Copy GDRCopy runtime libraries (but not the build artifacts)
COPY --from=framework_final /usr/lib/*-linux-gnu/libgdrapi.so* /usr/lib/
COPY --from=framework_final /usr/bin/gdrcopy_* /usr/bin/
COPY --from=framework_final /usr/src/gdrdrv-${GDRCOPY_VERSION} /usr/src/gdrdrv-${GDRCOPY_VERSION}

# Fix DeepEP IBGDA symlink in runtime
RUN ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so \
    && ldconfig

# Re-apply the pip index mirror: this stage's fresh base does not inherit the base
# stage's /etc/pip.conf, and build_dynamo.py below pip-installs ai-dynamo and maturin.
RUN if [ -n "${PIP_DEFAULT_INDEX}" ]; then \
    python3 -m pip config set global.index-url ${PIP_DEFAULT_INDEX}; \
fi

WORKDIR /sgl-workspace/sglang

########################################################
########## K3.1 Patch Chain ###########################
########################################################
#
# rebuilds the compiled parts the K3.1 optimizations rely on, in the runtime image:
#   1. Rust radix TreeCore   already built at the framework_final editable install
#   2. FlashInfer DynBlock   rebuilds the jit-cache routing modules for 1024 experts
#                            (K3.1 has 896) from the header patched by patch_flashinfer.py
#   3. FlashInfer Prims-TS   main 784bfaf5's prims_ts/ over 0.7.3, for --dsa-*-backend
#                            prims_ts
#   4. Dynamo 1.6.0          ai-dynamo-runtime rebuilt with the session-affinity patches
#                            and the K3.1 edits to dynamo.sglang
# Each script is a no-op when its marker matches what is installed.

# Empty = upstream. K31_RUST_MIRROR is an rsproxy.cn-layout host serving
# rustup-init.sh, the toolchain dists, and a sparse cargo index (dynamo rebuild only;
# TreeCore already built with the base stage's own toolchain).
ARG K31_RUST_MIRROR=
# e.g. https://ghproxy.net/https://github.com/ai-dynamo/dynamo
ARG K31_DYNAMO_REPO=
# e.g. https://ghproxy.net/https://codeload.github.com/flashinfer-ai/flashinfer/tar.gz/784bfaf5bda80259d11a41955ab8c979914197f1
ARG K31_FLASHINFER_ARCHIVE_URL=

RUN git config --global --add safe.directory /sgl-workspace/sglang \
    && git rev-parse HEAD > /sgl-workspace/sglang/.k31-source-commit

# FlashInfer DynBlock: rebuild the jit-cache modules in place; needs
# INSTALL_FLASHINFER_JIT_CACHE=1 (the default here) or the job list is empty.
COPY docker/k31-image/patch_flashinfer.py \
    docker/k31-image/build_flashinfer_dynblock.py /opt/k31-image/
RUN python3 /opt/k31-image/patch_flashinfer.py \
    && python3 /opt/k31-image/build_flashinfer_dynblock.py

COPY docker/k31-image/overlay_flashinfer_prims_ts.py /opt/k31-image/
RUN if [ -n "${K31_FLASHINFER_ARCHIVE_URL:-}" ]; then \
        export K31_FLASHINFER_ARCHIVE_URL="${K31_FLASHINFER_ARCHIVE_URL}"; \
    fi \
    && python3 /opt/k31-image/overlay_flashinfer_prims_ts.py

# COPY of a directory copies its contents, so the patches need their own
# destination to keep the build_dynamo.py-relative dynamo/ layout.
COPY docker/k31-image/build_dynamo.py /opt/k31-image/
COPY docker/k31-image/dynamo /opt/k31-image/dynamo
RUN if [ -n "${K31_DYNAMO_REPO:-}" ]; then \
        export K31_DYNAMO_REPO="${K31_DYNAMO_REPO}"; \
    fi \
    && if [ -n "${K31_RUST_MIRROR:-}" ]; then \
         export RUSTUP_DIST_SERVER="https://${K31_RUST_MIRROR}" \
         RUSTUP_UPDATE_ROOT="https://${K31_RUST_MIRROR}/rustup" \
         K31_RUSTUP_INIT_URL="https://${K31_RUST_MIRROR}/rustup-init.sh" \
         K31_CARGO_MIRROR="sparse+https://${K31_RUST_MIRROR}/index/"; \
       fi \
    && python3 /opt/k31-image/build_dynamo.py

# Fails the build if a DynBlock module was not rebuilt, the Prims-TS overlay or the
# patched Dynamo runtime is missing, or the Rust TreeCore does not load.
# Re-run any time: python3 /opt/k31-image/verify_image.py
COPY docker/k31-image/verify_image.py /opt/k31-image/
RUN python3 /opt/k31-image/verify_image.py \
    && echo "sglang-k31 $(cat /sgl-workspace/sglang/.k31-source-commit)"

# Keep build provenance at the end so metadata changes do not invalidate build layers.
ARG SGLANG_BUILD_COMMIT=unknown
ARG SGLANG_BUILD_URL=
ARG SGLANG_IMAGE_TAG=local/sglang:k31
ENV SGLANG_BUILD_COMMIT=${SGLANG_BUILD_COMMIT:-unknown} \
    SGLANG_BUILD_URL=${SGLANG_BUILD_URL:-} \
    SGLANG_IMAGE_TAG=${SGLANG_IMAGE_TAG:-local/sglang:k31}
LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang" \
      org.opencontainers.image.revision="${SGLANG_BUILD_COMMIT}" \
      org.opencontainers.image.version="${SGLANG_IMAGE_TAG}" \
      org.opencontainers.image.url="${SGLANG_BUILD_URL}" \
      ai.sglang.build.commit="${SGLANG_BUILD_COMMIT}" \
      ai.sglang.build.url="${SGLANG_BUILD_URL}" \
      ai.sglang.image.tag="${SGLANG_IMAGE_TAG}"

CMD ["/bin/bash"]
