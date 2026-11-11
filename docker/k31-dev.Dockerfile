# syntax=docker/dockerfile:1.7
# All-in-one dev/verify image for Kimi-K3.1 work on port/kimi-k31-pro (H200
# target). Everything pip/cargo could need for the near term is baked in
# (cargo included, so the editable install also compiles the Rust TreeCore
# extension); the sglang source tree is NOT -- clone or mount a checkout at
# runtime and `pip install -e "python[test]"`, which the dependency floor
# then resolves offline in seconds.
#
# Build from the repo root (no GPU needed; only docker/k31-image/ is sent):
#   docker buildx build -f docker/k31-dev.Dockerfile -t k31-dev:<tag> --load .
# Throttled network: route every download leg through mirrors; the defaults
# hit pypi.org / sh.rustup.rs / crates.io / github.com.
#   docker buildx build -f docker/k31-dev.Dockerfile -t k31-dev:<tag> --load \
#     --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
#     --build-arg K31_RUST_MIRROR=rsproxy.cn .
# The ~25GB base image is a separate leg from Docker Hub: set a registry mirror
# in /etc/docker/daemon.json, or retag an already-pulled copy as BASE_IMAGE.
#
# ---- First run: clone and install (editable inside the container) ----
#   docker run --rm -it --gpus all --ipc=host --network=host \
#     -v /models/Kimi-K3.1-Pro:/models/Kimi-K3.1-Pro:ro \
#     k31-dev:<tag> bash
#   git clone -b port/kimi-k31-pro https://github.com/xxsoul/sglang \
#     /sgl-workspace/sglang
#   cd /sgl-workspace/sglang && pip install -e "python[test]"
#   python3 /opt/k31-image/dev_verify.py   # full GPU + marlin probe
# Keep the clone across runs: -v k31-source:/sgl-workspace/sglang, or mount a
# host checkout the same way (-v $PWD:/sgl-workspace/sglang).
#
# ---- Multi-node launch (one block per node, identical except --node-rank) ----
#   GLOO_SOCKET_IFNAME=<nic> NCCL_SOCKET_IFNAME=<nic> SGLANG_HOST_IP=<node-ip> \
#   python -m sglang.launch_server --trust-remote-code \
#     --model-path /models/Kimi-K3.1-Pro --tp-size 16 --ep-size 16 \
#     --mem-fraction-static 0.85 --max-running-requests 64 \
#     --nnodes 2 --node-rank <rank> --dist-init-addr <node0-ip>:20000 \
#     --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 \
#     --host 0.0.0.0 --port 30000
#
# ---- Accuracy (sgl-eval runs from its own venv, never the server env) ----
#   /opt/sgl-eval-venv/bin/sgl-eval run gsm8k \
#     --base-url http://127.0.0.1:30000/v1 --num-examples 200 \
#     --load-preset-from-model-id moonshotai/Kimi-K3
ARG BASE_IMAGE=lmsysorg/sglang:nightly-dev-20261024-5994c559@sha256:40da9cac78a558237426971c048ab0f4f1a4b13bfb1b6e7a48d4d980e4163c8e
FROM ${BASE_IMAGE}

# Empty = upstream. K31_RUST_MIRROR is an rsproxy.cn-layout host serving
# rustup-init.sh, the toolchain dists, and a sparse cargo index.
ARG PIP_INDEX_URL=
ARG K31_RUST_MIRROR=
# e.g. https://ghproxy.net/https://github.com/ai-dynamo/dynamo
ARG K31_DYNAMO_REPO=
# e.g. https://ghproxy.net/https://codeload.github.com/flashinfer-ai/flashinfer/tar.gz/784bfaf5bda80259d11a41955ab8c979914197f1
ARG K31_FLASHINFER_ARCHIVE_URL=

RUN git config --global --add safe.directory /sgl-workspace/sglang

# Lock comes in its own COPY: it changes rarely, so the heavy pip layers stay
# cached while the k31-image scripts around it are still being iterated.
COPY docker/k31-image/k31-dev.lock /opt/k31-image/k31-dev.lock

# Drop the nightly's own editable install; the runtime clone replaces it,
# and pip install -e there must not inherit a stale upstream tree.
RUN pip uninstall -y sglang || true \
    && rm -rf /sgl-workspace/sglang
WORKDIR /sgl-workspace

# Dependency floor: exact pins resolved from the branch's pyproject [test]
# extra (docker/k31-image/k31-dev.lock). Regenerate with:
#   uv pip compile python/pyproject.toml --extra test --python-version 3.12 \
#     --no-emit-index-url -o docker/k31-image/k31-dev.lock
RUN export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}" \
    && pip install --no-deps -r /opt/k31-image/k31-dev.lock

# sgl-eval in its own venv (see the test extra pin comment in pyproject.toml).
RUN python3 -m venv /opt/sgl-eval-venv \
    && export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}" \
    && /opt/sgl-eval-venv/bin/pip install -q sgl-eval==0.1.3

# Minimal stable toolchain for the runtime `pip install -e`, which builds the
# Rust TreeCore extension (setup.py auto-discovers it when cargo is on PATH);
# K31_RUST_MIRROR additionally persists a crates.io source replacement into
# /opt/cargo/config.toml for that runtime build.
# build_dynamo.py still uses its own short-lived toolchain in a temp dir.
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
ENV PATH="/opt/cargo/bin:${PATH}"
RUN if [ -n "${K31_RUST_MIRROR:-}" ]; then \
         export RUSTUP_DIST_SERVER="https://${K31_RUST_MIRROR}" \
         RUSTUP_UPDATE_ROOT="https://${K31_RUST_MIRROR}/rustup" \
         K31_RUSTUP_INIT_URL="https://${K31_RUST_MIRROR}/rustup-init.sh"; \
       fi \
    && curl --proto '=https' --tlsv1.2 -sSf "${K31_RUSTUP_INIT_URL:-https://sh.rustup.rs}" \
        | sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path \
    && if [ -n "${K31_RUST_MIRROR:-}" ]; then \
         printf '[source.crates-io]\nreplace-with = "k31-mirror"\n[source.k31-mirror]\nregistry = "sparse+https://%s/index/"\n' \
           "${K31_RUST_MIRROR}" > /opt/cargo/config.toml; \
       fi

# Scripts copy per-stage so editing one never recompiles a neighbor layer;
# new scripts join the COPY of the stage they serve. Marker-idempotent.
COPY docker/k31-image/patch_flashinfer.py \
    docker/k31-image/build_flashinfer_dynblock.py /opt/k31-image/
RUN python3 /opt/k31-image/patch_flashinfer.py \
    && python3 /opt/k31-image/build_flashinfer_dynblock.py

COPY docker/k31-image/overlay_flashinfer_prims_ts.py /opt/k31-image/
RUN python3 /opt/k31-image/overlay_flashinfer_prims_ts.py

# COPY of a directory copies its contents, so the patches need their own
# destination to keep the build_dynamo.py-relative dynamo/ layout.
COPY docker/k31-image/build_dynamo.py /opt/k31-image/
COPY docker/k31-image/dynamo /opt/k31-image/dynamo
RUN export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}" \
    && if [ -n "${K31_RUST_MIRROR:-}" ]; then \
         export RUSTUP_DIST_SERVER="https://${K31_RUST_MIRROR}" \
         RUSTUP_UPDATE_ROOT="https://${K31_RUST_MIRROR}/rustup" \
         K31_RUSTUP_INIT_URL="https://${K31_RUST_MIRROR}/rustup-init.sh" \
         K31_CARGO_MIRROR="sparse+https://${K31_RUST_MIRROR}/index/"; \
       fi \
    && python3 /opt/k31-image/build_dynamo.py

# Fails the build if a DynBlock module was not rebuilt or the overlays are
# missing. Re-run verify_image.py any time (no GPU, no source tree needed);
# dev_verify.py --build-time checks the dependency floor.
COPY docker/k31-image/verify_image.py docker/k31-image/dev_verify.py /opt/k31-image/
RUN python3 /opt/k31-image/verify_image.py \
    && python3 /opt/k31-image/dev_verify.py --build-time \
    && echo "k31-dev lock $(sha256sum /opt/k31-image/k31-dev.lock | cut -c1-12)"

CMD ["/bin/bash"]
