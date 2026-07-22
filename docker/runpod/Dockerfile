# syntax=docker/dockerfile:1.7

ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.9.7
ARG RUNPOD_BASE_IMAGE=runpod/base:1.0.2-cuda1281-ubuntu2404

FROM ${UV_IMAGE} AS uv
FROM ${RUNPOD_BASE_IMAGE}

ARG JAX_VERSION=0.9.0
ARG JUPYTERLAB_VERSION=4.5.9
ARG IPYKERNEL_VERSION=7.2.0

ENV UV_PROJECT_ENVIRONMENT=/opt/beamz-venv \
    PATH="/opt/beamz-venv/bin:${PATH}" \
    PYTHONPATH=/workspace/beamz \
    BEAMZ_REPO_DIR=/workspace/beamz \
    BEAMZ_REPO_URL=https://github.com/beamzorg/beamz.git \
    BEAMZ_ENABLE_JAX_PERSISTENT_CACHE=1 \
    BEAMZ_JAX_CACHE_DIR=/workspace/.cache/beamz/jax_cache \
    BEAMZ_RASTER_CACHE=1 \
    BEAMZ_RASTER_CACHE_DIR=/workspace/.cache/beamz/raster_cache

COPY --from=uv /uv /uvx /bin/

WORKDIR /opt/beamz-build

# Install the locked BeamZ dependencies without baking a source checkout into the
# image. The live checkout is kept on RunPod's persistent /workspace volume.
COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    UV_CACHE_DIR=/root/.cache/uv \
    uv sync \
        --python 3.12 \
        --frozen \
        --no-install-project \
        --extra dev \
        --extra test && \
    UV_CACHE_DIR=/root/.cache/uv \
    uv export \
        --frozen \
        --no-emit-project \
        --no-hashes \
        --extra dev \
        --extra test \
        > /tmp/beamz-constraints.txt && \
    UV_CACHE_DIR=/root/.cache/uv \
    uv pip install \
        --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
        --constraint /tmp/beamz-constraints.txt \
        "jax[cuda12-local]==${JAX_VERSION}" \
        "jupyterlab==${JUPYTERLAB_VERSION}" \
        "ipykernel==${IPYKERNEL_VERSION}" && \
    rm /tmp/beamz-constraints.txt

# RunPod's /start.sh executes this hook before starting JupyterLab and SSH.
COPY --chmod=0755 docker/runpod/start.sh /pre_start.sh

WORKDIR /workspace

# Keep the base image's CMD ["/start.sh"].
