#!/usr/bin/env bash
set -euo pipefail

workspace="${RP_WORKSPACE:-/workspace}"
repo_dir="${BEAMZ_REPO_DIR:-${workspace}/beamz}"
repo_url="${BEAMZ_REPO_URL:-https://github.com/beamzorg/beamz.git}"

mkdir -p \
    "${workspace}" \
    "${BEAMZ_JAX_CACHE_DIR:-${workspace}/.cache/beamz/jax_cache}" \
    "${BEAMZ_RASTER_CACHE_DIR:-${workspace}/.cache/beamz/raster_cache}"

if [[ ! -e "${repo_dir}" ]]; then
    echo "Cloning BeamZ into ${repo_dir}..."
    GIT_TERMINAL_PROMPT=0 git clone "${repo_url}" "${repo_dir}"
elif [[ ! -d "${repo_dir}/.git" ]]; then
    echo "BeamZ startup error: ${repo_dir} exists but is not a Git checkout." >&2
    exit 1
else
    echo "Using the existing BeamZ checkout at ${repo_dir}."
fi

if ! git config --global --get-all safe.directory | grep -Fxq "${repo_dir}"; then
    git config --global --add safe.directory "${repo_dir}"
fi

echo "BeamZ environment ready. The checkout is not updated automatically."
