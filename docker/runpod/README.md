# BeamZ on RunPod

This image provides a reproducible BeamZ notebook and development environment
for RunPod GPU Pods. It installs the dependencies locked by the repository plus
CUDA-enabled JAX, JupyterLab, and an IPython kernel.

The image deliberately does not contain the BeamZ source tree. On first startup,
RunPod's pre-start hook clones BeamZ into `/workspace/beamz`. Later Pods reuse
that checkout from the attached volume and never pull or overwrite it
automatically.

## Build locally

Build from the repository root. RunPod GPU Pods use `linux/amd64` images.

```bash
docker build \
  --platform linux/amd64 \
  --file docker/runpod/Dockerfile \
  --tag beamz-runpod:local \
  .
```

The default build uses Python 3.12, JAX 0.9.0 against the CUDA 12.8.1 and cuDNN
libraries in the pinned RunPod base image, and the locked BeamZ development and
test dependencies. When the JAX version in `uv.lock` changes, update the
`JAX_VERSION` build argument in the Dockerfile at the same time.

## RunPod template

Configure a private custom Pod template with the following values after the
image has been built and placed in a registry:

| Setting | Value |
| --- | --- |
| Container image | The immutable tag for this image |
| Container disk | 30 GB |
| Volume mount path | `/workspace` |
| HTTP port | `8888` |
| TCP port | `22` |
| Docker entrypoint | Leave empty |
| Docker start command | Leave empty |

Attach a network volume when deploying if the checkout, notebooks, results, and
compilation caches must survive Pod termination. A regular Pod volume survives
stops but is deleted with its Pod.

Set `JUPYTER_PASSWORD` in the template to enable JupyterLab. Set `PUBLIC_KEY` to
an SSH public key if SSH access is wanted. Do not put passwords, private keys, or
registry credentials in this directory or bake them into the image.

The startup hook supports these optional environment variables:

- `BEAMZ_REPO_URL`: Git URL cloned on the first startup.
- `BEAMZ_REPO_DIR`: Checkout path; defaults to `/workspace/beamz`.

To update an existing checkout explicitly:

```bash
cd /workspace/beamz
git pull --ff-only
```

## Verify a GPU Pod

Open a terminal after the Pod starts and run:

```bash
python - <<'PY'
import jax
import beamz

print("BeamZ:", beamz.__version__)
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
assert jax.default_backend() == "gpu"
PY
```

BeamZ's JAX and raster caches are enabled and stored under
`/workspace/.cache/beamz`, so an attached network volume can reuse them across
replacement Pods.
