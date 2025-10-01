"""Benchmark the NumPy and JAX backends on a simple 2D FDTD update loop."""
import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np

# Allow running the script directly without installing the package.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from beamz.const import EPS_0, MU_0, LIGHT_SPEED
from beamz.simulation.backends import get_backend


@dataclass
class BenchmarkResult:
    backend: str
    mean_step_time: float
    total_time: float
    steps: int


def _create_backend(name: str):
    if name == "jax":
        # Force CPU execution to provide a fair comparison with NumPy.
        return get_backend("jax", device="cpu", use_jit=True)
    if name == "numpy":
        return get_backend("numpy")
    raise ValueError(f"Unsupported backend for benchmark: {name}")


def _block_until_ready(*arrays):
    for arr in arrays:
        block = getattr(arr, "block_until_ready", None)
        if callable(block):
            block()


def benchmark_backend(name: str, steps: int = 200, shape=(256, 256)) -> BenchmarkResult:
    backend = _create_backend(name)
    ny, nx = shape
    dx = dy = 0.02e-6
    dt = 0.99 * dx / (LIGHT_SPEED * np.sqrt(2))

    Ez = backend.zeros((ny, nx), dtype=np.complex128)
    Hx = backend.zeros((ny, nx - 1), dtype=np.complex128)
    Hy = backend.zeros((ny - 1, nx), dtype=np.complex128)

    sigma = backend.from_numpy(np.zeros((ny, nx), dtype=np.float32))
    eps_r = backend.from_numpy(np.ones((ny, nx), dtype=np.float32))

    # Warm up the kernels (especially important for JAX's JIT compilation).
    warmup_steps = 5
    for _ in range(warmup_steps):
        Hx, Hy = backend.update_h_fields(Hx, Hy, Ez, sigma, dx, dy, dt, MU_0, EPS_0)
        Ez = backend.update_e_field(Ez, Hx, Hy, sigma, eps_r, dx, dy, dt, EPS_0)
    _block_until_ready(Ez, Hx, Hy)

    start = time.perf_counter()
    for _ in range(steps):
        Hx, Hy = backend.update_h_fields(Hx, Hy, Ez, sigma, dx, dy, dt, MU_0, EPS_0)
        Ez = backend.update_e_field(Ez, Hx, Hy, sigma, eps_r, dx, dy, dt, EPS_0)
    _block_until_ready(Ez, Hx, Hy)
    end = time.perf_counter()

    total_time = end - start
    mean_step_time = total_time / steps
    return BenchmarkResult(name, mean_step_time, total_time, steps)


def main():
    print("Comparing backends on CPU…")
    for backend_name in ("jax", "numpy"):
        result = benchmark_backend(backend_name)
        print(f"{result.backend:>5} :: mean step {result.mean_step_time*1e6:8.2f} µs"
              f" (total {result.total_time:.3f} s over {result.steps} steps)")


if __name__ == "__main__":
    main()
