"""2D thermal benchmark with a manufactured analytical solution.

Run:
    python -m examples.thermal_benchmark_mms2d
    python examples/thermal_benchmark_mms2d.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure direct script execution resolves local beamz source tree.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from beamz import (
    Design,
    Material,
    StaticThermalConfig,
    ThermalScenario,
    ThermalSink,
    ThermalSource,
)


def build_manufactured_case(nx, ny, dx, t0, k, amplitude):
    x_idx = np.arange(nx, dtype=float) / max(nx - 1, 1)
    y_idx = np.arange(ny, dtype=float) / max(ny - 1, 1)
    X, Y = np.meshgrid(x_idx, y_idx)

    mode = np.sin(np.pi * X) * np.sin(np.pi * Y) + 0.35 * np.sin(
        2.0 * np.pi * X
    ) * np.sin(3.0 * np.pi * Y)
    t_ref = t0 + amplitude * mode

    lap = np.zeros_like(t_ref)
    lap[1:-1, 1:-1] = (t_ref[1:-1, 2:] - 2.0 * t_ref[1:-1, 1:-1] + t_ref[1:-1, :-2]) / (
        dx * dx
    ) + (t_ref[2:, 1:-1] - 2.0 * t_ref[1:-1, 1:-1] + t_ref[:-2, 1:-1]) / (dx * dx)
    source = -k * lap

    fixed_mask = np.zeros_like(t_ref, dtype=bool)
    fixed_mask[0, :] = True
    fixed_mask[-1, :] = True
    fixed_mask[:, 0] = True
    fixed_mask[:, -1] = True
    return t_ref, source, fixed_mask


def main():
    nx, ny = 81, 61
    dx = 1.0
    t0 = 300.0
    k = 12.0
    amplitude = 40.0

    t_ref, source, fixed_mask = build_manufactured_case(
        nx=nx, ny=ny, dx=dx, t0=t0, k=k, amplitude=amplitude
    )

    design = Design(
        width=float(nx * dx),
        height=float(ny * dx),
        material=Material(permittivity=1.0, k=k, rho=1.0, cp=1.0, T0=t0),
    )
    config = StaticThermalConfig(
        max_iters=20000,
        tol=1e-9,
    )
    scenario = ThermalScenario(
        sources=[
            ThermalSource(
                region=np.ones_like(source, dtype=bool),
                power_density_w_m3=source,
            )
        ],
        sinks=[ThermalSink(region=fixed_mask, temperature_k=t0)],
    )
    result = design.solve_thermal(
        resolution=dx,
        scenario=scenario,
        config=config,
    )

    t_num = np.asarray(result.temperature)
    err = t_num - t_ref
    abs_err = np.abs(err)
    rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(t_ref - t0), 1e-12))
    max_abs_err = float(np.max(abs_err))

    print(f"2D MMS relative L2 error: {rel_l2 * 100:.3f}%")
    print(f"2D MMS max abs error: {max_abs_err:.4f} K")

    extent = (0.0, nx * dx, 0.0, ny * dx)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))

    vmin = float(np.min(t_ref))
    vmax = float(np.max(t_ref))
    im0 = axes[0].imshow(
        t_num, origin="lower", extent=extent, cmap="inferno", vmin=vmin, vmax=vmax
    )
    axes[0].set_title("Numerical Temperature (K)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)

    im1 = axes[1].imshow(
        t_ref, origin="lower", extent=extent, cmap="inferno", vmin=vmin, vmax=vmax
    )
    axes[1].set_title("Analytical Temperature (K)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)

    im2 = axes[2].imshow(abs_err, origin="lower", extent=extent, cmap="viridis")
    axes[2].set_title("|Error| (K)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.03)

    plt.suptitle("2D Manufactured-Solution Thermal Benchmark", y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
