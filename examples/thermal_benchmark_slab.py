"""Thermal benchmark demo: 1D slab with generation + Robin boundary.

Run:
    python -m examples.thermal_benchmark_slab
    python examples/thermal_benchmark_slab.py
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
    ConvectionBC,
    Design,
    Material,
    StaticThermalConfig,
    ThermalScenario,
    ThermalSink,
    ThermalSource,
)


def analytical_slab_with_generation(x, k, q_vol, t_left, t_ambient, h, length):
    c1 = (
        q_vol * length
        - h * ((-q_vol * length * length) / (2.0 * k) + (t_left - t_ambient))
    ) / (k + h * length)
    return -q_vol / (2.0 * k) * x * x + c1 * x + t_left


def main():
    nx, ny = 81, 9
    dx = 1.0
    k = 5.0
    q_vol = 1.2
    t_left = 300.0
    t_ambient = 295.0
    h = 0.25

    design = Design(
        width=float(nx),
        height=float(ny),
        material=Material(permittivity=1.0, k=k, rho=1.0, cp=1.0, T0=t_left),
    )

    heater_mask = np.ones((ny, nx), dtype=bool)
    fixed_mask = np.zeros((ny, nx), dtype=bool)
    fixed_mask[:, 0] = True
    config = StaticThermalConfig(
        max_iters=20000,
        tol=1e-8,
    )
    scenario = ThermalScenario(
        sources=[ThermalSource(region=heater_mask, power_density_w_m3=q_vol)],
        sinks=[ThermalSink(region=fixed_mask, temperature_k=t_left)],
        convection=ConvectionBC(
            h_w_m2_k=h,
            ambient_temp_k=t_ambient,
            sides=("right",),
        ),
    )
    result = design.solve_thermal(
        resolution=dx,
        scenario=scenario,
        config=config,
    )

    t_num = np.asarray(result.temperature).mean(axis=0)
    x = np.arange(nx, dtype=float) * dx
    length = (nx - 1) * dx
    t_ref = analytical_slab_with_generation(
        x=x,
        k=k,
        q_vol=q_vol,
        t_left=t_left,
        t_ambient=t_ambient,
        h=h,
        length=length,
    )

    max_abs_err = float(np.max(np.abs(t_num - t_ref)))
    span = max(float(np.max(t_ref) - np.min(t_ref)), 1e-12)
    rel_err = max_abs_err / span
    print(f"Max abs error: {max_abs_err:.4f} K")
    print(f"Relative error: {rel_err*100:.3f}%")

    plt.figure(figsize=(7, 4))
    plt.plot(x, t_ref, "k--", linewidth=2, label="Analytical")
    plt.plot(x, t_num, color="#D24A3A", linewidth=2, label="BEAMZ static solver")
    plt.xlabel("x (arbitrary units)")
    plt.ylabel("Temperature (K)")
    plt.title("1D Slab Benchmark: Generation + Robin BC")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
