"""Practical thermal demo: MZI phase shifter heater sweep.

Run:
    python -m examples.thermal_mzi_phase_shifter
    python examples/thermal_mzi_phase_shifter.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle as PlotRectangle

# Ensure direct script execution resolves local beamz source tree.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from beamz import (
    Design,
    Material,
    Rectangle,
    StaticThermalConfig,
    ThermalBoundaryProfile,
    ThermalScenario,
    ThermalSource,
)


def main():
    # 2D cross-section (x horizontal, y vertical) for a heater-over-waveguide arm.
    W, H = 60e-6, 20e-6
    design = Design(
        width=W, height=H, material=Material(permittivity=1.0, k=0.026, T0=300.0)
    )

    si = Material(
        permittivity=3.48**2, k=130.0, rho=2330.0, cp=700.0, dn_dT=1.86e-4, T0=300.0
    )
    sio2 = Material(
        permittivity=1.44**2, k=1.38, rho=2200.0, cp=703.0, dn_dT=1.0e-5, T0=300.0
    )
    tin = Material(permittivity=1.0, k=25.0, rho=5200.0, cp=540.0, T0=300.0)

    substrate = Rectangle(position=(0.0, 0.0), width=W, height=12.0e-6, material=si)
    box = Rectangle(position=(0.0, 12.0e-6), width=W, height=2.0e-6, material=sio2)
    top_oxide = Rectangle(
        position=(0.0, 14.0e-6), width=W, height=3.5e-6, material=sio2
    )
    arm_core = Rectangle(
        position=(18e-6, 14.2e-6), width=24e-6, height=0.22e-6, material=si
    )
    heater = Rectangle(
        position=(18e-6, 16.4e-6), width=24e-6, height=0.3e-6, material=tin
    )

    for structure in [substrate, box, top_oxide, arm_core, heater]:
        design += structure

    resolution = 0.2e-6
    powers_w = np.linspace(0.0, 0.06, 10)
    scenario_base = ThermalScenario(
        extrusion_depth_m=200e-6,
        boundary_profile=ThermalBoundaryProfile.photonic_chip(
            sink_thickness_m=0.2e-6,
            sink_temperature_k=300.0,
            top_h_w_m2_k=10.0,
            ambient_temp_k=300.0,
        ),
    )
    tuning = design.sweep_mzi_heater(
        resolution=resolution,
        powers_w=powers_w,
        heater=ThermalSource(region=heater, name="upper_arm_heater"),
        optical_region=arm_core,
        arm_length_m=2.5e-3,
        wavelength_m=1.55e-6,
        group_index=4.2,
        scenario_base=scenario_base,
        config=StaticThermalConfig(max_iters=7000, tol=1e-6),
    )

    max_power_scenario = ThermalScenario(
        sources=[ThermalSource(region=heater, power_w=float(powers_w[-1]))],
        extrusion_depth_m=scenario_base.extrusion_depth_m,
        boundary_profile=scenario_base.boundary_profile,
    )
    solved = design.solve_thermal(
        resolution=resolution,
        scenario=max_power_scenario,
        config=StaticThermalConfig(max_iters=7000, tol=1e-6),
    )
    temperature = np.asarray(solved.temperature)

    print(f"Ppi estimate: {tuning.p_pi_w * 1e3:.2f} mW")
    print(
        f"Max temperature at {powers_w[-1] * 1e3:.1f} mW: {float(np.max(temperature)):.2f} K"
    )
    print(f"Peak phase shift: {float(np.max(tuning.delta_phi_rad)):.3f} rad")

    extent = (0, W * 1e6, 0, H * 1e6)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    im0 = axes[0].imshow(temperature, origin="lower", extent=extent, cmap="inferno")
    axes[0].set_title(f"Temperature at {powers_w[-1] * 1e3:.0f} mW")
    axes[0].set_xlabel("X (µm)")
    axes[0].set_ylabel("Y (µm)")
    fig.colorbar(im0, ax=axes[0], label="Temperature (K)", fraction=0.048, pad=0.03)

    for x, y, w, h in [
        (0.0, 0.0, W, 12.0e-6),
        (0.0, 12.0e-6, W, 2.0e-6),
        (0.0, 14.0e-6, W, 3.5e-6),
        (18e-6, 14.2e-6, 24e-6, 0.22e-6),
        (18e-6, 16.4e-6, 24e-6, 0.3e-6),
    ]:
        axes[0].add_patch(
            PlotRectangle(
                (x * 1e6, y * 1e6),
                w * 1e6,
                h * 1e6,
                fill=False,
                edgecolor="white",
                linewidth=1.0,
                alpha=0.6,
            )
        )

    axes[1].plot(
        tuning.power_w * 1e3, tuning.delta_phi_rad, color="#D24A3A", linewidth=2.2
    )
    axes[1].axhline(
        np.pi, color="black", linestyle="--", linewidth=1.2, label="π phase"
    )
    if np.isfinite(tuning.p_pi_w):
        axes[1].axvline(
            tuning.p_pi_w * 1e3,
            color="#1F77B4",
            linestyle=":",
            linewidth=1.6,
            label=f"Pπ={tuning.p_pi_w*1e3:.1f} mW",
        )
    axes[1].set_title("MZI Phase Shift vs Heater Power")
    axes[1].set_xlabel("Heater power (mW)")
    axes[1].set_ylabel("Δφ (rad)")
    axes[1].legend(loc="upper left")

    axes[2].plot(
        tuning.power_w * 1e3, tuning.delta_n_eff, color="#2A9D8F", linewidth=2.2
    )
    axes[2].set_title("Effective Index Shift")
    axes[2].set_xlabel("Heater power (mW)")
    axes[2].set_ylabel("Δn_eff")
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
