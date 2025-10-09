import numpy as np
import matplotlib.pyplot as plt

from beamz.const import LIGHT_SPEED, µm
from beamz.devices.mode import solve_modes


def main():
    plt.switch_backend("Agg")

    # Wavelength and grid
    wavelength = 1.55 * µm
    omega = 2 * np.pi * LIGHT_SPEED / wavelength
    dL = 0.02 * µm  # 20 nm transverse sampling

    # 1D slab waveguide profile (y-direction)
    height = 4.0 * µm
    ny = int(np.round(height / dL))
    y = np.linspace(-height / 2, height / 2, ny)

    n_core = 3.45
    n_clad = 1.44
    core_thickness = 0.5 * µm

    eps = np.full(ny, n_clad**2, dtype=float)
    core_mask = np.abs(y) <= (core_thickness / 2)
    eps[core_mask] = n_core**2

    # Solve for the first two modes propagating along +x
    neff, e_fields, h_fields, prop_axis = solve_modes(
        eps=eps,
        omega=omega,
        dL=dL,
        npml=20,
        m=2,
        direction="+x",
        return_fields=True,
    )

    print("Effective indices:", [float(np.real(n)) for n in neff])

    # Plot permittivity and fundamental |Ez| profile
    Ez0 = np.squeeze(e_fields[0][2])  # component index 2 is Ez for TE in 2D
    intensity0 = np.abs(Ez0) ** 2

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(y / µm, eps, color="black", label="εr")
    ax1.set_xlabel("y (µm)")
    ax1.set_ylabel("εr", color="black")
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(y / µm, intensity0 / np.max(intensity0 + 1e-18), color="crimson", label="|Ez|² (norm)")
    ax2.set_ylabel("|Ez|² (norm)")

    plt.title(f"1D Slab Modes, λ = {wavelength/µm:.2f} µm, neff₀ = {float(np.real(neff[0])):.3f}")
    fig.tight_layout()
    fig.savefig("modesolver_1d_mode0.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()


