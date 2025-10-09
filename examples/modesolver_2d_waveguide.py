import numpy as np
import matplotlib.pyplot as plt
from beamz.const import LIGHT_SPEED, µm
from beamz.devices.mode import tidy3d_mode_computation_wrapper

plt.switch_backend("Agg")
wavelength = 1.55 * µm
frequency = LIGHT_SPEED / wavelength
# 2D cross-section grid (y, z)
dy = dz = 0.0125 * µm
lateral_span = 2.0 * µm
substrate_thickness = 1.0 * µm
air_thickness = 1.0 * µm
total_height = substrate_thickness + air_thickness
ny = int(np.round(lateral_span / dy))
nz = int(np.round(total_height / dz))
n_core = 3.45
n_sub = 1.44
n_air = 1.00
core_wy = 0.6 * µm
core_wz = 0.22 * µm

# Build permittivity grid directly
core_offset = -0.1 * µm
y_samples = np.linspace(-lateral_span / 2 + dy/2, lateral_span / 2 - dy/2, ny)
z_samples = np.linspace(-substrate_thickness + dz/2 - core_offset, air_thickness - dz/2 - core_offset, nz)
eps = np.full((ny, nz), n_air**2, dtype=float)
eps[:, z_samples < 0] = n_sub**2
core_y_mask = np.abs(y_samples) <= core_wy / 2
core_z_mask = (z_samples >= core_offset) & (z_samples <= core_offset + core_wz)
eps[np.ix_(core_y_mask, core_z_mask)] = n_core**2
y_edges = np.linspace(-lateral_span / 2, lateral_span / 2, ny + 1)
z_edges = np.linspace(-substrate_thickness, air_thickness, nz + 1)
coords = [y_edges / µm, z_edges / µm]
y_centered = y_samples
z_centered = z_samples

modes = tidy3d_mode_computation_wrapper(
    frequency=frequency,
    permittivity_cross_section=eps,
    coords=coords,
    direction="+",
    num_modes=8,
    precision="double",
)

# Sort by descending neff (real part)
modes = sorted(modes, key=lambda m: float(np.real(m.neff)), reverse=True)
n_cols = min(3, len(modes))

# Define window size
window_um = (core_wy / 2 + 0.4 * core_wy) / µm

# Detailed field components (Ey, Ez, Hy, Hz) for first mode
if modes:
    mode0 = modes[0]
    Ex = np.array(mode0.Ex)
    Ey = np.array(mode0.Ey)
    Ez = np.array(mode0.Ez)
    Hx = np.array(mode0.Hx)
    Hy = np.array(mode0.Hy)
    Hz = np.array(mode0.Hz)

    fields = {
        "Ex": Ex,
        "Ey": Ey,
        "Ez": Ez,
        "Hx": Hx,
        "Hy": Hy,
        "Hz": Hz,
    }

    fig_comp, ax_comp = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    i = 0
    for (name, field), ax in zip(fields.items(), ax_comp.ravel()):
        ax.text(-0.05, 1.05, f"{chr(ord('a') + i)}", transform=ax.transAxes, fontsize=16, fontweight="bold")
        i += 1
        
        real_field = np.real(field)
        vmax = np.max(np.abs(real_field)) or 1.0
        im = ax.imshow(real_field.T / vmax, origin="lower",
                        extent=(y_centered[0] / µm, y_centered[-1] / µm, z_centered[0] / µm, z_centered[-1] / µm),
                        cmap="viridis", aspect="equal", vmin=-1, vmax=1)
        ax.set_xlim(-window_um, window_um)
        ax.set_ylim(-window_um, window_um)
        ax.set_aspect('equal')
        ax.set_title(f"Mode 0: Re({name}) (norm)")
        ax.set_xlabel("y (µm)")
        ax.set_ylabel("z (µm)")

    fig_comp.savefig("2D_modes.png", dpi=200, bbox_inches="tight")
    plt.close(fig_comp)