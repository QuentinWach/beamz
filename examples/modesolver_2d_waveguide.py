import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from beamz.const import LIGHT_SPEED, µm
try:
    import tidy3d  # noqa: F401
except ModuleNotFoundError as exc:
    tidy3d = None

from beamz.devices.mode import tidy3d_mode_computation_wrapper, ModeTupleType


def add_geometry_overlays(ax, y_edges_centered, z_edges_centered, core_wy, core_wz, substrate_thickness):
    y_min_um, y_max_um = y_edges_centered[0] / µm, y_edges_centered[-1] / µm
    z_min_um, z_max_um = z_edges_centered[0] / µm, z_edges_centered[-1] / µm
    width_um = y_max_um - y_min_um

    # Substrate: z < 0
    if z_min_um < 0:
        sub_height = min(0.0, z_max_um) - z_min_um
        ax.add_patch(Rectangle((y_min_um, z_min_um), width_um, sub_height,
                               facecolor="#e8f1ff", edgecolor="none", alpha=0.18, zorder=-10))
    # Air: z > core thickness
    air_start = substrate_thickness / µm
    if z_max_um > air_start:
        air_height = z_max_um - max(air_start, z_min_um)
        ax.add_patch(Rectangle((y_min_um, max(air_start, z_min_um)), width_um, air_height,
                               facecolor="#fff7e6", edgecolor="none", alpha=0.12, zorder=-10))
    # Core outline
    ax.add_patch(Rectangle((-core_wy/(2*µm), 0.0), core_wy/µm, core_wz/µm,
                           facecolor="none", edgecolor="black", linewidth=1.5))


def main():
    plt.switch_backend("Agg")

    if tidy3d is None:
        print("This example requires tidy3d. Please install tidy3d to run it.")
        return

    wavelength = 1.55 * µm
    frequency = LIGHT_SPEED / wavelength

    # 2D cross-section grid (y, z)
    dy = dz = 0.025 * µm
    lateral_span = 3.0 * µm
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
    y_samples = np.linspace(-lateral_span / 2 + dy/2, lateral_span / 2 - dy/2, ny)
    z_samples = np.linspace(-substrate_thickness + dz/2, air_thickness - dz/2, nz)
    eps = np.full((ny, nz), n_air**2, dtype=float)
    eps[:, z_samples < 0] = n_sub**2
    core_y_mask = np.abs(y_samples) <= core_wy / 2
    core_z_mask = (z_samples >= dz/2) & (z_samples <= core_wz - dz/2)
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

    # Save geometry visualization aligned with centered axes
    fig_geo, axg = plt.subplots(figsize=(6, 5))
    axg.set_xlim(y_centered[0] / µm, y_centered[-1] / µm)
    axg.set_ylim(z_centered[0] / µm, z_centered[-1] / µm)
    add_geometry_overlays(axg, y_centered, z_centered, core_wy, core_wz, substrate_thickness)
    axg.set_xlabel("y (µm)")
    axg.set_ylabel("z (µm)")
    axg.set_title("Waveguide Cross-section Geometry")
    axg.set_aspect('equal')
    fig_geo.tight_layout()
    fig_geo.savefig("modesolver_2d_geometry.png", dpi=200)
    plt.close(fig_geo)

    # Show multiple fields and modes: first row |Ez|, second row |Hy| for first up to 3 modes
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8), constrained_layout=True)
    for col in range(n_cols):
        mode: ModeTupleType = modes[col]
        Ez = np.array(mode.Ez)
        Hy = np.array(mode.Hy)
        Ez_mag = np.real(Ez)
        Hy_mag = np.real(Hy)

        ax_top = axes[0, col] if n_cols > 1 else axes[0]
        vmax_ez = np.max(np.abs(Ez_mag)) or 1.0
        im1 = ax_top.imshow(Ez_mag.T / vmax_ez, origin="lower",
                            extent=(y_centered[0] / µm, y_centered[-1] / µm, z_centered[0] / µm, z_centered[-1] / µm),
                            cmap="RdBu", aspect="equal", vmin=-1, vmax=1)
        add_geometry_overlays(ax_top, y_centered, z_centered, core_wy, core_wz, substrate_thickness)
        ax_top.set_title(f"Mode {col}: Re(Ez) (norm)\nneff = {float(np.real(mode.neff)):.3f}")
        ax_top.set_xlabel("y (µm)")
        ax_top.set_ylabel("z (µm)")
        plt.colorbar(im1, ax=ax_top, fraction=0.046, pad=0.04)

        ax_bot = axes[1, col] if n_cols > 1 else axes[1]
        vmax_hy = np.max(np.abs(Hy_mag)) or 1.0
        im2 = ax_bot.imshow(Hy_mag.T / vmax_hy, origin="lower",
                            extent=(y_centered[0] / µm, y_centered[-1] / µm, z_centered[0] / µm, z_centered[-1] / µm),
                            cmap="RdBu", aspect="equal", vmin=-1, vmax=1)
        add_geometry_overlays(ax_bot, y_centered, z_centered, core_wy, core_wz, substrate_thickness)
        ax_bot.set_title(f"Mode {col}: Re(Hy) (norm)")
        ax_bot.set_xlabel("y (µm)")
        ax_bot.set_ylabel("z (µm)")
        plt.colorbar(im2, ax=ax_bot, fraction=0.046, pad=0.04)

    fig.suptitle(f"2D Cross-section Modes, λ = {wavelength/µm:.2f} µm", y=1.02)
    fig.savefig("modesolver_2d_modes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

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
        for (name, field), ax in zip(fields.items(), ax_comp.ravel()):
            real_field = np.real(field)
            vmax = np.max(np.abs(real_field)) or 1.0
            im = ax.imshow(real_field.T / vmax, origin="lower",
                           extent=(y_centered[0] / µm, y_centered[-1] / µm, z_centered[0] / µm, z_centered[-1] / µm),
                           cmap="RdBu", aspect="equal", vmin=-1, vmax=1)
            add_geometry_overlays(ax, y_centered, z_centered, core_wy, core_wz, substrate_thickness)
            ax.set_title(f"Mode 0: Re({name}) (norm)")
            ax.set_xlabel("y (µm)")
            ax.set_ylabel("z (µm)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig_comp.suptitle(f"Mode 0 component fields, λ = {wavelength/µm:.2f} µm", y=1.02)
        fig_comp.savefig("modesolver_2d_mode0_components.png", dpi=200, bbox_inches="tight")
        plt.close(fig_comp)


if __name__ == "__main__":
    main()


