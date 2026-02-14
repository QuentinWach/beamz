import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle as PlotRectangle
from matplotlib.ticker import FuncFormatter

from beamz import Design, Material, Rectangle, ThermalConfig


def main():
    # Chip cross-section (top to bottom): air, heater, top oxide, BOX, Si substrate.
    # Use a taller substrate so heat can spread before reaching the thermal anchor.
    W, H = 30e-6, 18e-6
    # Air cladding (room-temperature thermal conductivity).
    design = Design(
        width=W, height=H, material=Material(permittivity=1.0, k=0.026, T0=300.0)
    )

    box = Material(
        permittivity=1.44**2, k=1.38, rho=2200.0, cp=703.0, dn_dT=1e-5, T0=300.0
    )
    top_oxide = Material(
        permittivity=1.44**2, k=1.38, rho=2200.0, cp=703.0, dn_dT=1e-5, T0=300.0
    )
    # Silicon handle wafer (effective substrate sink path).
    substrate = Material(
        permittivity=3.45**2, k=130.0, rho=2330.0, cp=700.0, dn_dT=1.86e-4, T0=300.0
    )
    # Metal heater (representative TiN-like thermal properties).
    heater = Material(permittivity=1.0, k=25.0, rho=5200.0, cp=540.0, T0=300.0)

    # Layers
    design += Rectangle(position=(0, 0.0), width=W, height=12.0e-6, material=substrate)
    design += Rectangle(position=(0, 12.0e-6), width=W, height=2.0e-6, material=box)
    design += Rectangle(
        position=(0, 14.0e-6), width=W, height=3.0e-6, material=top_oxide
    )
    design += Rectangle(
        position=(12e-6, 16.4e-6), width=6e-6, height=0.3e-6, material=heater
    )

    def heater_mask(x, y, z):
        return 12e-6 <= x <= 18e-6 and 16.4e-6 <= y <= 16.7e-6

    params = ThermalConfig(
        thermal_dt=1e-13,
        tau_avg=1e-13,
        max_iters=8000,
        tol=1e-6,
        # Robin BC proxy for natural convection to ambient air at the top surface.
        robin_h=10.0,
        robin_T_ambient=300.0,
        robin_sides=("top",),
    )

    def backside_sink_mask(x, y, z):
        # Backside thermal anchor: only a thin bottom slice is clamped to ambient.
        # This better approximates heat flowing into a heat sink/chuck than pinning
        # the entire substrate volume to 300 K.
        return 0.0 <= y <= 0.2e-6

    result = design.solve_static_thermal(
        resolution=0.1e-6,
        config=params,
        heater_mask=heater_mask,
        # Tuned for this stack to produce a realistic hotspot range (~300-380 K).
        heater_power=2e14,
        fixed_temp_mask=backside_sink_mask,
        fixed_temp_value=300.0,
    )
    eps_r, temperature = result.permittivity, result.temperature
    _ = eps_r

    # Compute heat flux for visualization (2D)
    dx = 0.1e-6
    grad_y, grad_x = np.gradient(temperature, dx, dx)
    k_grid, _, _, _, _ = design.get_thermal_grids(dx)
    qx = -k_grid * grad_x
    qy = -k_grid * grad_y
    qmag = np.sqrt(qx**2 + qy**2)
    # Mask air + fixed-temperature sink regions for visualization
    sink_mask = np.zeros_like(qmag, dtype=bool)
    for i in range(qmag.shape[0]):
        y = (i + 0.5) * dx
        if 0.0 <= y <= 0.2e-6:
            sink_mask[i, :] = True
    solid_mask = (k_grid > 0) & (~sink_mask)
    qmag_solid = np.where(solid_mask, qmag, 0.0)
    q_vis = np.log10(1.0 + qmag_solid)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 6.4))
    extent = (0, W * 1e6, 0, H * 1e6)

    im0 = ax0.imshow(temperature, origin="lower", extent=extent, cmap="inferno")
    temp_cbar = fig.colorbar(im0, ax=ax0, label="Temperature (K)")
    temp_cbar.formatter = FuncFormatter(lambda x, pos: f"{int(round(x))}K")
    temp_cbar.update_ticks()
    ax0.set_title("Heated Chip Cross-Section (Static Solve)")
    ax0.set_xlabel("X (µm)")
    ax0.set_ylabel("Y (µm)")

    # Draw structure outlines for clarity
    outline_color = "white"
    outline_alpha = 0.5
    structures = [
        (0, 0.0, W, 12.0e-6),  # substrate
        (0, 12.0e-6, W, 2.0e-6),  # BOX
        (0, 14.0e-6, W, 3.0e-6),  # top oxide
        (12e-6, 16.4e-6, 6e-6, 0.3e-6),  # heater
    ]
    for x, y, w, h in structures:
        ax0.add_patch(
            PlotRectangle(
                (x * 1e6, y * 1e6),
                w * 1e6,
                h * 1e6,
                fill=False,
                edgecolor=outline_color,
                linewidth=1.2,
                alpha=outline_alpha,
            )
        )

    im1 = ax1.imshow(
        q_vis,
        origin="lower",
        extent=extent,
        cmap="magma",
    )
    fig.colorbar(im1, ax=ax1, label="log10(1 + |Heat Flux|)")
    ax1.set_title("Heat Flux Magnitude + Direction")
    ax1.set_xlabel("X (µm)")
    ax1.set_ylabel("Y (µm)")

    # Streamlines for flux direction (mask air/sink regions)
    U = np.where(solid_mask, qx, np.nan)
    V = np.where(solid_mask, qy, np.nan)
    y = (np.arange(qmag.shape[0]) + 0.5) * dx * 1e6
    x = (np.arange(qmag.shape[1]) + 0.5) * dx * 1e6
    ax1.streamplot(
        x,
        y,
        U,
        V,
        color="white",
        linewidth=1.0,
        density=1.1,
        arrowsize=0.8,
    )

    # Draw structure outlines on flux plot too
    for x, y, w, h in structures:
        ax1.add_patch(
            PlotRectangle(
                (x * 1e6, y * 1e6),
                w * 1e6,
                h * 1e6,
                fill=False,
                edgecolor=outline_color,
                linewidth=1.0,
                alpha=outline_alpha,
            )
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
