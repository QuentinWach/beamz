import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle as PlotRectangle

from beamz import Design, Material, Rectangle, ThermalParams, apply_static_thermal

# Chip cross-section (top to bottom): air, heater, oxide, silicon, substrate
W, H = 20e-6, 8e-6
design = Design(width=W, height=H, material=Material(permittivity=1.0, k=0.03, T0=300.0))

oxide = Material(permittivity=1.44**2, k=1.4, rho=2200.0, cp=703.0, dn_dT=1e-5, T0=300.0)
silicon = Material(permittivity=3.48**2, k=148.0, rho=2330.0, cp=700.0, dn_dT=1.86e-4, T0=300.0)
substrate = Material(permittivity=3.4**2, k=120.0, rho=2650.0, cp=750.0, dn_dT=1.5e-4, T0=300.0)
heater = Material(permittivity=1.0, k=80.0, rho=5000.0, cp=300.0, T0=300.0)

# Layers
design += Rectangle(position=(0, 0.0), width=W, height=2.5e-6, material=substrate)
design += Rectangle(position=(0, 2.5e-6), width=W, height=2.0e-6, material=silicon)
design += Rectangle(position=(0, 4.5e-6), width=W, height=2.5e-6, material=oxide)
design += Rectangle(position=(7e-6, 6.7e-6), width=6e-6, height=0.4e-6, material=heater)

def heater_mask(x, y, z):
    return 7e-6 <= x <= 13e-6 and 6.7e-6 <= y <= 7.1e-6

params = ThermalParams(
    thermal_dt=1e-13,
    tau_avg=1e-13,
    steady_state=True,
    max_iters=6000,
    tol=1e-6,
)

def substrate_and_air_sink_mask(x, y, z):
    # Bottom substrate sink and top air sink above the heater
    return (0.0 <= y <= 2.5e-6) or (7.1e-6 <= y <= H)

eps_r, temperature = apply_static_thermal(
    design,
    resolution=0.1e-6,
    params=params,
    heater_mask=heater_mask,
    heater_power=5e16,
    fixed_temp_mask=substrate_and_air_sink_mask,
    fixed_temp_value=300.0,
)

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
    if 0.0 <= y <= 2.5e-6 or 7.1e-6 <= y <= H:
        sink_mask[i, :] = True
solid_mask = (k_grid > 0) & (~sink_mask)
qmag_solid = np.where(solid_mask, qmag, 0.0)
q_vis = np.log10(1.0 + qmag_solid)

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 6.4))
extent = (0, W * 1e6, 0, H * 1e6)

im0 = ax0.imshow(
    temperature,
    origin="lower",
    extent=extent,
    cmap="inferno"
)
temp_cbar = fig.colorbar(im0, ax=ax0, label="Temperature (K)")
temp_cbar.formatter = FuncFormatter(lambda x, pos: f"{x:.2f}K")
temp_cbar.update_ticks()
ax0.set_title("Heated Chip Cross-Section (Static Solve)")
ax0.set_xlabel("X (µm)")
ax0.set_ylabel("Y (µm)")

# Draw structure outlines for clarity
ax = ax0
outline_color = "white"
outline_alpha = 0.5
structures = [
    (0, 0.0, W, 2.5e-6),     # substrate
    (0, 2.5e-6, W, 2.0e-6),  # silicon
    (0, 4.5e-6, W, 2.5e-6),  # oxide
    (7e-6, 6.7e-6, 6e-6, 0.4e-6),  # heater
]
for x, y, w, h in structures:
    ax.add_patch(
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
