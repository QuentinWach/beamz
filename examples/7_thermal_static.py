import numpy as np
import matplotlib.pyplot as plt
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

eps_r, temperature = apply_static_thermal(
    design,
    resolution=0.1e-6,
    params=params,
    heater_mask=heater_mask,
    heater_power=8e12,
)

plt.figure(figsize=(7, 3.2))
plt.imshow(
    temperature,
    origin="lower",
    extent=(0, W * 1e6, 0, H * 1e6),
    cmap="inferno",
)
plt.colorbar(label="Temperature (K)")
plt.title("Heated Chip Cross-Section (Static Solve)")
plt.xlabel("X (µm)")
plt.ylabel("Y (µm)")

# Draw structure outlines for clarity
ax = plt.gca()
outline_color = "white"
outline_alpha = 0.3
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

plt.tight_layout()
plt.show()
