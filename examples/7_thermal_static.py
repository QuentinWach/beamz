import numpy as np
import matplotlib.pyplot as plt

from beamz import (
    Design,
    GaussianSource,
    LIGHT_SPEED,
    Material,
    PML,
    Rectangle,
    Simulation,
    ThermalParams,
    apply_static_thermal,
)

# Domain
W, H = 12e-6, 5e-6
design = Design(width=W, height=H, material=Material(permittivity=1.0))

# Core material
core = Material(
    permittivity=3.48**2,
    conductivity=1.0,
    k=150.0,
    rho=2330.0,
    cp=700.0,
    dn_dT=1.86e-4,
    T0=300.0,
)
design += Rectangle(position=(2e-6, 2e-6), width=8e-6, height=1e-6, material=core)

# Heater material
heater = Material(
    permittivity=1.0,
    conductivity=1.0,
    k=80.0,
    rho=5000.0,
    cp=300.0,
    dn_dT=0.0,
    T0=300.0,
)
design += Rectangle(position=(3e-6, 3.4e-6), width=6e-6, height=0.4e-6, material=heater)

def heater_mask(x, y, z):
    return 3e-6 <= x <= 9e-6 and 3.4e-6 <= y <= 3.8e-6

# Static thermal pre-solve
params = ThermalParams(
    thermal_dt=1e-13,
    tau_avg=1e-13,
    steady_state=True,
    max_iters=4000,
    tol=1e-6,
)
eps_r, temperature = apply_static_thermal(
    design,
    resolution=0.1e-6,
    params=params,
    heater_mask=heater_mask,
    heater_power=5e12,
)

# Visualize temperature field
plt.figure(figsize=(6, 3))
plt.imshow(
    temperature,
    origin="lower",
    extent=(0, W * 1e6, 0, H * 1e6),
    cmap="inferno",
)
plt.colorbar(label="Temperature (K)")
plt.title("Static Temperature Field")
plt.xlabel("X (µm)")
plt.ylabel("Y (µm)")
plt.tight_layout()
plt.show()

# EM run with fixed permittivity (example)
dx = 0.1e-6
dt = 0.99 * dx / (LIGHT_SPEED * np.sqrt(2))
time = np.arange(0, 300 * dt, dt)
frequency = LIGHT_SPEED / 1.55e-6
signal = np.sin(2 * np.pi * frequency * time)
source = GaussianSource(position=(1.5e-6, 2.5e-6), width=0.4e-6, signal=signal)

# Inject permittivity grid directly into the mesh via CustomMaterial
sim = Simulation(
    design=design,
    devices=[source],
    boundaries=[PML(thickness=1.0e-6)],
    time=time,
    resolution=dx,
)

sim.fields.update_materials(permittivity=eps_r)
sim.run()
