import numpy as np

from beamz import (
    Design,
    GaussianSource,
    LIGHT_SPEED,
    Material,
    PML,
    Rectangle,
    Simulation,
    ThermalParams,
    ThermoPhysics,
)

# Domain
W, H = 12e-6, 5e-6
design = Design(width=W, height=H, material=Material(permittivity=1.0))

# Core material with thermal properties
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

# FDTD params
dx = 0.1e-6
dt = 0.99 * dx / (LIGHT_SPEED * np.sqrt(2))
time = np.arange(0, 400 * dt, dt)

# Source
frequency = LIGHT_SPEED / 1.55e-6
signal = np.sin(2 * np.pi * frequency * time)
source = GaussianSource(position=(1.5e-6, 2.5e-6), width=0.4e-6, signal=signal)

# Thermal coupling
thermal = ThermoPhysics(
    ThermalParams(thermal_dt=1e-13, tau_avg=1e-13, T0=300.0)
)

sim = Simulation(
    design=design,
    devices=[source],
    boundaries=[PML(thickness=1.0e-6)],
    time=time,
    resolution=dx,
    thermal=thermal,
)

sim.run()

print(f"Max temperature: {float(np.max(np.asarray(thermal.T))):.2f} K")
