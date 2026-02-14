# Thermal Coupling

This guide shows how to run a transient thermal solve alongside the EM FDTD loop.

## Overview

- Joule heating: `Q = σ⟨|E|^2⟩`
- Heat equation: `rho * Cp * dT/dt = ∇·(k ∇T) + Q`
- Thermo-optic update: `n = n0 + dn/dT * (T - T0)`, `εr = n^2`

Thermal parameters are defined per material and rasterized onto the simulation grid.

## Example

```python
import numpy as np

from beamz import (
    Design,
    Material,
    Rectangle,
    GaussianSource,
    PML,
    Simulation,
    ThermalConfig,
    ThermalCoupling,
    LIGHT_SPEED,
)

W, H = 10e-6, 4e-6
design = Design(width=W, height=H, material=Material(permittivity=1.0))

core = Material(
    permittivity=3.48**2,
    conductivity=1.0,
    k=150,
    rho=2330,
    cp=700,
    dn_dT=1.86e-4,
    T0=300,
)
design += Rectangle(position=(2e-6, 1.5e-6), width=6e-6, height=1e-6, material=core)

dx = 0.1e-6
dt = 0.99 * dx / (LIGHT_SPEED * np.sqrt(2))
time = np.arange(0, 200 * dt, dt)

signal = np.sin(2 * np.pi * LIGHT_SPEED / 1.55e-6 * time)
source = GaussianSource(position=(1e-6, 2e-6), width=0.3e-6, signal=signal)

thermal = ThermalCoupling(
    ThermalConfig(thermal_dt=1e-13, tau_avg=1e-13, T0=300)
)

sim = Simulation(
    design=design,
    devices=[source],
    boundaries=[PML(thickness=1e-6)],
    time=time,
    resolution=dx,
    thermal=thermal,
)

sim.run()
```

## Static Pre-Solve (Heater Mask)

You can solve the steady-state heat equation before running EM to precompute
temperature gradients and update permittivity.

```python
import numpy as np

from beamz import (
    Design,
    Material,
    Rectangle,
    ThermalConfig,
)

W, H = 12e-6, 5e-6
design = Design(width=W, height=H, material=Material(permittivity=1.0))

heater = Material(permittivity=1.0, conductivity=1.0, k=80.0, rho=5000.0, cp=300.0)
design += Rectangle(position=(3e-6, 3.4e-6), width=6e-6, height=0.4e-6, material=heater)

def heater_mask(x, y, z):
    return 3e-6 <= x <= 9e-6 and 3.4e-6 <= y <= 3.8e-6

params = ThermalConfig(thermal_dt=1e-13, tau_avg=1e-13, steady_state=True)
result = design.solve_static_thermal(
    resolution=0.1e-6,
    config=params,
    heater_mask=heater_mask,
    heater_power=5e12,
)
eps_r, temperature = result.permittivity, result.temperature
```
