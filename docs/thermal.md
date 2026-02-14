# Thermal Coupling

This guide covers both thermal workflows in BEAMZ:
- EM-coupled transient thermal updates during FDTD (`ThermalCoupling`)
- static thermal pre-solve and heater tuning workflows (`Design.solve_thermal`, `Design.sweep_mzi_heater`)

## Overview

- Joule heating: `Q = σ⟨|E|^2⟩`
- Heat equation: `rho * Cp * dT/dt = ∇·(k ∇T) + Q`
- Spatial discretization: finite-volume-style `div(k grad T)` with harmonic face conductivity
- Thermo-optic update: `n = n0 + dn/dT * (T - T0)`, `εr = n^2`

Thermal parameters are defined per material (`k`, `rho`, `cp`, `dn_dT`, `T0`) and rasterized onto the simulation grid.

## Transient EM-Coupled Example

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

thermal = ThermalCoupling(ThermalConfig(thermal_dt=1e-13, tau_avg=1e-13, T0=300))

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

## Static Thermal Workflow (Designer API)

Use `Design.solve_thermal(...)` with:
- `StaticThermalConfig` for solver controls
- `ThermalScenario` for sources/sinks/boundaries
- optional `ThermalBoundaryProfile.photonic_chip(...)` preset

```python
import numpy as np

from beamz import (
    Design,
    Material,
    Rectangle,
    StaticThermalConfig,
    ThermalBoundaryProfile,
    ThermalScenario,
    ThermalSource,
)

W, H = 12e-6, 5e-6
design = Design(width=W, height=H, material=Material(permittivity=1.0, k=0.026, T0=300.0))

heater_mat = Material(permittivity=1.0, k=80.0, rho=5000.0, cp=300.0, T0=300.0)
heater = Rectangle(position=(3e-6, 3.4e-6), width=6e-6, height=0.4e-6, material=heater_mat)
design += heater

scenario = ThermalScenario(
    # Required in 2D when using total heater power (W)
    extrusion_depth_m=100e-6,
    boundary_profile=ThermalBoundaryProfile.photonic_chip(
        sink_thickness_m=0.2e-6,
        sink_temperature_k=300.0,
        top_h_w_m2_k=10.0,
        ambient_temp_k=300.0,
    ),
    sources=[
        ThermalSource(region=heater, power_w=0.02),
    ],
)

result = design.solve_thermal(
    resolution=0.1e-6,
    scenario=scenario,
    config=StaticThermalConfig(max_iters=8000, tol=1e-6),
)

eps_r, temperature = result.permittivity, result.temperature
```

### Region Definitions

`ThermalSource.region` and `ThermalSink.region` accept:
- structure references (for example `Rectangle`, `Ring`, ...)
- callables `(x, y, z) -> bool`
- bool arrays matching thermal grid shape
- iterables of structure/callable/array regions

### 2D Power Semantics

When a source uses `power_w` in 2D, `scenario.extrusion_depth_m` is required.
BEAMZ converts total heater power to volumetric source as:

`Q = power_w / (active_cells * dx * dy * extrusion_depth_m)`

If you already know volumetric heating, use `power_density_w_m3` directly.

## Practical Demo: MZI Heater Tuning

Run the practical photonic workflow demo:

```bash
python -m examples.thermal_mzi_phase_shifter
```

This demo uses:
- structure-referenced heater region
- photonic-chip boundary profile preset
- power sweep via `Design.sweep_mzi_heater(...)`
- outputs `ΔT`, `Δn_eff`, `Δφ`, and estimated `Pπ`

## Benchmarks

### 1D Analytical Slab (Dirichlet + Robin)

```bash
python -m examples.thermal_benchmark_slab
```

### 2D Manufactured-Solution Benchmark

```bash
python -m examples.thermal_benchmark_mms2d
```

## Migration Note (Breaking Change)

The old static signature was removed:

```python
# removed
Design.solve_static_thermal(..., heater_mask=..., heater_power=..., fixed_temp_mask=..., fixed_temp_value=...)
```

Use:

```python
Design.solve_thermal(..., scenario=ThermalScenario(...), config=StaticThermalConfig(...))
```
