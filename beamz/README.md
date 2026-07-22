## Module Structure

### `analysis/` - Modal Analysis, S-Parameters, and Plotting
Contains modal projection, port/S-parameter extraction, compact plotting helpers, and small result adapters used by examples and notebooks.

### `design/` - Parametric Design and Geometry
Defines the physical structure of the device through parametric geometry and materials.

### `devices/` - Field Sources and Monitors
Handles electromagnetic field injection (sources) and detection (monitors) that interact with the simulation fields.

### `simulation/` - FDTD Engine
Orchestrates the finite-difference time-domain (FDTD) simulation and field evolution.

### `optimization/` - Inverse Design Helpers
Contains topology optimization, autodiff utilities, adjoint field-history storage, and density polygonization.

### `const.py` - Physical Constants
Defines fundamental physical constants (light speed, vacuum permittivity/permeability) and unit conversions (µm, nm).

### Root foundations

`lattice.py` is intentionally separate from `const.py`: constants are dependency-free
scalars, while the lattice module owns the shared NumPy/JAX Yee geometry and material
sampling used by design, devices, simulation, and analysis. Merging them would make a
constant import load the numerical stack and would erase that dependency boundary.

The two private root helpers are cross-package foundations rather than package-owned
behavior: `_cache_tokens.py` provides canonical value hashing for immutable specs and
`_helpers.py` contains the small validation, unit-display, FDTD-step, logging, and
progress utilities shared by otherwise independent packages. Keep package-specific
helpers beside their owner instead of adding more root utility files.

## Code Architecture

The codebase uses immutable specifications with explicit runtime state:

- **Design geometry**: `Design` owns the background material and ordered, immutable structures. It does not own sources, monitors, or evolving fields.
- **Simulation orchestration**: `Simulation` combines a `Design` with source, monitor, boundary, time, and grid specifications. It lowers them into an immutable compiled program.
- **Runtime and results**: `SimulationState` contains the evolving Yee fields; `SimulationRun` keeps that continuation value separate from detached, immutable `SimulationResults`.
- **Device abstraction**: Sources, monitors, and boundaries are immutable device specifications compiled into grid-aware runtime data.
- **Separation of concerns**: Design geometry, devices, solver execution, analysis, and optimization remain separate packages; caches live outside immutable specifications and plans.
