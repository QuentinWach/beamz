## Module Structure

### `design/` - Parametric Design and Geometry
Defines the physical structure of the device through parametric geometry and materials.

### `devices/` - Field Sources and Monitors
Handles electromagnetic field injection (sources) and detection (monitors) that interact with the simulation fields.

### `simulation/` - FDTD Engine
Orchestrates the finite-difference time-domain (FDTD) simulation and field evolution.

### `optimization/` - Gradient-Based Optimization
Provides topology optimization tools using JAX for automatic differentiation and the adjoint method.

### `visual/` - Visualization and UI
Handles plotting, animation, and command-line interface interactions.

### `const.py` - Physical Constants
Defines fundamental physical constants (light speed, vacuum permittivity/permeability) and unit conversions (µm, nm).

## Code Architecture

The codebase follows a **modular high-level design** with **object-oriented patterns within modules**:

- **Design-centric**: The `Design` class serves as the central container for structures, materials, sources, and monitors.
- **Simulation orchestration**: The `Simulation` class references a `Design` and manages the FDTD time-stepping, delegating field updates to the `Fields` class.
- **Device abstraction**: Sources and monitors inherit from the `Device` base class, providing a unified interface for field manipulation.
- **Separation of concerns**: Design geometry is separate from simulation execution, which is separate from optimization and visualization.
