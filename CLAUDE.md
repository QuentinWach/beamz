# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BEAMZ is an electromagnetic simulation package using the FDTD (Finite-Difference Time-Domain) method for photonic integrated circuits. It features:
- High-level API for fast prototyping
- Inverse design module for topology optimization using the adjoint method
- JAX-based autodifferentiation for optimization
- Support for both 2D and 3D simulations

## Project Management with uv

This project uses **uv** for fast, reliable Python package and project management. uv provides reproducible environments and faster installation than pip.

### Prerequisites
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or on macOS with Homebrew:
brew install uv
```

### Quick Start
```bash
# Clone and setup in one command
git clone <repo-url> && cd beamz && uv sync
```

## Build and Development Commands

All commands can be run via **Makefile** (recommended) or directly with uv:

### Environment Setup
```bash
# Install package in development mode (recommended)
make install
# or: uv sync

# Install with all extras (dev, test, gpu dependencies)
make install-dev
# or: uv sync --all-extras

# Install specific extra only
uv sync --extra gpu
```

### Testing
```bash
# Run all tests with coverage
make test
# or: uv run pytest tests/ -v --tb=short --cov=beamz --cov-report=xml

# Run tests excluding slow ones
make test-fast
# or: uv run pytest tests/ -v -m "not slow" --tb=short

# Run a single test file
make test-single FILE=test_physics_energy.py
# or: uv run pytest tests/test_physics_energy.py -v

# Run tests by marker
uv run pytest -m design      # Only design module tests
uv run pytest -m simulation  # Only simulation module tests

# Run specific test function
uv run pytest tests/test_physics_energy.py::test_energy_conservation -v
```

### Code Quality
```bash
# Format code (black + isort)
make format
# or: uv run black beamz/ tests/ && uv run isort beamz/ tests/

# Check formatting without changing files
make format-check

# Lint
make lint
# or: uv run flake8 beamz/ tests/
```

### Documentation
```bash
# Serve documentation locally
make docs-serve
# or: uv run mkdocs serve

# Deploy documentation to GitHub Pages
make docs-deploy
# or: uv run mkdocs gh-deploy
```

### Versioning and Release
```bash
# Create a new version release with git tag
make version VERSION=0.1.X
# or: python release_version.py 0.1.X

# Create GitHub release (requires GITHUB_TOKEN)
python release_version.py 0.1.X --message "Release notes"
```

### Package Publishing
```bash
# Build distribution (uses uv build)
make build
# or: uv build

# Publish to PyPI
make publish
# or: uv run twine upload dist/*

# Note: The old patch_wheel.py step is no longer needed with uv/hatchling
```

### Clean Build Artifacts
```bash
make clean
```

### Dependency Management with uv

```bash
# Add a new dependency
uv add <package-name>

# Add a dev dependency
uv add --dev <package-name>

# Add with version constraint
uv add "numpy>=1.24,<2.0"

# Remove a dependency
uv remove <package-name>

# Update all dependencies
uv sync --upgrade

# Update a specific package
uv add --upgrade <package-name>

# Lock dependencies without installing
uv lock

# Show outdated packages
uv pip list --outdated
```

### Python Version Management

The project uses Python 3.11 by default (specified in `.python-version`). uv automatically uses this version.

```bash
# Check Python version being used
uv run python --version

# Use a different Python version temporarily
uv run --python 3.10 pytest tests/
```

## Architecture Overview

### Core Data Flow

The simulation pipeline follows this flow:
1. **Design** → 2. **Meshing** → 3. **Simulation** → 4. **Visualization/Optimization**

### Module Structure

#### `beamz/design/`
- **`core.py`**: `Design` class - the top-level container for structures, sources, and monitors
  - Manages structure lists and material properties
  - Provides `rasterize()` method to convert to simulation grid
  - Handles polygon unification for overlapping structures
- **`structures.py`**: Geometric primitives (Rectangle, Circle, Ring, Polygon, Taper, Sphere, etc.)
- **`materials.py`**: Material definitions (`Material`, `CustomMaterial`)
- **`meshing.py`**: Grid generation classes
  - `RegularGrid`: 2D meshing
  - `RegularGrid3D`: 3D meshing with depth handling
  - Converts design geometry to material property grids (permittivity, conductivity, permeability)

#### `beamz/simulation/`
- **`core.py`**: `Simulation` class - the FDTD engine
  - Owns time stepping logic
  - References material grids from Design (no duplication)
  - Coordinates field updates, source injection, and monitor recording
  - Handles boundary condition application (PML, ABC)
- **`fields.py`**: `Fields` class - field storage and update logic
  - Manages E/H field arrays on staggered Yee grid
  - Uses JAX for field arrays to enable GPU acceleration and autodiff
  - Provides `update()` method for FDTD time stepping
  - Supports both 2D (Ez, Hx, Hy) and 3D (Ex, Ey, Ez, Hx, Hy, Hz) simulations
- **`ops.py`**: Low-level FDTD operations (curl operators, field updates)
  - JAX-based implementations for performance
  - Separate functions for 2D vs 3D operations
- **`boundaries.py`**: Boundary conditions (PML, ABC, PeriodicBoundary)

#### `beamz/devices/`
- **`sources/`**: Electromagnetic sources
  - `mode.py`: `ModeSource` - waveguide mode excitation using mode solver (requires tidy3d for mode solving)
  - `gaussian.py`: `GaussianSource` - Gaussian beam sources
  - `signals.py`: Temporal signal functions (ramped_cosine, etc.)
- **`monitors/`**: Field and power monitors for data collection during simulation

#### `beamz/optimization/`
- **`topology.py`**: `TopologyManager` class - manages topology optimization
  - Uses JAX for autodifferentiation
  - Implements density-based topology optimization with filters
  - Supports conic, morphological, and blur filters
  - `compute_overlap_gradient()`: Calculates gradients using adjoint method
- **`autodiff.py`**: JAX-based autodiff utilities for gradient computation

#### `beamz/visual/`
- **`viz.py`**: Visualization functions for fields, animations, and results
  - `animate_manual_field()`: Create animated visualizations
  - `VideoRecorder`: Record simulation videos (handles 3D→2D slicing)
- **`helpers.py`**: UI utilities using Rich library for terminal output
  - `calc_optimal_fdtd_params()`: Calculate optimal resolution and timestep for stability

### Key Architectural Patterns

#### Material Grid Ownership
- `Design` owns the material grids (permittivity, conductivity, permeability)
- `Simulation` and `Fields` reference these grids (no duplication)
- Material grids are converted to JAX arrays in `Fields` for GPU acceleration

#### 2D vs 3D Simulation
- 2D simulations use Ez, Hx, Hy field components (TM mode) or Ex, Ey, Hz (TE mode)
- 3D simulations use all 6 field components (Ex, Ey, Ez, Hx, Hy, Hz)
- The `plane_2d` parameter specifies which plane for 2D projections ('xy', 'yz', 'xz')
- Field update logic automatically selected based on grid dimensionality

#### Source Injection Methods
- **New method**: Direct field injection via `inject()` method on source objects
- **Legacy method**: Source terms (J, M) collected and passed to field update
- Both coexist for backward compatibility

#### PML Implementation
- Uses effective conductivity approach (not split-field UPML)
- PML regions created once during initialization
- Conductivity profiles added to material grids before field updates

#### Topology Optimization Flow
1. Create `TopologyManager` with optimization region mask
2. For each optimization step:
   - Run forward simulation to compute objective
   - Run adjoint simulation with adjoint source at target
   - Compute gradient using `compute_overlap_gradient()` (field overlap integral)
   - Update density parameters using optimizer (Adam/SGD via optax)
   - Apply filters (conic/morphological/blur) and projection
   - Update material grid from physical density

### Important Constants and Units
- Defined in `beamz/const.py`
- Convenience units: `µm`, `nm` (both spellings supported)
- Physical constants: `LIGHT_SPEED`, `EPS_0`, `MU_0`

### Dependencies
- **Core**: numpy, matplotlib, scipy, shapely, rich, gdspy
- **Autodiff/Optimization**: jax, jaxlib, optax
- **Mode solving**: tidy3d (in dev dependencies, for ModeSource)
- **GPU acceleration**: torch (optional extra)

### Project Configuration Files

**Key files for dependency management:**
- **`pyproject.toml`**: Single source of truth for project metadata, dependencies, and tool configuration
  - Uses `hatchling` as build backend (modern, no setup.py needed)
  - Minimum Python version: 3.10 (required by jaxlib)
  - All tool configurations (pytest, black, isort) are in this file
- **`uv.lock`**: Lockfile for reproducible installations (committed to git)
- **`.python-version`**: Specifies Python 3.11 for development (committed to git)
- **`Makefile`**: Convenient shortcuts for common development tasks

**Legacy files (kept for compatibility):**
- **`setup.py`**: Still present but pyproject.toml is the source of truth
- **`pytest.ini`**: Can be removed (config now in pyproject.toml)

## Testing Notes

- Tests are organized by functionality (physics validation, benchmarks, mode sources)
- `tests/conftest.py` contains shared fixtures
- `tests/utils.py` contains test utilities
- Physics tests validate against analytical solutions (Fresnel reflection, PML absorption, etc.)
- Most tests use tidy3d for reference mode solving - install with `pip install -e ".[test]"`

## Common Development Patterns

### Creating a new simulation example
```python
from beamz import *
import numpy as np

# 1. Setup parameters
wavelength = 1.55*µm
resolution, dt = calc_optimal_fdtd_params(wavelength, n_max=2.5, dims=2)

# 2. Create design
design = Design(width=10*µm, height=5*µm, material=Material(1.444**2))
design += Rectangle(position=(0,0), width=5*µm, height=1*µm, material=Material(2.25**2))

# 3. Create sources and monitors
time = np.arange(0, 20*wavelength/LIGHT_SPEED, dt)
signal = ramped_cosine(time, 1.0, LIGHT_SPEED/wavelength, ramp_duration=3*wavelength/LIGHT_SPEED)
source = GaussianSource(center=(2*µm, 2.5*µm), width=1*µm, signal=signal)

# 4. Run simulation
sim = Simulation(design, devices=[source], boundaries=[PML(edges='all', thickness=1*µm)],
                 time=time, resolution=resolution)
sim.run(animate_live="Ez")
```

### Adding a new structure type
1. Create class in `beamz/design/structures.py`
2. Inherit from base structure or implement `vertices` property
3. Add to `__init__.py` exports
4. Ensure it works with `Design.rasterize()` meshing

### Version numbers
Version numbers must be updated in three files:
- `setup.py` (legacy, for compatibility)
- `pyproject.toml` (primary source)
- `beamz/__init__.py`

Use `release_version.py` script to update all three consistently.

## Best Practices for Development

### Adding New Dependencies
- Always use `uv add <package>` instead of manually editing pyproject.toml
- This ensures the lockfile stays in sync
- For dev tools: `uv add --dev <package>`
- For optional features: Add to appropriate `[project.optional-dependencies]` group in pyproject.toml

### Before Committing
```bash
# Ensure code is formatted
make format

# Run linting
make lint

# Run tests
make test-fast  # Quick validation
# or for full coverage:
make test
```

### Working with Virtual Environments
- uv automatically manages a `.venv/` directory
- Activate it manually if needed: `source .venv/bin/activate`
- Or always prefix commands with `uv run` (recommended)
- The `.venv/` directory is gitignored and should not be committed

### CI/CD Integration
The GitHub Actions workflows (`.github/workflows/tests.yml`) should be updated to use uv:
```yaml
- name: Install uv
  run: curl -LsSf https://astral.sh/uv/install.sh | sh

- name: Install dependencies
  run: uv sync --all-extras

- name: Run tests
  run: uv run pytest tests/ -v
```

### Troubleshooting

**Lock file conflicts:**
```bash
# Regenerate lock file
uv lock --upgrade
```

**Dependency resolution issues:**
```bash
# Clear cache and retry
uv cache clean
uv sync
```

**Python version issues:**
```bash
# List available Python versions
uv python list

# Install a specific Python version
uv python install 3.11

# Use specific Python for sync
uv sync --python 3.11
```
