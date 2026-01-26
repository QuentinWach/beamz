# BEAMZ Quick Start

## Installation

### For Users
```bash
pip install beamz
```

### For Developers

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup**:
   ```bash
   git clone https://github.com/QuentinWach/beamz.git
   cd beamz
   uv sync  # Installs everything automatically
   ```

## Common Commands

```bash
# Development
make install         # Install package
make test           # Run all tests
make test-fast      # Run quick tests
make format         # Format code
make lint           # Check code quality

# Testing
make test-single FILE=test_physics_energy.py  # Run one test file
uv run pytest tests/test_mode_source.py -v   # Run specific tests

# Documentation
make docs-serve     # View docs locally

# Dependencies
uv add <package>    # Add new dependency
uv remove <package> # Remove dependency
uv sync --upgrade   # Update all packages

# Building
make build          # Build distribution
make publish        # Upload to PyPI
```

## Example Usage

```python
from beamz import *
import numpy as np

# Setup
wavelength = 1.55*µm
resolution, dt = calc_optimal_fdtd_params(wavelength, n_max=2.5, dims=2)

# Create design
design = Design(width=10*µm, height=5*µm, material=Material(1.444**2))
design += Rectangle(position=(5*µm, 2.5*µm), width=5*µm, height=1*µm,
                    material=Material(2.25**2))

# Setup simulation
time = np.arange(0, 20*wavelength/LIGHT_SPEED, dt)
signal = ramped_cosine(time, 1.0, LIGHT_SPEED/wavelength,
                       ramp_duration=3*wavelength/LIGHT_SPEED)
source = GaussianSource(center=(2*µm, 2.5*µm), width=1*µm, signal=signal)

# Run
sim = Simulation(design, devices=[source],
                 boundaries=[PML(edges='all', thickness=1*µm)],
                 time=time, resolution=resolution)
sim.run(animate_live="Ez")
```

## Project Structure

```
beamz/
├── design/         # Geometry and materials (Design, Rectangle, Material)
├── simulation/     # FDTD engine (Simulation, Fields)
├── devices/        # Sources and monitors (ModeSource, GaussianSource, Monitor)
├── optimization/   # Topology optimization (TopologyManager)
└── visual/         # Visualization and UI helpers
```

## Need Help?

- 📖 **Docs**: [quentinwach.github.io/beamz](https://quentinwach.github.io/beamz)
- 🐛 **Issues**: [github.com/QuentinWach/beamz/issues](https://github.com/QuentinWach/beamz/issues)
- 💬 **Discussions**: [github.com/QuentinWach/beamz/discussions](https://github.com/QuentinWach/beamz/discussions)

## Key Features

- ⚡ **Fast**: JAX-accelerated FDTD with GPU support
- 🎨 **High-level API**: Design devices with just a few lines
- 🔧 **Topology Optimization**: Inverse design with autodiff
- 📊 **Visualization**: Built-in plotting and animations
- 🎯 **Mode Sources**: Realistic waveguide excitation
- 🧪 **Physics-Validated**: Extensive test suite
