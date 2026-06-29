# Getting Started

## Install

```bash
pip install beamz
```

For local development in this repository:

```bash
uv sync --all-extras
```

## Basic Workflow

A typical BEAMZ workflow has four steps:

1. Build a `Design` from materials and geometry.
2. Add sources and monitors.
3. Create and run a `Simulation`.
4. Convert or visualize the results.

```python
import beamz as bz
import numpy as np

um = bz.um

wl = 1.55 * um
time_end = 20 * wl / bz.LIGHT_SPEED
dx, dt = bz.dxdt(wl, n_max=3.48, dims=2, points_per_wavelength=10)
t = np.arange(0, time_end, dt)

cladding = bz.Material(permittivity=1.44**2)
core = bz.Material(permittivity=3.48**2)

design = bz.Design(width=6 * um, height=3 * um, material=cladding)
design += bz.Rectangle(
    position=(3 * um, 1.5 * um),
    width=4 * um,
    height=0.5 * um,
    material=core,
)

signal = bz.ramped_cosine(
    t,
    amplitude=1.0,
    frequency=bz.LIGHT_SPEED / wl,
    ramp_duration=3 * wl / bz.LIGHT_SPEED,
    t_max=time_end / 2,
)

source = bz.GaussianSource(
    position=(1 * um, 1.5 * um),
    width=wl / 6,
    signal=signal,
)
monitor = bz.Monitor(
    start=(5 * um, 0.75 * um),
    end=(5 * um, 2.25 * um),
    name="output",
)

sim = bz.Simulation(
    design=design,
    sources=[source],
    monitors=[monitor],
    boundaries=[bz.PML(edges="all", thickness=wl)],
    time=t,
    resolution=dx,
)

results = sim.save_video(
    "dipole.mp4",
    field="Ez",
    progress=False,
    animation_interval=3,
    video_fps=30,
    cmap="twilight_zero",
    clean_visualization=True,
)
```

{% video src="/assets/videos/dipole.mp4" /%}

The exact source, monitor, and simulation options depend on the dimensionality
and physics you need. See the API pages for the stable public entry points.

## Next Steps

- Browse the [API Reference](api/reference/index.md) for stable public imports.
- Open the Examples tab for runnable notebooks and rendered simulation outputs.
