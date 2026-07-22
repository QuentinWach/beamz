# Getting Started

**BEAMZ** is a **GPU-accelerated** [electromagnetic](https://en.wikipedia.org/wiki/Electromagnetism) simulation framework for photonic chip designers using the [FDTD (Finite-difference time-domain) method](https://en.wikipedia.org/wiki/Finite-difference_time-domain_method). It enables fast, large-scale simulations and offers a **familiar, high-level API** for quick prototyping with just a few lines of code, as well as an **inverse design module** for gradient-based optimization using the **adjoint method**.


## Install

Install BEAMZ using pip:

```bash
pip install beamz
```

For development installation, clone the repository and install in editable mode:

```bash
git clone https://github.com/quentinwach/beamz
cd beamz
pip install -e ".[dev]"
```


## Basic Workflow

In the core workflow, you

1. build a design,
2. add sources and monitors,
3. then run the compile simulation from these specs.

For example:

```python
import numpy as np
import beamz as bz

wl = 1.55 * bz.um
dx, dt = bz.dxdt(wl, n_max=3.48, dims=2, points_per_wavelength=10)
time = np.arange(0, 20 * wl / bz.LIGHT_SPEED, dt)

cladding = bz.Material(permittivity=1.44**2)
core = bz.Material(permittivity=3.48**2)

design = bz.Design(width=6 * bz.um, height=3 * bz.um, material=cladding)
design += bz.Rectangle(
    position=(1 * bz.um, 1.25 * bz.um),
    width=4 * bz.um,
    height=0.5 * bz.um,
    material=core,
)

signal = bz.ramped_cosine(
    time,
    amplitude=1.0,
    frequency=bz.LIGHT_SPEED / wl,
    ramp_duration=3 * wl / bz.LIGHT_SPEED,
    t_max=time[-1] / 2,
)
source = bz.GaussianSource(
    position=(1 * bz.um, 1.5 * bz.um),
    width=wl / 6,
    signal=signal,
)

monitor = bz.FieldRecorder(components=("Ez",), interval=10, name="field_frames")

simulation = bz.Simulation(
    design=design,
    sources=[source],
    monitors=[monitor],
    boundaries=[bz.PML(edges="all")],
    time=time,
    resolution=dx,
)
results = simulation.run()
ez_frames = results.monitor("field_frames").fields["Ez"]
```

`run()` is the normal execution method: it covers the complete time grid and
returns detached, immutable `SimulationResults`. Compilation happens lazily, so
there is no separate compilation step for ordinary use.

Use `advance()` only when you need the runtime state for chunking, checkpointing,
or branching:

```python
first = simulation.advance(num_steps=100)
second = simulation.advance(state=first.state, num_steps=100)

# first.results and second.results are durable analysis values.
# first.state remains reusable because continuation preserves inputs by default.
alternative = simulation.advance(state=first.state, num_steps=50)
```

For the lower-memory continuation path, `donate_state=True` explicitly transfers
the input buffers to JAX. Never access that input state after the call. `step()` is
the state-only, one-timestep primitive for debugging; it is not needed for normal
simulations.
