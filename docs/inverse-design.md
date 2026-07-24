# Differentiable FDTD inverse design

BeamZ keeps geometry, sources, boundaries, and monitors static while passing a
rectangular material region to the compiled FDTD scan as a dynamic JAX array.
The same executable can therefore be reused and differentiated across design
iterations.

```python
import jax.numpy as jnp

from beamz.optimization import DesignRegion, DifferentiableSimulation

region = DesignRegion(
    lower=(1e-6, 1e-6),
    upper=(2e-6, 2e-6),
    eps_min=1.0,
    eps_max=12.25,
)
trainable = DifferentiableSimulation(simulation, region)
density = jnp.ones(trainable.variable_shape)

def objective(result):
    return -jnp.abs(result.flux("output")[0])

value, gradient = trainable.value_and_grad(density, objective)
```

`DifferentiableResult.field()` and `DifferentiableResult.flux()` return JAX
arrays suitable for objectives. `run_results()` converts a run back to an
ordinary immutable `SimulationResults` for plotting and serialization.

## Modal objectives

An `InverseDesignProblem` calibrates a fixed mode basis at ports outside the
trainable region. The eigensolution is intentionally excluded from the
computational graph, matching the Ceviche challenge convention.

```python
projector = problem.port_projector(density, source_port="port1")
trainable = problem.differentiable("port1")

def loss(result):
    s11 = projector.s_parameter(
        result, source_port="port1", output_port="port1"
    )
    s21 = projector.s_parameter(
        result, source_port="port1", output_port="port2"
    )
    return jnp.mean(jnp.abs(s11)) - jnp.mean(jnp.abs(s21))

loss_value, gradient = trainable.value_and_grad(density, loss)
```

One broadband time-domain run acquires every requested DFT frequency.
`PortSweepResult.frequencies` describes its S-parameters, while
`field_frequencies` independently describes any full-domain field samples.

## Strict-foundry examples

The original 1550 nm bend from the main branch and the newer strict-foundry
O-band bend are both retained under wavelength-specific names:

```bash
uv run python examples/optimization/ceviche_bend_1550nm.py
uv run python examples/optimization/ceviche_bend_o_band.py
```

They are accompanied by exactly three additional device examples:

```bash
uv run python examples/optimization/ceviche_beam_splitter_o_band.py
uv run python examples/optimization/ceviche_mode_converter_o_band.py
uv run python examples/optimization/ceviche_wdm_o_band.py
```

Each file is intentionally a small runnable entry point. Shared optimization,
checkpointing, fabrication, evaluation, and plotting code lives in
`beamz.optimization.challenges`.

All three examples:

- generate binary layouts that satisfy the solid and void brush rules;
- use the paper's convolutional straight-through estimator and Adam constants;
- condition the design boundary on fixed access waveguides and exterior oxide;
- optimize six samples across the two O-band wavelength windows;
- store periodic topology snapshots rather than every full latent array; and
- write a final spectrum, field visualization, NPZ data, and JSON provenance.

The WDM uses a 40 nm default grid because its 6.4 µm design region would contain
640 by 640 generator cells at 10 nm. Its output explicitly distinguishes values
published in the article from access-domain coordinates reconstructed from the
rendered figure. `--evaluate-checkpoint` evaluates a saved WDM topology without
taking a fake optimizer step.

Use `--help` on an example to see its resolution, runtime, fabrication, snapshot,
and warm-start controls.

## Current scope

The trainable material path supports 2D TM and the full 3D Yee lattice.
Full-domain optimization monitors, differentiable modal projection, and the
strict-foundry brush generator are currently 2D. The differentiable runner does
not yet shard one optimization trajectory across multiple devices.
