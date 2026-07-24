# Plan: Next Steps!

Yes: [problems.py](/Users/quentinwach/Code/beamz-1/beamz/optimization/problems.py) should be removed, and BeamZ should have exactly one public execution path: `Simulation.run()`.

The right model is:

```text
parameters → ordinary Simulation.run() → ordinary results → scalar objective
                         ↑
                 differentiated by JAX
```

No `InverseDesignProblem`, no `DifferentiableSimulation`, and no optimization runner inside BeamZ.

## What Tidy3D does

Modern Tidy3D deliberately removed its old `JaxSimulation` family. Users now build an ordinary `td.Simulation`, call the regular `web.run()`, calculate a scalar from the returned simulation data, and apply `value_and_grad()` to the entire function. Its custom adjoint implementation is hidden behind the standard run operation. [Tidy3D’s official documentation](https://docs.flexcompute.com/projects/tidy3d/en/v2.12.0.dev0/api/plugins/autograd.html) explicitly presents:

```text
parameters → Simulation → web.run → SimulationData → scalar
```

Tidy3D still provides optional inverse-design utilities, but these are general transforms:

- Filters
- Projections
- Symmetries
- Penalties
- Parameterizations

They are separate from simulation execution. Its normal run function internally detects traced inputs and invokes the adjoint VJP; untraced simulations follow the ordinary path. [Tidy3D implementation](https://github.com/flexcompute/tidy3d/blob/567999f6debafceb74c2208603abe6010ba80cbc/tidy3d/web/api/autograd/autograd.py)

This is the most relevant model for BeamZ.

## What FDTDX does

FDTDX is more explicitly functional. An optimization objective performs:

```python
arrays, objects, info = fdtdx.apply_params(...)
state = fdtdx.run_fdtd(arrays, objects, config)
objective = detector_output(state)
```

The example then applies `jax.value_and_grad()` and updates the parameters with Optax. FDTDX does not define an inverse-design problem object or own the optimization loop. [Official Ceviche corner example](https://github.com/ymahlau/fdtdx/blob/f4e610c3908223abcfb6c79f491e98cd3d190294/examples/optimize_ceviche_corner.py)

Its solver selects ordinary, checkpointed, or reversible differentiation internally based on `GradientConfig`. That changes how the same simulation operation is differentiated, not the user-facing optimization model. [FDTDX execution implementation](https://github.com/ymahlau/fdtdx/blob/f4e610c3908223abcfb6c79f491e98cd3d190294/src/fdtdx/fdtd/wrapper.py)

BeamZ should use Tidy3D’s public surface and FDTDX’s JAX-native implementation philosophy.

## Proposed BeamZ API

A design should contain a general named material region:

```python
device = bz.MaterialGrid(
    name="device",
    lower=(x0, y0),
    upper=(x1, y1),
    materials=(oxide, silicon),
)

simulation = bz.Simulation(
    design=design + device,
    sources=[source],
    monitors=[input_port, output_port],
    ...
)
```

The run method accepts optional dynamic values:

```python
results = simulation.run(
    parameters={"device": density},
)
```

Without parameters, existing behavior remains unchanged:

```python
results = simulation.run()
```

An optimization is then ordinary JAX code:

```python
def objective(latent):
    density = bz.optimization.tanh_projection(
        bz.optimization.conic_filter(latent, radius=5),
        beta=beta,
    )

    results = simulation.run(parameters={"device": density})
    s21 = results["output"].mode_amplitude(direction="-", mode=0)
    return -jnp.mean(jnp.abs(s21) ** 2)


value_and_grad = jax.jit(jax.value_and_grad(objective))
value, gradient = value_and_grad(latent)
```

The optimizer belongs to the application:

```python
optimizer = optax.adam(1e-2)
```

BeamZ should not wrap Optax, manage epochs, save checkpoints, or define an optimization result type.

## Multiple sources and ports

The current `InverseDesignProblem` mainly manages multiple source simulations. Ordinary composition is sufficient:

```python
def objective(latent):
    parameters = {"device": parameterize(latent)}

    result_1 = simulation_port_1.run(parameters=parameters)
    result_2 = simulation_port_2.run(parameters=parameters)

    return splitter_loss(result_1, result_2)
```

JAX traces both calls as part of one objective. If batch execution becomes important later, it should be a general simulation batching capability—not an inverse-design problem abstraction.

## Results must remain differentiable

The largest internal change is not `Simulation.run()` itself; it is unifying the result types.

Currently:

- `Simulation.run()` returns detached `SimulationResults`.
- `DifferentiableSimulation.run()` returns `DifferentiableResult`.
- Analysis uses NumPy in several places.
- Differentiable modal projection requires `DifferentiablePortProjector`.

That duplication should disappear.

`SimulationResults` should become a JAX-compatible immutable PyTree whose numerical monitor arrays may be JAX arrays. Metadata remains static. Plotting, serialization, pandas, and xarray conversion may call `np.asarray()` only at their outer boundaries.

Then the same result supports both:

```python
results = simulation.run()             # ordinary use
gradient = jax.grad(objective)(params) # differentiated use
```

## Modal amplitudes should belong to monitors

`DifferentiablePortProjector` is compensating for modal projection occurring too late.

A `ModeMonitor` should compile its modal basis with the simulation and expose amplitudes directly:

```python
result["port2"].amplitudes
result["port2"].power
```

The modal basis is static because the port lies outside the design region. Its projection is simply JAX linear algebra applied to monitor fields. This makes it:

- Differentiable
- Reusable outside inverse design
- Available through normal results
- Consistent with mode-overlap detectors in FDTDX
- Independent of a “problem” object

The existing detached `beamz.analysis.s_parameters()` can remain as a convenience function, implemented using the same underlying amplitude calculation.

## Target package layout

```text
beamz/
├── design/
│   ├── structures.py       # MaterialGrid or equivalent
│   └── materials.py
├── devices/
│   └── monitors/
│       └── monitors.py     # ModeMonitor and compiled modal observable
├── simulation/
│   ├── api.py              # Simulation.run(parameters=...)
│   ├── compile.py          # Static compilation
│   ├── execute.py          # Dynamic material application and FDTD scan
│   └── results.py          # One JAX-compatible result representation
├── analysis/
│   ├── mode_projection.py
│   └── sparameters.py
└── optimization/
    ├── filters.py
    ├── projections.py
    ├── fabrication.py
    └── polygonize.py
```

Remove:

```text
beamz/optimization/problems.py
beamz/optimization/trainable.py
beamz/optimization/challenges/
```

The optimization package then contains only reusable mathematical tools. It knows nothing about ports, waveguide bends, optimization loops, output directories, or Ceviche.

## Internal execution design

`Simulation.run()` should internally distinguish only whether dynamic parameters are present:

```python
def run(self, *, parameters=None, ...):
    program = self.compile()
    dynamic_materials = program.materialize(parameters)
    state = execute(program, dynamic_materials)
    return SimulationResults.from_state(program, state)
```

The internal `execute()` helper is fine—it is implementation, not a competing public runner.

Initially, differentiability can use BeamZ’s existing rematerialized JAX scan. Later, checkpointed adjoints or reversible FDTD can replace its VJP without changing the API:

```python
jax.grad(lambda p: simulation.run(parameters=p)...)
```

That is the strongest lesson from both Tidy3D and FDTDX: memory strategy and adjoint implementation should be invisible behind the stable simulation operation.

## Refactoring sequence

1. Move all Ceviche implementations out of `beamz`.
2. Add a general named `MaterialGrid` structure.
3. Add `parameters=` to `Simulation.run()`.
4. Route the existing dynamic-permittivity code through that method.
5. Make monitor results JAX-compatible.
6. Integrate modal projection into `ModeMonitor`.
7. Rewrite one bend objective using direct `jax.value_and_grad()`.
8. Verify gradients against the current implementation and finite differences.
9. Migrate the splitter, converter, and WDM.
10. Delete `InverseDesignProblem`, `DifferentiableSimulation`, `DifferentiableResult`, and `PortSweepResult`.

The crucial quality rule is that we should not preserve the current abstractions under different filenames. Their responsibilities should be absorbed into general design, simulation, monitor, result, and analysis components. That produces a smaller API and leaves inverse design as normal composition of BeamZ primitives.