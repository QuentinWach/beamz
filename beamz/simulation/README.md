# Simulation architecture

BEAMZ keeps configuration, execution, and analysis separate:

1. `Simulation` is an immutable user-facing specification.
2. `Simulation.to_request()` resolves it into an immutable, data-only request.
3. The compiled planner turns that request into immutable JAX plans.
4. `SimulationState` owns all evolving fields, clocks, boundary buffers, and
   monitor accumulators.
5. `SimulationRun` separates continuation state from immutable `SimulationResults`.

Configuration changes use `updated_copy(...)` and always produce a new value:

```python
changed = sim.updated_copy(time=new_time, sources=new_sources)
results = changed.run()
```

The `Simulation` itself never owns fields, an engine, a current step, monitor
buffers, or an executable cache.

## Public API

The supported `beamz.simulation` surface is intentionally small:

- `Simulation`, `SimulationState`, `SimulationRun`, `SimulationResults`, `MonitorResults`
- `AutoTermination`, `RunTermination`
- `GridSpec`, `GaussianPulse`, `ModeSpec`
- `Absorber`, `PML`, `PEC`, `PMC`, `Port`

Numerical Yee helpers, mutable mesh builders, source/monitor lowering, update
kernels, and compiled plan types remain private implementation details rather
than forming a second public solver API.

Mirror-symmetric problems can use `Simulation(symmetry=(sx, sy, sz))`, where
each physical-axis entry is `0` (disabled), `1` (even electric-field parity), or
`-1` (odd electric-field parity). See the
[mirror-symmetry guide](../../docs/mirror-symmetry.md) for the workflow and
validation rules.

## Choosing an execution method

| Method | Intended use | Return value | Input-state ownership |
| --- | --- | --- | --- |
| `run()` | Normal complete or convergence-bounded simulation | `SimulationResults` | State is internal and safely donated |
| `advance()` | Chunking, continuation, checkpointing, or branching | `SimulationRun(results, state)` | Preserved by default; donation is explicit |
| `step()` | Single-timestep debugging and numerical verification | `SimulationState` | Preserved by default; donation is explicit |

All three methods use the same lazily compiled JAX engine. `compile()` is optional
and intended for advanced inspection or prewarming; there is no separate
"uncompiled" execution path.

Pass `AutoTermination` to `run(termination=...)` when the configured time grid
should be a maximum rather than a mandatory duration. Execution reuses a fixed
chunk program, waits for all sources to become inactive, and then requires the
configured energy and frequency-monitor residuals to pass for consecutive checks.
`SimulationResults.termination` contains the executed step count, stop reason,
and final diagnostics. It remains `None` for an ordinary full-grid run.

## Ownership

- `api.py`: immutable public `Simulation` specification and request creation.
- `model.py`: request, compiled-program, lattice, plan, and runtime-state values;
  compiled coefficients, sources, and monitors are stored directly without wrapper plans.
- `compile.py`: lowers a resolved request into an executable plan.
  Deterministic memory reporting lives here because it inspects that plan.
- `kernels.py`: all canonical 2D/3D Yee, material-loss, PEC, and packed CPML
  mathematics. PML, sponge, and PEC profile lowering lives with the boundary
  specifications in `beamz.devices._boundary_compile`.
- `sharding.py`: optional multi-device lowering, padding, placement, and cropping.
- `observe.py`: monitor accumulation plus the numerical interpretation and
  source normalization of those acquisitions.
- `results.py`: immutable execution-owned run outputs, decoding, and retained
  material context.
- `execute.py`: JAX execution, caching, continuation, and result assembly.

Memory reporting and observation stay with the compiled plan and monitor lifecycle
instead of adding one-function orchestration modules.

Shared Yee geometry is not owned here. `beamz.lattice` defines component
offsets, shapes, coordinates, public/canonical transforms, and material-grid
sampling. Boundary specifications and metallic-mask compilation live in
`beamz.devices.boundaries`. Geometry rasterization remains in `beamz.design`.

Public source, monitor, and boundary objects are canonical immutable Device specs.
Compilation consumes them directly and produces grid-aware internal plans;
there is no public adapter or lowering registry.

## Caches

Compiled programs, discretizations, and JIT execution artifacts live outside
immutable specs and plans in bounded private caches.
`Simulation.clear_compiled_cache()` is the supported explicit clear operation.
Cache identity is derived from
canonical immutable configuration tokens, never live object identity.

## Output policy

Library status and timing information are emitted through standard logging, never
stdout. Terminal progress is opt-in: `Simulation.run(progress=False)`,
`Simulation.advance(progress=False)`, and `Design.rasterize()` are silent, while
`progress=True` shows mesh and execution progress.
Physics-changing fallbacks raise an exception or a Python warning rather than
printing a message.

## Results and analysis

Results deep-freeze mappings and NumPy arrays. By default they retain only the
material regions required by configured analysis monitors. A full material
snapshot is opt-in with `store_full_materials=True`.

Named acquisitions are explicit and remain notebook-friendly:

```python
frames = results.monitors["line_fields"].fields["Ez"]
power = results.monitors["output"].power_history
flux = results.monitors["output"].flux
```

Frequency-domain acquisitions are normalized to source `0` by default. Select
another source with `Simulation(normalize_source=1)`, disable normalization with
`normalize_source=None`, or create another immutable view after a run:

```python
raw = results.renormalize(None)
source_1 = results.renormalize(1)
```

Execution owns raw monitor decoding and source normalization. Flux interpretation,
modal projection, S-parameters, labeled data, and plotting live in
`beamz.analysis`. Optional result and simulation convenience methods resolve those
adapters lazily, so execution does not import analysis or retain a live simulation.

For example:

```python
from beamz.analysis import s_parameters

results = sim.run()
s = s_parameters(results, source_port="o1", ports=ports)
```

Continuation preserves its input state by default, so branching and retries are safe:

```python
first = sim.advance(num_steps=100)
branch_a = sim.advance(num_steps=50, state=first.state)
branch_b = sim.advance(num_steps=50, state=first.state)
```

When the old state is no longer needed, explicitly transfer ownership to JAX to
reduce peak device memory:

```python
next_run = sim.advance(
    num_steps=50,
    state=first.state,
    donate_state=True,
)
# Do not use first.state after this call; its device buffers may have been recycled.
```

`Simulation.step()` remains available as an advanced state-only primitive for
single-timestep debugging and numerical verification; normal simulations should use
`run()`, and chunked simulations should use `advance()`.

## Dependency direction

Keep dependencies flowing toward orchestration:

`Design rasterization + Device specs -> lattice lowering -> compile -> kernels/observe -> execute -> results <- analysis`

Analysis may consume specs and results. A static architecture contract prevents
the simulation API, execution, result, monitor-result, normalization, and memory
modules from importing analysis behavior. Lazy plotting and labeled-data convenience
methods resolve analysis functions only when called.
