# Compiled FDTD Engine (v0.3)

BEAMZ v0.3 introduces a packed-data compiled simulation path:

- One compiled `jax.lax.scan` step over the full timestep loop.
- Sources compiled to static injection specs (`CompiledSourceSpec`).
- Monitors compiled to static accumulation specs (`CompiledMonitorSpec`).
- Material updates routed through compiled model interfaces (`CompiledMaterialSpec`).

## New APIs

- `beamz.simulation.compile_simulation(design, devices, boundaries, run_cfg)`
- `Simulation.compile(num_steps=...)`
- `Simulation.run_compiled(num_steps=..., record_interval=..., record_fields=...)`

## Runtime state types

- `EngineState`
- `MonitorState`
- `MaterialState`
- `RunState`

## Default behavior

- Default precision is `float32`.
- Monitor accumulation stores compressed outputs (`power_history`) by default.
- The legacy split-kernel path (`run_fast` / `run_jit_scan`) now delegates to `run_compiled`.

## Current scope

v0.3 first-class compiled source support:

- `GaussianSource`
- `ModeSource`

v0.3 first-class compiled monitor support:

- `Monitor` line/plane power accumulation

Thermal-coupled compiled stepping remains out-of-scope for this initial cut.
