# Compiled FDTD Engine (v0.3)

BEAMZ v0.3 introduces a packed-data compiled simulation path:

- One compiled loop (`jax.lax.scan` by default) over the full timestep loop.
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
- Default compiled loop primitive is `scan` (`BEAMZ_COMPILED_LOOP_KIND=scan`).
- Lossy shell split is disabled by default (`BEAMZ_ENABLE_E_SHELL_SPLIT=0`, `BEAMZ_ENABLE_H_SHELL_SPLIT=0`).
- Default source slab injection uses `dynamic_update_slice` (`BEAMZ_SOURCE_SINGLE_SLAB_DENSE=0`).
- Monitor accumulation stores compressed outputs (`power_history`) by default.
- The legacy split-kernel path (`run_fast` / `run_jit_scan`) now delegates to `run_compiled`.

## Performance debugging workflow

Use the benchmark runner to export IR and profile artifacts:

```bash
python benchmarks/compiled_engine_benchmark.py \
  --grid-n 352 \
  --steps 120 \
  --modes split_jit,compiled \
  --compiled-loop-kind scan \
  --no-enable-e-shell-split \
  --no-enable-h-shell-split \
  --dump-ir-dir benchmarks/results/ir/latest \
  --profile-dir benchmarks/results/trace/latest \
  --hlo-stats \
  --hlo-diagnostics \
  --csv benchmarks/results/compiled_3d_results.csv
```

Run source-isolation sweeps on a fixed 3D problem:

```bash
python benchmarks/compiled_engine_benchmark.py \
  --scenario fdtdx_coupler \
  --source-sweep \
  --domain-um 6,4,1.5 \
  --resolution-nm 25 \
  --courant-factor 0.99 \
  --sim-time-fs 200 \
  --modes split_jit,compiled \
  --csv benchmarks/results/compiled_3d_results.csv
```

This writes:

- `compiled_jaxpr.txt`
- `compiled_hlo_unoptimized.txt`
- `compiled_hlo_unoptimized.dot`
- `compiled_hlo_optimized.txt`
- `compiled_hlo_stats.json`

Use this to prioritize optimization work:

- Reduce `scatter` and `dynamic-update-slice`.
- Lower total `slice` count in the optimized HLO.
- Keep the compiled timestep path to one dominant fused kernel family.

`BEAMZ_SOURCE_SINGLE_SLAB_DENSE=1` removes `dynamic-update-slice` for single-slab sources,
but may reduce throughput on memory-bound CPU runs. Use it for IR experiments, not as the
default performance mode.

## Current scope

v0.3 first-class compiled source support:

- `GaussianSource`
- `ModeSource`

v0.3 first-class compiled monitor support:

- `Monitor` line/plane power accumulation

Thermal-coupled compiled stepping remains out-of-scope for this initial cut.
