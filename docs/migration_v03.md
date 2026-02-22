# Migration Guide: v0.2 -> v0.3

## What changed

The v0.3 engine executes FDTD through one compiled scan loop. Source/monitor work is no longer expected to execute as Python callbacks per timestep in the fast path.

## Replace old runtime entrypoints

- Old: `sim.run_fast(...)`
- Old: `sim.run_jit_scan(...)`
- New: `sim.run_compiled(...)`

`run_fast` and `run_jit_scan` still exist, but now call `run_compiled`.

## New compile-time packing APIs

- Sources: `compile_source_specs(...)`
- Monitors: `compile_monitor_specs(...)`
- Program: `compile_simulation(...)`

## Adjoint field storage

For standard field-overlap adjoint workflows, use disk chunking with memmaps:

- `beamz.optimization.adjoint_memmap.ForwardFieldChunkWriter`
- `beamz.optimization.adjoint_memmap.ReverseFieldChunkReader`
- `beamz.optimization.adjoint_memmap.compute_overlap_gradient_memmap`

## Notes

- Compiled run currently requires thermal coupling to be disabled.
- Default monitor behavior focuses on compact power accumulation.
