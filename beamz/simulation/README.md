# Simulation

Modules that orchestrate how the EM fields evolve depending on the design and devices.

+ core.py       / Main module to orchestrate the simulation.
+ fields.py     / Contains the Field class which owns the field data and defines the field update.
+ ops.py        / Contains the operations used by the field updates.
+ compiled.py   / v0.3 packed-data compiled engine (`run_compiled` / `compile_simulation`).

## Compiled Multi-Device Runs

`Simulation.run_compiled(..., sharding=ShardingConfig(...))` enables single-host
JAX sharding for 3D compiled simulations. The compiled engine keeps the same six
Yee component arrays and pads only the selected physical storage axis so every
component can be evenly split across devices. Full-PEC 3D runs use the same
execution path with active high-side Yee planes in storage. Public field arrays
and snapshots are cropped back to logical simulation shapes.

CPU-only structural testing can use fake host devices by setting this before the
Python process imports JAX:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4 pytest tests/test_compiled_sharding.py
```

For GPU performance runs, prefer validating JAX/NCCL peer-to-peer visibility
first, then run with explicit `num_devices`. Useful allocator settings are:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false
# Optional for memory debugging, usually slower:
XLA_PYTHON_CLIENT_ALLOCATOR=platform
```
