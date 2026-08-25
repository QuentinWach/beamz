# BeamZ CUDA component

CUDA is an optional native part of BeamZ, not a separately published package. This
directory builds the private component into the `beamz._cuda` module, which
registers typed JAX FFI targets while BeamZ remains usable with JAX alone. The
component's packaging metadata contains the `Private :: Do Not Upload` classifier,
so package indexes reject accidental publication.

The streamed backend replaces JAX's elementwise 3D Yee/CPML programs with fused
magnetic and electric CUDA launches. Bounded multi-step CUDA graphs own complete
3D runs, and one grouped program path covers arbitrary source batches and DFT
monitor schedules. JAX still owns tracing, buffers, and orchestration around that
small native interface, preserving BeamZ's public numerical semantics and JAX
fallback without duplicating configuration-specific FFI targets.

The native boundary has three typed FFI targets: one phase update, one complete
multi-step program, and the explicit Hopper experiment. The implementation is
split by responsibility:

- `ffi_handler.cc` decodes JAX buffers into a validated `BeamzProgramLaunch`;
- `program.cu` owns timestep order and selects the in-place or temporal schedule;
- `graph.cu` owns bounded capture, semantic cache keys, and replay;
- `update.cu` owns Yee and CPML kernels;
- `io.cu` owns source injection and DFT monitor accumulation.

For monitor-heavy graphs, `io.cu` first prepares one window and complex phase per
monitor/frequency into small XLA-owned scratch buffers, then reuses those values
across every gathered point and field component. Short plans retain the original
single-kernel path so an extra launch cannot dominate their work.

`abi_layout.json` is the source of truth for target names, layout selectors, and
positional buffer constants. After editing it, regenerate both language bindings
with `python scripts/generate_cuda_abi.py`; CI uses `--check` to reject drift.

On SM90, the experimental `beamz_cuda_hopper` target uses the same ABI and arithmetic
but maps each component to `32 × 4 × 2` spatial tiles. Each derivative input stages
only its directional halo in shared memory, reusing values across neighboring
updates while keeping the x direction warp-contiguous. Backend selection only
exposes this target on compute capability 9.0 or newer. It remains explicit-only
until hardware parity and throughput gates justify promotion; `auto` and generic
`cuda` use the streamed path.

Install BeamZ from this checkout, then build its optional component in a CUDA 12
development environment:

```console
python -m pip wheel ./cuda --no-deps
python -m pip install cuda/dist/beamz_cuda_component-*.whl
```

Release builds use precise CUDA division and square-root behavior. Approximate CUDA
intrinsics are available only for controlled experiments with
`-DBEAMZ_CUDA_FAST_MATH=ON`; they must pass the same hardware parity suite before a
result can be used for promotion.

The wheel compiles SASS for SM80, SM86, SM89, and SM90. `backend="auto"` detects and
registers it lazily; `backend="cuda_streamed"` requests it explicitly and
`backend="cuda_hopper"` requests the tiled target. The first
release supports one GPU and float32 3D grids. Multi-GPU and 2D simulations retain
the JAX backend; only the explicitly selected CPML recurrence state may use BF16.

BeamZ validates the component's explicit ABI version and complete streamed-target
manifest before registering any FFI handler. An older or partial component makes
`auto` fall back to JAX and causes explicit CUDA requests to fail with a
compatibility diagnostic.

Regular-grid, lossless CPML simulations with packed source groups use two
XLA-owned field banks. Alternating frozen inputs and outputs removes in-place
read/write hazards and enables safe spatial fusion in the CPML-free core. The
validated temporal, packed-source, material-codebook, and combined CPML queues
are unconditional; the legacy experimental schedule switches and their dead
kernels have been removed.

For memory-constrained CPML runs,
`BEAMZ_CUDA_CPML_PSI_PRECISION=bf16` stores only absorber recurrence state in
BF16 while fields and recurrence arithmetic remain FP32. The default is `fp32`;
use BF16 only after validating application-level accuracy. On GA102 the combined
queue uses a precision-specific `32 × 4` absorber tile and remaps the same 128
threads to a `64 × 2` recurrence-free core tile; FP32 retains its measured-optimal
`64 × 4` queue.

`BEAMZ_CUDA_DISABLE_GRAPH_CACHE=1` remains as a diagnostic switch.
`BEAMZ_CUDA_GRAPH_CACHE_CAPACITY` sets the completed-entry LRU target (default
`32`, valid range `0`–`4096`); a capacity of zero disables persistent graph
executables. In-flight entries may temporarily exceed the target so the native
cache never destroys an executable still referenced by a CUDA stream. These
choices and the CPML precision setting are snapshotted into the immutable
compiled-program key; native execution never rereads environment variables.

No CUDA result is promoted without all of the following on real hardware:

- compile with the oldest supported CUDA toolkit and import beside supported JAX;
- compare fields, CPML recurrence buffers, monitor accumulators, clocks, and
  continuation state with JAX over bare, lossy, PEC, CPML, source, and DFT cases;
- run Compute Sanitizer memcheck and racecheck;
- capture Nsight Compute memory throughput and the canonical H100 GCUPS records.

The CUDA workflow always compiles the private component and imports it through the
BeamZ namespace in a CUDA development container. It is retained only as a CI
artifact and is never published independently. Repositories with an H100
self-hosted runner can set the Actions variable
`BEAMZ_H100_RUNNER_ENABLED=true` and label that runner `h100` to additionally run the
32-step PEC/CPML parity envelope and publish canonical benchmark JSON artifacts.

The host FFI decoder deliberately has no CUDA-header dependency and can be checked
on developer machines with the JAX headers alone:

```console
clang++ -std=c++17 \
  -I"$(python -c 'import jax; print(jax.ffi.include_dir())')" -Icuda/src \
  -fsyntax-only cuda/src/ffi_handler.cc
```
