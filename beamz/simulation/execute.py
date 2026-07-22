"""Scan execution, caching, continuation, and runtime placement."""

from __future__ import annotations

import os
import pathlib
import platform
import sys
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from beamz._helpers import _finish_inline_progress, _print_inline_progress
from beamz.devices.sources.compiler import (
    BatchedSlabGroup,
    CompiledSourceSpec,
    batch_slab_specs,
)
from beamz.simulation.model import (
    CompiledProgram,
    SimulationState,
    UpdateCoefficients,
)

from . import kernels as update_runtime
from . import observe as monitor_runtime
from . import sharding as sharding_runtime
from .results import MonitorResults, SimulationResults, SimulationRun

SOURCE_PHASE_COMPONENTS = {
    "pre_e": ("Ex", "Ey", "Ez"),
    "h": ("Hx", "Hy", "Hz"),
    "e": ("Ex", "Ey", "Ez"),
}

# Partition sources by leapfrog phase: magnetic and electric injections must sit next to
# the update whose temporal staggering was used when their amplitudes were normalized.
SourceBatchMap = dict[
    tuple[str, str],
    tuple[BatchedSlabGroup | None, tuple[CompiledSourceSpec, ...]],
]


def compiled_source_batches(
    source_specs: tuple[CompiledSourceSpec, ...],
) -> SourceBatchMap:
    # Apply sources in their scheduled leapfrog phase so amplitude normalization
    # matches field time.
    return {
        (timing, component): batch_slab_specs(
            tuple(
                spec
                for spec in source_specs
                if spec.timing == timing and spec.component == component
            )
        )
        for timing, components in SOURCE_PHASE_COMPONENTS.items()
        for component in components
    }


def _apply_specs(
    arr: jnp.ndarray,
    abs_step: jnp.ndarray,
    specs: tuple[CompiledSourceSpec, ...],
) -> jnp.ndarray:
    # Apply irregular specs in declaration order because overlapping injections add.
    out = arr
    for spec in specs:
        safe_idx = jnp.clip(abs_step, 0, spec.waveform.shape[0] - 1)
        amp = spec.waveform[safe_idx]
        patch = (spec.coeff * amp).astype(out.dtype)
        if (
            spec.is_slab
            and spec.slab_starts is not None
            and spec.slab_sizes is not None
        ):
            cur = jax.lax.dynamic_slice(out, spec.slab_starts, spec.slab_sizes)
            out = jax.lax.dynamic_update_slice(out, cur + patch, spec.slab_starts)
        else:
            out = out.at[spec.index].add(patch)
    return out


def _apply_batched_slabs(
    arr: jnp.ndarray,
    abs_step: jnp.ndarray,
    group: BatchedSlabGroup,
    *,
    dense_single_slab: bool,
) -> jnp.ndarray:
    # Batch equal-shaped slabs so source count does not linearly grow generated code.
    safe_idx = jnp.clip(abs_step, 0, group.waveforms.shape[1] - 1)
    ndim = len(group.max_sizes)

    if group.n == 1:
        amp = group.waveforms[0, safe_idx]
        starts_0 = group.starts_tuple[0]
        if dense_single_slab:
            pad_width = tuple(
                (
                    starts_0[d],
                    int(arr.shape[d]) - starts_0[d] - group.max_sizes[d],
                )
                for d in range(ndim)
            )
            dense_coeff = jnp.pad(group.coeffs[0], pad_width)
            return arr + (dense_coeff * amp).astype(arr.dtype)
        patch = (group.coeffs[0] * amp).astype(arr.dtype)
        cur = jax.lax.dynamic_slice(arr, starts_0, group.max_sizes)
        return jax.lax.dynamic_update_slice(arr, cur + patch, starts_0)

    def body(i, out):
        # Carry prior additions so overlapping slabs accumulate rather than overwrite.
        amp = group.waveforms[i, safe_idx]
        patch = (group.coeffs[i] * amp).astype(out.dtype)
        starts_i = [group.starts[i, d] for d in range(ndim)]
        cur = jax.lax.dynamic_slice(out, starts_i, group.max_sizes)
        return jax.lax.dynamic_update_slice(out, cur + patch, starts_i)

    return jax.lax.fori_loop(0, group.n, body, arr)


def apply_source_phase(
    eng: SimulationState,
    abs_step: jnp.ndarray,
    batches: SourceBatchMap,
    timing: str,
    *,
    dense_single_slab: bool,
) -> SimulationState:
    # Apply sources in their scheduled leapfrog phase so amplitude normalization
    # matches field time.
    updates = {}
    for component in SOURCE_PHASE_COMPONENTS[timing]:
        field_name = component.lower()
        value = getattr(eng, field_name)
        batch, rest = batches[(timing, component)]
        if batch is not None:
            value = _apply_batched_slabs(
                value,
                abs_step,
                batch,
                dense_single_slab=dense_single_slab,
            )
        if rest:
            value = _apply_specs(value, abs_step, rest)
        updates[field_name] = value.astype(getattr(eng, field_name).dtype)
    return eng._replace(**updates)


ScanResult = SimulationState


def forward_step(
    carry,
    *,
    ctx: update_runtime.CompiledStepContext,
    coeffs: UpdateCoefficients,
    program,
    update_kernel: update_runtime.StepUpdateKernel,
):
    """Advance one compiled timestep."""
    cfg = ctx.config
    metallic = ctx.boundary.metallic

    # 1. Pre-E sources precede H because their waveform is normalized at this leapfrog
    # phase; the selected kernel then advances the canonical magnetic fields.
    state = apply_source_phase(
        carry,
        carry.current_step,
        ctx.source_batches,
        "pre_e",
        dense_single_slab=cfg.source_single_slab_dense,
    )
    state = update_kernel.update_h(state, ctx, coeffs)

    # 2. H-phase sources may overwrite constrained cells, so reapply the compiled masks
    # before E consumes the half-step magnetic fields.
    state = apply_source_phase(
        state,
        state.current_step,
        ctx.source_batches,
        "h",
        dense_single_slab=cfg.source_single_slab_dense,
    )
    hx, hy, hz = update_runtime.apply_post_source_boundaries(
        (state.hx, state.hy, state.hz),
        (metallic.hx_mask, metallic.hy_mask, metallic.hz_mask),
    )
    state = state._replace(hx=hx, hy=hy, hz=hz)

    # 3. Advance E, inject its sources, and restore its masks before observation.
    state = update_kernel.update_e(state, ctx, coeffs)
    state = apply_source_phase(
        state,
        state.current_step,
        ctx.source_batches,
        "e",
        dense_single_slab=cfg.source_single_slab_dense,
    )
    ex, ey, ez = update_runtime.apply_post_source_boundaries(
        (state.ex, state.ey, state.ez),
        (metallic.ex_mask, metallic.ey_mask, metallic.ez_mask),
    )
    state = state._replace(ex=ex, ey=ey, ez=ez)

    # 4. Observe only fully constrained end-of-step fields, then advance both clocks.
    t_phys = state.t + ctx.dt_scalar
    state = monitor_runtime.update_monitors(
        program,
        state,
        state.current_step,
        t_phys,
        ctx.dt_scalar,
        state.ex,
        state.ey,
        state.ez,
        state.hx,
        state.hy,
        state.hz,
    )
    return state._replace(
        t=t_phys,
        current_step=state.current_step + jnp.array(1, dtype=jnp.int32),
    )


def build_scan(program, *, donate_state: bool = False):
    """Build the jitted compiled scan for a program."""

    # 1. Pull immutable configuration out of the program before tracing. These values
    # select shapes and kernels, so they should remain static for executable reuse.
    # TODO(adjoint-checkpointing): Once trainable material arrays are dynamic
    # runtime inputs, expose chunked scan transitions here and rematerialize at
    # chunk boundaries. Keep parameter *values* out of the compiled-program key
    # so optimization iterations reuse one executable.

    cfg = program.config
    boundary = program.boundary
    resolution = float(cfg.resolution)
    dt = float(cfg.dt)
    dt_scalar = jnp.asarray(dt, dtype=jnp.float32)
    is_3d = cfg.is_3d

    # 2. Batch sources once; monitors are already canonical executable plans.
    source_batches = compiled_source_batches(program.sources)

    # 3. Assemble the shared step context and select the specialized update kernel before
    # JIT compilation begins.
    step_context = update_runtime.CompiledStepContext(
        config=cfg,
        boundary=boundary,
        source_batches=source_batches,
        resolution=resolution,
        dt=dt,
        dt_scalar=dt_scalar,
        is_3d=is_3d,
    )
    update_kernel = update_runtime.select_update_kernel(step_context)

    def run_scan(
        state: SimulationState,
        coeffs: UpdateCoefficients,
    ):
        # 4. Run the same transition through scan or fori_loop. The choice changes the
        # lowering strategy, not timestep semantics.
        if cfg.loop_kind == "scan":

            def _scan_body(carry, _unused):
                # Emit no per-step output because final state and explicit buffers hold results.
                return (
                    forward_step(
                        carry,
                        ctx=step_context,
                        coeffs=coeffs,
                        program=program,
                        update_kernel=update_kernel,
                    ),
                    None,
                )

            scan_out, _ = jax.lax.scan(
                _scan_body,
                state,
                xs=None,
                length=cfg.num_steps,
            )
        else:
            scan_out = jax.lax.fori_loop(
                0,
                cfg.num_steps,
                lambda _i, c: forward_step(
                    c,
                    ctx=step_context,
                    coeffs=coeffs,
                    program=program,
                    update_kernel=update_kernel,
                ),
                state,
            )
        return scan_out

    # 8. Buffer donation is an explicit ownership transfer. The default executable
    # preserves its input state; callers may opt into the lower-memory variant when
    # they no longer need that continuation value.
    donate_argnums = (0,) if donate_state else ()
    return jax.jit(run_scan, donate_argnums=donate_argnums)


# Runtime owns executable caching and device placement; immutable plans own numerical
# meaning. Separating those lifetimes makes continuation and cache reuse safe.


def _init_persistent_cache():
    # 1. Honor explicit opt-in and opt-out environment controls before changing JAX's
    # process-wide compilation-cache configuration.
    if os.environ.get("BEAMZ_ENABLE_JAX_PERSISTENT_CACHE", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    if os.environ.get("BEAMZ_DISABLE_JAX_PERSISTENT_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    # 2. Derive a cache namespace from JAX version, backend, architecture, and Python ABI
    # so incompatible executables are never reused.
    py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    backend = jax.default_backend()
    arch = platform.machine() or "unknown"
    cache_dir = os.environ.get(
        "BEAMZ_JAX_CACHE_DIR",
        str(
            pathlib.Path.home()
            / ".cache"
            / "beamz"
            / "jax_cache"
            / f"jax-{jax.__version__}"
            / backend
            / arch
            / py_tag
        ),
    )
    # 3. Preserve a user-provided JAX cache directory; otherwise configure the derived
    # Beamz location through the newest available JAX API with a compatibility fallback.
    if os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        return
    try:
        from jax.experimental.compilation_cache import compilation_cache as cc

        cc.set_cache_dir(cache_dir)
    except Exception:
        jax.config.update("jax_compilation_cache_dir", cache_dir)


@dataclass(slots=True)
class _ExecutionCache:
    # Cache lowered/compiled functions per runtime signature. Keeping it outside immutable
    # plan equality prevents compilation artifacts from changing semantic identity.
    compiled_scan: Callable[..., ScanResult] | None = None
    compiled_scan_donating: Callable[..., ScanResult] | None = None


_MAX_EXECUTION_CACHES = 8
_EXECUTION_CACHES: OrderedDict[int, tuple[object, _ExecutionCache]] = OrderedDict()


def execution_cache(program) -> _ExecutionCache:
    """Return bounded private executable state for an immutable plan."""
    # Key cached work by semantic execution inputs so equivalent states reuse
    # compilation safely.
    key = id(program)
    entry = _EXECUTION_CACHES.get(key)
    if entry is not None and entry[0] is program:
        _EXECUTION_CACHES.move_to_end(key)
        return entry[1]
    cache = _ExecutionCache()
    _EXECUTION_CACHES[key] = (program, cache)
    if len(_EXECUTION_CACHES) > _MAX_EXECUTION_CACHES:
        _EXECUTION_CACHES.popitem(last=False)
    return cache


def clear_execution_cache() -> None:
    # Key cached work by semantic execution inputs so equivalent states reuse
    # compilation safely.
    _EXECUTION_CACHES.clear()


def initial_program_state(
    program: CompiledProgram,
    *,
    t: float,
    current_step: int,
    continuation: SimulationState | None = None,
    monitor_steps: int | None = None,
) -> SimulationState:
    """Allocate or restore every runtime buffer required by a compiled plan."""
    cpml, layout = program.boundary.cpml, program.sharding.layout

    def field(name):
        # Fresh runs use the compiled lattice; continuations supply evolved canonical
        # arrays without reconstructing a mutable field container.
        return (
            jnp.array(getattr(program.grid, name))
            if continuation is None
            else getattr(continuation, name.lower())
        )

    def zeros(shape, dtype):
        shape = tuple(int(value) for value in shape)
        return (
            np.zeros(shape, dtype=np.dtype(dtype))
            if layout.enabled
            else jnp.zeros(shape, dtype=dtype)
        )

    # Continue compatible packed CPML memories; a changed boundary plan starts clean.
    def restore_psi(old, terms, dtype):
        shapes = tuple(term.slab.shape for term in terms)
        if len(old) == len(shapes) and all(
            tuple(value.shape) == shape
            for value, shape in zip(old, shapes, strict=True)
        ):
            return old
        return tuple(zeros(shape, dtype) for shape in shapes)

    old_h = () if continuation is None else continuation.cpml_psi_h_terms
    old_e = () if continuation is None else continuation.cpml_psi_e_terms
    monitor_values = monitor_runtime.empty_monitor_values(
        program,
        num_steps=max(
            1,
            int(program.config.num_steps if monitor_steps is None else monitor_steps),
        ),
    )
    recorder_buffers = monitor_values["recorded_fields"]
    recorder_compatible = (
        continuation is not None
        and len(continuation.recorded_fields) == len(recorder_buffers)
        and all(
            old.shape[1:] == new.shape[1:]
            for old, new in zip(
                continuation.recorded_fields, recorder_buffers, strict=True
            )
        )
    )
    # Reuse monitor history only when its compiled shapes still match this program.
    if (
        program.monitors
        and continuation is not None
        and continuation.counts.shape == monitor_values["counts"].shape
        and recorder_compatible
    ):
        monitor_values = {
            name: getattr(continuation, name) for name in monitor_runtime.MONITOR_FIELDS
        }
    return SimulationState(
        ex=field("Ex"),
        ey=field("Ey"),
        ez=field("Ez"),
        hx=field("Hx"),
        hy=field("Hy"),
        hz=field("Hz"),
        cpml_psi_h_terms=restore_psi(old_h, cpml.h_terms, field("Hx").dtype),
        cpml_psi_e_terms=restore_psi(old_e, cpml.e_terms, field("Ez").dtype),
        **monitor_values,
        t=jnp.asarray(t, dtype=jnp.float32),
        current_step=jnp.asarray(current_step, dtype=jnp.int32),
    )


def build_program_scan(program: CompiledProgram, *, donate_state: bool = False):
    """Build and cache the JIT scan only when execution first needs it."""
    cache = execution_cache(program)
    scan = build_scan(program, donate_state=bool(donate_state))
    if donate_state:
        cache.compiled_scan_donating = scan
    else:
        cache.compiled_scan = scan
    return scan


def program_is_compiled(
    program: CompiledProgram, *, donate_state: bool = False
) -> bool:
    """Return whether the requested ownership variant is already cached."""
    cache = execution_cache(program)
    return (
        cache.compiled_scan_donating if donate_state else cache.compiled_scan
    ) is not None


def run_program(
    program: CompiledProgram,
    state: SimulationState,
    *,
    donate_state: bool = False,
) -> SimulationState:
    """Place state and coefficients, then execute the program's cached scan."""
    state = sharding_runtime.prepare_state(
        program,
        state,
        replicated_fields=(*monitor_runtime.MONITOR_FIELDS, "t", "current_step"),
    )
    coeffs = sharding_runtime.place_tree(program, program.coefficients)
    cache = execution_cache(program)
    compiled_scan = (
        cache.compiled_scan_donating if donate_state else cache.compiled_scan
    ) or build_program_scan(program, donate_state=donate_state)
    return compiled_scan(state, coeffs)


def step_program(
    program: CompiledProgram,
    state: SimulationState,
    *,
    donate_state: bool = False,
) -> SimulationState:
    """Execute a program compiled for exactly one canonical timestep."""
    if program.config.num_steps != 1:
        raise ValueError("CompiledProgram.step() requires num_steps=1.")
    return run_program(program, state, donate_state=donate_state)


def _decode_monitor_results(sim, program, state) -> dict[str, MonitorResults]:
    """Detach packed monitor rows into the user-visible named mapping."""
    return {
        str(
            getattr(monitor, "name", None) or f"monitor_{spec.monitor_index}"
        ): MonitorResults.from_compiled_state(monitor, spec, state, program.config)
        for spec in program.monitors
        for monitor in (sim.monitors[spec.monitor_index],)
    }


def _compiled_source_launch_powers(program, source_count: int):
    """Return a source power only when all of its compiled terms agree."""
    values: list[list[float]] = [[] for _ in range(int(source_count))]
    for spec in program.sources:
        source_index = int(getattr(spec, "source_index", -1))
        power = getattr(spec, "launched_power", None)
        if 0 <= source_index < len(values) and power is not None:
            values[source_index].append(float(power))
    return tuple(
        None
        if not powers
        or not all(
            abs(value - powers[0]) <= 1e-10 * max(1.0, abs(powers[0]))
            for value in powers
        )
        else powers[0]
        for powers in values
    )


def runtime_inputs(
    program: CompiledProgram,
    state: SimulationState,
    *,
    monitor_steps: int,
) -> SimulationState:
    """Restore continuation buffers required by an already compiled program."""
    return initial_program_state(
        program,
        t=state.t,
        current_step=state.current_step,
        continuation=state,
        monitor_steps=monitor_steps,
    )


def execute_step(
    program: CompiledProgram,
    state: SimulationState,
    *,
    monitor_steps: int,
    donate_state: bool = False,
) -> SimulationState:
    """Advance one explicit state through a one-step program."""
    return sharding_runtime.crop_state(
        program,
        step_program(
            program,
            runtime_inputs(program, state, monitor_steps=monitor_steps),
            donate_state=donate_state,
        ),
    )


def compiled_xla_memory_analysis(
    program: CompiledProgram,
    state: SimulationState,
    *,
    monitor_steps: int,
) -> dict:
    """Return backend memory data for the exact placed program and state."""
    state = sharding_runtime.prepare_state(
        program,
        runtime_inputs(program, state, monitor_steps=monitor_steps),
        replicated_fields=(*monitor_runtime.MONITOR_FIELDS, "t", "current_step"),
    )
    coeffs = sharding_runtime.place_tree(program, program.coefficients)
    cache = execution_cache(program)
    compiled_scan = cache.compiled_scan or build_program_scan(program)
    compiled = compiled_scan.lower(state, coeffs).compile()
    analysis = getattr(compiled, "memory_analysis", lambda: None)()
    if analysis is None:
        return {"available": False}
    return {
        "available": True,
        **{
            name: value
            for name in dir(analysis)
            if not name.startswith("_")
            and isinstance(
                (value := getattr(analysis, name)),
                (int, float, str, bool, type(None)),
            )
        },
    }


def run_simulation_program(
    simulation,
    program: CompiledProgram,
    state: SimulationState,
    *,
    progress: bool,
    store_full_materials: bool,
    monitor_steps: int,
    donate_state: bool,
) -> SimulationRun:
    """Execute one continuation and separate durable results from runtime state."""
    compiling = bool(
        progress and not program_is_compiled(program, donate_state=donate_state)
    )
    if compiling:
        _print_inline_progress(0, 1, label="JIT compilation", unit="programs")
    state = run_program(
        program,
        runtime_inputs(program, state, monitor_steps=monitor_steps),
        donate_state=donate_state,
    )
    state.ez.block_until_ready()
    state = sharding_runtime.crop_state(program, state)
    if progress:
        if compiling:
            _finish_inline_progress()
        _print_inline_progress(program.config.num_steps, program.config.num_steps)
        _finish_inline_progress()
    results = SimulationResults.from_run(
        simulation,
        runtime_fields=program.grid,
        monitor_results=_decode_monitor_results(simulation, program, state),
        store_full_materials=store_full_materials,
        source_launch_powers=_compiled_source_launch_powers(
            program, len(simulation.sources)
        ),
    )
    return SimulationRun(results=results, state=state)


_init_persistent_cache()
