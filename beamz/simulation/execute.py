"""Scan execution, caching, continuation, and runtime placement."""

from __future__ import annotations

import os
import pathlib
import platform
import sys
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from beamz._helpers import _finish_inline_progress, _print_inline_progress
from beamz.const import EPS_0, MU_0
from beamz.devices.sources.compiler import (
    BatchedSlabGroup,
    CompiledSourceSpec,
    batch_slab_specs,
)
from beamz.simulation.model import (
    AutoTermination,
    CompiledProgram,
    SimulationState,
    UpdateCoefficients,
)

from . import kernels as update_runtime
from . import observe as monitor_runtime
from . import sharding as sharding_runtime
from .results import MonitorResults, RunTermination, SimulationResults, SimulationRun

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


_COMPONENT_OFFSETS_2D = {
    "Ex": (0.0, 0.5),
    "Ey": (0.5, 0.0),
    "Ez": (0.0, 0.0),
    "Hx": (0.5, 0.0),
    "Hy": (0.0, 0.5),
    "Hz": (0.5, 0.5),
}


def _axis_integration_weights(edges, offset: float, count: int) -> np.ndarray:
    """Return control-volume widths for one edge- or center-aligned Yee axis."""
    edges = np.asarray(edges, dtype=np.float64)
    widths = np.diff(edges)
    count = int(count)
    if offset == 0.5:
        if count != widths.size:
            raise ValueError("Center-aligned Yee support does not match the grid.")
        return widths
    if offset != 0.0 or count != edges.size:
        raise ValueError("Edge-aligned Yee support does not match the grid.")
    out = np.empty(count, dtype=np.float64)
    out[0] = 0.5 * widths[0]
    out[-1] = 0.5 * widths[-1]
    if count > 2:
        out[1:-1] = 0.5 * (widths[:-1] + widths[1:])
    return out


def _component_integration_weights(program: CompiledProgram, component: str):
    """Return physical integration weights on one native Yee support."""
    shape = tuple(int(value) for value in getattr(program.grid, component).shape)
    geometry = program.grid.geometry
    if program.config.is_3d:
        from beamz.lattice import component_axis_offsets_3d

        offsets = component_axis_offsets_3d(component)
        z = _axis_integration_weights(geometry.z_edges, offsets["z"], shape[0])
        y = _axis_integration_weights(geometry.y_edges, offsets["y"], shape[1])
        x = _axis_integration_weights(geometry.x_edges, offsets["x"], shape[2])
        return jnp.asarray(z[:, None, None] * y[None, :, None] * x[None, None, :])
    y_offset, x_offset = _COMPONENT_OFFSETS_2D[component]
    y = _axis_integration_weights(geometry.y_edges, y_offset, shape[0])
    x = _axis_integration_weights(geometry.x_edges, x_offset, shape[1])
    return jnp.asarray(y[:, None] * x[None, :])


def _energy_terms(program: CompiledProgram):
    """Precompute field, material, and measure names for energy diagnostics."""
    active = (
        {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
        if program.config.is_3d
        else (
            {"Ez", "Hx", "Hy"}
            if program.config.polarization_2d == "tm"
            else {"Ex", "Ey", "Hz"}
        )
    )
    extent = program.grid.geometry.extent
    domain_measure = float(
        extent[0] * extent[1] * (extent[2] if program.config.is_3d else 1.0)
    )
    terms = tuple(
        (
            component.lower(),
            getattr(program.grid, material),
            _component_integration_weights(program, component) / domain_measure,
            constant,
        )
        for component, material, constant in (
            ("Ex", "eps_ex", EPS_0),
            ("Ey", "eps_ey", EPS_0),
            ("Ez", "eps_ez", EPS_0),
            ("Hx", "mu_hx", MU_0),
            ("Hy", "mu_hy", MU_0),
            ("Hz", "mu_hz", MU_0),
        )
        if component in active
    )
    return terms, domain_measure


def _field_diagnostics(state: SimulationState, plan) -> tuple[float, float, bool]:
    """Compute integrated electromagnetic energy, maximum field, and finiteness."""
    terms, domain_measure = plan
    energy_density = jnp.asarray(0.0, dtype=jnp.float32)
    max_field = jnp.asarray(0.0, dtype=jnp.float32)
    finite = jnp.asarray(True)
    for field_name, material, measure, constant in terms:
        values = jnp.asarray(getattr(state, field_name))
        finite = finite & jnp.all(jnp.isfinite(values))
        max_field = jnp.maximum(max_field, jnp.max(jnp.abs(values), initial=0.0))
        density = jnp.asarray(material) * values * values
        energy_density = energy_density + 0.5 * float(constant) * jnp.sum(
            density * jnp.asarray(measure)
        )
    return float(energy_density) * domain_measure, float(max_field), bool(finite)


def _remaining_source_activity(
    program: CompiledProgram, total_steps: int
) -> np.ndarray:
    """Return the maximum future source amplitude relative to each term's peak."""
    total_steps = int(total_steps)
    activity = np.zeros(total_steps, dtype=np.float64)
    for source in program.sources:
        waveform = np.abs(np.asarray(source.waveform, dtype=np.float64).reshape(-1))
        if waveform.size == 0:
            continue
        peak = float(np.max(waveform, initial=0.0))
        if peak > 0.0:
            sampled = np.empty(total_steps, dtype=np.float64)
            copied = min(total_steps, waveform.size)
            sampled[:copied] = waveform[:copied]
            if copied < total_steps:
                sampled[copied:] = waveform[-1]
            activity = np.maximum(activity, sampled / peak)
    return np.maximum.accumulate(activity[::-1])[::-1] if activity.size else activity


def _source_residual(remaining_activity: np.ndarray, current_step: int) -> float:
    """Return the largest normalized source drive at or after ``current_step``."""
    index = int(current_step)
    if index < 0:
        raise ValueError("current_step must be non-negative.")
    return float(remaining_activity[index]) if index < remaining_activity.size else 0.0


def _selected_monitor_names(program: CompiledProgram, policy: AutoTermination):
    available = {
        spec.name
        for spec in program.monitors
        if spec.dft_enabled
        and spec.freq_count > 0
        and spec.dft_point_count > 0
        and np.any(np.asarray(spec.dft_component_mask) > 0.0)
    }
    if policy.monitor_names:
        missing = set(policy.monitor_names) - available
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(
                "Automatic termination monitors must be applicable "
                f"frequency-domain monitors: {names}."
            )
        return policy.monitor_names
    return tuple(spec.name for spec in program.monitors if spec.name in available)


def _monitor_vector(results: SimulationResults, names: tuple[str, ...]) -> np.ndarray:
    """Flatten selected raw DFT accumulators into one convergence vector."""
    values = []
    for name in names:
        monitor = results.monitors[name]
        fields = monitor._raw_dft_fields or monitor.dft_fields
        values.extend(
            np.asarray(fields[component], dtype=np.complex128).reshape(-1)
            for component in sorted(fields)
        )
    return np.concatenate(values) if values else np.empty(0, dtype=np.complex128)


def _relative_monitor_change(current: np.ndarray, previous: np.ndarray | None):
    """Return a scale-safe relative L2 change for cumulative monitor values."""
    if previous is None:
        return None
    if current.shape != previous.shape:
        raise ValueError("Automatic termination monitor values changed shape.")
    numerator = float(np.linalg.norm(current - previous))
    denominator = max(float(np.linalg.norm(current)), float(np.linalg.norm(previous)))
    if denominator <= np.finfo(float).tiny:
        return 0.0 if numerator <= np.finfo(float).tiny else np.inf
    return numerator / denominator


def _configured_dft_weight_sum(simulation, spec) -> float:
    """Return a monitor's DFT window weight through the configured time limit."""
    steps = np.arange(int(simulation.num_steps), dtype=np.int64)
    times = float(simulation.time[0]) + (steps.astype(np.float64) + 1.0) * float(
        simulation.dt
    )
    selected = (
        (steps % max(1, int(spec.dft_record_interval)) == 0)
        & (times >= float(spec.dft_t_start))
        & (times <= float(spec.dft_t_end))
    )
    if not np.any(selected):
        return 0.0
    if int(spec.dft_window_code) != 1 or not np.isfinite(spec.dft_t_end):
        return float(np.count_nonzero(selected))
    span = max(float(spec.dft_t_end) - float(spec.dft_t_start), 1e-30)
    tau = np.clip((times[selected] - float(spec.dft_t_start)) / span, 0.0, 1.0)
    return float(np.sum(0.5 * (1.0 - np.cos(2.0 * np.pi * tau))))


def _complete_converged_dft_weights(
    results: SimulationResults, simulation, program: CompiledProgram
) -> SimulationResults:
    """Normalize converged DFTs as though their negligible tail had been sampled."""
    monitors = dict(results.monitors)
    for spec in program.monitors:
        if not spec.dft_enabled or spec.freq_count <= 0:
            continue
        monitor = simulation.monitors[int(spec.monitor_index)]
        name = str(getattr(monitor, "name", None) or f"monitor_{spec.monitor_index}")
        result = monitors[name]
        total_weight = _configured_dft_weight_sum(simulation, spec)
        weights = np.full(result.dft_weight_sum.shape, total_weight, dtype=np.float64)
        monitors[name] = replace(result, dft_weight_sum=weights)
    return replace(results, monitors=monitors)


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
        metrics=program.metrics,
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


def run_until_terminated(
    simulation,
    policy: AutoTermination,
    *,
    progress: bool,
    store_full_materials: bool,
    sharding,
) -> SimulationResults:
    """Run reusable chunks until convergence, failure, or the configured time limit."""
    if not isinstance(policy, AutoTermination):
        raise TypeError("termination must be an AutoTermination instance or None.")
    chunk_steps = min(int(policy.chunk_steps), int(simulation.num_steps))
    first_program = simulation.compile(
        num_steps=chunk_steps, sharding=sharding, progress=progress
    )
    monitor_names = _selected_monitor_names(first_program, policy)
    monitor_tolerance = (
        None if policy.monitor_change is None else float(policy.monitor_change)
    )
    use_monitor = monitor_tolerance is not None and bool(monitor_names)
    if policy.field_decay == 0.0 and not use_monitor:
        raise ValueError(
            "Automatic termination has no applicable field or monitor criterion."
        )

    source_activity = _remaining_source_activity(first_program, simulation.num_steps)
    terms = _energy_terms(first_program)
    state = SimulationState.initial(first_program.grid, t=float(simulation.time[0]))
    previous_energy: float | None = None
    previous_monitor: np.ndarray | None = None
    energy = peak_energy = max_field = 0.0
    field_decay = monitor_change = None
    source_decay = _source_residual(source_activity, 0)
    successful_checks = growth_checks = 0
    reason = "time_limit"
    last_run: SimulationRun | None = None

    try:
        while int(state.current_step) < int(simulation.num_steps):
            current_step = int(state.current_step)
            remaining = int(simulation.num_steps) - current_step
            steps = min(chunk_steps, remaining)
            program = (
                first_program
                if steps == chunk_steps
                else simulation.compile(
                    num_steps=steps, sharding=sharding, progress=False
                )
            )
            last_run = run_simulation_program(
                simulation,
                program,
                state,
                progress=False,
                store_full_materials=store_full_materials,
                monitor_steps=remaining,
                donate_state=True,
            )
            state = last_run.state
            current_step = int(state.current_step)
            if progress:
                _print_inline_progress(current_step, int(simulation.num_steps))

            energy, max_field, fields_finite = _field_diagnostics(state, terms)
            current_monitor = _monitor_vector(last_run.results, monitor_names)
            monitors_finite = bool(np.isfinite(current_monitor).all())
            if not fields_finite or not np.isfinite(energy) or not monitors_finite:
                reason = "nonfinite"
                break

            peak_energy = max(peak_energy, energy)
            field_decay = (
                energy / peak_energy if peak_energy > np.finfo(float).tiny else 0.0
            )
            source_decay = _source_residual(source_activity, current_step)
            source_off = source_decay <= policy.source_decay
            monitor_change = _relative_monitor_change(current_monitor, previous_monitor)

            if (
                source_off
                and previous_energy is not None
                and previous_energy > np.finfo(float).tiny
                and energy > policy.growth_factor * previous_energy
            ):
                growth_checks += 1
            else:
                growth_checks = 0
            if growth_checks >= policy.growth_checks:
                reason = "diverged"
                break

            eligible = source_off and current_step >= policy.min_steps
            energy_stable = (
                policy.field_decay == 0.0 or field_decay <= policy.field_decay
            )
            monitor_stable = not use_monitor or (
                monitor_change is not None
                and monitor_tolerance is not None
                and monitor_change <= monitor_tolerance
            )
            if eligible and energy_stable and monitor_stable:
                successful_checks += 1
            else:
                successful_checks = 0
            if successful_checks >= policy.consecutive_checks:
                reason = "converged"
                break

            previous_energy = energy
            previous_monitor = current_monitor
    finally:
        if progress:
            _finish_inline_progress()

    if last_run is None:
        raise RuntimeError("Automatic termination executed no simulation steps.")
    report = RunTermination(
        reason=reason,
        steps=int(state.current_step),
        time=float(state.t),
        converged=reason == "converged",
        field_decay=field_decay,
        monitor_change=monitor_change if use_monitor else None,
        source_decay=source_decay,
        energy=energy,
        peak_energy=peak_energy,
        max_field=max_field,
        consecutive_checks=successful_checks,
    )
    results = last_run.results
    if reason == "converged":
        results = _complete_converged_dft_weights(results, simulation, first_program)
    return replace(results, termination=report)


_init_persistent_cache()
