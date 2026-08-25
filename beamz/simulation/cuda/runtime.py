"""Lower fused 3D Yee phases to the optional BeamZ CUDA typed FFI targets."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from beamz.simulation import _cuda_abi as abi
from beamz.simulation.backend import CUDA_ABI_VERSION
from beamz.simulation.model import SimulationState

_PHASE_H = 0
_PHASE_E = 1
_COMPONENT_CODE = {name: index for index, name in enumerate(("Hx", "Hy", "Hz"))}
_COMPONENT_CODE.update({name: index for index, name in enumerate(("Ex", "Ey", "Ez"))})
_EMPTY = jnp.empty((0,), dtype=jnp.float32)
_EMPTY_SOURCE_GROUP = (
    jnp.zeros((0, 1, 1, 1), dtype=jnp.float32),
    jnp.zeros((0, 1), dtype=jnp.float32),
    jnp.zeros((0, 3), dtype=jnp.int32),
)
_METRIC_KIND_CODE = {
    "isotropic_uniform": 0,
    "axis_uniform": 1,
    "rectilinear": 2,
}

_FIELD_COUNT = abi.FIELD_COUNT
_CPML_TERM_COUNT = abi.CPML_TERM_COUNT
_SOURCE_GROUP_COUNT = abi.SOURCE_GROUP_COUNT
_CPML_H_PSI_INPUT = abi.CPML_H_PSI_INPUT_OFFSET
_CPML_E_PSI_INPUT = abi.CPML_E_PSI_INPUT_OFFSET
_TEMPORAL_FIELD_WORKSPACE_INPUT = abi.TEMPORAL_FIELD_WORKSPACE_INPUT
_TEMPORAL_PSI_WORKSPACE_INPUT = abi.TEMPORAL_PSI_WORKSPACE_INPUT


def _metallic_edge_mask(edges: frozenset[str]) -> int:
    order = ("front", "back", "bottom", "top", "left", "right")
    return sum(1 << index for index, name in enumerate(order) if name in edges)


def _boundary_code(edges: frozenset[str], terms=()) -> int:
    """Pack PEC faces and a CPML thickness safe for every supplied phase.

    A non-zero CPML thickness selects a CUDA specialization which deliberately
    skips the per-term packed-slab descriptor.  It is therefore valid only when
    *all* supplied H and E recurrences have the same symmetric slab.  Phase-only
    FFI calls pass one phase; native program calls pass both phases.
    """
    thickness = 0
    if terms:
        first = int(terms[0].slab.low)
        if first > 0 and all(
            int(term.slab.low) == first and int(term.slab.high) == first
            for term in terms
        ):
            thickness = first
    return _metallic_edge_mask(edges) | (thickness << 8)


def _term_metadata(terms) -> jnp.ndarray:
    return jnp.asarray(
        [
            (
                _COMPONENT_CODE[term.component],
                int(term.axis),
                int(term.slab.low),
                int(term.slab.high),
                1 if float(term.sign) > 0.0 else -1,
            )
            for term in terms
        ],
        dtype=jnp.int32,
    ).reshape((len(terms), 5))


def _shape(value):
    return jax.ShapeDtypeStruct(value.shape, value.dtype)


def _fields(state):
    return (state.hx, state.hy, state.hz, state.ex, state.ey, state.ez)


def _ffi_call(target, result_values, aliases):
    return jax.ffi.ffi_call(
        target,
        tuple(_shape(value) for value in result_values),
        input_output_aliases=aliases,
        vmap_method="sequential",
    )


def _phase_metrics(ctx, phase: int):
    """Return CUDA-axis-ordered derivative metrics for one Yee phase."""
    metrics = ctx.metrics
    if phase == _PHASE_H:
        return (metrics.e_to_h_z, metrics.e_to_h_y, metrics.e_to_h_x)
    return (metrics.h_to_e_z, metrics.h_to_e_y, metrics.h_to_e_x)


def _metric_kind_code(ctx) -> np.int32:
    try:
        return np.int32(_METRIC_KIND_CODE[ctx.config.metric_kind])
    except KeyError as exc:
        raise ValueError(
            f"Unsupported CUDA derivative metric kind: {ctx.config.metric_kind!r}"
        ) from exc


def _program_attributes(
    ctx,
    nsteps: int,
    layout: int,
    *,
    monitor_count: int = 0,
    coincident_source_group_mask: int = 0,
):
    return {
        "abi_version": np.int32(CUDA_ABI_VERSION),
        "cuda_flags": np.int32(ctx.config.cuda_flags),
        "nsteps": np.int32(nsteps),
        "dt": np.float32(ctx.dt),
        "resolution": np.float32(ctx.resolution),
        "boundary_code": np.int32(
            _boundary_code(
                ctx.boundary.cpml.metallic_edges,
                (*ctx.boundary.cpml.h_terms, *ctx.boundary.cpml.e_terms),
            )
        ),
        "metric_kind": _metric_kind_code(ctx),
        "program_layout": np.int32(layout),
        "cpml_enabled": np.int32(ctx.boundary.cpml.enabled),
        "monitor_count": np.int32(monitor_count),
        "coincident_source_group_mask": np.int32(coincident_source_group_mask),
    }


def _ffi_phase(
    target: str,
    phase: int,
    targets,
    sources,
    materials,
    terms,
    psi_terms,
    metrics,
    *,
    metric_kind,
    dt,
    resolution,
    cuda_flags,
    metallic_edges,
):
    if len(terms) != len(psi_terms):
        raise ValueError(
            "CUDA CPML terms and recurrence buffers must have equal length"
        )
    if terms and len(terms) != 6:
        raise ValueError("3D CUDA CPML requires exactly six derivative terms per phase")
    metadata = _term_metadata(terms)
    term_arrays = tuple(
        value for term in terms for value in (term.a, term.b, term.inv_kappa)
    )
    arguments = (
        *targets,
        *sources,
        *materials,
        metadata,
        *term_arrays,
        *psi_terms,
        *metrics,
    )
    psi_start = 13 + 3 * len(terms)
    aliases = {0: 0, 1: 1, 2: 2}
    aliases.update({psi_start + index: 3 + index for index in range(len(psi_terms))})
    call = _ffi_call(target, (*targets, *psi_terms), aliases)
    outputs = call(
        *arguments,
        abi_version=np.int32(CUDA_ABI_VERSION),
        cuda_flags=np.int32(cuda_flags),
        phase=np.int32(phase),
        nterms=np.int32(len(terms)),
        dt=np.float32(dt),
        resolution=np.float32(resolution),
        boundary_code=np.int32(_boundary_code(metallic_edges, terms)),
        metric_kind=np.int32(metric_kind),
    )
    return tuple(outputs)


def update_h(state, ctx, coeffs) -> SimulationState:
    """Advance the three magnetic fields and optional CPML memory on CUDA."""
    target = (
        abi.CUDA_HOPPER_TARGET
        if ctx.config.backend == "cuda_hopper"
        else abi.CUDA_STREAMED_TARGET
    )
    terms = ctx.boundary.cpml.h_terms
    materials = (
        (
            coeffs.h_sigma_m_x,
            coeffs.h_sigma_m_y,
            coeffs.h_sigma_m_z,
            _EMPTY,
            _EMPTY,
            _EMPTY,
        )
        if ctx.config.backend == "cuda_hopper"
        else (
            coeffs.h_decay_x,
            coeffs.h_decay_y,
            coeffs.h_decay_z,
            coeffs.h_source_x,
            coeffs.h_source_y,
            coeffs.h_source_z,
        )
    )
    outputs = _ffi_phase(
        target,
        _PHASE_H,
        (state.hx, state.hy, state.hz),
        (state.ex, state.ey, state.ez),
        materials,
        terms,
        state.cpml_psi_h_terms,
        _phase_metrics(ctx, _PHASE_H),
        metric_kind=_metric_kind_code(ctx),
        dt=ctx.dt,
        resolution=ctx.resolution,
        cuda_flags=ctx.config.cuda_flags,
        metallic_edges=ctx.boundary.cpml.metallic_edges,
    )
    return state._replace(
        hx=outputs[0],
        hy=outputs[1],
        hz=outputs[2],
        cpml_psi_h_terms=outputs[3:],
    )


def update_e(state, ctx, coeffs) -> SimulationState:
    """Advance the three electric fields and optional CPML memory on CUDA."""
    target = (
        abi.CUDA_HOPPER_TARGET
        if ctx.config.backend == "cuda_hopper"
        else abi.CUDA_STREAMED_TARGET
    )
    terms = ctx.boundary.cpml.e_terms
    materials = (
        (
            coeffs.e_conductivity_x,
            coeffs.e_conductivity_y,
            coeffs.e_conductivity_z,
            coeffs.e_permittivity_x,
            coeffs.e_permittivity_y,
            coeffs.e_permittivity_z,
        )
        if ctx.config.backend == "cuda_hopper"
        else (
            coeffs.e_decay_x,
            coeffs.e_decay_y,
            coeffs.e_decay_z,
            coeffs.e_source_x,
            coeffs.e_source_y,
            coeffs.e_source_z,
        )
    )
    outputs = _ffi_phase(
        target,
        _PHASE_E,
        (state.ex, state.ey, state.ez),
        (state.hx, state.hy, state.hz),
        materials,
        terms,
        state.cpml_psi_e_terms,
        _phase_metrics(ctx, _PHASE_E),
        metric_kind=_metric_kind_code(ctx),
        dt=ctx.dt,
        resolution=ctx.resolution,
        cuda_flags=ctx.config.cuda_flags,
        metallic_edges=ctx.boundary.cpml.metallic_edges,
    )
    return state._replace(
        ex=outputs[0],
        ey=outputs[1],
        ez=outputs[2],
        cpml_psi_e_terms=outputs[3:],
    )


def _cpml_graph_io(state, ctx, coeffs):
    fields = _fields(state)
    h_terms = ctx.boundary.cpml.h_terms
    e_terms = ctx.boundary.cpml.e_terms
    h_materials = (
        coeffs.h_decay_x,
        coeffs.h_decay_y,
        coeffs.h_decay_z,
        coeffs.h_source_x,
        coeffs.h_source_y,
        coeffs.h_source_z,
    )
    e_materials = (
        coeffs.e_decay_x,
        coeffs.e_decay_y,
        coeffs.e_decay_z,
        coeffs.e_source_x,
        coeffs.e_source_y,
        coeffs.e_source_z,
    )
    h_payload = (
        *h_materials,
        _term_metadata(h_terms),
        *(value for term in h_terms for value in (term.a, term.b, term.inv_kappa)),
        *state.cpml_psi_h_terms,
    )
    e_payload = (
        *e_materials,
        _term_metadata(e_terms),
        *(value for term in e_terms for value in (term.a, term.b, term.inv_kappa)),
        *state.cpml_psi_e_terms,
    )
    arguments = (
        *fields,
        *h_payload,
        *e_payload,
        *_phase_metrics(ctx, _PHASE_H),
        *_phase_metrics(ctx, _PHASE_E),
    )
    result_values = (*fields, *state.cpml_psi_h_terms, *state.cpml_psi_e_terms)
    aliases = {index: index for index in range(_FIELD_COUNT)}
    aliases.update(
        {
            _CPML_H_PSI_INPUT + index: _FIELD_COUNT + index
            for index in range(_CPML_TERM_COUNT)
        }
    )
    aliases.update(
        {
            _CPML_E_PSI_INPUT + index: _FIELD_COUNT + _CPML_TERM_COUNT + index
            for index in range(_CPML_TERM_COUNT)
        }
    )
    return arguments, result_values, aliases


def _yee_graph_io(state, ctx, coeffs):
    fields = _fields(state)
    materials = (
        coeffs.h_decay_x,
        coeffs.h_decay_y,
        coeffs.h_decay_z,
        coeffs.h_source_x,
        coeffs.h_source_y,
        coeffs.h_source_z,
        coeffs.e_decay_x,
        coeffs.e_decay_y,
        coeffs.e_decay_z,
        coeffs.e_source_x,
        coeffs.e_source_y,
        coeffs.e_source_z,
    )
    arguments = (
        *fields,
        *materials,
        *_phase_metrics(ctx, _PHASE_H),
        *_phase_metrics(ctx, _PHASE_E),
    )
    return arguments, fields, {index: index for index in range(_FIELD_COUNT)}


def _graph_io(state, ctx, coeffs):
    return (
        _cpml_graph_io(state, ctx, coeffs)
        if ctx.boundary.cpml.enabled
        else _yee_graph_io(state, ctx, coeffs)
    )


def _replace_fields(state, outputs, start=0) -> SimulationState:
    return state._replace(
        hx=outputs[start],
        hy=outputs[start + 1],
        hz=outputs[start + 2],
        ex=outputs[start + 3],
        ey=outputs[start + 4],
        ez=outputs[start + 5],
    )


def _replace_graph_outputs(state, outputs) -> SimulationState:
    return _replace_fields(state, outputs)._replace(
        cpml_psi_h_terms=outputs[_FIELD_COUNT : _FIELD_COUNT + _CPML_TERM_COUNT],
        cpml_psi_e_terms=outputs[
            _FIELD_COUNT + _CPML_TERM_COUNT : _FIELD_COUNT + 2 * _CPML_TERM_COUNT
        ],
    )


def _temporal_cpml_graph_io(state, ctx, coeffs):
    arguments, _, _ = _cpml_graph_io(state, ctx, coeffs)
    fields = _fields(state)
    workspace = tuple(jnp.empty_like(value) for value in fields)
    psi = (*state.cpml_psi_h_terms, *state.cpml_psi_e_terms)
    psi_workspace = tuple(jnp.empty_like(value) for value in psi)
    result_values = (*fields, *workspace, *psi, *psi_workspace)
    aliases = {index: index for index in range(_FIELD_COUNT)}
    aliases.update(
        {
            _TEMPORAL_FIELD_WORKSPACE_INPUT + index: _FIELD_COUNT + index
            for index in range(_FIELD_COUNT)
        }
    )
    aliases.update(
        {
            _CPML_H_PSI_INPUT + index: 2 * _FIELD_COUNT + index
            for index in range(_CPML_TERM_COUNT)
        }
    )
    aliases.update(
        {
            _CPML_E_PSI_INPUT + index: 2 * _FIELD_COUNT + _CPML_TERM_COUNT + index
            for index in range(_CPML_TERM_COUNT)
        }
    )
    aliases.update(
        {
            _TEMPORAL_PSI_WORKSPACE_INPUT + index: 2 * _FIELD_COUNT
            + 2 * _CPML_TERM_COUNT
            + index
            for index in range(2 * _CPML_TERM_COUNT)
        }
    )
    return (*arguments, *workspace, *psi_workspace), result_values, aliases


def _replace_temporal_cpml_outputs(state, outputs, nsteps):
    field_start = 0 if nsteps % 2 == 0 else _FIELD_COUNT
    psi_start = 2 * _FIELD_COUNT
    if nsteps % 2:
        psi_start += 2 * _CPML_TERM_COUNT
    return _replace_fields(state, outputs, field_start)._replace(
        cpml_psi_h_terms=outputs[psi_start : psi_start + _CPML_TERM_COUNT],
        cpml_psi_e_terms=outputs[
            psi_start + _CPML_TERM_COUNT : psi_start + 2 * _CPML_TERM_COUNT
        ],
    )


def _source_group_arguments(groups):
    if len(groups) != _SOURCE_GROUP_COUNT:
        raise ValueError("CUDA source graph requires nine phase/component groups")
    return tuple(
        value
        for group in groups
        for value in (
            _EMPTY_SOURCE_GROUP
            if group is None
            else (group.coeffs, group.waveforms, group.starts)
        )
    )


def _coincident_source_group_mask(groups) -> int:
    """Encode equal-shape source groups whose slabs share one origin."""
    mask = 0
    for index, group in enumerate(groups):
        if group is None:
            continue
        starts = getattr(group, "starts_tuple", ())
        if starts and all(start == starts[0] for start in starts[1:]):
            mask |= 1 << index
    return mask


def _temporal_cpml_source_groups_supported(ctx, coeffs, nsteps: int) -> bool:
    """Whether a frozen second field bank can use the fused regular-grid core."""
    if (
        nsteps < 2
        or not ctx.boundary.cpml.enabled
        or ctx.config.metric_kind != "isotropic_uniform"
        or (
            _boundary_code(ctx.boundary.cpml.metallic_edges, ctx.boundary.cpml.h_terms)
            >> 8
        )
        <= 0
    ):
        return False
    h_values = (
        coeffs.h_decay_x,
        coeffs.h_decay_y,
        coeffs.h_decay_z,
        coeffs.h_source_x,
        coeffs.h_source_y,
        coeffs.h_source_z,
    )
    e_sources = (coeffs.e_source_x, coeffs.e_source_y, coeffs.e_source_z)
    return all(value.ndim == 0 for value in h_values) and all(
        value.ndim == 1 and value.dtype == jnp.int32 for value in e_sources
    )


def run_source_group_steps(state, ctx, coeffs, groups, nsteps: int) -> SimulationState:
    """Advance packed slab-source groups in every leapfrog phase on CUDA."""
    if nsteps < 1:
        raise ValueError("CUDA step count must be positive")
    source_arguments = _source_group_arguments(groups)
    use_temporal_cpml = _temporal_cpml_source_groups_supported(ctx, coeffs, nsteps)
    arguments, result_values, aliases = (
        _temporal_cpml_graph_io(state, ctx, coeffs)
        if use_temporal_cpml
        else _graph_io(state, ctx, coeffs)
    )
    call = _ffi_call(abi.CUDA_PROGRAM_TARGET, result_values, aliases)
    outputs = call(
        *arguments,
        *source_arguments,
        state.current_step,
        **_program_attributes(
            ctx,
            nsteps,
            (
                abi.PROGRAM_LAYOUT_SOURCE_TEMPORAL_CPML
                if use_temporal_cpml
                else abi.PROGRAM_LAYOUT_SOURCE_IN_PLACE
            ),
            coincident_source_group_mask=_coincident_source_group_mask(groups),
        ),
    )
    if use_temporal_cpml:
        return _replace_temporal_cpml_outputs(state, outputs, nsteps)
    if ctx.boundary.cpml.enabled:
        return _replace_graph_outputs(state, outputs)
    return _replace_fields(state, outputs)


def pack_dft_monitors(monitors):
    """Pack DFT gather plans with offsets into ragged accumulator arenas."""
    if not monitors:
        raise ValueError("CUDA DFT graph requires at least one monitor")
    max_points = max(int(monitor.dft_point_count) for monitor in monitors)
    max_frequencies = max(int(monitor.freq_count) for monitor in monitors)
    max_neighbors = max(
        int(indices.shape[1])
        for monitor in monitors
        for indices in monitor.dft_flat_idx
    )

    def pad_plan(value, points, neighbors, *, fill=0):
        return jnp.pad(
            value,
            ((0, points - value.shape[0]), (0, neighbors - value.shape[1])),
            constant_values=fill,
        )

    indices = jnp.stack(
        tuple(
            jnp.stack(
                tuple(
                    pad_plan(value, max_points, max_neighbors)
                    for value in monitor.dft_flat_idx
                )
            )
            for monitor in monitors
        )
    ).astype(jnp.int32)
    weights = jnp.stack(
        tuple(
            jnp.stack(
                tuple(
                    pad_plan(value, max_points, max_neighbors)
                    for value in monitor.dft_weights
                )
            )
            for monitor in monitors
        )
    ).astype(jnp.float32)
    frequencies = jnp.stack(
        tuple(
            jnp.pad(
                monitor.freq_hz,
                (0, max_frequencies - monitor.freq_count),
            )
            for monitor in monitors
        )
    ).astype(jnp.float32)
    component_masks = jnp.stack(
        tuple(monitor.dft_component_mask for monitor in monitors)
    ).astype(jnp.float32)
    counts = jnp.asarray(
        tuple(
            (
                monitor.freq_count,
                monitor.dft_point_count,
                monitor.dft_record_interval,
                monitor.dft_value_offset,
                monitor.dft_weight_offset,
            )
            for monitor in monitors
        ),
        dtype=jnp.int32,
    )
    codes = jnp.asarray(
        tuple(
            (monitor.dft_window_code, monitor.dft_normalization_code)
            for monitor in monitors
        ),
        dtype=jnp.int32,
    )
    windows = jnp.asarray(
        tuple(
            (monitor.dft_t_start, monitor.dft_t_end, monitor.dft_length_unit)
            for monitor in monitors
        ),
        dtype=jnp.float32,
    )
    return (
        indices,
        weights,
        frequencies,
        component_masks,
        counts,
        codes,
        windows,
    )


def run_program_steps(
    state, ctx, coeffs, groups, packed_monitors, nsteps: int
) -> SimulationState:
    """Advance arbitrary slab sources and packed vector DFTs in one CUDA graph."""
    if nsteps < 1:
        raise ValueError("CUDA step count must be positive")
    if state.dft_vec_re.dtype != jnp.float32:
        raise ValueError("CUDA program graph requires float32 DFT accumulators")
    source_arguments = _source_group_arguments(groups)
    use_temporal_cpml = _temporal_cpml_source_groups_supported(ctx, coeffs, nsteps)
    arguments, result_values, aliases = (
        _temporal_cpml_graph_io(state, ctx, coeffs)
        if use_temporal_cpml
        else _graph_io(state, ctx, coeffs)
    )
    state_output_count = len(result_values)
    result_values = (
        *result_values,
        state.dft_vec_re,
        state.dft_vec_im,
        state.dft_weight_sum,
    )
    monitor_output_start = len(arguments) + len(source_arguments) + len(packed_monitors)
    aliases = {
        **aliases,
        monitor_output_start: state_output_count,
        monitor_output_start + 1: state_output_count + 1,
        monitor_output_start + 2: state_output_count + 2,
    }
    call = _ffi_call(abi.CUDA_PROGRAM_TARGET, result_values, aliases)
    outputs = call(
        *arguments,
        *source_arguments,
        *packed_monitors,
        state.dft_vec_re,
        state.dft_vec_im,
        state.dft_weight_sum,
        state.t,
        state.current_step,
        **_program_attributes(
            ctx,
            nsteps,
            (
                abi.PROGRAM_LAYOUT_MONITOR_TEMPORAL_CPML
                if use_temporal_cpml
                else abi.PROGRAM_LAYOUT_MONITOR_IN_PLACE
            ),
            monitor_count=int(packed_monitors[0].shape[0]),
            coincident_source_group_mask=_coincident_source_group_mask(groups),
        ),
    )
    if use_temporal_cpml:
        next_state = _replace_temporal_cpml_outputs(state, outputs, nsteps)
    elif ctx.boundary.cpml.enabled:
        next_state = _replace_graph_outputs(state, outputs)
    else:
        next_state = _replace_fields(state, outputs)
    return next_state._replace(
        dft_vec_re=outputs[state_output_count],
        dft_vec_im=outputs[state_output_count + 1],
        dft_weight_sum=outputs[state_output_count + 2],
    )


def run_steps(state, ctx, coeffs, nsteps: int) -> SimulationState:
    """Advance a source-free, monitor-free Yee run through one CUDA FFI call."""
    if nsteps < 1:
        raise ValueError("CUDA step count must be positive")
    if _temporal_cpml_source_groups_supported(ctx, coeffs, nsteps):
        # The source-group target's empty groups add no graph nodes. Reusing it
        # gives plain CPML runs the same frozen-input field banks without
        # maintaining a second native handler for an identical update graph.
        return run_source_group_steps(
            state, ctx, coeffs, (None,) * _SOURCE_GROUP_COUNT, nsteps
        )
    fields = _fields(state)
    if ctx.boundary.cpml.enabled:
        arguments, result_values, aliases = _cpml_graph_io(state, ctx, coeffs)
        call = _ffi_call(abi.CUDA_PROGRAM_TARGET, result_values, aliases)
        outputs = call(
            *arguments,
            **_program_attributes(ctx, nsteps, abi.PROGRAM_LAYOUT_CPML_IN_PLACE),
        )
        return _replace_graph_outputs(state, outputs)
    materials = (
        coeffs.h_decay_x,
        coeffs.h_decay_y,
        coeffs.h_decay_z,
        coeffs.h_source_x,
        coeffs.h_source_y,
        coeffs.h_source_z,
        coeffs.e_decay_x,
        coeffs.e_decay_y,
        coeffs.e_decay_z,
        coeffs.e_source_x,
        coeffs.e_source_y,
        coeffs.e_source_z,
    )
    temporal_eligible = (
        nsteps >= 4 and _metallic_edge_mask(ctx.boundary.cpml.metallic_edges) == 63
    )
    if temporal_eligible:
        workspace = tuple(jnp.empty_like(value) for value in fields)
        call = _ffi_call(
            abi.CUDA_PROGRAM_TARGET,
            (*fields, *workspace),
            {index: index for index in range(2 * _FIELD_COUNT)},
        )
        outputs = call(
            *fields,
            *workspace,
            *materials,
            *_phase_metrics(ctx, _PHASE_H),
            *_phase_metrics(ctx, _PHASE_E),
            **_program_attributes(ctx, nsteps, abi.PROGRAM_LAYOUT_YEE_TEMPORAL),
        )
        return _replace_fields(state, outputs)
    call = _ffi_call(
        abi.CUDA_PROGRAM_TARGET,
        fields,
        {index: index for index in range(_FIELD_COUNT)},
    )
    outputs = call(
        *fields,
        *materials,
        *_phase_metrics(ctx, _PHASE_H),
        *_phase_metrics(ctx, _PHASE_E),
        **_program_attributes(ctx, nsteps, abi.PROGRAM_LAYOUT_YEE_IN_PLACE),
    )
    return _replace_fields(state, outputs)
