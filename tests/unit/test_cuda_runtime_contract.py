from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from beamz.simulation import _cuda_abi as abi
from beamz.simulation import backend as backend_runtime
from beamz.simulation import kernels
from beamz.simulation.compile import _elide_uniform_grid
from beamz.simulation.cuda import runtime as cuda_runtime
from beamz.simulation.execute import (
    CUDA_GRAPH_MAX_STEPS,
    build_scan,
    initial_program_state,
)
from tests.performance.h100_workloads import H100Workload


def _program_and_state(*, cpml: bool, source: bool = False, monitor: bool = False):
    workload = H100Workload(
        name="cuda_contract",
        shape_zyx=(8, 10, 12),
        timesteps=3,
        resolution=100e-9,
        pml_cells=2,
        heterogeneous=True,
        cpml=cpml,
        source=source,
        monitor=monitor,
    )
    simulation = workload.build()
    program = simulation.compile(backend="jax")
    state = initial_program_state(
        program,
        t=float(simulation.time[0]),
        current_step=0,
        monitor_steps=3,
    )
    config = replace(
        program.config,
        backend="cuda_streamed",
        cuda_flags=backend_runtime.CUDA_DEFAULT_FLAGS,
    )
    context = kernels.CompiledStepContext(
        config=config,
        boundary=program.boundary,
        source_batches={},
        metrics=program.metrics,
        resolution=config.resolution,
        dt=config.dt,
        dt_scalar=jnp.asarray(config.dt, dtype=jnp.float32),
        is_3d=True,
    )
    return program, state, context


def test_cuda_boundary_code_packs_only_uniform_two_sided_cpml():
    uniform = tuple(
        SimpleNamespace(slab=SimpleNamespace(low=4, high=4)) for _ in range(6)
    )
    asymmetric = (
        *uniform[:-1],
        SimpleNamespace(slab=SimpleNamespace(low=4, high=0)),
    )

    assert cuda_runtime._boundary_code(frozenset({"front"}), uniform) == (4 << 8) | 1
    assert cuda_runtime._boundary_code(frozenset({"front"}), asymmetric) == 1
    assert cuda_runtime._boundary_code(frozenset({"right"})) == 1 << 5


def test_cuda_program_boundary_code_requires_matching_h_and_e_cpml_slabs():
    _program, _state, context = _program_and_state(cpml=True)
    uniform_h = tuple(
        SimpleNamespace(slab=SimpleNamespace(low=4, high=4)) for _ in range(6)
    )
    staggered_e = (
        *uniform_h[:-1],
        SimpleNamespace(slab=SimpleNamespace(low=3, high=3)),
    )
    context = replace(
        context,
        boundary=SimpleNamespace(
            cpml=SimpleNamespace(
                enabled=True,
                metallic_edges=frozenset({"front"}),
                h_terms=uniform_h,
                e_terms=uniform_h,
            )
        ),
    )

    uniform_attributes = cuda_runtime._program_attributes(
        context, 3, abi.PROGRAM_LAYOUT_CPML_IN_PLACE
    )
    assert uniform_attributes["boundary_code"] == np.int32((4 << 8) | 1)

    context = replace(
        context,
        boundary=SimpleNamespace(
            cpml=SimpleNamespace(
                enabled=True,
                metallic_edges=frozenset({"front"}),
                h_terms=uniform_h,
                e_terms=staggered_e,
            )
        ),
    )
    attributes = cuda_runtime._program_attributes(
        context, 3, abi.PROGRAM_LAYOUT_CPML_IN_PLACE
    )

    # Program launches pass one boundary code to both phases.  A H-only check
    # would select the descriptor-free uniform path for E and index its shorter
    # packed slab with the H thickness.
    assert attributes["boundary_code"] == np.int32(1)


def test_cuda_cpml_bf16_state_is_explicit_and_preserves_continuation(monkeypatch):
    program, state, _context = _program_and_state(cpml=True)
    cuda_program = replace(
        program, config=replace(program.config, backend="cuda_streamed")
    )
    seeded = state._replace(
        cpml_psi_h_terms=tuple(value + 0.125 for value in state.cpml_psi_h_terms),
        cpml_psi_e_terms=tuple(value - 0.25 for value in state.cpml_psi_e_terms),
    )
    monkeypatch.setenv("BEAMZ_CUDA_CPML_PSI_PRECISION", "bf16")
    cuda_program = replace(
        cuda_program,
        config=replace(
            cuda_program.config,
            cuda_flags=backend_runtime.cuda_flags_from_env(),
        ),
    )

    converted = initial_program_state(
        cuda_program,
        t=float(cuda_program.config.dt),
        current_step=1,
        continuation=seeded,
        monitor_steps=3,
    )

    assert all(value.dtype == jnp.bfloat16 for value in converted.cpml_psi_h_terms)
    assert all(value.dtype == jnp.bfloat16 for value in converted.cpml_psi_e_terms)
    np.testing.assert_allclose(
        np.asarray(converted.cpml_psi_h_terms[0], dtype=np.float32), 0.125
    )
    np.testing.assert_allclose(
        np.asarray(converted.cpml_psi_e_terms[0], dtype=np.float32), -0.25
    )


def test_cuda_policy_rejects_unknown_psi_precision(monkeypatch):
    monkeypatch.setenv("BEAMZ_CUDA_CPML_PSI_PRECISION", "fp8")

    with np.testing.assert_raises_regex(ValueError, "must be 'fp32' or 'bf16'"):
        backend_runtime.cuda_flags_from_env()


def test_cuda_ffi_phase_packs_cpml_and_aliases_state(monkeypatch):
    program, state, context = _program_and_state(cpml=True)
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            nterms = int(attributes["nterms"])
            psi_start = 13 + 3 * nterms
            return (*arguments[:3], *arguments[psi_start : psi_start + nterms])

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)

    next_state = cuda_runtime.update_h(state, context, program.coefficients)

    target, results, options, arguments, attributes = captured[0]
    assert target == "beamz_cuda_streamed"
    assert len(results) == 9
    assert len(arguments) == 40
    assert attributes == {
        "abi_version": np.int32(cuda_runtime.CUDA_ABI_VERSION),
        "cuda_flags": np.int32(context.config.cuda_flags),
        "phase": np.int32(0),
        "nterms": np.int32(6),
        "dt": np.float32(context.dt),
        "resolution": np.float32(context.resolution),
        # Low six bits encode PEC faces; high bits carry the uniform CPML width.
        "boundary_code": np.int32(2 << 8),
        "metric_kind": np.int32(0),
    }
    assert options["input_output_aliases"] == {
        0: 0,
        1: 1,
        2: 2,
        31: 3,
        32: 4,
        33: 5,
        34: 6,
        35: 7,
        36: 8,
    }
    assert next_state.cpml_psi_h_terms == state.cpml_psi_h_terms


def test_cuda_ffi_phase_supports_non_cpml_yee_update(monkeypatch):
    program, state, context = _program_and_state(cpml=False)
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            return arguments[:3]

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)

    next_state = cuda_runtime.update_e(state, context, program.coefficients)

    _, results, options, arguments, attributes = captured[0]
    assert len(results) == 3
    assert len(arguments) == 16
    assert attributes["phase"] == 1
    assert attributes["nterms"] == 0
    assert options["input_output_aliases"] == {0: 0, 1: 1, 2: 2}
    assert next_state.cpml_psi_e_terms == ()


def test_cuda_multi_step_ffi_aliases_all_fields(monkeypatch):
    program, state, context = _program_and_state(cpml=False)
    coefficients = program.coefficients._replace(
        **{
            name: jnp.asarray(1.0, dtype=jnp.float32)
            for name in (
                "h_decay_x",
                "h_decay_y",
                "h_decay_z",
                "h_source_x",
                "h_source_y",
                "h_source_z",
                "e_decay_x",
                "e_decay_y",
                "e_decay_z",
                "e_source_x",
                "e_source_y",
                "e_source_z",
            )
        }
    )
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            return arguments[:12]

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)

    next_state = cuda_runtime.run_steps(state, context, coefficients, 7)

    target, results, options, arguments, attributes = captured[0]
    assert target == abi.CUDA_PROGRAM_TARGET
    assert len(results) == 12
    assert len(arguments) == 30
    assert options["input_output_aliases"] == {index: index for index in range(12)}
    assert attributes == {
        "abi_version": np.int32(cuda_runtime.CUDA_ABI_VERSION),
        "cuda_flags": np.int32(context.config.cuda_flags),
        "graph_cache_capacity": np.int32(context.config.cuda_graph_cache_capacity),
        "nsteps": np.int32(7),
        "dt": np.float32(context.dt),
        "resolution": np.float32(context.resolution),
        "boundary_code": np.int32(63),
        "metric_kind": np.int32(0),
        "program_layout": np.int32(abi.PROGRAM_LAYOUT_YEE_TEMPORAL),
        "cpml_enabled": np.int32(0),
        "monitor_count": np.int32(0),
        "coincident_source_group_mask": np.int32(0),
    }
    assert next_state.hx is state.hx
    assert next_state.ez is state.ez


def test_cuda_scan_replays_one_bounded_graph_and_advances_chunk_clocks(monkeypatch):
    program, state, _context = _program_and_state(cpml=False)
    requested_steps = 2 * CUDA_GRAPH_MAX_STEPS + 7
    program = replace(
        program,
        config=replace(
            program.config,
            backend="cuda_streamed",
            num_steps=requested_steps,
        ),
    )
    traced_step_counts = []

    def fake_run_steps(chunk_state, _context, _coefficients, nsteps):
        traced_step_counts.append(nsteps)
        return chunk_state._replace(ex=chunk_state.ex + np.float32(nsteps))

    monkeypatch.setattr(
        "beamz.simulation.cuda.run_steps",
        fake_run_steps,
    )

    next_state = build_scan(program)(state, program.coefficients)

    assert traced_step_counts == [CUDA_GRAPH_MAX_STEPS, 7]
    assert int(next_state.current_step) == requested_steps
    np.testing.assert_allclose(
        next_state.t,
        state.t + np.float32(program.config.dt * requested_steps),
    )
    np.testing.assert_allclose(next_state.ex, state.ex + np.float32(requested_steps))


def test_cuda_scan_routes_sources_to_grouped_graph(monkeypatch):
    program, state, _context = _program_and_state(cpml=True, source=True)
    program = replace(
        program,
        config=replace(program.config, backend="cuda_streamed"),
    )
    calls = []

    def fake_run_source_group_steps(
        chunk_state, _context, _coefficients, groups, nsteps
    ):
        calls.append((groups, nsteps))
        return chunk_state

    monkeypatch.setattr(
        "beamz.simulation.cuda.run_source_group_steps",
        fake_run_source_group_steps,
    )

    build_scan(program)(state, program.coefficients)

    assert len(calls) == 1
    assert sum(group is not None for group in calls[0][0]) == 1
    assert calls[0][1] == program.config.num_steps


@pytest.mark.parametrize("record_interval", [1, 2])
def test_cuda_scan_routes_monitors_to_general_program_graph(
    monkeypatch, record_interval
):
    program, state, _context = _program_and_state(cpml=True, source=True, monitor=True)
    program = replace(
        program,
        config=replace(program.config, backend="cuda_streamed"),
        monitors=(replace(program.monitors[0], dft_record_interval=record_interval),),
    )
    calls = []

    def fake_run_program_steps(
        chunk_state, _context, _coefficients, groups, monitors, nsteps
    ):
        calls.append((groups, monitors, nsteps))
        return chunk_state

    monkeypatch.setattr(
        "beamz.simulation.cuda.run_program_steps",
        fake_run_program_steps,
    )

    build_scan(program)(state, program.coefficients)

    assert len(calls) == 1
    assert calls[0][1] is not None
    assert calls[0][2] == program.config.num_steps


def test_cuda_cpml_multi_step_ffi_aliases_fields_and_psi(monkeypatch):
    program, state, context = _program_and_state(cpml=True)
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            return (*arguments[:6], *arguments[31:37], *arguments[62:68])

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)

    next_state = cuda_runtime.run_steps(state, context, program.coefficients, 7)

    target, results, options, arguments, attributes = captured[0]
    assert target == abi.CUDA_PROGRAM_TARGET
    assert len(results) == 18
    assert len(arguments) == 74
    assert options["input_output_aliases"] == {
        **{index: index for index in range(6)},
        **{31 + index: 6 + index for index in range(6)},
        **{62 + index: 12 + index for index in range(6)},
    }
    assert attributes["nsteps"] == np.int32(7)
    assert attributes["program_layout"] == np.int32(abi.PROGRAM_LAYOUT_CPML_IN_PLACE)
    assert next_state.cpml_psi_h_terms == state.cpml_psi_h_terms
    assert next_state.cpml_psi_e_terms == state.cpml_psi_e_terms


def test_cuda_cpml_multi_step_reuses_temporal_empty_source_graph(monkeypatch):
    program, state, context = _program_and_state(cpml=True)
    coefficients = program.coefficients._replace(
        h_decay_x=jnp.asarray(1.0, dtype=jnp.float32),
        h_decay_y=jnp.asarray(1.0, dtype=jnp.float32),
        h_decay_z=jnp.asarray(1.0, dtype=jnp.float32),
        h_source_x=jnp.asarray(1.0, dtype=jnp.float32),
        h_source_y=jnp.asarray(1.0, dtype=jnp.float32),
        h_source_z=jnp.asarray(1.0, dtype=jnp.float32),
        e_source_x=jnp.zeros((1,), dtype=jnp.int32),
        e_source_y=jnp.zeros((1,), dtype=jnp.int32),
        e_source_z=jnp.zeros((1,), dtype=jnp.int32),
    )
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            return (
                *arguments[:6],
                *arguments[74:80],
                *arguments[31:37],
                *arguments[62:68],
                *arguments[80:92],
            )

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)

    next_state = cuda_runtime.run_steps(state, context, coefficients, 3)

    target, results, options, arguments, attributes = captured[0]
    assert target == abi.CUDA_PROGRAM_TARGET
    assert len(results) == 36
    assert len(arguments) == 120
    assert all(arguments[index].shape[0] == 0 for index in range(92, 119, 3))
    assert options["input_output_aliases"][74] == 6
    assert attributes["coincident_source_group_mask"] == np.int32(0)
    assert attributes["program_layout"] == np.int32(
        abi.PROGRAM_LAYOUT_SOURCE_TEMPORAL_CPML
    )
    assert next_state.hx is arguments[74]


def test_cuda_source_group_graph_packs_all_phases_and_aliases_state(monkeypatch):
    program, state, context = _program_and_state(cpml=True)
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            return (*arguments[:6], *arguments[31:37], *arguments[62:68])

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)
    source_group = SimpleNamespace(
        coeffs=jnp.ones((2, 3, 4, 5), dtype=jnp.float32),
        waveforms=jnp.ones((2, 8), dtype=jnp.float32),
        starts=jnp.zeros((2, 3), dtype=jnp.int32),
        starts_tuple=((0, 0, 0), (0, 0, 0)),
    )
    groups = (source_group, None, None, None, None, None, None, None, None)

    next_state = cuda_runtime.run_source_group_steps(
        state, context, program.coefficients, groups, 3
    )

    target, results, options, arguments, attributes = captured[0]
    assert target == abi.CUDA_PROGRAM_TARGET
    assert len(results) == 18
    assert len(arguments) == 102
    assert arguments[74] is source_group.coeffs
    assert arguments[75] is source_group.waveforms
    assert arguments[76] is source_group.starts
    assert arguments[101] is state.current_step
    assert options["input_output_aliases"] == {
        **{index: index for index in range(6)},
        **{31 + index: 6 + index for index in range(6)},
        **{62 + index: 12 + index for index in range(6)},
    }
    assert attributes["nsteps"] == np.int32(3)
    assert attributes["cpml_enabled"] == np.int32(1)
    assert attributes["program_layout"] == np.int32(abi.PROGRAM_LAYOUT_SOURCE_IN_PLACE)
    assert attributes["coincident_source_group_mask"] == np.int32(1)
    assert next_state.cpml_psi_h_terms == state.cpml_psi_h_terms


def test_cuda_source_group_graph_uses_temporal_cpml_field_banks(monkeypatch):
    program, state, context = _program_and_state(cpml=True)
    coefficients = program.coefficients._replace(
        h_decay_x=jnp.asarray(1.0, dtype=jnp.float32),
        h_decay_y=jnp.asarray(1.0, dtype=jnp.float32),
        h_decay_z=jnp.asarray(1.0, dtype=jnp.float32),
        h_source_x=jnp.asarray(1.0, dtype=jnp.float32),
        h_source_y=jnp.asarray(1.0, dtype=jnp.float32),
        h_source_z=jnp.asarray(1.0, dtype=jnp.float32),
        e_source_x=jnp.zeros((1,), dtype=jnp.int32),
        e_source_y=jnp.zeros((1,), dtype=jnp.int32),
        e_source_z=jnp.zeros((1,), dtype=jnp.int32),
    )
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            return (
                *arguments[:6],
                *arguments[74:80],
                *arguments[31:37],
                *arguments[62:68],
                *arguments[80:92],
            )

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)

    next_state = cuda_runtime.run_source_group_steps(
        state, context, coefficients, (None,) * 9, 3
    )

    target, results, options, arguments, attributes = captured[0]
    assert target == abi.CUDA_PROGRAM_TARGET
    assert len(results) == 36
    assert len(arguments) == 120
    assert arguments[119] is state.current_step
    assert options["input_output_aliases"] == {
        **{index: index for index in range(6)},
        **{74 + index: 6 + index for index in range(6)},
        **{31 + index: 12 + index for index in range(6)},
        **{62 + index: 18 + index for index in range(6)},
        **{80 + index: 24 + index for index in range(12)},
    }
    assert attributes["cpml_enabled"] == np.int32(1)
    assert attributes["program_layout"] == np.int32(
        abi.PROGRAM_LAYOUT_SOURCE_TEMPORAL_CPML
    )
    assert attributes["nsteps"] == np.int32(3)
    assert next_state.hx is arguments[74]
    assert next_state.cpml_psi_h_terms == arguments[80:86]


def test_cuda_source_group_graph_requires_all_phase_component_slots():
    program, state, context = _program_and_state(cpml=False)

    with np.testing.assert_raises_regex(ValueError, "nine phase/component groups"):
        cuda_runtime.run_source_group_steps(
            state, context, program.coefficients, (None,) * 8, 3
        )


def test_cuda_program_graph_packs_monitor_batch_and_aliases_accumulators(monkeypatch):
    program, state, context = _program_and_state(cpml=True, source=True, monitor=True)
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            return (
                *arguments[:6],
                *arguments[31:37],
                *arguments[62:68],
                *arguments[108:111],
            )

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)
    source_group = SimpleNamespace(
        coeffs=jnp.ones((1, 2, 3, 4), dtype=jnp.float32),
        waveforms=jnp.ones((1, 8), dtype=jnp.float32),
        starts=jnp.zeros((1, 3), dtype=jnp.int32),
        starts_tuple=((0, 0, 0),),
    )
    groups = (source_group, None, None, None, None, None, None, None, None)
    packed = cuda_runtime.pack_dft_monitors(program.monitors)

    next_state = cuda_runtime.run_program_steps(
        state, context, program.coefficients, groups, packed, 3
    )

    target, results, options, arguments, attributes = captured[0]
    assert target == abi.CUDA_PROGRAM_TARGET
    assert len(results) == 21
    assert len(arguments) == 113
    assert packed[0].shape[:2] == (1, 6)
    assert options["input_output_aliases"][108] == 18
    assert options["input_output_aliases"][109] == 19
    assert options["input_output_aliases"][110] == 20
    assert attributes["monitor_count"] == np.int32(1)
    assert attributes["program_layout"] == np.int32(abi.PROGRAM_LAYOUT_MONITOR_IN_PLACE)
    assert attributes["coincident_source_group_mask"] == np.int32(1)
    np.testing.assert_array_equal(next_state.dft_vec_re, state.dft_vec_re)


def test_cuda_program_graph_uses_temporal_cpml_field_banks(monkeypatch):
    program, state, context = _program_and_state(cpml=True, source=True, monitor=True)
    coefficients = program.coefficients._replace(
        h_decay_x=jnp.asarray(1.0, dtype=jnp.float32),
        h_decay_y=jnp.asarray(1.0, dtype=jnp.float32),
        h_decay_z=jnp.asarray(1.0, dtype=jnp.float32),
        h_source_x=jnp.asarray(1.0, dtype=jnp.float32),
        h_source_y=jnp.asarray(1.0, dtype=jnp.float32),
        h_source_z=jnp.asarray(1.0, dtype=jnp.float32),
        e_source_x=jnp.zeros((1,), dtype=jnp.int32),
        e_source_y=jnp.zeros((1,), dtype=jnp.int32),
        e_source_z=jnp.zeros((1,), dtype=jnp.int32),
    )
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            return (
                *arguments[:6],
                *arguments[74:80],
                *arguments[31:37],
                *arguments[62:68],
                *arguments[80:92],
                *arguments[126:129],
            )

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)
    source_group = SimpleNamespace(
        coeffs=jnp.ones((1, 2, 3, 4), dtype=jnp.float32),
        waveforms=jnp.ones((1, 8), dtype=jnp.float32),
        starts=jnp.zeros((1, 3), dtype=jnp.int32),
        starts_tuple=((0, 0, 0),),
    )
    groups = (source_group, None, None, None, None, None, None, None, None)
    packed = cuda_runtime.pack_dft_monitors(program.monitors)

    next_state = cuda_runtime.run_program_steps(
        state, context, coefficients, groups, packed, 3
    )

    target, results, options, arguments, attributes = captured[0]
    assert target == abi.CUDA_PROGRAM_TARGET
    assert len(results) == 39
    assert len(arguments) == 131
    assert arguments[130] is state.current_step
    assert options["input_output_aliases"] == {
        **{index: index for index in range(6)},
        **{74 + index: 6 + index for index in range(6)},
        **{31 + index: 12 + index for index in range(6)},
        **{62 + index: 18 + index for index in range(6)},
        **{80 + index: 24 + index for index in range(12)},
        126: 36,
        127: 37,
        128: 38,
    }
    assert attributes["cpml_enabled"] == np.int32(1)
    assert attributes["monitor_count"] == np.int32(1)
    assert attributes["program_layout"] == np.int32(
        abi.PROGRAM_LAYOUT_MONITOR_TEMPORAL_CPML
    )
    assert next_state.hx is arguments[74]
    np.testing.assert_array_equal(next_state.dft_vec_re, state.dft_vec_re)


def test_cuda_backend_selects_hybrid_jax_orchestration_kernel():
    _program, _state, context = _program_and_state(cpml=True)

    selected = kernels.select_update_kernel(context)

    assert selected.kind == "cuda_streamed"
    assert selected.update_h is cuda_runtime.update_h
    assert selected.update_e is cuda_runtime.update_e


def test_hopper_backend_uses_sm90_tiled_target(monkeypatch):
    program, state, context = _program_and_state(cpml=False)
    context = replace(context, config=replace(context.config, backend="cuda_hopper"))
    targets = []

    def fake_ffi_call(target, result_metadata, **options):
        del result_metadata, options
        targets.append(target)
        return lambda *arguments, **attributes: arguments[:3]

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)

    cuda_runtime.update_h(state, context, program.coefficients)

    assert targets == ["beamz_cuda_hopper"]


def test_uniform_cuda_coefficients_are_compacted_without_rounding():
    uniform = jnp.full((3, 4, 5), np.float32(1.25))
    varied = uniform.at[1, 2, 3].set(np.nextafter(np.float32(1.25), np.float32(2.0)))

    compact = _elide_uniform_grid(uniform)

    assert compact.shape == ()
    assert float(compact) == 1.25
    assert _elide_uniform_grid(varied).shape == varied.shape
