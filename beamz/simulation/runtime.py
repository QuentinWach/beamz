import math
import os
from types import SimpleNamespace

import jax.numpy as jnp

from beamz.arrays import stack_host, to_host, to_scalar
from beamz.devices.monitors.monitors import Monitor
from beamz.simulation.compiled import (
    EngineState,
    MonitorState,
    compile_simulation,
    monitor_dft_point_size,
    monitor_frequency_size,
    monitor_state_size,
)


def compile_program(sim, num_steps=None):
    """Compile the packed-data simulation program."""
    spec = sim.spec
    state = sim.runtime
    if num_steps is None:
        num_steps = state.num_steps - state.current_step
    num_steps = int(num_steps)
    if num_steps <= 0:
        raise ValueError("num_steps must be > 0")

    loop_kind_env = os.getenv("BEAMZ_COMPILED_LOOP_KIND", "scan").strip().lower()
    if loop_kind_env in {"fori", "fori_loop", "fori-loop"}:
        loop_kind = "fori_loop"
    elif loop_kind_env == "scan":
        loop_kind = "scan"
    else:
        raise ValueError("Invalid BEAMZ_COMPILED_LOOP_KIND (use: scan, fori_loop).")
    e_shell_split = os.getenv("BEAMZ_ENABLE_E_SHELL_SPLIT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    h_shell_split = os.getenv("BEAMZ_ENABLE_H_SHELL_SPLIT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    source_single_slab_dense = os.getenv(
        "BEAMZ_SOURCE_SINGLE_SLAB_DENSE", ""
    ).strip().lower() in {"1", "true", "yes", "on"}

    signature = (
        num_steps,
        state.fields.permittivity.shape,
        spec.is_3d,
        spec.plane_2d,
        loop_kind,
        e_shell_split,
        h_shell_split,
        source_single_slab_dense,
    )
    cached = state.compiled_program_cache.get(signature)
    if cached is not None:
        state.compiled_program = cached
        state.compiled_program_signature = signature
        return cached

    run_cfg = SimpleNamespace(
        fields=state.fields,
        resolution=spec.resolution,
        dt=state.dt,
        num_steps=num_steps,
        plane_2d=spec.plane_2d,
        is_3d=spec.is_3d,
        total_steps=state.num_steps,
        t0=float(spec.time[0]),
        precision="float32",
        loop_kind=loop_kind,
        source_single_slab_dense=source_single_slab_dense,
    )
    program = compile_simulation(
        design=spec.design,
        devices=spec.devices,
        boundaries=spec.boundaries,
        run_cfg=run_cfg,
    )
    state.compiled_program_cache[signature] = program
    state.compiled_program = program
    state.compiled_program_signature = signature
    return program


def _make_chunk_monitor_state(sim, program):
    state = sim.runtime
    if (
        state.compiled_monitor_state is not None
        and program.monitor_specs
        and int(state.compiled_monitor_state.counts.shape[0]) == len(program.monitor_specs)
    ):
        return state.compiled_monitor_state
    if program.monitor_specs:
        records_horizon = max(1, int(state.num_steps - state.current_step))
        max_records = max(1, monitor_state_size(program.monitor_specs, records_horizon))
        max_freq = monitor_frequency_size(program.monitor_specs)
        max_points = monitor_dft_point_size(program.monitor_specs)
        return MonitorState(
            powers=jnp.zeros((len(program.monitor_specs), max_records), dtype=jnp.float32),
            timestamps=jnp.zeros((len(program.monitor_specs), max_records), dtype=jnp.float32),
            counts=jnp.zeros((len(program.monitor_specs),), dtype=jnp.int32),
            freq_flux_re=jnp.zeros((len(program.monitor_specs), max_freq), dtype=jnp.float32),
            freq_flux_im=jnp.zeros((len(program.monitor_specs), max_freq), dtype=jnp.float32),
            freq_phase_re=jnp.ones((len(program.monitor_specs), max_freq), dtype=jnp.float32),
            freq_phase_im=jnp.zeros((len(program.monitor_specs), max_freq), dtype=jnp.float32),
            dft_vec_re=jnp.zeros(
                (len(program.monitor_specs), 6, max_freq, max_points),
                dtype=jnp.float32,
            ),
            dft_vec_im=jnp.zeros(
                (len(program.monitor_specs), 6, max_freq, max_points),
                dtype=jnp.float32,
            ),
            dft_weight_sum=jnp.zeros((len(program.monitor_specs), max_freq), dtype=jnp.float32),
        )
    return MonitorState(
        powers=jnp.zeros((0, 0), dtype=jnp.float32),
        timestamps=jnp.zeros((0, 0), dtype=jnp.float32),
        counts=jnp.zeros((0,), dtype=jnp.int32),
        freq_flux_re=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_flux_im=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_phase_re=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_phase_im=jnp.zeros((0, 0), dtype=jnp.float32),
        dft_vec_re=jnp.zeros((0, 0, 0, 0), dtype=jnp.float32),
        dft_vec_im=jnp.zeros((0, 0, 0, 0), dtype=jnp.float32),
        dft_weight_sum=jnp.zeros((0, 0), dtype=jnp.float32),
    )


def run_compiled(sim, num_steps=None, record_interval=None, record_fields=None, progress=True):
    """Run the simulation using the compiled scan engine."""
    state = sim.runtime
    if num_steps is None:
        num_steps = state.num_steps - state.current_step
    num_steps = int(num_steps)
    if num_steps <= 0:
        return None

    if record_fields is None:
        record_fields = ["Ez"]

    record_every = int(record_interval) if record_interval else None
    if record_every is not None and record_every <= 0:
        raise ValueError("record_interval must be a positive integer")

    field_history = {name: [] for name in record_fields} if record_every else None
    if state.current_step == 0:
        state.compiled_monitor_state = None

    chunk_size = record_every if record_every else num_steps
    steps_remaining = num_steps
    steps_done = 0
    monitor_state = None

    while steps_remaining > 0:
        this_chunk = min(chunk_size, steps_remaining)
        program = sim.compile(num_steps=this_chunk)

        if progress and steps_done == 0 and program.compile_count == 0:
            print("● JIT compiling v0.3 packed FDTD program...", end=" ", flush=True)

        engine_state = EngineState(
            ex=state.fields.Ex,
            ey=state.fields.Ey,
            ez=state.fields.Ez,
            hx=state.fields.Hx,
            hy=state.fields.Hy,
            hz=state.fields.Hz,
            t=jnp.asarray(state.t, dtype=jnp.float32),
            current_step=jnp.asarray(state.current_step, dtype=jnp.int32),
        )

        if monitor_state is None:
            monitor_state = _make_chunk_monitor_state(sim, program)
        state.compiled_monitor_state = monitor_state

        engine_state, monitor_state, _ = program.run(
            engine_state=engine_state,
            monitor_state=monitor_state,
        )
        engine_state.ez.block_until_ready()
        state.compiled_monitor_state = monitor_state

        if progress and steps_done == 0:
            print("done!")

        state.fields.Ex = engine_state.ex
        state.fields.Ey = engine_state.ey
        state.fields.Ez = engine_state.ez
        state.fields.Hx = engine_state.hx
        state.fields.Hy = engine_state.hy
        state.fields.Hz = engine_state.hz
        state.t = to_scalar(engine_state.t, cast=float)
        state.current_step = to_scalar(engine_state.current_step, cast=int)

        if field_history is not None and (state.current_step % record_every == 0):
            for name in record_fields:
                if hasattr(state.fields, name):
                    field_history[name].append(
                        to_host(getattr(state.fields, name), copy=True)
                    )

        steps_done += this_chunk
        steps_remaining -= this_chunk

        if progress and num_steps > 0:
            pct = 100.0 * steps_done / num_steps
            print(
                f"\r● Progress: {pct:.0f}% ({steps_done}/{num_steps} steps)",
                end="",
                flush=True,
            )

    if progress:
        print()

    if monitor_state is not None:
        program.apply_monitor_state(monitor_state)

    result = {}
    if field_history is not None:
        result["fields"] = {key: stack_host(values) for key, values in field_history.items()}
    monitors = [device for device in sim.spec.devices if isinstance(device, Monitor)]
    if monitors:
        result["monitors"] = monitors
    return result if result else None


def run_compiled_until_decay(
    sim,
    monitors,
    *,
    min_time_s=0.0,
    chunk_steps=None,
    lookback_records=12,
    decay_ratio=1e-3,
    progress=True,
):
    """Run compiled chunks until monitor power decays after a minimum time."""
    state = sim.runtime
    total_steps = int(state.num_steps - state.current_step)
    if total_steps <= 0:
        return 0
    dt = float(state.dt)
    chunk_steps = (
        max(64, min(512, math.ceil(total_steps / 24.0)))
        if chunk_steps is None
        else max(1, int(chunk_steps))
    )
    lookback_records = max(2, int(lookback_records))
    min_steps = math.ceil(max(0.0, float(min_time_s)) / max(dt, 1e-30))
    steps_done = 0
    peak = 0.0

    while steps_done < total_steps:
        this_chunk = min(chunk_steps, total_steps - steps_done)
        sim.run_compiled(num_steps=this_chunk, progress=False)
        steps_done += this_chunk

        histories = [
            abs(to_host(mon.power_history, dtype=float))
            for mon in monitors
            if len(mon.power_history)
        ]
        tail = math.inf
        if histories:
            peak = max(peak, max(float(hist.max()) for hist in histories))
            tail = max(float(hist[-lookback_records:].max()) for hist in histories)

        if progress:
            pct = 100.0 * steps_done / max(total_steps, 1)
            print(
                f"\r● Progress: {pct:.0f}% ({steps_done}/{total_steps} steps)",
                end="",
                flush=True,
            )

        if (
            steps_done >= min_steps
            and peak > 0.0
            and math.isfinite(tail)
            and tail <= float(decay_ratio) * peak
        ):
            break

    if progress:
        print()
    return steps_done
