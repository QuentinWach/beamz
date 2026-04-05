import os
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

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
    if num_steps is None:
        num_steps = sim.num_steps - sim.current_step
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
        sim.fields.permittivity.shape,
        sim.is_3d,
        sim.plane_2d,
        loop_kind,
        e_shell_split,
        h_shell_split,
        source_single_slab_dense,
    )
    cached = sim._compiled_program_cache.get(signature)
    if cached is not None:
        sim._compiled_program = cached
        sim._compiled_program_signature = signature
        return cached

    run_cfg = SimpleNamespace(
        fields=sim.fields,
        resolution=sim.resolution,
        dt=sim.dt,
        num_steps=num_steps,
        plane_2d=sim.plane_2d,
        is_3d=sim.is_3d,
        total_steps=sim.num_steps,
        t0=float(sim.time[0]),
        precision="float32",
        loop_kind=loop_kind,
        source_single_slab_dense=source_single_slab_dense,
    )
    program = compile_simulation(
        design=sim.design,
        devices=sim.devices,
        boundaries=sim.boundaries,
        run_cfg=run_cfg,
    )
    sim._compiled_program_cache[signature] = program
    sim._compiled_program = program
    sim._compiled_program_signature = signature
    return program


def _make_chunk_monitor_state(sim, program):
    if (
        sim._compiled_monitor_state is not None
        and program.monitor_specs
        and int(np.asarray(sim._compiled_monitor_state.counts.shape[0]))
        == len(program.monitor_specs)
    ):
        return sim._compiled_monitor_state
    if program.monitor_specs:
        records_horizon = max(1, int(sim.num_steps - sim.current_step))
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
    if sim.thermal is not None and getattr(sim.thermal, "enabled", True):
        raise NotImplementedError("run_compiled currently does not support thermal coupling.")

    if num_steps is None:
        num_steps = sim.num_steps - sim.current_step
    num_steps = int(num_steps)
    if num_steps <= 0:
        return None

    if record_fields is None:
        record_fields = ["Ez"]

    record_every = int(record_interval) if record_interval else None
    if record_every is not None and record_every <= 0:
        raise ValueError("record_interval must be a positive integer")

    field_history = {name: [] for name in record_fields} if record_every else None
    if sim.current_step == 0:
        sim._compiled_monitor_state = None

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
            ex=sim.fields.Ex,
            ey=sim.fields.Ey,
            ez=sim.fields.Ez,
            hx=sim.fields.Hx,
            hy=sim.fields.Hy,
            hz=sim.fields.Hz,
            t=jnp.asarray(sim.t, dtype=jnp.float32),
            current_step=jnp.asarray(sim.current_step, dtype=jnp.int32),
        )

        if monitor_state is None:
            monitor_state = _make_chunk_monitor_state(sim, program)
        sim._compiled_monitor_state = monitor_state

        engine_state, monitor_state, _ = program.run(
            engine_state=engine_state,
            monitor_state=monitor_state,
        )
        engine_state.ez.block_until_ready()
        sim._compiled_monitor_state = monitor_state

        if progress and steps_done == 0:
            print("done!")

        sim.fields.Ex = engine_state.ex
        sim.fields.Ey = engine_state.ey
        sim.fields.Ez = engine_state.ez
        sim.fields.Hx = engine_state.hx
        sim.fields.Hy = engine_state.hy
        sim.fields.Hz = engine_state.hz
        sim.t = float(np.asarray(engine_state.t))
        sim.current_step = int(np.asarray(engine_state.current_step))

        if field_history is not None and (sim.current_step % record_every == 0):
            for name in record_fields:
                if hasattr(sim.fields, name):
                    field_history[name].append(np.array(getattr(sim.fields, name)))

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
        result["fields"] = {
            key: np.stack(values) if len(values) > 0 else np.zeros((0,))
            for key, values in field_history.items()
        }
    monitors = [device for device in sim.devices if isinstance(device, Monitor)]
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
    total_steps = int(sim.num_steps - sim.current_step)
    if total_steps <= 0:
        return 0
    dt = float(sim.dt)
    chunk_steps = (
        max(64, min(512, int(np.ceil(total_steps / 24.0))))
        if chunk_steps is None
        else max(1, int(chunk_steps))
    )
    lookback_records = max(2, int(lookback_records))
    min_steps = int(np.ceil(max(0.0, float(min_time_s)) / max(dt, 1e-30)))
    steps_done = 0
    peak = 0.0

    while steps_done < total_steps:
        this_chunk = min(chunk_steps, total_steps - steps_done)
        sim.run_compiled(num_steps=this_chunk, progress=False)
        steps_done += this_chunk

        histories = [
            np.abs(np.asarray(mon.power_history, dtype=np.float64))
            for mon in monitors
            if len(mon.power_history)
        ]
        tail = np.inf
        if histories:
            peak = max(peak, max(float(np.max(hist)) for hist in histories))
            tail = max(float(np.max(hist[-lookback_records:])) for hist in histories)

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
            and np.isfinite(tail)
            and tail <= float(decay_ratio) * peak
        ):
            break

    if progress:
        print()
    return steps_done


def run_fast(sim, num_steps=None, record_interval=None, record_fields=None, progress=True):
    return sim.run_compiled(
        num_steps=num_steps,
        record_interval=record_interval,
        record_fields=record_fields,
        progress=progress,
    )


def run_jit_scan(sim, num_steps=None, progress=True):
    return sim.run_compiled(
        num_steps=num_steps,
        record_interval=None,
        record_fields=None,
        progress=progress,
    )
