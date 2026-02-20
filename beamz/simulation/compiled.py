"""v0.3 compiled FDTD engine.

This module provides a packed-data simulation path where one compiled
`jax.lax.scan` step performs field updates, source injection, monitor
accumulation, and material model updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from beamz.devices.monitors.compiler import CompiledMonitorSpec, compile_monitor_specs
from beamz.devices.monitors.monitors import Monitor
from beamz.devices.sources.compiler import CompiledSourceSpec, compile_source_specs
from beamz.simulation import ops
from beamz.simulation.material_models import (
    CompiledMaterialSpec,
    MaterialState,
    create_material_model,
)


class EngineState(NamedTuple):
    """Runtime EM field state."""

    ex: jnp.ndarray
    ey: jnp.ndarray
    ez: jnp.ndarray
    hx: jnp.ndarray
    hy: jnp.ndarray
    hz: jnp.ndarray
    t: jnp.ndarray
    current_step: jnp.ndarray


class MonitorState(NamedTuple):
    """Packed monitor accumulators."""

    powers: jnp.ndarray
    timestamps: jnp.ndarray
    counts: jnp.ndarray


class RunState(NamedTuple):
    """Auxiliary run counters."""

    compile_count: jnp.ndarray


@dataclass(frozen=True)
class CompiledRunConfig:
    """Static compiled run configuration."""

    resolution: float
    dt: float
    num_steps: int
    plane_2d: str
    is_3d: bool
    precision: str = "float32"


@dataclass
class CompiledSimulation:
    """Compiled simulation program and packed static specs."""

    config: CompiledRunConfig
    material_spec: CompiledMaterialSpec
    source_specs: tuple[CompiledSourceSpec, ...]
    monitor_specs: tuple[CompiledMonitorSpec, ...]
    monitor_devices: tuple[Monitor, ...]

    # Static update coefficients (full-grid, dense updates; no per-step scatters)
    h_decay_x: jnp.ndarray
    h_source_x: jnp.ndarray
    h_decay_y: jnp.ndarray
    h_source_y: jnp.ndarray
    h_decay_z: jnp.ndarray
    h_source_z: jnp.ndarray
    e_decay_x: jnp.ndarray
    e_source_x: jnp.ndarray
    e_decay_y: jnp.ndarray
    e_source_y: jnp.ndarray
    e_decay_z: jnp.ndarray
    e_source_z: jnp.ndarray

    _compiled_scan: callable | None = None
    _compile_count: int = 0

    def _sources_for(self, timing: str, component: str) -> tuple[CompiledSourceSpec, ...]:
        return tuple(
            s for s in self.source_specs if s.timing == timing and s.component == component
        )

    def _apply_specs(
        self,
        arr: jnp.ndarray,
        t_idx: jnp.ndarray,
        specs: tuple[CompiledSourceSpec, ...],
    ) -> jnp.ndarray:
        out = arr
        for spec in specs:
            amp = spec.waveform[t_idx]
            out = out.at[spec.index].add(spec.coeff * amp)
        return out

    def _monitor_power_2d(
        self,
        spec: CompiledMonitorSpec,
        ez: jnp.ndarray,
        hx: jnp.ndarray,
        hy: jnp.ndarray,
    ) -> jnp.ndarray:
        ez_vals = ez[spec.y_ez, spec.x_ez] * spec.valid_ez
        hx_vals = hx[spec.y_hx, spec.x_hx] * spec.valid_hx
        hy_vals = hy[spec.y_hy, spec.x_hy] * spec.valid_hy

        sx = -ez_vals * hy_vals
        sy = ez_vals * hx_vals
        mag = jnp.sqrt(sx * sx + sy * sy)
        return jnp.sum(mag) * spec.power_scale

    def _monitor_power_3d(
        self,
        spec: CompiledMonitorSpec,
        ex: jnp.ndarray,
        ey: jnp.ndarray,
        ez: jnp.ndarray,
        hx: jnp.ndarray,
        hy: jnp.ndarray,
        hz: jnp.ndarray,
    ) -> jnp.ndarray:
        exs = ex[spec.ex_idx][: spec.min_dim0, : spec.min_dim1]
        eys = ey[spec.ey_idx][: spec.min_dim0, : spec.min_dim1]
        ezs = ez[spec.ez_idx][: spec.min_dim0, : spec.min_dim1]
        hxs = hx[spec.hx_idx][: spec.min_dim0, : spec.min_dim1]
        hys = hy[spec.hy_idx][: spec.min_dim0, : spec.min_dim1]
        hzs = hz[spec.hz_idx][: spec.min_dim0, : spec.min_dim1]

        sx = eys * hzs - ezs * hys
        sy = ezs * hxs - exs * hzs
        sz = exs * hys - eys * hxs
        mag = jnp.sqrt(sx * sx + sy * sy + sz * sz)
        return jnp.sum(mag) * spec.power_scale

    def _update_monitors(
        self,
        monitor_state: MonitorState,
        t_idx: jnp.ndarray,
        t_phys: jnp.ndarray,
        ex: jnp.ndarray,
        ey: jnp.ndarray,
        ez: jnp.ndarray,
        hx: jnp.ndarray,
        hy: jnp.ndarray,
        hz: jnp.ndarray,
    ) -> MonitorState:
        if not self.monitor_specs:
            return monitor_state

        powers = monitor_state.powers
        timestamps = monitor_state.timestamps
        counts = monitor_state.counts
        max_records = powers.shape[1]

        for mon in self.monitor_specs:
            should_record = (t_idx % mon.record_interval) == 0
            can_record = counts[mon.monitor_index] < max_records
            do_record = should_record & can_record & mon.accumulate_power

            power_val = jnp.where(
                do_record,
                self._monitor_power_3d(mon, ex, ey, ez, hx, hy, hz)
                if mon.is_3d
                else self._monitor_power_2d(mon, ez, hx, hy),
                jnp.array(0.0, dtype=jnp.float32),
            )

            slot = jnp.minimum(counts[mon.monitor_index], max_records - 1)
            old_power = powers[mon.monitor_index, slot]
            old_ts = timestamps[mon.monitor_index, slot]

            powers = powers.at[mon.monitor_index, slot].set(
                jnp.where(do_record, power_val, old_power)
            )
            timestamps = timestamps.at[mon.monitor_index, slot].set(
                jnp.where(do_record, t_phys, old_ts)
            )
            counts = counts.at[mon.monitor_index].set(
                counts[mon.monitor_index] + jnp.where(do_record, 1, 0)
            )

        return MonitorState(powers=powers, timestamps=timestamps, counts=counts)

    def _build_scan(self):
        material_model = create_material_model(self.material_spec)
        material_state0 = material_model.init_state(self.material_spec)

        resolution = float(self.config.resolution)
        dt = float(self.config.dt)
        plane_2d = self.config.plane_2d
        is_3d = self.config.is_3d

        pre_e_ex = self._sources_for("pre_e", "Ex")
        pre_e_ey = self._sources_for("pre_e", "Ey")
        pre_e_ez = self._sources_for("pre_e", "Ez")

        h_specs_x = self._sources_for("h", "Hx")
        h_specs_y = self._sources_for("h", "Hy")
        h_specs_z = self._sources_for("h", "Hz")

        e_specs_x = self._sources_for("e", "Ex")
        e_specs_y = self._sources_for("e", "Ey")
        e_specs_z = self._sources_for("e", "Ez")

        h_decay_x, h_source_x = self.h_decay_x, self.h_source_x
        h_decay_y, h_source_y = self.h_decay_y, self.h_source_y
        h_decay_z, h_source_z = self.h_decay_z, self.h_source_z
        e_decay_x, e_source_x = self.e_decay_x, self.e_source_x
        e_decay_y, e_source_y = self.e_decay_y, self.e_source_y
        e_decay_z, e_source_z = self.e_decay_z, self.e_source_z

        def body(carry, t_idx):
            eng, mon, mat = carry

            ex, ey, ez = eng.ex, eng.ey, eng.ez
            hx, hy, hz = eng.hx, eng.hy, eng.hz

            # Optional pre-E injections (legacy soft electric sources packed as data).
            ex = self._apply_specs(ex, t_idx, pre_e_ex)
            ey = self._apply_specs(ey, t_idx, pre_e_ey)
            ez = self._apply_specs(ez, t_idx, pre_e_ez)

            if is_3d:
                curl_ex, curl_ey, curl_ez = ops.curl_e_to_h_3d(ex, ey, ez, resolution)
            else:
                curl_ex, curl_ey, curl_ez = ops.curl_e_to_h_2d(
                    (ex, ey, ez),
                    resolution,
                    plane=plane_2d,
                )

            hx = h_decay_x * hx - h_source_x * curl_ex
            hy = h_decay_y * hy - h_source_y * curl_ey
            hz = h_decay_z * hz - h_source_z * curl_ez

            hx = self._apply_specs(hx, t_idx, h_specs_x)
            hy = self._apply_specs(hy, t_idx, h_specs_y)
            hz = self._apply_specs(hz, t_idx, h_specs_z)

            if is_3d:
                curl_hx, curl_hy, curl_hz = ops.curl_h_to_e_3d(
                    hx,
                    hy,
                    hz,
                    resolution,
                    ex_shape=ex.shape,
                    ey_shape=ey.shape,
                    ez_shape=ez.shape,
                )
            else:
                curl_hx, curl_hy, curl_hz = ops.curl_h_to_e_2d(
                    (hx, hy, hz),
                    resolution,
                    (ex.shape, ey.shape, ez.shape),
                    plane=plane_2d,
                )

            ex = e_decay_x * ex + e_source_x * curl_hx
            ey = e_decay_y * ey + e_source_y * curl_hy
            ez = e_decay_z * ez + e_source_z * curl_hz

            ex = self._apply_specs(ex, t_idx, e_specs_x)
            ey = self._apply_specs(ey, t_idx, e_specs_y)
            ez = self._apply_specs(ez, t_idx, e_specs_z)

            mat, _ = material_model.update(mat, ex, ey, ez, t_idx)

            t_phys = eng.t
            mon = self._update_monitors(mon, t_idx, t_phys, ex, ey, ez, hx, hy, hz)

            new_eng = EngineState(
                ex=ex,
                ey=ey,
                ez=ez,
                hx=hx,
                hy=hy,
                hz=hz,
                t=eng.t + dt,
                current_step=eng.current_step + jnp.array(1, dtype=jnp.int32),
            )
            return (new_eng, mon, mat), None

        @jax.jit
        def run_scan(engine_state: EngineState, monitor_state: MonitorState):
            t_idxs = jnp.arange(self.config.num_steps, dtype=jnp.int32)
            (engine_final, monitor_final, material_final), _ = jax.lax.scan(
                body,
                (engine_state, monitor_state, material_state0),
                t_idxs,
            )
            return engine_final, monitor_final, material_final

        self._compiled_scan = run_scan
        self._compile_count += 1

    @property
    def compile_count(self) -> int:
        return self._compile_count

    def run(
        self,
        engine_state: EngineState,
        monitor_state: MonitorState | None = None,
    ) -> tuple[EngineState, MonitorState, MaterialState]:
        """Execute the compiled simulation loop."""
        if monitor_state is None:
            if self.monitor_specs:
                max_records = max(1, monitor_state_size(self.monitor_specs, self.config.num_steps))
                monitor_state = MonitorState(
                    powers=jnp.zeros((len(self.monitor_specs), max_records), dtype=jnp.float32),
                    timestamps=jnp.zeros((len(self.monitor_specs), max_records), dtype=jnp.float32),
                    counts=jnp.zeros((len(self.monitor_specs),), dtype=jnp.int32),
                )
            else:
                monitor_state = MonitorState(
                    powers=jnp.zeros((0, 0), dtype=jnp.float32),
                    timestamps=jnp.zeros((0, 0), dtype=jnp.float32),
                    counts=jnp.zeros((0,), dtype=jnp.int32),
                )

        if self._compiled_scan is None:
            self._build_scan()

        eng, mon, mat = self._compiled_scan(engine_state, monitor_state)
        return eng, mon, mat

    def apply_monitor_state(self, monitor_state: MonitorState):
        """Push monitor-state buffers back to Monitor objects."""
        for spec in self.monitor_specs:
            dev = self.monitor_devices[spec.monitor_index]
            count = int(np.asarray(monitor_state.counts[spec.monitor_index]))
            powers = np.asarray(monitor_state.powers[spec.monitor_index, :count], dtype=float)
            ts = np.asarray(monitor_state.timestamps[spec.monitor_index, :count], dtype=float)

            dev.power_history = list(powers.tolist())
            dev.power_timestamps = list(ts.tolist())
            dev.power_accumulation_count = count


def monitor_state_size(specs: tuple[CompiledMonitorSpec, ...], num_steps: int) -> int:
    if not specs:
        return 0
    return int(
        max(
            int(np.ceil(num_steps / max(1, int(spec.record_interval)))) for spec in specs
        )
    )


def compile_simulation(design, devices, boundaries, run_cfg) -> CompiledSimulation:
    """Build a CompiledSimulation from design/devices/boundaries and a run config.

    Required run_cfg attributes:
    - fields
    - resolution
    - dt
    - num_steps
    - plane_2d
    - is_3d
    - t0
    """
    del design, boundaries

    fields = run_cfg.fields
    resolution = float(run_cfg.resolution)
    dt = float(run_cfg.dt)
    num_steps = int(run_cfg.num_steps)
    t0 = float(run_cfg.t0)

    source_specs = compile_source_specs(
        devices=devices,
        fields=fields,
        dt=dt,
        resolution=resolution,
        num_steps=num_steps,
        t0=t0,
    )

    monitor_specs, _ = compile_monitor_specs(
        devices=devices,
        fields=fields,
        resolution=resolution,
        num_steps=num_steps,
        dt=dt,
    )

    monitor_devices = tuple(d for d in devices if isinstance(d, Monitor))

    config = CompiledRunConfig(
        resolution=resolution,
        dt=dt,
        num_steps=num_steps,
        plane_2d=run_cfg.plane_2d,
        is_3d=bool(run_cfg.is_3d),
        precision=getattr(run_cfg, "precision", "float32"),
    )

    h_decay_x, h_source_x = ops.precompute_h_update_coefficients(fields.sigma_m_hx, dt)
    h_decay_y, h_source_y = ops.precompute_h_update_coefficients(fields.sigma_m_hy, dt)
    h_decay_z, h_source_z = ops.precompute_h_update_coefficients(fields.sigma_m_hz, dt)

    e_decay_x, e_source_x = ops.precompute_e_update_coefficients(
        shape=fields.Ex.shape,
        conductivity=fields.sig_x,
        permittivity=fields.eps_x,
        dt=dt,
        region=fields.region_x,
    )
    e_decay_y, e_source_y = ops.precompute_e_update_coefficients(
        shape=fields.Ey.shape,
        conductivity=fields.sig_y,
        permittivity=fields.eps_y,
        dt=dt,
        region=fields.region_y,
    )
    e_decay_z, e_source_z = ops.precompute_e_update_coefficients(
        shape=fields.Ez.shape,
        conductivity=fields.sig_z,
        permittivity=fields.eps_z,
        dt=dt,
        region=fields.region_z,
    )

    return CompiledSimulation(
        config=config,
        material_spec=CompiledMaterialSpec(model_kind="linear"),
        source_specs=source_specs,
        monitor_specs=monitor_specs,
        monitor_devices=monitor_devices,
        h_decay_x=h_decay_x,
        h_source_x=h_source_x,
        h_decay_y=h_decay_y,
        h_source_y=h_source_y,
        h_decay_z=h_decay_z,
        h_source_z=h_source_z,
        e_decay_x=e_decay_x,
        e_source_x=e_source_x,
        e_decay_y=e_decay_y,
        e_source_y=e_source_y,
        e_decay_z=e_decay_z,
        e_source_z=e_source_z,
    )
