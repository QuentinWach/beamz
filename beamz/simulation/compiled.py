"""v0.3 compiled FDTD engine.

This module provides a packed-data simulation path where one compiled
loop (`jax.lax.scan` or `jax.lax.fori_loop`) performs field updates,
source injection, monitor accumulation, and material model updates.
"""

from __future__ import annotations

import os
import pathlib
import platform
import sys
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from beamz.devices.monitors.compiler import (
    BatchedMonitorData,
    CompiledMonitorSpec,
)
from beamz.devices.monitors.monitors import Monitor
from beamz.devices.sources.compiler import (
    BatchedSlabGroup,
    CompiledSourceSpec,
)
from beamz.simulation import build, loop, shell, sources, track
from beamz.simulation.material_models import (
    CompiledMaterialSpec,
    MaterialState,
)


def _init_persistent_cache():
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
    if os.environ.get("JAX_COMPILATION_CACHE_DIR"):
        return
    try:
        from jax.experimental.compilation_cache import compilation_cache as cc

        cc.set_cache_dir(cache_dir)
    except Exception:
        jax.config.update("jax_compilation_cache_dir", cache_dir)


_init_persistent_cache()


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
    freq_flux_re: jnp.ndarray
    freq_flux_im: jnp.ndarray
    freq_phase_re: jnp.ndarray
    freq_phase_im: jnp.ndarray
    dft_vec_re: jnp.ndarray
    dft_vec_im: jnp.ndarray
    dft_weight_sum: jnp.ndarray


class UpdateCoefficients(NamedTuple):
    """Static update coefficients passed as runtime arguments to avoid constant capture."""

    h_decay_x: jnp.ndarray
    h_source_x: jnp.ndarray
    h_source_lossless_x: jnp.ndarray
    h_decay_y: jnp.ndarray
    h_source_y: jnp.ndarray
    h_source_lossless_y: jnp.ndarray
    h_decay_z: jnp.ndarray
    h_source_z: jnp.ndarray
    h_source_lossless_z: jnp.ndarray
    e_decay_x: jnp.ndarray
    e_source_x: jnp.ndarray
    e_source_lossless_x: jnp.ndarray
    e_decay_y: jnp.ndarray
    e_source_y: jnp.ndarray
    e_source_lossless_y: jnp.ndarray
    e_decay_z: jnp.ndarray
    e_source_z: jnp.ndarray
    e_source_lossless_z: jnp.ndarray


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
    loop_kind: str = "scan"
    source_single_slab_dense: bool = False


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
    h_source_lossless_x: jnp.ndarray
    h_decay_y: jnp.ndarray
    h_source_y: jnp.ndarray
    h_source_lossless_y: jnp.ndarray
    h_decay_z: jnp.ndarray
    h_source_z: jnp.ndarray
    h_source_lossless_z: jnp.ndarray
    e_decay_x: jnp.ndarray
    e_source_x: jnp.ndarray
    e_source_lossless_x: jnp.ndarray
    e_decay_y: jnp.ndarray
    e_source_y: jnp.ndarray
    e_source_lossless_y: jnp.ndarray
    e_decay_z: jnp.ndarray
    e_source_z: jnp.ndarray
    e_source_lossless_z: jnp.ndarray

    # Optional boundary-shell slabs where lossy update differs from lossless one.
    e_use_lossy_shell_x: bool
    e_lossy_shell_x: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]
    e_use_lossy_shell_y: bool
    e_lossy_shell_y: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]
    e_use_lossy_shell_z: bool
    e_lossy_shell_z: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]
    h_use_lossy_shell_x: bool
    h_lossy_shell_x: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]
    h_use_lossy_shell_y: bool
    h_lossy_shell_y: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]
    h_use_lossy_shell_z: bool
    h_lossy_shell_z: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]

    _compiled_scan: callable | None = None
    _compile_count: int = 0

    def _update_coefficients(self) -> UpdateCoefficients:
        """Build runtime coefficient container for jitted scan entrypoint."""
        return UpdateCoefficients(
            h_decay_x=self.h_decay_x,
            h_source_x=self.h_source_x,
            h_source_lossless_x=self.h_source_lossless_x,
            h_decay_y=self.h_decay_y,
            h_source_y=self.h_source_y,
            h_source_lossless_y=self.h_source_lossless_y,
            h_decay_z=self.h_decay_z,
            h_source_z=self.h_source_z,
            h_source_lossless_z=self.h_source_lossless_z,
            e_decay_x=self.e_decay_x,
            e_source_x=self.e_source_x,
            e_source_lossless_x=self.e_source_lossless_x,
            e_decay_y=self.e_decay_y,
            e_source_y=self.e_source_y,
            e_source_lossless_y=self.e_source_lossless_y,
            e_decay_z=self.e_decay_z,
            e_source_z=self.e_source_z,
            e_source_lossless_z=self.e_source_lossless_z,
        )

    def _sources_for(
        self, timing: str, component: str
    ) -> tuple[CompiledSourceSpec, ...]:
        return sources.sources_for(self, timing, component)

    def _apply_specs(
        self,
        arr: jnp.ndarray,
        abs_step: jnp.ndarray,
        specs: tuple[CompiledSourceSpec, ...],
    ) -> jnp.ndarray:
        return sources.apply_specs(self, arr, abs_step, specs)

    def _apply_batched_slabs(
        self,
        arr: jnp.ndarray,
        abs_step: jnp.ndarray,
        group: BatchedSlabGroup,
    ) -> jnp.ndarray:
        return sources.apply_batched_slabs(self, arr, abs_step, group)

    def _apply_source_group(
        self,
        arr: jnp.ndarray,
        abs_step: jnp.ndarray,
        batch: BatchedSlabGroup | None,
        rest: tuple[CompiledSourceSpec, ...],
    ) -> jnp.ndarray:
        return sources.apply_group(self, arr, abs_step, batch, rest)

    def _apply_lossy_shell(
        self,
        updated: jnp.ndarray,
        old: jnp.ndarray,
        curl: jnp.ndarray,
        decay: jnp.ndarray,
        source: jnp.ndarray,
        slabs: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...],
    ) -> jnp.ndarray:
        return shell.apply_lossy_shell(self, updated, old, curl, decay, source, slabs)

    def _apply_lossy_shell_from_lossless(
        self,
        updated_lossless: jnp.ndarray,
        old: jnp.ndarray,
        decay: jnp.ndarray,
        source: jnp.ndarray,
        source_lossless: jnp.ndarray,
        slabs: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...],
    ) -> jnp.ndarray:
        return shell.apply_lossy_shell_from_lossless(
            self,
            updated_lossless,
            old,
            decay,
            source,
            source_lossless,
            slabs,
        )

    def _monitor_power_2d(
        self,
        spec: CompiledMonitorSpec,
        ez: jnp.ndarray,
        hx: jnp.ndarray,
        hy: jnp.ndarray,
    ) -> jnp.ndarray:
        return track.monitor_power_2d(self, spec, ez, hx, hy)

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
        return track.monitor_power_3d(self, spec, ex, ey, ez, hx, hy, hz)

    def _update_monitors(
        self,
        monitor_state: MonitorState,
        abs_step: jnp.ndarray,
        t_phys: jnp.ndarray,
        dt_scalar: jnp.ndarray,
        ex: jnp.ndarray,
        ey: jnp.ndarray,
        ez: jnp.ndarray,
        hx: jnp.ndarray,
        hy: jnp.ndarray,
        hz: jnp.ndarray,
        batched_mon: BatchedMonitorData | None = None,
        monitors_2d: tuple[CompiledMonitorSpec, ...] = (),
    ) -> MonitorState:
        return track.update_monitors(
            self,
            monitor_state,
            abs_step,
            t_phys,
            dt_scalar,
            ex,
            ey,
            ez,
            hx,
            hy,
            hz,
            batched_mon=batched_mon,
            monitors_2d=monitors_2d,
        )

    def _build_scan(self):
        loop.build_scan(self)

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
            monitor_state = track.make_monitor_state(self, MonitorState)

        if self._compiled_scan is None:
            self._build_scan()

        eng, mon, mat = self._compiled_scan(
            engine_state,
            monitor_state,
            self._update_coefficients(),
        )
        return eng, mon, mat

    def apply_monitor_state(self, monitor_state: MonitorState):
        """Push monitor-state buffers back to Monitor objects."""
        track.apply_monitor_state(self, monitor_state)


def monitor_state_size(specs: tuple[CompiledMonitorSpec, ...], num_steps: int) -> int:
    return track.monitor_state_size(specs, num_steps)


def monitor_frequency_size(specs: tuple[CompiledMonitorSpec, ...]) -> int:
    return track.monitor_frequency_size(specs)


def monitor_dft_point_size(specs: tuple[CompiledMonitorSpec, ...]) -> int:
    return track.monitor_dft_point_size(specs)


def _edge_full_thickness(mask, axis: int) -> tuple[int, int]:
    return shell.edge_full_thickness(mask, axis)


def _region_offsets_and_sizes(
    field_shape: tuple[int, ...],
    region: tuple[slice, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    return shell.region_offsets_and_sizes(field_shape, region)


def _infer_lossy_shell_slabs(
    field_shape: tuple[int, ...],
    region: tuple[slice, ...],
    conductivity_region: jnp.ndarray,
) -> tuple[bool, tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]]:
    return shell.infer_lossy_shell_slabs(field_shape, region, conductivity_region)


def _lossy_fraction(
    field_shape: tuple[int, ...],
    region: tuple[slice, ...],
    conductivity_region: jnp.ndarray,
) -> float:
    return shell.lossy_fraction(field_shape, region, conductivity_region)


def compile_simulation(design, devices, boundaries, run_cfg) -> CompiledSimulation:
    return build.compile_simulation(
        design,
        devices,
        boundaries,
        run_cfg,
        compiled_cls=CompiledSimulation,
        config_cls=CompiledRunConfig,
    )
