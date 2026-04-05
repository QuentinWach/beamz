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
import numpy as np

from beamz.devices.monitors.compiler import (
    BatchedMonitorData,
    CompiledMonitorSpec,
    compile_batched_monitor_data,
    compile_monitor_specs,
)
from beamz.devices.monitors.monitors import Monitor
from beamz.devices.sources.compiler import (
    BatchedSlabGroup,
    CompiledSourceSpec,
    batch_slab_specs,
    compile_source_specs,
)
from beamz.simulation import ops
from beamz.simulation import shell, sources, track
from beamz.simulation.material_models import (
    CompiledMaterialSpec,
    MaterialState,
    create_material_model,
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
        material_model = create_material_model(self.material_spec)
        material_state0 = material_model.init_state(self.material_spec)

        resolution = float(self.config.resolution)
        dt = float(self.config.dt)
        dt_scalar = jnp.asarray(dt, dtype=jnp.float32)
        plane_2d = self.config.plane_2d
        is_3d = self.config.is_3d

        # Batch slab sources by (timing, component) for fori_loop application
        pre_e_ex_batch, pre_e_ex_rest = batch_slab_specs(
            self._sources_for("pre_e", "Ex")
        )
        pre_e_ey_batch, pre_e_ey_rest = batch_slab_specs(
            self._sources_for("pre_e", "Ey")
        )
        pre_e_ez_batch, pre_e_ez_rest = batch_slab_specs(
            self._sources_for("pre_e", "Ez")
        )

        h_batch_x, h_rest_x = batch_slab_specs(self._sources_for("h", "Hx"))
        h_batch_y, h_rest_y = batch_slab_specs(self._sources_for("h", "Hy"))
        h_batch_z, h_rest_z = batch_slab_specs(self._sources_for("h", "Hz"))

        e_batch_x, e_rest_x = batch_slab_specs(self._sources_for("e", "Ex"))
        e_batch_y, e_rest_y = batch_slab_specs(self._sources_for("e", "Ey"))
        e_batch_z, e_rest_z = batch_slab_specs(self._sources_for("e", "Ez"))

        # Batch 3D monitors for fori_loop power computation
        batched_mon = None
        monitors_2d: tuple[CompiledMonitorSpec, ...] = ()
        if self.monitor_specs and is_3d:
            has_dft_monitor = any(
                bool(getattr(s, "dft_enabled", False)) for s in self.monitor_specs
            )
            if has_dft_monitor:
                # Keep monitor path simple and deterministic for per-component DFT
                # accumulation in 3D modal extraction.
                batched_mon = None
                monitors_2d = tuple(self.monitor_specs)
            else:
                field_shapes = {
                    "Ex": tuple(self.e_source_x.shape),
                    "Ey": tuple(self.e_source_y.shape),
                    "Ez": tuple(self.e_source_z.shape),
                    "Hx": tuple(self.h_source_x.shape),
                    "Hy": tuple(self.h_source_y.shape),
                    "Hz": tuple(self.h_source_z.shape),
                }
                batched_mon = compile_batched_monitor_data(
                    self.monitor_specs, field_shapes
                )
                monitors_2d = tuple(s for s in self.monitor_specs if not s.is_3d)
        elif self.monitor_specs:
            monitors_2d = tuple(self.monitor_specs)

        def run_scan(
            engine_state: EngineState,
            monitor_state: MonitorState,
            coeffs: UpdateCoefficients,
        ):
            h_decay_x, h_source_x = coeffs.h_decay_x, coeffs.h_source_x
            h_source_lossless_x = coeffs.h_source_lossless_x
            h_decay_y, h_source_y = coeffs.h_decay_y, coeffs.h_source_y
            h_source_lossless_y = coeffs.h_source_lossless_y
            h_decay_z, h_source_z = coeffs.h_decay_z, coeffs.h_source_z
            h_source_lossless_z = coeffs.h_source_lossless_z
            e_decay_x, e_source_x = coeffs.e_decay_x, coeffs.e_source_x
            e_source_lossless_x = coeffs.e_source_lossless_x
            e_decay_y, e_source_y = coeffs.e_decay_y, coeffs.e_source_y
            e_source_lossless_y = coeffs.e_source_lossless_y
            e_decay_z, e_source_z = coeffs.e_decay_z, coeffs.e_source_z
            e_source_lossless_z = coeffs.e_source_lossless_z

            use_lossy_shell_ex = self.e_use_lossy_shell_x
            use_lossy_shell_ey = self.e_use_lossy_shell_y
            use_lossy_shell_ez = self.e_use_lossy_shell_z
            lossy_shell_ex = self.e_lossy_shell_x
            lossy_shell_ey = self.e_lossy_shell_y
            lossy_shell_ez = self.e_lossy_shell_z
            use_lossy_shell_hx = self.h_use_lossy_shell_x
            use_lossy_shell_hy = self.h_use_lossy_shell_y
            use_lossy_shell_hz = self.h_use_lossy_shell_z
            lossy_shell_hx = self.h_lossy_shell_x
            lossy_shell_hy = self.h_lossy_shell_y
            lossy_shell_hz = self.h_lossy_shell_z

            def body_with_coeffs(carry):
                eng, mon, mat = carry
                abs_step = eng.current_step

                ex, ey, ez = eng.ex, eng.ey, eng.ez
                hx, hy, hz = eng.hx, eng.hy, eng.hz

                ex = self._apply_source_group(
                    ex, abs_step, pre_e_ex_batch, pre_e_ex_rest
                )
                ey = self._apply_source_group(
                    ey, abs_step, pre_e_ey_batch, pre_e_ey_rest
                )
                ez = self._apply_source_group(
                    ez, abs_step, pre_e_ez_batch, pre_e_ez_rest
                )

                if is_3d:
                    any_h_shell = (
                        use_lossy_shell_hx or use_lossy_shell_hy or use_lossy_shell_hz
                    )
                    if any_h_shell:
                        # Shell path: lossless fused update, then lossy shell correction
                        # without explicit curl arrays.
                        hx_old, hy_old, hz_old = hx, hy, hz
                        hx, hy, hz = ops.fused_update_h_lossless_3d(
                            ex,
                            ey,
                            ez,
                            hx,
                            hy,
                            hz,
                            h_source_lossless_x,
                            h_source_lossless_y,
                            h_source_lossless_z,
                            resolution,
                        )
                        if use_lossy_shell_hx:
                            hx = self._apply_lossy_shell_from_lossless(
                                updated_lossless=hx,
                                old=hx_old,
                                decay=h_decay_x,
                                source=h_source_x,
                                source_lossless=h_source_lossless_x,
                                slabs=lossy_shell_hx,
                            )
                        if use_lossy_shell_hy:
                            hy = self._apply_lossy_shell_from_lossless(
                                updated_lossless=hy,
                                old=hy_old,
                                decay=h_decay_y,
                                source=h_source_y,
                                source_lossless=h_source_lossless_y,
                                slabs=lossy_shell_hy,
                            )
                        if use_lossy_shell_hz:
                            hz = self._apply_lossy_shell_from_lossless(
                                updated_lossless=hz,
                                old=hz_old,
                                decay=h_decay_z,
                                source=h_source_z,
                                source_lossless=h_source_lossless_z,
                                slabs=lossy_shell_hz,
                            )
                    else:
                        # Fused path: no intermediate curl arrays
                        hx, hy, hz = ops.fused_update_h_lossy_3d(
                            ex,
                            ey,
                            ez,
                            hx,
                            hy,
                            hz,
                            h_decay_x,
                            h_source_x,
                            h_decay_y,
                            h_source_y,
                            h_decay_z,
                            h_source_z,
                            resolution,
                        )
                else:
                    curl_ex, curl_ey, curl_ez = ops.curl_e_to_h_2d(
                        (ex, ey, ez),
                        resolution,
                        plane=plane_2d,
                    )

                    hx_old, hy_old, hz_old = hx, hy, hz

                    if use_lossy_shell_hx:
                        hx = hx_old - h_source_lossless_x * curl_ex
                        hx = self._apply_lossy_shell(
                            updated=hx,
                            old=hx_old,
                            curl=curl_ex,
                            decay=h_decay_x,
                            source=-h_source_x,
                            slabs=lossy_shell_hx,
                        )
                    else:
                        hx = h_decay_x * hx_old - h_source_x * curl_ex

                    if use_lossy_shell_hy:
                        hy = hy_old - h_source_lossless_y * curl_ey
                        hy = self._apply_lossy_shell(
                            updated=hy,
                            old=hy_old,
                            curl=curl_ey,
                            decay=h_decay_y,
                            source=-h_source_y,
                            slabs=lossy_shell_hy,
                        )
                    else:
                        hy = h_decay_y * hy_old - h_source_y * curl_ey

                    if use_lossy_shell_hz:
                        hz = hz_old - h_source_lossless_z * curl_ez
                        hz = self._apply_lossy_shell(
                            updated=hz,
                            old=hz_old,
                            curl=curl_ez,
                            decay=h_decay_z,
                            source=-h_source_z,
                            slabs=lossy_shell_hz,
                        )
                    else:
                        hz = h_decay_z * hz_old - h_source_z * curl_ez

                hx = self._apply_source_group(hx, abs_step, h_batch_x, h_rest_x)
                hy = self._apply_source_group(hy, abs_step, h_batch_y, h_rest_y)
                hz = self._apply_source_group(hz, abs_step, h_batch_z, h_rest_z)

                if is_3d:
                    any_e_shell = (
                        use_lossy_shell_ex or use_lossy_shell_ey or use_lossy_shell_ez
                    )
                    if any_e_shell:
                        # Shell path: lossless fused update, then lossy shell correction
                        # without explicit curl arrays.
                        ex_old, ey_old, ez_old = ex, ey, ez
                        ex, ey, ez = ops.fused_update_e_lossless_3d(
                            hx,
                            hy,
                            hz,
                            ex,
                            ey,
                            ez,
                            e_source_lossless_x,
                            e_source_lossless_y,
                            e_source_lossless_z,
                            resolution,
                        )
                        if use_lossy_shell_ex:
                            ex = self._apply_lossy_shell_from_lossless(
                                updated_lossless=ex,
                                old=ex_old,
                                decay=e_decay_x,
                                source=e_source_x,
                                source_lossless=e_source_lossless_x,
                                slabs=lossy_shell_ex,
                            )
                        if use_lossy_shell_ey:
                            ey = self._apply_lossy_shell_from_lossless(
                                updated_lossless=ey,
                                old=ey_old,
                                decay=e_decay_y,
                                source=e_source_y,
                                source_lossless=e_source_lossless_y,
                                slabs=lossy_shell_ey,
                            )
                        if use_lossy_shell_ez:
                            ez = self._apply_lossy_shell_from_lossless(
                                updated_lossless=ez,
                                old=ez_old,
                                decay=e_decay_z,
                                source=e_source_z,
                                source_lossless=e_source_lossless_z,
                                slabs=lossy_shell_ez,
                            )
                    else:
                        # Fused path: no intermediate curl arrays
                        ex, ey, ez = ops.fused_update_e_lossy_3d(
                            hx,
                            hy,
                            hz,
                            ex,
                            ey,
                            ez,
                            e_decay_x,
                            e_source_x,
                            e_decay_y,
                            e_source_y,
                            e_decay_z,
                            e_source_z,
                            resolution,
                        )
                else:
                    curl_hx, curl_hy, curl_hz = ops.curl_h_to_e_2d(
                        (hx, hy, hz),
                        resolution,
                        (ex.shape, ey.shape, ez.shape),
                        plane=plane_2d,
                    )

                    ex_old, ey_old, ez_old = ex, ey, ez

                    if use_lossy_shell_ex:
                        ex = ex_old + e_source_lossless_x * curl_hx
                        ex = self._apply_lossy_shell(
                            updated=ex,
                            old=ex_old,
                            curl=curl_hx,
                            decay=e_decay_x,
                            source=e_source_x,
                            slabs=lossy_shell_ex,
                        )
                    else:
                        ex = e_decay_x * ex_old + e_source_x * curl_hx

                    if use_lossy_shell_ey:
                        ey = ey_old + e_source_lossless_y * curl_hy
                        ey = self._apply_lossy_shell(
                            updated=ey,
                            old=ey_old,
                            curl=curl_hy,
                            decay=e_decay_y,
                            source=e_source_y,
                            slabs=lossy_shell_ey,
                        )
                    else:
                        ey = e_decay_y * ey_old + e_source_y * curl_hy

                    if use_lossy_shell_ez:
                        ez = ez_old + e_source_lossless_z * curl_hz
                        ez = self._apply_lossy_shell(
                            updated=ez,
                            old=ez_old,
                            curl=curl_hz,
                            decay=e_decay_z,
                            source=e_source_z,
                            slabs=lossy_shell_ez,
                        )
                    else:
                        ez = e_decay_z * ez_old + e_source_z * curl_hz

                ex = self._apply_source_group(ex, abs_step, e_batch_x, e_rest_x)
                ey = self._apply_source_group(ey, abs_step, e_batch_y, e_rest_y)
                ez = self._apply_source_group(ez, abs_step, e_batch_z, e_rest_z)

                mat, _ = material_model.update(mat, ex, ey, ez, abs_step)

                t_phys = eng.t
                mon = self._update_monitors(
                    mon,
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
                return (new_eng, mon, mat)

            if self.config.loop_kind == "scan":

                def _scan_body(carry, _unused):
                    return body_with_coeffs(carry), None

                (engine_final, monitor_final, material_final), _ = jax.lax.scan(
                    _scan_body,
                    (engine_state, monitor_state, material_state0),
                    xs=None,
                    length=self.config.num_steps,
                )
            else:
                init_carry = (engine_state, monitor_state, material_state0)
                engine_final, monitor_final, material_final = jax.lax.fori_loop(
                    0,
                    self.config.num_steps,
                    lambda _i, c: body_with_coeffs(c),
                    init_carry,
                )
            return engine_final, monitor_final, material_final

        # Use function-style JIT wrapping for compatibility with older JAX
        # versions where decorator kwargs require the callable as first arg.
        self._compiled_scan = jax.jit(run_scan, donate_argnums=(0, 1))
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
                max_records = max(
                    1, monitor_state_size(self.monitor_specs, self.config.num_steps)
                )
                max_freq = monitor_frequency_size(self.monitor_specs)
                max_points = monitor_dft_point_size(self.monitor_specs)
                monitor_state = MonitorState(
                    powers=jnp.zeros(
                        (len(self.monitor_specs), max_records), dtype=jnp.float32
                    ),
                    timestamps=jnp.zeros(
                        (len(self.monitor_specs), max_records), dtype=jnp.float32
                    ),
                    counts=jnp.zeros((len(self.monitor_specs),), dtype=jnp.int32),
                    freq_flux_re=jnp.zeros(
                        (len(self.monitor_specs), max_freq), dtype=jnp.float32
                    ),
                    freq_flux_im=jnp.zeros(
                        (len(self.monitor_specs), max_freq), dtype=jnp.float32
                    ),
                    freq_phase_re=jnp.ones(
                        (len(self.monitor_specs), max_freq), dtype=jnp.float32
                    ),
                    freq_phase_im=jnp.zeros(
                        (len(self.monitor_specs), max_freq), dtype=jnp.float32
                    ),
                    dft_vec_re=jnp.zeros(
                        (len(self.monitor_specs), 6, max_freq, max_points),
                        dtype=jnp.float32,
                    ),
                    dft_vec_im=jnp.zeros(
                        (len(self.monitor_specs), 6, max_freq, max_points),
                        dtype=jnp.float32,
                    ),
                    dft_weight_sum=jnp.zeros(
                        (len(self.monitor_specs), max_freq), dtype=jnp.float32
                    ),
                )
            else:
                monitor_state = MonitorState(
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


def _edge_full_thickness(mask: np.ndarray, axis: int) -> tuple[int, int]:
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
    """Build a CompiledSimulation from design/devices/boundaries and a run config.

    Required run_cfg attributes:
    - fields
    - resolution
    - dt
    - num_steps
    - plane_2d
    - is_3d
    Optional:
    - total_steps: full simulation length for absolute waveform indexing
    - t0: simulation time origin used when sampling source waveforms
    """
    del design, boundaries

    fields = run_cfg.fields
    resolution = float(run_cfg.resolution)
    dt = float(run_cfg.dt)
    num_steps = int(run_cfg.num_steps)
    total_steps = int(getattr(run_cfg, "total_steps", num_steps))
    t0 = float(getattr(run_cfg, "t0", 0.0))

    source_specs = compile_source_specs(
        devices=devices,
        fields=fields,
        dt=dt,
        resolution=resolution,
        num_steps=num_steps,
        t0=t0,
        total_steps=total_steps,
    )

    monitor_specs, _ = compile_monitor_specs(
        devices=devices,
        fields=fields,
        resolution=resolution,
        num_steps=num_steps,
        dt=dt,
    )

    monitor_devices = tuple(d for d in devices if isinstance(d, Monitor))

    loop_kind_raw = (
        str(
            getattr(
                run_cfg,
                "loop_kind",
                os.getenv("BEAMZ_COMPILED_LOOP_KIND", "scan"),
            )
        )
        .strip()
        .lower()
    )
    if loop_kind_raw in {"fori", "fori_loop", "fori-loop"}:
        loop_kind = "fori_loop"
    elif loop_kind_raw in {"scan"}:
        loop_kind = "scan"
    else:
        raise ValueError("Invalid compiled loop kind. Use one of: scan, fori_loop.")
    source_single_slab_dense = os.getenv(
        "BEAMZ_SOURCE_SINGLE_SLAB_DENSE",
        str(getattr(run_cfg, "source_single_slab_dense", False)),
    ).strip().lower() in {"1", "true", "yes", "on"}

    config = CompiledRunConfig(
        resolution=resolution,
        dt=dt,
        num_steps=num_steps,
        plane_2d=run_cfg.plane_2d,
        is_3d=bool(run_cfg.is_3d),
        precision=getattr(run_cfg, "precision", "float32"),
        loop_kind=loop_kind,
        source_single_slab_dense=source_single_slab_dense,
    )

    h_decay_x, h_source_x, h_source_lossless_x = ops.precompute_h_update_coefficients(
        fields.sigma_m_hx, dt
    )
    h_decay_y, h_source_y, h_source_lossless_y = ops.precompute_h_update_coefficients(
        fields.sigma_m_hy, dt
    )
    h_decay_z, h_source_z, h_source_lossless_z = ops.precompute_h_update_coefficients(
        fields.sigma_m_hz, dt
    )

    e_decay_x, e_source_x, e_source_lossless_x = ops.precompute_e_update_coefficients(
        shape=fields.Ex.shape,
        conductivity=fields.sig_x,
        permittivity=fields.eps_x,
        dt=dt,
        region=fields.region_x,
    )
    e_decay_y, e_source_y, e_source_lossless_y = ops.precompute_e_update_coefficients(
        shape=fields.Ey.shape,
        conductivity=fields.sig_y,
        permittivity=fields.eps_y,
        dt=dt,
        region=fields.region_y,
    )
    e_decay_z, e_source_z, e_source_lossless_z = ops.precompute_e_update_coefficients(
        shape=fields.Ez.shape,
        conductivity=fields.sig_z,
        permittivity=fields.eps_z,
        dt=dt,
        region=fields.region_z,
    )

    e_shell_frac_threshold = 0.35
    h_shell_frac_threshold = 0.20
    enable_e_shell_split = os.getenv(
        "BEAMZ_ENABLE_E_SHELL_SPLIT", ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    enable_h_shell_split = os.getenv(
        "BEAMZ_ENABLE_H_SHELL_SPLIT", ""
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if bool(run_cfg.is_3d):
        e_use_lossy_shell_x, e_lossy_shell_x = _infer_lossy_shell_slabs(
            field_shape=tuple(fields.Ex.shape),
            region=fields.region_x,
            conductivity_region=fields.sig_x,
        )
        e_use_lossy_shell_y, e_lossy_shell_y = _infer_lossy_shell_slabs(
            field_shape=tuple(fields.Ey.shape),
            region=fields.region_y,
            conductivity_region=fields.sig_y,
        )
        e_use_lossy_shell_z, e_lossy_shell_z = _infer_lossy_shell_slabs(
            field_shape=tuple(fields.Ez.shape),
            region=fields.region_z,
            conductivity_region=fields.sig_z,
        )
        h_use_lossy_shell_x, h_lossy_shell_x = _infer_lossy_shell_slabs(
            field_shape=tuple(fields.Hx.shape),
            region=(slice(None), slice(None), slice(None)),
            conductivity_region=fields.sigma_m_hx,
        )
        h_use_lossy_shell_y, h_lossy_shell_y = _infer_lossy_shell_slabs(
            field_shape=tuple(fields.Hy.shape),
            region=(slice(None), slice(None), slice(None)),
            conductivity_region=fields.sigma_m_hy,
        )
        h_use_lossy_shell_z, h_lossy_shell_z = _infer_lossy_shell_slabs(
            field_shape=tuple(fields.Hz.shape),
            region=(slice(None), slice(None), slice(None)),
            conductivity_region=fields.sigma_m_hz,
        )

        if enable_e_shell_split:
            e_use_lossy_shell_x = e_use_lossy_shell_x and (
                _lossy_fraction(tuple(fields.Ex.shape), fields.region_x, fields.sig_x)
                <= e_shell_frac_threshold
            )
            e_use_lossy_shell_y = e_use_lossy_shell_y and (
                _lossy_fraction(tuple(fields.Ey.shape), fields.region_y, fields.sig_y)
                <= e_shell_frac_threshold
            )
            e_use_lossy_shell_z = e_use_lossy_shell_z and (
                _lossy_fraction(tuple(fields.Ez.shape), fields.region_z, fields.sig_z)
                <= e_shell_frac_threshold
            )
        else:
            e_use_lossy_shell_x, e_use_lossy_shell_y, e_use_lossy_shell_z = (
                False,
                False,
                False,
            )
        if enable_h_shell_split:
            h_use_lossy_shell_x = h_use_lossy_shell_x and (
                _lossy_fraction(
                    tuple(fields.Hx.shape),
                    (slice(None), slice(None), slice(None)),
                    fields.sigma_m_hx,
                )
                <= h_shell_frac_threshold
            )
            h_use_lossy_shell_y = h_use_lossy_shell_y and (
                _lossy_fraction(
                    tuple(fields.Hy.shape),
                    (slice(None), slice(None), slice(None)),
                    fields.sigma_m_hy,
                )
                <= h_shell_frac_threshold
            )
            h_use_lossy_shell_z = h_use_lossy_shell_z and (
                _lossy_fraction(
                    tuple(fields.Hz.shape),
                    (slice(None), slice(None), slice(None)),
                    fields.sigma_m_hz,
                )
                <= h_shell_frac_threshold
            )
        else:
            h_use_lossy_shell_x, h_use_lossy_shell_y, h_use_lossy_shell_z = (
                False,
                False,
                False,
            )
    else:
        e_use_lossy_shell_x, e_lossy_shell_x = False, tuple()
        e_use_lossy_shell_y, e_lossy_shell_y = False, tuple()
        e_use_lossy_shell_z, e_lossy_shell_z = False, tuple()
        h_use_lossy_shell_x, h_lossy_shell_x = False, tuple()
        h_use_lossy_shell_y, h_lossy_shell_y = False, tuple()
        h_use_lossy_shell_z, h_lossy_shell_z = False, tuple()

    return CompiledSimulation(
        config=config,
        material_spec=CompiledMaterialSpec(model_kind="linear"),
        source_specs=source_specs,
        monitor_specs=monitor_specs,
        monitor_devices=monitor_devices,
        h_decay_x=h_decay_x,
        h_source_x=h_source_x,
        h_source_lossless_x=h_source_lossless_x,
        h_decay_y=h_decay_y,
        h_source_y=h_source_y,
        h_source_lossless_y=h_source_lossless_y,
        h_decay_z=h_decay_z,
        h_source_z=h_source_z,
        h_source_lossless_z=h_source_lossless_z,
        e_decay_x=e_decay_x,
        e_source_x=e_source_x,
        e_source_lossless_x=e_source_lossless_x,
        e_decay_y=e_decay_y,
        e_source_y=e_source_y,
        e_source_lossless_y=e_source_lossless_y,
        e_decay_z=e_decay_z,
        e_source_z=e_source_z,
        e_source_lossless_z=e_source_lossless_z,
        e_use_lossy_shell_x=e_use_lossy_shell_x,
        e_lossy_shell_x=e_lossy_shell_x,
        e_use_lossy_shell_y=e_use_lossy_shell_y,
        e_lossy_shell_y=e_lossy_shell_y,
        e_use_lossy_shell_z=e_use_lossy_shell_z,
        e_lossy_shell_z=e_lossy_shell_z,
        h_use_lossy_shell_x=h_use_lossy_shell_x,
        h_lossy_shell_x=h_lossy_shell_x,
        h_use_lossy_shell_y=h_use_lossy_shell_y,
        h_lossy_shell_y=h_lossy_shell_y,
        h_use_lossy_shell_z=h_use_lossy_shell_z,
        h_lossy_shell_z=h_lossy_shell_z,
    )
