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
        return tuple(
            s
            for s in self.source_specs
            if s.timing == timing and s.component == component
        )

    def _apply_specs(
        self,
        arr: jnp.ndarray,
        abs_step: jnp.ndarray,
        specs: tuple[CompiledSourceSpec, ...],
    ) -> jnp.ndarray:
        out = arr
        for spec in specs:
            safe_idx = jnp.clip(abs_step, 0, spec.waveform.shape[0] - 1)
            amp = spec.waveform[safe_idx]
            if (
                spec.is_slab
                and spec.slab_starts is not None
                and spec.slab_sizes is not None
            ):
                patch = spec.coeff * amp
                cur = jax.lax.dynamic_slice(out, spec.slab_starts, spec.slab_sizes)
                out = jax.lax.dynamic_update_slice(out, cur + patch, spec.slab_starts)
            else:
                out = out.at[spec.index].add(spec.coeff * amp)
        return out

    def _apply_batched_slabs(
        self,
        arr: jnp.ndarray,
        abs_step: jnp.ndarray,
        group: BatchedSlabGroup,
    ) -> jnp.ndarray:
        """Apply stacked slab sources via fori_loop (constant HLO size)."""
        safe_idx = jnp.clip(abs_step, 0, group.waveforms.shape[1] - 1)
        ndim = len(group.max_sizes)

        # Common hot path for benchmarks and many production setups:
        # one slab source only (e.g., single Gaussian source). Avoid nested
        # fori_loop overhead in the timestep kernel.
        if group.n == 1:
            amp = group.waveforms[0, safe_idx]
            starts_0 = group.starts_tuple[0]
            if self.config.source_single_slab_dense:
                # Optional DUS-free path for diagnostics: materialize a full-grid
                # coefficient tensor once and inject via dense add.
                pad_width = tuple(
                    (
                        starts_0[d],
                        int(arr.shape[d]) - starts_0[d] - group.max_sizes[d],
                    )
                    for d in range(ndim)
                )
                dense_coeff = jnp.pad(group.coeffs[0], pad_width)
                return arr + dense_coeff * amp
            patch = group.coeffs[0] * amp
            cur = jax.lax.dynamic_slice(arr, starts_0, group.max_sizes)
            return jax.lax.dynamic_update_slice(arr, cur + patch, starts_0)

        if group.n == 2:

            def apply_one(out, i: int):
                amp_i = group.waveforms[i, safe_idx]
                patch_i = group.coeffs[i] * amp_i
                starts_i = group.starts_tuple[i]
                cur_i = jax.lax.dynamic_slice(out, starts_i, group.max_sizes)
                return jax.lax.dynamic_update_slice(out, cur_i + patch_i, starts_i)

            return apply_one(apply_one(arr, 0), 1)

        def body(i, out):
            amp = group.waveforms[i, safe_idx]
            patch = group.coeffs[i] * amp
            starts_i = [group.starts[i, d] for d in range(ndim)]
            cur = jax.lax.dynamic_slice(out, starts_i, group.max_sizes)
            return jax.lax.dynamic_update_slice(out, cur + patch, starts_i)

        return jax.lax.fori_loop(0, group.n, body, arr)

    def _apply_source_group(
        self,
        arr: jnp.ndarray,
        abs_step: jnp.ndarray,
        batch: BatchedSlabGroup | None,
        rest: tuple[CompiledSourceSpec, ...],
    ) -> jnp.ndarray:
        """Apply batched slab sources then remaining non-slab sources."""
        if batch is not None:
            arr = self._apply_batched_slabs(arr, abs_step, batch)
        if rest:
            arr = self._apply_specs(arr, abs_step, rest)
        return arr

    def _apply_lossy_shell(
        self,
        updated: jnp.ndarray,
        old: jnp.ndarray,
        curl: jnp.ndarray,
        decay: jnp.ndarray,
        source: jnp.ndarray,
        slabs: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...],
    ) -> jnp.ndarray:
        """Apply lossy E update only on precomputed disjoint boundary slabs."""
        out = updated
        for starts, sizes in slabs:
            old_s = jax.lax.dynamic_slice(old, starts, sizes)
            curl_s = jax.lax.dynamic_slice(curl, starts, sizes)
            decay_s = jax.lax.dynamic_slice(decay, starts, sizes)
            source_s = jax.lax.dynamic_slice(source, starts, sizes)
            lossy_s = decay_s * old_s + source_s * curl_s
            out = jax.lax.dynamic_update_slice(out, lossy_s, starts)
        return out

    def _apply_lossy_shell_from_lossless(
        self,
        updated_lossless: jnp.ndarray,
        old: jnp.ndarray,
        decay: jnp.ndarray,
        source: jnp.ndarray,
        source_lossless: jnp.ndarray,
        slabs: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...],
    ) -> jnp.ndarray:
        """Apply lossy correction on shell slabs using old/lossless fields only.

        Given:
            lossless = old +/- source_lossless * curl
            lossy    = decay * old +/- source * curl
        we can eliminate curl on the shell via:
            beta = source / source_lossless
            lossy = (decay - beta) * old + beta * lossless
        """
        out = updated_lossless
        for starts, sizes in slabs:
            old_s = jax.lax.dynamic_slice(old, starts, sizes)
            lossless_s = jax.lax.dynamic_slice(updated_lossless, starts, sizes)
            decay_s = jax.lax.dynamic_slice(decay, starts, sizes)
            source_s = jax.lax.dynamic_slice(source, starts, sizes)
            source_ll_s = jax.lax.dynamic_slice(source_lossless, starts, sizes)
            beta = source_s / source_ll_s
            lossy_s = (decay_s - beta) * old_s + beta * lossless_s
            out = jax.lax.dynamic_update_slice(out, lossy_s, starts)
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
        if not self.monitor_specs:
            return monitor_state

        powers = monitor_state.powers
        timestamps = monitor_state.timestamps
        counts = monitor_state.counts
        freq_flux_re = monitor_state.freq_flux_re
        freq_flux_im = monitor_state.freq_flux_im
        freq_phase_re = monitor_state.freq_phase_re
        freq_phase_im = monitor_state.freq_phase_im
        dft_vec_re = monitor_state.dft_vec_re
        dft_vec_im = monitor_state.dft_vec_im
        dft_weight_sum = monitor_state.dft_weight_sum
        max_records = powers.shape[1]

        # Batched 3D monitors via fori_loop (constant HLO size)
        if batched_mon is not None:
            bm = batched_mon
            ex_flat = ex.ravel()
            ey_flat = ey.ravel()
            ez_flat = ez.ravel()
            hx_flat = hx.ravel()
            hy_flat = hy.ravel()
            hz_flat = hz.ravel()

            def _mon_body(i, carry):
                pwr, ts, cnt, f_re, f_im, ph_re, ph_im, d_re, d_im, d_w = carry
                mi = bm.monitor_indices[i]

                should_record = (abs_step % bm.record_intervals[i]) == 0
                can_record = cnt[mi] < max_records
                do_record = should_record & can_record & bm.accumulate_flags[i]

                mask = bm.valid_mask[i]
                exs = ex_flat[bm.ex_flat_idx[i]] * mask
                eys = ey_flat[bm.ey_flat_idx[i]] * mask
                ezs = ez_flat[bm.ez_flat_idx[i]] * mask
                hxs = hx_flat[bm.hx_flat_idx[i]] * mask
                hys = hy_flat[bm.hy_flat_idx[i]] * mask
                hzs = hz_flat[bm.hz_flat_idx[i]] * mask

                sx = eys * hzs - ezs * hys
                sy = ezs * hxs - exs * hzs
                sz = exs * hys - eys * hxs
                power_val = (
                    jnp.sum(jnp.sqrt(sx * sx + sy * sy + sz * sz)) * bm.power_scales[i]
                )
                axis_i = bm.normal_axes[i]
                normal_flux = (
                    jnp.sum(jnp.where(axis_i == 0, sx, jnp.where(axis_i == 1, sy, sz)))
                    * bm.power_scales[i]
                )
                flux_sample = jnp.where(axis_i < 0, power_val, normal_flux)

                slot = jnp.minimum(cnt[mi], max_records - 1)
                pwr = pwr.at[mi, slot].set(
                    jnp.where(do_record, power_val, pwr[mi, slot])
                )
                ts = ts.at[mi, slot].set(jnp.where(do_record, t_phys, ts[mi, slot]))
                cnt = cnt.at[mi].set(cnt[mi] + jnp.where(do_record, 1, 0))
                do_freq = bm.freq_enabled[i] & (
                    (abs_step % bm.freq_record_intervals[i]) == 0
                )
                mask_f = bm.freq_mask[i]
                row_f_re = f_re[mi]
                row_f_im = f_im[mi]
                row_ph_re = ph_re[mi]
                row_ph_im = ph_im[mi]
                delta_re = flux_sample * dt_scalar * row_ph_re * mask_f
                delta_im = flux_sample * dt_scalar * row_ph_im * mask_f
                row_f_re = row_f_re + jnp.where(do_freq, delta_re, 0.0)
                row_f_im = row_f_im + jnp.where(do_freq, delta_im, 0.0)
                rot_re = bm.freq_rot_re[i]
                rot_im = bm.freq_rot_im[i]
                next_ph_re = row_ph_re * rot_re - row_ph_im * rot_im
                next_ph_im = row_ph_re * rot_im + row_ph_im * rot_re
                row_ph_re = jnp.where(do_freq, next_ph_re, row_ph_re)
                row_ph_im = jnp.where(do_freq, next_ph_im, row_ph_im)
                f_re = f_re.at[mi].set(row_f_re)
                f_im = f_im.at[mi].set(row_f_im)
                ph_re = ph_re.at[mi].set(row_ph_re)
                ph_im = ph_im.at[mi].set(row_ph_im)
                do_dft = (
                    bm.dft_enabled[i]
                    & ((abs_step % bm.dft_record_intervals[i]) == 0)
                    & (t_phys >= bm.dft_t_start[i])
                    & (t_phys <= bm.dft_t_end[i])
                )
                two_pi = jnp.asarray(2.0 * np.pi, dtype=jnp.float32)
                span = jnp.maximum(bm.dft_t_end[i] - bm.dft_t_start[i], 1e-30)
                tau = jnp.clip((t_phys - bm.dft_t_start[i]) / span, 0.0, 1.0)
                w_hann = 0.5 * (1.0 - jnp.cos(two_pi * tau))
                w = jnp.where(
                    bm.dft_window_code[i] == 1,
                    w_hann,
                    jnp.asarray(1.0, dtype=jnp.float32),
                )
                w = jnp.where(do_dft, w, jnp.asarray(0.0, dtype=jnp.float32))

                f_hz = bm.freq_hz[i]
                mask_f = bm.freq_mask[i]
                phi = -two_pi * f_hz * t_phys
                ph_vec_re = jnp.cos(phi) * mask_f
                ph_vec_im = jnp.sin(phi) * mask_f

                vecs = jnp.stack((exs, eys, ezs, hxs, hys, hzs), axis=0)
                comp_mask = bm.dft_component_mask[i][:, None, None]
                delta_re_3d = w * comp_mask * jnp.einsum("f,cp->cfp", ph_vec_re, vecs)
                delta_im_3d = w * comp_mask * jnp.einsum("f,cp->cfp", ph_vec_im, vecs)
                d_re = d_re.at[mi].add(delta_re_3d)
                d_im = d_im.at[mi].add(delta_im_3d)
                d_w = d_w.at[mi].add(w * mask_f)

                return pwr, ts, cnt, f_re, f_im, ph_re, ph_im, d_re, d_im, d_w

            (
                powers,
                timestamps,
                counts,
                freq_flux_re,
                freq_flux_im,
                freq_phase_re,
                freq_phase_im,
                dft_vec_re,
                dft_vec_im,
                dft_weight_sum,
            ) = jax.lax.fori_loop(
                0,
                bm.n_monitors,
                _mon_body,
                (
                    powers,
                    timestamps,
                    counts,
                    freq_flux_re,
                    freq_flux_im,
                    freq_phase_re,
                    freq_phase_im,
                    dft_vec_re,
                    dft_vec_im,
                    dft_weight_sum,
                ),
            )

        # Remaining static monitor specs (2D or 3D).
        for mon in monitors_2d:
            should_record = (abs_step % mon.record_interval) == 0
            can_record = counts[mon.monitor_index] < max_records
            do_record = should_record & can_record & mon.accumulate_power
            do_freq = (
                (abs_step % mon.freq_record_interval) == 0
                if mon.accumulate_frequency and mon.freq_count > 0
                else jnp.array(False)
            )
            need_sample = do_record | do_freq

            power_sample = jnp.where(
                need_sample,
                (
                    self._monitor_power_3d(mon, ex, ey, ez, hx, hy, hz)
                    if mon.is_3d
                    else self._monitor_power_2d(mon, ez, hx, hy)
                ),
                jnp.array(0.0, dtype=jnp.float32),
            )
            power_val = jnp.where(
                do_record, power_sample, jnp.array(0.0, dtype=jnp.float32)
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
            if mon.accumulate_frequency and mon.freq_count > 0:
                mi = mon.monitor_index
                row_f_re = freq_flux_re[mi, : mon.freq_count]
                row_f_im = freq_flux_im[mi, : mon.freq_count]
                row_ph_re = freq_phase_re[mi, : mon.freq_count]
                row_ph_im = freq_phase_im[mi, : mon.freq_count]
                delta_re = power_sample * dt_scalar * row_ph_re
                delta_im = power_sample * dt_scalar * row_ph_im
                row_f_re = row_f_re + jnp.where(do_freq, delta_re, 0.0)
                row_f_im = row_f_im + jnp.where(do_freq, delta_im, 0.0)
                next_ph_re = row_ph_re * mon.freq_rot_re - row_ph_im * mon.freq_rot_im
                next_ph_im = row_ph_re * mon.freq_rot_im + row_ph_im * mon.freq_rot_re
                row_ph_re = jnp.where(do_freq, next_ph_re, row_ph_re)
                row_ph_im = jnp.where(do_freq, next_ph_im, row_ph_im)
                freq_flux_re = freq_flux_re.at[mi, : mon.freq_count].set(row_f_re)
                freq_flux_im = freq_flux_im.at[mi, : mon.freq_count].set(row_f_im)
                freq_phase_re = freq_phase_re.at[mi, : mon.freq_count].set(row_ph_re)
                freq_phase_im = freq_phase_im.at[mi, : mon.freq_count].set(row_ph_im)
            if mon.dft_enabled and mon.freq_count > 0 and mon.dft_point_count > 0:
                do_dft = (
                    ((abs_step % mon.dft_record_interval) == 0)
                    & (t_phys >= mon.dft_t_start)
                    & (t_phys <= mon.dft_t_end)
                )
                two_pi = jnp.asarray(2.0 * np.pi, dtype=jnp.float32)
                span = jnp.maximum(mon.dft_t_end - mon.dft_t_start, 1e-30)
                tau = jnp.clip((t_phys - mon.dft_t_start) / span, 0.0, 1.0)
                w_hann = 0.5 * (1.0 - jnp.cos(two_pi * tau))
                w = jnp.where(
                    mon.dft_window_code == 1,
                    w_hann,
                    jnp.asarray(1.0, dtype=jnp.float32),
                )
                w = jnp.where(do_dft, w, jnp.asarray(0.0, dtype=jnp.float32))

                if mon.is_3d:
                    ex_vals = ex[mon.ex_idx][: mon.min_dim0, : mon.min_dim1].reshape(-1)
                    ey_vals = ey[mon.ey_idx][: mon.min_dim0, : mon.min_dim1].reshape(-1)
                    ez_vals = ez[mon.ez_idx][: mon.min_dim0, : mon.min_dim1].reshape(-1)
                    hx_vals = hx[mon.hx_idx][: mon.min_dim0, : mon.min_dim1].reshape(-1)
                    hy_vals = hy[mon.hy_idx][: mon.min_dim0, : mon.min_dim1].reshape(-1)
                    hz_vals = hz[mon.hz_idx][: mon.min_dim0, : mon.min_dim1].reshape(-1)
                else:
                    ex_vals = ex[mon.y_ex, mon.x_ex] * mon.valid_ex
                    ey_vals = ey[mon.y_ey, mon.x_ey] * mon.valid_ey
                    ez_vals = ez[mon.y_ez, mon.x_ez] * mon.valid_ez
                    hx_vals = hx[mon.y_hx, mon.x_hx] * mon.valid_hx
                    hy_vals = hy[mon.y_hy, mon.x_hy] * mon.valid_hy
                    hz_vals = hz[mon.y_hz, mon.x_hz] * mon.valid_hz
                vecs = jnp.stack(
                    (ex_vals, ey_vals, ez_vals, hx_vals, hy_vals, hz_vals), axis=0
                )
                phase = -two_pi * mon.freq_hz * t_phys
                ph_re = jnp.cos(phase)
                ph_im = jnp.sin(phase)
                comp_mask = mon.dft_component_mask[:, None, None]
                delta_re = w * comp_mask * jnp.einsum("f,cp->cfp", ph_re, vecs)
                delta_im = w * comp_mask * jnp.einsum("f,cp->cfp", ph_im, vecs)
                mi = mon.monitor_index
                dft_vec_re = dft_vec_re.at[
                    mi, :, : mon.freq_count, : mon.dft_point_count
                ].add(delta_re[:, : mon.freq_count, : mon.dft_point_count])
                dft_vec_im = dft_vec_im.at[
                    mi, :, : mon.freq_count, : mon.dft_point_count
                ].add(delta_im[:, : mon.freq_count, : mon.dft_point_count])
                dft_weight_sum = dft_weight_sum.at[mi, : mon.freq_count].add(w)

        return MonitorState(
            powers=powers,
            timestamps=timestamps,
            counts=counts,
            freq_flux_re=freq_flux_re,
            freq_flux_im=freq_flux_im,
            freq_phase_re=freq_phase_re,
            freq_phase_im=freq_phase_im,
            dft_vec_re=dft_vec_re,
            dft_vec_im=dft_vec_im,
            dft_weight_sum=dft_weight_sum,
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

        @jax.jit(donate_argnums=(0, 1))
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
        for spec in self.monitor_specs:
            dev = self.monitor_devices[spec.monitor_index]
            count = int(np.asarray(monitor_state.counts[spec.monitor_index]))
            powers = np.asarray(
                monitor_state.powers[spec.monitor_index, :count], dtype=float
            )
            ts = np.asarray(
                monitor_state.timestamps[spec.monitor_index, :count], dtype=float
            )

            dev.power_history = list(powers.tolist())
            dev.power_timestamps = list(ts.tolist())
            dev.power_accumulation_count = count
            if spec.freq_count > 0:
                re = np.asarray(
                    monitor_state.freq_flux_re[spec.monitor_index, : spec.freq_count],
                    dtype=np.float32,
                )
                im = np.asarray(
                    monitor_state.freq_flux_im[spec.monitor_index, : spec.freq_count],
                    dtype=np.float32,
                )
                dev.frequency_flux_spectrum = (re + 1j * im).astype(np.complex64)
            else:
                dev.frequency_flux_spectrum = np.zeros((0,), dtype=np.complex64)

            if spec.dft_enabled and spec.freq_count > 0 and spec.dft_point_count > 0:
                comp_names = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
                comp_mask = (
                    np.asarray(spec.dft_component_mask, dtype=np.float32)
                    if spec.dft_component_mask is not None
                    else np.ones((6,), dtype=np.float32)
                )
                weight_sum = np.asarray(
                    monitor_state.dft_weight_sum[spec.monitor_index, : spec.freq_count],
                    dtype=np.float64,
                )
                dev._dft_weight_sum = weight_sum
                dev._dft_accum = {}
                for comp_i, comp_name in enumerate(comp_names):
                    if comp_mask[comp_i] <= 0.0:
                        continue
                    re = np.asarray(
                        monitor_state.dft_vec_re[
                            spec.monitor_index,
                            comp_i,
                            : spec.freq_count,
                            : spec.dft_point_count,
                        ],
                        dtype=np.float64,
                    )
                    im = np.asarray(
                        monitor_state.dft_vec_im[
                            spec.monitor_index,
                            comp_i,
                            : spec.freq_count,
                            : spec.dft_point_count,
                        ],
                        dtype=np.float64,
                    )
                    dev._dft_accum[comp_name] = re + 1j * im
            else:
                dev._dft_weight_sum = np.zeros((0,), dtype=np.float64)
                dev._dft_accum = {}


def monitor_state_size(specs: tuple[CompiledMonitorSpec, ...], num_steps: int) -> int:
    if not specs:
        return 0
    return int(
        max(
            int(np.ceil(num_steps / max(1, int(spec.record_interval))))
            for spec in specs
        )
    )


def monitor_frequency_size(specs: tuple[CompiledMonitorSpec, ...]) -> int:
    if not specs:
        return 0
    return int(max(int(spec.freq_count) for spec in specs))


def monitor_dft_point_size(specs: tuple[CompiledMonitorSpec, ...]) -> int:
    if not specs:
        return 0
    return int(max(int(getattr(spec, "dft_point_count", 0)) for spec in specs))


def _edge_full_thickness(mask: np.ndarray, axis: int) -> tuple[int, int]:
    """Count leading/trailing planes that are fully lossy along a given axis."""
    other_axes = tuple(i for i in range(mask.ndim) if i != axis)
    plane_all = mask.all(axis=other_axes)

    left = 0
    n = plane_all.shape[0]
    while left < n and bool(plane_all[left]):
        left += 1

    right = 0
    while right < (n - left) and bool(plane_all[n - 1 - right]):
        right += 1
    return left, right


def _region_offsets_and_sizes(
    field_shape: tuple[int, ...],
    region: tuple[slice, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Return region starts/sizes for slice-only regions with unit strides."""
    if len(field_shape) != len(region):
        return None

    starts: list[int] = []
    sizes: list[int] = []
    for dim, key in zip(field_shape, region):
        if not isinstance(key, slice):
            return None
        start, stop, step = key.indices(dim)
        if step != 1:
            return None
        starts.append(int(start))
        sizes.append(int(max(stop - start, 0)))
    return tuple(starts), tuple(sizes)


def _infer_lossy_shell_slabs(
    field_shape: tuple[int, ...],
    region: tuple[slice, ...],
    conductivity_region: jnp.ndarray,
) -> tuple[bool, tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]]:
    """Infer disjoint boundary-shell slabs from conductivity mask.

    Returns (enabled, slabs). Enabled is True only if the lossy mask can be represented as
    a standard disjoint shell decomposition in 3D.
    """
    if len(field_shape) != 3:
        return False, tuple()

    region_layout = _region_offsets_and_sizes(field_shape, region)
    if region_layout is None:
        return False, tuple()
    region_starts, region_sizes = region_layout
    if len(region_sizes) != 3 or any(s <= 0 for s in region_sizes):
        return False, tuple()

    local_mask = np.asarray(conductivity_region) > 0.0
    if tuple(local_mask.shape) != tuple(region_sizes):
        return False, tuple()
    if not local_mask.any():
        return False, tuple()

    zL, zR = _edge_full_thickness(local_mask, axis=0)
    yL, yR = _edge_full_thickness(local_mask, axis=1)
    xL, xR = _edge_full_thickness(local_mask, axis=2)

    nz, ny, nx = region_sizes
    z0, z1 = zL, nz - zR
    y0, y1 = yL, ny - yR
    x0, x1 = xL, nx - xR
    if z0 > z1 or y0 > y1 or x0 > x1:
        return False, tuple()

    slabs: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []

    def add_slab(starts: tuple[int, int, int], sizes: tuple[int, int, int]):
        if all(s > 0 for s in sizes):
            slabs.append((starts, sizes))

    # Disjoint shell: z faces, then y faces within z-core, then x faces within yz-core.
    add_slab((0, 0, 0), (zL, ny, nx))
    add_slab((z1, 0, 0), (zR, ny, nx))
    add_slab((z0, 0, 0), (max(z1 - z0, 0), yL, nx))
    add_slab((z0, y1, 0), (max(z1 - z0, 0), yR, nx))
    add_slab((z0, y0, 0), (max(z1 - z0, 0), max(y1 - y0, 0), xL))
    add_slab((z0, y0, x1), (max(z1 - z0, 0), max(y1 - y0, 0), xR))

    if not slabs:
        return False, tuple()

    recon = np.zeros(region_sizes, dtype=bool)
    for starts, sizes in slabs:
        z, y, x = starts
        dz, dy, dx = sizes
        recon[z : z + dz, y : y + dy, x : x + dx] = True

    if not np.array_equal(recon, local_mask):
        return False, tuple()

    z_off, y_off, x_off = region_starts
    global_slabs = tuple(
        (
            (starts[0] + z_off, starts[1] + y_off, starts[2] + x_off),
            sizes,
        )
        for starts, sizes in slabs
    )
    return True, global_slabs


def _lossy_fraction(
    field_shape: tuple[int, ...],
    region: tuple[slice, ...],
    conductivity_region: jnp.ndarray,
) -> float:
    """Fraction of lossy voxels for a field component."""
    full_mask = np.zeros(field_shape, dtype=bool)
    full_mask[region] = np.asarray(conductivity_region) > 0.0
    return float(full_mask.mean())


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
