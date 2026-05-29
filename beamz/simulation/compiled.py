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

from beamz.const import EPS_0
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
from beamz.shared_kernels import (
    advance_e_from_coefficients,
    advance_h_from_coefficients,
    apply_zero_mask,
    build_cpml_3d_terms,
    build_tm_xy_cpml_terms,
    full_tm_xy_component_to_centered_grid,
    monitor_dft_sample_scale,
    monitor_dft_should_accumulate,
    monitor_dft_window_weight,
    monitor_records_on_step,
    poynting_flux_2d,
    poynting_flux_3d,
    poynting_magnitude_2d,
    poynting_magnitude_3d,
    step_hits_interval,
)
from beamz.simulation import ops
from beamz.simulation.boundaries import (
    build_h_boundary_views_for_e_3d,
    cpml_update_e_from_h_3d,
    cpml_update_h_from_e_3d,
    create_metallic_boundary_masks,
    full_pec_curl_e_to_h_2d_xy,
    full_pec_curl_e_to_h_3d,
    full_pec_curl_h_to_e_2d_xy,
    full_pec_curl_h_to_e_3d,
    full_tm_2d_xy_masks,
    has_full_pec_3d,
    initialize_full_pec_3d_state,
    resolve_metallic_edges,
    tm_xy_cpml_curl_e_to_h_2d,
    tm_xy_cpml_curl_h_to_e_2d,
    tm_xy_curl_e_to_h_2d,
    tm_xy_curl_h_to_e_2d,
    xy_te_curl_e_to_h_2d,
    xy_te_curl_h_to_e_2d,
)
from beamz.simulation.material_models import (
    CompiledMaterialSpec,
    MaterialState,
    create_material_model,
)
from beamz.simulation.step_sequence import run_step_sequence
from beamz.simulation.yee import (
    sample_voxel_grid_at_tm_xy_full_component_2d,
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


def monitor_dft_accumulator_dtype():
    """Return the real dtype used for compiled monitor DFT accumulators.

    JAX's x64 mode is global, so BeamZ does not enable it implicitly here. When
    the host application has already enabled x64, DFT accumulators use it.
    """

    return jnp.float64 if bool(jax.config.jax_enable_x64) else jnp.float32


def _sample_centered_grid_targets_2d(
    field: jnp.ndarray,
    x_targets: jnp.ndarray,
    y_targets: jnp.ndarray,
    resolution: float,
) -> jnp.ndarray:
    dx = jnp.asarray(resolution, dtype=jnp.float32)
    x = jnp.asarray(x_targets, dtype=jnp.float32)
    y = jnp.asarray(y_targets, dtype=jnp.float32)
    nx = field.shape[1]
    ny = field.shape[0]

    fx = x / dx - 0.5
    fy = y / dx - 0.5

    x0 = jnp.floor(fx).astype(jnp.int32)
    y0 = jnp.floor(fy).astype(jnp.int32)
    ax = fx - x0.astype(jnp.float32)
    ay = fy - y0.astype(jnp.float32)

    x0 = jnp.clip(x0, 0, max(nx - 1, 0))
    y0 = jnp.clip(y0, 0, max(ny - 1, 0))
    x1 = jnp.clip(x0 + 1, 0, max(nx - 1, 0))
    y1 = jnp.clip(y0 + 1, 0, max(ny - 1, 0))

    f00 = field[y0, x0]
    f01 = field[y0, x1]
    f10 = field[y1, x0]
    f11 = field[y1, x1]
    one = jnp.asarray(1.0, dtype=field.dtype)
    axf = ax.astype(field.dtype)
    ayf = ay.astype(field.dtype)
    return (one - ayf) * ((one - axf) * f00 + axf * f01) + ayf * (
        (one - axf) * f10 + axf * f11
    )


def _empty_cpml_3d_terms(dtype=jnp.float32) -> tuple[jnp.ndarray, ...]:
    return tuple(jnp.zeros((0, 0, 0), dtype=dtype) for _ in range(6))


def _embed_cpml_3d_term_to_full_volume(
    term: jnp.ndarray,
    region: str,
    volume_shape: tuple[int, int, int],
    *,
    fill_value: float = 0.0,
) -> jnp.ndarray:
    out = jnp.full(
        volume_shape, jnp.asarray(fill_value, dtype=term.dtype), dtype=term.dtype
    )
    if region == "Hx":
        return out.at[:-1, :-1, :].set(term)
    if region == "Hy":
        return out.at[:-1, :, :-1].set(term)
    if region == "Hz":
        return out.at[:, :-1, :-1].set(term)
    if region == "Ex":
        return out.at[:, :, :-1].set(term)
    if region == "Ey":
        return out.at[:, :-1, :].set(term)
    if region == "Ez":
        return out.at[:-1, :, :].set(term)
    raise ValueError(f"Unsupported CPML 3D region {region!r}")


class EngineState(NamedTuple):
    """Runtime EM field state."""

    ex: jnp.ndarray
    ey: jnp.ndarray
    ez: jnp.ndarray
    hx: jnp.ndarray
    hy: jnp.ndarray
    hz: jnp.ndarray
    tm_ez: jnp.ndarray
    tm_hx: jnp.ndarray
    tm_hy: jnp.ndarray
    fp_ex: jnp.ndarray
    fp_ey: jnp.ndarray
    fp_ez: jnp.ndarray
    fp_hx: jnp.ndarray
    fp_hy: jnp.ndarray
    fp_hz: jnp.ndarray
    cpml_psi_h_terms: jnp.ndarray
    cpml_psi_e_terms: jnp.ndarray
    cpml3d_psi_h_terms: tuple[jnp.ndarray, ...]
    cpml3d_psi_e_terms: tuple[jnp.ndarray, ...]
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
    tm_h_decay_x: jnp.ndarray
    tm_h_source_x: jnp.ndarray
    tm_h_decay_y: jnp.ndarray
    tm_h_source_y: jnp.ndarray
    tm_e_decay_z: jnp.ndarray
    tm_e_source_z: jnp.ndarray


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
    snapshot_field: str | None = None
    snapshot_interval: int = 0


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
    tm_h_decay_x: jnp.ndarray
    tm_h_source_x: jnp.ndarray
    tm_h_decay_y: jnp.ndarray
    tm_h_source_y: jnp.ndarray
    tm_e_decay_z: jnp.ndarray
    tm_e_source_z: jnp.ndarray
    tm_ez_mask: jnp.ndarray
    tm_hx_mask: jnp.ndarray
    tm_hy_mask: jnp.ndarray
    tm_metallic_edges: frozenset[str]
    use_physical_tm_xy: bool
    use_cpml_tm_xy: bool
    cpml_sigma_h_terms: jnp.ndarray
    cpml_kappa_h_aux_terms: jnp.ndarray
    cpml_alpha_h_terms: jnp.ndarray
    cpml_kappa_h_direct_terms: jnp.ndarray
    cpml_sigma_e_terms: jnp.ndarray
    cpml_kappa_e_terms: jnp.ndarray
    cpml_alpha_e_terms: jnp.ndarray
    use_cpml_3d: bool
    cpml3d_a_h_terms: tuple[jnp.ndarray, ...]
    cpml3d_b_h_terms: tuple[jnp.ndarray, ...]
    cpml3d_inv_kappa_h_terms: tuple[jnp.ndarray, ...]
    cpml3d_a_e_terms: tuple[jnp.ndarray, ...]
    cpml3d_b_e_terms: tuple[jnp.ndarray, ...]
    cpml3d_inv_kappa_e_terms: tuple[jnp.ndarray, ...]
    cpml3d_metallic_edges: frozenset[str]
    full_pec_3d: bool
    fp_h_decay_x: jnp.ndarray
    fp_h_source_x: jnp.ndarray
    fp_h_decay_y: jnp.ndarray
    fp_h_source_y: jnp.ndarray
    fp_h_decay_z: jnp.ndarray
    fp_h_source_z: jnp.ndarray
    fp_e_decay_x: jnp.ndarray
    fp_e_source_x: jnp.ndarray
    fp_e_decay_y: jnp.ndarray
    fp_e_source_y: jnp.ndarray
    fp_e_decay_z: jnp.ndarray
    fp_e_source_z: jnp.ndarray
    fp_ex_mask: jnp.ndarray
    fp_ey_mask: jnp.ndarray
    fp_ez_mask: jnp.ndarray
    fp_hx_mask: jnp.ndarray
    fp_hy_mask: jnp.ndarray
    fp_hz_mask: jnp.ndarray

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

    # Explicit metallic-wall masks aligned to the Yee staggering.
    ex_metal_mask: jnp.ndarray
    ey_metal_mask: jnp.ndarray
    ez_metal_mask: jnp.ndarray
    hx_metal_mask: jnp.ndarray
    hy_metal_mask: jnp.ndarray
    hz_metal_mask: jnp.ndarray

    _compiled_scan: callable | None = None
    _compile_count: int = 0

    @staticmethod
    def _cpml_b_c(
        sigma: jnp.ndarray,
        kappa: jnp.ndarray,
        alpha: jnp.ndarray,
        dt: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        sigma = jnp.asarray(sigma, dtype=jnp.float32)
        kappa = jnp.maximum(jnp.asarray(kappa, dtype=jnp.float32), 1.0)
        alpha = jnp.asarray(alpha, dtype=jnp.float32)
        decay = (sigma / kappa + alpha) * (
            jnp.asarray(dt, dtype=jnp.float32) / jnp.asarray(EPS_0, dtype=jnp.float32)
        )
        b = jnp.expm1(-decay) + 1.0
        denom = sigma + kappa * alpha
        c = jnp.nan_to_num(
            ((b - 1.0) * sigma) / jnp.maximum(denom * kappa, 1e-30),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return b, c

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
            tm_h_decay_x=self.tm_h_decay_x,
            tm_h_source_x=self.tm_h_source_x,
            tm_h_decay_y=self.tm_h_decay_y,
            tm_h_source_y=self.tm_h_source_y,
            tm_e_decay_z=self.tm_e_decay_z,
            tm_e_source_z=self.tm_e_source_z,
        )

    def _sources_for(
        self, timing: str, component: str
    ) -> tuple[CompiledSourceSpec, ...]:
        return tuple(
            s
            for s in self.source_specs
            if s.timing == timing and s.component == component
        )

    def _snapshot_field_shape(self) -> tuple[int, ...]:
        field_name = self.config.snapshot_field
        if field_name == "Ex":
            return tuple(self.e_source_x.shape)
        if field_name == "Ey":
            return tuple(self.e_source_y.shape)
        if field_name == "Ez":
            return tuple(self.e_source_z.shape)
        if field_name == "Hx":
            return tuple(self.h_source_x.shape)
        if field_name == "Hy":
            return tuple(self.h_source_y.shape)
        if field_name == "Hz":
            return tuple(self.h_source_z.shape)
        raise ValueError(f"Unsupported snapshot field: {field_name}")

    def _empty_snapshot_state(
        self,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray] | None:
        if self.config.snapshot_field is None:
            return None
        max_snapshots = max(
            1,
            int(
                np.ceil(
                    self.config.num_steps / max(1, int(self.config.snapshot_interval))
                )
            ),
        )
        return (
            jnp.zeros(
                (max_snapshots, *self._snapshot_field_shape()),
                dtype=jnp.float32,
            ),
            jnp.zeros((max_snapshots,), dtype=jnp.int32),
            jnp.zeros((max_snapshots,), dtype=jnp.float32),
            jnp.zeros((), dtype=jnp.int32),
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

    def _apply_metal_mask(self, arr: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(mask, jnp.asarray(0.0, dtype=arr.dtype), arr)

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
        power_scale = jnp.asarray(spec.power_scale, dtype=jnp.float32)
        ez_vals = self._sample_monitor_component_2d(
            ez,
            spec.ez_interp_flat_idx,
            spec.ez_interp_weights,
            spec.x_ez,
            spec.y_ez,
            spec.valid_ez,
        )
        hx_vals = self._sample_monitor_component_2d(
            hx,
            spec.hx_interp_flat_idx,
            spec.hx_interp_weights,
            spec.x_hx,
            spec.y_hx,
            spec.valid_hx,
        )
        hy_vals = self._sample_monitor_component_2d(
            hy,
            spec.hy_interp_flat_idx,
            spec.hy_interp_weights,
            spec.x_hy,
            spec.y_hy,
            spec.valid_hy,
        )

        axis_i = int(spec.normal_axis)
        sign = jnp.asarray(spec.normal_sign, dtype=jnp.float32)
        flux_x = poynting_flux_2d(ez_vals, hx_vals, hy_vals, "x", sign)
        flux_y = poynting_flux_2d(ez_vals, hx_vals, hy_vals, "y", sign)
        mag = poynting_magnitude_2d(ez_vals, hx_vals, hy_vals)
        flux = jnp.where(axis_i == 0, flux_x, jnp.where(axis_i == 1, flux_y, mag))
        return jnp.asarray(jnp.sum(flux), dtype=jnp.float32) * power_scale

    @staticmethod
    def _sample_monitor_component_2d(
        field: jnp.ndarray,
        flat_idx: jnp.ndarray | None,
        weights: jnp.ndarray | None,
        x_idx: jnp.ndarray,
        y_idx: jnp.ndarray,
        valid: jnp.ndarray,
    ) -> jnp.ndarray:
        if flat_idx is not None and weights is not None:
            gathered = field.reshape(-1)[flat_idx]
            return jnp.sum(gathered * weights, axis=-1)
        return field[y_idx, x_idx] * valid

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
        power_scale = jnp.asarray(spec.power_scale, dtype=jnp.float32)
        exs = self._sample_monitor_component_3d(
            ex,
            spec.ex_interp_flat_idx,
            spec.ex_interp_weights,
            spec.min_dim0,
            spec.min_dim1,
        )
        eys = self._sample_monitor_component_3d(
            ey,
            spec.ey_interp_flat_idx,
            spec.ey_interp_weights,
            spec.min_dim0,
            spec.min_dim1,
        )
        ezs = self._sample_monitor_component_3d(
            ez,
            spec.ez_interp_flat_idx,
            spec.ez_interp_weights,
            spec.min_dim0,
            spec.min_dim1,
        )
        hxs = self._sample_monitor_component_3d(
            hx,
            spec.hx_interp_flat_idx,
            spec.hx_interp_weights,
            spec.min_dim0,
            spec.min_dim1,
        )
        hys = self._sample_monitor_component_3d(
            hy,
            spec.hy_interp_flat_idx,
            spec.hy_interp_weights,
            spec.min_dim0,
            spec.min_dim1,
        )
        hzs = self._sample_monitor_component_3d(
            hz,
            spec.hz_interp_flat_idx,
            spec.hz_interp_weights,
            spec.min_dim0,
            spec.min_dim1,
        )

        axis_i = int(spec.normal_axis)
        sign = jnp.asarray(spec.normal_sign, dtype=jnp.float32)
        flux_x = poynting_flux_3d(exs, eys, ezs, hxs, hys, hzs, "x", sign)
        flux_y = poynting_flux_3d(exs, eys, ezs, hxs, hys, hzs, "y", sign)
        flux_z = poynting_flux_3d(exs, eys, ezs, hxs, hys, hzs, "z", sign)
        mag = poynting_magnitude_3d(exs, eys, ezs, hxs, hys, hzs)
        flux = jnp.where(
            axis_i == 0,
            flux_x,
            jnp.where(axis_i == 1, flux_y, jnp.where(axis_i == 2, flux_z, mag)),
        )
        return jnp.asarray(jnp.sum(flux), dtype=jnp.float32) * power_scale

    @staticmethod
    def _sample_monitor_component_3d(
        field: jnp.ndarray,
        flat_idx: jnp.ndarray | None,
        weights: jnp.ndarray | None,
        dim0: int,
        dim1: int,
    ) -> jnp.ndarray:
        if flat_idx is None or weights is None:
            return jnp.zeros((int(dim0), int(dim1)), dtype=field.dtype)
        flat = field.reshape(-1)
        gathered = flat[flat_idx]
        sampled = jnp.sum(gathered * weights, axis=-1)
        return sampled.reshape((int(dim0), int(dim1)))

    def _monitor_dft_vectors_3d(
        self,
        spec: CompiledMonitorSpec,
        ex: jnp.ndarray,
        ey: jnp.ndarray,
        ez: jnp.ndarray,
        hx: jnp.ndarray,
        hy: jnp.ndarray,
        hz: jnp.ndarray,
    ) -> jnp.ndarray:
        ex_vals = self._sample_monitor_component_3d(
            ex,
            spec.ex_interp_flat_idx,
            spec.ex_interp_weights,
            spec.min_dim0,
            spec.min_dim1,
        ).reshape(-1)
        ey_vals = self._sample_monitor_component_3d(
            ey,
            spec.ey_interp_flat_idx,
            spec.ey_interp_weights,
            spec.min_dim0,
            spec.min_dim1,
        ).reshape(-1)
        ez_vals = self._sample_monitor_component_3d(
            ez,
            spec.ez_interp_flat_idx,
            spec.ez_interp_weights,
            spec.min_dim0,
            spec.min_dim1,
        ).reshape(-1)
        hx_vals = self._sample_monitor_component_3d(
            hx,
            spec.hx_interp_flat_idx,
            spec.hx_interp_weights,
            spec.min_dim0,
            spec.min_dim1,
        ).reshape(-1)
        hy_vals = self._sample_monitor_component_3d(
            hy,
            spec.hy_interp_flat_idx,
            spec.hy_interp_weights,
            spec.min_dim0,
            spec.min_dim1,
        ).reshape(-1)
        hz_vals = self._sample_monitor_component_3d(
            hz,
            spec.hz_interp_flat_idx,
            spec.hz_interp_weights,
            spec.min_dim0,
            spec.min_dim1,
        ).reshape(-1)
        return jnp.stack((ex_vals, ey_vals, ez_vals, hx_vals, hy_vals, hz_vals), axis=0)

    def _monitor_dft_vectors_2d(
        self,
        spec: CompiledMonitorSpec,
        ex: jnp.ndarray,
        ey: jnp.ndarray,
        ez: jnp.ndarray,
        hx: jnp.ndarray,
        hy: jnp.ndarray,
        hz: jnp.ndarray,
        tm_ez: jnp.ndarray | None = None,
        tm_hx: jnp.ndarray | None = None,
        tm_hy: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        ex_vals = self._sample_monitor_component_2d(
            ex,
            spec.ex_interp_flat_idx,
            spec.ex_interp_weights,
            spec.x_ex,
            spec.y_ex,
            spec.valid_ex,
        )
        ey_vals = self._sample_monitor_component_2d(
            ey,
            spec.ey_interp_flat_idx,
            spec.ey_interp_weights,
            spec.x_ey,
            spec.y_ey,
            spec.valid_ey,
        )
        hz_vals = self._sample_monitor_component_2d(
            hz,
            spec.hz_interp_flat_idx,
            spec.hz_interp_weights,
            spec.x_hz,
            spec.y_hz,
            spec.valid_hz,
        )

        if (
            spec.dft_normalization_code == 1
            and spec.dft_centered_tm_xy_sampling
            and tm_ez is not None
            and tm_hx is not None
            and tm_hy is not None
            and spec.dft_target_x is not None
            and spec.dft_target_y is not None
            and spec.dft_point_count > 0
        ):
            x_targets = spec.dft_target_x[: spec.dft_point_count]
            y_targets = spec.dft_target_y[: spec.dft_point_count]
            ez_center = full_tm_xy_component_to_centered_grid("Ez", tm_ez)
            hx_center = full_tm_xy_component_to_centered_grid("Hx", tm_hx)
            hy_center = full_tm_xy_component_to_centered_grid("Hy", tm_hy)
            ez_vals = _sample_centered_grid_targets_2d(
                ez_center,
                x_targets,
                y_targets,
                self.config.resolution,
            )
            hx_vals = _sample_centered_grid_targets_2d(
                hx_center,
                x_targets,
                y_targets,
                self.config.resolution,
            )
            hy_vals = _sample_centered_grid_targets_2d(
                hy_center,
                x_targets,
                y_targets,
                self.config.resolution,
            )
        else:
            ez_vals = self._sample_monitor_component_2d(
                ez,
                spec.ez_interp_flat_idx,
                spec.ez_interp_weights,
                spec.x_ez,
                spec.y_ez,
                spec.valid_ez,
            )
            hx_vals = self._sample_monitor_component_2d(
                hx,
                spec.hx_interp_flat_idx,
                spec.hx_interp_weights,
                spec.x_hx,
                spec.y_hx,
                spec.valid_hx,
            )
            hy_vals = self._sample_monitor_component_2d(
                hy,
                spec.hy_interp_flat_idx,
                spec.hy_interp_weights,
                spec.x_hy,
                spec.y_hy,
                spec.valid_hy,
            )

        return jnp.stack((ex_vals, ey_vals, ez_vals, hx_vals, hy_vals, hz_vals), axis=0)

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
        tm_ez: jnp.ndarray | None = None,
        tm_hx: jnp.ndarray | None = None,
        tm_hy: jnp.ndarray | None = None,
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
        dft_dtype = dft_vec_re.dtype
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

                should_record = monitor_records_on_step(
                    abs_step, bm.record_intervals[i]
                )
                can_record = cnt[mi] < max_records
                do_record = should_record & can_record & bm.accumulate_flags[i]
                do_freq = bm.freq_enabled[i] & step_hits_interval(
                    abs_step, bm.freq_record_intervals[i]
                )
                need_sample = do_record | do_freq

                def _sample_and_update(sample_carry):
                    pwr, ts, cnt, f_re, f_im, ph_re, ph_im, d_re, d_im, d_w = (
                        sample_carry
                    )
                    mask = bm.valid_mask[i]
                    exs = (
                        jnp.sum(
                            ex_flat[bm.ex_interp_flat_idx[i]]
                            * bm.ex_interp_weights[i],
                            axis=-1,
                        )
                        * mask
                    )
                    eys = (
                        jnp.sum(
                            ey_flat[bm.ey_interp_flat_idx[i]]
                            * bm.ey_interp_weights[i],
                            axis=-1,
                        )
                        * mask
                    )
                    ezs = (
                        jnp.sum(
                            ez_flat[bm.ez_interp_flat_idx[i]]
                            * bm.ez_interp_weights[i],
                            axis=-1,
                        )
                        * mask
                    )
                    hxs = (
                        jnp.sum(
                            hx_flat[bm.hx_interp_flat_idx[i]]
                            * bm.hx_interp_weights[i],
                            axis=-1,
                        )
                        * mask
                    )
                    hys = (
                        jnp.sum(
                            hy_flat[bm.hy_interp_flat_idx[i]]
                            * bm.hy_interp_weights[i],
                            axis=-1,
                        )
                        * mask
                    )
                    hzs = (
                        jnp.sum(
                            hz_flat[bm.hz_interp_flat_idx[i]]
                            * bm.hz_interp_weights[i],
                            axis=-1,
                        )
                        * mask
                    )

                    sx = eys * hzs - ezs * hys
                    sy = ezs * hxs - exs * hzs
                    sz = exs * hys - eys * hxs
                    power_val = (
                        jnp.sum(jnp.sqrt(sx * sx + sy * sy + sz * sz))
                        * bm.power_scales[i]
                    )
                    axis_i = bm.normal_axes[i]
                    normal_flux = (
                        jnp.sum(
                            jnp.where(axis_i == 0, sx, jnp.where(axis_i == 1, sy, sz))
                        )
                        * bm.normal_signs[i]
                        * bm.power_scales[i]
                    )
                    flux_sample = jnp.where(axis_i < 0, power_val, normal_flux)

                    slot = jnp.minimum(cnt[mi], max_records - 1)
                    pwr = pwr.at[mi, slot].set(
                        jnp.where(do_record, flux_sample, pwr[mi, slot])
                    )
                    ts = ts.at[mi, slot].set(jnp.where(do_record, t_phys, ts[mi, slot]))
                    cnt = cnt.at[mi].set(cnt[mi] + jnp.where(do_record, 1, 0))
                    mask_f = bm.freq_mask[i]
                    row_f_re = f_re[mi]
                    row_f_im = f_im[mi]
                    row_ph_re = ph_re[mi]
                    row_ph_im = ph_im[mi]
                    theta_now = (
                        jnp.asarray(2.0 * np.pi, dtype=jnp.float32)
                        * bm.freq_hz[i]
                        * t_phys
                    )
                    cur_ph_re = jnp.cos(theta_now)
                    cur_ph_im = jnp.sin(theta_now)
                    delta_re = flux_sample * dt_scalar * cur_ph_re * mask_f
                    delta_im = flux_sample * dt_scalar * cur_ph_im * mask_f
                    zero_freq = jnp.asarray(0.0, dtype=row_f_re.dtype)
                    row_f_re = row_f_re + jnp.where(do_freq, delta_re, zero_freq)
                    row_f_im = row_f_im + jnp.where(do_freq, delta_im, zero_freq)
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
                    return pwr, ts, cnt, f_re, f_im, ph_re, ph_im, d_re, d_im, d_w

                return jax.lax.cond(
                    need_sample,
                    _sample_and_update,
                    lambda sample_carry: sample_carry,
                    (pwr, ts, cnt, f_re, f_im, ph_re, ph_im, d_re, d_im, d_w),
                )

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
            should_record = monitor_records_on_step(abs_step, mon.record_interval)
            can_record = counts[mon.monitor_index] < max_records
            do_record = should_record & can_record & mon.accumulate_power
            do_freq = (
                step_hits_interval(abs_step, mon.freq_record_interval)
                if mon.accumulate_frequency and mon.freq_count > 0
                else jnp.array(False)
            )
            need_sample = do_record | do_freq

            def _sample_power(_unused):
                return (
                    self._monitor_power_3d(mon, ex, ey, ez, hx, hy, hz)
                    if mon.is_3d
                    else self._monitor_power_2d(mon, ez, hx, hy)
                )

            power_sample = jax.lax.cond(
                need_sample,
                _sample_power,
                lambda _unused: jnp.array(0.0, dtype=jnp.float32),
                operand=None,
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
                theta_now = (
                    jnp.asarray(2.0 * np.pi, dtype=jnp.float32)
                    * jnp.asarray(mon.freq_hz, dtype=jnp.float32)
                    * t_phys
                )
                cur_ph_re = jnp.cos(theta_now)
                cur_ph_im = jnp.sin(theta_now)
                delta_re = power_sample * dt_scalar * cur_ph_re
                delta_im = power_sample * dt_scalar * cur_ph_im
                zero_freq = jnp.asarray(0.0, dtype=row_f_re.dtype)
                row_f_re = row_f_re + jnp.where(do_freq, delta_re, zero_freq)
                row_f_im = row_f_im + jnp.where(do_freq, delta_im, zero_freq)
                next_ph_re = row_ph_re * mon.freq_rot_re - row_ph_im * mon.freq_rot_im
                next_ph_im = row_ph_re * mon.freq_rot_im + row_ph_im * mon.freq_rot_re
                row_ph_re = jnp.where(do_freq, next_ph_re, row_ph_re)
                row_ph_im = jnp.where(do_freq, next_ph_im, row_ph_im)
                freq_flux_re = freq_flux_re.at[mi, : mon.freq_count].set(row_f_re)
                freq_flux_im = freq_flux_im.at[mi, : mon.freq_count].set(row_f_im)
                freq_phase_re = freq_phase_re.at[mi, : mon.freq_count].set(row_ph_re)
                freq_phase_im = freq_phase_im.at[mi, : mon.freq_count].set(row_ph_im)
            if mon.dft_enabled and mon.freq_count > 0 and mon.dft_point_count > 0:
                do_dft = monitor_dft_should_accumulate(
                    mon.dft_enabled and mon.freq_count > 0 and mon.dft_point_count > 0,
                    abs_step,
                    t_phys,
                    mon.dft_t_start,
                    mon.dft_t_end,
                    mon.dft_record_interval,
                )

                def _accumulate_dft(carry):
                    d_re, d_im, d_w = carry
                    mi = mon.monitor_index
                    theta_now = (
                        jnp.asarray(2.0 * np.pi, dtype=dft_dtype)
                        * jnp.asarray(mon.freq_hz, dtype=dft_dtype)
                        * jnp.asarray(t_phys, dtype=dft_dtype)
                    )
                    dft_ph_re = jnp.cos(theta_now)
                    dft_ph_im = jnp.sin(theta_now)
                    w = jnp.asarray(
                        monitor_dft_window_weight(
                            t_phys,
                            mon.dft_t_start,
                            mon.dft_t_end,
                            mon.dft_window_code == 1,
                        ),
                        dtype=jnp.float32,
                    )
                    sample_scale = jnp.asarray(
                        monitor_dft_sample_scale(
                            w,
                            normalization_code=mon.dft_normalization_code,
                            base_dt=dt_scalar,
                            record_interval=mon.dft_record_interval,
                            length_unit=mon.dft_length_unit,
                        ),
                        dtype=dft_dtype,
                    )

                    if mon.is_3d:
                        vecs = self._monitor_dft_vectors_3d(
                            mon, ex, ey, ez, hx, hy, hz
                        )
                    else:
                        vecs = self._monitor_dft_vectors_2d(
                            mon,
                            ex,
                            ey,
                            ez,
                            hx,
                            hy,
                            hz,
                            tm_ez=tm_ez,
                            tm_hx=tm_hx,
                            tm_hy=tm_hy,
                        )
                    comp_mask = mon.dft_component_mask.astype(dft_dtype)[:, None, None]
                    delta_re = jnp.asarray(
                        sample_scale
                        * comp_mask
                        * jnp.einsum("f,cp->cfp", dft_ph_re, vecs.astype(dft_dtype)),
                        dtype=dft_dtype,
                    )
                    delta_im = jnp.asarray(
                        sample_scale
                        * comp_mask
                        * jnp.einsum("f,cp->cfp", dft_ph_im, vecs.astype(dft_dtype)),
                        dtype=dft_dtype,
                    )
                    d_re = d_re.at[mi, :, : mon.freq_count, : mon.dft_point_count].add(
                        delta_re[:, : mon.freq_count, : mon.dft_point_count]
                    )
                    d_im = d_im.at[mi, :, : mon.freq_count, : mon.dft_point_count].add(
                        delta_im[:, : mon.freq_count, : mon.dft_point_count]
                    )
                    d_w = d_w.at[mi, : mon.freq_count].add(
                        jnp.asarray(w, dtype=dft_dtype)
                    )
                    return d_re, d_im, d_w

                dft_vec_re, dft_vec_im, dft_weight_sum = jax.lax.cond(
                    do_dft,
                    _accumulate_dft,
                    lambda carry: carry,
                    (dft_vec_re, dft_vec_im, dft_weight_sum),
                )

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

        snapshot_field = self.config.snapshot_field
        snapshot_enabled = snapshot_field is not None
        snapshot_interval = max(1, int(self.config.snapshot_interval))

        def _snapshot_values(
            ex: jnp.ndarray,
            ey: jnp.ndarray,
            ez: jnp.ndarray,
            hx: jnp.ndarray,
            hy: jnp.ndarray,
            hz: jnp.ndarray,
            *,
            tm_ez: jnp.ndarray | None = None,
            tm_hx: jnp.ndarray | None = None,
            tm_hy: jnp.ndarray | None = None,
        ) -> jnp.ndarray:
            if snapshot_field == "Ex":
                return ex
            if snapshot_field == "Ey":
                return ey
            if snapshot_field == "Ez":
                return ez if tm_ez is None else tm_ez
            if snapshot_field == "Hx":
                return hx if tm_hx is None else tm_hx
            if snapshot_field == "Hy":
                return hy if tm_hy is None else tm_hy
            if snapshot_field == "Hz":
                return hz
            raise ValueError(f"Unsupported snapshot field: {snapshot_field}")

        def run_scan(
            engine_state: EngineState,
            monitor_state: MonitorState,
            coeffs: UpdateCoefficients,
            snapshot_state=None,
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
            tm_h_decay_x, tm_h_source_x = coeffs.tm_h_decay_x, coeffs.tm_h_source_x
            tm_h_decay_y, tm_h_source_y = coeffs.tm_h_decay_y, coeffs.tm_h_source_y
            tm_e_decay_z, tm_e_source_z = coeffs.tm_e_decay_z, coeffs.tm_e_source_z

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
            ex_metal_mask = self.ex_metal_mask
            ey_metal_mask = self.ey_metal_mask
            ez_metal_mask = self.ez_metal_mask
            hx_metal_mask = self.hx_metal_mask
            hy_metal_mask = self.hy_metal_mask
            hz_metal_mask = self.hz_metal_mask
            use_physical_tm_xy = self.use_physical_tm_xy
            use_cpml_tm_xy = self.use_cpml_tm_xy
            use_cpml_3d = self.use_cpml_3d
            full_pec_3d = self.full_pec_3d
            tm_ez_mask = self.tm_ez_mask
            tm_hx_mask = self.tm_hx_mask
            tm_hy_mask = self.tm_hy_mask
            tm_metallic_edges = self.tm_metallic_edges
            tm_full_pec = tm_metallic_edges == frozenset(
                {"left", "right", "bottom", "top"}
            )
            if use_cpml_tm_xy:
                if engine_state.cpml_psi_h_terms.shape != self.cpml_sigma_h_terms.shape:
                    engine_state = engine_state._replace(
                        cpml_psi_h_terms=jnp.zeros_like(self.cpml_sigma_h_terms)
                    )
                if engine_state.cpml_psi_e_terms.shape != self.cpml_sigma_e_terms.shape:
                    engine_state = engine_state._replace(
                        cpml_psi_e_terms=jnp.zeros_like(self.cpml_sigma_e_terms)
                    )
            if use_cpml_3d:
                if len(engine_state.cpml3d_psi_h_terms) != len(
                    self.cpml3d_b_h_terms
                ) or any(
                    psi.shape != coeff.shape
                    for psi, coeff in zip(
                        engine_state.cpml3d_psi_h_terms, self.cpml3d_b_h_terms
                    )
                ):
                    engine_state = engine_state._replace(
                        cpml3d_psi_h_terms=tuple(
                            jnp.zeros_like(term) for term in self.cpml3d_b_h_terms
                        )
                    )
                if len(engine_state.cpml3d_psi_e_terms) != len(
                    self.cpml3d_b_e_terms
                ) or any(
                    psi.shape != coeff.shape
                    for psi, coeff in zip(
                        engine_state.cpml3d_psi_e_terms, self.cpml3d_b_e_terms
                    )
                ):
                    engine_state = engine_state._replace(
                        cpml3d_psi_e_terms=tuple(
                            jnp.zeros_like(term) for term in self.cpml3d_b_e_terms
                        )
                    )

            def _split_carry(state):
                if snapshot_enabled:
                    return state
                eng, mon, mat = state
                return eng, mon, mat, None, None, None, None

            def _merge_carry(
                eng,
                mon,
                mat,
                snap_fields=None,
                snap_steps=None,
                snap_times=None,
                snap_count=None,
            ):
                if snapshot_enabled:
                    return (
                        eng,
                        mon,
                        mat,
                        snap_fields,
                        snap_steps,
                        snap_times,
                        snap_count,
                    )
                return (eng, mon, mat)

            def body_with_coeffs(carry):
                def _pre_e(state):
                    eng, mon, mat, snap_fields, snap_steps, snap_times, snap_count = (
                        _split_carry(state)
                    )
                    abs_step = eng.current_step
                    ex = self._apply_source_group(
                        eng.ex, abs_step, pre_e_ex_batch, pre_e_ex_rest
                    )
                    ey = self._apply_source_group(
                        eng.ey, abs_step, pre_e_ey_batch, pre_e_ey_rest
                    )
                    ez = self._apply_source_group(
                        eng.ez, abs_step, pre_e_ez_batch, pre_e_ez_rest
                    )
                    fp_ex, fp_ey, fp_ez = eng.fp_ex, eng.fp_ey, eng.fp_ez
                    if is_3d and full_pec_3d:
                        fp_ex = fp_ex.at[:-1, :-1, :-1].set(ex)
                        fp_ey = fp_ey.at[:-1, :-1, :-1].set(ey)
                        fp_ez = fp_ez.at[:-1, :-1, :-1].set(ez)
                    eng = eng._replace(
                        ex=ex, ey=ey, ez=ez, fp_ex=fp_ex, fp_ey=fp_ey, fp_ez=fp_ez
                    )
                    return _merge_carry(
                        eng, mon, mat, snap_fields, snap_steps, snap_times, snap_count
                    )

                def _prepare(state):
                    return state, None

                def _update_h(state, _payload):
                    eng, mon, mat, snap_fields, snap_steps, snap_times, snap_count = (
                        _split_carry(state)
                    )
                    ex, ey, ez = eng.ex, eng.ey, eng.ez
                    hx, hy, hz = eng.hx, eng.hy, eng.hz
                    fp_hx, fp_hy, fp_hz = eng.fp_hx, eng.fp_hy, eng.fp_hz
                    cpml_psi_h_terms = eng.cpml_psi_h_terms
                    cpml_psi_e_terms = eng.cpml_psi_e_terms
                    cpml3d_psi_h_terms = eng.cpml3d_psi_h_terms
                    cpml3d_psi_e_terms = eng.cpml3d_psi_e_terms
                    if use_cpml_tm_xy:
                        if cpml_psi_h_terms.shape != self.cpml_sigma_h_terms.shape:
                            cpml_psi_h_terms = jnp.zeros_like(self.cpml_sigma_h_terms)
                        if cpml_psi_e_terms.shape != self.cpml_sigma_e_terms.shape:
                            cpml_psi_e_terms = jnp.zeros_like(self.cpml_sigma_e_terms)
                    if use_cpml_3d:
                        if len(cpml3d_psi_h_terms) != len(self.cpml3d_b_h_terms) or any(
                            psi.shape != coeff.shape
                            for psi, coeff in zip(
                                cpml3d_psi_h_terms, self.cpml3d_b_h_terms
                            )
                        ):
                            cpml3d_psi_h_terms = tuple(
                                jnp.zeros_like(term) for term in self.cpml3d_b_h_terms
                            )
                        if len(cpml3d_psi_e_terms) != len(self.cpml3d_b_e_terms) or any(
                            psi.shape != coeff.shape
                            for psi, coeff in zip(
                                cpml3d_psi_e_terms, self.cpml3d_b_e_terms
                            )
                        ):
                            cpml3d_psi_e_terms = tuple(
                                jnp.zeros_like(term) for term in self.cpml3d_b_e_terms
                            )

                    if is_3d and full_pec_3d:
                        curl_ex, curl_ey, curl_ez = full_pec_curl_e_to_h_3d(
                            eng.fp_ex,
                            eng.fp_ey,
                            eng.fp_ez,
                            resolution,
                            fp_hx.shape,
                            fp_hy.shape,
                            fp_hz.shape,
                        )
                        fp_hx = apply_zero_mask(
                            advance_h_from_coefficients(
                                fp_hx, curl_ex, self.fp_h_decay_x, self.fp_h_source_x
                            ),
                            self.fp_hx_mask,
                        )
                        fp_hy = apply_zero_mask(
                            advance_h_from_coefficients(
                                fp_hy, curl_ey, self.fp_h_decay_y, self.fp_h_source_y
                            ),
                            self.fp_hy_mask,
                        )
                        fp_hz = apply_zero_mask(
                            advance_h_from_coefficients(
                                fp_hz, curl_ez, self.fp_h_decay_z, self.fp_h_source_z
                            ),
                            self.fp_hz_mask,
                        )
                        hx = fp_hx[:-1, :-1, :-1]
                        hy = fp_hy[:-1, :-1, :-1]
                        hz = fp_hz[:-1, :-1, :-1]
                    elif is_3d:
                        if self.use_cpml_3d:
                            hx, hy, hz, cpml3d_psi_h_terms = (
                                cpml_update_h_from_e_3d(
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
                                    a_h_terms=self.cpml3d_a_h_terms,
                                    b_h_terms=self.cpml3d_b_h_terms,
                                    inv_kappa_h_terms=self.cpml3d_inv_kappa_h_terms,
                                    psi_h_terms=cpml3d_psi_h_terms,
                                )
                            )
                        else:
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
                    elif use_physical_tm_xy:
                        if tm_full_pec:
                            curl_tm_hx, curl_tm_hy = full_pec_curl_e_to_h_2d_xy(
                                ez, resolution, hx.shape, hy.shape
                            )
                        elif self.use_cpml_tm_xy:
                            curl_tm_hx, curl_tm_hy, cpml_psi_h_terms = (
                                tm_xy_cpml_curl_e_to_h_2d(
                                    ez,
                                    resolution,
                                    sigma_h_terms=self.cpml_sigma_h_terms,
                                    kappa_h_aux_terms=self.cpml_kappa_h_aux_terms,
                                    alpha_h_terms=self.cpml_alpha_h_terms,
                                    kappa_h_direct_terms=self.cpml_kappa_h_direct_terms,
                                    psi_h_terms=cpml_psi_h_terms,
                                    dt=dt_scalar,
                                )
                            )
                        else:
                            curl_tm_hx, curl_tm_hy = tm_xy_curl_e_to_h_2d(
                                ez,
                                resolution,
                                hx.shape,
                                hy.shape,
                                tm_metallic_edges,
                            )
                        curl_ez = xy_te_curl_e_to_h_2d(
                            ex,
                            ey,
                            resolution,
                            hz.shape,
                        )
                        hx = apply_zero_mask(
                            advance_h_from_coefficients(
                                hx, curl_tm_hx, tm_h_decay_x, tm_h_source_x
                            ),
                            tm_hx_mask,
                        )
                        hy = apply_zero_mask(
                            advance_h_from_coefficients(
                                hy, curl_tm_hy, tm_h_decay_y, tm_h_source_y
                            ),
                            tm_hy_mask,
                        )
                        hz_old = hz
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

                    eng = eng._replace(
                        hx=hx,
                        hy=hy,
                        hz=hz,
                        fp_hx=fp_hx,
                        fp_hy=fp_hy,
                        fp_hz=fp_hz,
                        cpml_psi_h_terms=cpml_psi_h_terms,
                        cpml_psi_e_terms=cpml_psi_e_terms,
                        cpml3d_psi_h_terms=cpml3d_psi_h_terms,
                        cpml3d_psi_e_terms=cpml3d_psi_e_terms,
                    )
                    return _merge_carry(
                        eng, mon, mat, snap_fields, snap_steps, snap_times, snap_count
                    )

                def _post_h(state):
                    eng, mon, mat, snap_fields, snap_steps, snap_times, snap_count = (
                        _split_carry(state)
                    )
                    abs_step = eng.current_step
                    hx_post = self._apply_source_group(
                        eng.hx, abs_step, h_batch_x, h_rest_x
                    )
                    hy_post = self._apply_source_group(
                        eng.hy, abs_step, h_batch_y, h_rest_y
                    )
                    hz = self._apply_source_group(eng.hz, abs_step, h_batch_z, h_rest_z)
                    fp_hx, fp_hy, fp_hz = eng.fp_hx, eng.fp_hy, eng.fp_hz
                    if is_3d and full_pec_3d:
                        fp_hx = fp_hx.at[:-1, :-1, :-1].set(hx_post)
                        fp_hy = fp_hy.at[:-1, :-1, :-1].set(hy_post)
                        fp_hz = fp_hz.at[:-1, :-1, :-1].set(hz)
                    hx = self._apply_metal_mask(hx_post, hx_metal_mask)
                    hy = self._apply_metal_mask(hy_post, hy_metal_mask)
                    hz = self._apply_metal_mask(hz, hz_metal_mask)
                    eng = eng._replace(
                        hx=hx, hy=hy, hz=hz, fp_hx=fp_hx, fp_hy=fp_hy, fp_hz=fp_hz
                    )
                    return _merge_carry(
                        eng, mon, mat, snap_fields, snap_steps, snap_times, snap_count
                    )

                def _update_e(state, _payload):
                    eng, mon, mat, snap_fields, snap_steps, snap_times, snap_count = (
                        _split_carry(state)
                    )
                    ex, ey, ez = eng.ex, eng.ey, eng.ez
                    hx, hy, hz = eng.hx, eng.hy, eng.hz
                    fp_ex, fp_ey, fp_ez = eng.fp_ex, eng.fp_ey, eng.fp_ez
                    cpml_psi_e_terms = eng.cpml_psi_e_terms
                    cpml3d_psi_e_terms = eng.cpml3d_psi_e_terms

                    if is_3d and full_pec_3d:
                        curl_hx, curl_hy, curl_hz = full_pec_curl_h_to_e_3d(
                            eng.fp_hx,
                            eng.fp_hy,
                            eng.fp_hz,
                            resolution,
                            fp_ex.shape,
                            fp_ey.shape,
                            fp_ez.shape,
                        )
                        fp_ex = apply_zero_mask(
                            advance_e_from_coefficients(
                                fp_ex, curl_hx, self.fp_e_decay_x, self.fp_e_source_x
                            ),
                            self.fp_ex_mask,
                        )
                        fp_ey = apply_zero_mask(
                            advance_e_from_coefficients(
                                fp_ey, curl_hy, self.fp_e_decay_y, self.fp_e_source_y
                            ),
                            self.fp_ey_mask,
                        )
                        fp_ez = apply_zero_mask(
                            advance_e_from_coefficients(
                                fp_ez, curl_hz, self.fp_e_decay_z, self.fp_e_source_z
                            ),
                            self.fp_ez_mask,
                        )
                        ex = fp_ex[:-1, :-1, :-1]
                        ey = fp_ey[:-1, :-1, :-1]
                        ez = fp_ez[:-1, :-1, :-1]
                    elif is_3d:
                        if self.use_cpml_3d:
                            ex, ey, ez, cpml3d_psi_e_terms = (
                                cpml_update_e_from_h_3d(
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
                                    a_e_terms=self.cpml3d_a_e_terms,
                                    b_e_terms=self.cpml3d_b_e_terms,
                                    inv_kappa_e_terms=self.cpml3d_inv_kappa_e_terms,
                                    psi_e_terms=cpml3d_psi_e_terms,
                                    metallic_edges=self.cpml3d_metallic_edges,
                                )
                            )
                        else:
                            boundary_views = build_h_boundary_views_for_e_3d(
                                hx, hy, hz, None
                            )
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
                                boundary_views=boundary_views,
                            )
                    elif use_physical_tm_xy:
                        curl_hx, curl_hy = xy_te_curl_h_to_e_2d(
                            hz,
                            resolution,
                            ex.shape,
                            ey.shape,
                            tm_metallic_edges,
                        )
                        if tm_full_pec:
                            curl_tm_ez = full_pec_curl_h_to_e_2d_xy(
                                hx, hy, resolution, ez.shape
                            )
                        elif self.use_cpml_tm_xy:
                            curl_tm_ez, cpml_psi_e_terms = tm_xy_cpml_curl_h_to_e_2d(
                                hx,
                                hy,
                                resolution,
                                ez.shape,
                                tm_metallic_edges,
                                sigma_e_terms=self.cpml_sigma_e_terms,
                                kappa_e_terms=self.cpml_kappa_e_terms,
                                alpha_e_terms=self.cpml_alpha_e_terms,
                                psi_e_terms=cpml_psi_e_terms,
                                dt=dt_scalar,
                            )
                        else:
                            curl_tm_ez = tm_xy_curl_h_to_e_2d(
                                hx,
                                hy,
                                resolution,
                                ez.shape,
                                tm_metallic_edges,
                            )
                        ex_old, ey_old = ex, ey
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
                        ez = apply_zero_mask(
                            advance_e_from_coefficients(
                                ez, curl_tm_ez, tm_e_decay_z, tm_e_source_z
                            ),
                            tm_ez_mask,
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

                    eng = eng._replace(
                        ex=ex,
                        ey=ey,
                        ez=ez,
                        fp_ex=fp_ex,
                        fp_ey=fp_ey,
                        fp_ez=fp_ez,
                        cpml_psi_e_terms=cpml_psi_e_terms,
                        cpml3d_psi_e_terms=cpml3d_psi_e_terms,
                    )
                    return _merge_carry(
                        eng, mon, mat, snap_fields, snap_steps, snap_times, snap_count
                    )

                def _post_e(state):
                    eng, mon, mat, snap_fields, snap_steps, snap_times, snap_count = (
                        _split_carry(state)
                    )
                    abs_step = eng.current_step
                    ex = self._apply_source_group(eng.ex, abs_step, e_batch_x, e_rest_x)
                    ey = self._apply_source_group(eng.ey, abs_step, e_batch_y, e_rest_y)
                    ez = self._apply_source_group(eng.ez, abs_step, e_batch_z, e_rest_z)
                    fp_ex, fp_ey, fp_ez = eng.fp_ex, eng.fp_ey, eng.fp_ez
                    if is_3d and full_pec_3d:
                        fp_ex = fp_ex.at[:-1, :-1, :-1].set(ex)
                        fp_ey = fp_ey.at[:-1, :-1, :-1].set(ey)
                        fp_ez = fp_ez.at[:-1, :-1, :-1].set(ez)
                    ex = self._apply_metal_mask(ex, ex_metal_mask)
                    ey = self._apply_metal_mask(ey, ey_metal_mask)
                    ez = self._apply_metal_mask(ez, ez_metal_mask)
                    if use_physical_tm_xy:
                        ez = jnp.where(tm_ez_mask, jnp.asarray(0.0, dtype=ez.dtype), ez)
                    eng = eng._replace(
                        ex=ex, ey=ey, ez=ez, fp_ex=fp_ex, fp_ey=fp_ey, fp_ez=fp_ez
                    )
                    return _merge_carry(
                        eng, mon, mat, snap_fields, snap_steps, snap_times, snap_count
                    )

                def _finalize(state):
                    eng, mon, mat, snap_fields, snap_steps, snap_times, snap_count = (
                        _split_carry(state)
                    )
                    abs_step = eng.current_step
                    ex, ey, ez = eng.ex, eng.ey, eng.ez
                    hx, hy, hz = eng.hx, eng.hy, eng.hz
                    tm_ez, tm_hx, tm_hy = eng.tm_ez, eng.tm_hx, eng.tm_hy

                    mat, _ = material_model.update(mat, ex, ey, ez, abs_step)
                    t_phys = eng.t + dt_scalar
                    mon_tm_ez = ez if use_physical_tm_xy else tm_ez
                    mon_tm_hx = hx if use_physical_tm_xy else tm_hx
                    mon_tm_hy = hy if use_physical_tm_xy else tm_hy
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
                        tm_ez=mon_tm_ez,
                        tm_hx=mon_tm_hx,
                        tm_hy=mon_tm_hy,
                        batched_mon=batched_mon,
                        monitors_2d=monitors_2d,
                    )

                    new_tm_ez = ez if use_physical_tm_xy else tm_ez
                    new_tm_hx = hx if use_physical_tm_xy else tm_hx
                    new_tm_hy = hy if use_physical_tm_xy else tm_hy
                    eng = eng._replace(
                        tm_ez=new_tm_ez,
                        tm_hx=new_tm_hx,
                        tm_hy=new_tm_hy,
                        t=eng.t + dt,
                        current_step=eng.current_step + jnp.array(1, dtype=jnp.int32),
                    )
                    if not snapshot_enabled:
                        return _merge_carry(eng, mon, mat)

                    new_step = eng.current_step
                    new_time = eng.t
                    should_snapshot = (new_step % snapshot_interval) == 0
                    slot = jnp.minimum(snap_count, snap_fields.shape[0] - 1)
                    snapshot_values = _snapshot_values(
                        ex,
                        ey,
                        ez,
                        hx,
                        hy,
                        hz,
                        tm_ez=new_tm_ez if use_physical_tm_xy else None,
                        tm_hx=new_tm_hx if use_physical_tm_xy else None,
                        tm_hy=new_tm_hy if use_physical_tm_xy else None,
                    )
                    field_start = (slot,) + (0,) * snapshot_values.ndim
                    snap_fields = jax.lax.cond(
                        should_snapshot,
                        lambda buf: jax.lax.dynamic_update_slice(
                            buf,
                            snapshot_values[jnp.newaxis, ...],
                            field_start,
                        ),
                        lambda buf: buf,
                        snap_fields,
                    )
                    snap_steps = jax.lax.cond(
                        should_snapshot,
                        lambda buf: jax.lax.dynamic_update_slice(
                            buf,
                            new_step[jnp.newaxis],
                            (slot,),
                        ),
                        lambda buf: buf,
                        snap_steps,
                    )
                    snap_times = jax.lax.cond(
                        should_snapshot,
                        lambda buf: jax.lax.dynamic_update_slice(
                            buf,
                            new_time[jnp.newaxis],
                            (slot,),
                        ),
                        lambda buf: buf,
                        snap_times,
                    )
                    snap_count = snap_count + should_snapshot.astype(jnp.int32)
                    return _merge_carry(
                        eng, mon, mat, snap_fields, snap_steps, snap_times, snap_count
                    )

                return run_step_sequence(
                    carry,
                    pre_e=_pre_e,
                    prepare=_prepare,
                    update_h=_update_h,
                    post_h=_post_h,
                    update_e=_update_e,
                    post_e=_post_e,
                    finalize=_finalize,
                )

            if self.config.loop_kind == "scan":

                def _scan_body(carry, _unused):
                    return body_with_coeffs(carry), None

                init_carry = (
                    (engine_state, monitor_state, material_state0, *snapshot_state)
                    if snapshot_enabled
                    else (engine_state, monitor_state, material_state0)
                )
                scan_out, _ = jax.lax.scan(
                    _scan_body,
                    init_carry,
                    xs=None,
                    length=self.config.num_steps,
                )
            else:
                init_carry = (
                    (engine_state, monitor_state, material_state0, *snapshot_state)
                    if snapshot_enabled
                    else (engine_state, monitor_state, material_state0)
                )
                scan_out = jax.lax.fori_loop(
                    0,
                    self.config.num_steps,
                    lambda _i, c: body_with_coeffs(c),
                    init_carry,
                )
            if snapshot_enabled:
                (
                    engine_final,
                    monitor_final,
                    material_final,
                    snap_fields,
                    snap_steps,
                    snap_times,
                    snap_count,
                ) = scan_out
                return (
                    engine_final,
                    monitor_final,
                    material_final,
                    (snap_fields, snap_steps, snap_times, snap_count),
                )
            engine_final, monitor_final, material_final = scan_out
            return engine_final, monitor_final, material_final, None

        # Use function-style JIT wrapping for compatibility with older JAX
        # versions where decorator kwargs require the callable as first arg.
        if self.use_physical_tm_xy:
            donate_argnums = (1, 3) if snapshot_enabled else (1,)
        else:
            donate_argnums = (0, 1, 3) if snapshot_enabled else (0, 1)
        self._compiled_scan = jax.jit(run_scan, donate_argnums=donate_argnums)
        self._compile_count += 1

    @property
    def compile_count(self) -> int:
        return self._compile_count

    def run(
        self,
        engine_state: EngineState,
        monitor_state: MonitorState | None = None,
    ) -> tuple[
        EngineState,
        MonitorState,
        MaterialState,
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray] | None,
    ]:
        """Execute the compiled simulation loop."""
        if monitor_state is None:
            if self.monitor_specs:
                max_records = max(
                    1, monitor_state_size(self.monitor_specs, self.config.num_steps)
                )
                max_freq = monitor_frequency_size(self.monitor_specs)
                max_points = monitor_dft_point_size(self.monitor_specs)
                dft_dtype = monitor_dft_accumulator_dtype()
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
                        dtype=dft_dtype,
                    ),
                    dft_vec_im=jnp.zeros(
                        (len(self.monitor_specs), 6, max_freq, max_points),
                        dtype=dft_dtype,
                    ),
                    dft_weight_sum=jnp.zeros(
                        (len(self.monitor_specs), max_freq), dtype=dft_dtype
                    ),
                )
            else:
                dft_dtype = monitor_dft_accumulator_dtype()
                monitor_state = MonitorState(
                    powers=jnp.zeros((0, 0), dtype=jnp.float32),
                    timestamps=jnp.zeros((0, 0), dtype=jnp.float32),
                    counts=jnp.zeros((0,), dtype=jnp.int32),
                    freq_flux_re=jnp.zeros((0, 0), dtype=jnp.float32),
                    freq_flux_im=jnp.zeros((0, 0), dtype=jnp.float32),
                    freq_phase_re=jnp.zeros((0, 0), dtype=jnp.float32),
                    freq_phase_im=jnp.zeros((0, 0), dtype=jnp.float32),
                    dft_vec_re=jnp.zeros((0, 0, 0, 0), dtype=dft_dtype),
                    dft_vec_im=jnp.zeros((0, 0, 0, 0), dtype=dft_dtype),
                    dft_weight_sum=jnp.zeros((0, 0), dtype=dft_dtype),
                )

        if self._compiled_scan is None:
            self._build_scan()

        snapshot_state = self._empty_snapshot_state()
        if snapshot_state is None:
            eng, mon, mat, snapshots = self._compiled_scan(
                engine_state,
                monitor_state,
                self._update_coefficients(),
            )
        else:
            eng, mon, mat, snapshots = self._compiled_scan(
                engine_state,
                monitor_state,
                self._update_coefficients(),
                snapshot_state,
            )
        return eng, mon, mat, snapshots

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
            if spec.accumulate_frequency and spec.freq_count > 0:
                re = np.asarray(
                    monitor_state.freq_flux_re[spec.monitor_index, : spec.freq_count],
                    dtype=np.float32,
                )
                im = np.asarray(
                    monitor_state.freq_flux_im[spec.monitor_index, : spec.freq_count],
                    dtype=np.float32,
                )
                dev.power_spectrum = (re + 1j * im).astype(np.complex64)
            else:
                dev.power_spectrum = np.zeros((0,), dtype=np.complex64)
            dev._frequency_flux_spectrum_legacy = dev.power_spectrum

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
                dev._dft_base_dt = float(self.config.dt)
                if spec.is_3d:
                    dev._compiled_dft_shape_3d = (
                        int(spec.min_dim0),
                        int(spec.min_dim1),
                    )
                    axis = str(getattr(dev, "plane_normal", "z")).lower()
                    dev._compiled_dft_plane_axes = {
                        "x": ("z", "y"),
                        "y": ("z", "x"),
                        "z": ("y", "x"),
                    }.get(axis, ("y", "x"))
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
                try:
                    phasor_flux = np.asarray(dev.get_dft_flux(), dtype=np.float64)
                except ValueError:
                    pass
                else:
                    dev._frequency_flux_spectrum_legacy = phasor_flux.astype(
                        np.complex64
                    )
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


def compile_simulation(
    design, sources, monitors, boundaries, run_cfg
) -> CompiledSimulation:
    """Build a CompiledSimulation from design/sources/monitors/boundaries and a run config.

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
    del design

    fields = run_cfg.fields
    resolution = float(run_cfg.resolution)
    dt = float(run_cfg.dt)
    num_steps = int(run_cfg.num_steps)
    total_steps = int(getattr(run_cfg, "total_steps", num_steps))
    t0 = float(getattr(run_cfg, "t0", 0.0))

    source_specs = compile_source_specs(
        sources=sources,
        fields=fields,
        dt=dt,
        resolution=resolution,
        num_steps=num_steps,
        t0=t0,
        total_steps=total_steps,
    )

    monitor_specs, _ = compile_monitor_specs(
        monitors=monitors,
        fields=fields,
        resolution=resolution,
        num_steps=num_steps,
        dt=dt,
    )

    monitor_devices = tuple(monitors)

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
        snapshot_field=getattr(run_cfg, "snapshot_field", None),
        snapshot_interval=int(getattr(run_cfg, "snapshot_interval", 0) or 0),
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
    use_physical_tm_xy = False
    use_cpml_tm_xy = False
    use_cpml_3d = False
    empty_cpml_terms = jnp.zeros((2, 0, 0), dtype=jnp.float32)
    cpml_sigma_h_terms = empty_cpml_terms
    cpml_kappa_h_aux_terms = empty_cpml_terms
    cpml_alpha_h_terms = empty_cpml_terms
    cpml_kappa_h_direct_terms = empty_cpml_terms
    cpml_sigma_e_terms = empty_cpml_terms
    cpml_kappa_e_terms = empty_cpml_terms
    cpml_alpha_e_terms = empty_cpml_terms
    cpml3d_a_h_terms = _empty_cpml_3d_terms(jnp.float32)
    cpml3d_b_h_terms = _empty_cpml_3d_terms(jnp.float32)
    cpml3d_inv_kappa_h_terms = _empty_cpml_3d_terms(jnp.float32)
    cpml3d_a_e_terms = _empty_cpml_3d_terms(jnp.float32)
    cpml3d_b_e_terms = _empty_cpml_3d_terms(jnp.float32)
    cpml3d_inv_kappa_e_terms = _empty_cpml_3d_terms(jnp.float32)
    cpml3d_metallic_edges = frozenset()
    if bool(run_cfg.is_3d):
        cpml3d_metallic_edges = frozenset(
            resolve_metallic_edges(boundaries, is_3d=True)
        )
        use_cpml_3d = bool(
            getattr(fields, "has_cpml", False) and getattr(fields, "pml_data", None)
        )
        if use_cpml_3d:
            terms = build_cpml_3d_terms(fields.pml_data, dt=run_cfg.dt)
            if terms is not None:
                cpml3d_a_h_terms = terms.a_h_terms
                cpml3d_b_h_terms = terms.b_h_terms
                cpml3d_inv_kappa_h_terms = terms.inv_kappa_h_terms
                cpml3d_a_e_terms = terms.a_e_terms
                cpml3d_b_e_terms = terms.b_e_terms
                cpml3d_inv_kappa_e_terms = terms.inv_kappa_e_terms
    if not bool(run_cfg.is_3d) and run_cfg.plane_2d == "xy":
        tm_ez_shape = tuple(int(v) for v in fields.Ez.shape)
        # The physical full-state TMz lattice is the only supported xy-TM update
        # path.
        use_physical_tm_xy = True
        use_cpml_tm_xy = bool(
            getattr(fields, "has_cpml", False) and getattr(fields, "pml_data", None)
        )
        if use_cpml_tm_xy:
            # Keep CPML on the native full-TM representation only. This is closer
            # to FDTDX's architecture and avoids maintaining a second xy-TM CPML
            # implementation with slightly different staggering semantics.
            use_physical_tm_xy = True
            terms = build_tm_xy_cpml_terms(
                fields.pml_data.get("tm_xy_cpml"),
                ez_shape=tm_ez_shape,
            )
            if terms is None:
                use_cpml_tm_xy = False
            else:
                cpml_sigma_h_terms = terms.sigma_h_terms
                cpml_kappa_h_aux_terms = terms.kappa_h_aux_terms
                cpml_alpha_h_terms = terms.alpha_h_terms
                cpml_kappa_h_direct_terms = terms.kappa_h_direct_terms
                cpml_sigma_e_terms = terms.sigma_e_terms
                cpml_kappa_e_terms = terms.kappa_e_terms
                cpml_alpha_e_terms = terms.alpha_e_terms

    if use_physical_tm_xy:
        total_sigma = jnp.asarray(
            getattr(fields, "total_conductivity", fields.conductivity)
        )
        sigma_base = (
            total_sigma * jnp.asarray(fields.permeability) * ops.MU_0 / ops.EPS_0
        )
        tm_sigma_m_hx = sample_voxel_grid_at_tm_xy_full_component_2d(sigma_base, "Hx")
        tm_sigma_m_hy = sample_voxel_grid_at_tm_xy_full_component_2d(sigma_base, "Hy")
        tm_eps_z = sample_voxel_grid_at_tm_xy_full_component_2d(
            fields.permittivity, "Ez"
        )
        tm_sig_z = sample_voxel_grid_at_tm_xy_full_component_2d(total_sigma, "Ez")
        tm_h_decay_x, tm_h_source_x, _ = ops.precompute_h_update_coefficients(
            tm_sigma_m_hx, dt
        )
        tm_h_decay_y, tm_h_source_y, _ = ops.precompute_h_update_coefficients(
            tm_sigma_m_hy, dt
        )
        tm_e_decay_z, tm_e_source_z, _ = ops.precompute_e_update_coefficients(
            shape=tm_eps_z.shape,
            conductivity=tm_sig_z,
            permittivity=tm_eps_z,
            dt=dt,
            region=(slice(None), slice(None)),
        )
        metallic_edges_2d = frozenset(resolve_metallic_edges(boundaries, is_3d=False))
        tm_masks = full_tm_2d_xy_masks(
            tuple(fields.permittivity.shape), metallic_edges_2d
        )
        tm_ez_mask = tm_masks["Ez"]
        tm_hx_mask = tm_masks["Hx"]
        tm_hy_mask = tm_masks["Hy"]
    else:
        tm_h_decay_x = jnp.zeros((0, 0), dtype=jnp.float32)
        tm_h_source_x = jnp.zeros((0, 0), dtype=jnp.float32)
        tm_h_decay_y = jnp.zeros((0, 0), dtype=jnp.float32)
        tm_h_source_y = jnp.zeros((0, 0), dtype=jnp.float32)
        tm_e_decay_z = jnp.zeros((0, 0), dtype=jnp.float32)
        tm_e_source_z = jnp.zeros((0, 0), dtype=jnp.float32)
        tm_ez_mask = jnp.zeros(tuple(fields.Ez.shape), dtype=bool)
        tm_hx_mask = jnp.zeros((0, 0), dtype=bool)
        tm_hy_mask = jnp.zeros((0, 0), dtype=bool)
        metallic_edges_2d = frozenset()
    full_pec_3d = bool(has_full_pec_3d(boundaries))
    if bool(run_cfg.is_3d) and full_pec_3d:
        fp_state = initialize_full_pec_3d_state(fields)
        fp_h_decay_x, fp_h_source_x, _ = ops.precompute_h_update_coefficients(
            fp_state.sigma_m_hx, dt
        )
        fp_h_decay_y, fp_h_source_y, _ = ops.precompute_h_update_coefficients(
            fp_state.sigma_m_hy, dt
        )
        fp_h_decay_z, fp_h_source_z, _ = ops.precompute_h_update_coefficients(
            fp_state.sigma_m_hz, dt
        )
        fp_e_decay_x, fp_e_source_x, _ = ops.precompute_e_update_coefficients(
            shape=fp_state.Ex.shape,
            conductivity=fp_state.sig_x_region,
            permittivity=fp_state.eps_x_region,
            dt=dt,
            region=(slice(1, -1), slice(1, -1), slice(None)),
        )
        fp_e_decay_y, fp_e_source_y, _ = ops.precompute_e_update_coefficients(
            shape=fp_state.Ey.shape,
            conductivity=fp_state.sig_y_region,
            permittivity=fp_state.eps_y_region,
            dt=dt,
            region=(slice(1, -1), slice(None), slice(1, -1)),
        )
        fp_e_decay_z, fp_e_source_z, _ = ops.precompute_e_update_coefficients(
            shape=fp_state.Ez.shape,
            conductivity=fp_state.sig_z_region,
            permittivity=fp_state.eps_z_region,
            dt=dt,
            region=(slice(None), slice(1, -1), slice(1, -1)),
        )
        fp_ex_mask = fp_state.masks["Ex"]
        fp_ey_mask = fp_state.masks["Ey"]
        fp_ez_mask = fp_state.masks["Ez"]
        fp_hx_mask = fp_state.masks["Hx"]
        fp_hy_mask = fp_state.masks["Hy"]
        fp_hz_mask = fp_state.masks["Hz"]
    else:
        fp_h_decay_x = jnp.zeros((0, 0, 0), dtype=jnp.float32)
        fp_h_source_x = jnp.zeros((0, 0, 0), dtype=jnp.float32)
        fp_h_decay_y = jnp.zeros((0, 0, 0), dtype=jnp.float32)
        fp_h_source_y = jnp.zeros((0, 0, 0), dtype=jnp.float32)
        fp_h_decay_z = jnp.zeros((0, 0, 0), dtype=jnp.float32)
        fp_h_source_z = jnp.zeros((0, 0, 0), dtype=jnp.float32)
        fp_e_decay_x = jnp.zeros((0, 0, 0), dtype=jnp.float32)
        fp_e_source_x = jnp.zeros((0, 0, 0), dtype=jnp.float32)
        fp_e_decay_y = jnp.zeros((0, 0, 0), dtype=jnp.float32)
        fp_e_source_y = jnp.zeros((0, 0, 0), dtype=jnp.float32)
        fp_e_decay_z = jnp.zeros((0, 0, 0), dtype=jnp.float32)
        fp_e_source_z = jnp.zeros((0, 0, 0), dtype=jnp.float32)
        fp_ex_mask = jnp.zeros((0, 0, 0), dtype=bool)
        fp_ey_mask = jnp.zeros((0, 0, 0), dtype=bool)
        fp_ez_mask = jnp.zeros((0, 0, 0), dtype=bool)
        fp_hx_mask = jnp.zeros((0, 0, 0), dtype=bool)
        fp_hy_mask = jnp.zeros((0, 0, 0), dtype=bool)
        fp_hz_mask = jnp.zeros((0, 0, 0), dtype=bool)
    metallic_masks = create_metallic_boundary_masks(
        fields,
        boundaries,
        is_3d=bool(run_cfg.is_3d),
        plane_2d=run_cfg.plane_2d,
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
        tm_h_decay_x=tm_h_decay_x,
        tm_h_source_x=tm_h_source_x,
        tm_h_decay_y=tm_h_decay_y,
        tm_h_source_y=tm_h_source_y,
        tm_e_decay_z=tm_e_decay_z,
        tm_e_source_z=tm_e_source_z,
        tm_ez_mask=tm_ez_mask,
        tm_hx_mask=tm_hx_mask,
        tm_hy_mask=tm_hy_mask,
        tm_metallic_edges=metallic_edges_2d,
        use_physical_tm_xy=use_physical_tm_xy,
        use_cpml_tm_xy=use_cpml_tm_xy,
        cpml_sigma_h_terms=cpml_sigma_h_terms,
        cpml_kappa_h_aux_terms=cpml_kappa_h_aux_terms,
        cpml_alpha_h_terms=cpml_alpha_h_terms,
        cpml_kappa_h_direct_terms=cpml_kappa_h_direct_terms,
        cpml_sigma_e_terms=cpml_sigma_e_terms,
        cpml_kappa_e_terms=cpml_kappa_e_terms,
        cpml_alpha_e_terms=cpml_alpha_e_terms,
        use_cpml_3d=use_cpml_3d,
        cpml3d_a_h_terms=cpml3d_a_h_terms,
        cpml3d_b_h_terms=cpml3d_b_h_terms,
        cpml3d_inv_kappa_h_terms=cpml3d_inv_kappa_h_terms,
        cpml3d_a_e_terms=cpml3d_a_e_terms,
        cpml3d_b_e_terms=cpml3d_b_e_terms,
        cpml3d_inv_kappa_e_terms=cpml3d_inv_kappa_e_terms,
        cpml3d_metallic_edges=cpml3d_metallic_edges,
        full_pec_3d=full_pec_3d,
        fp_h_decay_x=fp_h_decay_x,
        fp_h_source_x=fp_h_source_x,
        fp_h_decay_y=fp_h_decay_y,
        fp_h_source_y=fp_h_source_y,
        fp_h_decay_z=fp_h_decay_z,
        fp_h_source_z=fp_h_source_z,
        fp_e_decay_x=fp_e_decay_x,
        fp_e_source_x=fp_e_source_x,
        fp_e_decay_y=fp_e_decay_y,
        fp_e_source_y=fp_e_source_y,
        fp_e_decay_z=fp_e_decay_z,
        fp_e_source_z=fp_e_source_z,
        fp_ex_mask=fp_ex_mask,
        fp_ey_mask=fp_ey_mask,
        fp_ez_mask=fp_ez_mask,
        fp_hx_mask=fp_hx_mask,
        fp_hy_mask=fp_hy_mask,
        fp_hz_mask=fp_hz_mask,
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
        ex_metal_mask=metallic_masks["Ex"],
        ey_metal_mask=metallic_masks["Ey"],
        ez_metal_mask=metallic_masks["Ez"],
        hx_metal_mask=metallic_masks["Hx"],
        hy_metal_mask=metallic_masks["Hy"],
        hz_metal_mask=metallic_masks["Hz"],
    )
