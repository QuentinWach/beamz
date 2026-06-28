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
import warnings
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Literal, NamedTuple

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
from beamz.shared_kernels import (
    CPML_3D_E_DERIVATIVES,
    CPML_3D_H_DERIVATIVES,
    advance_e_from_coefficients,
    advance_h_from_coefficients,
    apply_zero_mask,
    build_cpml_3d_primitive_terms,
    build_cpml_3d_terms,
    build_tm_xy_cpml_terms,
    cpml_precompute_native_terms,
    fit_array_to_shape,
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
    cpml_update_e_from_h_3d_packed_psi,
    cpml_update_h_from_e_3d,
    cpml_update_h_from_e_3d_packed_psi,
    create_metallic_boundary_masks,
    full_pec_curl_e_to_h_2d_xy,
    full_pec_curl_h_to_e_2d_xy,
    full_pec_e_update_coefficients_3d,
    full_pec_h_update_coefficients_3d,
    full_pec_update_e_from_h_3d,
    full_pec_update_h_from_e_3d,
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


def _array_nbytes(arr) -> int:
    try:
        shape = tuple(int(v) for v in arr.shape)
        dtype = np.dtype(arr.dtype)
    except Exception:
        return 0
    return int(np.prod(shape, dtype=np.int64) * dtype.itemsize)


def _array_memory_entry(name: str, arr, category: str, residency: str) -> dict | None:
    try:
        shape = tuple(int(v) for v in arr.shape)
        dtype = np.dtype(arr.dtype)
    except Exception:
        return None
    return {
        "name": str(name),
        "category": str(category),
        "residency": str(residency),
        "shape": list(shape),
        "dtype": str(dtype),
        "bytes": _array_nbytes(arr),
    }


def _add_array_entries(
    entries: list[dict],
    name: str,
    value,
    *,
    category: str,
    residency: str = "persistent",
) -> None:
    entry = _array_memory_entry(name, value, category, residency)
    if entry is not None:
        entries.append(entry)
        return
    if isinstance(value, tuple):
        for idx, item in enumerate(value):
            _add_array_entries(
                entries,
                f"{name}[{idx}]",
                item,
                category=category,
                residency=residency,
            )


def _memory_report(entries: list[dict]) -> dict:
    totals_by_category: dict[str, int] = {}
    totals_by_residency: dict[str, int] = {}
    for entry in entries:
        byte_count = int(entry["bytes"])
        totals_by_category[entry["category"]] = (
            totals_by_category.get(entry["category"], 0) + byte_count
        )
        totals_by_residency[entry["residency"]] = (
            totals_by_residency.get(entry["residency"], 0) + byte_count
        )
    total = int(sum(int(entry["bytes"]) for entry in entries))
    return {
        "total_bytes": total,
        "total_gib": total / 1024**3,
        "totals_by_category": totals_by_category,
        "totals_by_residency": totals_by_residency,
        "entries": entries,
    }


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


def _cpml_3d_h_term_shapes(hx, hy, hz) -> tuple[tuple[int, ...], ...]:
    return (hx.shape, hx.shape, hy.shape, hy.shape, hz.shape, hz.shape)


def _cpml_3d_e_term_shapes(ex, ey, ez) -> tuple[tuple[int, ...], ...]:
    return (ex.shape, ex.shape, ey.shape, ey.shape, ez.shape, ez.shape)


def _cpml_3d_shapes_match(terms, shapes) -> bool:
    return len(terms) == len(shapes) and all(
        tuple(term.shape) == tuple(shape) for term, shape in zip(terms, shapes)
    )


def _zeros_for_cpml_3d_shapes(shapes, dtype) -> tuple[jnp.ndarray, ...]:
    return tuple(jnp.zeros(shape, dtype=dtype) for shape in shapes)


def _cpml_active_slab_counts(a_term, inv_kappa_term, axis: int) -> tuple[int, int]:
    a = np.asarray(a_term)
    inv_kappa = np.asarray(inv_kappa_term)
    if a.size == 0:
        return 0, 0
    active = (np.abs(a) > 1e-30) | (np.abs(inv_kappa - 1.0) > 1e-7)
    reduce_axes = tuple(i for i in range(active.ndim) if i != int(axis))
    if reduce_axes:
        active = np.any(active, axis=reduce_axes)
    active = np.asarray(active, dtype=bool)
    low = 0
    while low < active.size and bool(active[low]):
        low += 1
    high = 0
    while high < active.size - low and bool(active[active.size - high - 1]):
        high += 1
    return int(low), int(high)


def _cpml_packed_slab_shape(
    full_shape: tuple[int, int, int], axis: int, low: int, high: int
) -> tuple[int, int, int]:
    if int(low) + int(high) > int(full_shape[int(axis)]):
        raise ValueError(
            "Packed CPML slab counts exceed derivative axis length: "
            f"shape={full_shape}, axis={axis}, low={low}, high={high}"
        )
    shape = list(full_shape)
    shape[int(axis)] = int(low) + int(high)
    return tuple(shape)


def _build_cpml_packed_slab_specs(
    a_terms: tuple[jnp.ndarray, ...],
    inv_kappa_terms: tuple[jnp.ndarray, ...],
    full_shapes: tuple[tuple[int, int, int], ...],
    derivative_specs,
) -> tuple[CpmlPackedSlabSpec, ...]:
    axis_index = {"z": 0, "y": 1, "x": 2}
    specs = []
    for a_term, inv_kappa_term, full_shape, derivative_spec in zip(
        a_terms, inv_kappa_terms, full_shapes, derivative_specs, strict=True
    ):
        axis = axis_index[derivative_spec.derivative_axis]
        low, high = _cpml_active_slab_counts(a_term, inv_kappa_term, axis)
        specs.append(
            CpmlPackedSlabSpec(
                axis=axis,
                low=low,
                high=high,
                shape=_cpml_packed_slab_shape(full_shape, axis, low, high),
            )
        )
    return tuple(specs)


def _cpml_packed_slab_shapes(
    specs: tuple[CpmlPackedSlabSpec, ...],
) -> tuple[tuple[int, int, int], ...]:
    return tuple(spec.shape for spec in specs)


def _cpml_psi_state_bytes(shapes, dtype) -> int:
    dtype = np.dtype(dtype)
    return int(sum(int(np.prod(shape, dtype=np.int64)) for shape in shapes)) * int(
        dtype.itemsize
    )


def _should_pack_cpml_3d_psi(shapes, dtype) -> bool:
    raw = os.getenv("BEAMZ_CPML_PACKED_PSI", "auto").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    threshold_raw = os.getenv("BEAMZ_CPML_PACKED_PSI_MIN_GIB", "1.0").strip()
    try:
        threshold_gib = float(threshold_raw)
    except ValueError:
        threshold_gib = 1.0
    threshold_bytes = max(0, int(threshold_gib * 1024**3))
    return _cpml_psi_state_bytes(shapes, dtype) >= threshold_bytes


def _empty_like_rank(arr: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros((0,) * arr.ndim, dtype=arr.dtype)


def _snapshot_warning_threshold_bytes() -> int:
    raw = os.getenv("BEAMZ_SNAPSHOT_WARN_GIB", "2.0").strip()
    try:
        gib = float(raw)
    except ValueError:
        gib = 2.0
    return max(0, int(gib * 1024**3))


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
    h_sigma_m_x: jnp.ndarray
    h_decay_y: jnp.ndarray
    h_source_y: jnp.ndarray
    h_source_lossless_y: jnp.ndarray
    h_sigma_m_y: jnp.ndarray
    h_decay_z: jnp.ndarray
    h_source_z: jnp.ndarray
    h_source_lossless_z: jnp.ndarray
    h_sigma_m_z: jnp.ndarray
    e_decay_x: jnp.ndarray
    e_source_x: jnp.ndarray
    e_source_lossless_x: jnp.ndarray
    e_conductivity_x: jnp.ndarray
    e_inv_permittivity_x: jnp.ndarray
    e_decay_y: jnp.ndarray
    e_source_y: jnp.ndarray
    e_source_lossless_y: jnp.ndarray
    e_conductivity_y: jnp.ndarray
    e_inv_permittivity_y: jnp.ndarray
    e_decay_z: jnp.ndarray
    e_source_z: jnp.ndarray
    e_source_lossless_z: jnp.ndarray
    e_conductivity_z: jnp.ndarray
    e_inv_permittivity_z: jnp.ndarray
    tm_h_decay_x: jnp.ndarray
    tm_h_source_x: jnp.ndarray
    tm_h_decay_y: jnp.ndarray
    tm_h_source_y: jnp.ndarray
    tm_e_decay_z: jnp.ndarray
    tm_e_source_z: jnp.ndarray


class RunState(NamedTuple):
    """Auxiliary run counters."""

    compile_count: jnp.ndarray


class CpmlPackedSlabSpec(NamedTuple):
    """Static packed low/high CPML slab layout for one derivative term."""

    axis: int
    low: int
    high: int
    shape: tuple[int, int, int]


@dataclass(frozen=True)
class ShardingConfig:
    """Optional single-host JAX sharding configuration for compiled 3D runs."""

    enabled: bool = False
    axis: Literal["auto", "z", "y", "x"] = "auto"
    num_devices: int | None = None
    backend: Literal["cpu", "gpu"] | None = None


@dataclass(frozen=True)
class StorageLayout:
    """Logical-to-storage layout for padded compiled component arrays."""

    enabled: bool
    pec_full_storage: bool
    logical_base_shape: tuple[int, int, int]
    axis_name: str
    axis: int
    num_devices: int
    backend: str | None
    logical_shapes: dict[str, tuple[int, ...]]
    active_shapes: dict[str, tuple[int, ...]]
    storage_shapes: dict[str, tuple[int, ...]]
    padding: dict[str, tuple[tuple[int, int], ...]]
    valid_masks: dict[str, jnp.ndarray | None]


_COMPONENT_NAMES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
_AXIS_TO_INDEX = {"z": 0, "y": 1, "x": 2}
_INDEX_TO_AXIS = ("z", "y", "x")
_MESH_AXIS = "fdtd"


def _disabled_sharding_config() -> ShardingConfig:
    return ShardingConfig(enabled=False)


def normalize_sharding_config(value) -> ShardingConfig:
    """Normalize public sharding input into a stable config object."""

    if value is None or value is False:
        return _disabled_sharding_config()
    if isinstance(value, ShardingConfig):
        return value
    if value is True:
        return ShardingConfig(enabled=True)
    if isinstance(value, dict):
        raw = dict(value)
        enabled = bool(raw.pop("enabled", True))
        return ShardingConfig(enabled=enabled, **raw)
    raise TypeError(
        "sharding must be None, bool, dict, or beamz.simulation.ShardingConfig"
    )


def sharding_cache_token(value) -> tuple:
    cfg = normalize_sharding_config(value)
    return (bool(cfg.enabled), cfg.axis, cfg.num_devices, cfg.backend)


def _select_sharding_axis(axis: str, grid_shape: tuple[int, int, int]) -> str:
    axis = str(axis).lower()
    if axis != "auto":
        if axis not in _AXIS_TO_INDEX:
            raise ValueError("sharding axis must be one of: 'auto', 'z', 'y', 'x'")
        return axis
    lengths = tuple(int(v) for v in grid_shape)
    best = max(range(3), key=lambda idx: (lengths[idx], -idx))
    return _INDEX_TO_AXIS[best]


def _jax_devices_for_config(cfg: ShardingConfig) -> tuple[jax.Device, ...]:
    if not cfg.enabled:
        return ()
    try:
        devices = (
            tuple(jax.devices(cfg.backend)) if cfg.backend else tuple(jax.devices())
        )
    except Exception as exc:
        backend = cfg.backend or "default"
        raise ValueError(
            f"No JAX devices are available for backend {backend!r}"
        ) from exc
    if not devices:
        backend = cfg.backend or "default"
        raise ValueError(f"No JAX devices are available for backend {backend!r}")
    num_devices = len(devices) if cfg.num_devices is None else int(cfg.num_devices)
    if num_devices <= 1:
        return ()
    if num_devices > len(devices):
        raise ValueError(
            f"Requested {num_devices} sharding devices, but only {len(devices)} "
            f"are available for backend {cfg.backend or 'default'}"
        )
    return devices[:num_devices]


def _pad_shape_for_devices(
    shape: tuple[int, ...], axis: int, num_devices: int
) -> tuple[int, ...]:
    out = list(int(v) for v in shape)
    size = out[int(axis)]
    remainder = size % int(num_devices)
    if remainder:
        out[int(axis)] = size + (int(num_devices) - remainder)
    return tuple(out)


def _pad_width(
    base_shape: tuple[int, ...], storage_shape: tuple[int, ...]
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (0, max(0, int(storage_shape[i]) - int(base_shape[i])))
        for i in range(len(base_shape))
    )


def _valid_storage_mask(
    active_shape: tuple[int, ...], storage_shape: tuple[int, ...]
) -> jnp.ndarray | None:
    if tuple(int(v) for v in active_shape) == tuple(int(v) for v in storage_shape):
        return None
    mask = np.ones(tuple(int(v) for v in storage_shape), dtype=bool)
    active_region = tuple(slice(0, int(v)) for v in active_shape)
    mask[active_region] = False
    return jnp.asarray(mask)


def _build_storage_layout(
    fields,
    cfg: ShardingConfig,
    *,
    is_3d: bool,
    full_pec_3d: bool = False,
) -> tuple[StorageLayout, tuple[jax.Device, ...]]:
    logical_base_shape = tuple(int(v) for v in getattr(fields, "permittivity").shape)
    if not cfg.enabled:
        logical_shapes = {
            name: tuple(int(v) for v in getattr(fields, name).shape)
            for name in _COMPONENT_NAMES
        }
        active_shapes = {
            name: tuple(int(v) + 1 for v in shape) if full_pec_3d else shape
            for name, shape in logical_shapes.items()
        }
        masks = {
            name: _valid_storage_mask(active_shapes[name], active_shapes[name])
            for name in _COMPONENT_NAMES
        }
        padding = {
            name: tuple((0, 0) for _ in active_shapes[name])
            for name in _COMPONENT_NAMES
        }
        return (
            StorageLayout(
                enabled=False,
                pec_full_storage=bool(full_pec_3d),
                logical_base_shape=logical_base_shape,
                axis_name="z",
                axis=0,
                num_devices=1,
                backend=cfg.backend,
                logical_shapes=logical_shapes,
                active_shapes=active_shapes,
                storage_shapes=dict(active_shapes),
                padding=padding,
                valid_masks=masks,
            ),
            (),
        )
    if not is_3d:
        raise NotImplementedError("compiled sharding currently supports 3D runs only")
    if len(logical_base_shape) != 3:
        raise ValueError(
            f"3D sharding requires a 3-axis grid, got {logical_base_shape}"
        )

    devices = _jax_devices_for_config(cfg)
    if not devices:
        return _build_storage_layout(
            fields,
            ShardingConfig(enabled=False, backend=cfg.backend),
            is_3d=is_3d,
            full_pec_3d=full_pec_3d,
        )
    axis_name = _select_sharding_axis(cfg.axis, logical_base_shape)
    axis = _AXIS_TO_INDEX[axis_name]
    num_devices = len(devices)
    logical_shapes = {
        name: tuple(int(v) for v in getattr(fields, name).shape)
        for name in _COMPONENT_NAMES
    }
    active_shapes = {
        name: tuple(int(v) + 1 for v in shape) if full_pec_3d else shape
        for name, shape in logical_shapes.items()
    }
    storage_shapes = {
        name: _pad_shape_for_devices(shape, axis, num_devices)
        for name, shape in active_shapes.items()
    }
    padding = {
        name: _pad_width(active_shapes[name], storage_shapes[name])
        for name in _COMPONENT_NAMES
    }
    masks = {
        name: _valid_storage_mask(active_shapes[name], storage_shapes[name])
        for name in _COMPONENT_NAMES
    }
    return (
        StorageLayout(
            enabled=True,
            pec_full_storage=bool(full_pec_3d),
            logical_base_shape=logical_base_shape,
            axis_name=axis_name,
            axis=axis,
            num_devices=num_devices,
            backend=cfg.backend,
            logical_shapes=logical_shapes,
            active_shapes=active_shapes,
            storage_shapes=storage_shapes,
            padding=padding,
            valid_masks=masks,
        ),
        devices,
    )


def _pad_high_to_shape(arr, shape: tuple[int, ...], *, pad_value=0.0) -> jnp.ndarray:
    return fit_array_to_shape(jnp.asarray(arr), shape, pad_value=pad_value)


def _crop_high_to_shape(arr, shape: tuple[int, ...]) -> jnp.ndarray:
    slices = tuple(slice(0, int(v)) for v in shape)
    return jnp.asarray(arr)[slices]


def _pad_all_high_planes_3d(arr: jnp.ndarray) -> jnp.ndarray:
    out = jnp.asarray(arr)
    for axis in range(3):
        tail = jnp.take(out, indices=jnp.array([out.shape[axis] - 1]), axis=axis)
        out = jnp.concatenate([out, tail], axis=axis)
    return out


class _StorageFieldsProxy:
    """Shallow fields proxy with padded component storage arrays."""

    def __init__(
        self,
        base,
        overrides: dict[str, object],
        layout: StorageLayout,
    ) -> None:
        self._base = base
        self._overrides = dict(overrides)
        self._logical_component_shapes = dict(layout.logical_shapes)
        self._storage_component_shapes = dict(layout.storage_shapes)
        self._logical_base_shape_3d = tuple(layout.logical_base_shape)
        self._storage_layout = layout

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._base, name)


def _pad_pml_data_for_storage(fields, layout: StorageLayout):
    pml_data = getattr(fields, "pml_data", None)
    if (not layout.enabled and not layout.pec_full_storage) or not isinstance(
        pml_data, dict
    ):
        return pml_data
    out = dict(pml_data)
    axis_index = {"z": 0, "y": 1, "x": 2}

    def _pad_cpml_profile(key: str, spec, *, neutral: float):
        if key not in out:
            return
        arr = jnp.asarray(out[key])
        storage_shape = layout.storage_shapes[spec.target_component]
        axis = axis_index[spec.derivative_axis]
        if arr.ndim == 3 and all(int(arr.shape[i]) == 1 for i in range(3) if i != axis):
            compact_shape = tuple(
                int(storage_shape[i]) if i == axis else 1 for i in range(3)
            )
            out[key] = _pad_high_to_shape(arr, compact_shape, pad_value=neutral)
        else:
            out[key] = _pad_high_to_shape(arr, storage_shape, pad_value=neutral)

    for spec in CPML_3D_H_DERIVATIVES:
        for suffix, neutral in (("sigma", 0.0), ("kappa", 1.0), ("alpha", 0.0)):
            key = f"cpml3d_{spec.name}_{suffix}"
            _pad_cpml_profile(key, spec, neutral=neutral)
    for spec in CPML_3D_E_DERIVATIVES:
        for suffix, neutral in (("sigma", 0.0), ("kappa", 1.0), ("alpha", 0.0)):
            key = f"cpml3d_{spec.name}_{suffix}"
            _pad_cpml_profile(key, spec, neutral=neutral)
    return out


def _make_storage_fields_proxy(fields, layout: StorageLayout):
    if not layout.enabled and not layout.pec_full_storage:
        return fields
    overrides: dict[str, object] = {}
    for name in _COMPONENT_NAMES:
        overrides[name] = _pad_high_to_shape(
            getattr(fields, name), layout.storage_shapes[name], pad_value=0.0
        )
    neutral_component_arrays = {
        "eps_x": ("Ex", 1.0),
        "eps_y": ("Ey", 1.0),
        "eps_z": ("Ez", 1.0),
        "sig_x": ("Ex", 0.0),
        "sig_y": ("Ey", 0.0),
        "sig_z": ("Ez", 0.0),
        "sigma_m_hx": ("Hx", 0.0),
        "sigma_m_hy": ("Hy", 0.0),
        "sigma_m_hz": ("Hz", 0.0),
    }
    for name, (component, neutral) in neutral_component_arrays.items():
        if hasattr(fields, name):
            overrides[name] = _pad_high_to_shape(
                getattr(fields, name),
                layout.storage_shapes[component],
                pad_value=neutral,
            )
    overrides["pml_data"] = _pad_pml_data_for_storage(fields, layout)
    return _StorageFieldsProxy(fields, overrides, layout)


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
    sharding: ShardingConfig = ShardingConfig()


@dataclass
class CompiledSimulation:
    """Compiled simulation program and packed static specs."""

    config: CompiledRunConfig
    material_spec: CompiledMaterialSpec
    source_specs: tuple[CompiledSourceSpec, ...]
    monitor_specs: tuple[CompiledMonitorSpec, ...]
    monitor_devices: tuple[Monitor, ...]
    field_shape_ex: tuple[int, ...]
    field_shape_ey: tuple[int, ...]
    field_shape_ez: tuple[int, ...]
    field_shape_hx: tuple[int, ...]
    field_shape_hy: tuple[int, ...]
    field_shape_hz: tuple[int, ...]
    storage_layout: StorageLayout
    sharding_devices: tuple[jax.Device, ...]
    storage_shape_ex: tuple[int, ...]
    storage_shape_ey: tuple[int, ...]
    storage_shape_ez: tuple[int, ...]
    storage_shape_hx: tuple[int, ...]
    storage_shape_hy: tuple[int, ...]
    storage_shape_hz: tuple[int, ...]

    # Static update coefficients (full-grid, dense updates; no per-step scatters)
    h_decay_x: jnp.ndarray
    h_source_x: jnp.ndarray
    h_source_lossless_x: jnp.ndarray
    h_sigma_m_x: jnp.ndarray
    h_decay_y: jnp.ndarray
    h_source_y: jnp.ndarray
    h_source_lossless_y: jnp.ndarray
    h_sigma_m_y: jnp.ndarray
    h_decay_z: jnp.ndarray
    h_source_z: jnp.ndarray
    h_source_lossless_z: jnp.ndarray
    h_sigma_m_z: jnp.ndarray
    e_decay_x: jnp.ndarray
    e_source_x: jnp.ndarray
    e_source_lossless_x: jnp.ndarray
    e_conductivity_x: jnp.ndarray
    e_inv_permittivity_x: jnp.ndarray
    e_decay_y: jnp.ndarray
    e_source_y: jnp.ndarray
    e_source_lossless_y: jnp.ndarray
    e_conductivity_y: jnp.ndarray
    e_inv_permittivity_y: jnp.ndarray
    e_decay_z: jnp.ndarray
    e_source_z: jnp.ndarray
    e_source_lossless_z: jnp.ndarray
    e_conductivity_z: jnp.ndarray
    e_inv_permittivity_z: jnp.ndarray
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
    cpml3d_sigma_h_terms: tuple[jnp.ndarray, ...]
    cpml3d_kappa_h_terms: tuple[jnp.ndarray, ...]
    cpml3d_alpha_h_terms: tuple[jnp.ndarray, ...]
    cpml3d_sigma_e_terms: tuple[jnp.ndarray, ...]
    cpml3d_kappa_e_terms: tuple[jnp.ndarray, ...]
    cpml3d_alpha_e_terms: tuple[jnp.ndarray, ...]
    cpml3d_metallic_edges: frozenset[str]
    use_primitive_cpml_3d_terms: bool
    use_cpml_3d_packed_psi: bool
    cpml3d_h_slab_specs: tuple[CpmlPackedSlabSpec, ...]
    cpml3d_e_slab_specs: tuple[CpmlPackedSlabSpec, ...]
    cpml3d_h_psi_shapes: tuple[tuple[int, int, int], ...]
    cpml3d_e_psi_shapes: tuple[tuple[int, int, int], ...]
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

    # Explicit metallic-wall masks aligned to the Yee staggering.
    ex_metal_mask: jnp.ndarray
    ey_metal_mask: jnp.ndarray
    ez_metal_mask: jnp.ndarray
    hx_metal_mask: jnp.ndarray
    hy_metal_mask: jnp.ndarray
    hz_metal_mask: jnp.ndarray
    ex_storage_mask: jnp.ndarray | None
    ey_storage_mask: jnp.ndarray | None
    ez_storage_mask: jnp.ndarray | None
    hx_storage_mask: jnp.ndarray | None
    hy_storage_mask: jnp.ndarray | None
    hz_storage_mask: jnp.ndarray | None

    _compiled_scan: callable | None = None
    _compile_count: int = 0
    _sharding_mesh: object | None = None

    def _update_coefficients(self) -> UpdateCoefficients:
        """Build runtime coefficient container for jitted scan entrypoint."""
        return UpdateCoefficients(
            **{name: getattr(self, name) for name in UpdateCoefficients._fields}
        )

    def _component_logical_shape(self, component: str) -> tuple[int, ...]:
        return self.storage_layout.logical_shapes[component]

    def _component_active_shape(self, component: str) -> tuple[int, ...]:
        return self.storage_layout.active_shapes[component]

    def _component_storage_shape(self, component: str) -> tuple[int, ...]:
        return self.storage_layout.storage_shapes[component]

    def _component_pec_mask(self, component: str) -> jnp.ndarray:
        return {
            "Ex": self.fp_ex_mask,
            "Ey": self.fp_ey_mask,
            "Ez": self.fp_ez_mask,
            "Hx": self.fp_hx_mask,
            "Hy": self.fp_hy_mask,
            "Hz": self.fp_hz_mask,
        }[component]

    def _pad_component(self, component: str, arr: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(arr)
        if self.storage_layout.pec_full_storage:
            if tuple(arr.shape) == self._component_logical_shape(component):
                arr = _pad_all_high_planes_3d(arr)
            arr = _pad_high_to_shape(
                arr, self._component_storage_shape(component), pad_value=0.0
            )
            return apply_zero_mask(arr, self._component_pec_mask(component))
        return _pad_high_to_shape(
            arr, self._component_storage_shape(component), pad_value=0.0
        )

    def _crop_active_component(self, component: str, arr: jnp.ndarray) -> jnp.ndarray:
        return _crop_high_to_shape(arr, self._component_active_shape(component))

    def _pad_active_component(self, component: str, arr: jnp.ndarray) -> jnp.ndarray:
        out = _pad_high_to_shape(
            arr, self._component_storage_shape(component), pad_value=0.0
        )
        if self.storage_layout.pec_full_storage:
            return apply_zero_mask(out, self._component_pec_mask(component))
        return out

    def _crop_component(self, component: str, arr: jnp.ndarray) -> jnp.ndarray:
        return _crop_high_to_shape(arr, self._component_logical_shape(component))

    def _device_mesh(self):
        if not self.storage_layout.enabled:
            return None
        if self._sharding_mesh is None:
            devices = np.asarray(self.sharding_devices, dtype=object)
            self._sharding_mesh = jax.sharding.Mesh(devices, (_MESH_AXIS,))
        return self._sharding_mesh

    def _replicated_sharding(self):
        mesh = self._device_mesh()
        if mesh is None:
            return None
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def _axis_sharding(self, arr: jnp.ndarray):
        mesh = self._device_mesh()
        if mesh is None:
            return None
        arr = jnp.asarray(arr)
        axis = int(self.storage_layout.axis)
        if (
            arr.ndim > axis
            and int(arr.shape[axis]) > 0
            and int(arr.shape[axis]) % int(self.storage_layout.num_devices) == 0
        ):
            spec = [None] * arr.ndim
            spec[axis] = _MESH_AXIS
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))
        return self._replicated_sharding()

    def _place_array(self, arr, *, shard_arrays: bool = True):
        if not self.storage_layout.enabled:
            return arr
        arr = jnp.asarray(arr)
        sharding = (
            self._axis_sharding(arr) if shard_arrays else self._replicated_sharding()
        )
        return jax.device_put(arr, sharding)

    def _place_pytree(self, tree, *, shard_arrays: bool = True):
        if not self.storage_layout.enabled:
            return tree
        return jax.tree_util.tree_map(
            lambda arr: self._place_array(arr, shard_arrays=shard_arrays),
            tree,
        )

    def prepare_engine_state(self, engine_state: EngineState) -> EngineState:
        """Pad logical component fields and place runtime state for execution."""

        engine_state = engine_state._replace(
            ex=self._pad_component("Ex", engine_state.ex),
            ey=self._pad_component("Ey", engine_state.ey),
            ez=self._pad_component("Ez", engine_state.ez),
            hx=self._pad_component("Hx", engine_state.hx),
            hy=self._pad_component("Hy", engine_state.hy),
            hz=self._pad_component("Hz", engine_state.hz),
        )
        if not self.storage_layout.enabled:
            return engine_state
        return self._place_pytree(engine_state, shard_arrays=True)

    def crop_engine_state(self, engine_state: EngineState) -> EngineState:
        """Return an EngineState with public component fields cropped to logical shape."""

        if not self.storage_layout.enabled and not self.storage_layout.pec_full_storage:
            return engine_state
        return engine_state._replace(
            ex=self._crop_component("Ex", engine_state.ex),
            ey=self._crop_component("Ey", engine_state.ey),
            ez=self._crop_component("Ez", engine_state.ez),
            hx=self._crop_component("Hx", engine_state.hx),
            hy=self._crop_component("Hy", engine_state.hy),
            hz=self._crop_component("Hz", engine_state.hz),
        )

    def _place_update_coefficients(
        self, coeffs: UpdateCoefficients
    ) -> UpdateCoefficients:
        if not self.storage_layout.enabled:
            return coeffs
        return self._place_pytree(coeffs, shard_arrays=True)

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
            return self.field_shape_ex
        if field_name == "Ey":
            return self.field_shape_ey
        if field_name == "Ez":
            return self.field_shape_ez
        if field_name == "Hx":
            return self.field_shape_hx
        if field_name == "Hy":
            return self.field_shape_hy
        if field_name == "Hz":
            return self.field_shape_hz
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
        field_shape = self._snapshot_field_shape()
        snapshot_bytes = int(
            max_snapshots
            * np.prod(field_shape, dtype=np.int64)
            * np.dtype(np.float32).itemsize
        )
        warn_threshold = _snapshot_warning_threshold_bytes()
        if warn_threshold > 0 and snapshot_bytes >= warn_threshold:
            warnings.warn(
                "Compiled field snapshots will allocate "
                f"{snapshot_bytes / 1024**3:.2f} GiB for "
                f"{max_snapshots} {self.config.snapshot_field} frames. "
                "Increase snapshot_interval or disable store_snapshots for large runs.",
                RuntimeWarning,
                stacklevel=2,
            )
        return (
            jnp.zeros(
                (max_snapshots, *field_shape),
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
                patch = (spec.coeff * amp).astype(out.dtype)
                cur = jax.lax.dynamic_slice(out, spec.slab_starts, spec.slab_sizes)
                out = jax.lax.dynamic_update_slice(out, cur + patch, spec.slab_starts)
            else:
                patch = (spec.coeff * amp).astype(out.dtype)
                out = out.at[spec.index].add(patch)
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
                return arr + (dense_coeff * amp).astype(arr.dtype)
            patch = (group.coeffs[0] * amp).astype(arr.dtype)
            cur = jax.lax.dynamic_slice(arr, starts_0, group.max_sizes)
            return jax.lax.dynamic_update_slice(arr, cur + patch, starts_0)

        if group.n == 2:

            def apply_one(out, i: int):
                amp_i = group.waveforms[i, safe_idx]
                patch_i = (group.coeffs[i] * amp_i).astype(out.dtype)
                starts_i = group.starts_tuple[i]
                cur_i = jax.lax.dynamic_slice(out, starts_i, group.max_sizes)
                return jax.lax.dynamic_update_slice(out, cur_i + patch_i, starts_i)

            return apply_one(apply_one(arr, 0), 1)

        def body(i, out):
            amp = group.waveforms[i, safe_idx]
            patch = (group.coeffs[i] * amp).astype(out.dtype)
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

    @staticmethod
    def _apply_metal_edges_3d(
        arr: jnp.ndarray, component: str, metallic_edges: frozenset[str]
    ) -> jnp.ndarray:
        """Zero compact-grid Yee samples constrained by low-side PEC walls."""

        out = arr
        zero = jnp.asarray(0.0, dtype=arr.dtype)
        if component == "Ex":
            if "front" in metallic_edges:
                out = out.at[0, :, :].set(zero)
            if "bottom" in metallic_edges:
                out = out.at[:, 0, :].set(zero)
        elif component == "Ey":
            if "front" in metallic_edges:
                out = out.at[0, :, :].set(zero)
            if "left" in metallic_edges:
                out = out.at[:, :, 0].set(zero)
        elif component == "Ez":
            if "bottom" in metallic_edges:
                out = out.at[:, 0, :].set(zero)
            if "left" in metallic_edges:
                out = out.at[:, :, 0].set(zero)
        elif component == "Hx":
            if "left" in metallic_edges:
                out = out.at[:, :, 0].set(zero)
        elif component == "Hy":
            if "bottom" in metallic_edges:
                out = out.at[:, 0, :].set(zero)
        elif component == "Hz":
            if "front" in metallic_edges:
                out = out.at[0, :, :].set(zero)
        else:
            raise ValueError(f"Unsupported 3D field component {component!r}")
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
                            ex_flat[bm.ex_interp_flat_idx[i]] * bm.ex_interp_weights[i],
                            axis=-1,
                        )
                        * mask
                    )
                    eys = (
                        jnp.sum(
                            ey_flat[bm.ey_interp_flat_idx[i]] * bm.ey_interp_weights[i],
                            axis=-1,
                        )
                        * mask
                    )
                    ezs = (
                        jnp.sum(
                            ez_flat[bm.ez_interp_flat_idx[i]] * bm.ez_interp_weights[i],
                            axis=-1,
                        )
                        * mask
                    )
                    hxs = (
                        jnp.sum(
                            hx_flat[bm.hx_interp_flat_idx[i]] * bm.hx_interp_weights[i],
                            axis=-1,
                        )
                        * mask
                    )
                    hys = (
                        jnp.sum(
                            hy_flat[bm.hy_interp_flat_idx[i]] * bm.hy_interp_weights[i],
                            axis=-1,
                        )
                        * mask
                    )
                    hzs = (
                        jnp.sum(
                            hz_flat[bm.hz_interp_flat_idx[i]] * bm.hz_interp_weights[i],
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
                        vecs = self._monitor_dft_vectors_3d(mon, ex, ey, ez, hx, hy, hz)
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
            dft_3d = tuple(
                s
                for s in self.monitor_specs
                if s.is_3d and bool(getattr(s, "dft_enabled", False))
            )
            batchable_3d = tuple(
                s
                for s in self.monitor_specs
                if s.is_3d and not bool(getattr(s, "dft_enabled", False))
            )
            field_shapes = {
                "Ex": self.storage_shape_ex,
                "Ey": self.storage_shape_ey,
                "Ez": self.storage_shape_ez,
                "Hx": self.storage_shape_hx,
                "Hy": self.storage_shape_hy,
                "Hz": self.storage_shape_hz,
            }
            batched_mon = compile_batched_monitor_data(batchable_3d, field_shapes)
            # Keep DFT monitors unbatched for deterministic per-component modal
            # accumulation while still batching ordinary 3D power/frequency monitors.
            monitors_2d = tuple(s for s in self.monitor_specs if not s.is_3d) + dft_3d
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
                return self._crop_component("Ex", ex)
            if snapshot_field == "Ey":
                return self._crop_component("Ey", ey)
            if snapshot_field == "Ez":
                return self._crop_component("Ez", ez) if tm_ez is None else tm_ez
            if snapshot_field == "Hx":
                return self._crop_component("Hx", hx) if tm_hx is None else tm_hx
            if snapshot_field == "Hy":
                return self._crop_component("Hy", hy) if tm_hy is None else tm_hy
            if snapshot_field == "Hz":
                return self._crop_component("Hz", hz)
            raise ValueError(f"Unsupported snapshot field: {snapshot_field}")

        def run_scan(
            engine_state: EngineState,
            monitor_state: MonitorState,
            coeffs: UpdateCoefficients,
            snapshot_state=None,
        ):
            h_decay_x, h_source_x = coeffs.h_decay_x, coeffs.h_source_x
            h_sigma_m_x = coeffs.h_sigma_m_x
            h_decay_y, h_source_y = coeffs.h_decay_y, coeffs.h_source_y
            h_sigma_m_y = coeffs.h_sigma_m_y
            h_decay_z, h_source_z = coeffs.h_decay_z, coeffs.h_source_z
            h_sigma_m_z = coeffs.h_sigma_m_z
            e_decay_x, e_source_x = coeffs.e_decay_x, coeffs.e_source_x
            e_conductivity_x = coeffs.e_conductivity_x
            e_inv_permittivity_x = coeffs.e_inv_permittivity_x
            e_decay_y, e_source_y = coeffs.e_decay_y, coeffs.e_source_y
            e_conductivity_y = coeffs.e_conductivity_y
            e_inv_permittivity_y = coeffs.e_inv_permittivity_y
            e_decay_z, e_source_z = coeffs.e_decay_z, coeffs.e_source_z
            e_conductivity_z = coeffs.e_conductivity_z
            e_inv_permittivity_z = coeffs.e_inv_permittivity_z
            tm_h_decay_x, tm_h_source_x = coeffs.tm_h_decay_x, coeffs.tm_h_source_x
            tm_h_decay_y, tm_h_source_y = coeffs.tm_h_decay_y, coeffs.tm_h_source_y
            tm_e_decay_z, tm_e_source_z = coeffs.tm_e_decay_z, coeffs.tm_e_source_z
            ex_metal_mask = self.ex_metal_mask
            ey_metal_mask = self.ey_metal_mask
            ez_metal_mask = self.ez_metal_mask
            hx_metal_mask = self.hx_metal_mask
            hy_metal_mask = self.hy_metal_mask
            hz_metal_mask = self.hz_metal_mask
            ex_storage_mask = self.ex_storage_mask
            ey_storage_mask = self.ey_storage_mask
            ez_storage_mask = self.ez_storage_mask
            hx_storage_mask = self.hx_storage_mask
            hy_storage_mask = self.hy_storage_mask
            hz_storage_mask = self.hz_storage_mask
            use_physical_tm_xy = self.use_physical_tm_xy
            use_cpml_tm_xy = self.use_cpml_tm_xy
            use_cpml_3d = self.use_cpml_3d
            full_pec_3d = self.full_pec_3d
            metallic_edges_3d = self.cpml3d_metallic_edges
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
                h_psi_shapes = self.cpml3d_h_psi_shapes
                e_psi_shapes = self.cpml3d_e_psi_shapes
                if not _cpml_3d_shapes_match(
                    engine_state.cpml3d_psi_h_terms, h_psi_shapes
                ):
                    engine_state = engine_state._replace(
                        cpml3d_psi_h_terms=_zeros_for_cpml_3d_shapes(
                            h_psi_shapes, engine_state.hx.dtype
                        )
                    )
                if not _cpml_3d_shapes_match(
                    engine_state.cpml3d_psi_e_terms, e_psi_shapes
                ):
                    engine_state = engine_state._replace(
                        cpml3d_psi_e_terms=_zeros_for_cpml_3d_shapes(
                            e_psi_shapes, engine_state.ex.dtype
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
                    eng = eng._replace(
                        ex=ex.astype(eng.ex.dtype),
                        ey=ey.astype(eng.ey.dtype),
                        ez=ez.astype(eng.ez.dtype),
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
                        h_psi_shapes = self.cpml3d_h_psi_shapes
                        e_psi_shapes = self.cpml3d_e_psi_shapes
                        if not _cpml_3d_shapes_match(cpml3d_psi_h_terms, h_psi_shapes):
                            cpml3d_psi_h_terms = _zeros_for_cpml_3d_shapes(
                                h_psi_shapes, hx.dtype
                            )
                        if not _cpml_3d_shapes_match(cpml3d_psi_e_terms, e_psi_shapes):
                            cpml3d_psi_e_terms = _zeros_for_cpml_3d_shapes(
                                e_psi_shapes, ex.dtype
                            )

                    if is_3d and full_pec_3d:
                        hx_active, hy_active, hz_active = full_pec_update_h_from_e_3d(
                            self._crop_active_component("Ex", ex),
                            self._crop_active_component("Ey", ey),
                            self._crop_active_component("Ez", ez),
                            self._crop_active_component("Hx", hx),
                            self._crop_active_component("Hy", hy),
                            self._crop_active_component("Hz", hz),
                            resolution,
                            h_decay=(
                                h_decay_x,
                                h_decay_y,
                                h_decay_z,
                            ),
                            h_source=(
                                h_source_x,
                                h_source_y,
                                h_source_z,
                            ),
                            h_mask=(
                                self._crop_active_component("Hx", self.fp_hx_mask),
                                self._crop_active_component("Hy", self.fp_hy_mask),
                                self._crop_active_component("Hz", self.fp_hz_mask),
                            ),
                        )
                        hx = self._pad_active_component("Hx", hx_active)
                        hy = self._pad_active_component("Hy", hy_active)
                        hz = self._pad_active_component("Hz", hz_active)
                    elif is_3d:
                        if self.use_cpml_3d:
                            if self.use_cpml_3d_packed_psi:
                                hx, hy, hz, cpml3d_psi_h_terms = (
                                    cpml_update_h_from_e_3d_packed_psi(
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
                                        slab_specs=self.cpml3d_h_slab_specs,
                                    )
                                )
                            else:
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
                                        dt=dt_scalar,
                                        psi_h_terms=cpml3d_psi_h_terms,
                                    )
                                )
                        else:
                            hx, hy, hz = ops.fused_update_h_lossy_3d_material(
                                ex,
                                ey,
                                ez,
                                hx,
                                hy,
                                hz,
                                h_sigma_m_x,
                                h_sigma_m_y,
                                h_sigma_m_z,
                                dt_scalar,
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
                        hz = h_decay_z * hz - h_source_z * curl_ez
                    else:
                        curl_ex, curl_ey, curl_ez = ops.curl_e_to_h_2d(
                            (ex, ey, ez),
                            resolution,
                            plane=plane_2d,
                        )
                        hx = h_decay_x * hx - h_source_x * curl_ex
                        hy = h_decay_y * hy - h_source_y * curl_ey
                        hz = h_decay_z * hz - h_source_z * curl_ez

                    eng = eng._replace(
                        hx=hx.astype(eng.hx.dtype),
                        hy=hy.astype(eng.hy.dtype),
                        hz=hz.astype(eng.hz.dtype),
                        fp_hx=fp_hx.astype(eng.fp_hx.dtype),
                        fp_hy=fp_hy.astype(eng.fp_hy.dtype),
                        fp_hz=fp_hz.astype(eng.fp_hz.dtype),
                        cpml_psi_h_terms=cpml_psi_h_terms.astype(
                            eng.cpml_psi_h_terms.dtype
                        ),
                        cpml_psi_e_terms=cpml_psi_e_terms.astype(
                            eng.cpml_psi_e_terms.dtype
                        ),
                        cpml3d_psi_h_terms=tuple(
                            term.astype(ref.dtype)
                            for term, ref in zip(
                                cpml3d_psi_h_terms,
                                eng.cpml3d_psi_h_terms,
                                strict=True,
                            )
                        ),
                        cpml3d_psi_e_terms=tuple(
                            term.astype(ref.dtype)
                            for term, ref in zip(
                                cpml3d_psi_e_terms,
                                eng.cpml3d_psi_e_terms,
                                strict=True,
                            )
                        ),
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
                    if is_3d:
                        if full_pec_3d:
                            hx = apply_zero_mask(hx_post, self.fp_hx_mask)
                            hy = apply_zero_mask(hy_post, self.fp_hy_mask)
                            hz = apply_zero_mask(hz, self.fp_hz_mask)
                        else:
                            hx = self._apply_metal_edges_3d(
                                hx_post, "Hx", metallic_edges_3d
                            )
                            hy = self._apply_metal_edges_3d(
                                hy_post, "Hy", metallic_edges_3d
                            )
                            hz = self._apply_metal_edges_3d(hz, "Hz", metallic_edges_3d)
                            hx = apply_zero_mask(hx, hx_storage_mask)
                            hy = apply_zero_mask(hy, hy_storage_mask)
                            hz = apply_zero_mask(hz, hz_storage_mask)
                    else:
                        hx = self._apply_metal_mask(hx_post, hx_metal_mask)
                        hy = self._apply_metal_mask(hy_post, hy_metal_mask)
                        hz = self._apply_metal_mask(hz, hz_metal_mask)
                    eng = eng._replace(
                        hx=hx.astype(eng.hx.dtype),
                        hy=hy.astype(eng.hy.dtype),
                        hz=hz.astype(eng.hz.dtype),
                        fp_hx=fp_hx.astype(eng.fp_hx.dtype),
                        fp_hy=fp_hy.astype(eng.fp_hy.dtype),
                        fp_hz=fp_hz.astype(eng.fp_hz.dtype),
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
                        ex_active, ey_active, ez_active = full_pec_update_e_from_h_3d(
                            self._crop_active_component("Hx", hx),
                            self._crop_active_component("Hy", hy),
                            self._crop_active_component("Hz", hz),
                            self._crop_active_component("Ex", ex),
                            self._crop_active_component("Ey", ey),
                            self._crop_active_component("Ez", ez),
                            resolution,
                            e_decay=(
                                e_decay_x,
                                e_decay_y,
                                e_decay_z,
                            ),
                            e_source=(
                                e_source_x,
                                e_source_y,
                                e_source_z,
                            ),
                            e_mask=(
                                self._crop_active_component("Ex", self.fp_ex_mask),
                                self._crop_active_component("Ey", self.fp_ey_mask),
                                self._crop_active_component("Ez", self.fp_ez_mask),
                            ),
                        )
                        ex = self._pad_active_component("Ex", ex_active)
                        ey = self._pad_active_component("Ey", ey_active)
                        ez = self._pad_active_component("Ez", ez_active)
                    elif is_3d:
                        if self.use_cpml_3d:
                            if self.use_cpml_3d_packed_psi:
                                ex, ey, ez, cpml3d_psi_e_terms = (
                                    cpml_update_e_from_h_3d_packed_psi(
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
                                        slab_specs=self.cpml3d_e_slab_specs,
                                        metallic_edges=self.cpml3d_metallic_edges,
                                    )
                                )
                            else:
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
                                        dt=dt_scalar,
                                        psi_e_terms=cpml3d_psi_e_terms,
                                        metallic_edges=self.cpml3d_metallic_edges,
                                    )
                                )
                        else:
                            boundary_views = build_h_boundary_views_for_e_3d(
                                hx, hy, hz, None
                            )
                            ex, ey, ez = ops.fused_update_e_lossy_3d_material(
                                hx,
                                hy,
                                hz,
                                ex,
                                ey,
                                ez,
                                e_conductivity_x,
                                e_inv_permittivity_x,
                                e_conductivity_y,
                                e_inv_permittivity_y,
                                e_conductivity_z,
                                e_inv_permittivity_z,
                                dt_scalar,
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
                        ex = e_decay_x * ex + e_source_x * curl_hx
                        ey = e_decay_y * ey + e_source_y * curl_hy
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
                        ex = e_decay_x * ex + e_source_x * curl_hx
                        ey = e_decay_y * ey + e_source_y * curl_hy
                        ez = e_decay_z * ez + e_source_z * curl_hz

                    eng = eng._replace(
                        ex=ex.astype(eng.ex.dtype),
                        ey=ey.astype(eng.ey.dtype),
                        ez=ez.astype(eng.ez.dtype),
                        fp_ex=fp_ex.astype(eng.fp_ex.dtype),
                        fp_ey=fp_ey.astype(eng.fp_ey.dtype),
                        fp_ez=fp_ez.astype(eng.fp_ez.dtype),
                        cpml_psi_e_terms=cpml_psi_e_terms.astype(
                            eng.cpml_psi_e_terms.dtype
                        ),
                        cpml3d_psi_e_terms=tuple(
                            term.astype(ref.dtype)
                            for term, ref in zip(
                                cpml3d_psi_e_terms,
                                eng.cpml3d_psi_e_terms,
                                strict=True,
                            )
                        ),
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
                    if is_3d:
                        if full_pec_3d:
                            ex = apply_zero_mask(ex, self.fp_ex_mask)
                            ey = apply_zero_mask(ey, self.fp_ey_mask)
                            ez = apply_zero_mask(ez, self.fp_ez_mask)
                        else:
                            ex = self._apply_metal_edges_3d(ex, "Ex", metallic_edges_3d)
                            ey = self._apply_metal_edges_3d(ey, "Ey", metallic_edges_3d)
                            ez = self._apply_metal_edges_3d(ez, "Ez", metallic_edges_3d)
                            ex = apply_zero_mask(ex, ex_storage_mask)
                            ey = apply_zero_mask(ey, ey_storage_mask)
                            ez = apply_zero_mask(ez, ez_storage_mask)
                    else:
                        ex = self._apply_metal_mask(ex, ex_metal_mask)
                        ey = self._apply_metal_mask(ey, ey_metal_mask)
                        ez = self._apply_metal_mask(ez, ez_metal_mask)
                    if use_physical_tm_xy:
                        ez = jnp.where(tm_ez_mask, jnp.asarray(0.0, dtype=ez.dtype), ez)
                    eng = eng._replace(
                        ex=ex.astype(eng.ex.dtype),
                        ey=ey.astype(eng.ey.dtype),
                        ez=ez.astype(eng.ez.dtype),
                        fp_ex=fp_ex.astype(eng.fp_ex.dtype),
                        fp_ey=fp_ey.astype(eng.fp_ey.dtype),
                        fp_ez=fp_ez.astype(eng.fp_ez.dtype),
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
                    slot = jnp.minimum(
                        snap_count,
                        jnp.asarray(snap_fields.shape[0] - 1, dtype=snap_count.dtype),
                    )
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
                    snapshot_values = snapshot_values.astype(snap_fields.dtype)
                    field_start = (slot,) + tuple(
                        jnp.asarray(0, dtype=slot.dtype)
                        for _ in range(snapshot_values.ndim)
                    )
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

    def memory_estimate(self, *, include_runtime: bool = True) -> dict:
        """Return a JSON-friendly estimate of compiled-program array memory."""
        entries: list[dict] = []
        referenced_entries: list[dict] = []

        update_names = set(UpdateCoefficients._fields)
        referenced_update_names = {
            "h_sigma_m_x",
            "h_sigma_m_y",
            "h_sigma_m_z",
            "e_conductivity_x",
            "e_conductivity_y",
            "e_conductivity_z",
        }
        for name in referenced_update_names:
            _add_array_entries(
                referenced_entries,
                name,
                getattr(self, name),
                category="compiled_referenced_inputs",
                residency="reference",
            )

        for name in update_names:
            if name in referenced_update_names:
                continue
            _add_array_entries(
                entries,
                name,
                getattr(self, name),
                category="compiled_update_coefficients",
            )

        for name in (
            "tm_ez_mask",
            "tm_hx_mask",
            "tm_hy_mask",
            "cpml_sigma_h_terms",
            "cpml_kappa_h_aux_terms",
            "cpml_alpha_h_terms",
            "cpml_kappa_h_direct_terms",
            "cpml_sigma_e_terms",
            "cpml_kappa_e_terms",
            "cpml_alpha_e_terms",
            "cpml3d_a_h_terms",
            "cpml3d_b_h_terms",
            "cpml3d_inv_kappa_h_terms",
            "cpml3d_a_e_terms",
            "cpml3d_b_e_terms",
            "cpml3d_inv_kappa_e_terms",
            "cpml3d_sigma_h_terms",
            "cpml3d_kappa_h_terms",
            "cpml3d_alpha_h_terms",
            "cpml3d_sigma_e_terms",
            "cpml3d_kappa_e_terms",
            "cpml3d_alpha_e_terms",
            "fp_h_decay_x",
            "fp_h_source_x",
            "fp_h_decay_y",
            "fp_h_source_y",
            "fp_h_decay_z",
            "fp_h_source_z",
            "fp_e_decay_x",
            "fp_e_source_x",
            "fp_e_decay_y",
            "fp_e_source_y",
            "fp_e_decay_z",
            "fp_e_source_z",
            "fp_ex_mask",
            "fp_ey_mask",
            "fp_ez_mask",
            "fp_hx_mask",
            "fp_hy_mask",
            "fp_hz_mask",
            "ex_metal_mask",
            "ey_metal_mask",
            "ez_metal_mask",
            "hx_metal_mask",
            "hy_metal_mask",
            "hz_metal_mask",
            "ex_storage_mask",
            "ey_storage_mask",
            "ez_storage_mask",
            "hx_storage_mask",
            "hy_storage_mask",
            "hz_storage_mask",
        ):
            _add_array_entries(
                entries,
                name,
                getattr(self, name),
                category="compiled_static_terms",
            )

        for spec_i, spec in enumerate(self.source_specs):
            _add_array_entries(
                entries,
                f"source_specs[{spec_i}].coeff",
                spec.coeff,
                category="source_specs",
            )
            _add_array_entries(
                entries,
                f"source_specs[{spec_i}].waveform",
                spec.waveform,
                category="source_specs",
            )

        for spec_i, spec in enumerate(self.monitor_specs):
            for field_info in dataclass_fields(spec):
                value = getattr(spec, field_info.name)
                _add_array_entries(
                    entries,
                    f"monitor_specs[{spec_i}].{field_info.name}",
                    value,
                    category="monitor_specs",
                )

        if include_runtime and self.monitor_specs:
            max_records = max(
                1, monitor_state_size(self.monitor_specs, self.config.num_steps)
            )
            max_freq = monitor_frequency_size(self.monitor_specs)
            max_points = monitor_dft_point_size(self.monitor_specs)
            dft_dtype = np.dtype(
                np.float64 if jax.config.jax_enable_x64 else np.float32
            )
            monitor_shapes = {
                "powers": (len(self.monitor_specs), max_records),
                "timestamps": (len(self.monitor_specs), max_records),
                "counts": (len(self.monitor_specs),),
                "freq_flux_re": (len(self.monitor_specs), max_freq),
                "freq_flux_im": (len(self.monitor_specs), max_freq),
                "freq_phase_re": (len(self.monitor_specs), max_freq),
                "freq_phase_im": (len(self.monitor_specs), max_freq),
                "dft_vec_re": (
                    len(self.monitor_specs),
                    6,
                    max_freq,
                    max_points,
                ),
                "dft_vec_im": (
                    len(self.monitor_specs),
                    6,
                    max_freq,
                    max_points,
                ),
                "dft_weight_sum": (len(self.monitor_specs), max_freq),
            }
            for name, shape in monitor_shapes.items():
                dtype = (
                    np.dtype(np.int32)
                    if name == "counts"
                    else dft_dtype
                    if name.startswith("dft_")
                    else np.dtype(np.float32)
                )
                entries.append(
                    {
                        "name": f"monitor_state.{name}",
                        "category": "monitor_state",
                        "residency": "runtime",
                        "shape": [int(v) for v in shape],
                        "dtype": str(dtype),
                        "bytes": int(np.prod(shape, dtype=np.int64) * dtype.itemsize),
                    }
                )

        if include_runtime:
            snapshot_state = self._empty_snapshot_state()
            if snapshot_state is not None:
                for name, arr in zip(
                    ("fields", "steps", "times", "count"),
                    snapshot_state,
                    strict=True,
                ):
                    _add_array_entries(
                        entries,
                        f"snapshot_state.{name}",
                        arr,
                        category="snapshot_state",
                        residency="runtime",
                    )

        report = _memory_report(entries)
        report["referenced_inputs"] = _memory_report(referenced_entries)
        report["config"] = {
            "num_steps": int(self.config.num_steps),
            "is_3d": bool(self.config.is_3d),
            "loop_kind": self.config.loop_kind,
            "use_cpml_3d": bool(self.use_cpml_3d),
            "use_primitive_cpml_3d_terms": bool(self.use_primitive_cpml_3d_terms),
            "use_cpml_3d_packed_psi": bool(self.use_cpml_3d_packed_psi),
            "sharding": {
                "enabled": bool(self.storage_layout.enabled),
                "axis": self.storage_layout.axis_name,
                "num_devices": int(self.storage_layout.num_devices),
                "backend": self.storage_layout.backend,
                "logical_shapes": {
                    name: [int(v) for v in shape]
                    for name, shape in self.storage_layout.logical_shapes.items()
                },
                "storage_shapes": {
                    name: [int(v) for v in shape]
                    for name, shape in self.storage_layout.storage_shapes.items()
                },
            },
        }
        if self.storage_layout.enabled:
            report["per_device_total_bytes"] = int(
                np.ceil(report["total_bytes"] / self.storage_layout.num_devices)
            )
            report["per_device_total_gib"] = report["per_device_total_bytes"] / 1024**3
        return report

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

        engine_state = self.prepare_engine_state(engine_state)
        monitor_state = self._place_pytree(monitor_state, shard_arrays=False)
        coeffs = self._place_update_coefficients(self._update_coefficients())

        if self._compiled_scan is None:
            self._build_scan()

        snapshot_state = self._empty_snapshot_state()
        if snapshot_state is not None:
            snapshot_state = self._place_pytree(snapshot_state, shard_arrays=False)
        if snapshot_state is None:
            eng, mon, mat, snapshots = self._compiled_scan(
                engine_state,
                monitor_state,
                coeffs,
            )
        else:
            eng, mon, mat, snapshots = self._compiled_scan(
                engine_state,
                monitor_state,
                coeffs,
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


def _has_positive_conductivity(conductivity_region: jnp.ndarray) -> bool:
    return bool(np.asarray(conductivity_region).size) and bool(
        (np.asarray(conductivity_region) > 0.0).any()
    )


def _precompute_e_lossless_source_coefficient(
    *,
    shape: tuple[int, int, int],
    permittivity: jnp.ndarray,
    dt: float,
    region: tuple[slice, slice, slice],
) -> jnp.ndarray:
    source = jnp.zeros(shape, dtype=jnp.float32)
    local = (dt / (ops.EPS_0 * permittivity)).astype(jnp.float32)
    return source.at[region].set(local)


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

    logical_fields = run_cfg.fields
    resolution = float(run_cfg.resolution)
    dt = float(run_cfg.dt)
    num_steps = int(run_cfg.num_steps)
    total_steps = int(getattr(run_cfg, "total_steps", num_steps))
    t0 = float(getattr(run_cfg, "t0", 0.0))
    sharding_cfg = normalize_sharding_config(getattr(run_cfg, "sharding", None))
    full_pec_3d_static = bool(run_cfg.is_3d and has_full_pec_3d(boundaries))
    storage_layout, sharding_devices = _build_storage_layout(
        logical_fields,
        sharding_cfg,
        is_3d=bool(run_cfg.is_3d),
        full_pec_3d=full_pec_3d_static,
    )
    fields = _make_storage_fields_proxy(logical_fields, storage_layout)
    effective_sharding = (
        ShardingConfig(
            enabled=True,
            axis=storage_layout.axis_name,
            num_devices=storage_layout.num_devices,
            backend=sharding_cfg.backend,
        )
        if storage_layout.enabled
        else ShardingConfig(
            enabled=False,
            axis=sharding_cfg.axis,
            num_devices=sharding_cfg.num_devices,
            backend=sharding_cfg.backend,
        )
    )

    source_specs = compile_source_specs(
        sources=sources,
        fields=logical_fields,
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
        sharding=effective_sharding,
    )

    has_cpml_3d = bool(
        run_cfg.is_3d
        and getattr(fields, "has_cpml", False)
        and getattr(fields, "pml_data", None)
    )
    use_3d_material_coefficients = bool(
        run_cfg.is_3d and (not has_cpml_3d) and (not full_pec_3d_static)
    )

    empty3 = jnp.zeros((0, 0, 0), dtype=jnp.float32)
    if use_3d_material_coefficients:
        h_decay_x = h_source_x = h_decay_y = h_source_y = empty3
        h_decay_z = h_source_z = empty3
        h_source_lossless_x = h_source_lossless_y = h_source_lossless_z = empty3
        h_sigma_m_x = fields.sigma_m_hx
        h_sigma_m_y = fields.sigma_m_hy
        h_sigma_m_z = fields.sigma_m_hz
    else:
        (
            (h_decay_x, h_source_x, h_source_lossless_x),
            (
                h_decay_y,
                h_source_y,
                h_source_lossless_y,
            ),
            (h_decay_z, h_source_z, h_source_lossless_z),
        ) = (
            ops.precompute_h_update_coefficients(fields.sigma_m_hx, dt),
            ops.precompute_h_update_coefficients(fields.sigma_m_hy, dt),
            ops.precompute_h_update_coefficients(fields.sigma_m_hz, dt),
        )
        h_sigma_m_x = h_sigma_m_y = h_sigma_m_z = empty3

    if use_3d_material_coefficients:
        e_decay_x = e_source_x = e_decay_y = e_source_y = empty3
        e_decay_z = e_source_z = empty3
        e_source_lossless_x = e_source_lossless_y = e_source_lossless_z = empty3
        e_conductivity_x = fields.sig_x
        e_conductivity_y = fields.sig_y
        e_conductivity_z = fields.sig_z
        e_inv_permittivity_x = 1.0 / fields.eps_x
        e_inv_permittivity_y = 1.0 / fields.eps_y
        e_inv_permittivity_z = 1.0 / fields.eps_z
    else:
        (
            (e_decay_x, e_source_x, e_source_lossless_x),
            (
                e_decay_y,
                e_source_y,
                e_source_lossless_y,
            ),
            (
                e_decay_z,
                e_source_z,
                e_source_lossless_z,
            ),
        ) = (
            ops.precompute_e_update_coefficients(
                shape=fields.Ex.shape,
                conductivity=fields.sig_x,
                permittivity=fields.eps_x,
                dt=dt,
                region=fields.region_x,
            ),
            ops.precompute_e_update_coefficients(
                shape=fields.Ey.shape,
                conductivity=fields.sig_y,
                permittivity=fields.eps_y,
                dt=dt,
                region=fields.region_y,
            ),
            ops.precompute_e_update_coefficients(
                shape=fields.Ez.shape,
                conductivity=fields.sig_z,
                permittivity=fields.eps_z,
                dt=dt,
                region=fields.region_z,
            ),
        )
        e_conductivity_x = e_conductivity_y = e_conductivity_z = empty3
        e_inv_permittivity_x = e_inv_permittivity_y = e_inv_permittivity_z = empty3
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
    cpml3d_sigma_h_terms = _empty_cpml_3d_terms(jnp.float32)
    cpml3d_kappa_h_terms = _empty_cpml_3d_terms(jnp.float32)
    cpml3d_alpha_h_terms = _empty_cpml_3d_terms(jnp.float32)
    cpml3d_sigma_e_terms = _empty_cpml_3d_terms(jnp.float32)
    cpml3d_kappa_e_terms = _empty_cpml_3d_terms(jnp.float32)
    cpml3d_alpha_e_terms = _empty_cpml_3d_terms(jnp.float32)
    cpml3d_metallic_edges = frozenset()
    use_primitive_cpml_3d_terms = False
    use_cpml_3d_packed_psi = False
    cpml3d_h_slab_specs = tuple(
        CpmlPackedSlabSpec(axis=0, low=0, high=0, shape=(0, 0, 0)) for _ in range(6)
    )
    cpml3d_e_slab_specs = tuple(
        CpmlPackedSlabSpec(axis=0, low=0, high=0, shape=(0, 0, 0)) for _ in range(6)
    )
    cpml3d_h_psi_shapes = tuple((0, 0, 0) for _ in range(6))
    cpml3d_e_psi_shapes = tuple((0, 0, 0) for _ in range(6))
    if bool(run_cfg.is_3d):
        cpml3d_metallic_edges = frozenset(
            resolve_metallic_edges(boundaries, is_3d=True)
        )
        use_cpml_3d = bool(
            getattr(fields, "has_cpml", False) and getattr(fields, "pml_data", None)
        )
        if use_cpml_3d:
            primitive_terms = build_cpml_3d_primitive_terms(fields.pml_data)
            if primitive_terms is not None:
                use_primitive_cpml_3d_terms = True
                cpml3d_sigma_h_terms = primitive_terms.sigma_h_terms
                cpml3d_kappa_h_terms = primitive_terms.kappa_h_terms
                cpml3d_alpha_h_terms = primitive_terms.alpha_h_terms
                cpml3d_sigma_e_terms = primitive_terms.sigma_e_terms
                cpml3d_kappa_e_terms = primitive_terms.kappa_e_terms
                cpml3d_alpha_e_terms = primitive_terms.alpha_e_terms
                (
                    cpml3d_a_h_terms,
                    cpml3d_b_h_terms,
                    cpml3d_inv_kappa_h_terms,
                ) = cpml_precompute_native_terms(
                    cpml3d_sigma_h_terms,
                    cpml3d_kappa_h_terms,
                    cpml3d_alpha_h_terms,
                    run_cfg.dt,
                )
                (
                    cpml3d_a_e_terms,
                    cpml3d_b_e_terms,
                    cpml3d_inv_kappa_e_terms,
                ) = cpml_precompute_native_terms(
                    cpml3d_sigma_e_terms,
                    cpml3d_kappa_e_terms,
                    cpml3d_alpha_e_terms,
                    run_cfg.dt,
                )
                h_full_shapes = _cpml_3d_h_term_shapes(fields.Hx, fields.Hy, fields.Hz)
                e_full_shapes = _cpml_3d_e_term_shapes(fields.Ex, fields.Ey, fields.Ez)
                cpml3d_h_slab_specs = _build_cpml_packed_slab_specs(
                    cpml3d_a_h_terms,
                    cpml3d_inv_kappa_h_terms,
                    h_full_shapes,
                    CPML_3D_H_DERIVATIVES,
                )
                cpml3d_e_slab_specs = _build_cpml_packed_slab_specs(
                    cpml3d_a_e_terms,
                    cpml3d_inv_kappa_e_terms,
                    e_full_shapes,
                    CPML_3D_E_DERIVATIVES,
                )
                full_psi_shapes = (*h_full_shapes, *e_full_shapes)
                use_cpml_3d_packed_psi = _should_pack_cpml_3d_psi(
                    full_psi_shapes, fields.Hx.dtype
                )
                if use_cpml_3d_packed_psi:
                    cpml3d_h_psi_shapes = _cpml_packed_slab_shapes(cpml3d_h_slab_specs)
                    cpml3d_e_psi_shapes = _cpml_packed_slab_shapes(cpml3d_e_slab_specs)
                else:
                    cpml3d_h_psi_shapes = h_full_shapes
                    cpml3d_e_psi_shapes = e_full_shapes
            else:
                terms = build_cpml_3d_terms(fields.pml_data, dt=run_cfg.dt)
                if terms is not None:
                    cpml3d_a_h_terms = terms.a_h_terms
                    cpml3d_b_h_terms = terms.b_h_terms
                    cpml3d_inv_kappa_h_terms = terms.inv_kappa_h_terms
                    cpml3d_a_e_terms = terms.a_e_terms
                    cpml3d_b_e_terms = terms.b_e_terms
                    cpml3d_inv_kappa_e_terms = terms.inv_kappa_e_terms
                    cpml3d_h_psi_shapes = _cpml_3d_h_term_shapes(
                        fields.Hx, fields.Hy, fields.Hz
                    )
                    cpml3d_e_psi_shapes = _cpml_3d_e_term_shapes(
                        fields.Ex, fields.Ey, fields.Ez
                    )
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
        fp_state = initialize_full_pec_3d_state(logical_fields)
        fp_has_loss = any(
            _has_positive_conductivity(arr)
            for arr in (
                fp_state.sigma_m_hx,
                fp_state.sigma_m_hy,
                fp_state.sigma_m_hz,
                fp_state.sig_x_region,
                fp_state.sig_y_region,
                fp_state.sig_z_region,
            )
        )
        fp_ex_mask = _pad_high_to_shape(
            fp_state.masks["Ex"], storage_layout.storage_shapes["Ex"], pad_value=True
        )
        fp_ey_mask = _pad_high_to_shape(
            fp_state.masks["Ey"], storage_layout.storage_shapes["Ey"], pad_value=True
        )
        fp_ez_mask = _pad_high_to_shape(
            fp_state.masks["Ez"], storage_layout.storage_shapes["Ez"], pad_value=True
        )
        fp_hx_mask = _pad_high_to_shape(
            fp_state.masks["Hx"], storage_layout.storage_shapes["Hx"], pad_value=True
        )
        fp_hy_mask = _pad_high_to_shape(
            fp_state.masks["Hy"], storage_layout.storage_shapes["Hy"], pad_value=True
        )
        fp_hz_mask = _pad_high_to_shape(
            fp_state.masks["Hz"], storage_layout.storage_shapes["Hz"], pad_value=True
        )
        if fp_has_loss:
            (
                (fp_h_decay_x, fp_h_decay_y, fp_h_decay_z),
                (
                    fp_h_source_x,
                    fp_h_source_y,
                    fp_h_source_z,
                ),
            ) = full_pec_h_update_coefficients_3d(fp_state, dt)
            (
                (fp_e_decay_x, fp_e_decay_y, fp_e_decay_z),
                (
                    fp_e_source_x,
                    fp_e_source_y,
                    fp_e_source_z,
                ),
            ) = full_pec_e_update_coefficients_3d(fp_state, dt)
        else:
            fp_h_decay_x = jnp.zeros((0, 0, 0), dtype=jnp.float32)
            fp_h_source_x = jnp.asarray(dt / ops.MU_0, dtype=jnp.float32)
            fp_h_decay_y = jnp.zeros((0, 0, 0), dtype=jnp.float32)
            fp_h_source_y = jnp.asarray(dt / ops.MU_0, dtype=jnp.float32)
            fp_h_decay_z = jnp.zeros((0, 0, 0), dtype=jnp.float32)
            fp_h_source_z = jnp.asarray(dt / ops.MU_0, dtype=jnp.float32)
            fp_e_decay_x = jnp.zeros((0, 0, 0), dtype=jnp.float32)
            fp_e_source_x = _precompute_e_lossless_source_coefficient(
                shape=tuple(fp_state.Ex.shape),
                permittivity=fp_state.eps_x_region,
                dt=dt,
                region=(slice(1, -1), slice(1, -1), slice(None)),
            )
            fp_e_decay_y = jnp.zeros((0, 0, 0), dtype=jnp.float32)
            fp_e_source_y = _precompute_e_lossless_source_coefficient(
                shape=tuple(fp_state.Ey.shape),
                permittivity=fp_state.eps_y_region,
                dt=dt,
                region=(slice(1, -1), slice(None), slice(1, -1)),
            )
            fp_e_decay_z = jnp.zeros((0, 0, 0), dtype=jnp.float32)
            fp_e_source_z = _precompute_e_lossless_source_coefficient(
                shape=tuple(fp_state.Ez.shape),
                permittivity=fp_state.eps_z_region,
                dt=dt,
                region=(slice(None), slice(1, -1), slice(1, -1)),
            )
        h_decay_x = fp_h_decay_x
        h_source_x = fp_h_source_x
        h_decay_y = fp_h_decay_y
        h_source_y = fp_h_source_y
        h_decay_z = fp_h_decay_z
        h_source_z = fp_h_source_z
        e_decay_x = fp_e_decay_x
        e_source_x = fp_e_source_x
        e_decay_y = fp_e_decay_y
        e_source_y = fp_e_source_y
        e_decay_z = fp_e_decay_z
        e_source_z = fp_e_source_z
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
    if bool(run_cfg.is_3d):
        empty_mask3 = jnp.zeros((0, 0, 0), dtype=bool)
        metallic_masks = {
            "Ex": empty_mask3,
            "Ey": empty_mask3,
            "Ez": empty_mask3,
            "Hx": empty_mask3,
            "Hy": empty_mask3,
            "Hz": empty_mask3,
        }

    h_source_lossless_x = _empty_like_rank(h_source_lossless_x)
    h_source_lossless_y = _empty_like_rank(h_source_lossless_y)
    h_source_lossless_z = _empty_like_rank(h_source_lossless_z)
    e_source_lossless_x = _empty_like_rank(e_source_lossless_x)
    e_source_lossless_y = _empty_like_rank(e_source_lossless_y)
    e_source_lossless_z = _empty_like_rank(e_source_lossless_z)

    return CompiledSimulation(
        config=config,
        material_spec=CompiledMaterialSpec(model_kind="linear"),
        source_specs=source_specs,
        monitor_specs=monitor_specs,
        monitor_devices=monitor_devices,
        field_shape_ex=tuple(int(v) for v in logical_fields.Ex.shape),
        field_shape_ey=tuple(int(v) for v in logical_fields.Ey.shape),
        field_shape_ez=tuple(int(v) for v in logical_fields.Ez.shape),
        field_shape_hx=tuple(int(v) for v in logical_fields.Hx.shape),
        field_shape_hy=tuple(int(v) for v in logical_fields.Hy.shape),
        field_shape_hz=tuple(int(v) for v in logical_fields.Hz.shape),
        storage_layout=storage_layout,
        sharding_devices=sharding_devices,
        storage_shape_ex=storage_layout.storage_shapes["Ex"],
        storage_shape_ey=storage_layout.storage_shapes["Ey"],
        storage_shape_ez=storage_layout.storage_shapes["Ez"],
        storage_shape_hx=storage_layout.storage_shapes["Hx"],
        storage_shape_hy=storage_layout.storage_shapes["Hy"],
        storage_shape_hz=storage_layout.storage_shapes["Hz"],
        h_decay_x=h_decay_x,
        h_source_x=h_source_x,
        h_source_lossless_x=h_source_lossless_x,
        h_sigma_m_x=h_sigma_m_x,
        h_decay_y=h_decay_y,
        h_source_y=h_source_y,
        h_source_lossless_y=h_source_lossless_y,
        h_sigma_m_y=h_sigma_m_y,
        h_decay_z=h_decay_z,
        h_source_z=h_source_z,
        h_source_lossless_z=h_source_lossless_z,
        h_sigma_m_z=h_sigma_m_z,
        e_decay_x=e_decay_x,
        e_source_x=e_source_x,
        e_source_lossless_x=e_source_lossless_x,
        e_conductivity_x=e_conductivity_x,
        e_inv_permittivity_x=e_inv_permittivity_x,
        e_decay_y=e_decay_y,
        e_source_y=e_source_y,
        e_source_lossless_y=e_source_lossless_y,
        e_conductivity_y=e_conductivity_y,
        e_inv_permittivity_y=e_inv_permittivity_y,
        e_decay_z=e_decay_z,
        e_source_z=e_source_z,
        e_source_lossless_z=e_source_lossless_z,
        e_conductivity_z=e_conductivity_z,
        e_inv_permittivity_z=e_inv_permittivity_z,
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
        cpml3d_sigma_h_terms=cpml3d_sigma_h_terms,
        cpml3d_kappa_h_terms=cpml3d_kappa_h_terms,
        cpml3d_alpha_h_terms=cpml3d_alpha_h_terms,
        cpml3d_sigma_e_terms=cpml3d_sigma_e_terms,
        cpml3d_kappa_e_terms=cpml3d_kappa_e_terms,
        cpml3d_alpha_e_terms=cpml3d_alpha_e_terms,
        cpml3d_metallic_edges=cpml3d_metallic_edges,
        use_primitive_cpml_3d_terms=use_primitive_cpml_3d_terms,
        use_cpml_3d_packed_psi=use_cpml_3d_packed_psi,
        cpml3d_h_slab_specs=cpml3d_h_slab_specs,
        cpml3d_e_slab_specs=cpml3d_e_slab_specs,
        cpml3d_h_psi_shapes=cpml3d_h_psi_shapes,
        cpml3d_e_psi_shapes=cpml3d_e_psi_shapes,
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
        ex_metal_mask=metallic_masks["Ex"],
        ey_metal_mask=metallic_masks["Ey"],
        ez_metal_mask=metallic_masks["Ez"],
        hx_metal_mask=metallic_masks["Hx"],
        hy_metal_mask=metallic_masks["Hy"],
        hz_metal_mask=metallic_masks["Hz"],
        ex_storage_mask=storage_layout.valid_masks["Ex"],
        ey_storage_mask=storage_layout.valid_masks["Ey"],
        ez_storage_mask=storage_layout.valid_masks["Ez"],
        hx_storage_mask=storage_layout.valid_masks["Hx"],
        hy_storage_mask=storage_layout.valid_masks["Hy"],
        hz_storage_mask=storage_layout.valid_masks["Hz"],
    )
