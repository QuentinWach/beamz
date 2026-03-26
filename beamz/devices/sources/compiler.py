"""Compile high-level source objects into static packed source specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, MU_0
from beamz.devices.sources.gaussian import GaussianSource
from beamz.devices.sources.mode import ModeSource, _get_3d_huygens_terms


@dataclass(frozen=True)
class BatchedSlabGroup:
    """Stacked slab sources for a single (timing, component) group.

    Enables fori_loop-based application that keeps HLO size constant
    regardless of source count.
    """

    waveforms: jnp.ndarray  # (n, total_steps)
    coeffs: jnp.ndarray  # (n, *max_sizes)
    starts: jnp.ndarray  # (n, ndim) int32
    starts_tuple: tuple[tuple[int, ...], ...]  # static starts for tiny-n fast paths
    max_sizes: tuple[int, ...]  # static — used for dynamic_slice
    n: int  # static — number of specs


def batch_slab_specs(
    specs: tuple[CompiledSourceSpec, ...],
) -> tuple[BatchedSlabGroup | None, tuple[CompiledSourceSpec, ...]]:
    """Split specs into a batched slab group and remaining non-slab specs."""
    slab = [
        s
        for s in specs
        if s.is_slab and s.slab_starts is not None and s.slab_sizes is not None
    ]
    rest = tuple(
        s
        for s in specs
        if not (s.is_slab and s.slab_starts is not None and s.slab_sizes is not None)
    )
    if not slab:
        return None, specs
    ndim = len(slab[0].slab_sizes)
    max_sizes = tuple(max(s.slab_sizes[d] for s in slab) for d in range(ndim))
    padded = []
    for s in slab:
        pad_width = tuple((0, max_sizes[d] - s.slab_sizes[d]) for d in range(ndim))
        padded.append(jnp.pad(s.coeff, pad_width))
    return (
        BatchedSlabGroup(
            waveforms=jnp.stack([s.waveform for s in slab]),
            coeffs=jnp.stack(padded),
            starts=jnp.array([list(s.slab_starts) for s in slab], dtype=jnp.int32),
            starts_tuple=tuple(tuple(int(v) for v in s.slab_starts) for s in slab),
            max_sizes=max_sizes,
            n=len(slab),
        ),
        rest,
    )


@dataclass(frozen=True)
class CompiledSourceSpec:
    """Single packed source term consumed by compiled step kernels."""

    component: str
    timing: str  # "pre_e", "e", "h"
    index: tuple[Any, ...]
    coeff: jnp.ndarray
    waveform: jnp.ndarray
    is_slab: bool = False
    slab_starts: tuple[int, ...] | None = None
    slab_sizes: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SourceExecutionGroup:
    """One compiled source execution group for a (timing, component) slot."""

    batch: BatchedSlabGroup | None
    rest: tuple[CompiledSourceSpec, ...]


@dataclass(frozen=True)
class SourceExecutionPlan:
    """Compiled source execution groups arranged by timestep phase."""

    pre_e_ex: SourceExecutionGroup
    pre_e_ey: SourceExecutionGroup
    pre_e_ez: SourceExecutionGroup
    h_x: SourceExecutionGroup
    h_y: SourceExecutionGroup
    h_z: SourceExecutionGroup
    e_x: SourceExecutionGroup
    e_y: SourceExecutionGroup
    e_z: SourceExecutionGroup


def apply_source_specs(
    arr: jnp.ndarray,
    abs_step: jnp.ndarray,
    specs: tuple[CompiledSourceSpec, ...],
) -> jnp.ndarray:
    """Apply unpacked source specs directly."""
    out = arr
    for spec in specs:
        safe_idx = jnp.clip(abs_step, 0, spec.waveform.shape[0] - 1)
        amp = spec.waveform[safe_idx]
        if spec.is_slab and spec.slab_starts is not None and spec.slab_sizes is not None:
            patch = spec.coeff * amp
            cur = jax.lax.dynamic_slice(out, spec.slab_starts, spec.slab_sizes)
            out = jax.lax.dynamic_update_slice(out, cur + patch, spec.slab_starts)
        else:
            out = out.at[spec.index].add(spec.coeff * amp)
    return out


def apply_batched_slab_sources(
    arr: jnp.ndarray,
    abs_step: jnp.ndarray,
    group: BatchedSlabGroup,
    *,
    source_single_slab_dense: bool = False,
) -> jnp.ndarray:
    """Apply stacked slab sources via a constant-size fori_loop."""
    safe_idx = jnp.clip(abs_step, 0, group.waveforms.shape[1] - 1)
    ndim = len(group.max_sizes)

    if group.n == 1:
        amp = group.waveforms[0, safe_idx]
        starts_0 = group.starts_tuple[0]
        if source_single_slab_dense:
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


def apply_source_group(
    arr: jnp.ndarray,
    abs_step: jnp.ndarray,
    group: SourceExecutionGroup,
    *,
    source_single_slab_dense: bool = False,
) -> jnp.ndarray:
    """Apply a compiled source execution group."""
    if group.batch is not None:
        arr = apply_batched_slab_sources(
            arr,
            abs_step,
            group.batch,
            source_single_slab_dense=source_single_slab_dense,
        )
    if group.rest:
        arr = apply_source_specs(arr, abs_step, group.rest)
    return arr


def build_source_applier(
    group: SourceExecutionGroup,
    *,
    source_single_slab_dense: bool = False,
):
    """Return the narrowest applier closure for a source execution group."""
    if group.batch is None and not group.rest:
        return lambda arr, _abs_step: arr
    if group.batch is not None and not group.rest:
        return lambda arr, abs_step: apply_batched_slab_sources(
            arr,
            abs_step,
            group.batch,
            source_single_slab_dense=source_single_slab_dense,
        )
    if group.batch is None:
        return lambda arr, abs_step: apply_source_specs(arr, abs_step, group.rest)
    return lambda arr, abs_step: apply_source_group(
        arr,
        abs_step,
        group,
        source_single_slab_dense=source_single_slab_dense,
    )


def build_source_execution_plan(
    specs: tuple[CompiledSourceSpec, ...],
) -> SourceExecutionPlan:
    """Group compiled source specs by timestep phase and field component."""

    def _group(timing: str, component: str) -> SourceExecutionGroup:
        matched = tuple(
            spec
            for spec in specs
            if spec.timing == timing and spec.component == component
        )
        batch, rest = batch_slab_specs(matched)
        return SourceExecutionGroup(batch=batch, rest=rest)

    return SourceExecutionPlan(
        pre_e_ex=_group("pre_e", "Ex"),
        pre_e_ey=_group("pre_e", "Ey"),
        pre_e_ez=_group("pre_e", "Ez"),
        h_x=_group("h", "Hx"),
        h_y=_group("h", "Hy"),
        h_z=_group("h", "Hz"),
        e_x=_group("e", "Ex"),
        e_y=_group("e", "Ey"),
        e_z=_group("e", "Ez"),
    )


def _as_slab_spec(
    component: str,
    timing: str,
    index: tuple[Any, ...],
    coeff,
    waveform: jnp.ndarray,
    target_shape: tuple[int, ...],
) -> CompiledSourceSpec:
    """Build a packed source spec, using slab metadata for slice/int indices."""
    starts: list[int] = []
    sizes: list[int] = []

    if len(index) == len(target_shape):
        for dim, key in enumerate(index):
            if isinstance(key, slice):
                start, stop, step = key.indices(target_shape[dim])
                if step != 1:
                    break
                starts.append(int(start))
                sizes.append(int(stop - start))
                continue
            if isinstance(key, (int, np.integer)):
                idx = int(key)
                if idx < 0:
                    idx += int(target_shape[dim])
                if idx < 0 or idx >= int(target_shape[dim]):
                    break
                starts.append(idx)
                sizes.append(1)
                continue
            break
        else:
            coeff_np = np.asarray(coeff, dtype=np.float32)
            slab_sizes = tuple(sizes)
            expected = int(np.prod(slab_sizes))
            if coeff_np.size == expected:
                coeff_np = coeff_np.reshape(slab_sizes)
                return CompiledSourceSpec(
                    component=component,
                    timing=timing,
                    index=index,
                    coeff=jnp.asarray(coeff_np, dtype=jnp.float32),
                    waveform=waveform,
                    is_slab=True,
                    slab_starts=tuple(starts),
                    slab_sizes=slab_sizes,
                )

    return CompiledSourceSpec(
        component=component,
        timing=timing,
        index=index,
        coeff=jnp.asarray(coeff, dtype=jnp.float32),
        waveform=waveform,
    )


def _sample_waveform(
    get_signal_value,
    t0: float,
    dt: float,
    num_steps: int,
    offset_fn,
    total_steps: int | None = None,
):
    n = total_steps if total_steps is not None else num_steps
    start = float(t0)
    vals = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        t = start + i * dt
        vals[i] = float(get_signal_value(offset_fn(t, dt), dt))
    return jnp.asarray(vals)


def _match_shape(profile: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    profile = np.asarray(profile)
    if profile.shape == target_shape:
        return profile
    if profile.ndim != len(target_shape):
        return np.zeros(target_shape, dtype=profile.dtype)

    slices = tuple(
        slice(0, min(profile.shape[i], target_shape[i])) for i in range(profile.ndim)
    )
    trimmed = profile[slices]
    out = np.zeros(target_shape, dtype=profile.dtype)
    insert = tuple(slice(0, trimmed.shape[i]) for i in range(trimmed.ndim))
    out[insert] = trimmed
    return out


def _mode_3d_profiles_and_indices(src: ModeSource):
    profiles = {
        "Ex": getattr(src, "_Ex_profile", None),
        "Ey": getattr(src, "_Ey_profile", None),
        "Ez": getattr(src, "_Ez_profile", None),
        "Hx": getattr(src, "_Hx_profile", None),
        "Hy": getattr(src, "_Hy_profile", None),
        "Hz": getattr(src, "_Hz_profile", None),
    }
    indices = {
        "Ex": getattr(src, "_Ex_indices", None),
        "Ey": getattr(src, "_Ey_indices", None),
        "Ez": getattr(src, "_Ez_indices", None),
        "Hx": getattr(src, "_Hx_indices", None),
        "Hy": getattr(src, "_Hy_indices", None),
        "Hz": getattr(src, "_Hz_indices", None),
    }
    return profiles, indices


def compile_source_specs(
    devices: list,
    fields,
    dt: float,
    resolution: float,
    num_steps: int,
    t0: float,
    total_steps: int | None = None,
) -> tuple[CompiledSourceSpec, ...]:
    """Compile source devices into packed source specs.

    v0.3 first-class support: GaussianSource and ModeSource.
    """
    specs: list[CompiledSourceSpec] = []

    for device in devices:
        if isinstance(device, GaussianSource):
            specs.extend(
                _compile_gaussian_source(
                    device=device,
                    fields=fields,
                    dt=dt,
                    num_steps=num_steps,
                    t0=t0,
                    resolution=resolution,
                    total_steps=total_steps,
                )
            )
        elif isinstance(device, ModeSource):
            specs.extend(
                _compile_mode_source(
                    device=device,
                    fields=fields,
                    dt=dt,
                    num_steps=num_steps,
                    t0=t0,
                    resolution=resolution,
                    total_steps=total_steps,
                )
            )

    return tuple(specs)


def _compile_gaussian_source(
    device: GaussianSource,
    fields,
    dt: float,
    num_steps: int,
    t0: float,
    resolution: float,
    total_steps: int | None = None,
) -> tuple[CompiledSourceSpec, ...]:
    # Initialize spatial profile once.
    is_3d = len(device.position) >= 3 if hasattr(device.position, "__len__") else False
    if device._spatial_profile_ez is None:
        device._init_spatial_profile(fields.Ez.shape, resolution, is_3d)

    idx = device._grid_indices
    eps_region = np.asarray(fields.permittivity[idx])
    profile = np.asarray(device._spatial_profile_ez)

    coeff = -profile * dt / (EPS_0 * eps_region)
    waveform = _sample_waveform(
        device._get_signal_value,
        t0=t0,
        dt=dt,
        num_steps=num_steps,
        offset_fn=lambda t, dt_: t + 0.5 * dt_,
        total_steps=total_steps,
    )

    return (
        _as_slab_spec(
            component="Ez",
            timing="pre_e",
            index=idx,
            coeff=coeff,
            waveform=waveform,
            target_shape=tuple(fields.Ez.shape),
        ),
    )


def _compile_mode_source(
    device: ModeSource,
    fields,
    dt: float,
    num_steps: int,
    t0: float,
    resolution: float,
    total_steps: int | None = None,
) -> tuple[CompiledSourceSpec, ...]:
    if (
        (not getattr(device, "_initialized", False))
        or (getattr(device, "_grid_shape", None) != fields.permittivity.shape)
        or (getattr(device, "_resolution", None) is None)
        or (not np.isclose(getattr(device, "_resolution", 0.0), resolution))
    ):
        device.initialize(fields.permittivity, resolution, dt=dt)

    is_3d = bool(getattr(device, "_is_3d", False))

    h_waveform = _sample_waveform(
        device._get_signal_value,
        t0=t0,
        dt=dt,
        num_steps=num_steps,
        offset_fn=lambda t, dt_: t + 0.5 * dt_,
        total_steps=total_steps,
    )

    dt_physical = float(getattr(device, "_dt_physical", 0.0))
    # E injection is applied after the E update within each Yee step; use the
    # same half-step base time as H plus physical plane delay to keep the 3D
    # Huygens pair phase-consistent with the 2D implementation.
    e_waveform = _sample_waveform(
        device._get_signal_value,
        t0=t0,
        dt=dt,
        num_steps=num_steps,
        offset_fn=lambda t, dt_: t + 0.5 * dt_ + dt_physical,
        total_steps=total_steps,
    )

    if is_3d:
        return _compile_mode_source_3d(
            device,
            fields,
            dt,
            resolution,
            h_waveform,
            e_waveform,
        )
    return _compile_mode_source_2d(
        device,
        fields,
        dt,
        resolution,
        h_waveform,
        e_waveform,
    )


def _build_coeff(
    profile: np.ndarray,
    target: np.ndarray,
    dt: float,
    scale_denom: np.ndarray,
) -> jnp.ndarray:
    profile = _match_shape(np.asarray(profile), target.shape)
    coeff = profile * dt / scale_denom
    return jnp.asarray(coeff, dtype=jnp.float32)


def _compile_mode_source_2d(
    src: ModeSource,
    fields,
    dt: float,
    resolution: float,
    h_waveform: jnp.ndarray,
    e_waveform: jnp.ndarray,
) -> tuple[CompiledSourceSpec, ...]:
    specs: list[CompiledSourceSpec] = []

    if src.pol == "tm":
        if src._h_indices is not None and src._my_profile is not None:
            comp = src._h_component
            idx = src._h_indices
            target = np.asarray(getattr(fields, comp)[idx])
            mu = np.asarray(fields.permeability[idx])
            coeff = _build_coeff(
                profile=-np.asarray(src._my_profile),
                target=target,
                dt=dt,
                scale_denom=MU_0 * mu * resolution,
            )
            specs.append(
                _as_slab_spec(
                    component=comp,
                    timing="h",
                    index=idx,
                    coeff=coeff,
                    waveform=h_waveform,
                    target_shape=tuple(getattr(fields, comp).shape),
                )
            )

        if src._ez_indices is not None and src._jz_profile is not None:
            idx = src._ez_indices
            target = np.asarray(fields.Ez[idx])
            eps = np.asarray(fields.permittivity[idx])
            coeff = _build_coeff(
                profile=np.asarray(src._jz_profile),
                target=target,
                dt=dt,
                scale_denom=EPS_0 * eps * resolution,
            )
            specs.append(
                _as_slab_spec(
                    component="Ez",
                    timing="e",
                    index=idx,
                    coeff=coeff,
                    waveform=e_waveform,
                    target_shape=tuple(fields.Ez.shape),
                )
            )
    else:
        if src._hz_indices is not None and src._mz_profile is not None:
            idx = src._hz_indices
            target = np.asarray(fields.Hz[idx])
            mu = np.asarray(fields.permeability[idx])
            coeff = _build_coeff(
                profile=np.asarray(src._mz_profile),
                target=target,
                dt=dt,
                scale_denom=MU_0 * mu * resolution,
            )
            specs.append(
                _as_slab_spec(
                    component="Hz",
                    timing="h",
                    index=idx,
                    coeff=coeff,
                    waveform=h_waveform,
                    target_shape=tuple(fields.Hz.shape),
                )
            )

        if src._e_indices is not None:
            comp = src._e_component
            prof = src._jx_profile if comp == "Ex" else src._jy_profile
            if prof is not None:
                idx = src._e_indices
                target = np.asarray(getattr(fields, comp)[idx])
                eps = np.asarray(fields.permittivity[idx])
                coeff = _build_coeff(
                    profile=-np.asarray(prof),
                    target=target,
                    dt=dt,
                    scale_denom=EPS_0 * eps * resolution,
                )
                specs.append(
                    _as_slab_spec(
                        component=comp,
                        timing="e",
                        index=idx,
                        coeff=coeff,
                        waveform=e_waveform,
                        target_shape=tuple(getattr(fields, comp).shape),
                    )
                )

    return tuple(specs)


def _compile_mode_source_3d(
    src: ModeSource,
    fields,
    dt: float,
    resolution: float,
    h_waveform: jnp.ndarray,
    e_waveform: jnp.ndarray,
) -> tuple[CompiledSourceSpec, ...]:
    specs: list[CompiledSourceSpec] = []
    profiles, indices = _mode_3d_profiles_and_indices(src)

    e_terms, h_terms = _get_3d_huygens_terms(src._axis, src.pol)

    for h_comp, e_source, sign in h_terms:
        idx = indices[h_comp]
        prof = profiles[e_source]
        if idx is None or prof is None:
            continue
        target = np.asarray(getattr(fields, h_comp)[idx])
        mu = np.asarray(fields.permeability[idx])
        coeff = _build_coeff(
            profile=sign * np.asarray(prof),
            target=target,
            dt=dt,
            scale_denom=MU_0 * mu * resolution,
        )
        specs.append(
            _as_slab_spec(
                component=h_comp,
                timing="h",
                index=idx,
                coeff=coeff,
                waveform=h_waveform,
                target_shape=tuple(getattr(fields, h_comp).shape),
            )
        )

    for e_comp, h_source, sign in e_terms:
        idx = indices[e_comp]
        prof = profiles[h_source]
        if idx is None or prof is None:
            continue
        target = np.asarray(getattr(fields, e_comp)[idx])
        eps = np.asarray(fields.permittivity[idx])
        coeff = _build_coeff(
            profile=sign * np.asarray(prof),
            target=target,
            dt=dt,
            scale_denom=EPS_0 * eps * resolution,
        )
        specs.append(
            _as_slab_spec(
                component=e_comp,
                timing="e",
                index=idx,
                coeff=coeff,
                waveform=e_waveform,
                target_shape=tuple(getattr(fields, e_comp).shape),
            )
        )

    return tuple(specs)
