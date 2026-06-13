"""Compile high-level source objects into static packed source specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.devices.sources._materials import (
    component_permeability_at,
    component_permittivity_at,
)
from beamz.devices.sources.gaussian import GaussianSource
from beamz.devices.sources.mode import ModeSource


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


def source_supports_compiled_specs(device) -> bool:
    """Return True when a source can emit packed compiled-step source specs."""
    return isinstance(device, (GaussianSource, ModeSource)) or hasattr(
        device, "compile_source_specs"
    )


def apply_compiled_source_specs(
    field: jnp.ndarray,
    abs_step: int,
    specs: tuple[CompiledSourceSpec, ...],
) -> jnp.ndarray:
    """Apply packed source specs to one field component for a single step."""
    out = field
    step_idx = jnp.asarray(abs_step, dtype=jnp.int32)
    for spec in specs:
        if spec.waveform.size == 0:
            continue
        safe_idx = jnp.clip(step_idx, 0, spec.waveform.shape[0] - 1)
        patch = spec.coeff * spec.waveform[safe_idx]
        if (
            spec.is_slab
            and spec.slab_starts is not None
            and spec.slab_sizes is not None
        ):
            cur = jax.lax.dynamic_slice(out, spec.slab_starts, spec.slab_sizes)
            out = jax.lax.dynamic_update_slice(out, cur + patch, spec.slab_starts)
            continue
        target = out[spec.index]
        if patch.shape != target.shape:
            if patch.size == target.size:
                patch = jnp.reshape(patch, target.shape)
            else:
                patch = jnp.broadcast_to(patch, target.shape)
        out = out.at[spec.index].add(patch)
    return out


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


def _analytic_waveform_samples(
    device: ModeSource,
    *,
    t0: float,
    dt: float,
    num_steps: int,
    total_steps: int | None = None,
) -> np.ndarray:
    n = int(total_steps if total_steps is not None else num_steps)
    start = float(t0)
    vals = np.zeros((n,), dtype=np.complex128)
    for i in range(n):
        t = start + i * float(dt)
        vals[i] = complex(
            device._get_signal_value(t, dt),
            device._get_signal_quadrature_value(t, dt),
        )
    return vals


def _partition_weights_by_frequency(
    fft_frequencies: np.ndarray,
    profile_frequencies: np.ndarray,
) -> np.ndarray:
    """Return a smooth frequency partition for broadband modal source profiles."""
    nodes = np.sort(np.unique(np.asarray(profile_frequencies, dtype=float).reshape(-1)))
    if nodes.size == 0:
        raise ValueError("profile_frequencies must contain at least one frequency.")
    if np.any(nodes <= 0.0):
        raise ValueError("profile_frequencies must be strictly positive.")
    abs_freq = np.abs(np.asarray(fft_frequencies, dtype=float).reshape(-1))
    weights = np.zeros((nodes.size, abs_freq.size), dtype=np.float64)
    if nodes.size == 1:
        weights[0, :] = 1.0
        return weights

    for idx, freq in enumerate(nodes):
        if idx == 0:
            right = nodes[idx + 1]
            mask = abs_freq <= right
            weights[idx, mask] = np.where(
                abs_freq[mask] <= freq,
                1.0,
                (right - abs_freq[mask]) / max(right - freq, 1e-30),
            )
            continue
        if idx == nodes.size - 1:
            left = nodes[idx - 1]
            mask = abs_freq >= left
            weights[idx, mask] = np.where(
                abs_freq[mask] >= freq,
                1.0,
                (abs_freq[mask] - left) / max(freq - left, 1e-30),
            )
            continue
        left = nodes[idx - 1]
        right = nodes[idx + 1]
        left_mask = (abs_freq >= left) & (abs_freq <= freq)
        right_mask = (abs_freq >= freq) & (abs_freq <= right)
        weights[idx, left_mask] = (abs_freq[left_mask] - left) / max(freq - left, 1e-30)
        weights[idx, right_mask] = (right - abs_freq[right_mask]) / max(
            right - freq, 1e-30
        )

    total = np.sum(weights, axis=0)
    empty = total <= 1e-30
    if np.any(empty):
        nearest = np.argmin(np.abs(abs_freq[empty, None] - nodes[None, :]), axis=1)
        weights[:, empty] = 0.0
        weights[nearest, np.where(empty)[0]] = 1.0
        total = np.sum(weights, axis=0)
    return weights / np.maximum(total, 1e-30)


def _analytic_subband_waveforms(
    analytic_waveform: np.ndarray,
    *,
    dt: float,
    profile_frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an analytic source waveform into profile-frequency subbands."""
    waveform = np.asarray(analytic_waveform, dtype=np.complex128).reshape(-1)
    if waveform.size == 0:
        return (
            np.asarray(profile_frequencies, dtype=float).reshape(-1),
            np.zeros((0, 0), dtype=np.complex128),
        )
    nodes = np.sort(np.unique(np.asarray(profile_frequencies, dtype=float).reshape(-1)))
    spectrum = np.fft.fft(waveform)
    fft_freqs = np.fft.fftfreq(waveform.size, d=float(dt))
    weights = _partition_weights_by_frequency(fft_freqs, nodes)
    subbands = np.fft.ifft(weights * spectrum[None, :], axis=1)
    return nodes, np.asarray(subbands, dtype=np.complex128)


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


def compile_source_specs(
    sources: list,
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

    for device in sources:
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
        elif hasattr(device, "compile_source_specs"):
            specs.extend(
                tuple(
                    device.compile_source_specs(
                        fields=fields,
                        dt=dt,
                        num_steps=num_steps,
                        t0=t0,
                        resolution=resolution,
                        total_steps=total_steps,
                    )
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
        device._init_spatial_profile(
            fields.Ez.shape,
            resolution,
            is_3d,
            plane_2d=getattr(fields, "plane_2d", None),
        )

    idx = device._grid_indices
    eps_region = np.asarray(component_permittivity_at(fields, "Ez", idx))
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
    needs_initialize = (
        (not getattr(device, "_initialized", False))
        or (getattr(device, "_grid_shape", None) != fields.permittivity.shape)
        or (getattr(device, "_resolution", None) is None)
        or (not np.isclose(getattr(device, "_resolution", 0.0), resolution))
    )
    if not needs_initialize and bool(getattr(device, "_is_3d", False)):
        launch_dt = getattr(device, "_launch_dt", None)
        needs_initialize = (
            launch_dt is None
            or (not np.isclose(float(launch_dt), float(dt)))
            or (getattr(device, "_k_num_axis", None) is None)
            or (getattr(device, "_omega_launch", None) is None)
        )
    if needs_initialize:
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
        profile_frequencies = getattr(device, "profile_frequencies", None)
        if profile_frequencies is None and getattr(device, "num_freqs", None):
            source_time = getattr(device, "source_time", None)
            freq0 = float(
                getattr(source_time, "freq0", LIGHT_SPEED / float(device.wavelength))
            )
            fwidth = float(getattr(source_time, "fwidth", freq0 / 10.0))
            count = int(getattr(device, "num_freqs"))
            if count > 1:
                k = np.arange(count, dtype=float)
                profile_frequencies = np.sort(
                    freq0
                    + 1.5 * fwidth * np.cos((2.0 * k + 1.0) * np.pi / (2.0 * count))
                )
        if profile_frequencies is not None:
            profile_frequencies = np.asarray(profile_frequencies, dtype=float).reshape(
                -1
            )
            if profile_frequencies.size > 1:
                return _compile_mode_source_3d_multifrequency(
                    device,
                    fields,
                    dt,
                    resolution,
                    num_steps=num_steps,
                    t0=t0,
                    total_steps=total_steps,
                    profile_frequencies=profile_frequencies,
                )
        phasor_waveform = _sample_waveform(
            device._get_signal_value,
            t0=t0,
            dt=dt,
            num_steps=num_steps,
            offset_fn=lambda t, dt_: t,
            total_steps=total_steps,
        )
        phasor_quadrature_waveform = _sample_waveform(
            device._get_signal_quadrature_value,
            t0=t0,
            dt=dt,
            num_steps=num_steps,
            offset_fn=lambda t, dt_: t,
            total_steps=total_steps,
        )
        return _compile_mode_source_3d(
            device,
            fields,
            dt,
            resolution,
            phasor_waveform,
            phasor_waveform,
            phasor_quadrature_waveform,
            phasor_quadrature_waveform,
            t0=t0,
        )
    return _compile_mode_source_2d(
        device,
        fields,
        dt,
        resolution,
        h_waveform,
        e_waveform,
    )


def _compile_mode_source_3d_multifrequency(
    src: ModeSource,
    fields,
    dt: float,
    resolution: float,
    *,
    num_steps: int,
    t0: float,
    total_steps: int | None,
    profile_frequencies: np.ndarray,
) -> tuple[CompiledSourceSpec, ...]:
    """Compile a broadband 3D ModeSource as a profile-frequency filter bank."""
    analytic = _analytic_waveform_samples(
        src,
        t0=t0,
        dt=dt,
        num_steps=num_steps,
        total_steps=total_steps,
    )
    nodes, subbands = _analytic_subband_waveforms(
        analytic,
        dt=dt,
        profile_frequencies=profile_frequencies,
    )
    specs: list[CompiledSourceSpec] = []
    common_kwargs = dict(
        grid=src.grid,
        center=src.center,
        width=src.width,
        pol=src.pol,
        signal=src.signal,
        direction=src.direction,
        height=src.height,
        signal_quadrature=getattr(src, "signal_quadrature", None),
        source_time=getattr(src, "source_time", None),
        num_freqs=None,
        power=getattr(src, "power", 1.0),
        mode_index=getattr(src, "mode_index", 0),
        mode_target_neff=getattr(src, "mode_target_neff", None),
        mode_num_modes=getattr(src, "mode_num_modes", None),
        mode_eps_profile_full=getattr(src, "mode_eps_profile_full", None),
        mode_crop_slices=getattr(src, "mode_crop_slices", None),
    )
    for freq, waveform in zip(nodes, subbands, strict=True):
        profile_src = ModeSource(
            wavelength=LIGHT_SPEED / float(freq),
            profile_frequencies=None,
            **common_kwargs,
        )
        profile_src.initialize(fields.permittivity, resolution, dt=dt)
        real_waveform = jnp.asarray(np.real(waveform), dtype=jnp.float32)
        quadrature_waveform = jnp.asarray(np.imag(waveform), dtype=jnp.float32)
        specs.extend(
            _compile_mode_source_3d(
                profile_src,
                fields,
                dt,
                resolution,
                real_waveform,
                real_waveform,
                quadrature_waveform,
                quadrature_waveform,
                t0=t0,
            )
        )
    return tuple(specs)


def _build_coeff(
    profile: np.ndarray,
    target: np.ndarray,
    dt: float,
    scale_denom: np.ndarray,
) -> jnp.ndarray:
    profile = _match_shape(np.asarray(profile), target.shape)
    coeff = profile * dt / scale_denom
    return jnp.asarray(coeff, dtype=jnp.float32)


def _append_phasor_source_specs(
    specs: list[CompiledSourceSpec],
    *,
    component: str,
    timing: str,
    index: tuple[Any, ...],
    profile: np.ndarray,
    target: np.ndarray,
    dt: float,
    scale_denom: np.ndarray,
    waveform: jnp.ndarray,
    quadrature_waveform: jnp.ndarray | None,
    target_shape: tuple[int, ...],
    imag_tol: float = 1e-30,
) -> None:
    """Append real compiled specs for Re(profile * analytic_waveform)."""
    profile_c = np.asarray(profile, dtype=np.complex128)
    real_coeff = _build_coeff(
        profile=np.real(profile_c),
        target=target,
        dt=dt,
        scale_denom=scale_denom,
    )
    specs.append(
        _as_slab_spec(
            component=component,
            timing=timing,
            index=index,
            coeff=real_coeff,
            waveform=waveform,
            target_shape=target_shape,
        )
    )

    if quadrature_waveform is None:
        return
    imag_peak = float(np.max(np.abs(np.imag(profile_c)))) if profile_c.size else 0.0
    if imag_peak <= float(imag_tol):
        return

    imag_coeff = _build_coeff(
        profile=-np.imag(profile_c),
        target=target,
        dt=dt,
        scale_denom=scale_denom,
    )
    specs.append(
        _as_slab_spec(
            component=component,
            timing=timing,
            index=index,
            coeff=imag_coeff,
            waveform=quadrature_waveform,
            target_shape=target_shape,
        )
    )


def _nonzero_bbox(arr: np.ndarray, *, atol: float = 0.0):
    values = np.asarray(arr)
    if values.size == 0:
        return None
    mask = np.abs(values) > float(atol)
    if not np.any(mask):
        return None
    coords = np.argwhere(mask)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


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
            mu = np.asarray(component_permeability_at(fields, comp, idx))
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
            eps = np.asarray(component_permittivity_at(fields, "Ez", idx))
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
            mu = np.asarray(component_permeability_at(fields, "Hz", idx))
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
                eps = np.asarray(component_permittivity_at(fields, comp, idx))
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
    h_quadrature_waveform: jnp.ndarray | None = None,
    e_quadrature_waveform: jnp.ndarray | None = None,
    t0: float = 0.0,
) -> tuple[CompiledSourceSpec, ...]:
    specs: list[CompiledSourceSpec] = []
    del resolution, e_waveform, e_quadrature_waveform, t0

    for residual in src._compute_discrete_3d_phasor_residuals(fields, dt=float(dt)):
        component = residual.component
        index = residual.index
        target = np.asarray(getattr(fields, component)[index])
        _append_phasor_source_specs(
            specs,
            component=component,
            timing=residual.timing,
            index=index,
            profile=np.asarray(residual.residual, dtype=np.complex128),
            target=target,
            dt=1.0,
            scale_denom=np.asarray(1.0, dtype=np.float64),
            waveform=h_waveform,
            quadrature_waveform=h_quadrature_waveform,
            target_shape=tuple(getattr(fields, component).shape),
        )

    return tuple(specs)
