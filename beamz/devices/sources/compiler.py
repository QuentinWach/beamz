"""Compile high-level source objects into static packed source specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, MU_0
from beamz.devices.sources.gaussian import GaussianSource
from beamz.devices.sources.mode import ModeSource, _get_3d_huygens_terms


@dataclass(frozen=True)
class CompiledSourceSpec:
    """Single packed source term consumed by compiled step kernels."""

    component: str
    timing: str  # "pre_e", "e", "h"
    index: tuple[Any, ...]
    coeff: jnp.ndarray
    waveform: jnp.ndarray


def _sample_waveform(get_signal_value, t0: float, dt: float, num_steps: int, offset_fn):
    vals = np.zeros((num_steps,), dtype=np.float32)
    for i in range(num_steps):
        t = t0 + i * dt
        vals[i] = float(get_signal_value(offset_fn(t, dt), dt))
    return jnp.asarray(vals)


def _match_shape(profile: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    profile = np.asarray(profile)
    if profile.shape == target_shape:
        return profile
    if profile.ndim != len(target_shape):
        return np.zeros(target_shape, dtype=profile.dtype)

    slices = tuple(slice(0, min(profile.shape[i], target_shape[i])) for i in range(profile.ndim))
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
    )

    return (
        CompiledSourceSpec(
            component="Ez",
            timing="pre_e",
            index=idx,
            coeff=jnp.asarray(coeff, dtype=jnp.float32),
            waveform=waveform,
        ),
    )


def _compile_mode_source(
    device: ModeSource,
    fields,
    dt: float,
    num_steps: int,
    t0: float,
    resolution: float,
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
    )

    dt_physical = float(getattr(device, "_dt_physical", 0.0))
    if is_3d:
        e_waveform = _sample_waveform(
            device._get_signal_value,
            t0=t0,
            dt=dt,
            num_steps=num_steps,
            offset_fn=lambda t, dt_: t + dt_ + dt_physical,
        )
    else:
        e_waveform = _sample_waveform(
            device._get_signal_value,
            t0=t0,
            dt=dt,
            num_steps=num_steps,
            offset_fn=lambda t, dt_: t + 0.5 * dt_ + dt_physical,
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
                CompiledSourceSpec(
                    component=comp,
                    timing="h",
                    index=idx,
                    coeff=coeff,
                    waveform=h_waveform,
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
                CompiledSourceSpec(
                    component="Ez",
                    timing="e",
                    index=idx,
                    coeff=coeff,
                    waveform=e_waveform,
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
                CompiledSourceSpec(
                    component="Hz",
                    timing="h",
                    index=idx,
                    coeff=coeff,
                    waveform=h_waveform,
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
                    CompiledSourceSpec(
                        component=comp,
                        timing="e",
                        index=idx,
                        coeff=coeff,
                        waveform=e_waveform,
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
            CompiledSourceSpec(
                component=h_comp,
                timing="h",
                index=idx,
                coeff=coeff,
                waveform=h_waveform,
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
            CompiledSourceSpec(
                component=e_comp,
                timing="e",
                index=idx,
                coeff=coeff,
                waveform=e_waveform,
            )
        )

    return tuple(specs)
