"""Compile high-level source objects into static packed source specs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from beamz.arrays import to_host
from beamz.const import EPS_0, MU_0
from beamz.devices.sources.inject import _get_3d_huygens_terms
from beamz.devices.sources.setup import (
    initialize_gaussian_state,
    initialize_mode_state,
    sample_signal,
)
from beamz.devices.sources.spec import (
    GaussianSourceSpec,
    ModeSourceSpec,
    source_to_spec,
)
from beamz.devices.sources.state import (
    GaussianSourceState,
    ModeSourceState,
    source_state_for,
)


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
            coeff_np = to_host(coeff, dtype=np.float32)
            slab_sizes = tuple(sizes)
            expected = int(math.prod(slab_sizes))
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
    spec,
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
        vals[i] = float(sample_signal(spec, offset_fn(t, dt), dt))
    return jnp.asarray(vals)


def _match_shape(profile: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    profile = to_host(profile)
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


def _mode_3d_profiles_and_indices(state: ModeSourceState):
    profiles = {
        "Ex": state.Ex_profile,
        "Ey": state.Ey_profile,
        "Ez": state.Ez_profile,
        "Hx": state.Hx_profile,
        "Hy": state.Hy_profile,
        "Hz": state.Hz_profile,
    }
    indices = {
        "Ex": state.Ex_indices,
        "Ey": state.Ey_indices,
        "Ez": state.Ez_indices,
        "Hx": state.Hx_indices,
        "Hy": state.Hy_indices,
        "Hz": state.Hz_indices,
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
    source_states: tuple[object | None, ...] | list[object | None] | None = None,
) -> tuple[CompiledSourceSpec, ...]:
    """Compile source devices into packed source specs.

    v0.3 first-class support: GaussianSourceSpec and ModeSourceSpec.
    """
    specs: list[CompiledSourceSpec] = []

    devices = tuple(devices)
    if source_states is None:
        source_states = (None,) * len(devices)
    else:
        source_states = tuple(source_states)
        if len(source_states) != len(devices):
            raise ValueError("source_states must match devices length when provided")

    for device, state_override in zip(devices, source_states):
        try:
            spec = source_to_spec(device)
        except TypeError:
            continue
        state = source_state_for(spec, source=device, state=state_override)
        if isinstance(spec, GaussianSourceSpec):
            specs.extend(
                _compile_gaussian_source(
                    spec=spec,
                    state=state,
                    fields=fields,
                    dt=dt,
                    num_steps=num_steps,
                    t0=t0,
                    resolution=resolution,
                    total_steps=total_steps,
                )
            )
        elif isinstance(spec, ModeSourceSpec):
            specs.extend(
                _compile_mode_source(
                    spec=spec,
                    state=state,
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
    spec: GaussianSourceSpec,
    state: GaussianSourceState,
    fields,
    dt: float,
    num_steps: int,
    t0: float,
    resolution: float,
    total_steps: int | None = None,
) -> tuple[CompiledSourceSpec, ...]:
    # Initialize spatial profile once.
    if state.spatial_profile_ez is None:
        initialize_gaussian_state(spec, state, fields.Ez.shape, resolution)

    idx = state.grid_indices
    eps_region = to_host(fields.permittivity[idx])
    profile = to_host(state.spatial_profile_ez)

    coeff = -profile * dt / (EPS_0 * eps_region)
    waveform = _sample_waveform(
        spec,
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
    spec: ModeSourceSpec,
    state: ModeSourceState,
    fields,
    dt: float,
    num_steps: int,
    t0: float,
    resolution: float,
    total_steps: int | None = None,
) -> tuple[CompiledSourceSpec, ...]:
    if (
        (not state.initialized)
        or (state.grid_shape != fields.permittivity.shape)
        or (state.resolution is None)
        or (
            not math.isclose(
                float(state.resolution),
                float(resolution),
                rel_tol=1e-5,
                abs_tol=1e-8,
            )
        )
    ):
        initialize_mode_state(spec, state, fields.permittivity, resolution, dt=dt)

    is_3d = bool(state.is_3d)

    h_waveform = _sample_waveform(
        spec,
        t0=t0,
        dt=dt,
        num_steps=num_steps,
        offset_fn=lambda t, dt_: t + 0.5 * dt_,
        total_steps=total_steps,
    )

    dt_physical = float(state.dt_physical)
    # E injection is applied after the E update within each Yee step; use the
    # same half-step base time as H plus physical plane delay to keep the 3D
    # Huygens pair phase-consistent with the 2D implementation.
    e_waveform = _sample_waveform(
        spec,
        t0=t0,
        dt=dt,
        num_steps=num_steps,
        offset_fn=lambda t, dt_: t + 0.5 * dt_ + dt_physical,
        total_steps=total_steps,
    )

    if is_3d:
        return _compile_mode_source_3d(
            spec,
            state,
            fields,
            dt,
            resolution,
            h_waveform,
            e_waveform,
        )
    return _compile_mode_source_2d(
        spec,
        state,
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
    profile = _match_shape(profile, target.shape)
    coeff = profile * dt / scale_denom
    return jnp.asarray(coeff, dtype=jnp.float32)


def _compile_mode_source_2d(
    spec: ModeSourceSpec,
    state: ModeSourceState,
    fields,
    dt: float,
    resolution: float,
    h_waveform: jnp.ndarray,
    e_waveform: jnp.ndarray,
) -> tuple[CompiledSourceSpec, ...]:
    pol = spec.pol
    specs: list[CompiledSourceSpec] = []

    if pol == "tm":
        if state.h_indices is not None and state.my_profile is not None:
            comp = state.h_component
            idx = state.h_indices
            target = to_host(getattr(fields, comp)[idx])
            mu = to_host(fields.permeability[idx])
            coeff = _build_coeff(
                profile=-to_host(state.my_profile),
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

        if state.ez_indices is not None and state.jz_profile is not None:
            idx = state.ez_indices
            target = to_host(fields.Ez[idx])
            eps = to_host(fields.permittivity[idx])
            coeff = _build_coeff(
                profile=to_host(state.jz_profile),
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
        if state.hz_indices is not None and state.mz_profile is not None:
            idx = state.hz_indices
            target = to_host(fields.Hz[idx])
            mu = to_host(fields.permeability[idx])
            coeff = _build_coeff(
                profile=to_host(state.mz_profile),
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

        if state.e_indices is not None:
            comp = state.e_component
            prof = state.jx_profile if comp == "Ex" else state.jy_profile
            if prof is not None:
                idx = state.e_indices
                target = to_host(getattr(fields, comp)[idx])
                eps = to_host(fields.permittivity[idx])
                coeff = _build_coeff(
                    profile=-to_host(prof),
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
    spec: ModeSourceSpec,
    state: ModeSourceState,
    fields,
    dt: float,
    resolution: float,
    h_waveform: jnp.ndarray,
    e_waveform: jnp.ndarray,
) -> tuple[CompiledSourceSpec, ...]:
    specs: list[CompiledSourceSpec] = []
    profiles, indices = _mode_3d_profiles_and_indices(state)
    pol = spec.pol

    e_terms, h_terms = _get_3d_huygens_terms(state.axis, pol)

    for h_comp, e_source, sign in h_terms:
        idx = indices[h_comp]
        prof = profiles[e_source]
        if idx is None or prof is None:
            continue
        target = to_host(getattr(fields, h_comp)[idx])
        mu = to_host(fields.permeability[idx])
        coeff = _build_coeff(
            profile=sign * to_host(prof),
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
        target = to_host(getattr(fields, e_comp)[idx])
        eps = to_host(fields.permittivity[idx])
        coeff = _build_coeff(
            profile=sign * to_host(prof),
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
