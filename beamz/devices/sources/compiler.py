"""Compile source specs into static packed source update specs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import singledispatch
from typing import Any, Literal, cast

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.lattice import (
    canonical_component_2d,
    component_axis_offsets_3d,
    component_material_at,
    public_component_2d,
)

from .mode_launch import (
    Mode2DLaunchPlan,
    Mode3DLaunchPlan,
    ModeLaunchPlan,
    plan_mode_source_launch,
)
from .planar_tfsf import (
    compute_discrete_3d_e_phasor_residuals,
    compute_discrete_3d_h_phasor_residuals,
)
from .specs import (
    CustomSource,
    FieldAxis3D,
    FieldIndex3D,
    FieldProfile3D,
    GaussianBeamSource,
    GaussianSource,
    ModeSource,
)
from .time import (
    analytic_subband_waveforms,
    interpolate_time_signal,
    sample_source_waveforms,
    sample_waveform,
)

_analytic_subband_waveforms = analytic_subband_waveforms
_sample_waveform = sample_waveform


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
    slab: list[tuple[CompiledSourceSpec, tuple[int, ...], tuple[int, ...]]] = []
    for spec in specs:
        if (
            spec.is_slab
            and spec.slab_starts is not None
            and spec.slab_sizes is not None
        ):
            slab.append((spec, spec.slab_starts, spec.slab_sizes))
    rest = tuple(
        s
        for s in specs
        if not (s.is_slab and s.slab_starts is not None and s.slab_sizes is not None)
    )
    if not slab:
        return None, specs
    ndim = len(slab[0][2])
    max_sizes = tuple(max(sizes[d] for _s, _starts, sizes in slab) for d in range(ndim))
    padded = []
    for spec, _starts, sizes in slab:
        pad_width = tuple((0, max_sizes[d] - sizes[d]) for d in range(ndim))
        padded.append(jnp.pad(spec.coeff, pad_width))
    return (
        BatchedSlabGroup(
            waveforms=jnp.stack([spec.waveform for spec, _starts, _sizes in slab]),
            coeffs=jnp.stack(padded),
            starts=jnp.array(
                [list(starts) for _s, starts, _sizes in slab], dtype=jnp.int32
            ),
            starts_tuple=tuple(
                tuple(int(v) for v in starts) for _s, starts, _sizes in slab
            ),
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
    source_index: int = -1
    launched_power: float | None = None


@dataclass(frozen=True, slots=True)
class SpatialFieldProfile:
    """One component of a lowered source, in field-update units."""

    component: str
    timing: str
    values: np.ndarray


@dataclass(frozen=True, slots=True)
class TemporalWaveform:
    """Sampled real drive and its optional analytic quadrature."""

    values: jnp.ndarray
    quadrature: jnp.ndarray | None = None


@dataclass(frozen=True, slots=True)
class SourceSupport:
    """Target cells for a spatial field profile."""

    index: tuple[Any, ...]
    target_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class InjectionPlanEntry:
    """Normalized profile, waveform, and support for one field component."""

    profile: SpatialFieldProfile
    waveform: TemporalWaveform
    support: SourceSupport
    launched_power: float | None = None


@dataclass(frozen=True, slots=True)
class CompiledInjectionPlan:
    """Source-independent injection plan produced by every source lowerer."""

    entries: tuple[InjectionPlanEntry, ...]


@dataclass(frozen=True, slots=True)
class SourceLoweringContext:
    fields: Any
    resolution: float
    dt: float
    t0: float
    num_steps: int
    total_steps: int
    domain: Any | None = None
    grid: Any | None = None


@singledispatch
def lower_source(
    source: object,
    ctx: SourceLoweringContext,
) -> CompiledInjectionPlan:
    del ctx
    raise TypeError(
        f"Unsupported source object {type(source).__name__!s}. "
        "Pass a built-in source or CustomSource."
    )


@lower_source.register
def _lower_custom_source(
    source: CustomSource,
    ctx: SourceLoweringContext,
) -> CompiledInjectionPlan:
    component = source.component
    coeff = source.coeff
    target_shape = source.target_shape
    if ctx.domain is not None and not bool(ctx.domain.is_3d):
        polarization = ctx.domain.polarization_2d
        canonical = canonical_component_2d(component, ctx.domain.plane_2d, polarization)
        if canonical is None:
            raise ValueError(
                f"2D plane {ctx.domain.plane_2d!r} does not support {component!r}; "
                f"only {polarization.upper()}z components are active."
            )
        _, sign = public_component_2d(canonical, ctx.domain.plane_2d, polarization)
        component = canonical
        coeff = sign * np.asarray(coeff)
        target_shape = tuple(getattr(ctx.fields, canonical).shape)
    return CompiledInjectionPlan(
        (
            _injection_entry(
                component=component,
                timing=source.timing,
                index=source.index,
                values=coeff,
                waveform=TemporalWaveform(
                    jnp.asarray(source.waveform, dtype=jnp.float32)
                ),
                target_shape=target_shape,
            ),
        )
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


def _injection_entry(
    *,
    component: str,
    timing: str,
    index: tuple[Any, ...],
    values,
    waveform: TemporalWaveform,
    target_shape: tuple[int, ...],
    launched_power: float | None = None,
) -> InjectionPlanEntry:
    return InjectionPlanEntry(
        profile=SpatialFieldProfile(
            component=component,
            timing=timing,
            values=np.asarray(values),
        ),
        waveform=waveform,
        support=SourceSupport(index=index, target_shape=target_shape),
        launched_power=launched_power,
    )


def _compile_injection_plan(
    plan: CompiledInjectionPlan,
    *,
    source_index: int = -1,
    imag_tol: float = 1e-30,
) -> tuple[CompiledSourceSpec, ...]:
    """Emit runtime specs through the single source scheduling path."""
    specs: list[CompiledSourceSpec] = []
    for entry in plan.entries:
        profile = entry.profile
        support = entry.support
        values = np.asarray(profile.values, dtype=np.complex128)
        parts = [(np.real(values), entry.waveform.values)]
        imag_peak = float(np.max(np.abs(np.imag(values)))) if values.size else 0.0
        if entry.waveform.quadrature is not None and imag_peak > float(imag_tol):
            parts.append((-np.imag(values), entry.waveform.quadrature))

        for coeff, waveform in parts:
            spec = _as_slab_spec(
                component=profile.component,
                timing=profile.timing,
                index=support.index,
                coeff=coeff,
                waveform=waveform,
                target_shape=support.target_shape,
            )
            specs.append(
                replace(
                    spec,
                    source_index=source_index,
                    launched_power=entry.launched_power,
                )
            )
    return tuple(specs)


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
    source_specs: tuple[object, ...],
    fields,
    dt: float,
    resolution: float,
    num_steps: int,
    t0: float,
    total_steps: int,
    domain=None,
    grid=None,
) -> tuple[CompiledSourceSpec, ...]:
    """Compile immutable source request objects into packed update specs."""
    ctx = SourceLoweringContext(
        fields=fields,
        resolution=resolution,
        dt=dt,
        t0=t0,
        num_steps=num_steps,
        total_steps=total_steps,
        domain=domain,
        grid=grid,
    )
    specs: list[CompiledSourceSpec] = []
    for source_index, source in enumerate(source_specs):
        specs.extend(
            _compile_injection_plan(
                lower_source(source, ctx), source_index=source_index
            )
        )

    return tuple(specs)


@lower_source.register
def _lower_gaussian_beam_source(
    source: GaussianBeamSource,
    ctx: SourceLoweringContext,
) -> CompiledInjectionPlan:
    if _source_requires_rectilinear_operator(ctx):
        raise NotImplementedError(
            "GaussianBeamSource on a rectilinear grid requires a metric-aware "
            "phasor-residual operator; use GaussianSource or a uniform grid."
        )
    waveform = _sample_temporal_waveform(source.source_time, ctx)
    return _field_profile_injection_plan(
        gaussian_beam_field_profile(source, ctx.fields, resolution=ctx.resolution),
        waveform,
        ctx,
        max_shift=source.max_shift,
    )


@lower_source.register
def _lower_gaussian_source(
    source: GaussianSource,
    ctx: SourceLoweringContext,
) -> CompiledInjectionPlan:
    is_3d = np.asarray(ctx.fields.permittivity).ndim == 3
    is_te = not is_3d and ctx.domain is not None and ctx.domain.polarization_2d == "te"
    component = "Hz" if is_te else "Ez"
    target = getattr(ctx.fields, component)
    index, profile = gaussian_spatial_profile(
        source,
        ez_shape=target.shape,
        resolution=ctx.resolution,
        is_3d=is_3d,
        plane_2d=getattr(ctx.fields, "plane_2d", None),
        component=component,
        grid=ctx.grid,
    )
    material_region = np.asarray(component_material_at(ctx.fields, component, index))
    constant = MU_0 if is_te else EPS_0
    values = -np.asarray(profile) * ctx.dt / (constant * material_region)
    waveform = TemporalWaveform(
        sample_waveform(
            lambda time, step: _signal_value(source.signal, time, step),
            t0=ctx.t0,
            dt=ctx.dt,
            num_steps=ctx.num_steps,
            offset_fn=lambda t, dt: t + 0.5 * dt,
            total_steps=ctx.total_steps,
        )
    )
    return CompiledInjectionPlan(
        (
            _injection_entry(
                component=component,
                timing="h" if is_te else "pre_e",
                index=index,
                values=values,
                waveform=waveform,
                target_shape=tuple(target.shape),
            ),
        )
    )


@lower_source.register
def _lower_mode_source(
    source: ModeSource,
    ctx: SourceLoweringContext,
) -> CompiledInjectionPlan:
    if (
        np.asarray(ctx.fields.permittivity).ndim == 3
        and source.profile_frequencies().size > 1
    ):
        return _lower_broadband_mode_source(source, ctx)

    launch_plan = plan_mode_source_launch(
        source,
        ctx.fields,
        resolution=ctx.resolution,
        dt=ctx.dt,
        **({"grid": ctx.grid} if ctx.grid is not None else {}),
    )
    is_3d = isinstance(launch_plan, Mode3DLaunchPlan) or hasattr(
        launch_plan, "residuals"
    )
    waveform = _sample_temporal_waveform(
        source.source_time,
        ctx,
        offset_half_step=not is_3d,
        include_quadrature=is_3d,
    )
    return _mode_launch_injection_plan(launch_plan, waveform, ctx)


def _signal_value(signal, time, dt):
    if signal is None:
        return 0.0
    return interpolate_time_signal(signal, time, dt)


def _sample_temporal_waveform(
    signal,
    ctx: SourceLoweringContext,
    *,
    offset_half_step: bool = False,
    include_quadrature: bool = True,
) -> TemporalWaveform:
    values, quadrature = sample_source_waveforms(
        signal,
        t0=ctx.t0,
        dt=ctx.dt,
        num_steps=ctx.num_steps,
        total_steps=ctx.total_steps,
        offset_fn=(lambda t, dt: t + 0.5 * dt)
        if offset_half_step
        else (lambda t, dt: t),
    )
    return TemporalWaveform(
        values=jnp.asarray(values, dtype=jnp.float32),
        quadrature=(
            jnp.asarray(quadrature, dtype=jnp.float32) if include_quadrature else None
        ),
    )


def _lower_broadband_mode_source(
    src: ModeSource,
    ctx: SourceLoweringContext,
) -> CompiledInjectionPlan:
    """Lower a broadband mode into frequency-partitioned injection entries."""
    sampled = _sample_temporal_waveform(src.source_time, ctx)
    analytic = np.asarray(sampled.values, dtype=np.float64) + 1j * np.asarray(
        sampled.quadrature, dtype=np.float64
    )
    nodes, subbands = analytic_subband_waveforms(
        analytic,
        dt=ctx.dt,
        profile_frequencies=src.profile_frequencies(),
    )
    plans = []
    for freq in nodes:
        profile_source_time = replace(src.source_time, freq0=float(freq))
        profile_src = src.updated_copy(
            source_time=profile_source_time,
            mode_spec=replace(src.mode_spec, num_freqs=1),
        )
        plan = plan_mode_source_launch(
            profile_src,
            ctx.fields,
            resolution=ctx.resolution,
            dt=ctx.dt,
            **({"grid": ctx.grid} if ctx.grid is not None else {}),
        )
        plans.append(plan)

    scales = _broadband_launch_amplitude_scales(nodes, plans)
    plans = [
        replace(plan, launch_amplitude_scale=float(scale))
        if isinstance(plan, Mode3DLaunchPlan)
        else plan
        for plan, scale in zip(plans, scales, strict=True)
    ]

    entries: list[InjectionPlanEntry] = []
    for plan, waveform in zip(plans, subbands, strict=True):
        temporal = TemporalWaveform(
            values=jnp.asarray(np.real(waveform), dtype=jnp.float32),
            quadrature=jnp.asarray(np.imag(waveform), dtype=jnp.float32),
        )
        entries.extend(_mode_launch_injection_plan(plan, temporal, ctx).entries)
    return CompiledInjectionPlan(tuple(entries))


def _broadband_launch_amplitude_scales(nodes, plans) -> np.ndarray:
    """Return the monitor-power correction for every profile-frequency node."""
    node_arr = np.asarray(nodes, dtype=float).reshape(-1)
    fallback = np.asarray(
        [float(getattr(plan, "launch_amplitude_scale", 1.0) or 1.0) for plan in plans],
        dtype=float,
    )
    if node_arr.size == 0 or fallback.size != node_arr.size:
        return fallback

    ratios = np.asarray(
        [getattr(plan, "launch_power_ratio", np.nan) for plan in plans],
        dtype=float,
    )
    valid = np.isfinite(ratios) & (ratios > 1e-24)
    scales = fallback.copy()
    scales[valid] = 1.0 / np.sqrt(ratios[valid])
    return scales


def _mode_launch_injection_plan(
    plan: ModeLaunchPlan,
    waveform: TemporalWaveform,
    ctx: SourceLoweringContext,
) -> CompiledInjectionPlan:
    if isinstance(plan, Mode2DLaunchPlan):
        entries = []
        for entry in plan.entries:
            component = entry.component
            material_scale = MU_0 if component.startswith("H") else EPS_0
            denominator = (
                material_scale
                * np.asarray(component_material_at(ctx.fields, component, entry.index))
                * ctx.resolution
            )
            target = np.asarray(getattr(ctx.fields, component)[entry.index])
            values = (
                _match_shape(np.asarray(entry.profile), target.shape)
                * ctx.dt
                / denominator
            )
            entries.append(
                _injection_entry(
                    component=component,
                    timing=entry.timing,
                    index=entry.index,
                    values=values,
                    waveform=waveform,
                    target_shape=tuple(getattr(ctx.fields, component).shape),
                )
            )
        return CompiledInjectionPlan(tuple(entries))

    if not (isinstance(plan, Mode3DLaunchPlan) or hasattr(plan, "residuals")):
        raise TypeError(f"Unsupported mode launch plan {type(plan).__name__}.")
    launch_scale = float(getattr(plan, "launch_amplitude_scale", 1.0) or 1.0)
    launched_power = getattr(plan, "launched_power", None)
    return CompiledInjectionPlan(
        tuple(
            _injection_entry(
                component=residual.component,
                timing=residual.timing,
                index=residual.index,
                values=np.asarray(residual.residual, dtype=np.complex128)
                * launch_scale,
                waveform=waveform,
                target_shape=tuple(getattr(ctx.fields, residual.component).shape),
                launched_power=(
                    None if launched_power is None else float(launched_power)
                ),
            )
            for residual in plan.residuals
        )
    )


def gaussian_spatial_profile(
    source,
    *,
    ez_shape,
    resolution: float,
    is_3d: bool,
    plane_2d=None,
    component: str = "Ez",
    grid=None,
):
    """Compute a Gaussian source profile from semantic source data."""
    if grid is not None:
        offsets = (
            component_axis_offsets_3d(component)
            if is_3d
            else {
                "Ez": {"y": 0.0, "x": 0.0},
                "Hx": {"y": 0.5, "x": 0.0},
                "Hy": {"y": 0.0, "x": 0.5},
                "Ex": {"y": 0.0, "x": 0.5},
                "Ey": {"y": 0.5, "x": 0.0},
                "Hz": {"y": 0.5, "x": 0.5},
            }[component]
        )
        axes = ("z", "y", "x") if is_3d else ("y", "x")
        positions = tuple(float(value) for value in source.position)
        centers = dict(zip(("x", "y", "z"), positions, strict=False))
        coordinates = {
            axis: np.asarray(
                grid.axis_edges(axis) if offsets[axis] == 0.0 else grid.centers(axis)
            )
            for axis in axes
        }
        slices = []
        selected = []
        for axis in axes:
            values = coordinates[axis]
            center = centers[axis]
            active = np.flatnonzero(np.abs(values - center) <= 4.0 * source.width)
            if active.size:
                start, stop = int(active[0]), int(active[-1] + 1)
            else:
                nearest = int(np.argmin(np.abs(values - center)))
                start, stop = nearest, nearest + 1
            slices.append(slice(start, stop))
            selected.append(jnp.asarray(values[start:stop]))
        mesh = jnp.meshgrid(*selected, indexing="ij")
        distance_sq = sum(
            (values - centers[axis]) ** 2
            for axis, values in zip(axes, mesh, strict=True)
        )
        return tuple(slices), jnp.exp(-distance_sq / (2 * source.width**2))

    sigma_grid = source.width / resolution
    radius_grid = int(np.ceil(4 * sigma_grid))

    if is_3d:
        x0, y0, z0 = source.position
        nz, ny, nx = ez_shape
        cx, cy, cz = (int(round(c / resolution)) for c in (x0, y0, z0))
        x0i, x1i = max(0, cx - radius_grid), min(nx, cx + radius_grid + 1)
        y0i, y1i = max(0, cy - radius_grid), min(ny, cy + radius_grid + 1)
        z0i, z1i = max(0, cz - radius_grid), min(nz, cz + radius_grid + 1)
        idx = (slice(z0i, z1i), slice(y0i, y1i), slice(x0i, x1i))
        z, y, x = jnp.meshgrid(
            (jnp.arange(z0i, z1i) + 0.5) * resolution,
            (jnp.arange(y0i, y1i) + 0.5) * resolution,
            (jnp.arange(x0i, x1i) + 0.5) * resolution,
            indexing="ij",
        )
        distance_sq = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2
    else:
        x0, y0 = source.position[:2]
        ny, nx = ez_shape
        cx, cy = int(round(x0 / resolution)), int(round(y0 / resolution))
        x0i, x1i = max(0, cx - radius_grid), min(nx, cx + radius_grid + 1)
        y0i, y1i = max(0, cy - radius_grid), min(ny, cy + radius_grid + 1)
        idx = (slice(y0i, y1i), slice(x0i, x1i))
        offset = 0.0 if plane_2d == "xy" else 0.5
        x, y = jnp.meshgrid(
            (jnp.arange(x0i, x1i) + offset) * resolution,
            (jnp.arange(y0i, y1i) + offset) * resolution,
            indexing="xy",
        )
        distance_sq = (x - x0) ** 2 + (y - y0) ** 2

    return idx, jnp.exp(-distance_sq / (2 * source.width**2))


def _source_requires_rectilinear_operator(ctx: SourceLoweringContext) -> bool:
    if ctx.grid is None:
        return False
    is_3d = np.asarray(ctx.fields.permittivity).ndim == 3
    axes = ("x", "y", "z") if is_3d else ("x", "y")
    return ctx.grid.metric_kind_for(axes) != "isotropic_uniform"


Direction3D = Literal["+x", "-x", "+y", "-y", "+z", "-z"]


_AXES: tuple[FieldAxis3D, FieldAxis3D, FieldAxis3D] = ("x", "y", "z")


_INDEX_AXES: tuple[FieldAxis3D, FieldAxis3D, FieldAxis3D] = ("z", "y", "x")


_AXIS_TO_VECTOR: dict[FieldAxis3D, np.ndarray] = {
    "x": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    "y": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    "z": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
}


_TRANSVERSE_AXES: dict[FieldAxis3D, tuple[FieldAxis3D, FieldAxis3D]] = {
    "x": ("y", "z"),
    "y": ("z", "x"),
    "z": ("x", "y"),
}


_FIELD_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def gaussian_beam_wavelength(source) -> float:
    if source.wavelength is not None:
        return float(source.wavelength)
    freq0 = getattr(source.source_time, "freq0", None)
    if freq0 is None or not np.isfinite(float(freq0)) or float(freq0) <= 0.0:
        raise ValueError(
            "GaussianBeamSource requires wavelength=... or a positive source_time.freq0."
        )
    return float(LIGHT_SPEED / float(freq0))


def gaussian_beam_waist_radius(source) -> float:
    if source.waist_radius is not None:
        return float(source.waist_radius)
    values = np.asarray(source.size, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("GaussianBeamSource size must not be empty.")
    return 0.25 * float(np.min(values))


def gaussian_beam_field_profile(source, fields, *, resolution: float) -> FieldProfile3D:
    permittivity = np.asarray(fields.permittivity)
    if permittivity.ndim != 3:
        raise ValueError("GaussianBeamSource currently supports 3D simulations only.")
    return GaussianBeamProfile(
        center=source.center,
        size=source.size,
        direction=source.direction,
        angle_theta=source.angle_theta,
        angle_phi=source.angle_phi,
        pol_angle=source.pol_angle,
        waist_radius=gaussian_beam_waist_radius(source),
        waist_distance=source.waist_distance,
        wavelength=gaussian_beam_wavelength(source),
        background_index=source.background_index,
        power=source.power,
    ).field_profile(
        resolution=float(resolution),
        grid_shape=tuple(map(int, permittivity.shape)),
    )


def _parse_direction(direction: str) -> tuple[FieldAxis3D, float]:
    if direction not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
        raise ValueError(f"Unsupported Gaussian beam direction {direction!r}.")
    return cast(FieldAxis3D, direction[1]), (1.0 if direction[0] == "+" else -1.0)


def _unit_vector(axis: FieldAxis3D) -> np.ndarray:
    return _AXIS_TO_VECTOR[axis].copy()


def _normalize(vector: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 1e-30:
        raise ValueError(f"{name} must have non-zero finite norm.")
    return arr / norm


def _as_xyz(value, *, name: str) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values.")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _component_field_shape(
    component: str,
    grid_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    offsets = component_axis_offsets_3d(component)
    dims = {"z": int(grid_shape[0]), "y": int(grid_shape[1]), "x": int(grid_shape[2])}
    return (
        max(0, dims["z"] - (1 if float(offsets["z"]) == 0.5 else 0)),
        max(0, dims["y"] - (1 if float(offsets["y"]) == 0.5 else 0)),
        max(0, dims["x"] - (1 if float(offsets["x"]) == 0.5 else 0)),
    )


@dataclass(frozen=True)
class GaussianBeamProfile:
    """Generate a Gaussian beam as a prepared planar ``FieldProfile3D``."""

    center: tuple[float, float, float]
    size: float | tuple[float, float] | tuple[float, float, float]
    direction: Direction3D
    angle_theta: float
    angle_phi: float
    pol_angle: float
    waist_radius: float
    waist_distance: float
    wavelength: float
    background_index: float = 1.0
    power: float = 1.0

    def __post_init__(self):
        _as_xyz(self.center, name="center")
        for name in (
            "angle_theta",
            "angle_phi",
            "pol_angle",
            "waist_radius",
            "waist_distance",
            "wavelength",
            "background_index",
            "power",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if float(self.waist_radius) <= 0.0:
            raise ValueError("waist_radius must be positive.")
        if float(self.wavelength) <= 0.0:
            raise ValueError("wavelength must be positive.")
        if float(self.background_index) <= 0.0:
            raise ValueError("background_index must be positive.")
        if float(self.power) < 0.0:
            raise ValueError("power must be non-negative.")
        _parse_direction(self.direction)
        self._transverse_extents()

    @property
    def axis(self) -> FieldAxis3D:
        axis, _sign = _parse_direction(self.direction)
        return axis

    @property
    def direction_sign(self) -> float:
        _axis, sign = _parse_direction(self.direction)
        return sign

    @property
    def omega(self) -> float:
        return float(2.0 * np.pi * LIGHT_SPEED / float(self.wavelength))

    @property
    def medium_wavenumber(self) -> float:
        return float(2.0 * np.pi * float(self.background_index) / self.wavelength)

    def propagation_unit_vector(self) -> np.ndarray:
        axis = self.axis
        normal = self.direction_sign * _unit_vector(axis)
        t1_axis, t2_axis = _TRANSVERSE_AXES[axis]
        t1 = _unit_vector(t1_axis)
        t2 = _unit_vector(t2_axis)
        theta = float(self.angle_theta)
        phi = float(self.angle_phi)
        return _normalize(
            np.cos(theta) * normal
            + np.sin(theta) * (np.cos(phi) * t1 + np.sin(phi) * t2),
            name="propagation direction",
        )

    def propagation_vector(self) -> np.ndarray:
        return self.medium_wavenumber * self.propagation_unit_vector()

    def electric_unit_vector(self) -> np.ndarray:
        axis = self.axis
        t1_axis, t2_axis = _TRANSVERSE_AXES[axis]
        t1 = _unit_vector(t1_axis)
        t2 = _unit_vector(t2_axis)
        seed = np.cos(float(self.pol_angle)) * t1 + np.sin(float(self.pol_angle)) * t2
        k_hat = self.propagation_unit_vector()
        projected = seed - float(np.dot(seed, k_hat)) * k_hat
        if float(np.linalg.norm(projected)) <= 1e-12:
            projected = np.cross(k_hat, t1)
        return _normalize(projected, name="electric polarization")

    def magnetic_unit_vector(self) -> np.ndarray:
        return _normalize(
            np.cross(self.propagation_unit_vector(), self.electric_unit_vector()),
            name="magnetic polarization",
        )

    def field_profile(
        self,
        *,
        resolution: float,
        grid_shape: tuple[int, ...],
    ) -> FieldProfile3D:
        """Sample the Gaussian beam on a Yee source plane."""
        resolution = float(resolution)
        if not np.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("resolution must be positive and finite.")
        if len(tuple(grid_shape)) != 3:
            raise ValueError("grid_shape must be a 3D cell shape in (z, y, x) order.")
        grid_shape = (int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]))
        if any(v <= 1 for v in grid_shape):
            raise ValueError("grid_shape dimensions must be greater than one.")

        axis = self.axis
        direction_sign = self.direction_sign
        center_xyz: dict[FieldAxis3D, float] = dict(
            zip(_AXES, _as_xyz(self.center, name="center"), strict=True)
        )
        transverse_slices = self._transverse_slices(resolution, grid_shape)
        phase_ref_coord = float(center_xyz[axis])
        phase_plane_coord = float(center_xyz[axis])
        k_vector = self.propagation_vector()
        k_axis = float(k_vector[_AXES.index(axis)])
        e_hat = self.electric_unit_vector()
        h_hat = (
            float(self.background_index)
            / float(np.sqrt(MU_0 / EPS_0))
            * np.cross(self.propagation_unit_vector(), e_hat)
        )
        amplitude = self._power_amplitude_scale(
            resolution,
            transverse_slices,
            k_normal_abs=abs(
                float(np.dot(self.propagation_unit_vector(), _unit_vector(axis)))
            ),
        )

        components: dict[str, np.ndarray] = {}
        indices: dict[str, FieldIndex3D] = {}
        for component in _FIELD_COMPONENTS:
            index = self._component_index(
                component,
                resolution=resolution,
                grid_shape=grid_shape,
                transverse_slices=transverse_slices,
            )
            coords = self._component_coordinate_arrays(
                component,
                index,
                resolution=resolution,
            )
            scalar = amplitude * self._scalar_profile(
                coords,
                center_xyz=center_xyz,
                phase_ref_coord=phase_ref_coord,
                k_vector=k_vector,
                k_axis=k_axis,
            )
            vector = e_hat if component.startswith("E") else h_hat
            component_axis = cast(FieldAxis3D, component[1].lower())
            components[component] = (
                scalar * float(vector[_AXES.index(component_axis)])
            ).astype(np.complex128, copy=False)
            indices[component] = index

        return FieldProfile3D(
            components=components,
            indices=indices,
            axis=axis,
            direction_sign=direction_sign,
            omega=self.omega,
            k_axis=k_axis,
            phase_ref_coord=phase_ref_coord,
            phase_plane_coord=phase_plane_coord,
        )

    def _transverse_extents(self) -> tuple[float, float]:
        values = np.asarray(self.size, dtype=np.float64).reshape(-1)
        if values.size == 1:
            out = (float(values[0]), float(values[0]))
        elif values.size == 2:
            out = (float(values[0]), float(values[1]))
        elif values.size == 3:
            extents: dict[FieldAxis3D, float] = dict(zip(_AXES, values, strict=True))
            t1_axis, t2_axis = _TRANSVERSE_AXES[self.axis]
            out = (float(extents[t1_axis]), float(extents[t2_axis]))
        else:
            raise ValueError("size must be a scalar, 2-tuple, or 3-tuple.")
        if any((not np.isfinite(v)) or v <= 0.0 for v in out):
            raise ValueError("size extents must be positive finite values.")
        return out

    def _transverse_slices(
        self,
        resolution: float,
        grid_shape: tuple[int, int, int],
    ) -> dict[FieldAxis3D, slice]:
        center_xyz: dict[FieldAxis3D, float] = dict(
            zip(_AXES, _as_xyz(self.center, name="center"), strict=True)
        )
        extents = self._transverse_extents()
        out: dict[FieldAxis3D, slice] = {}
        dims = {"z": grid_shape[0], "y": grid_shape[1], "x": grid_shape[2]}
        for axis_name, extent in zip(_TRANSVERSE_AXES[self.axis], extents, strict=True):
            center = float(center_xyz[axis_name])
            start = int(np.floor((center - 0.5 * extent) / resolution))
            stop = int(np.ceil((center + 0.5 * extent) / resolution))
            start = max(0, min(start, int(dims[axis_name]) - 1))
            stop = max(start + 1, min(stop, int(dims[axis_name])))
            out[axis_name] = slice(start, stop)
        return out

    def _component_index(
        self,
        component: str,
        *,
        resolution: float,
        grid_shape: tuple[int, int, int],
        transverse_slices: dict[FieldAxis3D, slice],
    ) -> tuple[slice, slice, slice]:
        center_xyz = dict(zip(_AXES, _as_xyz(self.center, name="center"), strict=True))
        offsets = component_axis_offsets_3d(component)
        field_shape = _component_field_shape(component, grid_shape)
        items: list[int | slice] = []
        for dim, axis_name in enumerate(_INDEX_AXES):
            if axis_name == self.axis:
                raw = int(
                    round(
                        float(center_xyz[axis_name]) / resolution - offsets[axis_name]
                    )
                )
                items.append(max(0, min(raw, int(field_shape[dim]) - 1)))
                continue
            source_slice = transverse_slices[axis_name]
            start = max(0, min(int(source_slice.start or 0), int(field_shape[dim]) - 1))
            stop = max(
                start + 1,
                min(int(source_slice.stop or start + 1), int(field_shape[dim])),
            )
            items.append(slice(start, stop))
        return tuple(items)  # type: ignore[return-value]

    def _component_coordinate_arrays(
        self,
        component: str,
        index: tuple[slice, slice, slice],
        *,
        resolution: float,
    ) -> dict[FieldAxis3D, np.ndarray]:
        offsets = component_axis_offsets_3d(component)
        axis_values: dict[FieldAxis3D, np.ndarray | float] = {}
        mesh_axes: list[FieldAxis3D] = []
        mesh_values: list[np.ndarray] = []
        for axis_name, item in zip(_INDEX_AXES, index, strict=True):
            if isinstance(item, slice):
                values = (
                    np.arange(
                        int(item.start or 0), int(item.stop or 0), dtype=np.float64
                    )
                    + float(offsets[axis_name])
                ) * resolution
                axis_values[axis_name] = values
                mesh_axes.append(axis_name)
                mesh_values.append(values)
            else:
                axis_values[axis_name] = (
                    int(item) + float(offsets[axis_name])
                ) * resolution

        meshes = np.meshgrid(*mesh_values, indexing="ij")
        coords: dict[FieldAxis3D, np.ndarray] = {}
        shape = meshes[0].shape if meshes else ()
        for axis_name, axis_value in axis_values.items():
            if axis_name in mesh_axes:
                coords[axis_name] = meshes[mesh_axes.index(axis_name)]
            else:
                coords[axis_name] = np.full(shape, float(axis_value), dtype=np.float64)
        return coords

    def _scalar_profile(
        self,
        coords: Mapping[FieldAxis3D, np.ndarray],
        *,
        center_xyz: Mapping[FieldAxis3D, float],
        phase_ref_coord: float,
        k_vector: np.ndarray,
        k_axis: float,
    ) -> np.ndarray:
        r = np.stack(
            [coords[axis] - float(center_xyz[axis]) for axis in _AXES],
            axis=0,
        )
        e_hat = self.electric_unit_vector()
        v_hat = self.magnetic_unit_vector()
        u_coord = np.tensordot(e_hat, r, axes=(0, 0))
        v_coord = np.tensordot(v_hat, r, axes=(0, 0))
        rho2 = u_coord**2 + v_coord**2
        radius, curvature, gouy = self._beam_radius_curvature_gouy()
        envelope = np.exp(-rho2 / max(radius**2, 1e-300))
        phase = -np.tensordot(k_vector, r, axes=(0, 0))
        phase += float(k_axis) * (coords[self.axis] - float(phase_ref_coord))
        if np.isfinite(curvature):
            phase += -self.medium_wavenumber * rho2 / (2.0 * curvature)
        phase += gouy
        return envelope * np.exp(1j * phase)

    def _beam_radius_curvature_gouy(self) -> tuple[float, float, float]:
        waist = float(self.waist_radius)
        wavelength_medium = float(self.wavelength) / float(self.background_index)
        rayleigh = np.pi * waist**2 / wavelength_medium
        z = float(self.waist_distance)
        radius = waist * np.sqrt(1.0 + (z / rayleigh) ** 2)
        curvature = np.inf if abs(z) <= 1e-30 else z * (1.0 + (rayleigh / z) ** 2)
        gouy = float(np.arctan2(z, rayleigh))
        return float(radius), float(curvature), gouy

    def _power_amplitude_scale(
        self,
        resolution: float,
        transverse_slices: Mapping[FieldAxis3D, slice],
        *,
        k_normal_abs: float,
    ) -> float:
        if float(self.power) == 0.0:
            return 0.0
        center_xyz: dict[FieldAxis3D, float] = dict(
            zip(_AXES, _as_xyz(self.center, name="center"), strict=True)
        )
        t_axes = _TRANSVERSE_AXES[self.axis]
        coords = []
        for axis_name in t_axes:
            item = transverse_slices[axis_name]
            coords.append(
                (np.arange(int(item.start or 0), int(item.stop or 0)) + 0.5)
                * resolution
                - float(center_xyz[axis_name])
            )
        a, b = np.meshgrid(coords[0], coords[1], indexing="ij")
        radius, _curvature, _gouy = self._beam_radius_curvature_gouy()
        envelope2 = np.exp(-2.0 * (a**2 + b**2) / max(radius**2, 1e-300))
        eta = float(np.sqrt(MU_0 / EPS_0)) / float(self.background_index)
        flux = (
            0.5
            * max(float(k_normal_abs), 1e-30)
            / eta
            * float(np.sum(envelope2))
            * float(resolution) ** 2
        )
        if (not np.isfinite(flux)) or flux <= 1e-300:
            return 0.0
        return float(np.sqrt(float(self.power) / flux))


def scale_field_profile(profile: FieldProfile3D, power: float) -> FieldProfile3D:
    if not isinstance(profile, FieldProfile3D):
        raise TypeError("profile must be a FieldProfile3D instance")
    power = float(power)
    if not np.isfinite(power) or power < 0.0:
        raise ValueError(
            f"Custom field profile source power must be non-negative, got {power!r}."
        )
    if power == 1.0:
        return profile
    scale = float(np.sqrt(power))
    return replace(
        profile,
        components={
            name: np.asarray(value, dtype=np.complex128) * scale
            for name, value in profile.components.items()
        },
    )


def field_profile_phasor_residuals(
    profile: FieldProfile3D,
    fields,
    *,
    dt: float,
    resolution: float,
    max_shift: int = 1,
):
    return (
        *compute_discrete_3d_h_phasor_residuals(
            profile,
            fields,
            resolution=float(resolution),
            max_shift=int(max(1, max_shift)),
            dt=float(dt),
        ),
        *compute_discrete_3d_e_phasor_residuals(
            profile,
            fields,
            resolution=float(resolution),
            max_shift=int(max(1, max_shift)),
            dt=float(dt),
        ),
    )


def _field_profile_injection_plan(
    profile: FieldProfile3D,
    waveform: TemporalWaveform,
    ctx: SourceLoweringContext,
    *,
    max_shift: int = 1,
    power: float = 1.0,
) -> CompiledInjectionPlan:
    profile = scale_field_profile(profile, power)
    return CompiledInjectionPlan(
        tuple(
            _injection_entry(
                component=residual.component,
                timing=residual.timing,
                index=residual.index,
                values=np.asarray(residual.residual, dtype=np.complex128),
                waveform=waveform,
                target_shape=tuple(getattr(ctx.fields, residual.component).shape),
            )
            for residual in field_profile_phasor_residuals(
                profile,
                ctx.fields,
                dt=ctx.dt,
                resolution=ctx.resolution,
                max_shift=max_shift,
            )
        )
    )
