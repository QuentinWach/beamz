"""Pure mode-launch planning for ModeSource compilation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.devices._immutable import readonly_array
from beamz.devices._placement import (
    snap_mode_source_region,
    snap_mode_source_region_grid,
)
from beamz.devices.modes.discrete import solve_beamz_mode
from beamz.devices.modes.fields import _modal_power, _numeric_wave_number
from beamz.devices.modes.plane import solve_mode_plane_3d, solve_modes

from . import planar_tfsf
from .solve import _require_cell_mode_materials
from .specs import FieldProfile3D, ModeSource

logger = logging.getLogger(__name__)


def _to_real_profile(profile, imag_ratio_warn=1e-2, eps=1e-30):
    """Project a launch profile to real-valued injection coefficients."""
    arr = np.asarray(profile, dtype=np.complex128)
    real = np.real(arr)
    real_peak = float(np.max(np.abs(real))) if arr.size else 0.0
    imag_peak = float(np.max(np.abs(np.imag(arr)))) if arr.size else 0.0
    if real_peak > eps and imag_peak / real_peak > imag_ratio_warn:
        logger.debug(
            "Mode profile has non-negligible imaginary content before real projection: "
            "imag/real peak ratio=%.3e",
            imag_peak / real_peak,
        )
    return real


def _power_scale(power) -> float:
    value = float(power)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            f"ModeSource power must be a non-negative finite value, got {power!r}."
        )
    return float(np.sqrt(value))


def _scale_profiles_for_power(profiles, power):
    """Scale unit-power modal profiles to the requested launched power."""
    scale = _power_scale(power)
    if scale == 1.0:
        return profiles
    for key, value in profiles.items():
        if value is not None:
            profiles[key] = np.asarray(value) * scale
    return profiles


def _scale_pair_for_power(first, second, power):
    scale = _power_scale(power)
    if scale == 1.0:
        return first, second
    return np.asarray(first) * scale, np.asarray(second) * scale


@dataclass(frozen=True)
class ModePlanEntry:
    component: str
    timing: str
    index: tuple[Any, ...]
    profile: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "profile", readonly_array(self.profile))


@dataclass(frozen=True)
class Mode2DLaunchPlan:
    entries: tuple[ModePlanEntry, ...]
    normal_spacing: float | None = None


@dataclass(frozen=True)
class Mode3DLaunchPlan:
    residuals: tuple[planar_tfsf.ModeSource3DResidual, ...]
    field_profile: FieldProfile3D | None = None
    launch_power_ratio: float | None = None
    launch_amplitude_scale: float = 1.0
    unscaled_launched_power: float | None = None

    @property
    def launched_power(self) -> float | None:
        """Return the compiler-estimated net power after launch scaling."""
        power = self.unscaled_launched_power
        if power is None or not np.isfinite(float(power)):
            return None
        return float(power) * float(self.launch_amplitude_scale) ** 2


ModeLaunchPlan = Mode2DLaunchPlan | Mode3DLaunchPlan


def _orient_3d_profiles_for_launch(profiles, axis: str, direction_sign: float):
    out = dict(profiles)
    if axis != "y":
        return out
    if direction_sign > 0.0:
        if out.get("Ex") is not None:
            out["Ex"] = -out["Ex"]
        if out.get("Hz") is not None:
            out["Hz"] = -out["Hz"]
    else:
        if out.get("Ez") is not None:
            out["Ez"] = -out["Ez"]
        if out.get("Hx") is not None:
            out["Hx"] = -out["Hx"]
    return out


def _modal_power_2d(e_profile, h_profile, signed_flux_sign, dl):
    e = np.asarray(e_profile, dtype=np.complex128).reshape(-1)
    h = np.asarray(h_profile, dtype=np.complex128).reshape(-1)
    n = int(min(e.size, h.size))
    if n <= 0:
        return 0.0
    measure = np.asarray(dl, dtype=float)
    weights = measure if measure.ndim == 0 else measure.reshape(-1)[:n]
    p = 0.5 * np.real(
        np.sum(weights * float(signed_flux_sign) * e[:n] * np.conjugate(h[:n]))
    )
    return float(p)


def _normalize_2d_pair_by_power(h_profile, e_profile, signed_flux_sign, dl, eps=1e-30):
    h = np.asarray(h_profile)
    e = np.asarray(e_profile)
    p = _modal_power_2d(e, h, signed_flux_sign=signed_flux_sign, dl=dl)
    if np.isfinite(p) and abs(p) > eps:
        scale = np.sqrt(1.0 / abs(p))
        h = h * scale
        e = e * scale
    return h, e


def _impedance_match_e_profile(e_profile, h_profile, z_target, eps=1e-12):
    e = np.asarray(e_profile, dtype=np.complex128).reshape(-1)
    h = np.asarray(h_profile, dtype=np.complex128).reshape(-1)
    n = int(min(e.size, h.size))
    if n <= 0:
        return e_profile
    denom = np.sum(h[:n] * np.conjugate(h[:n]))
    if abs(denom) <= eps:
        return e_profile
    z_est = np.sum(e[:n] * np.conjugate(h[:n])) / denom
    z_mag = float(abs(z_est))
    if (not np.isfinite(z_mag)) or z_mag <= eps:
        return e_profile
    return np.asarray(e_profile) * (float(abs(z_target)) / z_mag)


def _make_1d_window(width_cells, alpha=0.3):
    if width_cells > 2:
        from scipy.signal.windows import tukey

        return tukey(width_cells, alpha=alpha)
    return np.ones(max(1, width_cells))


def _solve_2d_mode(
    source: ModeSource,
    eps_profile,
    omega: float,
    resolution: float,
    axis: str,
    polarization: str,
    grid_edges: np.ndarray | None = None,
):
    del axis
    mode_spec = source.mode_spec
    eps_profile_arr = np.asarray(eps_profile)
    n_local_max = float(np.sqrt(max(float(np.max(np.real(eps_profile_arr))), 1e-12)))
    target_neff = (
        mode_spec.target_neff
        if mode_spec.target_neff is not None
        else 0.98 * n_local_max
    )
    mode_count = max(
        int(mode_spec.num_modes),
        int(mode_spec.mode_index) + 1,
        3,
    )
    neff_val, e_fields, h_fields, _ = solve_modes(
        eps=eps_profile,
        omega=omega,
        dL=resolution,
        m=mode_count,
        direction=cast(
            Literal["+x", "-x", "+y", "-y", "+z", "-z"],
            source.signed_direction,
        ),
        filter_pol=cast(Literal["te", "tm"], polarization),
        target_neff=target_neff,
        return_fields=True,
        grid_edges=None if grid_edges is None else (np.asarray(grid_edges),),
    )
    mode_idx = min(int(mode_spec.mode_index), len(neff_val) - 1)
    return np.asarray(neff_val)[mode_idx], e_fields[mode_idx], h_fields[mode_idx]


def _normalize_2d_launch_pair(first, second, *, flux_sign, resolution, power):
    first, second = _normalize_2d_pair_by_power(
        first, second, signed_flux_sign=flux_sign, dl=resolution
    )
    first, second = _to_real_profile(first), _to_real_profile(second)
    first, second = _normalize_2d_pair_by_power(
        first, second, signed_flux_sign=flux_sign, dl=resolution
    )
    return _scale_pair_for_power(first, second, power)


def _plane_index(axis, normal_index, support):
    return (support, normal_index) if axis == "x" else (normal_index, support)


def _plan_2d_entries(
    source,
    e_mode,
    h_mode,
    neff,
    center_idx,
    offset_idx,
    snapped,
    shape,
    resolution,
    polarization,
    measure=None,
):
    axis = source.axis
    transverse_axis = "y" if axis == "x" else "x"
    interval = snapped.axis_interval(transverse_axis)
    start, end = int(interval.start), int(interval.stop)
    dir_sign = 1.0 if source.direction == "+" else -1.0
    z_target = np.sqrt(MU_0 / EPS_0) / max(np.real(neff), 1e-6)

    if polarization == "tm":
        h_raw = np.squeeze(h_mode[1 if axis == "x" else 0])
        e_raw = np.squeeze(e_mode[2])
        phase_ref = np.angle(h_raw.flatten()[np.argmax(np.abs(h_raw))])
        h_aligned = h_raw * np.exp(-1j * phase_ref)
        e_aligned = _impedance_match_e_profile(
            e_raw * np.exp(-1j * phase_ref), h_aligned, z_target
        )
        window = _make_1d_window(end - start)
        h_crop, e_crop = h_aligned[start:end], e_aligned[start:end]
        if len(h_crop) == len(window):
            h_crop, e_crop = h_crop * window, e_crop * window
        jz_profile = -dir_sign * h_crop
        magnetic_profile = dir_sign * (1.0 if axis == "x" else -1.0) * e_crop
        flux_sign = -1.0 if axis == "x" else 1.0
        jz_profile, magnetic_profile = _normalize_2d_launch_pair(
            jz_profile,
            magnetic_profile,
            flux_sign=flux_sign,
            resolution=resolution if measure is None else measure,
            power=source.power,
        )
        support = slice(start, end)
        h_component = "Hy" if axis == "x" else "Hx"
        return (
            ModePlanEntry(
                h_component,
                "h",
                _plane_index(axis, offset_idx, support),
                -np.asarray(magnetic_profile),
            ),
            ModePlanEntry(
                "Ez",
                "e",
                _plane_index(axis, center_idx, support),
                np.asarray(jz_profile),
            ),
        )

    h_raw = np.squeeze(h_mode[2])
    e_raw = np.squeeze(e_mode[1 if axis == "x" else 0])
    h_staggered = 0.5 * (h_raw[:-1] + h_raw[1:])
    e_staggered = 0.5 * (e_raw[:-1] + e_raw[1:])
    phase_ref = np.angle(h_staggered.flatten()[np.argmax(np.abs(h_staggered))])
    h_profile = h_staggered * np.exp(-1j * phase_ref)
    e_profile = _impedance_match_e_profile(
        e_staggered * np.exp(-1j * phase_ref), h_profile, z_target
    )
    stop = min(end, len(h_profile))
    window = _make_1d_window(stop - start)
    h_crop = h_profile[start:stop]
    e_crop = e_profile[start : min(end, len(e_profile))]
    if len(h_crop) == len(window):
        h_crop, e_crop = h_crop * window, e_crop * window
    electric_profile = dir_sign * h_crop
    mz_profile = -dir_sign * e_crop
    electric_profile, mz_profile = _normalize_2d_launch_pair(
        electric_profile,
        mz_profile,
        flux_sign=1.0 if axis == "x" else -1.0,
        resolution=resolution if measure is None else measure,
        power=source.power,
    )
    normal_cells = shape[1] if axis == "x" else shape[0]
    hz_index = (
        max(0, offset_idx - 1)
        if source.direction == "+"
        else min(normal_cells - 2, offset_idx)
    )
    support = slice(start, stop)
    electric_component = "Ey" if axis == "x" else "Ex"
    return (
        ModePlanEntry(
            "Hz",
            "h",
            _plane_index(axis, hz_index, support),
            np.asarray(mz_profile),
        ),
        ModePlanEntry(
            electric_component,
            "e",
            _plane_index(axis, offset_idx, support),
            -np.asarray(electric_profile),
        ),
    )


def _plan_2d_mode_source(
    source: ModeSource,
    fields,
    resolution: float,
    dt: float | None,
    grid=None,
) -> Mode2DLaunchPlan:
    polarization = str(getattr(fields, "polarization_2d", "tm"))
    requested = source.mode_spec.polarization
    if requested is not None and requested != polarization:
        raise ValueError(
            f"ModeSource polarization {requested!r} does not match the "
            f"Simulation polarization {polarization!r}."
        )
    if np.any(np.asarray(fields.conductivity) != 0.0):
        raise ValueError(
            "ModeSource does not yet support conductive material profiles; the "
            "mode solve would not match lossy FDTD propagation."
        )
    _require_cell_mode_materials(
        getattr(fields, "material_grid", None),
        operation="2D ModeSource",
    )
    permittivity = np.asarray(fields.permittivity)
    ny, nx = permittivity.shape
    axis = source.axis
    if axis == "z":
        raise ValueError(
            "direction '+z'/'-z' requires a 3D permittivity grid; received 2D data"
        )
    metric_grid = (
        grid
        if grid is not None and grid.metric_kind_for(("x", "y")) != "isotropic_uniform"
        else None
    )
    source_center = tuple(float(v) for v in source.center)
    source_width = float(source.transverse_size[0])
    direction_sign = 1.0 if source.direction == "+" else -1.0
    if metric_grid is not None:
        snapped = snap_mode_source_region_grid(
            center=source_center,
            width=source_width,
            height=None,
            axis=axis,
            direction_sign=direction_sign,
            is_3d=False,
            grid=metric_grid,
        )
    else:
        snapped = snap_mode_source_region(
            center=source_center,
            width=source_width,
            height=None,
            axis=axis,
            direction_sign=direction_sign,
            is_3d=False,
            grid_shape=(ny, nx),
            resolution=float(resolution),
        )
    center_idx = int(snapped.plane_index)
    if snapped.companion_index is None:
        raise ValueError("A mode source launch needs a companion Yee plane.")
    offset_idx = int(snapped.companion_index)
    material_grid = getattr(fields, "material_grid", None)
    if (
        material_grid is not None
        and material_grid.uses_direct_yee_materials
        and metric_grid is None
    ):
        component = (
            "eps_z" if polarization == "tm" else ("eps_y" if axis == "x" else "eps_x")
        )
        direct = np.asarray(getattr(fields, component))
        normal_index = min(center_idx, direct.shape[1 if axis == "x" else 0] - 1)
        eps_profile = (
            direct[:, normal_index] if axis == "x" else direct[normal_index, :]
        )
    else:
        eps_profile = (
            permittivity[:, center_idx] if axis == "x" else permittivity[center_idx, :]
        )
    omega = 2.0 * np.pi * source.frequency
    interval = snapped.axis_interval("y" if axis == "x" else "x")
    if interval is None:
        raise RuntimeError("A 2D mode source is missing its transverse interval.")
    if metric_grid is not None:
        transverse_axis = "y" if axis == "x" else "x"
        grid_edges = np.asarray(metric_grid.axis_edges(transverse_axis))
        measure = np.asarray(metric_grid.cell_widths(transverse_axis))[
            int(interval.start) : int(interval.stop)
        ]
    else:
        grid_edges = None
        measure = None
    neff, e_mode, h_mode = _solve_2d_mode(
        source,
        eps_profile,
        omega,
        resolution,
        axis,
        polarization,
        grid_edges=grid_edges,
    )
    entries = _plan_2d_entries(
        source,
        e_mode,
        h_mode,
        neff,
        center_idx,
        offset_idx,
        snapped,
        (ny, nx),
        resolution,
        polarization,
        measure=measure,
    )
    normal_spacing = (
        float(metric_grid.cell_widths(axis)[center_idx])
        if metric_grid is not None
        else None
    )
    return Mode2DLaunchPlan(entries=entries, normal_spacing=normal_spacing)


def _plan_3d_mode_source(
    source: ModeSource,
    fields,
    resolution: float,
    dt: float | None,
    grid=None,
) -> Mode3DLaunchPlan:
    if np.any(np.asarray(fields.conductivity) != 0.0):
        raise ValueError(
            "ModeSource does not yet support conductive material profiles; the "
            "mode solve would not match lossy FDTD propagation."
        )
    permittivity = np.asarray(fields.permittivity)
    nz, ny, nx = permittivity.shape
    axis = source.axis
    metric_grid = (
        grid if grid is not None and grid.metric_kind != "isotropic_uniform" else None
    )
    width, height = source.transverse_size
    source_center = tuple(float(v) for v in source.center)
    source_width = float(width)
    source_height = float(height)
    direction_sign = 1.0 if source.direction == "+" else -1.0
    if grid is not None:
        snapped = snap_mode_source_region_grid(
            center=source_center,
            width=source_width,
            height=source_height,
            axis=axis,
            direction_sign=direction_sign,
            is_3d=True,
            grid=grid,
        )
    else:
        snapped = snap_mode_source_region(
            center=source_center,
            width=source_width,
            height=source_height,
            axis=axis,
            direction_sign=direction_sign,
            is_3d=True,
            grid_shape=(nz, ny, nx),
            resolution=float(resolution),
        )
    center_idx = int(snapped.plane_index)
    if snapped.companion_index is None:
        raise ValueError("A mode source launch needs a companion Yee plane.")
    offset_idx = int(snapped.companion_index)
    omega = 2.0 * np.pi * source.frequency
    solver_direction = "+y" if axis == "y" else source.signed_direction
    discrete_mode = solve_mode_plane_3d(
        permittivity,
        np.asarray(fields.permeability),
        material_tensors=getattr(fields.material_grid, "tensors", None),
        yee_materials=(
            getattr(fields.material_grid, "yee_materials", None)
            if fields.material_grid.uses_direct_yee_materials
            else None
        ),
        frequency=source.frequency,
        resolution=resolution,
        dt=dt,
        axis=axis,
        grid_shape=(nz, ny, nx),
        center=source.center,
        width=width,
        height=height,
        plane_index=center_idx,
        offset_index=offset_idx,
        direction=source.signed_direction,
        solver_direction=solver_direction,
        mode_index=source.mode_spec.mode_index,
        polarization=source.mode_spec.polarization,
        target_neff=source.mode_spec.target_neff,
        num_modes=max(
            int(source.mode_spec.num_modes),
            int(source.mode_spec.mode_index) + 1,
        ),
        snapped_region=snapped,
        solver=solve_beamz_mode,
        grid=metric_grid,
    )
    profiles = {
        name: np.asarray(value, dtype=np.complex128)
        for name, value in discrete_mode.profiles.items()
    }
    profiles = _scale_profiles_for_power(profiles, source.power)
    profiles = _orient_3d_profiles_for_launch(
        profiles,
        axis,
        1.0 if source.direction == "+" else -1.0,
    )
    indices = {
        name: index
        for name, index in dict(discrete_mode.component_indices).items()
        if name in profiles
    }
    k_axis = float(discrete_mode.k_num_axis)
    if dt is None:
        k_axis = float(np.real(discrete_mode.neff)) * float(omega) / LIGHT_SPEED
    elif not np.isfinite(k_axis):
        k_axis = _numeric_wave_number(omega, dt, resolution, discrete_mode.neff)
    field_profile = FieldProfile3D(
        components=profiles,
        indices=indices,  # type: ignore[arg-type]
        axis=axis,  # type: ignore[arg-type]
        direction_sign=1.0 if source.direction == "+" else -1.0,
        omega=float(omega),
        k_axis=float(k_axis),
        phase_ref_coord=float(discrete_mode.phase_reference_coord),
        phase_plane_coord=float(discrete_mode.phase_plane_coord),
        grid=metric_grid,
        power_weights=discrete_mode.integration_weights,
    )
    residuals = (
        *planar_tfsf.compute_discrete_3d_h_phasor_residuals(
            field_profile,
            fields,
            resolution=float(resolution),
            max_shift=12,
            dt=float(dt or 0.0),
        ),
        *planar_tfsf.compute_discrete_3d_e_phasor_residuals(
            field_profile,
            fields,
            resolution=float(resolution),
            max_shift=12,
            dt=float(dt or 0.0),
        ),
    )
    residuals = tuple(residuals)
    launch_power_ratio, unscaled_launched_power = _launch_power_diagnostics_3d(
        source,
        field_profile,
        residuals,
        fields,
        resolution=float(resolution),
        dt=dt,
        requested_power=float(source.power),
    )
    launch_amplitude_scale = _launch_amplitude_scale(launch_power_ratio)
    return Mode3DLaunchPlan(
        residuals=residuals,
        field_profile=field_profile,
        launch_power_ratio=launch_power_ratio,
        launch_amplitude_scale=launch_amplitude_scale,
        unscaled_launched_power=unscaled_launched_power,
    )


def _launch_amplitude_scale(launch_power_ratio: float | None) -> float:
    if launch_power_ratio is None:
        return 1.0
    ratio = float(launch_power_ratio)
    if (not np.isfinite(ratio)) or ratio <= 1e-24:
        return 1.0
    return float(1.0 / np.sqrt(ratio))


def _launch_power_diagnostics_3d(
    source: ModeSource,
    field_profile: FieldProfile3D,
    residuals: tuple[planar_tfsf.ModeSource3DResidual, ...],
    fields,
    *,
    resolution: float,
    dt: float | None,
    requested_power: float,
) -> tuple[float | None, float | None]:
    del source, residuals, fields, resolution, dt
    if (not np.isfinite(requested_power)) or requested_power <= 1e-30:
        return None, None
    weights = field_profile.power_weights
    if not weights:
        return None, None
    launched_power = _modal_power(
        field_profile.components,
        axis=field_profile.axis,
        measure=weights,
        direction_sign=float(field_profile.direction_sign),
    )
    ratio = float(launched_power / float(requested_power))
    if (not np.isfinite(ratio)) or ratio <= 1e-24:
        ratio = None
    if (not np.isfinite(launched_power)) or launched_power <= 1e-24:
        launched_power = None
    return ratio, launched_power


def plan_mode_source_launch(
    source: ModeSource,
    fields,
    *,
    resolution: float,
    dt: float | None,
    grid=None,
) -> ModeLaunchPlan:
    permittivity = np.asarray(fields.permittivity)
    if permittivity.ndim == 3:
        return _plan_3d_mode_source(
            source,
            fields,
            float(resolution),
            dt,
            grid=grid if grid is not None else getattr(fields, "geometry", None),
        )
    if permittivity.ndim == 2:
        return _plan_2d_mode_source(
            source,
            fields,
            float(resolution),
            dt,
            grid=grid if grid is not None else getattr(fields, "geometry", None),
        )
    raise ValueError("ModeSource expects a 2D or 3D permittivity grid.")


__all__ = [
    "Mode2DLaunchPlan",
    "Mode3DLaunchPlan",
    "ModeLaunchPlan",
    "ModePlanEntry",
    "plan_mode_source_launch",
]
