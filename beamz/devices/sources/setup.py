import math

import jax.numpy as jnp
import numpy as np

from beamz.arrays import to_host
from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.devices.sources.profiles_2d import (
    _align_2d_impedance_pair,
    _crop_window_2d_pair,
    _finalize_2d_launch_pair,
    _stagger_2d_pair,
)
from beamz.devices.sources.profiles_3d import _build_3d_profiles
from beamz.devices.sources.profiles_common import (
    _axis_index_from_component_indices,
    _component_axis_coord,
    _dominant_3d_pair,
    _parse_direction,
    _remap_3d_solver_components,
    _select_3d_impedance_index,
    _select_3d_phase_ref,
    _select_core_confined_mode_index,
    _solve_numeric_k_axis,
    _numeric_phase_delay,
)
from beamz.devices.sources.spec import GaussianSourceSpec, ModeSourceSpec
from beamz.devices.sources.state import GaussianSourceState, ModeSourceState
from beamz.devices.sources.solve import solve_modes


def _center_index(coord: float, step: float, limit: int) -> int:
    return max(0, min(limit - 1, int(round(coord / step - 0.5))))


def _transverse_bounds(center: float, width: float, resolution: float, limit: int):
    center_idx = int(round(center / resolution))
    half_width_idx = int(round((width / 2) / resolution))
    start = max(0, center_idx - half_width_idx)
    end = min(limit, center_idx + half_width_idx)
    return start, end, slice(start, end)


_SETUP_2D_TM = {
    "x": {
        "h_component": "Hx",
        "ez_attr": "_ez_indices",
        "h_attr": "_h_indices",
        "profile_attrs": ("_jz_profile", "_my_profile"),
        "field_signs": (1.0, 1.0),
        "flux_sign": -1.0,
        "h_mode_index": 1,
        "e_mode_index": 2,
    },
    "y": {
        "h_component": "Hy",
        "ez_attr": "_ez_indices",
        "h_attr": "_h_indices",
        "profile_attrs": ("_jz_profile", "_my_profile"),
        "field_signs": (1.0, -1.0),
        "flux_sign": 1.0,
        "h_mode_index": 1,
        "e_mode_index": 2,
    },
}

_SETUP_2D_TE = {
    "x": {
        "e_component": "Ey",
        "hz_attr": "_hz_indices",
        "e_attr": "_e_indices",
        "profile_attrs": ("_jy_profile", "_mz_profile"),
        "field_signs": (1.0, -1.0),
        "flux_sign": 1.0,
        "h_mode_index": 2,
        "e_mode_index": 1,
        "offset_limit_axis": 0,
    },
    "y": {
        "e_component": "Ex",
        "hz_attr": "_hz_indices",
        "e_attr": "_e_indices",
        "profile_attrs": ("_jx_profile", "_mz_profile"),
        "field_signs": (-1.0, -1.0),
        "flux_sign": -1.0,
        "h_mode_index": 2,
        "e_mode_index": 1,
        "offset_limit_axis": 1,
    },
}

_MODE_SPEC_MAP = {
    "_direction_axis": "direction_axis",
    "_direction_sign": "direction_sign",
}
_MODE_STATE_MAP = {
    "_Ex_profile": "Ex_profile",
    "_Ey_profile": "Ey_profile",
    "_Ez_profile": "Ez_profile",
    "_Hx_profile": "Hx_profile",
    "_Hy_profile": "Hy_profile",
    "_Hz_profile": "Hz_profile",
    "_Ex_indices": "Ex_indices",
    "_Ey_indices": "Ey_indices",
    "_Ez_indices": "Ez_indices",
    "_Hx_indices": "Hx_indices",
    "_Hy_indices": "Hy_indices",
    "_Hz_indices": "Hz_indices",
    "_jz_profile": "jz_profile",
    "_my_profile": "my_profile",
    "_mz_profile": "mz_profile",
    "_jy_profile": "jy_profile",
    "_jx_profile": "jx_profile",
    "_ez_indices": "ez_indices",
    "_h_indices": "h_indices",
    "_hz_indices": "hz_indices",
    "_e_indices": "e_indices",
    "_h_component": "h_component",
    "_e_component": "e_component",
    "_neff": "neff",
    "_impedance_neff": "impedance_neff",
    "_dt_physical": "dt_physical",
    "_launch_dt": "launch_dt",
    "_initialized": "initialized",
    "_resolution": "resolution",
    "_is_3d": "is_3d",
    "_grid_shape": "grid_shape",
    "_eps_profile_2d": "eps_profile_2d",
    "_axis": "axis",
    "_transverse_start": "transverse_start",
    "_transverse_end": "transverse_end",
    "_x_start": "x_start",
    "_x_end": "x_end",
    "_y_start": "y_start",
    "_y_end": "y_end",
    "_z_start": "z_start",
    "_z_end": "z_end",
}


class _ModeSourceAdapter:
    def __init__(self, spec, state):
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "state", state)

    def __getattr__(self, name):
        if name in {"spec", "state"}:
            return object.__getattribute__(self, name)
        mapped = _MODE_SPEC_MAP.get(name, name)
        spec = object.__getattribute__(self, "spec")
        if hasattr(spec, mapped):
            return getattr(spec, mapped)
        state = object.__getattribute__(self, "state")
        if name in _MODE_STATE_MAP:
            return getattr(state, _MODE_STATE_MAP[name])
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in {"spec", "state"}:
            object.__setattr__(self, name, value)
            return
        if name in _MODE_STATE_MAP:
            setattr(self.state, _MODE_STATE_MAP[name], value)
            return
        raise AttributeError(name)


def sample_signal(spec, time, dt):
    signal = spec.signal
    idx_float = float(time / dt)
    idx_low = int(np.floor(idx_float))
    idx_high = idx_low + 1
    frac = idx_float - idx_low

    if 0 <= idx_low < len(signal) - 1:
        return (1.0 - frac) * signal[idx_low] + frac * signal[idx_high]
    if idx_low == len(signal) - 1:
        return signal[idx_low]
    return 0.0


def initialize_gaussian_state(spec, state, ez_shape, resolution):
    if not isinstance(spec, GaussianSourceSpec):
        raise TypeError("initialize_gaussian_state expects a GaussianSourceSpec")
    if not isinstance(state, GaussianSourceState):
        raise TypeError("initialize_gaussian_state expects a GaussianSourceState")

    is_3d = len(spec.position) >= 3 if hasattr(spec.position, "__len__") else False
    sigma_grid = spec.width / resolution
    radius_grid = int(np.ceil(4 * sigma_grid))

    if is_3d:
        x0, y0, z0 = spec.position
        nz, ny, nx = ez_shape
        cx, cy, cz = (int(round(c / resolution)) for c in (x0, y0, z0))

        x_start, x_end = max(0, cx - radius_grid), min(nx, cx + radius_grid + 1)
        y_start, y_end = max(0, cy - radius_grid), min(ny, cy + radius_grid + 1)
        z_start, z_end = max(0, cz - radius_grid), min(nz, cz + radius_grid + 1)

        state.grid_indices = (
            slice(z_start, z_end),
            slice(y_start, y_end),
            slice(x_start, x_end),
        )

        x_coords = (jnp.arange(x_start, x_end) + 0.5) * resolution
        y_coords = (jnp.arange(y_start, y_end) + 0.5) * resolution
        z_coords = (jnp.arange(z_start, z_end) + 0.5) * resolution
        Z, Y, X = jnp.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
        dist_sq = (X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2
    else:
        x0, y0 = spec.position
        ny, nx = ez_shape
        cx, cy = int(round(x0 / resolution)), int(round(y0 / resolution))

        x_start, x_end = max(0, cx - radius_grid), min(nx, cx + radius_grid + 1)
        y_start, y_end = max(0, cy - radius_grid), min(ny, cy + radius_grid + 1)

        state.grid_indices = (slice(y_start, y_end), slice(x_start, x_end))

        x_coords = (jnp.arange(x_start, x_end) + 0.5) * resolution
        y_coords = (jnp.arange(y_start, y_end) + 0.5) * resolution
        X, Y = jnp.meshgrid(x_coords, y_coords, indexing="xy")
        dist_sq = (X - x0) ** 2 + (Y - y0) ** 2

    state.spatial_profile_ez = jnp.exp(-dist_sq / (2 * spec.width**2))


def initialize_mode_state(spec, state, permittivity, resolution, dt=None):
    if not isinstance(spec, ModeSourceSpec):
        raise TypeError("initialize_mode_state expects a ModeSourceSpec")
    if not isinstance(state, ModeSourceState):
        raise TypeError("initialize_mode_state expects a ModeSourceState")
    return initialize(_ModeSourceAdapter(spec, state), permittivity, resolution, dt=dt)


def _transverse_span(source, axis, ny, nx, resolution):
    if axis == "x":
        return _transverse_bounds(source.center[1], source.width, resolution, ny)
    return _transverse_bounds(source.center[0], source.width, resolution, nx)


def _offset_plane_index(direction, offset_idx, limit):
    if direction.startswith("+"):
        return max(0, offset_idx - 1)
    return min(limit - 2, offset_idx)


def _axis_slice_data(spec, permittivity, *, dx, dy, dz, nx, ny, nz, is_3d):
    axis = spec.direction_axis
    if axis == "x":
        center_idx = _center_index(spec.center[0], dx, nx)
        offset_idx = max(0, center_idx - 1) if spec.direction == "+x" else min(nx - 2, center_idx + 1)
        return axis, center_idx, offset_idx, (
            permittivity[:, :, center_idx] if is_3d else permittivity[:, center_idx]
        )
    if axis == "y":
        center_idx = _center_index(spec.center[1], dy, ny)
        offset_idx = max(0, center_idx - 1) if spec.direction == "+y" else min(ny - 2, center_idx + 1)
        return axis, center_idx, offset_idx, (
            permittivity[:, center_idx, :] if is_3d else permittivity[center_idx, :]
        )
    center_idx = _center_index(spec.center[2], dz, nz)
    offset_idx = max(0, center_idx - 1) if spec.direction == "+z" else min(nz - 2, center_idx + 1)
    return axis, center_idx, offset_idx, permittivity[center_idx, :, :]


def _setup_2d_tm(
    source, e_mode, h_mode, center_idx, offset_idx, axis, ny, nx, resolution, dir_sign, z_target
):
    cfg = _SETUP_2D_TM[axis]
    start, end, transverse_slice = _transverse_span(source, axis, ny, nx, resolution)
    source.state.transverse_start = start
    source.state.transverse_end = end

    if axis == "x":
        ez_indices = (transverse_slice, center_idx)
        h_indices = (transverse_slice, offset_idx)
    else:
        ez_indices = (center_idx, transverse_slice)
        h_indices = (offset_idx, transverse_slice)

    setattr(source, cfg["ez_attr"], ez_indices)
    setattr(source, cfg["h_attr"], h_indices)
    source._h_component = cfg["h_component"]

    h_profile, e_profile = _align_2d_impedance_pair(
        h_mode[cfg["h_mode_index"]],
        e_mode[cfg["e_mode_index"]],
        z_target,
    )
    h_cropped, e_cropped = _crop_window_2d_pair(h_profile, e_profile, start, end)
    first, second = _finalize_2d_launch_pair(
        h_cropped,
        e_cropped,
        sign_h=dir_sign * cfg["field_signs"][0],
        sign_e=dir_sign * cfg["field_signs"][1],
        signed_flux_sign=cfg["flux_sign"],
        resolution=resolution,
    )
    setattr(source, cfg["profile_attrs"][0], first)
    setattr(source, cfg["profile_attrs"][1], second)


def _setup_2d_te(
    source, e_mode, h_mode, center_idx, offset_idx, axis, ny, nx, resolution, dir_sign, z_target
):
    cfg = _SETUP_2D_TE[axis]
    start, end, _ = _transverse_span(source, axis, ny, nx, resolution)
    source.state.transverse_start = start
    source.state.transverse_end = end

    plane_limit = nx if cfg["offset_limit_axis"] == 1 else ny
    hz_plane = _offset_plane_index(source.direction, offset_idx, plane_limit)
    line_limit = (ny - 1) if axis == "x" else (nx - 1)
    transverse_slice = slice(start, min(end, line_limit))

    if axis == "x":
        hz_indices = (transverse_slice, hz_plane)
        e_indices = (transverse_slice, offset_idx)
    else:
        hz_indices = (hz_plane, transverse_slice)
        e_indices = (offset_idx, transverse_slice)

    setattr(source, cfg["hz_attr"], hz_indices)
    setattr(source, cfg["e_attr"], e_indices)
    source._e_component = cfg["e_component"]

    h_profile, e_profile = _stagger_2d_pair(
        np.squeeze(h_mode[cfg["h_mode_index"]]),
        np.squeeze(e_mode[cfg["e_mode_index"]]),
    )
    h_profile, e_profile = _align_2d_impedance_pair(h_profile, e_profile, z_target)
    h_cropped, e_cropped = _crop_window_2d_pair(h_profile, e_profile, start, end)
    first, second = _finalize_2d_launch_pair(
        h_cropped,
        e_cropped,
        sign_h=dir_sign * cfg["field_signs"][0],
        sign_e=dir_sign * cfg["field_signs"][1],
        signed_flux_sign=cfg["flux_sign"],
        resolution=resolution,
    )
    setattr(source, cfg["profile_attrs"][0], first)
    setattr(source, cfg["profile_attrs"][1], second)


def initialize(source, permittivity, resolution, dt=None):
    """Compute the mode and configure source profiles and indices."""
    spec = source.spec
    state = source.state
    dx = dy = resolution
    is_3d = permittivity.ndim == 3
    state.resolution = resolution
    state.is_3d = is_3d

    if is_3d:
        nz, ny, nx = permittivity.shape
        dz = resolution
        state.grid_shape = (nz, ny, nx)
        height = spec.width if spec.height is None else spec.height
    else:
        ny, nx = permittivity.shape
        nz = 1
        dz = resolution
        state.grid_shape = (ny, nx)
        height = None

    axis = spec.direction_axis
    if (not is_3d) and axis == "z":
        raise ValueError(
            "direction '+z'/'-z' requires a 3D permittivity grid; received 2D data"
        )
    state.axis = axis
    state.dt_physical = 0.0
    state.launch_dt = dt
    state.transverse_start = None
    state.transverse_end = None

    axis, center_idx, offset_idx, eps_profile = _axis_slice_data(
        spec,
        permittivity,
        dx=dx,
        dy=dy,
        dz=dz,
        nx=nx,
        ny=ny,
        nz=nz,
        is_3d=is_3d,
    )
    state.eps_profile_2d = eps_profile if is_3d else None

    omega = 2 * math.pi * LIGHT_SPEED / spec.wavelength
    dL = dz if is_3d else (dy if axis == "x" else dx)
    solver_direction = spec.direction
    if is_3d and axis in {"x", "y"}:
        solver_direction = ("-" if spec.direction.startswith("+") else "+") + axis

    eps_profile_arr = to_host(eps_profile)
    n_local_max = math.sqrt(max(float(np.real(eps_profile_arr).max()), 1e-12))
    target_neff = 0.98 * n_local_max

    mode_candidates = 3
    try:
        neff_val, e_fields, h_fields, _ = solve_modes(
            eps=eps_profile,
            omega=omega,
            dL=dL,
            m=mode_candidates,
            direction=solver_direction,
            filter_pol=spec.pol,
            target_neff=target_neff,
            return_fields=True,
        )
    except ValueError:
        neff_val, e_fields, h_fields, _ = solve_modes(
            eps=eps_profile,
            omega=omega,
            dL=dL,
            m=1,
            direction=solver_direction,
            filter_pol=spec.pol,
            target_neff=target_neff,
            return_fields=True,
        )

    mode_idx = _select_core_confined_mode_index(eps_profile, e_fields, neff_val)
    state.neff = neff_val[mode_idx]
    e_mode = e_fields[mode_idx]
    h_mode = h_fields[mode_idx]

    ex_raw = jnp.asarray(jnp.squeeze(e_mode[0]))
    ey_raw = jnp.asarray(jnp.squeeze(e_mode[1]))
    ez_raw = jnp.asarray(jnp.squeeze(e_mode[2]))
    hx_raw = jnp.asarray(jnp.squeeze(h_mode[0]))
    hy_raw = jnp.asarray(jnp.squeeze(h_mode[1]))
    hz_raw = jnp.asarray(jnp.squeeze(h_mode[2]))

    if is_3d:
        ex_raw, ey_raw, ez_raw, hx_raw, hy_raw, hz_raw = _remap_3d_solver_components(
            ex_raw, ey_raw, ez_raw, hx_raw, hy_raw, hz_raw, axis
        )

    if is_3d:
        ref_field = _select_3d_phase_ref(
            axis, spec.pol, ex_raw, ey_raw, ez_raw, hx_raw, hy_raw, hz_raw
        )
    elif spec.pol == "tm":
        ex_max = jnp.max(jnp.abs(ex_raw))
        ey_max = jnp.max(jnp.abs(ey_raw))
        ez_max = jnp.max(jnp.abs(ez_raw))
        ref_field = jnp.where(
            ex_max > ey_max,
            jnp.where(ex_max > ez_max, ex_raw, ez_raw),
            jnp.where(ey_max > ez_max, ey_raw, ez_raw),
        )
    else:
        ref_field = ey_raw if axis == "x" else ex_raw
        ref_field = jnp.where(jnp.max(jnp.abs(ref_field)) < 1e-9, ez_raw, ref_field)

    idx_max = jnp.argmax(jnp.abs(ref_field))
    phase_ref = jnp.angle(ref_field.flatten()[idx_max])

    ex_aligned = ex_raw * jnp.exp(-1j * phase_ref)
    ey_aligned = ey_raw * jnp.exp(-1j * phase_ref)
    ez_aligned = ez_raw * jnp.exp(-1j * phase_ref)
    hx_aligned = hx_raw * jnp.exp(-1j * phase_ref)
    hy_aligned = hy_raw * jnp.exp(-1j * phase_ref)
    hz_aligned = hz_raw * jnp.exp(-1j * phase_ref)

    if is_3d:
        state.impedance_neff = _select_3d_impedance_index(
            axis,
            spec.pol,
            state.eps_profile_2d,
            ex_aligned,
            ey_aligned,
            ez_aligned,
            hx_aligned,
            hy_aligned,
            hz_aligned,
        )
        setup_3d(
            source,
            ex_aligned,
            ey_aligned,
            ez_aligned,
            hx_aligned,
            hy_aligned,
            hz_aligned,
            center_idx,
            offset_idx,
            axis,
            nz,
            ny,
            nx,
            resolution,
            height=height,
            omega=omega,
            dt=dt,
        )
    else:
        state.impedance_neff = None
        setup_2d(
            source, e_mode, h_mode, center_idx, offset_idx, axis, ny, nx, resolution
        )

    compute_dt_physical(source, axis, is_3d, dx, dy, dz, dt=dt)
    state.initialized = True


def setup_3d(
    source,
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    center_idx,
    offset_idx,
    axis,
    nz,
    ny,
    nx,
    resolution,
    height,
    omega,
    dt,
):
    """Set up full 6-component injection for 3D simulations."""
    profiles, indices, extra = _build_3d_profiles(
        ex,
        ey,
        ez,
        hx,
        hy,
        hz,
        axis=axis,
        direction=source.direction,
        center=source.center,
        width=source.width,
        height=height,
        center_idx=center_idx,
        offset_idx=offset_idx,
        grid_shape=(nz, ny, nx),
        resolution=resolution,
        impedance_neff=(
            source.state.impedance_neff
            if source.state.impedance_neff is not None
            else source.state.neff
        ),
        omega=omega,
        dt=dt,
    )

    source._Ex_profile = profiles.get("Ex")
    source._Ey_profile = profiles.get("Ey")
    source._Ez_profile = profiles.get("Ez")
    source._Hx_profile = profiles.get("Hx")
    source._Hy_profile = profiles.get("Hy")
    source._Hz_profile = profiles.get("Hz")

    source._Ex_indices = indices.get("Ex")
    source._Ey_indices = indices.get("Ey")
    source._Ez_indices = indices.get("Ez")
    source._Hx_indices = indices.get("Hx")
    source._Hy_indices = indices.get("Hy")
    source._Hz_indices = indices.get("Hz")

    for key, val in extra.items():
        setattr(source, key, val)

    source._jz_profile = source._Hz_profile
    source._my_profile = source._Ez_profile


def setup_2d(source, e_mode, h_mode, center_idx, offset_idx, axis, ny, nx, resolution):
    """2D injection setup using explicit global component mapping."""
    dir_sign = 1.0 if source.direction.startswith("+") else -1.0
    eta_0 = math.sqrt(MU_0 / EPS_0)
    z_target = eta_0 / max(float(np.real(source._neff)), 1e-6)

    if source.pol == "tm":
        _setup_2d_tm(
            source,
            e_mode,
            h_mode,
            center_idx,
            offset_idx,
            axis,
            ny,
            nx,
            resolution,
            dir_sign,
            z_target,
        )
        return

    _setup_2d_te(
        source,
        e_mode,
        h_mode,
        center_idx,
        offset_idx,
        axis,
        ny,
        nx,
        resolution,
        dir_sign,
        z_target,
    )
def compute_dt_physical(source, axis, is_3d, dx, dy, dz=None, dt=None):
    """Compute physical time shift between E and H injection planes."""
    if source._neff is None:
        return
    if dz is None:
        dz = dx

    coord_e = 0.0
    coord_h = 0.0

    if is_3d:
        e_comp, h_comp = _dominant_3d_pair(axis, source.pol)
        e_indices = getattr(source, f"_{e_comp}_indices", None)
        h_indices = getattr(source, f"_{h_comp}_indices", None)
        e_axis_idx = _axis_index_from_component_indices(e_indices, axis)
        h_axis_idx = _axis_index_from_component_indices(h_indices, axis)
        coord_e = _component_axis_coord(e_comp, e_axis_idx, axis, dx, dy, dz)
        coord_h = _component_axis_coord(h_comp, h_axis_idx, axis, dx, dy, dz)
    else:
        if axis == "x":
            if source.pol == "tm":
                idx_e = source._ez_indices[1] if source._ez_indices else 0
                idx_h = source._h_indices[1] if source._h_indices else 0
            else:
                idx_e = source._e_indices[1] if source._e_indices else 0
                idx_h = source._hz_indices[1] if source._hz_indices else 0
            coord_e = (idx_e + 0.5) * dx
            coord_h = (idx_h + 1.0) * dx
        else:
            if source.pol == "tm":
                idx_e = source._ez_indices[0] if source._ez_indices else 0
                idx_h = source._h_indices[0] if source._h_indices else 0
            else:
                idx_e = source._e_indices[0] if source._e_indices else 0
                idx_h = source._hz_indices[0] if source._hz_indices else 0
            coord_e = (idx_e + 0.5) * dy
            coord_h = (idx_h + 1.0) * dy

    delta_s = float(coord_e - coord_h)
    if is_3d and dt is not None:
        omega = 2 * math.pi * LIGHT_SPEED / source.wavelength
        d_axis = {"x": dx, "y": dy, "z": dz}[axis]
        k_num = _solve_numeric_k_axis(omega, dt, d_axis, source._neff)
        source._dt_physical = _numeric_phase_delay(omega, k_num, delta_s)
    else:
        source._dt_physical = delta_s * float(np.real(source._neff)) / LIGHT_SPEED
