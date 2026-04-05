import math

import jax.numpy as jnp
import numpy as np

from beamz.arrays import to_host
from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.devices.sources.profiles import (
    _axis_index_from_component_indices,
    _build_3d_profiles,
    _component_axis_coord,
    _dominant_3d_pair,
    _impedance_match_e_profile,
    _parse_direction,
    _remap_3d_solver_components,
    _select_3d_impedance_index,
    _select_3d_phase_ref,
    _select_core_confined_mode_index,
    _solve_numeric_k_axis,
    _normalize_2d_pair_by_power,
    _numeric_phase_delay,
    _to_real_profile,
)
from beamz.devices.sources.solve import solve_modes


def _center_index(coord: float, step: float, limit: int) -> int:
    return max(0, min(limit - 1, int(round(coord / step - 0.5))))


def _transverse_bounds(center: float, width: float, resolution: float, limit: int):
    center_idx = int(round(center / resolution))
    half_width_idx = int(round((width / 2) / resolution))
    start = max(0, center_idx - half_width_idx)
    end = min(limit, center_idx + half_width_idx)
    return start, end, slice(start, end)


def _align_impedance_pair(h_field, e_field, z_target):
    h_profile = np.squeeze(h_field)
    e_profile = np.squeeze(e_field)
    idx_max = np.argmax(np.abs(h_profile))
    phase_ref = np.angle(h_profile.flatten()[idx_max])
    h_profile = h_profile * np.exp(-1j * phase_ref)
    e_profile = e_profile * np.exp(-1j * phase_ref)
    return h_profile, _impedance_match_e_profile(e_profile, h_profile, z_target)


def _crop_window_pair(h_profile, e_profile, start: int, end: int):
    stop = min(end, len(h_profile), len(e_profile))
    width_cells = max(0, stop - start)
    window = make_1d_window(width_cells)
    h_cropped = h_profile[start:stop]
    e_cropped = e_profile[start:stop]
    if len(h_cropped) == len(window):
        h_cropped = h_cropped * window
        e_cropped = e_cropped * window
    return h_cropped, e_cropped


def _finalize_2d_pair(h_profile, e_profile, sign_h, sign_e, signed_flux_sign, resolution):
    h_profile = sign_h * h_profile
    e_profile = sign_e * e_profile
    h_profile, e_profile = _normalize_2d_pair_by_power(
        h_profile,
        e_profile,
        signed_flux_sign=signed_flux_sign,
        dl=resolution,
    )
    h_profile = _to_real_profile(h_profile)
    e_profile = _to_real_profile(e_profile)
    return _normalize_2d_pair_by_power(
        h_profile,
        e_profile,
        signed_flux_sign=signed_flux_sign,
        dl=resolution,
    )


def _stagger_pair(h_field, e_field):
    return 0.5 * (h_field[:-1] + h_field[1:]), 0.5 * (e_field[:-1] + e_field[1:])


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

    if axis == "x":
        center_idx = _center_index(spec.center[0], dx, nx)
        if spec.direction == "+x":
            offset_idx = max(0, center_idx - 1)
        else:
            offset_idx = min(nx - 2, center_idx + 1)

        if is_3d:
            eps_profile = permittivity[:, :, center_idx]
            state.eps_profile_2d = eps_profile
        else:
            eps_profile = permittivity[:, center_idx]
            state.eps_profile_2d = None

    elif axis == "y":
        center_idx = _center_index(spec.center[1], dy, ny)
        if spec.direction == "+y":
            offset_idx = max(0, center_idx - 1)
        else:
            offset_idx = min(ny - 2, center_idx + 1)

        if is_3d:
            eps_profile = permittivity[:, center_idx, :]
            state.eps_profile_2d = eps_profile
        else:
            eps_profile = permittivity[center_idx, :]
            state.eps_profile_2d = None

    else:
        center_idx = _center_index(spec.center[2], dz, nz)
        if spec.direction == "+z":
            offset_idx = max(0, center_idx - 1)
        else:
            offset_idx = min(nz - 2, center_idx + 1)

        eps_profile = permittivity[center_idx, :, :]
        state.eps_profile_2d = eps_profile

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

    if axis == "x":
        setup_2d_x(
            source,
            e_mode,
            h_mode,
            center_idx,
            offset_idx,
            ny,
            nx,
            resolution,
            dir_sign,
            z_target,
        )
    else:
        setup_2d_y(
            source,
            e_mode,
            h_mode,
            center_idx,
            offset_idx,
            ny,
            nx,
            resolution,
            dir_sign,
            z_target,
        )


def setup_2d_x(
    source,
    e_mode,
    h_mode,
    center_idx,
    offset_idx,
    ny,
    nx,
    resolution,
    dir_sign,
    z_target,
):
    """2D injection setup for x-propagation."""
    y_start, y_end, y_slice = _transverse_bounds(source.center[1], source.width, resolution, ny)
    source.state.transverse_start = y_start
    source.state.transverse_end = y_end

    if source.pol == "tm":
        source._ez_indices = (y_slice, center_idx)
        source._h_indices = (y_slice, offset_idx)
        source._h_component = "Hx"
        hy_profile, ez_profile = _align_impedance_pair(h_mode[1], e_mode[2], z_target)
        hy_cropped, ez_cropped = _crop_window_pair(hy_profile, ez_profile, y_start, y_end)
        source._jz_profile, source._my_profile = _finalize_2d_pair(
            hy_cropped,
            ez_cropped,
            dir_sign,
            dir_sign,
            -1.0,
            resolution,
        )

    else:
        hz_col = (
            max(0, offset_idx - 1)
            if source.direction == "+x"
            else min(nx - 2, offset_idx)
        )

        source._hz_indices = (slice(y_start, min(y_end, ny - 1)), hz_col)
        source._e_indices = (slice(y_start, min(y_end, ny - 1)), offset_idx)
        source._e_component = "Ey"
        hz_profile, ey_profile = _stagger_pair(np.squeeze(h_mode[2]), np.squeeze(e_mode[1]))
        hz_profile, ey_profile = _align_impedance_pair(hz_profile, ey_profile, z_target)
        hz_cropped, ey_cropped = _crop_window_pair(hz_profile, ey_profile, y_start, y_end)
        source._jy_profile, source._mz_profile = _finalize_2d_pair(
            hz_cropped,
            ey_cropped,
            dir_sign,
            -dir_sign,
            1.0,
            resolution,
        )


def setup_2d_y(
    source,
    e_mode,
    h_mode,
    center_idx,
    offset_idx,
    ny,
    nx,
    resolution,
    dir_sign,
    z_target,
):
    """2D injection setup for y-propagation."""
    x_start, x_end, x_slice = _transverse_bounds(source.center[0], source.width, resolution, nx)
    source.state.transverse_start = x_start
    source.state.transverse_end = x_end

    if source.pol == "tm":
        source._ez_indices = (center_idx, x_slice)
        source._h_indices = (offset_idx, x_slice)
        source._h_component = "Hy"
        hx_profile, ez_profile = _align_impedance_pair(h_mode[1], e_mode[2], z_target)
        hx_cropped, ez_cropped = _crop_window_pair(hx_profile, ez_profile, x_start, x_end)
        source._jz_profile, source._my_profile = _finalize_2d_pair(
            hx_cropped,
            ez_cropped,
            dir_sign,
            -dir_sign,
            1.0,
            resolution,
        )

    else:
        hz_row = (
            max(0, offset_idx - 1)
            if source.direction == "+y"
            else min(ny - 2, offset_idx)
        )

        source._hz_indices = (hz_row, slice(x_start, min(x_end, nx - 1)))
        source._e_indices = (offset_idx, slice(x_start, min(x_end, nx - 1)))
        source._e_component = "Ex"
        hz_profile, ex_profile = _stagger_pair(np.squeeze(h_mode[2]), np.squeeze(e_mode[1]))
        hz_profile, ex_profile = _align_impedance_pair(hz_profile, ex_profile, z_target)
        hz_cropped, ex_cropped = _crop_window_pair(hz_profile, ex_profile, x_start, x_end)
        source._jx_profile, source._mz_profile = _finalize_2d_pair(
            hz_cropped,
            ex_cropped,
            -dir_sign,
            -dir_sign,
            -1.0,
            resolution,
        )


def make_1d_window(width_cells, alpha=0.3):
    """Create a 1D Tukey window for smooth edges."""
    if width_cells > 2:
        from scipy.signal.windows import tukey

        return tukey(width_cells, alpha=alpha)
    return np.ones(max(1, width_cells))


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
