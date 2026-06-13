"""Numerical operations for FDTD field updates: curls, field advancement, material handling on staggered Yee grids."""

import jax.numpy as jnp

from beamz.const import EPS_0, MU_0
from beamz.simulation.yee import (
    sample_voxel_grid_at_component_2d,
    sample_voxel_grid_at_component_3d,
    sample_voxel_grid_at_e_component_3d_centered,
    sample_voxel_grid_at_tm_xy_full_component_2d,
)


def _raise_xy_tm_native_helper_error(function_name: str) -> None:
    raise ValueError(
        f"{function_name} does not handle plane='xy'. Use the native TMz helpers in "
        "beamz.simulation.boundaries "
        "(tm_xy_curl_e_to_h_2d/full_pec_curl_e_to_h_2d_xy and "
        "tm_xy_curl_h_to_e_2d/full_pec_curl_h_to_e_2d_xy)."
    )


def curl_e_to_h_2d(e_fields, resolution, plane="xy"):
    """Compute curl of E-field for H update in 2D on staggered Yee grid for arbitrary plane."""
    # Unpack E-fields based on plane
    if plane == "xy":
        _raise_xy_tm_native_helper_error("curl_e_to_h_2d")

    elif plane == "yz":
        # E = (Ex, Ey, Ez) with ∂/∂x = 0
        # Dimensions (nz, ny). Axis 0=z, Axis 1=y.
        Ex, Ey, Ez = e_fields
        resolution = jnp.asarray(resolution, dtype=Ex.dtype)
        # ∇×E = (∂Ez/∂y - ∂Ey/∂z)x̂ + (∂Ex/∂z)ŷ + (-∂Ex/∂y)ẑ

        # Hx = ∂Ez/∂y - ∂Ey/∂z
        # Ez(nz-1, ny). ∂Ez/∂y -> diff(axis=1). (nz-1, ny-1).
        # Ey(nz, ny-1). ∂Ey/∂z -> diff(axis=0). (nz-1, ny-1).
        curl_ex = jnp.diff(Ez, axis=1) / resolution - jnp.diff(Ey, axis=0) / resolution

        # Hy = ∂Ex/∂z
        # Ex(nz, ny). ∂Ex/∂z -> diff(axis=0). (nz-1, ny).
        curl_ey = jnp.diff(Ex, axis=0) / resolution

        # Hz = -∂Ex/∂y
        # Ex(nz, ny). ∂Ex/∂y -> diff(axis=1). (nz, ny-1).
        curl_ez = -jnp.diff(Ex, axis=1) / resolution

        return curl_ex, curl_ey, curl_ez

    elif plane == "xz":
        # E = (Ex, Ey, Ez) with ∂/∂y = 0
        # Dimensions (nz, nx). Axis 0=z, Axis 1=x.
        Ex, Ey, Ez = e_fields
        resolution = jnp.asarray(resolution, dtype=Ex.dtype)
        # ∇×E = (-∂Ey/∂z)x̂ + (∂Ex/∂z - ∂Ez/∂x)ŷ + (∂Ey/∂x)ẑ

        # Hx = -∂Ey/∂z
        # Ey(nz, nx). ∂Ey/∂z -> diff(axis=0). (nz-1, nx).
        curl_ex = -jnp.diff(Ey, axis=0) / resolution

        # Hy = ∂Ex/∂z - ∂Ez/∂x
        # Ex(nz, nx-1). ∂Ex/∂z -> diff(axis=0). (nz-1, nx-1).
        # Ez(nz-1, nx). ∂Ez/∂x -> diff(axis=1). (nz-1, nx-1).
        curl_ey = jnp.diff(Ex, axis=0) / resolution - jnp.diff(Ez, axis=1) / resolution

        # Hz = ∂Ey/∂x
        # Ey(nz, nx). ∂Ey/∂x -> diff(axis=1). (nz, nx-1).
        curl_ez = jnp.diff(Ey, axis=1) / resolution

        return curl_ex, curl_ey, curl_ez

    raise ValueError(f"Invalid plane: {plane}")


def curl_h_to_e_2d(h_fields, resolution, e_shapes, plane="xy"):
    """Compute curl of H-field for E update in 2D for arbitrary plane."""
    # e_shapes is tuple of shapes for (Ex, Ey, Ez) to handle boundary padding

    if plane == "xy":
        del h_fields, resolution, e_shapes
        _raise_xy_tm_native_helper_error("curl_h_to_e_2d")

    elif plane == "yz":
        # ∂/∂x = 0
        Hx, Hy, Hz = h_fields
        dtype = Hx.dtype
        resolution = jnp.asarray(resolution, dtype=dtype)
        # Ex ~ ∂Hz/∂y - ∂Hy/∂z
        # Ey ~ ∂Hx/∂z
        # Ez ~ -∂Hx/∂y

        # Ex (nz, ny). Hz(nz-1, ny). Hy(nz, ny-1).
        curl_ex = jnp.zeros(e_shapes[0], dtype=dtype)
        curl_ex = curl_ex.at[1:-1, 1:-1].set(
            (Hz[1:-1, 1:] - Hz[1:-1, :-1]) / resolution
            - (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) / resolution
        )

        # Ey ~ ∂Hx/∂z
        # Ey (nz, ny-1). Hx (nz-1, ny-1).
        curl_ey = jnp.zeros(e_shapes[1], dtype=dtype)
        curl_ey = curl_ey.at[1:-1, :].set((Hx[1:, :] - Hx[:-1, :]) / resolution)

        # Ez ~ -∂Hx/∂y
        # Ez (nz-1, ny). Hx (nz-1, ny-1).
        curl_ez = jnp.zeros(e_shapes[2], dtype=dtype)
        curl_ez = curl_ez.at[:, 1:-1].set(-(Hx[:, 1:] - Hx[:, :-1]) / resolution)

        return curl_ex, curl_ey, curl_ez

    elif plane == "xz":
        # ∂/∂y = 0
        Hx, Hy, Hz = h_fields
        dtype = Hx.dtype
        resolution = jnp.asarray(resolution, dtype=dtype)
        # (∇×H)_x = ∂Hz/∂y - ∂Hy/∂z = -∂Hy/∂z
        # (∇×H)_y = ∂Hx/∂z - ∂Hz/∂x
        # (∇×H)_z = ∂Hy/∂x - ∂Hx/∂y(0) = ∂Hy/∂x

        # Ex ~ -∂Hy/∂z
        # Ex (nz, nx-1). Hy (nz-1, nx-1).
        curl_ex = jnp.zeros(e_shapes[0], dtype=dtype)
        curl_ex = curl_ex.at[1:-1, :].set(-(Hy[1:, :] - Hy[:-1, :]) / resolution)

        # Ey ~ ∂Hx/∂z - ∂Hz/∂x
        # Ey (nz, nx). Hx (nz-1, nx). Hz (nz, nx-1).
        curl_ey = jnp.zeros(e_shapes[1], dtype=dtype)
        dHx_dz = (Hx[1:, :] - Hx[:-1, :]) / resolution
        dHz_dx = (Hz[:, 1:] - Hz[:, :-1]) / resolution
        curl_ey = curl_ey.at[1:-1, 1:-1].set(dHx_dz[:, 1:-1] - dHz_dx[1:-1, :])

        # Ez ~ ∂Hy/∂x
        # Ez (nz-1, nx). Hy (nz-1, nx-1).
        curl_ez = jnp.zeros(e_shapes[2], dtype=dtype)
        curl_ez = curl_ez.at[:, 1:-1].set((Hy[:, 1:] - Hy[:, :-1]) / resolution)

        return curl_ex, curl_ey, curl_ez

    raise ValueError(f"Invalid plane: {plane}")


def material_slice_for_e_2d_component(permittivity, conductivity, component, plane):
    """Extract material parameters for a specific E-component in 2D plane."""
    # component: 'x', 'y', or 'z'
    # plane: 'xy', 'yz', 'xz'

    s_mid = slice(1, -1)
    s_all = slice(None)

    if plane == "xy":
        # Grid (ny, nx). Component staggering:
        # Ex (ny, nx-1) - staggered in x
        # Ey (ny-1, nx) - staggered in y
        # Ez (ny+1, nx+1) - native TMz node lattice

        if component == "x":
            # Ex (ny, nx-1). Update region: Ex[1:-1, :] -> (ny-2, nx-1)
            # Material at Ex[i, j] positions: use material[i, j] from permittivity[i, j]
            # So material[1:-1, :-1] gives (ny-2, nx-1) from permittivity(ny, nx)
            # Region [1:-1, :] is used for field/curl, material is pre-sliced to match
            region = (s_mid, s_all)  # [1:-1, :] for field update
        elif component == "y":
            # Ey (ny-1, nx). Update region: Ey[:, 1:-1] -> (ny-1, nx-2)
            # Material at Ey[i, j] positions: use material[i, j] from permittivity[i, j]
            # So material[:-1, 1:-1] gives (ny-1, nx-2) from permittivity(ny, nx)
            # Region [:, 1:-1] is used for field/curl, material is pre-sliced to match
            region = (s_all, s_mid)  # [:, 1:-1] for field update
        elif component == "z":
            # Native TMz Ez spans the full node lattice and is updated everywhere;
            # PEC masking is applied separately to constrained boundary nodes.
            region = (s_all, s_all)

    elif plane == "yz":
        # Grid (nz, ny)
        if component == "x":
            # Ex (nz, ny) - normal to plane
            region = (s_mid, s_mid)
        elif component == "y":
            # Ey (nz, ny-1) - in plane y
            region = (s_mid, s_all)
        elif component == "z":
            # Ez (nz-1, ny)
            region = (s_all, s_mid)

    elif plane == "xz":
        # Grid (nz, nx)
        if component == "x":
            # Ex (nz, nx-1)
            region = (s_all, s_mid)
        elif component == "y":
            # Ey (nz, nx) - normal
            region = (s_mid, s_mid)
        elif component == "z":
            # Ez (nz-1, nx)
            region = (s_mid, s_all)

    field_component = {"x": "Ex", "y": "Ey", "z": "Ez"}[component]
    field_shape = {
        "xy": {
            "Ex": (permittivity.shape[0], permittivity.shape[1] - 1),
            "Ey": (permittivity.shape[0] - 1, permittivity.shape[1]),
            "Ez": (permittivity.shape[0] + 1, permittivity.shape[1] + 1),
        },
        "yz": {
            "Ex": (permittivity.shape[0], permittivity.shape[1]),
            "Ey": (permittivity.shape[0], permittivity.shape[1] - 1),
            "Ez": (permittivity.shape[0] - 1, permittivity.shape[1]),
        },
        "xz": {
            "Ex": (permittivity.shape[0], permittivity.shape[1] - 1),
            "Ey": (permittivity.shape[0], permittivity.shape[1]),
            "Ez": (permittivity.shape[0] - 1, permittivity.shape[1]),
        },
    }[plane][field_component]

    if plane == "xy" and field_component == "Ez":
        eps = sample_voxel_grid_at_tm_xy_full_component_2d(permittivity, "Ez")[region]
        sig = sample_voxel_grid_at_tm_xy_full_component_2d(conductivity, "Ez")[region]
    else:
        eps = sample_voxel_grid_at_component_2d(
            permittivity,
            field_component,
            plane,
            stored_shape=field_shape,
            region=region,
        )
        sig = sample_voxel_grid_at_component_2d(
            conductivity,
            field_component,
            plane,
            stored_shape=field_shape,
            region=region,
        )

    return eps, sig, region


def magnetic_conductivity_terms_2d_full(
    conductivity, permeability, hx_shape, hy_shape, hz_shape, plane
):
    """Compute magnetic conductivity for all H-components in 2D."""
    # sigma_m = sigma * mu * MU_0 / EPS_0
    base_term = conductivity * permeability * MU_0 / EPS_0

    if plane not in {"xy", "yz", "xz"}:
        raise ValueError(f"Invalid plane: {plane}")

    if plane == "xy":
        sigma_m_hx = sample_voxel_grid_at_tm_xy_full_component_2d(base_term, "Hx")
        sigma_m_hy = sample_voxel_grid_at_tm_xy_full_component_2d(base_term, "Hy")
    else:
        sigma_m_hx = sample_voxel_grid_at_component_2d(
            base_term, "Hx", plane, stored_shape=hx_shape
        )
        sigma_m_hy = sample_voxel_grid_at_component_2d(
            base_term, "Hy", plane, stored_shape=hy_shape
        )
    sigma_m_hz = sample_voxel_grid_at_component_2d(
        base_term,
        "Hz",
        plane,
        stored_shape=hz_shape,
    )

    assert sigma_m_hx.shape == hx_shape, (
        f"sigma_m_hx shape mismatch: {sigma_m_hx.shape} vs {hx_shape}"
    )
    assert sigma_m_hy.shape == hy_shape, (
        f"sigma_m_hy shape mismatch: {sigma_m_hy.shape} vs {hy_shape}"
    )
    assert sigma_m_hz.shape == hz_shape, (
        f"sigma_m_hz shape mismatch: {sigma_m_hz.shape} vs {hz_shape}"
    )

    return sigma_m_hx, sigma_m_hy, sigma_m_hz


def curl_e_to_h_3d(ex, ey, ez, resolution):
    """Compute curl of E-field for H update in 3D: ∂H/∂t = -∇×E/μ₀."""
    # Full 3D curl: ∇×E = [(∂Ez/∂y - ∂Ey/∂z)x̂ + (∂Ex/∂z - ∂Ez/∂x)ŷ + (∂Ey/∂x - ∂Ex/∂y)ẑ]
    # Field shapes: Ex(nz, ny, nx-1), Ey(nz, ny-1, nx), Ez(nz-1, ny, nx)
    # H-field shapes: Hx(nz-1, ny-1, nx), Hy(nz-1, ny, nx-1), Hz(nz, ny-1, nx-1)

    # Hx update from x-component: (∇×E)_x = ∂Ez/∂y - ∂Ey/∂z
    # Hx is at (z-1/2, y-1/2, x), need curl at that position
    # Ez is at (z-1/2, y, x), ∂Ez/∂y -> diff along y axis: (nz-1, ny-1, nx)
    # Ey is at (z, y-1/2, x), ∂Ey/∂z -> diff along z axis: (nz-1, ny-1, nx)
    dEz_dy = (ez[:, 1:, :] - ez[:, :-1, :]) / resolution  # (nz-1, ny-1, nx)
    dEy_dz = (ey[1:, :, :] - ey[:-1, :, :]) / resolution  # (nz-1, ny-1, nx)
    curl_ex = dEz_dy - dEy_dz  # (nz-1, ny-1, nx) matches Hx shape

    # Hy update from y-component: (∇×E)_y = ∂Ex/∂z - ∂Ez/∂x
    # Hy is at (z-1/2, y, x-1/2), need curl at that position
    # Ex is at (z, y, x-1/2), ∂Ex/∂z -> diff along z axis: (nz-1, ny, nx-1)
    # Ez is at (z-1/2, y, x), ∂Ez/∂x -> diff along x axis: (nz-1, ny, nx-1)
    dEx_dz = (ex[1:, :, :] - ex[:-1, :, :]) / resolution  # (nz-1, ny, nx-1)
    dEz_dx = (ez[:, :, 1:] - ez[:, :, :-1]) / resolution  # (nz-1, ny, nx-1)
    curl_ey = dEx_dz - dEz_dx  # (nz-1, ny, nx-1) matches Hy shape

    # Hz update from z-component: (∇×E)_z = ∂Ey/∂x - ∂Ex/∂y
    # Hz is at (z, y-1/2, x-1/2), need curl at that position
    # Ey is at (z, y-1/2, x), ∂Ey/∂x -> diff along x axis: (nz, ny-1, nx-1)
    # Ex is at (z, y, x-1/2), ∂Ex/∂y -> diff along y axis: (nz, ny-1, nx-1)
    dEy_dx = (ey[:, :, 1:] - ey[:, :, :-1]) / resolution  # (nz, ny-1, nx-1)
    dEx_dy = (ex[:, 1:, :] - ex[:, :-1, :]) / resolution  # (nz, ny-1, nx-1)
    curl_ez = dEy_dx - dEx_dy  # (nz, ny-1, nx-1) matches Hz shape

    return (curl_ex, curl_ey, curl_ez)


def _adjacent_difference(arr, axis, resolution):
    """Pure local adjacent difference with no embedded boundary semantics."""

    resolution = jnp.asarray(resolution, dtype=arr.dtype)
    moved = jnp.moveaxis(arr, axis, 0)
    diff = (moved[1:] - moved[:-1]) / resolution
    return jnp.moveaxis(diff, 0, axis)


def curl_h_to_e_3d(
    hx,
    hy,
    hz,
    resolution,
    ex_shape=None,
    ey_shape=None,
    ez_shape=None,
    *,
    boundary_views,
):
    """Compute curl of H-field for E update in 3D: ∂E/∂t = ∇×H/(ε₀εᵣ)."""
    # Full 3D curl: ∇×H = [(∂Hz/∂y - ∂Hy/∂z)x̂ + (∂Hx/∂z - ∂Hz/∂x)ŷ + (∂Hy/∂x - ∂Hx/∂y)ẑ]
    # Field shapes: Hx(nz-1, ny-1, nx), Hy(nz-1, ny, nx-1), Hz(nz, ny-1, nx-1)
    # E-field shapes: Ex(nz, ny, nx-1), Ey(nz, ny-1, nx), Ez(nz-1, ny, nx)

    # Determine target shapes from E-field shapes if provided
    if ex_shape is None:
        ex_shape = (hz.shape[0], hz.shape[1] + 1, hz.shape[2])
    if ey_shape is None:
        ey_shape = (hx.shape[0] + 1, hx.shape[1], hx.shape[2])
    if ez_shape is None:
        ez_shape = (hy.shape[0], hy.shape[1], hy.shape[2] + 1)

    curl_hx = _adjacent_difference(
        boundary_views["hz_y"], axis=1, resolution=resolution
    ) - (_adjacent_difference(boundary_views["hy_z"], axis=0, resolution=resolution))

    curl_hy = _adjacent_difference(
        boundary_views["hx_z"], axis=0, resolution=resolution
    ) - (_adjacent_difference(boundary_views["hz_x"], axis=2, resolution=resolution))

    curl_hz = _adjacent_difference(
        boundary_views["hy_x"], axis=2, resolution=resolution
    ) - (_adjacent_difference(boundary_views["hx_y"], axis=1, resolution=resolution))

    # Preserve shape contracts when callers pass explicit target shapes.
    assert curl_hx.shape == ex_shape, (
        f"curl_hx shape mismatch: {curl_hx.shape} vs {ex_shape}"
    )
    assert curl_hy.shape == ey_shape, (
        f"curl_hy shape mismatch: {curl_hy.shape} vs {ey_shape}"
    )
    assert curl_hz.shape == ez_shape, (
        f"curl_hz shape mismatch: {curl_hz.shape} vs {ez_shape}"
    )

    return (curl_hx, curl_hy, curl_hz)


def fused_update_h_lossless_3d(
    ex, ey, ez, hx, hy, hz, h_src_ll_x, h_src_ll_y, h_src_ll_z, resolution
):
    """H_new = H_old - source_lossless * curl_E (no intermediate curl arrays)."""
    inv_res = jnp.asarray(1.0, dtype=hx.dtype) / jnp.asarray(resolution, dtype=hx.dtype)
    hx = (
        hx
        - h_src_ll_x
        * ((ez[:, 1:, :] - ez[:, :-1, :]) - (ey[1:, :, :] - ey[:-1, :, :]))
        * inv_res
    )
    hy = (
        hy
        - h_src_ll_y
        * ((ex[1:, :, :] - ex[:-1, :, :]) - (ez[:, :, 1:] - ez[:, :, :-1]))
        * inv_res
    )
    hz = (
        hz
        - h_src_ll_z
        * ((ey[:, :, 1:] - ey[:, :, :-1]) - (ex[:, 1:, :] - ex[:, :-1, :]))
        * inv_res
    )
    return hx, hy, hz


def fused_update_h_lossy_3d(
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    h_decay_x,
    h_src_x,
    h_decay_y,
    h_src_y,
    h_decay_z,
    h_src_z,
    resolution,
):
    """H_new = decay * H_old - source * curl_E (no intermediate curl arrays)."""
    inv_res = jnp.asarray(1.0, dtype=hx.dtype) / jnp.asarray(resolution, dtype=hx.dtype)
    curl_ex = (
        (ez[:, 1:, :] - ez[:, :-1, :]) - (ey[1:, :, :] - ey[:-1, :, :])
    ) * inv_res
    hx = h_decay_x * hx - h_src_x * curl_ex
    curl_ey = (
        (ex[1:, :, :] - ex[:-1, :, :]) - (ez[:, :, 1:] - ez[:, :, :-1])
    ) * inv_res
    hy = h_decay_y * hy - h_src_y * curl_ey
    curl_ez = (
        (ey[:, :, 1:] - ey[:, :, :-1]) - (ex[:, 1:, :] - ex[:, :-1, :])
    ) * inv_res
    hz = h_decay_z * hz - h_src_z * curl_ez
    return hx, hy, hz


def fused_update_h_lossy_3d_material(
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    h_sigma_m_x,
    h_sigma_m_y,
    h_sigma_m_z,
    dt,
    resolution,
):
    """H_new from sigma_m grids without persistent dense decay/source grids."""
    one = jnp.asarray(1.0, dtype=hx.dtype)
    half = jnp.asarray(0.5, dtype=hx.dtype)
    inv_res = one / jnp.asarray(resolution, dtype=hx.dtype)
    dt_over_mu0 = jnp.asarray(dt, dtype=hx.dtype) / jnp.asarray(MU_0, dtype=hx.dtype)
    half_dt_over_mu0 = half * dt_over_mu0

    alpha_x = h_sigma_m_x * half_dt_over_mu0
    denom_x = one + alpha_x
    hx = (one - alpha_x) / denom_x * hx - (dt_over_mu0 / denom_x) * (
        (ez[:, 1:, :] - ez[:, :-1, :]) - (ey[1:, :, :] - ey[:-1, :, :])
    ) * inv_res

    alpha_y = h_sigma_m_y * half_dt_over_mu0
    denom_y = one + alpha_y
    hy = (one - alpha_y) / denom_y * hy - (dt_over_mu0 / denom_y) * (
        (ex[1:, :, :] - ex[:-1, :, :]) - (ez[:, :, 1:] - ez[:, :, :-1])
    ) * inv_res

    alpha_z = h_sigma_m_z * half_dt_over_mu0
    denom_z = one + alpha_z
    hz = (one - alpha_z) / denom_z * hz - (dt_over_mu0 / denom_z) * (
        (ey[:, :, 1:] - ey[:, :, :-1]) - (ex[:, 1:, :] - ex[:, :-1, :])
    ) * inv_res

    return hx, hy, hz


def fused_update_e_lossless_3d(
    hx,
    hy,
    hz,
    ex,
    ey,
    ez,
    e_src_ll_x,
    e_src_ll_y,
    e_src_ll_z,
    resolution,
    *,
    boundary_views,
):
    """E_new = E_old + source_lossless * curl_H (inline pad, no named curl arrays)."""
    ex = ex + e_src_ll_x * (
        _adjacent_difference(boundary_views["hz_y"], axis=1, resolution=resolution)
        - _adjacent_difference(boundary_views["hy_z"], axis=0, resolution=resolution)
    )
    ey = ey + e_src_ll_y * (
        _adjacent_difference(boundary_views["hx_z"], axis=0, resolution=resolution)
        - _adjacent_difference(boundary_views["hz_x"], axis=2, resolution=resolution)
    )
    ez = ez + e_src_ll_z * (
        _adjacent_difference(boundary_views["hy_x"], axis=2, resolution=resolution)
        - _adjacent_difference(boundary_views["hx_y"], axis=1, resolution=resolution)
    )
    return ex, ey, ez


def fused_update_e_lossy_3d_material(
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
    dt,
    resolution,
    *,
    boundary_views,
):
    """E_new from sigma/epsilon grids without persistent dense decay/source grids."""
    one = jnp.asarray(1.0, dtype=ex.dtype)
    half = jnp.asarray(0.5, dtype=ex.dtype)
    dt_over_eps0 = jnp.asarray(dt, dtype=ex.dtype) / jnp.asarray(EPS_0, dtype=ex.dtype)
    half_dt_over_eps0 = half * dt_over_eps0

    beta_x = e_conductivity_x * half_dt_over_eps0 * e_inv_permittivity_x
    denom_x = one + beta_x
    curl_hx = _adjacent_difference(
        boundary_views["hz_y"], axis=1, resolution=resolution
    ) - (_adjacent_difference(boundary_views["hy_z"], axis=0, resolution=resolution))
    ex = (one - beta_x) / denom_x * ex + (
        dt_over_eps0 * e_inv_permittivity_x
    ) / denom_x * curl_hx

    beta_y = e_conductivity_y * half_dt_over_eps0 * e_inv_permittivity_y
    denom_y = one + beta_y
    curl_hy = _adjacent_difference(
        boundary_views["hx_z"], axis=0, resolution=resolution
    ) - (_adjacent_difference(boundary_views["hz_x"], axis=2, resolution=resolution))
    ey = (one - beta_y) / denom_y * ey + (
        dt_over_eps0 * e_inv_permittivity_y
    ) / denom_y * curl_hy

    beta_z = e_conductivity_z * half_dt_over_eps0 * e_inv_permittivity_z
    denom_z = one + beta_z
    curl_hz = _adjacent_difference(
        boundary_views["hy_x"], axis=2, resolution=resolution
    ) - (_adjacent_difference(boundary_views["hx_y"], axis=1, resolution=resolution))
    ez = (one - beta_z) / denom_z * ez + (
        dt_over_eps0 * e_inv_permittivity_z
    ) / denom_z * curl_hz

    return ex, ey, ez


def fused_update_e_lossless_3d_inv_permittivity(
    hx,
    hy,
    hz,
    ex,
    ey,
    ez,
    e_inv_permittivity_x,
    e_inv_permittivity_y,
    e_inv_permittivity_z,
    dt,
    resolution,
    *,
    boundary_views,
):
    """E_new = E_old + dt/(eps0*eps_r) * curl_H without dense source grids."""
    dt_over_eps0 = jnp.asarray(dt, dtype=ex.dtype) / jnp.asarray(EPS_0, dtype=ex.dtype)
    ex = ex + dt_over_eps0 * e_inv_permittivity_x * (
        _adjacent_difference(boundary_views["hz_y"], axis=1, resolution=resolution)
        - _adjacent_difference(boundary_views["hy_z"], axis=0, resolution=resolution)
    )
    ey = ey + dt_over_eps0 * e_inv_permittivity_y * (
        _adjacent_difference(boundary_views["hx_z"], axis=0, resolution=resolution)
        - _adjacent_difference(boundary_views["hz_x"], axis=2, resolution=resolution)
    )
    ez = ez + dt_over_eps0 * e_inv_permittivity_z * (
        _adjacent_difference(boundary_views["hy_x"], axis=2, resolution=resolution)
        - _adjacent_difference(boundary_views["hx_y"], axis=1, resolution=resolution)
    )

    return ex, ey, ez


def fused_update_e_lossy_3d(
    hx,
    hy,
    hz,
    ex,
    ey,
    ez,
    e_decay_x,
    e_src_x,
    e_decay_y,
    e_src_y,
    e_decay_z,
    e_src_z,
    resolution,
    *,
    boundary_views,
):
    """E_new = decay * E_old + source * curl_H (inline pad, no named curl arrays)."""
    curl_hx = _adjacent_difference(
        boundary_views["hz_y"], axis=1, resolution=resolution
    ) - (_adjacent_difference(boundary_views["hy_z"], axis=0, resolution=resolution))
    ex = e_decay_x * ex + e_src_x * curl_hx
    curl_hy = _adjacent_difference(
        boundary_views["hx_z"], axis=0, resolution=resolution
    ) - (_adjacent_difference(boundary_views["hz_x"], axis=2, resolution=resolution))
    ey = e_decay_y * ey + e_src_y * curl_hy
    curl_hz = _adjacent_difference(
        boundary_views["hy_x"], axis=2, resolution=resolution
    ) - (_adjacent_difference(boundary_views["hx_y"], axis=1, resolution=resolution))
    ez = e_decay_z * ez + e_src_z * curl_hz
    return ex, ey, ez


def magnetic_conductivity_terms_3d(
    conductivity, permeability, hx_shape, hy_shape, hz_shape
):
    """Compute magnetic conductivity σ_m = σ * μ₀μᵣ/ε₀ for H-field PML absorption in 3D."""
    if conductivity.ndim < 3:
        return (jnp.zeros(hx_shape), jnp.zeros(hy_shape), jnp.zeros(hz_shape))
    sigma_base = conductivity * permeability * MU_0 / EPS_0
    sigma_m_hx = sample_voxel_grid_at_component_3d(
        sigma_base, "Hx", stored_shape=hx_shape
    )
    sigma_m_hy = sample_voxel_grid_at_component_3d(
        sigma_base, "Hy", stored_shape=hy_shape
    )
    sigma_m_hz = sample_voxel_grid_at_component_3d(
        sigma_base, "Hz", stored_shape=hz_shape
    )
    return (sigma_m_hx, sigma_m_hy, sigma_m_hz)


def advance_h_field(field, curl, sigma_m, dt):
    """Advance H-field one time step via Crank-Nicolson: ∂H/∂t = -∇×E/μ₀ - σ_m*H/μ₀.

    FUNCTIONAL version - returns NEW array instead of mutating input.
    """
    # Faraday's law with magnetic loss: μ₀∂H/∂t = -∇×E - σ_m*H
    # Crank-Nicolson (implicit midpoint): H^(n+1) = [(1 - α)/(1 + α)]H^n - [Δt/μ₀/(1 + α)]∇×E^(n+1/2)
    # where α = σ_m*Δt/(2μ₀) ensures second-order accuracy and unconditional stability
    denom = 1.0 + sigma_m * (dt / (2.0 * MU_0))  # Denominator: 1 + α
    factor = (1.0 - sigma_m * (dt / (2.0 * MU_0))) / denom
    source_coeff = (dt / MU_0) / denom
    # Return NEW array (functional style for JAX)
    return field * factor - source_coeff * curl


def advance_e_field(field, curl, conductivity, permittivity, dt, region):
    """Advance E-field one time step via Crank-Nicolson: ∂E/∂t = ∇×H/(ε₀εᵣ) - σE/(ε₀εᵣ).

    FUNCTIONAL version - returns NEW array using .at[].set() for indexed updates.
    """
    # Ampere's law with electric loss: ε₀εᵣ∂E/∂t = ∇×H - σE
    # Crank-Nicolson: E^(n+1) = [(1 - β)/(1 + β)]E^n + [Δt/(ε₀εᵣ)/(1 + β)]∇×H^(n+1/2)
    # where β = σΔt/(2ε₀εᵣ) for stability and second-order temporal accuracy
    # Note: conductivity and permittivity are already sliced to the interior region
    denom = 1.0 + conductivity * (
        dt / (2.0 * EPS_0 * permittivity)
    )  # Denominator: 1 + β
    factor = (1.0 - conductivity * (dt / (2.0 * EPS_0 * permittivity))) / denom
    source = (dt / (EPS_0 * permittivity)) / denom

    # Compute new values for the interior region
    new_values = field[region] * factor + source * curl[region]

    # Use .at[].set() for functional update (JAX immutable arrays)
    return field.at[region].set(new_values)


def precompute_h_update_coefficients(sigma_m, dt):
    """Precompute static H update coefficients for dense in-loop updates."""
    denom = 1.0 + sigma_m * (dt / (2.0 * MU_0))
    decay = (1.0 - sigma_m * (dt / (2.0 * MU_0))) / denom
    source = (dt / MU_0) / denom
    source_lossless = jnp.full_like(sigma_m, dt / MU_0, dtype=jnp.float32)
    return decay.astype(jnp.float32), source.astype(jnp.float32), source_lossless


def precompute_e_update_coefficients(shape, conductivity, permittivity, dt, region):
    """Precompute full-grid E update coefficients with boundary-safe masks."""
    dtype = jnp.float32
    decay = jnp.ones(shape, dtype=dtype)
    source = jnp.zeros(shape, dtype=dtype)
    source_lossless = jnp.zeros(shape, dtype=dtype)

    denom = 1.0 + conductivity * (dt / (2.0 * EPS_0 * permittivity))
    local_decay = (1.0 - conductivity * (dt / (2.0 * EPS_0 * permittivity))) / denom
    local_source = (dt / (EPS_0 * permittivity)) / denom
    local_source_lossless = dt / (EPS_0 * permittivity)

    decay = decay.at[region].set(local_decay.astype(dtype))
    source = source.at[region].set(local_source.astype(dtype))
    source_lossless = source_lossless.at[region].set(
        local_source_lossless.astype(dtype)
    )
    return decay, source, source_lossless


def material_slice_for_e_3d(permittivity, conductivity, orientation):
    """Extract material parameters at staggered Yee grid positions for E-field components in 3D."""
    # Each E-field component lives at different staggered positions on Yee grid:
    # Ex at (z, y, x-1/2), Ey at (z, y-1/2, x), Ez at (z-1/2, y, x)
    # Only the staggered axis is reduced in size for each component. The other two
    # axes represent valid Yee locations all the way to the metallic walls and must
    # therefore be updated explicitly; metallic boundary enforcement happens
    # separately via masks after the update step.
    component = {"x": "Ex", "y": "Ey", "z": "Ez"}[orientation]
    f_region = (slice(None), slice(None), slice(None))
    eps = sample_voxel_grid_at_e_component_3d_centered(permittivity, component)
    sig = sample_voxel_grid_at_e_component_3d_centered(conductivity, component)
    return eps, sig, f_region
