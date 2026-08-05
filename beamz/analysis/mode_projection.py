import numpy as np

from beamz.analysis.data import AnalysisData
from beamz.analysis.modal_projection.colocation import (
    _discrete_mode_projection_grids_3d,
)
from beamz.analysis.modal_projection.diagnostics import (
    _modal_projection_reconstruction_diagnostics_from_matrix,
)
from beamz.analysis.modal_projection.geometry import (
    _analysis_plane_sample_area,
    _modal_projection_plane_delay_s,
    _mode_components_for_port,
    _monitor_analysis_plane_3d,
)
from beamz.devices._placement import (
    line_region_points,
    snap_axis_aligned_line_region_grid,
    snap_plane_region_grid,
)
from beamz.devices.modes.discrete import DISCRETE_MODE_CONTRACT, solve_beamz_mode
from beamz.devices.modes.fields import _modal_overlap, _normalize_profiles
from beamz.devices.modes.plane import solve_mode_plane_3d, solve_modes
from beamz.devices.monitors.monitors import ModeMonitor


def _material_arrays(sim):
    if not isinstance(sim, AnalysisData):
        raise TypeError("Modal projection requires AnalysisData.")
    region = sim.materials
    if region is None:
        return None
    return (
        np.asarray(region.permittivity),
        np.asarray(region.permeability),
        tuple(int(v) for v in region.origin),
        tuple(int(v) for v in region.full_shape),
    )


def _monitor_profile_slice(sim, monitor, axis, pad_cells):
    materials = _material_arrays(sim)
    if materials is None:
        raise RuntimeError(
            "Detached modal analysis requires a monitor material region or "
            "store_full_materials=True."
        )
    perm, _mu, origin, full_shape = materials
    if perm.ndim == 3:
        z_idx, y_idx, x_idx = monitor.get_grid_slice_3d(
            sim.resolution,
            sim.resolution,
            sim.resolution,
            full_shape,
        )

        def _clamp(idx, limit):
            if isinstance(idx, slice):
                start = 0 if idx.start is None else int(idx.start)
                stop = limit if idx.stop is None else int(idx.stop)
                start = max(0, min(start, max(limit - 1, 0)))
                stop = max(start + 1, min(stop, limit))
                return slice(start, stop)
            ii = int(idx)
            return max(0, min(ii, limit - 1))

        z_idx = _clamp(z_idx, full_shape[0])
        y_idx = _clamp(y_idx, full_shape[1])
        x_idx = _clamp(x_idx, full_shape[2])
        global_indices = (z_idx, y_idx, x_idx)
        local_indices = []
        for idx, offset in zip(global_indices, origin, strict=True):
            if isinstance(idx, slice):
                local_indices.append(
                    slice(
                        None if idx.start is None else int(idx.start) - offset,
                        None if idx.stop is None else int(idx.stop) - offset,
                    )
                )
            else:
                local_indices.append(int(idx) - offset)
        z_idx, y_idx, x_idx = local_indices
        eps_slice = np.asarray(perm[z_idx, y_idx, x_idx], dtype=np.complex128)
        if eps_slice.ndim != 2:
            eps_slice = np.atleast_2d(eps_slice)
        npts = int(eps_slice.size)
        local_idx = np.arange(npts, dtype=int)
        d_area = float(sim.resolution) * float(sim.resolution)
        return eps_slice, local_idx, d_area
    if perm.ndim != 2:
        raise NotImplementedError("Modal extraction supports 2D or 3D only.")
    grid = sim.coordinates.grid
    if grid is not None and grid.metric_kind_for(("x", "y")) != "isotropic_uniform":
        snapped = snap_axis_aligned_line_region_grid(monitor.start, monitor.end, grid)
        if snapped is None:
            raise ValueError(f"Monitor '{monitor.name}' is not axis aligned.")
        points = line_region_points(snapped)
        if not points:
            raise ValueError(f"Monitor '{monitor.name}' contains no sample points.")
        if axis == "x":
            plane = int(snapped.plane_index)
            eps_profile_full = perm[:, plane - origin[1]]
            sample_idx = np.asarray(
                [point[1] - origin[0] for point in points], dtype=int
            )
            transverse_axis = "y"
        else:
            plane = int(snapped.plane_index)
            eps_profile_full = perm[plane - origin[0], :]
            sample_idx = np.asarray(
                [point[0] - origin[1] for point in points], dtype=int
            )
            transverse_axis = "x"
        lo = max(0, int(np.min(sample_idx)) - int(pad_cells))
        hi = min(len(eps_profile_full), int(np.max(sample_idx)) + int(pad_cells) + 1)
        local_idx = np.clip(sample_idx - lo, 0, max(hi - lo - 1, 0))
        edges = np.asarray(grid.axis_edges(transverse_axis))[lo : hi + 1]
        weights = np.asarray(grid.cell_widths(transverse_axis))[
            sample_idx + (origin[0] if axis == "x" else origin[1])
        ]
        return (
            np.asarray(eps_profile_full[lo:hi], dtype=np.complex128),
            local_idx,
            weights,
            edges,
        )
    points = monitor.get_grid_points_2d(sim.resolution, sim.resolution)
    if not points:
        raise ValueError(f"Monitor '{monitor.name}' contains no sample points.")
    p = np.asarray(points, dtype=float)
    if axis == "x":
        x_idx = int(np.clip(round(float(np.mean(p[:, 0]))), 0, full_shape[1] - 1))
        eps_profile_full = perm[:, x_idx - origin[1]]
        sample_idx = np.asarray(
            [int(np.clip(pi[1], 0, full_shape[0] - 1)) - origin[0] for pi in points],
            dtype=int,
        )
    else:
        y_idx = int(np.clip(round(float(np.mean(p[:, 1]))), 0, full_shape[0] - 1))
        eps_profile_full = perm[y_idx - origin[0], :]
        sample_idx = np.asarray(
            [int(np.clip(pi[0], 0, full_shape[1] - 1)) - origin[1] for pi in points],
            dtype=int,
        )
    lo = max(0, int(np.min(sample_idx)) - int(pad_cells))
    hi = min(len(eps_profile_full), int(np.max(sample_idx)) + int(pad_cells) + 1)
    local_idx = np.clip(sample_idx - lo, 0, max(hi - lo - 1, 0))
    if len(points) > 1:
        step_idx = np.diff(np.asarray(points, dtype=float), axis=0)
        dl = float(np.mean(np.linalg.norm(step_idx, axis=1))) * float(sim.resolution)
    else:
        dl = float(sim.resolution)
    dl = max(dl, float(sim.resolution) * 1e-9)
    return (
        np.asarray(eps_profile_full[lo:hi], dtype=np.complex128),
        local_idx,
        dl,
        None,
    )


def _build_discrete_port_projection_3d(
    sim,
    *,
    spec,
    monitor,
    frequency,
    parts,
    direction_sign,
    analysis_coords0,
    analysis_coords1,
):
    if not isinstance(monitor, ModeMonitor):
        raise TypeError("3D modal projection requires a canonical ModeMonitor.")
    materials = _material_arrays(sim)
    if materials is None:
        raise RuntimeError(
            "3D modal projection requires monitor-local materials or "
            "store_full_materials=True."
        )
    perm, permeability, origin, full_shape = materials
    if perm.ndim != 3:
        raise ValueError("The discrete 3D modal contract requires 3D materials.")

    axis = parts["axis"]
    axis_index = {"z": 0, "y": 1, "x": 2}[axis]
    grid = sim.coordinates.grid
    if grid is not None and grid.metric_kind != "isotropic_uniform":
        snapped_region = snap_plane_region_grid(
            center=monitor.center,
            size=monitor.size,
            plane_normal=axis,
            grid=grid,
        )
        plane_index = int(snapped_region.plane_index)
    else:
        z_idx, y_idx, x_idx = monitor.get_grid_slice_3d(
            sim.resolution,
            sim.resolution,
            sim.resolution,
            full_shape,
        )
        normal_index = {"z": z_idx, "y": y_idx, "x": x_idx}[axis]
        if isinstance(normal_index, slice):
            start = 0 if normal_index.start is None else int(normal_index.start)
            stop = (
                full_shape[axis_index]
                if normal_index.stop is None
                else int(normal_index.stop)
            )
            plane_index = int(
                np.clip(
                    (start + max(start + 1, stop) - 1) // 2,
                    0,
                    full_shape[axis_index] - 1,
                )
            )
        else:
            plane_index = int(np.clip(int(normal_index), 0, full_shape[axis_index] - 1))
        snapped_region = monitor.get_snapped_region(
            dx=float(sim.resolution),
            dy=float(sim.resolution),
            dz=float(sim.resolution),
            field_shape=full_shape,
        )
    if direction_sign > 0.0:
        offset_index = max(0, plane_index - 1)
    else:
        offset_index = min(max(full_shape[axis_index] - 2, 0), plane_index + 1)

    mode_spec = monitor.mode_spec
    num_modes = int(
        max(
            int(mode_spec.num_modes or 0),
            int(spec.mode_index) + 1,
        )
    )
    center = tuple(float(value) for value in monitor.center)
    size = tuple(float(value) for value in monitor.size_spec)
    if axis == "x":
        width, height = size[1], size[2]
    elif axis == "y":
        width, height = size[0], size[2]
    else:
        width, height = size[0], size[1]

    discrete_mode = solve_mode_plane_3d(
        perm,
        permeability,
        frequency=frequency,
        resolution=sim.resolution,
        dt=sim.dt,
        axis=axis,
        grid_shape=full_shape,
        center=center,
        width=width,
        height=height,
        plane_index=plane_index,
        offset_index=offset_index,
        direction=spec.projection_direction,
        mode_index=spec.mode_index,
        polarization=str(spec.polarization).lower(),
        target_neff=mode_spec.target_neff,
        num_modes=num_modes,
        snapped_region=snapped_region,
        material_origin_zyx=origin,
        solver=solve_beamz_mode,
        grid=grid
        if grid is not None and grid.metric_kind != "isotropic_uniform"
        else None,
    )
    proj_components = tuple(parts.get("projection_components_3d", ()))
    if not proj_components:
        raise ValueError(f"Port axis {axis!r} has no 3D projection components.")
    d_area = _analysis_plane_sample_area(
        analysis_coords0,
        analysis_coords1,
        float(sim.resolution),
    )
    if grid is not None and grid.metric_kind != "isotropic_uniform":
        transverse_axes = {
            "x": ("z", "y"),
            "y": ("z", "x"),
            "z": ("y", "x"),
        }[axis]
        interval0 = snapped_region.axis_interval(transverse_axes[0])
        interval1 = snapped_region.axis_interval(transverse_axes[1])
        weights0 = np.asarray(grid.cell_widths(transverse_axes[0]))[
            int(interval0.start) : int(interval0.stop)
        ]
        weights1 = np.asarray(grid.cell_widths(transverse_axes[1]))[
            int(interval1.start) : int(interval1.stop)
        ]
        integration_weights = (weights0[:, None] * weights1[None, :]).reshape(-1)
    else:
        integration_weights = float(d_area)
    _, plus_components = _discrete_mode_projection_grids_3d(
        sim,
        discrete_mode,
        discrete_mode.backward_profiles,
        monitor=monitor,
        axis=axis,
        components=proj_components,
        analysis_coords0=analysis_coords0,
        analysis_coords1=analysis_coords1,
    )
    _, minus_components = _discrete_mode_projection_grids_3d(
        sim,
        discrete_mode,
        discrete_mode.profiles,
        monitor=monitor,
        axis=axis,
        components=proj_components,
        analysis_coords0=analysis_coords0,
        analysis_coords1=analysis_coords1,
    )
    if any(
        name not in plus_components or name not in minus_components
        for name in proj_components
    ):
        missing_components = [
            name
            for name in proj_components
            if name not in plus_components or name not in minus_components
        ]
        raise RuntimeError(
            "Discrete mode is missing projection components: "
            + ", ".join(missing_components)
        )

    plus_components = {
        name: np.asarray(plus_components[name], dtype=np.complex128)
        for name in proj_components
    }
    minus_components = {
        name: np.asarray(minus_components[name], dtype=np.complex128)
        for name in proj_components
    }
    _normalize_profiles(
        plus_components,
        axis=axis,
        measure=integration_weights,
        direction_sign=float(direction_sign),
        max_scale=1e6,
    )
    _normalize_profiles(
        minus_components,
        axis=axis,
        measure=integration_weights,
        direction_sign=float(direction_sign),
        max_scale=1e6,
    )
    overlap_matrix = np.asarray(
        [
            [
                _modal_overlap(
                    plus_components,
                    plus_components,
                    axis,
                    integration_weights,
                    direction_sign=direction_sign,
                ),
                _modal_overlap(
                    plus_components,
                    minus_components,
                    axis,
                    integration_weights,
                    direction_sign=direction_sign,
                ),
            ],
            [
                _modal_overlap(
                    minus_components,
                    plus_components,
                    axis,
                    integration_weights,
                    direction_sign=direction_sign,
                ),
                _modal_overlap(
                    minus_components,
                    minus_components,
                    axis,
                    integration_weights,
                    direction_sign=direction_sign,
                ),
            ],
        ],
        dtype=np.complex128,
    )
    mode_neff = float(np.real(np.asarray(discrete_mode.neff)))
    if not np.isfinite(mode_neff) or mode_neff <= 0.0:
        raise RuntimeError(
            f"Discrete mode returned invalid effective index {mode_neff!r}."
        )
    projection = {
        "e_component": parts["e_component"],
        "h_component": parts["h_component"],
        "components": tuple(proj_components),
        "condition_number": float(np.linalg.cond(overlap_matrix)),
        "mode_neff": mode_neff,
        "mode_components": {
            name: np.asarray(plus_components[name], dtype=np.complex128)
            for name in proj_components
        },
        "mode_components_bwd": {
            name: np.asarray(minus_components[name], dtype=np.complex128)
            for name in proj_components
        },
        "overlap_matrix": overlap_matrix,
        "axis": axis,
        "direction_sign": float(direction_sign),
        "d_area": float(d_area),
        "integration_weights": integration_weights,
        "discrete_contract": DISCRETE_MODE_CONTRACT,
        "analysis_coords0": np.asarray(analysis_coords0, dtype=np.float64),
        "analysis_coords1": np.asarray(analysis_coords1, dtype=np.float64),
    }
    projection["modal_plane_delay_s"] = _modal_projection_plane_delay_s(
        sim,
        spec,
        frequency,
        projection["mode_neff"],
    )
    return projection


def _build_port_projection_2d(
    sim,
    *,
    spec,
    monitor,
    frequency,
    parts,
    mode_pad_cells,
):
    profile_slice = _monitor_profile_slice(sim, monitor, parts["axis"], mode_pad_cells)
    if len(profile_slice) == 3:
        eps_profile, local_idx, dl = profile_slice
        grid_edges = None
    else:
        eps_profile, local_idx, dl, grid_edges = profile_slice
    eps_profile = np.asarray(eps_profile, dtype=np.complex128)
    target_neff = 0.98 * np.sqrt(max(float(np.max(np.real(eps_profile))), 1e-12))
    mode_count = int(spec.mode_index) + 1
    neffs, e_fields, h_fields, _ = solve_modes(
        eps=eps_profile,
        omega=2.0 * np.pi * float(frequency),
        dL=float(sim.resolution),
        m=mode_count,
        direction=spec.projection_direction,
        filter_pol=spec.polarization,
        target_neff=target_neff,
        return_fields=True,
        **({"grid_edges": (grid_edges,)} if grid_edges is not None else {}),
    )
    if len(neffs) <= int(spec.mode_index):
        raise ValueError(
            f"Mode solver returned {len(neffs)} modes for requested "
            f"mode_index={spec.mode_index} on port {spec.name!r}."
        )

    mode_index = int(spec.mode_index)
    e_profile = np.asarray(
        np.squeeze(e_fields[mode_index][parts["e_mode_index"]]),
        dtype=np.complex128,
    )
    h_profile = np.asarray(
        np.squeeze(h_fields[mode_index][parts["h_mode_index"]]),
        dtype=np.complex128,
    )
    if e_profile.ndim > 1:
        e_profile = e_profile[:, 0]
    if h_profile.ndim > 1:
        h_profile = h_profile[:, 0]
    e_profile = e_profile[local_idx]
    h_profile = h_profile[local_idx]

    measure = np.asarray(dl, dtype=float)
    weights = measure if measure.ndim == 0 else measure.reshape(-1)[: e_profile.size]
    power = 0.5 * np.real(
        np.sum(
            weights * parts["signed_flux_sign"] * e_profile * np.conjugate(h_profile)
        )
    )
    normalization = np.sqrt(max(abs(power), 1e-30))
    e_forward = e_profile / normalization
    h_forward = h_profile / normalization
    mode_matrix = np.column_stack(
        [
            np.concatenate([e_forward, h_forward]),
            np.concatenate([e_forward, -h_forward]),
        ]
    )
    if np.ndim(weights) == 0:
        # Preserve the established uniform-grid solve exactly.  Multiplying both
        # sides by a tiny constant measure is algebraically redundant, but it can
        # move the SVD pseudoinverse across a numerical rank threshold because E
        # and H use very different SI scales.
        projection_weights = None
        pinv = np.linalg.pinv(mode_matrix)
    else:
        projection_weights = np.concatenate(
            [
                np.broadcast_to(np.sqrt(weights), e_forward.shape),
                np.broadcast_to(np.sqrt(weights), h_forward.shape),
            ]
        )
        pinv = np.linalg.pinv(projection_weights[:, None] * mode_matrix)
    projection = {
        "e_component": parts["e_component"],
        "h_component": parts["h_component"],
        "components": (parts["e_component"], parts["h_component"]),
        "mode_matrix": mode_matrix,
        "condition_number": float(np.linalg.cond(mode_matrix)),
        "pinv": pinv,
        "mode_neff": float(np.real(np.asarray(neffs[mode_index]))),
    }
    if projection_weights is not None:
        projection["projection_weights"] = projection_weights
    projection["modal_plane_delay_s"] = _modal_projection_plane_delay_s(
        sim,
        spec,
        frequency,
        projection["mode_neff"],
    )
    return projection


def _build_port_projection(
    sim,
    spec,
    monitor,
    frequency,
    cache,
    mode_pad_cells=6,
):
    key = (spec.name, monitor.name, float(frequency))
    cached = cache.get(key)
    if cached is not None:
        return cached

    parts = _mode_components_for_port(spec)
    if sim.is_3d:
        analysis_coords0, analysis_coords1 = _monitor_analysis_plane_3d(
            sim, monitor, parts["axis"]
        )
        projection = _build_discrete_port_projection_3d(
            sim,
            spec=spec,
            monitor=monitor,
            frequency=frequency,
            parts=parts,
            direction_sign=1.0,
            analysis_coords0=analysis_coords0,
            analysis_coords1=analysis_coords1,
        )
    else:
        projection = _build_port_projection_2d(
            sim,
            spec=spec,
            monitor=monitor,
            frequency=frequency,
            parts=parts,
            mode_pad_cells=mode_pad_cells,
        )

    cache[key] = projection
    return projection


def _project_modal_coefficients_3d_group(field_components, projections):
    """Project one 3D monitor field onto a coupled forward/backward mode set."""
    projections = tuple(projections)
    if not projections:
        return [], np.nan, np.nan, {}

    first = projections[0]
    components = tuple(first.get("components", ()))
    axis = str(first.get("axis", "")).lower()
    d_area = first.get("integration_weights", first.get("d_area", 1.0))
    direction_sign = float(first.get("direction_sign", 1.0))
    if len(components) == 0 or axis not in {"x", "y", "z"}:
        raise ValueError("3D modal group projection is missing components or axis.")

    basis = []
    for proj in projections:
        if tuple(proj.get("components", ())) != components:
            raise ValueError("Grouped 3D modal projections must share components.")
        if str(proj.get("axis", "")).lower() != axis:
            raise ValueError("Grouped 3D modal projections must share an axis.")
        basis.append(
            {
                name: np.asarray(
                    proj.get("mode_components", {}).get(name, []),
                    dtype=np.complex128,
                ).reshape(-1)
                for name in components
            }
        )
        basis.append(
            {
                name: np.asarray(
                    proj.get("mode_components_bwd", {}).get(name, []),
                    dtype=np.complex128,
                ).reshape(-1)
                for name in components
            }
        )

    rhs = np.asarray(
        [
            _modal_overlap(
                field_components,
                mode,
                axis,
                d_area,
                direction_sign=direction_sign,
            )
            for mode in basis
        ],
        dtype=np.complex128,
    )
    overlap = np.asarray(
        [
            [
                _modal_overlap(
                    basis_i,
                    basis_j,
                    axis,
                    d_area,
                    direction_sign=direction_sign,
                )
                for basis_j in basis
            ]
            for basis_i in basis
        ],
        dtype=np.complex128,
    )
    system = overlap.T
    cond = float(np.linalg.cond(system))
    if (
        not np.all(np.isfinite(system))
        or not np.all(np.isfinite(rhs))
        or not np.isfinite(cond)
    ):
        raise ValueError("Invalid grouped 3D modal overlap system.")
    if cond < 1e8:
        coeff = np.linalg.solve(system, rhs)
    else:
        coeff = np.linalg.pinv(system) @ rhs

    field_parts = [
        np.asarray(field_components[name], dtype=np.complex128).reshape(-1)
        for name in components
    ]
    component_slices = []
    offset = 0
    for name, part in zip(components, field_parts, strict=True):
        next_offset = offset + int(part.size)
        component_slices.append((name, offset, next_offset))
        offset = next_offset
    field_vec = np.concatenate(field_parts)
    mode_matrix = np.column_stack(
        [
            np.concatenate(
                [
                    np.asarray(mode[name], dtype=np.complex128).reshape(-1)
                    for name in components
                ]
            )
            for mode in basis
        ]
    )
    diagnostics = _modal_projection_reconstruction_diagnostics_from_matrix(
        field_vec,
        mode_matrix,
        coeff,
        component_slices=component_slices,
    )
    diagnostics["projected_signed_power"] = float(
        np.real(np.dot(coeff, overlap @ np.conjugate(coeff)))
    )
    residual = diagnostics["residual"]
    return (
        [
            (np.complex128(coeff[2 * idx]), np.complex128(coeff[2 * idx + 1]))
            for idx in range(len(projections))
        ],
        residual,
        cond,
        diagnostics,
    )
