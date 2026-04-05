"""Modal projection helpers for simulation analysis."""

from __future__ import annotations

import numpy as np

from beamz.devices.sources.mode import (
    _make_3d_mode_basis_profiles,
    _modal_overlap_3d_profiles,
)
from beamz.devices.sources.solve import solve_modes


def remap_3d_solver_components(ex, ey, ez, hx, hy, hz, axis):
    """Match solve_modes x-basis output to the requested global propagation axis."""
    if axis == "x":
        return ex, ey, ez, hx, hy, hz
    if axis == "y":
        return -ey, ex, ez, -hy, hx, hz
    if axis == "z":
        return ey, ez, ex, hy, hz, hx
    raise ValueError(f"Unsupported axis {axis!r} for 3D mode remap.")


def monitor_profile_slice(sim, monitor, axis, pad_cells):
    perm = np.asarray(sim.fields.permittivity)
    if perm.ndim == 3:
        z_idx, y_idx, x_idx = monitor.get_grid_slice_3d(
            sim.resolution,
            sim.resolution,
            sim.resolution,
            perm.shape,
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

        z_idx = _clamp(z_idx, perm.shape[0])
        y_idx = _clamp(y_idx, perm.shape[1])
        x_idx = _clamp(x_idx, perm.shape[2])
        eps_slice = np.asarray(perm[z_idx, y_idx, x_idx], dtype=np.complex128)
        if eps_slice.ndim != 2:
            eps_slice = np.atleast_2d(eps_slice)
        npts = int(eps_slice.size)
        local_idx = np.arange(npts, dtype=int)
        d_area = float(sim.resolution) * float(sim.resolution)
        return eps_slice, local_idx, d_area
    if perm.ndim != 2:
        raise NotImplementedError("Modal extraction supports 2D or 3D only.")
    points = monitor.get_grid_points_2d(sim.resolution, sim.resolution)
    if not points:
        raise ValueError(f"Monitor '{monitor.name}' contains no sample points.")
    p = np.asarray(points, dtype=float)
    if axis == "x":
        x_idx = int(np.clip(round(float(np.mean(p[:, 0]))), 0, perm.shape[1] - 1))
        eps_profile_full = perm[:, x_idx]
        sample_idx = np.asarray(
            [int(np.clip(pi[1], 0, perm.shape[0] - 1)) for pi in points], dtype=int
        )
    else:
        y_idx = int(np.clip(round(float(np.mean(p[:, 1]))), 0, perm.shape[0] - 1))
        eps_profile_full = perm[y_idx, :]
        sample_idx = np.asarray(
            [int(np.clip(pi[0], 0, perm.shape[1] - 1)) for pi in points], dtype=int
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
    return np.asarray(eps_profile_full[lo:hi], dtype=np.complex128), local_idx, dl


def build_port_projection(sim, spec, monitor, frequency, cache, mode_pad_cells=6):
    return _build_port_projection(
        sim,
        spec,
        monitor,
        frequency,
        cache,
        mode_pad_cells=mode_pad_cells,
    )


def wanted_components(sim, parts):
    if sim.is_3d:
        return tuple(
            parts.get(
                "projection_components_3d",
                (parts["e_component"], parts["h_component"]),
            )
        )
    return (parts["e_component"], parts["h_component"])


def load_component_cache(cache, monitor, component, loader):
    key = (monitor.name, component)
    if key not in cache:
        _, cache[key] = loader(monitor, component)
    return cache[key]


def resolve_projection(sim, spec, monitor, frequency, projection_cache, last_valid):
    proj = sim._build_port_projection(spec, monitor, float(frequency), projection_cache)
    proj_neff = float(proj.get("mode_neff", np.nan))
    if (not np.isfinite(proj_neff)) or (proj_neff <= 1e-6):
        if last_valid is not None:
            return last_valid, last_valid
        return proj, last_valid
    return proj, proj


def modal_coefficients_from_samples(sim, proj, samples):
    proj_components = tuple(
        proj.get("components", (proj["e_component"], proj["h_component"]))
    )
    if sim.is_3d:
        field_components = {
            comp: np.asarray(samples[comp], dtype=np.complex128)
            for comp in proj_components
        }
        coeff = sim._project_modal_coefficients_3d(field_components, proj)
    else:
        field_vec = np.concatenate(
            [np.asarray(samples[comp], dtype=np.complex128) for comp in proj_components]
        )
        coeff = proj["pinv"] @ field_vec
    return np.asarray(coeff, dtype=np.complex128)


def extract_monitor_coefficients(
    sim,
    spec,
    monitor,
    frequencies,
    *,
    projection_frequencies,
    projection_cache,
    sample_cache,
    loader,
    include_metadata=False,
):
    parts = sim._mode_components_for_port(spec)
    for component in wanted_components(sim, parts):
        load_component_cache(sample_cache, monitor, component, loader)

    a_plus = np.zeros(frequencies.size, dtype=np.complex128)
    a_minus = np.zeros(frequencies.size, dtype=np.complex128)
    condition_number = None
    mode_neff = None
    if include_metadata:
        condition_number = np.zeros(frequencies.size, dtype=float)
        mode_neff = np.full(frequencies.size, np.nan, dtype=float)

    last_valid_proj = None
    for idx, f_mode in enumerate(projection_frequencies):
        proj, last_valid_proj = resolve_projection(
            sim, spec, monitor, f_mode, projection_cache, last_valid_proj
        )
        proj_components = tuple(
            proj.get("components", (proj["e_component"], proj["h_component"]))
        )
        coeff = modal_coefficients_from_samples(
            sim,
            proj,
            {
                comp: sample_cache[(monitor.name, comp)][idx]
                for comp in proj_components
            },
        )
        a_plus[idx], a_minus[idx] = coeff[0], coeff[1]
        if include_metadata:
            condition_number[idx] = float(proj.get("condition_number", np.nan))
            mode_neff[idx] = float(proj.get("mode_neff", np.nan))

    return a_plus, a_minus, condition_number, mode_neff


def extract_reference_waves(
    sim,
    spec,
    monitor,
    frequencies,
    *,
    projection_frequencies,
    projection_cache,
    sample_cache,
    loader,
    include_metadata=False,
):
    a_plus, a_minus, condition_number, mode_neff = extract_monitor_coefficients(
        sim,
        spec,
        monitor,
        frequencies,
        projection_frequencies=projection_frequencies,
        projection_cache=projection_cache,
        sample_cache=sample_cache,
        loader=loader,
        include_metadata=include_metadata,
    )
    data = {
        "a_incident": a_plus,
        "a_incident_plus": a_plus,
        "a_incident_minus": a_minus,
    }
    if include_metadata:
        data["reference_condition_number"] = condition_number
        data["reference_mode_neff"] = mode_neff
    return data


def extract_cw_monitor_coefficients(
    sim,
    spec,
    monitor,
    frequency,
    *,
    projection_cache,
    steady_start_time,
    avg_cycles,
    window,
):
    parts = sim._mode_components_for_port(spec)
    proj = sim._build_port_projection(spec, monitor, frequency, projection_cache)
    coeff = proj["pinv"] @ np.concatenate(
        [
            sim._demodulate_monitor_component(
                monitor,
                component,
                frequency=frequency,
                t_start=steady_start_time,
                avg_cycles=avg_cycles,
                window=window,
            )
            for component in (parts["e_component"], parts["h_component"])
        ]
    )
    return np.complex128(coeff[0]), np.complex128(coeff[1])


def _build_port_projection(
    sim,
    spec,
    monitor,
    frequency,
    cache,
    mode_pad_cells=6,
    *,
    solve_modes_fn=solve_modes,
    mode_basis_builder=_make_3d_mode_basis_profiles,
    modal_overlap_fn=_modal_overlap_3d_profiles,
):
    key = (spec.name, monitor.name, float(frequency))
    cached = cache.get(key)
    if cached is not None:
        return cached

    parts = sim._mode_components_for_port(spec)
    eps_profile, local_idx, dl = sim._monitor_profile_slice(
        monitor, parts["axis"], mode_pad_cells
    )
    solver_direction = spec.direction
    if sim.is_3d:
        solver_direction = "+" + parts["axis"]
    omega = 2.0 * np.pi * float(frequency)
    eps_profile_arr = np.asarray(eps_profile)
    n_local_max = float(np.sqrt(max(float(np.max(np.real(eps_profile_arr))), 1e-12)))
    target_neff = 0.98 * n_local_max
    neff_vals, e_fields, h_fields, _ = solve_modes_fn(
        eps=eps_profile,
        omega=omega,
        dL=float(sim.resolution),
        m=spec.mode_index + 1,
        direction=solver_direction,
        filter_pol=spec.polarization,
        target_neff=target_neff,
        return_fields=True,
    )

    mode = int(spec.mode_index)
    if sim.is_3d:
        ex_full = np.asarray(np.squeeze(e_fields[mode][0]), dtype=np.complex128)
        ey_full = np.asarray(np.squeeze(e_fields[mode][1]), dtype=np.complex128)
        ez_full = np.asarray(np.squeeze(e_fields[mode][2]), dtype=np.complex128)
        hx_full = np.asarray(np.squeeze(h_fields[mode][0]), dtype=np.complex128)
        hy_full = np.asarray(np.squeeze(h_fields[mode][1]), dtype=np.complex128)
        hz_full = np.asarray(np.squeeze(h_fields[mode][2]), dtype=np.complex128)
        ex_full, ey_full, ez_full, hx_full, hy_full, hz_full = (
            sim._remap_3d_solver_components(
                ex_full,
                ey_full,
                ez_full,
                hx_full,
                hy_full,
                hz_full,
                parts["axis"],
            )
        )
        comp_full = {
            "Ex": ex_full,
            "Ey": ey_full,
            "Ez": ez_full,
            "Hx": hx_full,
            "Hy": hy_full,
            "Hz": hz_full,
        }
        for name in tuple(comp_full.keys()):
            arr = np.asarray(comp_full[name], dtype=np.complex128)
            if arr.ndim == 1:
                arr = arr[:, None]
            comp_full[name] = arr
        proj_components = tuple(parts.get("projection_components_3d", ()))
        mon_dim0 = 0
        mon_dim1 = 0
        try:
            shape_map = {
                "Ex": tuple(np.asarray(sim.fields.Ex).shape),
                "Ey": tuple(np.asarray(sim.fields.Ey).shape),
                "Ez": tuple(np.asarray(sim.fields.Ez).shape),
                "Hx": tuple(np.asarray(sim.fields.Hx).shape),
                "Hy": tuple(np.asarray(sim.fields.Hy).shape),
                "Hz": tuple(np.asarray(sim.fields.Hz).shape),
            }

            def _clamp_idx(idx, limit):
                if isinstance(idx, slice):
                    start = 0 if idx.start is None else int(idx.start)
                    stop = limit if idx.stop is None else int(idx.stop)
                    start = max(0, min(start, max(limit - 1, 0)))
                    stop = max(start + 1, min(stop, limit))
                    return slice(start, stop)
                ii = int(idx)
                return max(0, min(ii, limit - 1))

            def _slice_len(idx, limit):
                if isinstance(idx, slice):
                    start = 0 if idx.start is None else int(idx.start)
                    stop = limit if idx.stop is None else int(idx.stop)
                    step = 1 if idx.step is None else int(idx.step)
                    if step <= 0:
                        raise ValueError("Only positive slice steps are supported.")
                    span = max(0, stop - start)
                    return 0 if span <= 0 else 1 + (span - 1) // step
                return 1

            dims0 = []
            dims1 = []
            for cname in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                shp = shape_map[cname]
                z_idx, y_idx, x_idx = monitor.get_grid_slice_3d(
                    sim.resolution,
                    sim.resolution,
                    sim.resolution,
                    shp,
                )
                z_idx = _clamp_idx(z_idx, shp[0])
                y_idx = _clamp_idx(y_idx, shp[1])
                x_idx = _clamp_idx(x_idx, shp[2])
                slice_lens = [
                    _slice_len(idx, lim)
                    for idx, lim in (
                        (z_idx, shp[0]),
                        (y_idx, shp[1]),
                        (x_idx, shp[2]),
                    )
                    if isinstance(idx, slice)
                ]
                if len(slice_lens) >= 2:
                    d0, d1 = int(slice_lens[0]), int(slice_lens[1])
                elif len(slice_lens) == 1:
                    d0, d1 = int(slice_lens[0]), 1
                else:
                    d0, d1 = 1, 1
                dims0.append(max(d0, 1))
                dims1.append(max(d1, 1))
            mon_dim0 = min(dims0)
            mon_dim1 = min(dims1)
        except Exception:
            mon_dim0 = min(int(comp_full[c].shape[0]) for c in proj_components)
            mon_dim1 = min(int(comp_full[c].shape[1]) for c in proj_components)

        try:
            n_monitor = min(
                int(
                    np.asarray(
                        monitor.get_dft_component(comp_name),
                        dtype=np.complex128,
                    ).shape[1]
                )
                for comp_name in proj_components
            )
            if mon_dim0 > 0 and mon_dim1 > 0:
                n_monitor = min(n_monitor, int(mon_dim0 * mon_dim1))
            if mon_dim0 > 0:
                mon_dim1 = max(1, min(mon_dim1, int(n_monitor // max(mon_dim0, 1))))
            if mon_dim1 > 0:
                mon_dim0 = max(1, min(mon_dim0, int(n_monitor // max(mon_dim1, 1))))
        except Exception:
            n_monitor = int(mon_dim0 * mon_dim1)

        crop_dim0 = int(min(mon_dim0, *(comp_full[c].shape[0] for c in proj_components)))
        crop_dim1 = int(min(mon_dim1, *(comp_full[c].shape[1] for c in proj_components)))
        if crop_dim0 <= 0 or crop_dim1 <= 0:
            crop_dim0 = int(min(comp_full[c].shape[0] for c in proj_components))
            crop_dim1 = int(min(comp_full[c].shape[1] for c in proj_components))
        n_target = int(crop_dim0 * crop_dim1)
        if n_monitor > 0:
            n_target = min(n_target, n_monitor)
        if n_target <= 0:
            raise ValueError(f"Monitor '{monitor.name}' has zero 3D projection points.")

        crop_dim1 = max(1, min(crop_dim1, n_target))
        crop_dim0 = max(1, min(crop_dim0, n_target // crop_dim1))
        n_target = int(crop_dim0 * crop_dim1)

        comp_samples = {}
        for name, arr in comp_full.items():
            a = np.asarray(arr, dtype=np.complex128)
            if a.ndim == 1:
                a = a[:, None]
            a = a[:crop_dim0, :crop_dim1]
            comp_samples[name] = a.reshape(-1)[:n_target]
    else:
        e_fwd_full = np.asarray(
            np.squeeze(e_fields[mode][parts["e_mode_index"]]), dtype=np.complex128
        )
        h_fwd_full = np.asarray(
            np.squeeze(h_fields[mode][parts["h_mode_index"]]), dtype=np.complex128
        )
        if e_fwd_full.ndim > 1:
            e_fwd_full = e_fwd_full[:, 0]
        if h_fwd_full.ndim > 1:
            h_fwd_full = h_fwd_full[:, 0]
        e_fwd = e_fwd_full[local_idx]
        h_fwd = h_fwd_full[local_idx]
        proj_components = (parts["e_component"], parts["h_component"])
        comp_samples = {parts["e_component"]: e_fwd, parts["h_component"]: h_fwd}

    if sim.is_3d:
        h_ref = comp_samples.get(parts["h_component"], np.zeros((0,), dtype=np.complex128))
    else:
        h_ref = h_fwd
    if h_ref.size:
        i_max = int(np.argmax(np.abs(h_ref)))
        phase = np.angle(h_ref[i_max])
        phase_rot = np.exp(-1j * phase)
        for name in tuple(comp_samples.keys()):
            comp_samples[name] = comp_samples[name] * phase_rot

    if sim.is_3d:
        raw_mode_components = {
            name: np.asarray(comp_samples[name], dtype=np.complex128).reshape(-1)
            for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
            if name in comp_samples
        }
        mode_components, mode_components_bwd = mode_basis_builder(
            raw_mode_components,
            axis=parts["axis"],
            d_area=float(dl),
        )
        comp_samples = mode_components
        fwd_vec = np.concatenate([mode_components[c] for c in proj_components])
        bwd_vec = np.concatenate([mode_components_bwd[c] for c in proj_components])
        mode_matrix = np.column_stack([fwd_vec, bwd_vec])
        overlap_matrix = np.asarray(
            [
                [
                    modal_overlap_fn(
                        mode_components, mode_components, parts["axis"], float(dl)
                    ),
                    modal_overlap_fn(
                        mode_components_bwd, mode_components, parts["axis"], float(dl)
                    ),
                ],
                [
                    modal_overlap_fn(
                        mode_components, mode_components_bwd, parts["axis"], float(dl)
                    ),
                    modal_overlap_fn(
                        mode_components_bwd,
                        mode_components_bwd,
                        parts["axis"],
                        float(dl),
                    ),
                ],
            ],
            dtype=np.complex128,
        )
    else:
        if e_fwd_full.ndim > 1:
            e_fwd_full = e_fwd_full[:, 0]
        if h_fwd_full.ndim > 1:
            h_fwd_full = h_fwd_full[:, 0]
        e_fwd = e_fwd_full[local_idx]
        h_fwd = h_fwd_full[local_idx]
        pm = 0.5 * np.real(
            np.sum(parts["signed_flux_sign"] * e_fwd * np.conjugate(h_fwd)) * dl
        )
        norm = np.sqrt(max(abs(pm), 1e-30))
        e_fwd = e_fwd / norm
        h_fwd = h_fwd / norm
        e_bwd = e_fwd.copy()
        h_bwd = -h_fwd.copy()
        mode_matrix = np.column_stack(
            [
                np.concatenate([e_fwd, h_fwd]),
                np.concatenate([e_bwd, h_bwd]),
            ]
        )

    projection = {
        "e_component": parts["e_component"],
        "h_component": parts["h_component"],
        "components": tuple(proj_components),
        "mode_matrix": mode_matrix,
        "condition_number": float(
            np.linalg.cond(overlap_matrix if sim.is_3d else mode_matrix)
        ),
        "pinv": np.linalg.pinv(mode_matrix),
        "mode_neff": float(np.real(np.asarray(neff_vals[mode]))),
    }
    if sim.is_3d:
        projection["mode_components"] = {
            name: np.asarray(comp_samples[name], dtype=np.complex128)
            for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
            if name in comp_samples
        }
        projection["mode_components_bwd"] = {
            name: np.asarray(mode_components_bwd[name], dtype=np.complex128)
            for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
            if name in mode_components_bwd
        }
        projection["overlap_matrix"] = np.asarray(overlap_matrix, dtype=np.complex128)
        projection["axis"] = parts["axis"]
        projection["d_area"] = float(dl)
        projection["power_norm"] = 1.0
    cache[key] = projection
    return projection


def modal_power_3d(mode_components, axis, d_area):
    ex = np.asarray(
        mode_components.get("Ex", np.zeros((0,), dtype=np.complex128)),
        dtype=np.complex128,
    )
    ey = np.asarray(
        mode_components.get("Ey", np.zeros((0,), dtype=np.complex128)),
        dtype=np.complex128,
    )
    ez = np.asarray(
        mode_components.get("Ez", np.zeros((0,), dtype=np.complex128)),
        dtype=np.complex128,
    )
    hx = np.asarray(
        mode_components.get("Hx", np.zeros((0,), dtype=np.complex128)),
        dtype=np.complex128,
    )
    hy = np.asarray(
        mode_components.get("Hy", np.zeros((0,), dtype=np.complex128)),
        dtype=np.complex128,
    )
    hz = np.asarray(
        mode_components.get("Hz", np.zeros((0,), dtype=np.complex128)),
        dtype=np.complex128,
    )
    n = int(min(ex.size, ey.size, ez.size, hx.size, hy.size, hz.size))
    if n <= 0:
        return 0.0
    ex = ex[:n]
    ey = ey[:n]
    ez = ez[:n]
    hx = hx[:n]
    hy = hy[:n]
    hz = hz[:n]
    if axis == "x":
        s_axis = ey * np.conjugate(hz) - ez * np.conjugate(hy)
    elif axis == "y":
        s_axis = ez * np.conjugate(hx) - ex * np.conjugate(hz)
    else:
        s_axis = ex * np.conjugate(hy) - ey * np.conjugate(hx)
    return float(0.5 * np.real(np.sum(s_axis) * float(d_area)))


def project_modal_coefficients_3d(
    field_components, projection, apply_calibration=True
):
    return _project_modal_coefficients_3d(
        field_components,
        projection,
        apply_calibration=apply_calibration,
    )


def _project_modal_coefficients_3d(
    field_components,
    projection,
    apply_calibration=True,
    *,
    modal_overlap_fn=_modal_overlap_3d_profiles,
):
    del apply_calibration
    mode_components = projection.get("mode_components", None)
    mode_components_bwd = projection.get("mode_components_bwd", None)
    overlap_matrix = projection.get("overlap_matrix", None)
    axis = str(projection.get("axis", "")).lower()
    d_area = float(projection.get("d_area", 1.0))
    if (
        isinstance(mode_components, dict)
        and isinstance(mode_components_bwd, dict)
        and overlap_matrix is not None
        and axis in {"x", "y", "z"}
    ):
        rhs = np.asarray(
            [
                modal_overlap_fn(
                    field_components,
                    mode_components,
                    axis,
                    d_area,
                ),
                modal_overlap_fn(
                    field_components,
                    mode_components_bwd,
                    axis,
                    d_area,
                ),
            ],
            dtype=np.complex128,
        )
        overlap = np.asarray(overlap_matrix, dtype=np.complex128)
        cond = float(np.linalg.cond(overlap))
        if (
            not np.all(np.isfinite(overlap))
            or not np.all(np.isfinite(rhs))
            or not np.isfinite(cond)
        ):
            raise ValueError("Invalid 3D modal overlap system.")
        if cond < 1e8:
            coeff = np.linalg.solve(overlap, rhs)
        else:
            coeff = np.linalg.pinv(overlap) @ rhs
        return np.complex128(coeff[0]), np.complex128(coeff[1])

    components = tuple(projection.get("components", ()))
    if len(components) == 0:
        raise ValueError("3D projection missing component list.")

    vec_parts = []
    for comp in components:
        if comp not in field_components:
            raise ValueError(
                f"Missing field component '{comp}' for 3D modal projection."
            )
        vec_parts.append(
            np.asarray(field_components[comp], dtype=np.complex128).reshape(-1)
        )
    field_vec = np.concatenate(vec_parts).astype(np.complex128, copy=False)

    pinv = np.asarray(
        projection.get("pinv", np.zeros((2, 0), dtype=np.complex128)),
        dtype=np.complex128,
    )
    if pinv.ndim != 2 or pinv.shape[0] < 2:
        raise ValueError("Invalid 3D projection pseudo-inverse shape.")
    n_expected = int(pinv.shape[1])
    if field_vec.size != n_expected:
        if field_vec.size > n_expected:
            field_vec = field_vec[:n_expected]
        else:
            field_vec = np.pad(field_vec, (0, n_expected - field_vec.size))
    coeff = pinv @ field_vec
    a_plus = coeff[0]
    a_minus = coeff[1]
    return np.complex128(a_plus), np.complex128(a_minus)
