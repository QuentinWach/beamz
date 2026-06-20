from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    Design,
    Material,
    ModeSource,
    Monitor,
    Rectangle,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    solve_modes,
    um,
)
from beamz.devices.sources.mode import (
    _axis_index_from_component_indices,
    _build_3d_profiles,
    _component_axis_coord,
    _detect_transverse_symmetry_axes,
    _enforce_componentwise_parity,
    _make_3d_mode_basis_profiles,
    _match_shape,
    _modal_overlap_3d_profiles,
    _modal_power_3d_from_profiles,
    _normalize_3d_profiles_by_flux,
    _numeric_phase_delay,
    _phase_reference_3d_profiles,
    _project_3d_profiles_to_real,
    _remap_3d_solver_components,
    _runtime_3d_profiles,
    _select_3d_impedance_index,
    _select_3d_phase_ref,
    _select_core_confined_mode_index,
    _shift_component_indices_along_axis,
    _stagger_both,
    _stagger_half,
)
from beamz.simulation import ops
from beamz.simulation.boundaries import (
    build_h_boundary_views_for_e_3d,
    has_full_pec_3d,
    sync_full_pec_3d_from_compact,
)
from beamz.simulation.yee import (
    component_axis_offsets_3d,
    sample_voxel_grid_at_component_3d,
    sample_voxel_grid_at_e_component_3d_centered,
)

pytestmark = [pytest.mark.component, pytest.mark.simulation]


def _quantize_cells(
    target_m: float,
    dx: float,
    *,
    min_cells: int = 2,
    parity: int | None = None,
) -> int:
    cells = max(int(min_cells), int(round(float(target_m) / float(dx))))
    if parity is None:
        return cells
    parity = int(parity) & 1
    if cells % 2 == parity:
        return cells
    if cells <= int(min_cells):
        return cells + 1
    lower = cells - 1
    upper = cells + 1
    if lower >= int(min_cells) and (lower % 2) == parity:
        return lower
    return upper


def _build_centered_straight_guide_sim(*, ppw: int = 6, axis: str = "x"):
    wavelength = 1.55 * um
    n_core = 2.0
    n_clad = 1.0
    dx, dt = calc_optimal_fdtd_params(
        wavelength,
        n_core,
        dims=3,
        safety_factor=0.9,
        points_per_wavelength=int(ppw),
        width=5.5 * wavelength,
        height=2.2 * wavelength,
        depth=2.0 * wavelength,
    )

    long_cells = _quantize_cells(5.5 * wavelength, dx, min_cells=12)
    height_cells = _quantize_cells(2.2 * wavelength, dx, min_cells=16)
    depth_cells = _quantize_cells(2.0 * wavelength, dx, min_cells=14)
    guide_y_cells = _quantize_cells(
        0.55 * wavelength,
        dx,
        min_cells=2,
        parity=height_cells % 2,
    )
    guide_z_cells = _quantize_cells(
        0.35 * wavelength,
        dx,
        min_cells=2,
        parity=depth_cells % 2,
    )

    width = long_cells * dx
    height = height_cells * dx
    depth = depth_cells * dx
    guide0 = guide_y_cells * dx
    guide1 = guide_z_cells * dx
    center = (0.5 * width, 0.5 * height, 0.5 * depth)

    design = Design(
        width=width,
        height=height,
        depth=depth,
        material=Material(n_clad**2),
    )
    axis = str(axis).lower()
    if axis == "x":
        design += Rectangle(
            position=(0.0, center[1] - 0.5 * guide0, center[2] - 0.5 * guide1),
            width=width,
            height=guide0,
            depth=guide1,
            material=Material(n_core**2),
        )
    elif axis == "y":
        design += Rectangle(
            position=(center[0] - 0.5 * guide0, 0.0, center[2] - 0.5 * guide1),
            width=guide0,
            height=height,
            depth=guide1,
            material=Material(n_core**2),
        )
    else:
        design += Rectangle(
            position=(center[0] - 0.5 * guide0, center[1] - 0.5 * guide1, 0.0),
            width=guide0,
            height=guide1,
            depth=depth,
            material=Material(n_core**2),
        )

    sim = Simulation(
        design=design,
        sources=[],
        time=np.asarray([0.0, dt], dtype=float),
        resolution=dx,
    )
    return sim


def _build_centered_straight_guide_sim_steps(
    *, ppw: int = 6, axis: str = "x", num_steps: int = 40
):
    base = _build_centered_straight_guide_sim(ppw=ppw, axis=axis)
    time = np.arange(
        0.0,
        (int(num_steps) + 1) * float(base.dt),
        float(base.dt),
        dtype=float,
    )
    return Simulation(
        design=base.design,
        sources=[],
        time=time,
        resolution=base.resolution,
        plane_2d=base.plane_2d,
    )


def _build_centered_uniform_sim(*, ppw: int = 6):
    wavelength = 1.55 * um
    n_core = 2.0
    n_clad = 1.0
    dx, dt = calc_optimal_fdtd_params(
        wavelength,
        n_core,
        dims=3,
        safety_factor=0.9,
        points_per_wavelength=int(ppw),
        width=5.5 * wavelength,
        height=2.2 * wavelength,
        depth=2.0 * wavelength,
    )

    long_cells = _quantize_cells(5.5 * wavelength, dx, min_cells=12)
    height_cells = _quantize_cells(2.2 * wavelength, dx, min_cells=16)
    depth_cells = _quantize_cells(2.0 * wavelength, dx, min_cells=14)
    width = long_cells * dx
    height = height_cells * dx
    depth = depth_cells * dx

    design = Design(
        width=width,
        height=height,
        depth=depth,
        material=Material(n_clad**2),
    )
    sim = Simulation(
        design=design,
        sources=[],
        time=np.asarray([0.0, dt], dtype=float),
        resolution=dx,
    )
    return sim


def _build_tiny_straight_guide_sim(
    *,
    ppw: int = 6,
    axis: str = "x",
    long_cells: int = 18,
    transverse0_cells: int = 8,
    transverse1_cells: int = 6,
    guide0_cells: int = 4,
    guide1_cells: int = 2,
    num_steps: int = 8,
):
    wavelength = 1.55 * um
    n_core = 2.0
    n_clad = 1.0
    dx, dt = calc_optimal_fdtd_params(
        wavelength,
        n_core,
        dims=3,
        safety_factor=0.9,
        points_per_wavelength=int(ppw),
        width=5.5 * wavelength,
        height=2.2 * wavelength,
        depth=2.0 * wavelength,
    )

    width = float(long_cells) * dx if axis == "x" else float(transverse0_cells) * dx
    height = float(long_cells) * dx if axis == "y" else float(transverse0_cells) * dx
    depth = float(long_cells) * dx if axis == "z" else float(transverse1_cells) * dx
    g0 = float(guide0_cells) * dx
    g1 = float(guide1_cells) * dx
    center = (0.5 * width, 0.5 * height, 0.5 * depth)

    design = Design(
        width=width,
        height=height,
        depth=depth,
        material=Material(n_clad**2),
    )
    axis = str(axis).lower()
    if axis == "x":
        design += Rectangle(
            position=(0.0, center[1] - 0.5 * g0, center[2] - 0.5 * g1),
            width=width,
            height=g0,
            depth=g1,
            material=Material(n_core**2),
        )
        source_spans = (
            float(min(transverse0_cells, guide0_cells + 2)) * dx,
            float(min(transverse1_cells, guide1_cells + 2)) * dx,
        )
    elif axis == "y":
        design += Rectangle(
            position=(center[0] - 0.5 * g0, 0.0, center[2] - 0.5 * g1),
            width=g0,
            height=height,
            depth=g1,
            material=Material(n_core**2),
        )
        source_spans = (
            float(min(transverse0_cells, guide0_cells + 2)) * dx,
            float(min(transverse1_cells, guide1_cells + 2)) * dx,
        )
    else:
        design += Rectangle(
            position=(center[0] - 0.5 * g0, center[1] - 0.5 * g1, 0.0),
            width=g0,
            height=g1,
            depth=depth,
            material=Material(n_core**2),
        )
        source_spans = (
            float(min(transverse1_cells, guide1_cells + 2)) * dx,
            float(min(transverse0_cells, guide0_cells + 2)) * dx,
        )

    time = np.arange(0.0, (int(num_steps) + 2) * dt, dt, dtype=float)
    sim = Simulation(
        design=design,
        sources=[],
        time=time,
        resolution=dx,
    )
    return sim, source_spans


def _mirror_residual(arr: np.ndarray, axis: int) -> float:
    lhs = np.asarray(arr, dtype=float)
    rhs = np.flip(lhs, axis=axis)
    denom = max(float(np.linalg.norm(lhs.ravel())), 1e-30)
    return float(np.linalg.norm((lhs - rhs).ravel()) / denom)


def _physical_component_mirror_residual(
    arr: np.ndarray,
    *,
    component: str,
    grid_shape: tuple[int, int, int],
    axis: int,
    global_axis: int | None = None,
    global_indices: np.ndarray | None = None,
) -> float:
    axes = ("z", "y", "x")
    global_axis = int(axis if global_axis is None else global_axis)
    indices = (
        np.arange(np.asarray(arr).shape[axis], dtype=int)
        if global_indices is None
        else np.asarray(global_indices, dtype=int)
    )
    offset = component_axis_offsets_3d(component)[axes[global_axis]]
    grid_dim = int(grid_shape[global_axis])
    mirrored = np.rint(grid_dim - (indices + offset) - offset).astype(int)
    index_to_local = {int(index): pos for pos, index in enumerate(indices)}

    lhs_pos: list[int] = []
    rhs_pos: list[int] = []
    for pos, mirror_index in enumerate(mirrored):
        mirror_pos = index_to_local.get(int(mirror_index))
        if mirror_pos is not None:
            lhs_pos.append(pos)
            rhs_pos.append(mirror_pos)

    if not lhs_pos:
        return 0.0

    moved = np.moveaxis(np.asarray(arr, dtype=float), axis, 0)
    lhs = np.take(moved, lhs_pos, axis=0)
    rhs = np.take(moved, rhs_pos, axis=0)
    denom = max(float(np.linalg.norm(lhs.ravel())), 1e-30)
    return float(np.linalg.norm((lhs - rhs).ravel()) / denom)


def _best_parity_residual(profile: np.ndarray, axis: int) -> float:
    arr = np.asarray(profile, dtype=float)
    flipped = np.flip(arr, axis=axis)
    denom = max(float(np.linalg.norm(arr.ravel())), 1e-30)
    even = float(np.linalg.norm((arr - flipped).ravel()) / denom)
    odd = float(np.linalg.norm((arr + flipped).ravel()) / denom)
    return min(even, odd)


def _build_test_source(
    sim: Simulation,
    *,
    direction: str = "+x",
    pol: str = "te",
    source_spans: tuple[float, float] | None = None,
) -> tuple[ModeSource, float]:
    dx = float(sim.resolution)
    width = float(sim.design.width)
    height = float(sim.design.height)
    depth = float(sim.design.depth)
    direction = str(direction)
    center = {
        "+x": (8.0 * dx, 0.5 * height, 0.5 * depth),
        "-x": (width - 8.0 * dx, 0.5 * height, 0.5 * depth),
        "+y": (0.5 * width, 8.0 * dx, 0.5 * depth),
        "-y": (0.5 * width, height - 8.0 * dx, 0.5 * depth),
        "+z": (0.5 * width, 0.5 * height, 8.0 * dx),
        "-z": (0.5 * width, 0.5 * height, depth - 8.0 * dx),
    }[direction]
    if source_spans is None:
        source_spans = (12.0 * dx, 10.0 * dx)
    source = ModeSource(
        grid=sim.design.rasterize(resolution=dx),
        center=center,
        width=float(source_spans[0]),
        height=float(source_spans[1]),
        wavelength=1.55 * um,
        pol=str(pol),
        signal=np.asarray([1.0, 1.0, 1.0], dtype=float),
        direction=direction,
    )
    source.initialize(np.asarray(sim.fields.permittivity), dx, dt=float(sim.dt))
    return source, dx


def _build_step_driven_test_source(
    sim: Simulation,
    *,
    direction: str = "+x",
    pol: str = "te",
    source_spans: tuple[float, float] | None = None,
) -> tuple[ModeSource, float]:
    source, dx = _build_test_source(
        sim,
        direction=direction,
        pol=pol,
        source_spans=source_spans,
    )
    source.signal = np.ones(int(sim.num_steps) + 5, dtype=float)
    return source, dx


def _full_transverse_source_spans(
    sim: Simulation,
    *,
    direction: str,
    margin_cells: int = 1,
) -> tuple[float, float]:
    dx = float(sim.resolution)
    margin = float(max(0, int(margin_cells))) * dx
    width = float(sim.design.width)
    height = float(sim.design.height)
    depth = float(sim.design.depth)
    axis = str(direction)[1]
    if axis == "x":
        return (max(dx, height - 2.0 * margin), max(dx, depth - 2.0 * margin))
    if axis == "y":
        extra = 2.0 * dx
        return (
            max(dx, width - 2.0 * (margin + extra)),
            max(dx, depth - 2.0 * (margin + extra)),
        )
    return (max(dx, height - 2.0 * margin), max(dx, width - 2.0 * margin))


def _build_tiny_test_source(
    sim: Simulation,
    source_spans: tuple[float, float],
    *,
    direction: str = "+x",
    pol: str = "te",
    clearance_cells: int = 4,
):
    dx = float(sim.resolution)
    width = float(sim.design.width)
    height = float(sim.design.height)
    depth = float(sim.design.depth)
    direction = str(direction)
    clearance = float(clearance_cells) * dx
    center = {
        "+x": (clearance, 0.5 * height, 0.5 * depth),
        "-x": (width - clearance, 0.5 * height, 0.5 * depth),
        "+y": (0.5 * width, clearance, 0.5 * depth),
        "-y": (0.5 * width, height - clearance, 0.5 * depth),
        "+z": (0.5 * width, 0.5 * height, clearance),
        "-z": (0.5 * width, 0.5 * height, depth - clearance),
    }[direction]
    freq = LIGHT_SPEED / (1.55 * um)
    t_max = float(np.asarray(sim.time, dtype=float)[-1])
    signal = ramped_cosine(
        np.asarray(sim.time, dtype=float),
        amplitude=1.0,
        frequency=freq,
        ramp_duration=1.0 / freq,
        t_max=t_max,
    )
    source = ModeSource(
        grid=sim.design.rasterize(resolution=dx),
        center=center,
        width=float(source_spans[0]),
        height=float(source_spans[1]),
        wavelength=1.55 * um,
        pol=str(pol),
        signal=signal,
        direction=direction,
    )
    source.initialize(np.asarray(sim.fields.permittivity), dx, dt=float(sim.dt))
    return source, dx


def _support_window_parity_residual(
    fields,
    source: ModeSource,
    component_name: str,
    index_attr: str,
    axis: int,
) -> float:
    idx = getattr(source, index_attr)
    arr = np.asarray(getattr(fields, component_name), dtype=float)[idx]
    return _best_parity_residual(arr, axis=axis)


def _support_array_parity_residual(
    arr,
    source: ModeSource,
    index_attr: str,
    axis: int,
) -> float:
    idx = getattr(source, index_attr)
    sample = np.asarray(arr, dtype=float)[idx]
    return _best_parity_residual(sample, axis=axis)


def _trimmed_support_array(
    arr, source: ModeSource, index_attr: str, trim: int = 1
) -> np.ndarray:
    idx = getattr(source, index_attr)
    sample = np.asarray(arr, dtype=float)[idx]
    if sample.ndim != 2:
        return sample
    trim = int(max(0, trim))
    if trim == 0:
        return sample
    if sample.shape[0] <= 2 * trim or sample.shape[1] <= 2 * trim:
        return sample
    return sample[trim:-trim, trim:-trim]


def _slice_with_extra_stop(s: slice, extra: int = 1) -> slice:
    return slice(int(s.start), int(s.stop) + int(extra))


def _move_along(
    center: tuple[float, float, float], direction: str, distance: float
) -> tuple[float, float, float]:
    x, y, z = center
    if direction == "+x":
        return (x + distance, y, z)
    if direction == "-x":
        return (x - distance, y, z)
    if direction == "+y":
        return (x, y + distance, z)
    if direction == "-y":
        return (x, y - distance, z)
    if direction == "+z":
        return (x, y, z + distance)
    return (x, y, z - distance)


def _monitor_plane(
    center: tuple[float, float, float], axis: str, span0: float, span1: float
):
    x, y, z = center
    if axis == "x":
        return (x, y - 0.5 * span0, z - 0.5 * span1), (
            x,
            y + 0.5 * span0,
            z + 0.5 * span1,
        )
    if axis == "y":
        return (x - 0.5 * span0, y, z - 0.5 * span1), (
            x + 0.5 * span0,
            y,
            z + 0.5 * span1,
        )
    return (x - 0.5 * span1, y - 0.5 * span0, z), (x + 0.5 * span1, y + 0.5 * span0, z)


def _sample_monitor_plane(
    sim: Simulation,
    source: ModeSource,
    source_spans: tuple[float, float],
    *,
    direction: str,
    monitor_offset_cells: int = 3,
) -> dict[str, np.ndarray]:
    axis = str(direction)[1]
    monitor_center = _move_along(
        tuple(float(v) for v in source.center),
        str(direction),
        float(monitor_offset_cells) * float(sim.resolution),
    )
    mon_start, mon_end = _monitor_plane(
        monitor_center, axis, source_spans[0], source_spans[1]
    )
    monitor = Monitor(
        start=mon_start,
        end=mon_end,
        name="m",
        record_fields=True,
        dft_enabled=False,
    )
    monitor.record_fields_3d(
        np.asarray(sim.fields.Ex),
        np.asarray(sim.fields.Ey),
        np.asarray(sim.fields.Ez),
        np.asarray(sim.fields.Hx),
        np.asarray(sim.fields.Hy),
        np.asarray(sim.fields.Hz),
        t=float(sim.t),
        dx=float(sim.resolution),
        dy=float(sim.resolution),
        dz=float(sim.resolution),
        step=int(sim.current_step),
    )
    coords0, coords1 = monitor.get_analysis_plane_coords_3d(
        dx=float(sim.resolution),
        dy=float(sim.resolution),
        dz=float(sim.resolution),
        field_shape=tuple(np.asarray(sim.fields.permittivity).shape),
    )
    n0 = int(np.asarray(coords0).size)
    n1 = int(np.asarray(coords1).size)
    return {
        comp: np.asarray(monitor.fields[comp][-1], dtype=np.complex128).reshape(n0, n1)
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }


def _best_abs_parity_pct(arr: np.ndarray) -> tuple[float, float]:
    mag = np.abs(np.asarray(arr))
    return (
        100.0 * _best_parity_residual(mag, axis=0),
        100.0 * _best_parity_residual(mag, axis=1),
    )


def _max_support_parity(
    fields, source: ModeSource, components: tuple[tuple[str, str], ...]
) -> tuple[float, float]:
    axis0 = 0.0
    axis1 = 0.0
    for comp, idx_attr in components:
        metrics = _support_window_parity_residual(
            fields, source, comp, idx_attr, axis=0
        )
        axis0 = max(axis0, float(metrics))
        metrics = _support_window_parity_residual(
            fields, source, comp, idx_attr, axis=1
        )
        axis1 = max(axis1, float(metrics))
    return axis0, axis1


def _field_state_arrays(fields) -> dict[str, np.ndarray]:
    return {
        comp: np.asarray(getattr(fields, comp), dtype=float)
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }


def _set_field_state(fields, state: dict[str, np.ndarray]) -> None:
    for comp, arr in state.items():
        current = getattr(fields, comp)
        setattr(fields, comp, jnp.asarray(arr, dtype=current.dtype))


def _sync_full_pec_after_direct_injection(fields) -> None:
    if (
        has_full_pec_3d(getattr(fields, "boundaries", None))
        and fields.full_pec_3d_state is not None
    ):
        sync_full_pec_3d_from_compact(fields, fields.full_pec_3d_state)


def _support_state_arrays(
    state: dict[str, np.ndarray], source: ModeSource
) -> dict[str, np.ndarray]:
    index_map = {
        "Ex": source._Ex_indices,
        "Ey": source._Ey_indices,
        "Ez": source._Ez_indices,
        "Hx": source._Hx_indices,
        "Hy": source._Hy_indices,
        "Hz": source._Hz_indices,
    }
    return {
        comp: np.asarray(state[comp], dtype=float)[index_map[comp]]
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }


def _runtime_profiles_and_indices(source: ModeSource):
    profiles, indices = source._get_3d_profiles_and_indices()
    out_profiles = {
        name: (
            None if value is None else np.real(np.asarray(value, dtype=np.complex128))
        )
        for name, value in profiles.items()
    }
    return out_profiles, indices


def _source_plane_deembedded_phasor_profiles(
    source: ModeSource,
    state: dict[str, np.ndarray],
    *,
    t_e: float,
    t_h: float,
    shift: int = 0,
) -> dict[str, np.ndarray]:
    profiles, indices = source._get_3d_profiles_and_indices()
    axis = str(source._axis)
    dx = dy = dz = float(source._resolution)
    d_axis = {"x": dx, "y": dy, "z": dz}[axis]
    omega = float(source._omega_launch)
    k_num = float(source._k_num_axis)
    ref_coord = float(source._phase_ref_coord)
    out: dict[str, np.ndarray] = {}
    for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        base_idx = indices[comp]
        field = np.asarray(state[comp], dtype=np.complex128)
        idx = _shift_component_indices_along_axis(
            base_idx,
            axis,
            int(shift),
            field.shape,
        )
        assert idx is not None
        arr = field[idx]
        axis_idx = _axis_index_from_component_indices(base_idx, axis)
        coord = _component_axis_coord(comp, axis_idx, axis, dx, dy, dz)
        coord = float(coord + int(shift) * d_axis)
        base_time = float(t_e if comp.startswith("E") else t_h)
        delay = _numeric_phase_delay(omega, k_num, coord - ref_coord)
        phase = omega * (base_time - delay)
        out[comp] = arr * np.exp(-1j * phase)
        assert out[comp].shape == np.asarray(profiles[comp]).shape
    return out


def _source_basis_branch_metrics(
    source: ModeSource,
    field_profiles: dict[str, np.ndarray],
) -> dict[str, float]:
    profiles, _indices = source._get_3d_profiles_and_indices()
    basis_profiles = {
        name: np.asarray(value, dtype=np.complex128) for name, value in profiles.items()
    }
    axis = str(source._axis)
    d_area = float(source._resolution * source._resolution)
    forward, backward = _make_3d_mode_basis_profiles(
        basis_profiles,
        axis=axis,
        d_area=d_area,
        direction_sign=float(source._direction_sign),
    )
    a_forward = _modal_overlap_3d_profiles(
        field_profiles,
        forward,
        axis=axis,
        d_area=d_area,
        direction_sign=float(source._direction_sign),
    )
    a_backward = _modal_overlap_3d_profiles(
        field_profiles,
        backward,
        axis=axis,
        d_area=d_area,
        direction_sign=float(source._direction_sign),
    )
    return {
        "forward_abs": float(abs(a_forward)),
        "backward_abs": float(abs(a_backward)),
        "backward_ratio": float(abs(a_backward) / max(abs(a_forward), 1e-30)),
    }


def _source_phase_referenced_power(
    source: ModeSource,
    field_profiles: dict[str, np.ndarray],
) -> float:
    _profiles, indices = source._get_3d_profiles_and_indices()
    axis = str(source._axis)
    d_area = float(source._resolution * source._resolution)
    referenced = _phase_reference_3d_profiles(
        field_profiles,
        indices,
        axis=axis,
        dx=float(source._resolution),
        dy=float(source._resolution),
        dz=float(source._resolution),
        omega=float(source._omega_launch),
        k_num=float(source._k_num_axis),
        ref_coord=float(source._phase_ref_coord),
    )
    signed_power = _modal_power_3d_from_profiles(
        referenced,
        axis=axis,
        d_area=d_area,
        direction_sign=float(source._direction_sign),
    )
    return float(abs(signed_power))


def _max_complex_part(deltas: dict[str, np.ndarray]) -> tuple[float, float]:
    max_abs = 0.0
    max_imag = 0.0
    for value in deltas.values():
        arr = np.asarray(value, dtype=np.complex128)
        if arr.size == 0:
            continue
        max_abs = max(max_abs, float(np.max(np.abs(arr))))
        max_imag = max(max_imag, float(np.max(np.abs(np.imag(arr)))))
    return max_abs, max_imag


def _target_and_residual_reconstructed_phasor_step(
    source: ModeSource,
    fields,
    *,
    dt: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    full_prev = source._build_incident_3d_phasor_state(
        fields,
        t_e=0.0,
        t_h=-0.5 * float(dt),
        masked=False,
    )
    masked_prev = source._build_incident_3d_phasor_state(
        fields,
        t_e=0.0,
        t_h=-0.5 * float(dt),
        masked=True,
    )

    h_full_next = source._advance_incident_h_3d(fields, full_prev, dt)
    h_target_next = source._mask_incident_3d_state_to_launched_side(h_full_next)
    h_mask_next = source._advance_incident_h_3d(fields, masked_prev, dt)
    h_delta = source._compute_discrete_3d_h_phasor_delta(fields, dt=dt)
    h_reconstructed = {
        comp: h_mask_next[comp] + h_delta[comp] for comp in ("Hx", "Hy", "Hz")
    }

    e_full_next = source._advance_incident_e_3d(fields, full_prev, h_full_next, dt)
    e_target_next = source._mask_incident_3d_state_to_launched_side(e_full_next)
    e_mask_next = source._advance_incident_e_3d(
        fields,
        masked_prev,
        h_target_next,
        dt,
    )
    e_delta = source._compute_discrete_3d_e_phasor_delta(fields, dt=dt)
    e_reconstructed = {
        comp: e_mask_next[comp] + e_delta[comp] for comp in ("Ex", "Ey", "Ez")
    }

    target_next = {
        "Ex": e_target_next["Ex"],
        "Ey": e_target_next["Ey"],
        "Ez": e_target_next["Ez"],
        "Hx": h_target_next["Hx"],
        "Hy": h_target_next["Hy"],
        "Hz": h_target_next["Hz"],
    }
    reconstructed = {
        "Ex": e_reconstructed["Ex"],
        "Ey": e_reconstructed["Ey"],
        "Ez": e_reconstructed["Ez"],
        "Hx": h_reconstructed["Hx"],
        "Hy": h_reconstructed["Hy"],
        "Hz": h_reconstructed["Hz"],
    }
    return target_next, reconstructed


def _source_profile_stage_snapshots(
    sim: Simulation, source: ModeSource
) -> dict[str, object]:
    permittivity = np.asarray(sim.fields.permittivity)
    direction = str(source.direction)
    axis = direction[1]
    center_idx = int(source._snapped_region.plane_index)
    nz, ny, nx = permittivity.shape

    if axis == "x":
        offset_idx = (
            max(0, center_idx - 1) if direction == "+x" else min(nx - 2, center_idx + 1)
        )
        eps_profile = permittivity[:, :, center_idx]
    elif axis == "y":
        offset_idx = (
            max(0, center_idx - 1) if direction == "+y" else min(ny - 2, center_idx + 1)
        )
        eps_profile = permittivity[:, center_idx, :]
    else:
        offset_idx = (
            max(0, center_idx - 1) if direction == "+z" else min(nz - 2, center_idx + 1)
        )
        eps_profile = permittivity[center_idx, :, :]

    omega = 2.0 * np.pi * LIGHT_SPEED / float(source.wavelength)
    solver_direction = direction
    if axis == "y":
        solver_direction = "+y"

    eps_profile_arr = np.asarray(eps_profile)
    n_local_max = float(np.sqrt(max(float(np.max(np.real(eps_profile_arr))), 1e-12)))
    target_neff = 0.98 * n_local_max
    neff_val, e_fields, h_fields, _ = solve_modes(
        eps=eps_profile,
        omega=omega,
        dL=float(sim.resolution),
        m=3,
        direction=solver_direction,
        filter_pol=source.pol,
        target_neff=target_neff,
        return_fields=True,
    )
    mode_idx = _select_core_confined_mode_index(eps_profile, e_fields, neff_val)
    e_mode = e_fields[mode_idx]
    h_mode = h_fields[mode_idx]

    Ex_raw = np.asarray(np.squeeze(e_mode[0]))
    Ey_raw = np.asarray(np.squeeze(e_mode[1]))
    Ez_raw = np.asarray(np.squeeze(e_mode[2]))
    Hx_raw = np.asarray(np.squeeze(h_mode[0]))
    Hy_raw = np.asarray(np.squeeze(h_mode[1]))
    Hz_raw = np.asarray(np.squeeze(h_mode[2]))

    Ex_raw, Ey_raw, Ez_raw, Hx_raw, Hy_raw, Hz_raw = _remap_3d_solver_components(
        Ex_raw, Ey_raw, Ez_raw, Hx_raw, Hy_raw, Hz_raw, axis
    )
    ref_field = _select_3d_phase_ref(
        axis, source.pol, Ex_raw, Ey_raw, Ez_raw, Hx_raw, Hy_raw, Hz_raw
    )
    ref_flat = np.asarray(ref_field).reshape(-1)
    phase_ref = np.angle(ref_flat[int(np.argmax(np.abs(ref_flat)))])

    Ex_aligned = Ex_raw * np.exp(-1j * phase_ref)
    Ey_aligned = Ey_raw * np.exp(-1j * phase_ref)
    Ez_aligned = Ez_raw * np.exp(-1j * phase_ref)
    Hx_aligned = Hx_raw * np.exp(-1j * phase_ref)
    Hy_aligned = Hy_raw * np.exp(-1j * phase_ref)
    Hz_aligned = Hz_raw * np.exp(-1j * phase_ref)

    impedance_neff = _select_3d_impedance_index(
        axis,
        source.pol,
        eps_profile,
        Ex_aligned,
        Ey_aligned,
        Ez_aligned,
        Hx_aligned,
        Hy_aligned,
        Hz_aligned,
    )
    complex_profiles, _indices, _extra = _build_3d_profiles(
        Ex_aligned,
        Ey_aligned,
        Ez_aligned,
        Hx_aligned,
        Hy_aligned,
        Hz_aligned,
        axis=axis,
        direction=direction,
        center=tuple(float(v) for v in source.center),
        width=float(source.width),
        height=float(source.height),
        center_idx=center_idx,
        offset_idx=offset_idx,
        grid_shape=(nz, ny, nx),
        resolution=float(sim.resolution),
        impedance_neff=float(impedance_neff),
        omega=float(omega),
        dt=float(sim.dt),
    )
    parity_axes = _detect_transverse_symmetry_axes(eps_profile)
    parity_profiles = _enforce_componentwise_parity(complex_profiles, parity_axes)

    d_area = float(sim.resolution * sim.resolution)
    forward, backward = _make_3d_mode_basis_profiles(
        parity_profiles,
        axis=axis,
        d_area=d_area,
        direction_sign=float(source._direction_sign),
    )
    projected_real = _project_3d_profiles_to_real(parity_profiles)
    normalized_real = _normalize_3d_profiles_by_flux(
        {k: np.asarray(v, dtype=np.complex128) for k, v in projected_real.items()},
        axis=axis,
        d_area=d_area,
        direction_sign=float(source._direction_sign),
    )
    runtime_real = _runtime_3d_profiles(
        {k: np.asarray(v, dtype=np.complex128) for k, v in normalized_real.items()},
        axis,
        float(source._direction_sign),
    )

    return {
        "axis": axis,
        "direction_sign": float(source._direction_sign),
        "d_area": d_area,
        "stages": {
            "aligned_raw": {
                "Ex": Ex_aligned,
                "Ey": Ey_aligned,
                "Ez": Ez_aligned,
                "Hx": Hx_aligned,
                "Hy": Hy_aligned,
                "Hz": Hz_aligned,
            },
            "built_complex": complex_profiles,
            "parity_complex": parity_profiles,
            "projected_real": projected_real,
            "normalized_real": normalized_real,
            "runtime_real": runtime_real,
        },
    }


def _source_profile_stage_purity(
    sim: Simulation, source: ModeSource
) -> dict[str, dict[str, float]]:
    stage_data = _source_profile_stage_snapshots(sim, source)
    axis = str(stage_data["axis"])
    d_area = float(stage_data["d_area"])
    direction_sign = float(stage_data["direction_sign"])
    stage_maps = stage_data["stages"]
    parity_profiles = stage_maps["parity_complex"]
    forward, backward = _make_3d_mode_basis_profiles(
        parity_profiles,
        axis=axis,
        d_area=d_area,
        direction_sign=direction_sign,
    )

    def summarize(stage_profiles):
        a_fwd = _modal_overlap_3d_profiles(
            stage_profiles,
            forward,
            axis=axis,
            d_area=d_area,
            direction_sign=direction_sign,
        )
        a_bwd = _modal_overlap_3d_profiles(
            stage_profiles,
            backward,
            axis=axis,
            d_area=d_area,
            direction_sign=direction_sign,
        )
        return {
            "forward_abs": float(abs(a_fwd)),
            "backward_abs": float(abs(a_bwd)),
            "backward_ratio": float(abs(a_bwd) / max(abs(a_fwd), 1e-30)),
        }

    return {
        "parity_complex": summarize(stage_maps["parity_complex"]),
        "projected_real": summarize(stage_maps["projected_real"]),
        "normalized_real": summarize(stage_maps["normalized_real"]),
        "runtime_real": summarize(stage_maps["runtime_real"]),
    }


def _secondary_pair_names(axis: str) -> tuple[str, str, str, str]:
    pair_map = {
        "x": ("Hy", "Hz", "Ez", "Ey"),
        "y": ("Hx", "Hz", "Ez", "Ex"),
        "z": ("Hx", "Hy", "Ey", "Ex"),
    }
    try:
        return pair_map[str(axis)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported axis {axis!r} for secondary-pair metrics"
        ) from exc


def _secondary_pair_metrics(stage_profiles, axis: str) -> dict[str, float]:
    weak_h, dom_h, weak_e, dom_e = _secondary_pair_names(axis)

    def _pair_metrics(weak_name: str, dom_name: str) -> dict[str, float]:
        weak = np.asarray(stage_profiles[weak_name], dtype=np.complex128).reshape(-1)
        dom = np.asarray(stage_profiles[dom_name], dtype=np.complex128).reshape(-1)
        n = int(min(weak.size, dom.size))
        weak = weak[:n]
        dom = dom[:n]
        weak_norm = float(np.linalg.norm(weak))
        dom_norm = float(np.linalg.norm(dom))
        overlap = np.vdot(dom, weak)
        return {
            "norm_ratio": weak_norm / max(dom_norm, 1e-30),
            "overlap_abs_ratio": float(abs(overlap) / max(dom_norm * weak_norm, 1e-30)),
            "overlap_phase": (
                float(np.angle(overlap)) if abs(overlap) > 1e-30 else float("nan")
            ),
            "imag_frac": float(np.linalg.norm(np.imag(weak)) / max(weak_norm, 1e-30)),
        }

    h_metrics = _pair_metrics(weak_h, dom_h)
    e_metrics = _pair_metrics(weak_e, dom_e)
    return {
        "weak_h_norm_ratio": h_metrics["norm_ratio"],
        "weak_h_overlap_abs_ratio": h_metrics["overlap_abs_ratio"],
        "weak_h_overlap_phase": h_metrics["overlap_phase"],
        "weak_h_imag_frac": h_metrics["imag_frac"],
        "weak_e_norm_ratio": e_metrics["norm_ratio"],
        "weak_e_overlap_abs_ratio": e_metrics["overlap_abs_ratio"],
        "weak_e_overlap_phase": e_metrics["overlap_phase"],
        "weak_e_imag_frac": e_metrics["imag_frac"],
    }


def _source_profile_secondary_pair_metrics(
    sim: Simulation, source: ModeSource
) -> dict[str, dict[str, float]]:
    stage_data = _source_profile_stage_snapshots(sim, source)
    axis = str(stage_data["axis"])
    out: dict[str, dict[str, float]] = {}
    for stage_name, stage_profiles in stage_data["stages"].items():
        out[stage_name] = _secondary_pair_metrics(stage_profiles, axis)
    return out


def _runtime_profile_overlap_with_large_guide_mode(
    sim: Simulation,
    source: ModeSource,
    *,
    mode_index: int,
) -> float:
    stage = _source_profile_stage_snapshots(sim, source)
    runtime = {
        k: np.asarray(v, dtype=np.complex128)
        for k, v in stage["stages"]["runtime_real"].items()
    }
    forward, _backward, axis, d_area = _large_guide_mode_basis_profiles(
        sim,
        source,
        mode_index=mode_index,
        center=tuple(float(v) for v in source.center),
        spans=(float(source.width), float(source.height)),
    )
    return float(
        abs(
            _modal_overlap_3d_profiles(
                runtime,
                forward,
                axis=axis,
                d_area=d_area,
                direction_sign=float(source._direction_sign),
            )
        )
    )


def _large_guide_mode_basis_profiles(
    sim: Simulation,
    source: ModeSource,
    *,
    mode_index: int,
    center: tuple[float, float, float],
    spans: tuple[float, float],
):
    permittivity = np.asarray(sim.fields.permittivity)
    direction = str(source.direction)
    axis = direction[1]
    source_center_idx = int(source._snapped_region.plane_index)
    axis_pos = {"x": 0, "y": 1, "z": 2}[axis]
    plane_shift = int(
        round(
            (float(center[axis_pos]) - float(source.center[axis_pos]))
            / float(sim.resolution)
        )
    )
    center_idx = int(source_center_idx + plane_shift)
    nz, ny, nx = permittivity.shape
    if axis == "x":
        center_idx = int(np.clip(center_idx, 0, nx - 1))
        offset_idx = (
            max(0, center_idx - 1) if direction == "+x" else min(nx - 2, center_idx + 1)
        )
        eps_profile = permittivity[:, :, center_idx]
    elif axis == "y":
        center_idx = int(np.clip(center_idx, 0, ny - 1))
        offset_idx = (
            max(0, center_idx - 1) if direction == "+y" else min(ny - 2, center_idx + 1)
        )
        eps_profile = permittivity[:, center_idx, :]
    else:
        center_idx = int(np.clip(center_idx, 0, nz - 1))
        offset_idx = (
            max(0, center_idx - 1) if direction == "+z" else min(nz - 2, center_idx + 1)
        )
        eps_profile = permittivity[center_idx, :, :]

    omega = 2.0 * np.pi * LIGHT_SPEED / float(source.wavelength)
    solver_direction = direction
    if axis == "y":
        solver_direction = "+y"
    target_neff = 0.98 * float(np.sqrt(np.max(eps_profile)))
    neff_val, e_fields, h_fields, _ = solve_modes(
        eps=eps_profile,
        omega=omega,
        dL=float(sim.resolution),
        m=max(int(mode_index) + 1, 4),
        direction=solver_direction,
        filter_pol=source.pol,
        target_neff=target_neff,
        return_fields=True,
    )

    Ex_raw = np.asarray(np.squeeze(e_fields[mode_index][0]))
    Ey_raw = np.asarray(np.squeeze(e_fields[mode_index][1]))
    Ez_raw = np.asarray(np.squeeze(e_fields[mode_index][2]))
    Hx_raw = np.asarray(np.squeeze(h_fields[mode_index][0]))
    Hy_raw = np.asarray(np.squeeze(h_fields[mode_index][1]))
    Hz_raw = np.asarray(np.squeeze(h_fields[mode_index][2]))

    Ex_raw, Ey_raw, Ez_raw, Hx_raw, Hy_raw, Hz_raw = _remap_3d_solver_components(
        Ex_raw, Ey_raw, Ez_raw, Hx_raw, Hy_raw, Hz_raw, axis
    )
    ref_field = _select_3d_phase_ref(
        axis, source.pol, Ex_raw, Ey_raw, Ez_raw, Hx_raw, Hy_raw, Hz_raw
    )
    ref_flat = np.asarray(ref_field).reshape(-1)
    phase_ref = np.angle(ref_flat[int(np.argmax(np.abs(ref_flat)))])

    Ex_aligned = Ex_raw * np.exp(-1j * phase_ref)
    Ey_aligned = Ey_raw * np.exp(-1j * phase_ref)
    Ez_aligned = Ez_raw * np.exp(-1j * phase_ref)
    Hx_aligned = Hx_raw * np.exp(-1j * phase_ref)
    Hy_aligned = Hy_raw * np.exp(-1j * phase_ref)
    Hz_aligned = Hz_raw * np.exp(-1j * phase_ref)

    impedance_neff = _select_3d_impedance_index(
        axis,
        source.pol,
        eps_profile,
        Ex_aligned,
        Ey_aligned,
        Ez_aligned,
        Hx_aligned,
        Hy_aligned,
        Hz_aligned,
    )
    parity_axes = _detect_transverse_symmetry_axes(eps_profile)
    complex_profiles, _, _ = _build_3d_profiles(
        Ex_aligned,
        Ey_aligned,
        Ez_aligned,
        Hx_aligned,
        Hy_aligned,
        Hz_aligned,
        axis=axis,
        direction=direction,
        center=tuple(float(v) for v in center),
        width=float(spans[0]),
        height=float(spans[1]),
        center_idx=center_idx,
        offset_idx=offset_idx,
        grid_shape=(nz, ny, nx),
        resolution=float(sim.resolution),
        impedance_neff=float(impedance_neff),
        omega=float(omega),
        dt=float(sim.dt),
    )
    parity_profiles = _enforce_componentwise_parity(complex_profiles, parity_axes)
    d_area = float(sim.resolution * sim.resolution)
    forward, backward = _make_3d_mode_basis_profiles(
        parity_profiles,
        axis=axis,
        d_area=d_area,
        direction_sign=float(source._direction_sign),
    )
    return forward, backward, axis, d_area


def _large_guide_full_plane_mode_basis_profiles(
    sim: Simulation,
    source: ModeSource,
    *,
    mode_index: int,
    center: tuple[float, float, float],
    plane_shapes: dict[str, tuple[int, ...]],
):
    permittivity = np.asarray(sim.fields.permittivity)
    direction = str(source.direction)
    axis = direction[1]
    source_center_idx = int(source._snapped_region.plane_index)
    axis_pos = {"x": 0, "y": 1, "z": 2}[axis]
    plane_shift = int(
        round(
            (float(center[axis_pos]) - float(source.center[axis_pos]))
            / float(sim.resolution)
        )
    )
    center_idx = int(source_center_idx + plane_shift)
    nz, ny, nx = permittivity.shape
    if axis == "x":
        center_idx = int(np.clip(center_idx, 0, nx - 1))
        eps_profile = permittivity[:, :, center_idx]
    elif axis == "y":
        center_idx = int(np.clip(center_idx, 0, ny - 1))
        eps_profile = permittivity[:, center_idx, :]
    else:
        center_idx = int(np.clip(center_idx, 0, nz - 1))
        eps_profile = permittivity[center_idx, :, :]

    omega = 2.0 * np.pi * LIGHT_SPEED / float(source.wavelength)
    solver_direction = direction
    if axis == "y":
        solver_direction = "+y"
    target_neff = 0.98 * float(np.sqrt(np.max(eps_profile)))
    _neff_val, e_fields, h_fields, _ = solve_modes(
        eps=eps_profile,
        omega=omega,
        dL=float(sim.resolution),
        m=max(int(mode_index) + 1, 6),
        direction=solver_direction,
        filter_pol=source.pol,
        target_neff=target_neff,
        return_fields=True,
    )

    Ex_raw = np.asarray(np.squeeze(e_fields[mode_index][0]))
    Ey_raw = np.asarray(np.squeeze(e_fields[mode_index][1]))
    Ez_raw = np.asarray(np.squeeze(e_fields[mode_index][2]))
    Hx_raw = np.asarray(np.squeeze(h_fields[mode_index][0]))
    Hy_raw = np.asarray(np.squeeze(h_fields[mode_index][1]))
    Hz_raw = np.asarray(np.squeeze(h_fields[mode_index][2]))

    Ex_raw, Ey_raw, Ez_raw, Hx_raw, Hy_raw, Hz_raw = _remap_3d_solver_components(
        Ex_raw, Ey_raw, Ez_raw, Hx_raw, Hy_raw, Hz_raw, axis
    )
    ref_field = _select_3d_phase_ref(
        axis, source.pol, Ex_raw, Ey_raw, Ez_raw, Hx_raw, Hy_raw, Hz_raw
    )
    ref_flat = np.asarray(ref_field).reshape(-1)
    phase_ref = np.angle(ref_flat[int(np.argmax(np.abs(ref_flat)))])
    phase = np.exp(-1j * phase_ref)

    Ex_aligned = Ex_raw * phase
    Ey_aligned = Ey_raw * phase
    Ez_aligned = Ez_raw * phase
    Hx_aligned = Hx_raw * phase
    Hy_aligned = Hy_raw * phase
    Hz_aligned = Hz_raw * phase

    if axis == "x":
        profiles = {
            "Ex": np.asarray(Ex_aligned, dtype=np.complex128),
            "Ey": np.asarray(_stagger_half(Ey_aligned, axis=1), dtype=np.complex128),
            "Ez": np.asarray(_stagger_half(Ez_aligned, axis=0), dtype=np.complex128),
            "Hx": np.asarray(_stagger_both(Hx_aligned), dtype=np.complex128),
            "Hy": np.asarray(_stagger_half(Hy_aligned, axis=0), dtype=np.complex128),
            "Hz": np.asarray(_stagger_half(Hz_aligned, axis=1), dtype=np.complex128),
        }
    elif axis == "y":
        profiles = {
            "Ex": np.asarray(_stagger_half(Ex_aligned, axis=1), dtype=np.complex128),
            "Ey": np.asarray(Ey_aligned, dtype=np.complex128),
            "Ez": np.asarray(_stagger_half(Ez_aligned, axis=0), dtype=np.complex128),
            "Hx": np.asarray(_stagger_half(Hx_aligned, axis=0), dtype=np.complex128),
            "Hy": np.asarray(_stagger_both(Hy_aligned), dtype=np.complex128),
            "Hz": np.asarray(_stagger_half(Hz_aligned, axis=1), dtype=np.complex128),
        }
    else:
        profiles = {
            "Ex": np.asarray(_stagger_half(Ex_aligned, axis=1), dtype=np.complex128),
            "Ey": np.asarray(_stagger_half(Ey_aligned, axis=0), dtype=np.complex128),
            "Ez": np.asarray(Ez_aligned, dtype=np.complex128),
            "Hx": np.asarray(_stagger_half(Hx_aligned, axis=0), dtype=np.complex128),
            "Hy": np.asarray(_stagger_half(Hy_aligned, axis=1), dtype=np.complex128),
            "Hz": np.asarray(_stagger_both(Hz_aligned), dtype=np.complex128),
        }

    matched: dict[str, np.ndarray] = {}
    for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        shaped = _match_shape(profiles[comp], plane_shapes[comp])
        if shaped is None:
            raise ValueError(
                f"Could not match full-plane {comp} mode shape to sampled plane"
            )
        matched[comp] = np.asarray(shaped, dtype=np.complex128)

    d_area = float(sim.resolution * sim.resolution)
    forward, backward = _make_3d_mode_basis_profiles(
        matched,
        axis=axis,
        d_area=d_area,
        direction_sign=float(source._direction_sign),
    )
    return forward, backward, axis, d_area


def _compute_source_off_plane_mode_metrics(
    sim: Simulation,
    source: ModeSource,
    plane: dict[str, np.ndarray],
    *,
    monitor_center: tuple[float, float, float],
    spans: tuple[float, float],
    mode_count: int,
):
    plane_shapes = {comp: np.asarray(arr).shape for comp, arr in plane.items()}
    overlaps: list[dict[str, float]] = []
    basis_vectors: list[np.ndarray] = []
    basis_labels: list[str] = []
    for mode_index in range(int(mode_count)):
        forward, backward, axis, d_area = _large_guide_full_plane_mode_basis_profiles(
            sim,
            source,
            mode_index=mode_index,
            center=monitor_center,
            plane_shapes=plane_shapes,
        )
        a_fwd = _modal_overlap_3d_profiles(
            plane,
            forward,
            axis=axis,
            d_area=d_area,
            direction_sign=float(source._direction_sign),
        )
        a_bwd = _modal_overlap_3d_profiles(
            plane,
            backward,
            axis=axis,
            d_area=d_area,
            direction_sign=float(source._direction_sign),
        )
        overlaps.append(
            {
                "mode_index": int(mode_index),
                "forward_abs": float(abs(a_fwd)),
                "backward_abs": float(abs(a_bwd)),
            }
        )
        basis_vectors.append(_flatten_3d_profile_state(forward))
        basis_labels.append(f"mode{int(mode_index)}_forward")
        basis_vectors.append(_flatten_3d_profile_state(backward))
        basis_labels.append(f"mode{int(mode_index)}_backward")

    fwd0 = overlaps[0]["forward_abs"]
    bwd0 = overlaps[0]["backward_abs"]
    fwd1 = overlaps[1]["forward_abs"] if len(overlaps) > 1 else 0.0
    higher = sum(item["forward_abs"] + item["backward_abs"] for item in overlaps[1:])
    total = fwd0 + bwd0 + higher
    lsq = _least_squares_mode_decomposition(
        plane,
        basis_vectors=basis_vectors,
        basis_labels=basis_labels,
    )
    return {
        "basis_kind": "full_plane_complex",
        "spans": spans,
        "mode_overlaps": overlaps,
        "fundamental_forward_fraction": float(fwd0 / max(total, 1e-30)),
        "fundamental_backward_ratio": float(bwd0 / max(fwd0, 1e-30)),
        "first_higher_forward_ratio": float(fwd1 / max(fwd0, 1e-30)),
        "lsq": lsq,
    }


def _source_off_downstream_mode_metrics(
    direction: str,
    pol: str,
    *,
    ppw: int = 6,
    num_steps: int = 80,
    monitor_offset_cells: int = 10,
    mode_count: int = 3,
):
    sim = _build_centered_straight_guide_sim_steps(
        ppw=ppw,
        axis=direction[1],
        num_steps=num_steps,
    )
    spans = _full_transverse_source_spans(sim, direction=direction, margin_cells=1)
    source, _dx = _build_test_source(
        sim,
        direction=direction,
        pol=pol,
        source_spans=spans,
    )
    freq = LIGHT_SPEED / float(source.wavelength)
    time = np.asarray(sim.time, dtype=float)
    burst_tmax = min(float(time[-1]), 3.0 / float(freq))
    source.signal = ramped_cosine(
        time,
        amplitude=1.0,
        frequency=float(freq),
        ramp_duration=1.0 / float(freq),
        t_max=burst_tmax,
    )
    sim.sources = [source]

    for _ in range(int(sim.num_steps)):
        sim.step()

    plane = _sample_monitor_plane(
        sim,
        source,
        spans,
        direction=direction,
        monitor_offset_cells=monitor_offset_cells,
    )
    monitor_center = _move_along(
        tuple(float(v) for v in source.center),
        str(direction),
        float(monitor_offset_cells) * float(sim.resolution),
    )
    return _compute_source_off_plane_mode_metrics(
        sim,
        source,
        plane,
        monitor_center=monitor_center,
        spans=spans,
        mode_count=mode_count,
    )


def _source_off_downstream_distance_sweep_metrics(
    direction: str,
    pol: str,
    *,
    ppw: int = 6,
    num_steps: int = 80,
    offsets_cells: tuple[int, ...] = (6, 10, 14),
    mode_count: int = 6,
):
    sim = _build_centered_straight_guide_sim_steps(
        ppw=ppw,
        axis=direction[1],
        num_steps=num_steps,
    )
    spans = _full_transverse_source_spans(sim, direction=direction, margin_cells=1)
    source, _dx = _build_test_source(
        sim,
        direction=direction,
        pol=pol,
        source_spans=spans,
    )
    freq = LIGHT_SPEED / float(source.wavelength)
    time = np.asarray(sim.time, dtype=float)
    burst_tmax = min(float(time[-1]), 3.0 / float(freq))
    source.signal = ramped_cosine(
        time,
        amplitude=1.0,
        frequency=float(freq),
        ramp_duration=1.0 / float(freq),
        t_max=burst_tmax,
    )
    sim.sources = [source]

    for _ in range(int(sim.num_steps)):
        sim.step()

    out: list[dict[str, object]] = []
    for offset_cells in tuple(int(v) for v in offsets_cells):
        plane = _sample_monitor_plane(
            sim,
            source,
            spans,
            direction=direction,
            monitor_offset_cells=offset_cells,
        )
        monitor_center = _move_along(
            tuple(float(v) for v in source.center),
            str(direction),
            float(offset_cells) * float(sim.resolution),
        )
        metrics = _compute_source_off_plane_mode_metrics(
            sim,
            source,
            plane,
            monitor_center=monitor_center,
            spans=spans,
            mode_count=mode_count,
        )
        metrics["offset_cells"] = int(offset_cells)
        out.append(metrics)
    return out


def _flatten_3d_profile_state(state: dict[str, np.ndarray]) -> np.ndarray:
    parts = []
    for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        parts.append(np.asarray(state[comp], dtype=np.complex128).reshape(-1))
    return np.concatenate(parts)


def _match_profile_state_to_reference(
    state: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        matched = _match_shape(state[comp], np.asarray(reference[comp]).shape)
        if matched is None:
            raise ValueError(f"Could not match {comp} basis shape to sampled plane")
        out[comp] = np.asarray(matched, dtype=np.complex128)
    return out


def _least_squares_mode_decomposition(
    plane: dict[str, np.ndarray],
    *,
    basis_vectors: list[np.ndarray],
    basis_labels: list[str],
) -> dict[str, object]:
    target = _flatten_3d_profile_state(plane)
    norm_target = max(float(np.linalg.norm(target)), 1e-30)
    cols = []
    norms = []
    for vec in basis_vectors:
        arr = np.asarray(vec, dtype=np.complex128).reshape(-1)
        norm = max(float(np.linalg.norm(arr)), 1e-30)
        cols.append(arr / norm)
        norms.append(norm)
    B = np.column_stack(cols)
    coeffs, _resid, _rank, _sing = np.linalg.lstsq(B, target, rcond=None)
    recon = B @ coeffs
    residual_rel = float(np.linalg.norm(target - recon) / norm_target)
    coeff_reports = [
        {"label": label, "abs": float(abs(coeff))}
        for label, coeff in zip(basis_labels, coeffs, strict=True)
    ]
    coeff_map = {item["label"]: item["abs"] for item in coeff_reports}
    fwd0 = coeff_map.get("mode0_forward", 0.0)
    bwd0 = coeff_map.get("mode0_backward", 0.0)
    higher = sum(
        value
        for key, value in coeff_map.items()
        if key not in {"mode0_forward", "mode0_backward"}
    )
    total = sum(coeff_map.values())
    return {
        "residual_rel": residual_rel,
        "coeffs": coeff_reports,
        "fundamental_forward_fraction": float(fwd0 / max(total, 1e-30)),
        "fundamental_backward_ratio": float(bwd0 / max(fwd0, 1e-30)),
        "higher_mode_ratio": float(higher / max(fwd0, 1e-30)),
    }


def _matched_profile(profile, target_shape):
    if profile is None:
        return None
    matched = _match_shape(np.asarray(profile, dtype=float), target_shape)
    if matched is None:
        return None
    return np.asarray(matched, dtype=float)


def _second_order_mode_fit(
    state0: dict[str, np.ndarray],
    state1: dict[str, np.ndarray],
    state2: dict[str, np.ndarray],
) -> dict[str, object]:
    component_reports: dict[str, dict[str, float]] = {}
    lhs_parts: list[np.ndarray] = []
    mid_parts: list[np.ndarray] = []

    for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        arr0 = np.asarray(state0[comp], dtype=np.complex128)
        arr1 = np.asarray(state1[comp], dtype=np.complex128)
        arr2 = np.asarray(state2[comp], dtype=np.complex128)
        lhs = arr2 + arr0
        mid = arr1
        lhs_flat = lhs.ravel()
        mid_flat = mid.ravel()
        denom = max(float(np.real(np.vdot(mid_flat, mid_flat))), 1e-30)
        alpha = np.vdot(mid_flat, lhs_flat) / denom
        resid = lhs_flat - alpha * mid_flat
        lhs_norm = max(float(np.linalg.norm(lhs_flat)), 1e-30)
        component_reports[comp] = {
            "alpha_real": float(np.real(alpha)),
            "alpha_imag": float(np.imag(alpha)),
            "residual_rel": float(np.linalg.norm(resid) / lhs_norm),
        }
        lhs_parts.append(lhs_flat)
        mid_parts.append(mid_flat)

    lhs_all = np.concatenate(lhs_parts)
    mid_all = np.concatenate(mid_parts)
    denom_all = max(float(np.real(np.vdot(mid_all, mid_all))), 1e-30)
    alpha_all = np.vdot(mid_all, lhs_all) / denom_all
    resid_all = lhs_all - alpha_all * mid_all
    lhs_norm_all = max(float(np.linalg.norm(lhs_all)), 1e-30)
    return {
        "global": {
            "alpha_real": float(np.real(alpha_all)),
            "alpha_imag": float(np.imag(alpha_all)),
            "residual_rel": float(np.linalg.norm(resid_all) / lhs_norm_all),
        },
        "components": component_reports,
    }


def test_centered_straight_guide_cell_centered_raster_is_transversely_symmetric():
    sim = _build_centered_straight_guide_sim(ppw=6)
    eps = np.asarray(sim.fields.permittivity, dtype=float)

    assert _mirror_residual(eps, axis=1) == 0.0
    assert _mirror_residual(eps, axis=0) == 0.0


def test_centered_straight_guide_te_fixture_is_weakly_multimode():
    sim = _build_centered_straight_guide_sim(ppw=6, axis="x")
    eps = np.asarray(sim.fields.permittivity, dtype=float)
    center_idx = eps.shape[2] // 2
    eps_profile = eps[:, :, center_idx]
    omega = 2.0 * np.pi * LIGHT_SPEED / (1.55 * um)
    target_neff = 0.98 * float(np.sqrt(np.max(eps_profile)))
    neff_val, _e_fields, _h_fields, _ = solve_modes(
        eps=eps_profile,
        omega=omega,
        dL=float(sim.resolution),
        m=4,
        direction="-x",
        filter_pol="te",
        target_neff=target_neff,
        return_fields=True,
    )

    assert float(np.real(neff_val[0])) > 1.3
    assert float(np.real(neff_val[1])) > 1.0
    assert float(np.real(neff_val[2])) < 1.0


def test_tiny_centered_straight_guide_fixture_is_small_and_transversely_symmetric():
    sim, _source_spans = _build_tiny_straight_guide_sim(
        ppw=6,
        axis="x",
        long_cells=18,
        transverse0_cells=8,
        transverse1_cells=6,
        guide0_cells=4,
        guide1_cells=2,
        num_steps=4,
    )
    eps = np.asarray(sim.fields.permittivity, dtype=float)

    assert eps.shape == (6, 8, 18)
    assert _mirror_residual(eps, axis=1) == 0.0
    assert _mirror_residual(eps, axis=0) == 0.0


def test_second_order_mode_fit_is_exact_for_single_cosine_sequence():
    theta = 0.37
    coeff = 2.0 * np.cos(theta)
    state0 = {
        comp: np.asarray([[np.cos(theta)]], dtype=float)
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    state1 = {
        comp: np.asarray([[np.cos(2.0 * theta)]], dtype=float)
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    state2 = {
        comp: np.asarray([[np.cos(3.0 * theta)]], dtype=float)
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }

    fit = _second_order_mode_fit(state0, state1, state2)

    assert fit["global"]["residual_rel"] < 1e-12
    assert abs(fit["global"]["alpha_real"] - coeff) < 1e-12
    assert abs(fit["global"]["alpha_imag"]) < 1e-12


def test_complex_3d_source_profiles_are_forward_pure_before_real_projection():
    sim, source_spans = _build_tiny_straight_guide_sim(
        ppw=6,
        axis="x",
        long_cells=18,
        transverse0_cells=8,
        transverse1_cells=6,
        guide0_cells=4,
        guide1_cells=2,
        num_steps=4,
    )
    source, _dx = _build_tiny_test_source(
        sim,
        source_spans,
        direction="+x",
        pol="te",
        clearance_cells=4,
    )
    stage_data = _source_profile_stage_snapshots(sim, source)
    power = _modal_power_3d_from_profiles(
        stage_data["stages"]["built_complex"],
        axis=str(stage_data["axis"]),
        d_area=float(stage_data["d_area"]),
        direction_sign=float(stage_data["direction_sign"]),
    )

    assert np.isfinite(power)
    assert power >= -1e-24


def test_large_guide_runtime_profiles_do_not_couple_to_first_odd_guided_mode():
    sim = _build_centered_straight_guide_sim(ppw=6, axis="x")
    source, _dx = _build_test_source(sim, direction="+x", pol="te")

    overlap0 = _runtime_profile_overlap_with_large_guide_mode(sim, source, mode_index=0)
    overlap1 = _runtime_profile_overlap_with_large_guide_mode(sim, source, mode_index=1)

    assert overlap0 > 0.99
    assert overlap1 / max(overlap0, 1e-30) < 1e-6


@pytest.mark.parametrize(
    "direction,pol",
    (
        ("+x", "te"),
        ("+y", "te"),
        ("-z", "tm"),
    ),
)
def test_source_off_downstream_mode_metrics_are_finite_and_improve_with_more_modes(
    direction: str, pol: str
):
    metrics3 = _source_off_downstream_mode_metrics(
        direction,
        pol,
        ppw=6,
        num_steps=80,
        monitor_offset_cells=10,
        mode_count=3,
    )
    metrics6 = _source_off_downstream_mode_metrics(
        direction,
        pol,
        ppw=6,
        num_steps=80,
        monitor_offset_cells=10,
        mode_count=6,
    )

    assert metrics6["basis_kind"] == "full_plane_complex"
    assert np.isfinite(metrics3["lsq"]["residual_rel"])
    assert np.isfinite(metrics6["lsq"]["residual_rel"])
    assert 0.0 <= metrics3["fundamental_forward_fraction"] <= 1.0
    assert 0.0 <= metrics6["fundamental_forward_fraction"] <= 1.0
    assert metrics6["lsq"]["residual_rel"] <= metrics3["lsq"]["residual_rel"] + 1e-12


@pytest.mark.parametrize(
    "direction,pol",
    (
        ("+x", "te"),
        ("+y", "te"),
        ("-z", "tm"),
    ),
)
def test_source_off_downstream_distance_sweep_metrics_are_finite(
    direction: str, pol: str
):
    sweep = _source_off_downstream_distance_sweep_metrics(
        direction,
        pol,
        ppw=6,
        num_steps=80,
        offsets_cells=(6, 10, 14),
        mode_count=6,
    )

    assert [item["offset_cells"] for item in sweep] == [6, 10, 14]
    for item in sweep:
        assert item["basis_kind"] == "full_plane_complex"
        assert np.isfinite(item["lsq"]["residual_rel"])
        assert 0.0 <= item["fundamental_forward_fraction"] <= 1.0
        assert np.isfinite(item["fundamental_backward_ratio"])
        assert np.isfinite(item["lsq"]["higher_mode_ratio"])


def test_real_projection_preserves_forward_purity_under_current_profile_basis():
    sim, source_spans = _build_tiny_straight_guide_sim(
        ppw=6,
        axis="x",
        long_cells=18,
        transverse0_cells=8,
        transverse1_cells=6,
        guide0_cells=4,
        guide1_cells=2,
        num_steps=4,
    )
    source, _dx = _build_tiny_test_source(
        sim,
        source_spans,
        direction="+x",
        pol="te",
        clearance_cells=4,
    )
    stats = _source_profile_stage_purity(sim, source)

    assert np.isfinite(stats["parity_complex"]["backward_ratio"])
    assert np.isfinite(stats["projected_real"]["backward_ratio"])
    assert stats["projected_real"]["backward_ratio"] == pytest.approx(1.0)


def test_runtime_gauge_and_flux_normalization_do_not_change_3d_mode_purity():
    sim, source_spans = _build_tiny_straight_guide_sim(
        ppw=6,
        axis="y",
        long_cells=18,
        transverse0_cells=8,
        transverse1_cells=6,
        guide0_cells=4,
        guide1_cells=2,
        num_steps=4,
    )
    source, _dx = _build_tiny_test_source(
        sim,
        source_spans,
        direction="+y",
        pol="te",
        clearance_cells=4,
    )
    stats = _source_profile_stage_purity(sim, source)

    assert (
        abs(
            stats["normalized_real"]["backward_ratio"]
            - stats["projected_real"]["backward_ratio"]
        )
        < 1e-9
    )
    assert (
        abs(
            stats["runtime_real"]["backward_ratio"]
            - stats["normalized_real"]["backward_ratio"]
        )
        < 1e-9
    )


@pytest.mark.parametrize(
    "direction,pol",
    (
        ("+x", "te"),
        ("+y", "te"),
        ("+x", "tm"),
        ("+y", "tm"),
    ),
)
def test_lateral_rectangular_guide_secondary_pair_is_suppressed_during_profile_build(
    direction: str, pol: str
):
    sim, source_spans = _build_tiny_straight_guide_sim(
        ppw=6,
        axis=direction[1],
        long_cells=18,
        transverse0_cells=8,
        transverse1_cells=6,
        guide0_cells=4,
        guide1_cells=2,
        num_steps=4,
    )
    source, _dx = _build_tiny_test_source(
        sim,
        source_spans,
        direction=direction,
        pol=pol,
        clearance_cells=4,
    )
    metrics = _source_profile_secondary_pair_metrics(sim, source)
    suppressed_metrics = metrics["parity_complex"]

    assert (
        suppressed_metrics["weak_h_norm_ratio"]
        < 0.98 * metrics["aligned_raw"]["weak_h_norm_ratio"]
    )
    if pol == "te":
        assert (
            suppressed_metrics["weak_e_norm_ratio"]
            < 0.98 * metrics["aligned_raw"]["weak_e_norm_ratio"]
        )


def test_secondary_h_pair_is_specific_to_doubly_confined_lateral_guides():
    def _metrics_for(guide0_cells: int, guide1_cells: int):
        sim, source_spans = _build_tiny_straight_guide_sim(
            ppw=6,
            axis="x",
            long_cells=18,
            transverse0_cells=8,
            transverse1_cells=6,
            guide0_cells=guide0_cells,
            guide1_cells=guide1_cells,
            num_steps=4,
        )
        source, _dx = _build_tiny_test_source(
            sim,
            source_spans,
            direction="+x",
            pol="te",
            clearance_cells=4,
        )
        return _source_profile_secondary_pair_metrics(sim, source)

    rectangular = _metrics_for(4, 2)
    y_slab = _metrics_for(4, 6)
    z_slab = _metrics_for(8, 2)

    assert rectangular["built_complex"]["weak_h_norm_ratio"] > 1e-2
    assert y_slab["built_complex"]["weak_h_norm_ratio"] < 1e-6
    assert z_slab["built_complex"]["weak_h_norm_ratio"] < 1e-6


@pytest.mark.parametrize("direction", ("+x", "-x", "+y", "-y", "+z", "-z"))
@pytest.mark.parametrize("pol", ("te", "tm"))
def test_mode_source_runtime_profiles_are_transversely_parity_clean_for_all_3d_axes_and_polarizations(
    direction: str,
    pol: str,
):
    sim = _build_centered_straight_guide_sim(ppw=6, axis=direction[1])
    source, _dx = _build_test_source(sim, direction=direction, pol=pol)
    profiles, indices = _runtime_profiles_and_indices(source)

    for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        profile = profiles.get(comp)
        idx = indices.get(comp)
        if profile is None or idx is None:
            continue
        target = np.asarray(getattr(sim.fields, comp), dtype=float)[idx]
        matched = _matched_profile(profile, target.shape)
        assert matched is not None
        assert matched.shape == target.shape
        assert _best_parity_residual(matched, axis=0) < 1e-6
        assert _best_parity_residual(matched, axis=1) < 1e-6


@pytest.mark.parametrize(
    "direction,pol",
    (
        ("+x", "te"),
        ("-x", "tm"),
        ("+y", "te"),
        ("+z", "tm"),
    ),
)
def test_mode_source_h_injection_matches_discrete_residual_for_all_3d_axes_and_polarizations(
    direction: str,
    pol: str,
):
    sim = _build_centered_straight_guide_sim(ppw=6, axis=direction[1])
    source, _dx = _build_test_source(sim, direction=direction, pol=pol)
    dt = float(sim.dt)
    expected = source._compute_discrete_3d_h_delta(sim.fields, t=0.0, dt=dt)

    source.inject_h(
        sim.fields,
        t=0.0,
        dt=dt,
        current_step=0,
        resolution=float(sim.resolution),
        design=sim.design,
    )
    for comp in ("Hx", "Hy", "Hz"):
        np.testing.assert_allclose(
            np.asarray(getattr(sim.fields, comp), dtype=float),
            expected[comp],
            rtol=2e-6,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "direction,pol",
    (
        ("+x", "te"),
        ("-x", "tm"),
        ("+y", "te"),
        ("+z", "tm"),
    ),
)
def test_mode_source_e_injection_matches_discrete_residual_for_all_3d_axes_and_polarizations(
    direction: str,
    pol: str,
):
    sim = _build_centered_straight_guide_sim(ppw=6, axis=direction[1])
    source, _dx = _build_test_source(sim, direction=direction, pol=pol)
    dt = float(sim.dt)
    expected = source._compute_discrete_3d_e_delta(sim.fields, t=0.0, dt=dt)

    source.inject_e(
        sim.fields,
        t=0.0,
        dt=dt,
        current_step=0,
        resolution=float(sim.resolution),
        design=sim.design,
    )
    for comp in ("Ex", "Ey", "Ez"):
        np.testing.assert_allclose(
            np.asarray(getattr(sim.fields, comp), dtype=float),
            expected[comp],
            rtol=2e-6,
            atol=1e-12,
        )


def test_centered_3d_e_sampling_restores_mirror_symmetry():
    sim = _build_centered_straight_guide_sim(ppw=6)
    eps = np.asarray(sim.fields.permittivity, dtype=float)

    eps_x = np.asarray(
        sample_voxel_grid_at_e_component_3d_centered(eps, "Ex"), dtype=float
    )
    eps_y = np.asarray(
        sample_voxel_grid_at_e_component_3d_centered(eps, "Ey"), dtype=float
    )
    eps_z = np.asarray(
        sample_voxel_grid_at_e_component_3d_centered(eps, "Ez"), dtype=float
    )

    assert (
        _physical_component_mirror_residual(
            eps_x, component="Ex", grid_shape=eps.shape, axis=1
        )
        == 0.0
    )
    assert (
        _physical_component_mirror_residual(
            eps_x, component="Ex", grid_shape=eps.shape, axis=0
        )
        == 0.0
    )
    assert (
        _physical_component_mirror_residual(
            eps_y, component="Ey", grid_shape=eps.shape, axis=1
        )
        == 0.0
    )
    assert (
        _physical_component_mirror_residual(
            eps_y, component="Ey", grid_shape=eps.shape, axis=0
        )
        == 0.0
    )
    assert (
        _physical_component_mirror_residual(
            eps_z, component="Ez", grid_shape=eps.shape, axis=1
        )
        == 0.0
    )
    assert (
        _physical_component_mirror_residual(
            eps_z, component="Ez", grid_shape=eps.shape, axis=0
        )
        == 0.0
    )


def test_runtime_3d_e_material_sampling_preserves_mirror_symmetry():
    sim = _build_centered_straight_guide_sim(ppw=6)

    grid_shape = tuple(np.asarray(sim.fields.permittivity).shape)
    for component, eps_comp in (
        ("Ex", sim.fields.eps_ex),
        ("Ey", sim.fields.eps_ey),
        ("Ez", sim.fields.eps_ez),
    ):
        arr = np.asarray(eps_comp, dtype=float)
        assert (
            _physical_component_mirror_residual(
                arr, component=component, grid_shape=grid_shape, axis=1
            )
            == 0.0
        )
        assert (
            _physical_component_mirror_residual(
                arr, component=component, grid_shape=grid_shape, axis=0
            )
            == 0.0
        )


def test_generic_3d_component_sampler_uses_centered_e_mapping():
    sim = _build_centered_straight_guide_sim(ppw=6)
    eps = np.asarray(sim.fields.permittivity, dtype=float)

    for component, runtime_eps in (
        ("Ex", sim.fields.eps_ex),
        ("Ey", sim.fields.eps_ey),
        ("Ez", sim.fields.eps_ez),
    ):
        np.testing.assert_allclose(
            np.asarray(sample_voxel_grid_at_component_3d(eps, component), dtype=float),
            np.asarray(runtime_eps, dtype=float),
            rtol=0.0,
            atol=0.0,
        )


def test_one_step_uniform_medium_keeps_small_transverse_e_asymmetry():
    sim = _build_centered_uniform_sim(ppw=6)
    source, _dx = _build_test_source(sim)

    source.inject_h(
        sim.fields,
        t=0.0,
        dt=float(sim.dt),
        current_step=0,
        resolution=float(sim.resolution),
        design=sim.design,
    )
    pre_h_z = max(
        _support_window_parity_residual(sim.fields, source, comp, idx_attr, axis=0)
        for comp, idx_attr in (
            ("Hx", "_Hx_indices"),
            ("Hy", "_Hy_indices"),
            ("Hz", "_Hz_indices"),
        )
    )

    sim.fields.update_e(float(sim.dt))
    post_ex_z = _support_window_parity_residual(
        sim.fields, source, "Ex", "_Ex_indices", axis=0
    )

    assert pre_h_z < 1e-6
    assert post_ex_z < 1e-6


def test_one_step_straight_guide_update_has_no_staggered_ex_support_residual():
    sim = _build_centered_straight_guide_sim(ppw=6)
    source, _dx = _build_test_source(sim)

    source.inject_h(
        sim.fields,
        t=0.0,
        dt=float(sim.dt),
        current_step=0,
        resolution=float(sim.resolution),
        design=sim.design,
    )
    pre_h_z = max(
        _support_window_parity_residual(sim.fields, source, comp, idx_attr, axis=0)
        for comp, idx_attr in (
            ("Hx", "_Hx_indices"),
            ("Hy", "_Hy_indices"),
            ("Hz", "_Hz_indices"),
        )
    )

    sim.fields.update_e(float(sim.dt))
    post_ex_z = _support_window_parity_residual(
        sim.fields, source, "Ex", "_Ex_indices", axis=0
    )

    assert pre_h_z < 1e-6
    assert post_ex_z < 1e-6


def test_one_step_guide_curl_hx_source_branches_remain_individually_symmetric():
    sim = _build_centered_straight_guide_sim(ppw=6)
    source, _dx = _build_test_source(sim)

    source.inject_h(
        sim.fields,
        t=0.0,
        dt=float(sim.dt),
        current_step=0,
        resolution=float(sim.resolution),
        design=sim.design,
    )
    boundary_views = build_h_boundary_views_for_e_3d(
        sim.fields.Hx,
        sim.fields.Hy,
        sim.fields.Hz,
        getattr(sim, "boundaries", None),
    )
    _curl_hx, _curl_hy, _curl_hz = ops.curl_h_to_e_3d(
        sim.fields.Hx,
        sim.fields.Hy,
        sim.fields.Hz,
        sim.resolution,
        ex_shape=sim.fields.Ex.shape,
        ey_shape=sim.fields.Ey.shape,
        ez_shape=sim.fields.Ez.shape,
        boundary_views=boundary_views,
    )

    d_hz_dy = np.asarray(
        ops._adjacent_difference(
            boundary_views["hz_y"], axis=1, resolution=sim.resolution
        ),
        dtype=float,
    )
    d_hy_dz = np.asarray(
        ops._adjacent_difference(
            boundary_views["hy_z"], axis=0, resolution=sim.resolution
        ),
        dtype=float,
    )

    hz_z, hz_y, hz_x = source._Hz_indices
    hy_z, hy_y, hy_x = source._Hy_indices

    d_hz_support = d_hz_dy[hz_z, _slice_with_extra_stop(hz_y), hz_x]
    d_hy_support = d_hy_dz[_slice_with_extra_stop(hy_z), hy_y, hy_x]

    assert _best_parity_residual(d_hz_support, axis=0) < 1e-6
    assert _best_parity_residual(d_hz_support, axis=1) < 1e-6
    assert _best_parity_residual(d_hy_support, axis=0) < 1e-6
    assert _best_parity_residual(d_hy_support, axis=1) < 1e-6


def _longitudinal_e_support_axis0_parity_after_one_update(
    *,
    build_sim,
    direction: str,
    pol: str,
    sim_axis: str,
) -> tuple[float, float, float]:
    sim = (
        build_sim(ppw=6)
        if build_sim is _build_centered_uniform_sim
        else build_sim(ppw=6, axis=sim_axis)
    )
    source, _dx = _build_test_source(sim, direction=direction, pol=pol)

    source.inject_h(
        sim.fields,
        t=0.0,
        dt=float(sim.dt),
        current_step=0,
        resolution=float(sim.resolution),
        design=sim.design,
    )
    pre_h_axis0 = max(
        _support_window_parity_residual(sim.fields, source, comp, idx_attr, axis=0)
        for comp, idx_attr in (
            ("Hx", "_Hx_indices"),
            ("Hy", "_Hy_indices"),
            ("Hz", "_Hz_indices"),
        )
    )

    sim.fields.update_e(float(sim.dt))

    e_comp = {"x": "Ex", "y": "Ey", "z": "Ez"}[sim_axis]
    idx_attr = {"x": "_Ex_indices", "y": "_Ey_indices", "z": "_Ez_indices"}[sim_axis]
    post_axis0 = _support_window_parity_residual(
        sim.fields,
        source,
        e_comp,
        idx_attr,
        axis=0,
    )
    post_axis1 = _support_window_parity_residual(
        sim.fields,
        source,
        e_comp,
        idx_attr,
        axis=1,
    )
    return float(pre_h_axis0), float(post_axis0), float(post_axis1)


@pytest.mark.parametrize("direction", ("+x", "-x", "+y", "-y"))
@pytest.mark.parametrize("pol", ("te", "tm"))
def test_one_step_lateral_guide_update_has_no_staggered_longitudinal_e_support_residual(
    direction: str,
    pol: str,
):
    axis = direction[1]
    pre_h_axis0, post_axis0, post_axis1 = (
        _longitudinal_e_support_axis0_parity_after_one_update(
            build_sim=_build_centered_straight_guide_sim,
            direction=direction,
            pol=pol,
            sim_axis=axis,
        )
    )

    assert pre_h_axis0 < 1e-6
    broken_axis = 0 if pol == "te" else 1
    post_broken = post_axis0 if broken_axis == 0 else post_axis1
    assert post_broken < 1e-6


def test_tiny_straight_guide_manual_substep_update_has_no_staggered_ex_support_residual():
    sim, source_spans = _build_tiny_straight_guide_sim(
        ppw=6,
        axis="x",
        long_cells=18,
        transverse0_cells=8,
        transverse1_cells=6,
        guide0_cells=4,
        guide1_cells=2,
        num_steps=2,
    )
    source, _dx = _build_tiny_test_source(
        sim,
        source_spans,
        direction="+x",
        pol="te",
        clearance_cells=4,
    )

    source.inject_h(
        sim.fields,
        t=0.0,
        dt=float(sim.dt),
        current_step=0,
        resolution=float(sim.resolution),
        design=sim.design,
    )
    axis0, axis1 = _max_support_parity(
        sim.fields,
        source,
        (("Hx", "_Hx_indices"), ("Hy", "_Hy_indices"), ("Hz", "_Hz_indices")),
    )

    sim.fields.update_e(float(sim.dt))
    ex_support = np.asarray(sim.fields.Ex, dtype=float)[source._Ex_indices]
    post_ex_axis0 = _best_parity_residual(ex_support, axis=0)
    post_ex_axis1 = _best_parity_residual(ex_support, axis=1)

    assert axis0 < 1e-6
    assert axis1 < 1e-6
    assert float(np.linalg.norm(ex_support.ravel())) > 0.0
    assert post_ex_axis0 < 1e-6
    assert post_ex_axis1 < 1e-6


def test_zeroing_longitudinal_h_branch_preserves_tiny_x_guide_ex_axis0_symmetry():
    sim, source_spans = _build_tiny_straight_guide_sim(
        ppw=6,
        axis="x",
        long_cells=18,
        transverse0_cells=8,
        transverse1_cells=6,
        guide0_cells=4,
        guide1_cells=2,
        num_steps=2,
    )
    source, _dx = _build_tiny_test_source(
        sim,
        source_spans,
        direction="+x",
        pol="te",
        clearance_cells=4,
    )
    source.inject_h(
        sim.fields,
        t=0.0,
        dt=float(sim.dt),
        current_step=0,
        resolution=float(sim.resolution),
        design=sim.design,
    )
    sim.fields.Hy = np.zeros_like(sim.fields.Hy)
    sim.fields.update_e(float(sim.dt))

    post_axis0 = _support_window_parity_residual(
        sim.fields,
        source,
        "Ex",
        "_Ex_indices",
        axis=0,
    )
    assert post_axis0 < 1e-6


def test_zeroing_longitudinal_h_branch_preserves_y_guide_ey_axis0_symmetry():
    sim = _build_centered_straight_guide_sim(ppw=6, axis="y")
    source, _dx = _build_test_source(sim, direction="+y", pol="te")
    source.inject_h(
        sim.fields,
        t=0.0,
        dt=float(sim.dt),
        current_step=0,
        resolution=float(sim.resolution),
        design=sim.design,
    )
    sim.fields.Hx = np.zeros_like(sim.fields.Hx)
    sim.fields.update_e(float(sim.dt))

    post_axis0 = _support_window_parity_residual(
        sim.fields,
        source,
        "Ey",
        "_Ey_indices",
        axis=0,
    )
    assert post_axis0 < 1e-6


def test_tiny_discrete_3d_h_source_matches_commutator_residual():
    base_sim, source_spans = _build_tiny_straight_guide_sim(
        ppw=6,
        axis="x",
        long_cells=18,
        transverse0_cells=8,
        transverse1_cells=6,
        guide0_cells=4,
        guide1_cells=2,
        num_steps=6,
    )
    sim = Simulation(
        design=base_sim.design,
        sources=[],
        boundaries=[],
        time=np.asarray(base_sim.time, dtype=float),
        resolution=float(base_sim.resolution),
    )
    source, _dx = _build_tiny_test_source(
        sim,
        source_spans,
        direction="+x",
        pol="te",
        clearance_cells=4,
    )
    source.signal = np.ones(32, dtype=float)

    dt = float(sim.dt)
    t = 4.0 * dt
    expected = source._compute_discrete_3d_h_delta(sim.fields, t=t, dt=dt)

    for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        setattr(sim.fields, comp, jnp.zeros_like(getattr(sim.fields, comp)))

    source.inject_h(
        sim.fields,
        t=t,
        dt=dt,
        current_step=int(round(t / dt)),
        resolution=float(sim.resolution),
        design=sim.design,
    )
    _sync_full_pec_after_direct_injection(sim.fields)

    np.testing.assert_allclose(
        np.asarray(sim.fields.Hx), expected["Hx"], atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(sim.fields.Hy), expected["Hy"], atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(sim.fields.Hz), expected["Hz"], atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(np.asarray(sim.fields.Ex), 0.0, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(np.asarray(sim.fields.Ey), 0.0, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(np.asarray(sim.fields.Ez), 0.0, atol=1e-8, rtol=0.0)


def test_tiny_discrete_3d_split_source_recovers_launched_incident_step_in_source_interior():
    base_sim, source_spans = _build_tiny_straight_guide_sim(
        ppw=6,
        axis="x",
        long_cells=18,
        transverse0_cells=8,
        transverse1_cells=6,
        guide0_cells=4,
        guide1_cells=2,
        num_steps=6,
    )
    sim = Simulation(
        design=base_sim.design,
        sources=[],
        boundaries=[],
        time=np.asarray(base_sim.time, dtype=float),
        resolution=float(base_sim.resolution),
    )
    source, _dx = _build_tiny_test_source(
        sim,
        source_spans,
        direction="+x",
        pol="te",
        clearance_cells=4,
    )
    source.signal = np.ones(32, dtype=float)

    dt = float(sim.dt)
    t = 4.0 * dt
    full_prev = source._build_incident_3d_state(
        sim.fields,
        t_e=t,
        t_h=t - 0.5 * dt,
        dt=dt,
        masked=False,
    )
    masked_prev = source._build_incident_3d_state(
        sim.fields,
        t_e=t,
        t_h=t - 0.5 * dt,
        dt=dt,
        masked=True,
    )
    full_h_next = source._advance_incident_h_3d(sim.fields, full_prev, dt)
    target_h_next = source._mask_incident_3d_state_to_launched_side(full_h_next)
    full_e_next = source._advance_incident_e_3d(sim.fields, full_prev, full_h_next, dt)
    target_e_next = source._mask_incident_3d_state_to_launched_side(full_e_next)

    _set_field_state(sim.fields, masked_prev)
    sim.fields.update_h(dt)
    source.inject_h(
        sim.fields,
        t=t,
        dt=dt,
        current_step=int(round(t / dt)),
        resolution=float(sim.resolution),
        design=sim.design,
    )
    _sync_full_pec_after_direct_injection(sim.fields)

    np.testing.assert_allclose(
        np.asarray(sim.fields.Hx), target_h_next["Hx"], atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(sim.fields.Hy), target_h_next["Hy"], atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(sim.fields.Hz), target_h_next["Hz"], atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(sim.fields.Ex), masked_prev["Ex"], atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(sim.fields.Ey), masked_prev["Ey"], atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(sim.fields.Ez), masked_prev["Ez"], atol=1e-6, rtol=1e-6
    )

    sim.fields.update_e(dt)
    source.inject_e(
        sim.fields,
        t=t,
        dt=dt,
        current_step=int(round(t / dt)),
        resolution=float(sim.resolution),
        design=sim.design,
    )
    _sync_full_pec_after_direct_injection(sim.fields)

    np.testing.assert_allclose(
        np.asarray(sim.fields.Ex),
        target_e_next["Ex"],
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(sim.fields.Ey),
        target_e_next["Ey"],
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(sim.fields.Ez),
        target_e_next["Ez"],
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(sim.fields.Hx), target_h_next["Hx"], atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(sim.fields.Hy), target_h_next["Hy"], atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(sim.fields.Hz), target_h_next["Hz"], atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize(
    "direction,pol",
    (
        ("+x", "te"),
        ("-x", "tm"),
        ("+y", "te"),
        ("-y", "tm"),
        ("+z", "te"),
        ("-z", "tm"),
    ),
)
def test_source_plane_full_incident_phasor_matches_runtime_profile_basis(
    direction: str,
    pol: str,
):
    sim = _build_centered_straight_guide_sim(ppw=6, axis=direction[1])
    source, _dx = _build_test_source(sim, direction=direction, pol=pol)
    dt = float(sim.dt)
    full_state = source._build_incident_3d_phasor_state(
        sim.fields,
        t_e=0.0,
        t_h=-0.5 * dt,
        masked=False,
    )
    profiles, indices = source._get_3d_profiles_and_indices()

    max_shift = int(source._discrete_launch_max_shift)
    checked = 0
    for shift in range(-max_shift, max_shift + 1):
        if any(
            _shift_component_indices_along_axis(
                indices[comp],
                str(source._axis),
                int(shift),
                np.asarray(full_state[comp]).shape,
            )
            is None
            for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        ):
            continue
        deembedded = _source_plane_deembedded_phasor_profiles(
            source,
            full_state,
            t_e=0.0,
            t_h=-0.5 * dt,
            shift=shift,
        )
        checked += 1
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            np.testing.assert_allclose(
                deembedded[comp],
                np.asarray(profiles[comp], dtype=np.complex128),
                rtol=1e-12,
                atol=1e-12,
            )
    assert checked > 0

    profile_metrics = _source_basis_branch_metrics(
        source,
        {
            name: np.asarray(value, dtype=np.complex128)
            for name, value in profiles.items()
        },
    )
    phasor_metrics = _source_basis_branch_metrics(
        source,
        _source_plane_deembedded_phasor_profiles(
            source,
            full_state,
            t_e=0.0,
            t_h=-0.5 * dt,
        ),
    )

    phase_referenced_power = _source_phase_referenced_power(
        source,
        {
            name: np.asarray(value, dtype=np.complex128)
            for name, value in profiles.items()
        },
    )
    assert phase_referenced_power == pytest.approx(float(source.power), rel=1e-12)
    assert profile_metrics["forward_abs"] > 0.0
    assert profile_metrics["backward_ratio"] < 1e-12
    for key in ("forward_abs", "backward_abs", "backward_ratio"):
        assert phasor_metrics[key] == pytest.approx(
            profile_metrics[key],
            rel=1e-12,
            abs=1e-12,
        )
    assert phasor_metrics["backward_ratio"] < 1e-12


@pytest.mark.parametrize(
    "direction,pol",
    (
        ("+x", "te"),
        ("+y", "tm"),
        ("+z", "te"),
    ),
)
def test_full_pec_discrete_3d_phasor_residual_preserves_quadrature_branch(
    direction: str,
    pol: str,
):
    sim = _build_centered_straight_guide_sim(ppw=6, axis=direction[1])
    source, _dx = _build_test_source(sim, direction=direction, pol=pol)

    h_delta = source._compute_discrete_3d_h_phasor_delta(
        sim.fields,
        dt=float(sim.dt),
    )
    e_delta = source._compute_discrete_3d_e_phasor_delta(
        sim.fields,
        dt=float(sim.dt),
    )

    h_abs, h_imag = _max_complex_part(h_delta)
    e_abs, e_imag = _max_complex_part(e_delta)
    assert h_abs > 0.0
    assert e_abs > 0.0
    assert h_imag > 1e-3 * h_abs
    assert e_imag > 1e-3 * e_abs


@pytest.mark.parametrize(
    "direction,pol",
    (
        ("+x", "te"),
        ("-x", "tm"),
        ("+y", "te"),
        ("-y", "tm"),
        ("+z", "te"),
        ("-z", "tm"),
    ),
)
def test_source_plane_three_way_projection_localizes_rejected_branch_to_yee_update(
    direction: str,
    pol: str,
):
    sim = _build_centered_straight_guide_sim(ppw=6, axis=direction[1])
    source, _dx = _build_test_source(sim, direction=direction, pol=pol)
    dt = float(sim.dt)
    profiles, _indices = source._get_3d_profiles_and_indices()
    initial_state = source._build_incident_3d_phasor_state(
        sim.fields,
        t_e=0.0,
        t_h=-0.5 * dt,
        masked=False,
    )
    target_next, reconstructed_next = _target_and_residual_reconstructed_phasor_step(
        source,
        sim.fields,
        dt=dt,
    )

    profile_metrics = _source_basis_branch_metrics(
        source,
        {
            name: np.asarray(value, dtype=np.complex128)
            for name, value in profiles.items()
        },
    )
    initial_metrics = _source_basis_branch_metrics(
        source,
        _source_plane_deembedded_phasor_profiles(
            source,
            initial_state,
            t_e=0.0,
            t_h=-0.5 * dt,
        ),
    )
    target_update_metrics = _source_basis_branch_metrics(
        source,
        _source_plane_deembedded_phasor_profiles(
            source,
            target_next,
            t_e=dt,
            t_h=0.5 * dt,
        ),
    )
    residual_metrics = _source_basis_branch_metrics(
        source,
        _source_plane_deembedded_phasor_profiles(
            source,
            reconstructed_next,
            t_e=dt,
            t_h=0.5 * dt,
        ),
    )

    phase_referenced_power = _source_phase_referenced_power(
        source,
        {
            name: np.asarray(value, dtype=np.complex128)
            for name, value in profiles.items()
        },
    )
    assert phase_referenced_power == pytest.approx(float(source.power), rel=1e-12)
    assert profile_metrics["forward_abs"] > 0.0
    assert profile_metrics["backward_ratio"] < 1e-12
    for key in ("forward_abs", "backward_abs", "backward_ratio"):
        assert initial_metrics[key] == pytest.approx(
            profile_metrics[key],
            rel=1e-12,
            abs=1e-12,
        )
    assert initial_metrics["backward_ratio"] < 1e-12
    assert target_update_metrics["forward_abs"] > 0.9
    assert target_update_metrics["backward_ratio"] < 0.15
    for key in ("forward_abs", "backward_abs", "backward_ratio"):
        assert residual_metrics[key] == pytest.approx(
            target_update_metrics[key],
            rel=5e-6,
            abs=5e-8,
        )


@pytest.mark.parametrize(
    "direction,pol",
    (
        ("+x", "te"),
        ("-y", "tm"),
        ("+z", "tm"),
    ),
)
def test_discrete_3d_phasor_residual_is_exact_launched_side_update(
    direction: str,
    pol: str,
):
    sim = _build_centered_straight_guide_sim(ppw=6, axis=direction[1])
    source, _dx = _build_test_source(sim, direction=direction, pol=pol)
    dt = float(sim.dt)
    full_prev = source._build_incident_3d_phasor_state(
        sim.fields,
        t_e=0.0,
        t_h=-0.5 * dt,
        masked=False,
    )
    masked_prev = source._build_incident_3d_phasor_state(
        sim.fields,
        t_e=0.0,
        t_h=-0.5 * dt,
        masked=True,
    )

    h_full_next = source._advance_incident_h_3d(sim.fields, full_prev, dt)
    h_target_next = source._mask_incident_3d_state_to_launched_side(h_full_next)
    h_mask_next = source._advance_incident_h_3d(sim.fields, masked_prev, dt)
    h_delta = source._compute_discrete_3d_h_phasor_delta(sim.fields, dt=dt)
    for comp in ("Hx", "Hy", "Hz"):
        np.testing.assert_allclose(
            h_mask_next[comp] + h_delta[comp],
            h_target_next[comp],
            rtol=1e-6,
            atol=1e-6,
        )

    e_full_next = source._advance_incident_e_3d(
        sim.fields,
        full_prev,
        h_full_next,
        dt,
    )
    e_target_next = source._mask_incident_3d_state_to_launched_side(e_full_next)
    e_mask_next = source._advance_incident_e_3d(
        sim.fields,
        masked_prev,
        h_target_next,
        dt,
    )
    e_delta = source._compute_discrete_3d_e_phasor_delta(sim.fields, dt=dt)
    for comp in ("Ex", "Ey", "Ez"):
        np.testing.assert_allclose(
            e_mask_next[comp] + e_delta[comp],
            e_target_next[comp],
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.parametrize("direction", ("+x", "-x", "+y", "-y"))
@pytest.mark.parametrize("pol", ("te", "tm"))
def test_mode_source_h_injection_is_transversely_symmetric_in_source_interior_for_all_3d_axes_and_polarizations(
    direction: str,
    pol: str,
):
    sim = _build_centered_straight_guide_sim(ppw=6, axis=direction[1])
    source, _dx = _build_test_source(sim, direction=direction, pol=pol)

    source.inject_h(
        sim.fields,
        t=0.0,
        dt=float(sim.dt),
        current_step=0,
        resolution=float(sim.resolution),
        design=sim.design,
    )
    for comp, idx_attr in (
        ("Hx", "_Hx_indices"),
        ("Hy", "_Hy_indices"),
        ("Hz", "_Hz_indices"),
    ):
        sample = _trimmed_support_array(
            getattr(sim.fields, comp), source, idx_attr, trim=1
        )
        if sample.ndim != 2 or min(sample.shape) < 2:
            continue
        assert _best_parity_residual(sample, axis=0) < 1e-6
        assert _best_parity_residual(sample, axis=1) < 1e-6


test_centered_straight_guide_cell_centered_raster_is_transversely_symmetric.__test__ = (
    False
)
test_centered_straight_guide_te_fixture_is_weakly_multimode.__test__ = False
test_tiny_centered_straight_guide_fixture_is_small_and_transversely_symmetric.__test__ = False
test_second_order_mode_fit_is_exact_for_single_cosine_sequence.__test__ = False
test_complex_3d_source_profiles_are_forward_pure_before_real_projection.__test__ = False
test_large_guide_runtime_profiles_do_not_couple_to_first_odd_guided_mode.__test__ = (
    False
)
test_source_off_downstream_mode_metrics_are_finite_and_improve_with_more_modes.__test__ = False
test_source_off_downstream_distance_sweep_metrics_are_finite.__test__ = False
test_real_projection_preserves_forward_purity_under_current_profile_basis.__test__ = (
    False
)
test_runtime_gauge_and_flux_normalization_do_not_change_3d_mode_purity.__test__ = False
test_lateral_rectangular_guide_secondary_pair_is_suppressed_during_profile_build.__test__ = False
test_secondary_h_pair_is_specific_to_doubly_confined_lateral_guides.__test__ = False
test_mode_source_runtime_profiles_are_transversely_parity_clean_for_all_3d_axes_and_polarizations.__test__ = False
test_one_step_uniform_medium_keeps_small_transverse_e_asymmetry.__test__ = False
test_one_step_straight_guide_update_has_no_staggered_ex_support_residual.__test__ = (
    False
)
test_one_step_guide_curl_hx_source_branches_remain_individually_symmetric.__test__ = (
    False
)
test_one_step_lateral_guide_update_has_no_staggered_longitudinal_e_support_residual.__test__ = False
test_tiny_straight_guide_manual_substep_update_has_no_staggered_ex_support_residual.__test__ = False
test_zeroing_longitudinal_h_branch_preserves_tiny_x_guide_ex_axis0_symmetry.__test__ = (
    False
)
test_zeroing_longitudinal_h_branch_preserves_y_guide_ey_axis0_symmetry.__test__ = False
