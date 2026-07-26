from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal, cast, overload

import numpy as np

from beamz.devices.modes import solve_grid as solve_fdfd_grid
from beamz.devices.modes.specs import ModeData, ModeSpec

from .specs import plane_axis_and_spans

SignedAxis = Literal["+x", "-x", "+y", "-y", "+z", "-z"]


@overload
def solve_modes(
    eps: np.ndarray,
    omega: float,
    dL: float,
    npml: int = 0,
    m: int = 1,
    direction: SignedAxis = "+x",
    filter_pol: Literal["te", "tm"] | None = None,
    return_fields: Literal[True] = True,
    propagation_axis: SignedAxis | None = None,
    target_neff: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]: ...


@overload
def solve_modes(
    eps: np.ndarray,
    omega: float,
    dL: float,
    npml: int = 0,
    m: int = 1,
    direction: SignedAxis = "+x",
    filter_pol: Literal["te", "tm"] | None = None,
    return_fields: Literal[False] = False,
    propagation_axis: SignedAxis | None = None,
    target_neff: float | None = None,
) -> tuple[np.ndarray, np.ndarray]: ...


def solve_modes(
    eps: np.ndarray,
    omega: float,
    dL: float,
    npml: int = 0,
    m: int = 1,
    direction: SignedAxis = "+x",
    filter_pol: Literal["te", "tm"] | None = None,
    return_fields: bool = False,
    propagation_axis: SignedAxis | None = None,
    target_neff: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Solve a BeamZ material profile with one native FDFD eigensolve."""
    eps_array = np.asarray(eps, dtype=np.complex128)
    if eps_array.ndim not in {1, 2}:
        raise ValueError("solve_modes expects a 1D or 2D permittivity array")
    if npml < 0:
        raise ValueError("npml must be non-negative")
    if m <= 0:
        raise ValueError("m must be positive")

    axis_hint = propagation_axis or direction
    axis = str(axis_hint)[-1].lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported propagation axis {axis_hint!r}.")
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]

    plane = eps_array[:, None] if eps_array.ndim == 1 else eps_array
    edges = tuple(
        tuple(np.arange(size + 1, dtype=float) * float(dL) / 1e-6)
        for size in plane.shape
    )
    result = solve_fdfd_grid(
        eps_xx=plane,
        x_edges=edges[0],
        y_edges=edges[1],
        freqs=[float(omega) / (2.0 * np.pi)],
        direction="+" if str(direction).startswith("+") else "-",
        num_modes=2 * int(m) + 5,
        target_neff=target_neff,
        pml=(int(npml), int(npml)),
        normal_axis=cast(Literal[0, 1, 2], axis_index),
    )
    fields = {
        name: _mode_planes(data, axis) for name, data in result.field_components.items()
    }
    order = _mode_order(result.n_complex.values[0], fields, axis, filter_pol)[: int(m)]
    neffs = np.asarray(result.n_complex.values[0, order], dtype=np.complex128)
    electric = np.stack([fields[name][order] for name in ("Ex", "Ey", "Ez")], axis=1)
    magnetic = np.stack([fields[name][order] for name in ("Hx", "Hy", "Hz")], axis=1)
    if axis_index == 1:
        magnetic = -magnetic
    if eps_array.ndim == 1:
        electric = electric[..., 0]
        magnetic = magnetic[..., 0]

    if return_fields:
        return neffs, electric, magnetic, axis_index

    dominant = np.argmax(
        np.linalg.norm(electric.reshape(len(order), 3, -1), axis=2), axis=1
    )
    vectors = [
        np.ravel(electric[index, component]) for index, component in enumerate(dominant)
    ]
    return neffs, np.column_stack(vectors)


def _mode_planes(data_array, normal_axis: str) -> np.ndarray:
    selected = data_array.isel(f=0)
    values = np.take(
        np.asarray(selected.values),
        0,
        axis=selected.dims.index(normal_axis),
    )
    return np.moveaxis(values, -1, 0)


def _mode_order(neffs, fields, axis, polarization):
    descending = np.argsort(np.real(neffs))[::-1]
    if polarization is None:
        return descending
    first, second = {
        "x": ("Ey", "Ez"),
        "y": ("Ex", "Ez"),
        "z": ("Ex", "Ey"),
    }[axis]
    axes = tuple(range(1, fields[first].ndim))
    first_power = np.sum(np.abs(fields[first]) ** 2, axis=axes)
    second_power = np.sum(np.abs(fields[second]) ** 2, axis=axes)
    fraction = first_power / np.maximum(first_power + second_power, np.finfo(float).eps)
    matching = fraction >= 0.5 if polarization == "te" else fraction < 0.5
    return np.concatenate(
        (descending[matching[descending]], descending[~matching[descending]])
    )


def _profile_crop_slices(eps_profile, *, profile_axes, center, size, resolution):
    grid_axis_to_coord_index = {0: 2, 1: 1, 2: 0}
    slices = []
    for dim, grid_axis in enumerate(profile_axes):
        coord_index = grid_axis_to_coord_index[int(grid_axis)]
        span = float(size[coord_index])
        if span <= 0.0 or not np.isfinite(span):
            slices.append(slice(None))
            continue
        midpoint = float(center[coord_index])
        start = int(np.floor((midpoint - 0.5 * span) / resolution))
        stop = int(np.ceil((midpoint + 0.5 * span) / resolution))
        start = int(np.clip(start, 0, eps_profile.shape[dim] - 1))
        stop = int(np.clip(stop, start + 1, eps_profile.shape[dim]))
        slices.append(slice(start, stop))
    return tuple(slices)


def _simulation_grid_view(simulation):
    fields = getattr(simulation, "fields", None)
    if fields is None or not hasattr(fields, "permittivity"):
        return simulation.design.rasterize(resolution=simulation.resolution)
    return SimpleNamespace(
        permittivity=fields.permittivity,
        conductivity=getattr(fields, "conductivity", None),
        permeability=getattr(fields, "permeability", None),
        resolution=float(simulation.resolution),
        design=simulation.design,
        shape=tuple(int(v) for v in fields.permittivity.shape),
        width=float(getattr(simulation.design, "width", 0.0) or 0.0),
        height=float(getattr(simulation.design, "height", 0.0) or 0.0),
        depth=float(getattr(simulation.design, "depth", 0.0) or 0.0),
    )


def _resolve_solver_direction(axis: str, direction=None) -> str:
    axis = str(axis).lower()
    if direction is None:
        return f"-{axis}"
    direction_str = str(direction).lower()
    if direction_str in {"+", "-"}:
        return f"{direction_str}{axis}"
    if direction_str not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
        raise ValueError(
            "mode solve direction must be one of '+', '-', '+x', '-x', '+y', "
            f"'-y', '+z', or '-z', got {direction!r}."
        )
    if direction_str[-1] != axis:
        raise ValueError(
            f"mode solve direction {direction_str!r} does not match {axis!r}-normal plane."
        )
    return direction_str


@dataclass(frozen=True)
class ModePlaneContext:
    eps: np.ndarray
    axis: str
    center: tuple[float, float, float]
    axis_index: int
    grid_index: int
    eps_profile_full: np.ndarray
    crop_slices: tuple[slice, ...]


def mode_plane_context(*, simulation, plane) -> ModePlaneContext:
    grid = _simulation_grid_view(simulation)
    eps = np.asarray(grid.permittivity)
    axis, center, _spans = plane_axis_and_spans(plane)
    offset = getattr(simulation, "coordinate_offset", (0.0, 0.0, 0.0))
    center = tuple(c + o for c, o in zip(center, offset, strict=True))
    axis_index = {"z": 0, "y": 1, "x": 2}[axis]
    grid_index = int(
        np.clip(
            round(center[{"z": 2, "y": 1, "x": 0}[axis]] / simulation.resolution),
            0,
            eps.shape[axis_index] - 1,
        )
    )
    eps_profile_full = np.take(eps, grid_index, axis=axis_index)
    profile_axes = tuple(index for index in range(eps.ndim) if index != axis_index)
    crop_slices = _profile_crop_slices(
        eps_profile_full,
        profile_axes=profile_axes,
        center=center,
        size=tuple(float(value) for value in plane.size),
        resolution=float(simulation.resolution),
    )
    if len(center) != 3:
        raise ValueError(f"Mode-plane centers require three coordinates: {center!r}")
    return ModePlaneContext(
        eps=eps,
        axis=axis,
        center=(center[0], center[1], center[2]),
        axis_index=axis_index,
        grid_index=grid_index,
        eps_profile_full=eps_profile_full,
        crop_slices=crop_slices,
    )


def solve_mode_plane(
    *, simulation, plane, mode_spec: ModeSpec | None, freqs, direction=None
) -> ModeData:
    """Solve modal fields on a finite simulation plane."""
    spec = mode_spec or ModeSpec()
    frequencies = np.asarray(freqs, dtype=float).reshape(-1)
    if frequencies.size == 0:
        raise ValueError("Mode solving requires at least one frequency.")

    context = mode_plane_context(simulation=simulation, plane=plane)
    eps_profile = context.eps_profile_full[context.crop_slices]
    solver_direction = _resolve_solver_direction(context.axis, direction)
    solved = [
        solve_modes(
            eps=eps_profile,
            omega=2.0 * np.pi * float(frequency),
            dL=simulation.resolution,
            m=int(spec.num_modes),
            direction=cast(SignedAxis, solver_direction),
            filter_pol=cast(Literal["te", "tm"] | None, spec.polarization),
            target_neff=spec.target_neff,
            return_fields=True,
        )
        for frequency in frequencies
    ]
    return ModeData(
        frequencies=frequencies,
        neffs=np.asarray([item[0] for item in solved]),
        e_fields=np.asarray([item[1] for item in solved]),
        h_fields=np.asarray([item[2] for item in solved]),
        eps_profiles=np.repeat(eps_profile[None, ...], len(frequencies), axis=0),
        eps_profile_fulls=np.repeat(
            context.eps_profile_full[None, ...], len(frequencies), axis=0
        ),
        resolution=float(simulation.resolution),
        solver_direction=solver_direction,
        axis=context.axis,
        center=context.center,
        plane=plane,
        crop_slices=context.crop_slices,
    )
