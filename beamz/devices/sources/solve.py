from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal, Tuple, Union, cast, overload

import numpy as np

from beamz.devices.modes import ModePlaneSpec, solve_beamz_mode, solve_grid
from beamz.devices.modes.discrete import DISCRETE_MODE_CONTRACT
from beamz.devices.modes.specs import ModeData, ModeSpec

from .specs import plane_axis_and_spans


def solve_beamz_mode_plane(**spec_kwargs):
    """Solve a BeamZ mode plane through the native discrete launch contract."""
    return solve_beamz_mode(ModePlaneSpec(**spec_kwargs))


ModeTupleType = namedtuple("Mode", ["neff", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
"""A named tuple containing the mode fields and effective index."""


def compute_mode_polarization_fraction(
    mode: ModeTupleType,
    tangential_axes: tuple[int, int],
    pol: Literal["te", "tm"],
) -> float:
    E_fields = [mode.Ex, mode.Ey, mode.Ez]
    E1 = E_fields[tangential_axes[0]]
    E2 = E_fields[tangential_axes[1]]

    if pol == "te":
        # Common convention:
        # - TE: dominant in-plane transverse E component for the selected propagation axis.
        # - TM: dominant orthogonal transverse E component.
        #
        # For +x propagation this maps to TE~Ey and TM~Ez.
        numerator = np.sum(np.abs(E1) ** 2)
    elif pol == "tm":
        numerator = np.sum(np.abs(E2) ** 2)
    else:
        raise ValueError(f"pol must be 'te' or 'tm', but got {pol}")

    denominator = np.sum(np.abs(E1) ** 2 + np.abs(E2) ** 2) + 1e-18
    return numerator / denominator


def _field_plane(data_array, normal_axis: int, mode_index: int) -> np.ndarray:
    """Extract a BeamZ transverse field plane from a native mode component."""
    normal_dim = ("x", "y", "z")[normal_axis]
    selected = data_array.isel(f=0, mode_index=mode_index)
    normal_position = selected.dims.index(normal_dim)
    return np.take(np.asarray(selected.values), indices=0, axis=normal_position)


def _remap_mode_tuple_to_global(
    mode: ModeTupleType,
    local_axis_to_global: tuple[int, int, int],
) -> ModeTupleType:
    """Map solver-local component labels to BeamZ global Cartesian labels."""

    def _remap_components(components):
        out = [None, None, None]
        for local_axis, global_axis in enumerate(local_axis_to_global):
            out[int(global_axis)] = components[int(local_axis)]
        if any(component is None for component in out):
            raise ValueError(
                f"Invalid local-to-global axis mapping: {local_axis_to_global!r}"
            )
        return out

    Ex, Ey, Ez = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple(_remap_components((mode.Ex, mode.Ey, mode.Ez))),
    )
    Hx, Hy, Hz = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray],
        tuple(_remap_components((mode.Hx, mode.Hy, mode.Hz))),
    )
    transform = np.zeros((3, 3), dtype=float)
    for local_axis, global_axis in enumerate(local_axis_to_global):
        transform[int(global_axis), int(local_axis)] = 1.0
    axial_sign = float(round(np.linalg.det(transform)))
    return ModeTupleType(
        neff=mode.neff,
        Ex=Ex,
        Ey=Ey,
        Ez=Ez,
        Hx=axial_sign * Hx,
        Hy=axial_sign * Hy,
        Hz=axial_sign * Hz,
    )


def sort_modes(
    modes: list[ModeTupleType],
    filter_pol: Union[Literal["te", "tm"], None],
    tangential_axes: tuple[int, int],
) -> list[ModeTupleType]:
    if filter_pol is None:
        return sorted(modes, key=lambda m: float(np.real(m.neff)), reverse=True)

    def is_matching(mode: ModeTupleType) -> bool:
        frac = compute_mode_polarization_fraction(mode, tangential_axes, filter_pol)
        return frac >= 0.5

    matching = [m for m in modes if is_matching(m)]
    non_matching = [m for m in modes if not is_matching(m)]

    matching_sorted = sorted(
        matching, key=lambda m: float(np.real(m.neff)), reverse=True
    )
    non_matching_sorted = sorted(
        non_matching, key=lambda m: float(np.real(m.neff)), reverse=True
    )

    return matching_sorted + non_matching_sorted


def compute_mode(
    frequency: float,
    inv_permittivities: np.ndarray,
    inv_permeabilities: Union[np.ndarray, float],
    resolution: float,
    direction: Literal["+", "-"],
    mode_index: int = 0,
    filter_pol: Union[Literal["te", "tm"], None] = None,
    target_neff: Union[float, None] = None,
    local_axis_to_global: tuple[int, int, int] = (0, 1, 2),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    inv_permittivities = np.asarray(inv_permittivities, dtype=np.complex128)
    if inv_permittivities.ndim == 1:
        inv_permittivities = inv_permittivities[np.newaxis, :, np.newaxis]
    elif inv_permittivities.ndim == 2:
        inv_permittivities = inv_permittivities[np.newaxis, :, :]
    elif inv_permittivities.ndim > 3:
        raise ValueError(
            f"Invalid shape of inv_permittivities: {inv_permittivities.shape}"
        )

    if isinstance(inv_permeabilities, np.ndarray):
        inv_permeabilities = np.asarray(inv_permeabilities, dtype=np.complex128)
        if inv_permeabilities.ndim == 1:
            inv_permeabilities = inv_permeabilities[np.newaxis, :, np.newaxis]
        elif inv_permeabilities.ndim == 2:
            inv_permeabilities = inv_permeabilities[np.newaxis, :, :]
        elif inv_permeabilities.ndim > 3:
            raise ValueError(
                f"Invalid shape of inv_permeabilities: {inv_permeabilities.shape}"
            )
    else:
        inv_permeabilities = np.asarray(inv_permeabilities, dtype=np.complex128)

    singleton_axes = [
        idx for idx, size in enumerate(inv_permittivities.shape) if size == 1
    ]
    if not singleton_axes:
        raise ValueError(
            "At least one singleton dimension is required to denote the propagation axis"
        )
    propagation_axis = singleton_axes[0]

    cross_axes = [ax for ax in range(inv_permittivities.ndim) if ax != propagation_axis]
    if not cross_axes:
        raise ValueError("Need at least one transverse axis for mode computation")
    is_1d_profile = len(singleton_axes) >= 2

    permittivities = 1 / inv_permittivities
    coords = [
        np.arange(permittivities.shape[dim] + 1) * resolution / 1e-6
        for dim in cross_axes
    ]
    permittivity_squeezed = np.take(permittivities, indices=0, axis=propagation_axis)
    if permittivity_squeezed.ndim == 1:
        permittivity_squeezed = permittivity_squeezed[:, np.newaxis]

    if inv_permeabilities.ndim == inv_permittivities.ndim:
        permeability = 1 / inv_permeabilities
        permeability_squeezed = np.take(permeability, indices=0, axis=propagation_axis)
        if permeability_squeezed.ndim == 1:
            permeability_squeezed = permeability_squeezed[:, np.newaxis]
    else:
        permeability_squeezed = 1 / inv_permeabilities.item()

    if np.ndim(permeability_squeezed) == 0:
        mu_xx = np.full_like(permittivity_squeezed, permeability_squeezed)
    else:
        mu_xx = np.asarray(permeability_squeezed, dtype=np.complex128)

    result = solve_grid(
        eps_xx=permittivity_squeezed,
        mu_xx=mu_xx,
        x_edges=coords[0],
        y_edges=coords[1],
        freqs=[frequency],
        direction=direction,
        num_modes=2 * (mode_index + 1) + 5,
        target_neff=target_neff,
        normal_axis=cast(Literal[0, 1, 2], propagation_axis),
    )
    modes = []
    for idx in range(result.n_complex.shape[1]):
        local_mode = ModeTupleType(
            neff=result.n_complex.values[0, idx],
            Ex=_field_plane(result.field_components["Ex"], propagation_axis, idx),
            Ey=_field_plane(result.field_components["Ey"], propagation_axis, idx),
            Ez=_field_plane(result.field_components["Ez"], propagation_axis, idx),
            Hx=_field_plane(result.field_components["Hx"], propagation_axis, idx),
            Hy=_field_plane(result.field_components["Hy"], propagation_axis, idx),
            Hz=_field_plane(result.field_components["Hz"], propagation_axis, idx),
        )
        modes.append(_remap_mode_tuple_to_global(local_mode, local_axis_to_global))
    tangential_values = tuple(ax for ax in range(3) if ax != propagation_axis)
    tangential_axes = (tangential_values[0], tangential_values[1])
    modes = sort_modes(modes, filter_pol, tangential_axes)
    if mode_index >= len(modes):
        raise ValueError(
            f"Requested mode index {mode_index}, but only {len(modes)} modes available"
        )

    mode = modes[mode_index]

    E = np.stack([mode.Ex, mode.Ey, mode.Ez], axis=0).astype(np.complex128)
    H = np.stack([mode.Hx, mode.Hy, mode.Hz], axis=0).astype(np.complex128)
    if propagation_axis == 1:
        H = -H
    if is_1d_profile:
        E = E[..., 0]
        H = H[..., 0]

    E_norm, H_norm = _normalize_by_poynting_flux(E, H, axis=propagation_axis)
    return E_norm, H_norm, np.asarray(mode.neff, dtype=np.complex128), propagation_axis


@overload
def solve_modes(
    eps: np.ndarray,
    omega: float,
    dL: float,
    npml: int = 0,
    m: int = 1,
    direction: Literal["+x", "-x", "+y", "-y", "+z", "-z"] = "+x",
    filter_pol: Literal["te", "tm"] | None = None,
    return_fields: Literal[True] = True,
    propagation_axis: Literal["+x", "-x", "+y", "-y", "+z", "-z"] | None = None,
    target_neff: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]: ...


@overload
def solve_modes(
    eps: np.ndarray,
    omega: float,
    dL: float,
    npml: int = 0,
    m: int = 1,
    direction: Literal["+x", "-x", "+y", "-y", "+z", "-z"] = "+x",
    filter_pol: Literal["te", "tm"] | None = None,
    return_fields: Literal[False] = False,
    propagation_axis: Literal["+x", "-x", "+y", "-y", "+z", "-z"] | None = None,
    target_neff: float | None = None,
) -> tuple[np.ndarray, np.ndarray]: ...


def solve_modes(
    eps: np.ndarray,
    omega: float,
    dL: float,
    npml: int = 0,
    m: int = 1,
    direction: Literal["+x", "-x", "+y", "-y", "+z", "-z"] = "+x",
    filter_pol: Union[Literal["te", "tm"], None] = None,
    return_fields: bool = False,
    propagation_axis: Union[Literal["+x", "-x", "+y", "-y", "+z", "-z"], None] = None,
    target_neff: Union[float, None] = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, int]
]:
    if eps.ndim not in [1, 2]:
        raise ValueError("solve_modes expects a 1D or 2D permittivity array")
    if npml < 0:
        raise ValueError("npml must be non-negative")

    freq = omega / (2 * np.pi)
    axis_hint = propagation_axis if propagation_axis is not None else direction
    axis_char = (
        str(axis_hint)[1] if str(axis_hint).startswith(("+", "-")) else str(axis_hint)
    )
    axis_index = {"x": 0, "y": 1, "z": 2}.get(axis_char)
    if axis_index is None:
        raise ValueError(
            f"Unsupported propagation axis '{axis_hint}'. Use one of ±x/±y/±z."
        )

    # Reshape eps to 3D for compute_mode (axis, trans1, trans2)
    # compute_mode expects (prop_axis, trans1, trans2) where prop_axis is singleton
    local_axis_to_global = (0, 1, 2)
    if eps.ndim == 1:
        line = np.asarray(eps, dtype=np.complex128)
        if axis_index == 0:
            inv_eps = (1.0 / line).reshape(1, line.size, 1)
            local_axis_to_global = (0, 1, 2)
        elif axis_index == 1:
            inv_eps = (1.0 / line).reshape(line.size, 1, 1)
            local_axis_to_global = (0, 1, 2)
        else:
            inv_eps = (1.0 / line).reshape(line.size, 1, 1)
            local_axis_to_global = (0, 1, 2)
    else:
        eps_arr = 1.0 / np.asarray(eps, dtype=np.complex128)
        if axis_index == 0:
            inv_eps = eps_arr[np.newaxis, :, :]
            local_axis_to_global = (0, 2, 1)
        elif axis_index == 1:
            inv_eps = eps_arr[:, np.newaxis, :]
            local_axis_to_global = (2, 1, 0)
        else:
            inv_eps = eps_arr[:, :, np.newaxis]
            local_axis_to_global = (1, 0, 2)

    direction_flag = "+" if direction.startswith("+") else "-"

    neffs: list[complex] = []
    e_fields: list[np.ndarray] = []
    h_fields: list[np.ndarray] = []
    mode_vectors: list[np.ndarray] = []

    for mode_index in range(m):
        E_full, H_full, neff, prop_axis = compute_mode(
            frequency=freq,
            inv_permittivities=inv_eps,
            inv_permeabilities=1.0,
            resolution=dL,
            direction=direction_flag,
            mode_index=mode_index,
            filter_pol=filter_pol,
            target_neff=target_neff,
            local_axis_to_global=local_axis_to_global,
        )

        neffs.append(complex(np.asarray(neff).item()))
        if return_fields:
            e_fields.append(E_full)
            h_fields.append(H_full)
        else:
            component_norms = [np.linalg.norm(np.squeeze(E_full[i])) for i in range(3)]
            component_idx = int(np.argmax(component_norms))
            field_line = np.squeeze(E_full[component_idx])
            if field_line.ndim > 1:
                field_line = field_line[:, 0]
            max_amp = np.max(np.abs(field_line)) or 1.0
            mode_vectors.append(field_line / max_amp)

    neff_array = np.asarray(neffs, dtype=np.complex128)

    if return_fields:
        return (
            neff_array,
            np.stack(e_fields) if e_fields else np.empty((0, 3, 0, 0)),
            np.stack(h_fields) if h_fields else np.empty((0, 3, 0, 0)),
            prop_axis,
        )

    if not mode_vectors:
        return neff_array, np.zeros((eps.size, 0), dtype=np.complex128)

    return neff_array, np.column_stack(mode_vectors)


def _normalize_by_poynting_flux(
    E: np.ndarray, H: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    S = np.cross(E, np.conjugate(H), axis=0)
    power = float(np.real(np.sum(S[axis])))

    # Guard against tiny/negative/NaN power from numerical noise
    if not np.isfinite(power) or abs(power) < 1e-18:
        # Fallback: normalize by field amplitude
        e_norm = float(np.linalg.norm(E))
        if e_norm > 1e-18 and np.isfinite(e_norm):
            return E / e_norm, H / e_norm
        return E, H
    # Normalize by magnitude of power to avoid sqrt of negative
    scale = np.sqrt(abs(power))
    if scale == 0.0 or not np.isfinite(scale):
        # Fallback: normalize by field amplitude
        e_norm = float(np.linalg.norm(E))
        if e_norm > 1e-18 and np.isfinite(e_norm):
            return E / e_norm, H / e_norm
        return E, H
    E_norm = E / scale
    H_norm = H / scale
    # Final NaN check
    if not np.all(np.isfinite(E_norm)) or not np.all(np.isfinite(H_norm)):
        return E, H
    return E_norm, H_norm


def solve_discrete_mode_plane(**spec_kwargs: Any):
    """Return a native BeamZ DiscreteMode or fail with a clear contract error."""

    discrete_mode = solve_beamz_mode_plane(**spec_kwargs)
    if discrete_mode is None:
        raise RuntimeError(
            "The native mode solver returned None for the required "
            f"{DISCRETE_MODE_CONTRACT} contract."
        )
    missing = [
        name
        for name in (
            "neff",
            "profiles",
            "backward_profiles",
            "component_indices",
            "phase_reference_coord",
            "phase_plane_coord",
            "k_num_axis",
            "power_scale",
        )
        if not hasattr(discrete_mode, name)
    ]
    if missing:
        raise RuntimeError(
            "The native mode solver returned an incompatible DiscreteMode object; "
            f"missing {', '.join(missing)}."
        )
    return discrete_mode


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
    plane_size = tuple(float(value) for value in plane.size)
    profile_axes = tuple(idx for idx in range(eps.ndim) if idx != axis_index)
    crop_slices = _profile_crop_slices(
        eps_profile_full,
        profile_axes=profile_axes,
        center=center,
        size=plane_size,
        resolution=float(simulation.resolution),
    )
    if len(center) != 3:
        raise ValueError(f"Mode-plane centers require three coordinates: {center!r}")
    center_3d = (center[0], center[1], center[2])
    return ModePlaneContext(
        eps=eps,
        axis=axis,
        center=center_3d,
        axis_index=axis_index,
        grid_index=grid_index,
        eps_profile_full=eps_profile_full,
        crop_slices=crop_slices,
    )


def solve_mode_plane(
    *, simulation, plane, mode_spec: ModeSpec | None, freqs, direction=None
) -> ModeData:
    """Solve modal fields on a finite simulation plane."""

    spec = mode_spec if mode_spec is not None else ModeSpec()
    freq_arr = np.asarray(freqs, dtype=float).reshape(-1)
    if freq_arr.size == 0:
        raise ValueError("Mode solving requires at least one frequency.")

    ctx = mode_plane_context(simulation=simulation, plane=plane)
    eps_profile = ctx.eps_profile_full[ctx.crop_slices]
    solver_direction = _resolve_solver_direction(ctx.axis, direction)
    neffs_by_freq = []
    e_by_freq = []
    h_by_freq = []
    eps_by_freq = []
    eps_full_by_freq = []
    for freq in freq_arr:
        neffs, e_fields, h_fields, _ = solve_modes(
            eps=eps_profile,
            omega=2.0 * np.pi * float(freq),
            dL=simulation.resolution,
            m=int(spec.num_modes),
            direction=cast(
                Literal["+x", "-x", "+y", "-y", "+z", "-z"], solver_direction
            ),
            filter_pol=cast(Literal["te", "tm"] | None, spec.polarization),
            target_neff=spec.target_neff,
            return_fields=True,
        )
        neffs_by_freq.append(np.asarray(neffs))
        e_by_freq.append(np.asarray(e_fields))
        h_by_freq.append(np.asarray(h_fields))
        eps_by_freq.append(np.asarray(eps_profile))
        eps_full_by_freq.append(np.asarray(ctx.eps_profile_full))

    return ModeData(
        frequencies=freq_arr,
        neffs=np.asarray(neffs_by_freq),
        e_fields=np.asarray(e_by_freq),
        h_fields=np.asarray(h_by_freq),
        eps_profiles=np.asarray(eps_by_freq),
        eps_profile_fulls=np.asarray(eps_full_by_freq),
        resolution=float(simulation.resolution),
        solver_direction=solver_direction,
        axis=ctx.axis,
        center=ctx.center,
        plane=plane,
        crop_slices=ctx.crop_slices,
    )
