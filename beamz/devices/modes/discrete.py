"""BEAMZ-facing discrete mode contract.

This module is intentionally small and data-oriented. BeamZ owns geometry and
Yee-grid placement; the native mode package solves and converts fields into
component planes that the simulation can inject without another interpretation
layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, cast

import numpy as np

from ._yee import refine_x_mode_at_fixed_beta, validate_x_mode_refinement
from .solver import solve_grid

AxisName = Literal["x", "y", "z"]
DirectionName = Literal["+x", "-x", "+y", "-y", "+z", "-z"]
PolarizationName = Literal["te", "tm"]
ComponentIndex = tuple[slice | int, slice | int, slice | int]

_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
# Legacy schema identifier retained for saved diagnostics and external consumers.
# It names the v1 data shape and no longer implies an external package dependency.
DISCRETE_MODE_CONTRACT = "micromode.beamz.DiscreteMode/v1"
_AXIS_INDEX: dict[AxisName, Literal[0, 1, 2]] = {"x": 0, "y": 1, "z": 2}
_AXIS_NAMES: tuple[AxisName, AxisName, AxisName] = ("x", "y", "z")
_YEE_OFFSETS_3D = {
    "Ex": {"z": 0.0, "y": 0.0, "x": 0.5},
    "Ey": {"z": 0.0, "y": 0.5, "x": 0.0},
    "Ez": {"z": 0.5, "y": 0.0, "x": 0.0},
    "Hx": {"z": 0.5, "y": 0.5, "x": 0.0},
    "Hy": {"z": 0.5, "y": 0.0, "x": 0.5},
    "Hz": {"z": 0.0, "y": 0.5, "x": 0.5},
}


class _ModeCandidate(TypedDict):
    neff: complex
    fields: dict[str, np.ndarray]


@dataclass(frozen=True)
class ModePlaneSpec:
    """Exact mode-plane metadata passed to the BeamZ-native solver.

    ``scalar_permittivity`` uses ``transverse_axes`` order, not the solver's
    internal local-axis order. For example, an x-normal BeamZ plane is usually
    stored as ``("z", "y")``.
    """

    scalar_permittivity: np.ndarray
    frequency: float
    resolution: float
    dt: float | None
    axis: AxisName
    direction: DirectionName
    transverse_axes: tuple[AxisName, AxisName]
    grid_shape: tuple[int, int, int]
    component_shapes: dict[str, tuple[int, int, int]]
    center: tuple[float, float, float]
    width: float
    height: float
    plane_index: int
    offset_index: int
    mode_index: int = 0
    polarization: PolarizationName | None = None
    target_neff: float | None = None
    num_modes: int | None = None
    solver_direction: DirectionName | None = None
    aperture_pad_cells: int = 2
    aperture_window_alpha: float = 0.2
    phase_reference: str = "dominant_h_real_positive"
    time_convention: str = "exp(-i omega t); E at integer steps, H at half steps"
    component_offsets: dict[str, dict[str, float]] = field(
        default_factory=lambda: dict(_YEE_OFFSETS_3D)
    )
    component_permittivity: dict[str, np.ndarray] = field(default_factory=dict)
    component_permeability: dict[str, np.ndarray] = field(default_factory=dict)
    # Guarded by joint field, power, energy, and discrete-Maxwell validation.
    yee_refinement: bool = True
    boundary: str = "beamz-finite-aperture"

    def __post_init__(self) -> None:
        eps = np.asarray(self.scalar_permittivity, dtype=np.complex128)
        if eps.ndim != 2:
            raise ValueError("scalar_permittivity must be a 2D transverse plane")
        object.__setattr__(self, "scalar_permittivity", eps)

        axis = str(self.axis).lower()
        if axis not in _AXIS_INDEX:
            raise ValueError("axis must be one of 'x', 'y', or 'z'")
        object.__setattr__(self, "axis", axis)

        direction = str(self.direction).lower()
        if direction not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
            raise ValueError(
                "direction must be one of '+x', '-x', '+y', '-y', '+z', '-z'"
            )
        if direction[1] != axis:
            raise ValueError("direction axis must match axis")
        object.__setattr__(self, "direction", direction)

        solver_direction = (
            self.direction
            if self.solver_direction is None
            else str(self.solver_direction).lower()
        )
        if solver_direction not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
            raise ValueError(
                "solver_direction must be one of '+x', '-x', '+y', '-y', '+z', '-z'"
            )
        if solver_direction[1] != axis:
            raise ValueError("solver_direction axis must match axis")
        object.__setattr__(self, "solver_direction", solver_direction)

        transverse_axes = tuple(str(value).lower() for value in self.transverse_axes)
        expected_axes = tuple(value for value in _AXIS_NAMES if value != axis)
        if set(transverse_axes) != set(expected_axes) or len(transverse_axes) != 2:
            raise ValueError(
                f"transverse_axes must be a permutation of {expected_axes!r}"
            )
        object.__setattr__(self, "transverse_axes", transverse_axes)

        grid_shape = tuple(int(v) for v in self.grid_shape)
        if len(grid_shape) != 3 or any(v <= 1 for v in grid_shape):
            raise ValueError("grid_shape must contain three dimensions larger than one")
        object.__setattr__(self, "grid_shape", grid_shape)

        component_shapes = {
            name: tuple(int(v) for v in shape)
            for name, shape in self.component_shapes.items()
        }
        missing_shapes = set(_COMPONENTS).difference(component_shapes)
        if missing_shapes:
            raise ValueError(
                f"component_shapes missing: {', '.join(sorted(missing_shapes))}"
            )
        object.__setattr__(self, "component_shapes", component_shapes)

        if float(self.frequency) <= 0.0 or not np.isfinite(float(self.frequency)):
            raise ValueError("frequency must be finite and positive")
        if float(self.resolution) <= 0.0 or not np.isfinite(float(self.resolution)):
            raise ValueError("resolution must be finite and positive")
        if self.dt is not None and (
            float(self.dt) <= 0.0 or not np.isfinite(float(self.dt))
        ):
            raise ValueError("dt must be finite and positive when provided")
        object.__setattr__(self, "frequency", float(self.frequency))
        object.__setattr__(self, "resolution", float(self.resolution))
        object.__setattr__(self, "dt", None if self.dt is None else float(self.dt))
        object.__setattr__(self, "center", tuple(float(v) for v in self.center))
        object.__setattr__(self, "width", float(self.width))
        object.__setattr__(self, "height", float(self.height))
        object.__setattr__(self, "plane_index", int(self.plane_index))
        object.__setattr__(self, "offset_index", int(self.offset_index))
        object.__setattr__(self, "mode_index", int(self.mode_index))
        object.__setattr__(
            self, "num_modes", None if self.num_modes is None else int(self.num_modes)
        )
        object.__setattr__(self, "aperture_pad_cells", int(self.aperture_pad_cells))
        object.__setattr__(
            self, "aperture_window_alpha", float(self.aperture_window_alpha)
        )
        object.__setattr__(self, "yee_refinement", bool(self.yee_refinement))

        if self.polarization is not None:
            pol = str(self.polarization).lower()
            if pol not in {"te", "tm"}:
                raise ValueError("polarization must be 'te', 'tm', or None")
            object.__setattr__(self, "polarization", pol)


@dataclass(frozen=True)
class DiscreteMode:
    """Mode solved and shaped directly for BEAMZ component lattices."""

    neff: complex
    profiles: dict[str, np.ndarray]
    backward_profiles: dict[str, np.ndarray]
    component_indices: dict[str, ComponentIndex]
    axis: AxisName
    direction: DirectionName
    transverse_axes: tuple[AxisName, AxisName]
    phase_reference_component: str
    phase_reference_coord: float
    phase_plane_coord: float
    k_num_axis: float
    power_scale: float
    diagnostics: dict[str, object]

    def component(self, name: str) -> np.ndarray:
        """Return one component profile."""
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise ValueError(f"Unknown component {name!r}") from exc


def solve_beamz_mode(spec: ModePlaneSpec) -> DiscreteMode:
    """Solve and return a BEAMZ-shaped discrete mode."""

    if not isinstance(spec, ModePlaneSpec):
        raise TypeError("spec must be a ModePlaneSpec")

    solver_axes = _solver_axes_for_axis(spec.axis)
    eps_solver = _transpose_between_axes(
        spec.scalar_permittivity, spec.transverse_axes, solver_axes
    )
    dx_um = spec.resolution / 1e-6
    x_edges = tuple(float(v) for v in np.arange(eps_solver.shape[0] + 1) * dx_um)
    y_edges = tuple(float(v) for v in np.arange(eps_solver.shape[1] + 1) * dx_um)
    mode_count = (
        spec.num_modes if spec.num_modes is not None else 2 * (spec.mode_index + 1) + 5
    )

    result = solve_grid(
        eps_xx=eps_solver,
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[spec.frequency],
        direction="+" if str(spec.solver_direction).startswith("+") else "-",
        num_modes=mode_count,
        target_neff=spec.target_neff,
        normal_axis=_AXIS_INDEX[spec.axis],
    )
    candidates = _candidate_modes(result, spec)
    candidates = _sort_modes(candidates, spec)
    if spec.mode_index >= len(candidates):
        raise ValueError(
            f"Requested mode_index={spec.mode_index}, but only {len(candidates)} modes are available"
        )

    selected = candidates[spec.mode_index]
    phase_component = _select_phase_reference_component(
        spec.axis, spec.polarization, selected["fields"]
    )
    phase_ref = _dominant_phase(selected["fields"][phase_component])
    aligned = {
        name: value * np.exp(-1j * phase_ref)
        for name, value in selected["fields"].items()
    }

    profiles, indices, extra = _build_profiles(aligned, spec)
    symmetric_axes = _detect_transverse_symmetry_axes(spec.scalar_permittivity)
    if symmetric_axes:
        profiles = _enforce_componentwise_parity(profiles, symmetric_axes)

    phase_ref_coord = _component_axis_coord(
        phase_component,
        _axis_index_from_component_indices(indices.get(phase_component), spec.axis),
        spec,
    )
    omega = 2.0 * np.pi * spec.frequency
    k_num = _solve_numeric_k_axis(omega, spec.dt, spec.resolution, selected["neff"])
    boundary_neff = _boundary_refractive_index(spec.scalar_permittivity)
    yee_refinement_eligible = (
        spec.axis == "x"
        and bool(spec.component_permittivity)
        and float(np.real(selected["neff"])) > boundary_neff
    )
    yee_refinement_requested = bool(spec.yee_refinement)
    yee_refinement_attempted = yee_refinement_requested and yee_refinement_eligible
    yee_refinement_accepted = False
    yee_refinement_rejection_reason = ""
    yee_validation: dict[str, Any] = {}
    yee_residual = np.nan
    yee_frequency_ratio = np.nan
    yee_initial_frequency_ratio = np.nan
    if yee_refinement_requested and not yee_refinement_eligible:
        yee_refinement_rejection_reason = (
            "mode is not eligible for x-normal guided-mode refinement"
        )
    if yee_refinement_attempted:
        seed_profiles = {
            name: np.asarray(value, dtype=np.complex128).copy()
            for name, value in profiles.items()
        }
        seed_k_num = float(k_num)
        try:
            (
                candidate,
                yee_residual,
                yee_frequency_ratio,
                candidate_k_num,
                yee_initial_frequency_ratio,
            ) = refine_x_mode_at_fixed_beta(
                seed_profiles,
                indices,
                component_permittivity=spec.component_permittivity,
                component_permeability=spec.component_permeability,
                omega=omega,
                dt=spec.dt,
                resolution=spec.resolution,
                k_num=seed_k_num,
                direction_sign=_direction_sign(spec.direction),
            )
            yee_refinement_accepted, yee_validation = validate_x_mode_refinement(
                seed_profiles,
                candidate,
                indices,
                component_permittivity=spec.component_permittivity,
                component_permeability=spec.component_permeability,
                omega=omega,
                dt=spec.dt,
                resolution=spec.resolution,
                k_num=candidate_k_num,
                direction_sign=_direction_sign(spec.direction),
            )
            yee_refinement_rejection_reason = str(
                yee_validation.get("rejection_reason", "")
            )
            if yee_refinement_accepted:
                profiles = candidate
                k_num = float(candidate_k_num)
            else:
                profiles = seed_profiles
                k_num = seed_k_num
        except Exception as exc:
            profiles = seed_profiles
            k_num = seed_k_num
            yee_refinement_rejection_reason = f"{type(exc).__name__}: {exc}"
    profiles, power_scale, signed_power = _normalize_profiles_by_phase_referenced_flux(
        profiles,
        indices,
        axis=spec.axis,
        d_area=spec.resolution * spec.resolution,
        direction_sign=_direction_sign(spec.direction),
        omega=omega,
        k_num=k_num,
        ref_coord=phase_ref_coord,
        resolution=spec.resolution,
    )
    profiles = _runtime_oriented_profiles(
        profiles, spec.axis, _direction_sign(spec.direction)
    )
    if signed_power < 0.0:
        profiles = _backward_mode_from_forward(profiles)
        signed_power = -signed_power
    backward_profiles = _backward_mode_from_forward(profiles)

    diagnostics = {
        "contract": DISCRETE_MODE_CONTRACT,
        "normal_axis": spec.axis,
        "transverse_axes": spec.transverse_axes,
        "solver_axes": solver_axes,
        "solver_direction": spec.solver_direction,
        "mode_index": spec.mode_index,
        "selected_neff": complex(selected["neff"]),
        "phase_reference": spec.phase_reference,
        "time_convention": spec.time_convention,
        "aperture_window_alpha": spec.aperture_window_alpha,
        "yee_refinement": yee_refinement_accepted,
        "yee_refinement_requested": yee_refinement_requested,
        "yee_refinement_eligible": yee_refinement_eligible,
        "yee_refinement_attempted": yee_refinement_attempted,
        "yee_refinement_accepted": yee_refinement_accepted,
        "yee_refinement_rejection_reason": yee_refinement_rejection_reason,
        "yee_refinement_validation": yee_validation,
        "boundary_neff": float(boundary_neff),
        "yee_residual": float(yee_residual),
        "yee_frequency_ratio": float(yee_frequency_ratio),
        "yee_initial_frequency_ratio": float(yee_initial_frequency_ratio),
        "power_before_phase_reference": float(extra.get("initial_power", np.nan)),
        "power_after_phase_reference": float(signed_power),
        "solver_info": result.solver_info or {},
    }
    return DiscreteMode(
        neff=complex(selected["neff"]),
        profiles=profiles,
        backward_profiles=backward_profiles,
        component_indices=indices,
        axis=spec.axis,
        direction=spec.direction,
        transverse_axes=spec.transverse_axes,
        phase_reference_component=phase_component,
        phase_reference_coord=float(phase_ref_coord),
        phase_plane_coord=float((spec.plane_index + 0.5) * spec.resolution),
        k_num_axis=float(k_num),
        power_scale=float(power_scale),
        diagnostics=diagnostics,
    )


def _boundary_refractive_index(permittivity: np.ndarray) -> float:
    """Return the largest refractive index touching the mode-plane boundary."""
    eps = np.asarray(permittivity, dtype=np.complex128)
    if eps.ndim != 2 or min(eps.shape) == 0:
        return 0.0
    boundary = np.concatenate((eps[0], eps[-1], eps[1:-1, 0], eps[1:-1, -1]))
    return float(np.sqrt(max(float(np.max(np.real(boundary))), 0.0)))


def _candidate_modes(result, spec: ModePlaneSpec) -> list[_ModeCandidate]:
    candidates = []
    count = int(result.n_complex.shape[1])
    for mode_index in range(count):
        fields = {
            component: _field_plane(
                result.field_components[component],
                spec.axis,
                spec.transverse_axes,
                mode_index,
            )
            for component in _COMPONENTS
        }
        candidates.append(
            {"neff": complex(result.n_complex.values[0, mode_index]), "fields": fields}
        )
    return candidates


def _field_plane(
    data_array,
    axis: AxisName,
    transverse_axes: tuple[AxisName, AxisName],
    mode_index: int,
) -> np.ndarray:
    selected = data_array.isel(f=0, mode_index=mode_index)
    normal_dim = axis
    if normal_dim in selected.dims:
        selected = selected.isel({normal_dim: 0})
    selected = selected.transpose(*transverse_axes)
    return np.asarray(selected.values, dtype=np.complex128)


def _sort_modes(
    candidates: list[_ModeCandidate], spec: ModePlaneSpec
) -> list[_ModeCandidate]:
    if spec.polarization is None:
        return sorted(
            candidates, key=lambda item: float(np.real(item["neff"])), reverse=True
        )

    def matches(item: _ModeCandidate) -> bool:
        return (
            _polarization_fraction(item["fields"], spec.axis, spec.polarization) >= 0.5
        )

    matching = [item for item in candidates if matches(item)]
    rest = [item for item in candidates if not matches(item)]
    return sorted(
        matching, key=lambda item: float(np.real(item["neff"])), reverse=True
    ) + sorted(rest, key=lambda item: float(np.real(item["neff"])), reverse=True)


def _polarization_fraction(
    fields: dict[str, np.ndarray], axis: AxisName, polarization: PolarizationName | None
) -> float:
    if polarization is None:
        return 1.0
    tangential_axes = tuple(idx for idx in range(3) if idx != _AXIS_INDEX[axis])
    first = fields[f"E{_AXIS_NAMES[tangential_axes[0]]}"]
    second = fields[f"E{_AXIS_NAMES[tangential_axes[1]]}"]
    numerator = np.sum(
        np.abs(first) ** 2 if polarization == "te" else np.abs(second) ** 2
    )
    denominator = np.sum(np.abs(first) ** 2 + np.abs(second) ** 2) + 1e-18
    return float(np.real(numerator / denominator))


def _select_phase_reference_component(
    axis: AxisName,
    polarization: PolarizationName | None,
    fields: dict[str, np.ndarray],
) -> str:
    preferred = {
        ("x", "tm"): "Hy",
        ("x", "te"): "Hz",
        ("y", "tm"): "Hx",
        ("y", "te"): "Hz",
        ("z", "tm"): "Hx",
        ("z", "te"): "Hy",
    }
    if polarization is not None and (axis, polarization) in preferred:
        candidate = preferred[(axis, polarization)]
        if np.max(np.abs(fields[candidate])) >= 1e-9:
            return candidate
    tangential_h = {"x": ("Hy", "Hz"), "y": ("Hx", "Hz"), "z": ("Hx", "Hy")}[axis]
    strengths = [float(np.max(np.abs(fields[name]))) for name in tangential_h]
    return tangential_h[int(np.argmax(strengths))]


def _dominant_phase(field: np.ndarray) -> float:
    flat = np.asarray(field, dtype=np.complex128).reshape(-1)
    if flat.size == 0:
        return 0.0
    return float(np.angle(flat[int(np.argmax(np.abs(flat)))]))


def _build_profiles(
    fields: dict[str, np.ndarray],
    spec: ModePlaneSpec,
) -> tuple[dict[str, np.ndarray], dict[str, ComponentIndex], dict[str, float]]:
    axis = spec.axis
    if axis == "x":
        return _build_x_profiles(fields, spec)
    if axis == "y":
        return _build_y_profiles(fields, spec)
    return _build_z_profiles(fields, spec)


def _build_x_profiles(
    fields: dict[str, np.ndarray],
    spec: ModePlaneSpec,
) -> tuple[dict[str, np.ndarray], dict[str, ComponentIndex], dict[str, float]]:
    ex_s = fields["Ex"]
    ey_s = _stagger_half(fields["Ey"], axis=1)
    ez_s = _stagger_half(fields["Ez"], axis=0)
    hx_s = _stagger_both(fields["Hx"])
    hy_s = _stagger_half(fields["Hy"], axis=0)
    hz_s = _stagger_half(fields["Hz"], axis=1)
    nz, ny, _nx = spec.grid_shape
    y_start, y_end = _padded_bounds(
        spec.center[1], spec.width, spec.resolution, ny, spec.aperture_pad_cells
    )
    z_start, z_end = _padded_bounds(
        spec.center[2], spec.height, spec.resolution, nz, spec.aperture_pad_cells
    )
    staggered = {"Ex": ex_s, "Ey": ey_s, "Ez": ez_s, "Hx": hx_s, "Hy": hy_s, "Hz": hz_s}
    indices: dict[str, ComponentIndex] = {
        "Ex": (
            *_support_slices("Ex", "x", z_start, z_end, y_start, y_end, ex_s.shape),
            spec.offset_index,
        ),
        "Ey": (
            *_support_slices("Ey", "x", z_start, z_end, y_start, y_end, ey_s.shape),
            spec.plane_index,
        ),
        "Ez": (
            *_support_slices("Ez", "x", z_start, z_end, y_start, y_end, ez_s.shape),
            spec.plane_index,
        ),
        "Hx": (
            *_support_slices("Hx", "x", z_start, z_end, y_start, y_end, hx_s.shape),
            spec.plane_index,
        ),
        "Hy": (
            *_support_slices("Hy", "x", z_start, z_end, y_start, y_end, hy_s.shape),
            spec.offset_index,
        ),
        "Hz": (
            *_support_slices("Hz", "x", z_start, z_end, y_start, y_end, hz_s.shape),
            spec.offset_index,
        ),
    }
    profiles = _crop_window_all(
        staggered, z_start, z_end, y_start, y_end, _direction_sign(spec.direction), spec
    )
    initial_power = _normalize_profiles_by_flux(
        profiles,
        axis="x",
        d_area=spec.resolution**2,
        direction_sign=_direction_sign(spec.direction),
    )
    return profiles, indices, {"initial_power": initial_power}


def _build_y_profiles(
    fields: dict[str, np.ndarray],
    spec: ModePlaneSpec,
) -> tuple[dict[str, np.ndarray], dict[str, ComponentIndex], dict[str, float]]:
    ex_s = _stagger_half(fields["Ex"], axis=1)
    ey_s = fields["Ey"]
    ez_s = _stagger_half(fields["Ez"], axis=0)
    hx_s = _stagger_half(fields["Hx"], axis=0)
    hy_s = _stagger_both(fields["Hy"])
    hz_s = _stagger_half(fields["Hz"], axis=1)
    nz, _ny, nx = spec.grid_shape
    x_start, x_end = _padded_bounds(
        spec.center[0], spec.width, spec.resolution, nx, spec.aperture_pad_cells
    )
    z_start, z_end = _padded_bounds(
        spec.center[2], spec.height, spec.resolution, nz, spec.aperture_pad_cells
    )
    staggered = {"Ex": ex_s, "Ey": ey_s, "Ez": ez_s, "Hx": hx_s, "Hy": hy_s, "Hz": hz_s}
    indices: dict[str, ComponentIndex] = {
        "Ex": (
            *_support_slices("Ex", "y", z_start, z_end, x_start, x_end, ex_s.shape),
            spec.plane_index,
        ),
        "Ey": (
            *_support_slices("Ey", "y", z_start, z_end, x_start, x_end, ey_s.shape),
            spec.offset_index,
        ),
        "Ez": (
            *_support_slices("Ez", "y", z_start, z_end, x_start, x_end, ez_s.shape),
            spec.plane_index,
        ),
        "Hx": (
            *_support_slices("Hx", "y", z_start, z_end, x_start, x_end, hx_s.shape),
            spec.offset_index,
        ),
        "Hy": (
            *_support_slices("Hy", "y", z_start, z_end, x_start, x_end, hy_s.shape),
            spec.plane_index,
        ),
        "Hz": (
            *_support_slices("Hz", "y", z_start, z_end, x_start, x_end, hz_s.shape),
            spec.offset_index,
        ),
    }
    indices = {name: (idx[0], idx[2], idx[1]) for name, idx in indices.items()}
    profiles = _crop_window_all(
        staggered, z_start, z_end, x_start, x_end, _direction_sign(spec.direction), spec
    )
    if _direction_sign(spec.direction) < 0.0:
        for component in ("Ex", "Ey", "Ez"):
            profiles[component] = -profiles[component]
    initial_power = _normalize_profiles_by_flux(
        profiles,
        axis="y",
        d_area=spec.resolution**2,
        direction_sign=_direction_sign(spec.direction),
    )
    return profiles, indices, {"initial_power": initial_power}


def _build_z_profiles(
    fields: dict[str, np.ndarray],
    spec: ModePlaneSpec,
) -> tuple[dict[str, np.ndarray], dict[str, ComponentIndex], dict[str, float]]:
    ex_s = _stagger_half(fields["Ex"], axis=1)
    ey_s = _stagger_half(fields["Ey"], axis=0)
    ez_s = fields["Ez"]
    hx_s = _stagger_half(fields["Hx"], axis=0)
    hy_s = _stagger_half(fields["Hy"], axis=1)
    hz_s = _stagger_both(fields["Hz"])
    nz, ny, nx = spec.grid_shape
    x_start, x_end = _padded_bounds(
        spec.center[0], spec.width, spec.resolution, nx, spec.aperture_pad_cells
    )
    y_start, y_end = _padded_bounds(
        spec.center[1], spec.height, spec.resolution, ny, spec.aperture_pad_cells
    )
    e_z_idx = int(np.clip(spec.plane_index, 0, nz - 1))
    h_z_idx = int(np.clip(spec.offset_index, 0, max(nz - 2, 0)))
    ez_z_idx = int(np.clip(spec.plane_index, 0, max(nz - 2, 0)))
    hz_z_idx = int(np.clip(spec.offset_index, 0, nz - 1))
    staggered = {"Ex": ex_s, "Ey": ey_s, "Ez": ez_s, "Hx": hx_s, "Hy": hy_s, "Hz": hz_s}
    indices: dict[str, ComponentIndex] = {
        "Ex": (
            e_z_idx,
            *_support_slices("Ex", "z", y_start, y_end, x_start, x_end, ex_s.shape),
        ),
        "Ey": (
            e_z_idx,
            *_support_slices("Ey", "z", y_start, y_end, x_start, x_end, ey_s.shape),
        ),
        "Ez": (
            ez_z_idx,
            *_support_slices("Ez", "z", y_start, y_end, x_start, x_end, ez_s.shape),
        ),
        "Hx": (
            h_z_idx,
            *_support_slices("Hx", "z", y_start, y_end, x_start, x_end, hx_s.shape),
        ),
        "Hy": (
            h_z_idx,
            *_support_slices("Hy", "z", y_start, y_end, x_start, x_end, hy_s.shape),
        ),
        "Hz": (
            hz_z_idx,
            *_support_slices("Hz", "z", y_start, y_end, x_start, x_end, hz_s.shape),
        ),
    }
    profiles = _crop_window_all(
        staggered, y_start, y_end, x_start, x_end, _direction_sign(spec.direction), spec
    )
    initial_power = _normalize_profiles_by_flux(
        profiles,
        axis="z",
        d_area=spec.resolution**2,
        direction_sign=_direction_sign(spec.direction),
    )
    return profiles, indices, {"initial_power": initial_power}


def _transpose_between_axes(
    values: np.ndarray,
    src_axes: tuple[AxisName, AxisName],
    dst_axes: tuple[AxisName, AxisName],
) -> np.ndarray:
    return np.transpose(
        np.asarray(values, dtype=np.complex128),
        [src_axes.index(axis) for axis in dst_axes],
    )


def _solver_axes_for_axis(axis: AxisName) -> tuple[AxisName, AxisName]:
    return cast(
        tuple[AxisName, AxisName],
        tuple(value for value in _AXIS_NAMES if value != axis),
    )


def _stagger_half(field: np.ndarray, axis: int) -> np.ndarray:
    if field.shape[axis] <= 1:
        return field
    if axis == 0:
        return 0.5 * (field[:-1, :] + field[1:, :])
    return 0.5 * (field[:, :-1] + field[:, 1:])


def _stagger_both(field: np.ndarray) -> np.ndarray:
    out = field
    if out.shape[1] > 1:
        out = 0.5 * (out[:, :-1] + out[:, 1:])
    if out.shape[0] > 1:
        out = 0.5 * (out[:-1, :] + out[1:, :])
    return out


def _padded_bounds(
    center_value: float,
    extent: float,
    resolution: float,
    limit: int,
    pad_cells: int,
) -> tuple[int, int]:
    padded = float(extent) + 2.0 * max(0, int(pad_cells)) * float(resolution)
    center_idx = round(float(center_value) / float(resolution))
    half = max(1, round(0.5 * padded / float(resolution)))
    return max(0, center_idx - half), min(int(limit), center_idx + half)


def _support_slices(
    component: str,
    axis: AxisName,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
    field_shape: tuple[int, int],
) -> tuple[slice, slice]:
    row_axis, col_axis = {"x": ("z", "y"), "y": ("z", "x"), "z": ("y", "x")}[axis]
    row_stop = _support_stop_for_offset(
        row_start, row_stop, _YEE_OFFSETS_3D[component][row_axis]
    )
    col_stop = _support_stop_for_offset(
        col_start, col_stop, _YEE_OFFSETS_3D[component][col_axis]
    )
    return (
        slice(row_start, min(row_stop, int(field_shape[0]))),
        slice(col_start, min(col_stop, int(field_shape[1]))),
    )


def _support_stop_for_offset(start: int, stop: int, offset: float) -> int:
    if float(offset) == 0.5 and int(stop) - int(start) > 1:
        return int(stop) - 1
    return int(stop)


def _crop_window_all(
    staggered: dict[str, np.ndarray],
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
    direction_sign: float,
    spec: ModePlaneSpec,
) -> dict[str, np.ndarray]:
    profiles = {}
    row_axis, col_axis = {"x": ("z", "y"), "y": ("z", "x"), "z": ("y", "x")}[spec.axis]
    for component, values in staggered.items():
        comp_row_stop = _support_stop_for_offset(
            row_start, row_stop, _YEE_OFFSETS_3D[component][row_axis]
        )
        comp_col_stop = _support_stop_for_offset(
            col_start, col_stop, _YEE_OFFSETS_3D[component][col_axis]
        )
        row_end = min(comp_row_stop, values.shape[0])
        col_end = min(comp_col_stop, values.shape[1])
        cropped = values[row_start:row_end, col_start:col_end]
        window = _tukey2d(
            cast(tuple[int, int], cropped.shape), alpha=spec.aperture_window_alpha
        )
        profiles[component] = direction_sign * cropped * window
    return profiles


def _tukey2d(shape: tuple[int, int], alpha: float) -> np.ndarray:
    rows, cols = shape
    return _tukey(rows, alpha)[:, None] * _tukey(cols, alpha)[None, :]


def _tukey(count: int, alpha: float) -> np.ndarray:
    if count <= 0:
        return np.ones((0,), dtype=np.float64)
    if count == 1 or count <= 2:
        return np.ones((count,), dtype=np.float64)
    n = np.arange(count, dtype=np.float64)
    width = max(float(alpha) * (count - 1) / 2.0, np.finfo(float).eps)
    left = 0.5 * (1.0 + np.cos(np.pi * (n / width - 1.0)))
    right = 0.5 * (1.0 + np.cos(np.pi * ((n - (count - 1 - width)) / width)))
    return np.where(n < width, left, np.where(n > (count - 1) - width, right, 1.0))


def _normalize_profiles_by_flux(
    profiles: dict[str, np.ndarray],
    axis: AxisName,
    d_area: float,
    direction_sign: float,
) -> float:
    flux = _modal_power_from_profiles(
        profiles, axis=axis, d_area=d_area, direction_sign=direction_sign
    )
    if np.isfinite(flux) and abs(flux) > np.finfo(float).tiny:
        scale = float(np.sqrt(1.0 / abs(flux)))
        for key, value in profiles.items():
            profiles[key] = np.asarray(value, dtype=np.complex128) * scale
    return float(flux)


def _normalize_profiles_by_phase_referenced_flux(
    profiles: dict[str, np.ndarray],
    indices: dict[str, ComponentIndex],
    *,
    axis: AxisName,
    d_area: float,
    direction_sign: float,
    omega: float,
    k_num: float,
    ref_coord: float,
    resolution: float,
) -> tuple[dict[str, np.ndarray], float, float]:
    referenced = _phase_reference_profiles(
        profiles,
        indices,
        axis=axis,
        omega=omega,
        k_num=k_num,
        ref_coord=ref_coord,
        resolution=resolution,
    )
    flux = _modal_power_from_profiles(
        referenced,
        axis=axis,
        d_area=d_area,
        direction_sign=direction_sign,
    )
    if (not np.isfinite(flux)) or abs(flux) <= np.finfo(float).tiny:
        return profiles, 1.0, float(flux)
    scale = float(np.sqrt(1.0 / abs(flux)))
    return (
        {
            key: np.asarray(value, dtype=np.complex128) * scale
            for key, value in profiles.items()
        },
        scale,
        float(flux) * scale**2,
    )


def _phase_reference_profiles(
    profiles: dict[str, np.ndarray],
    indices: dict[str, ComponentIndex],
    *,
    axis: AxisName,
    omega: float,
    k_num: float,
    ref_coord: float,
    resolution: float,
) -> dict[str, np.ndarray]:
    out = {}
    dummy_spec = _CoordSpec(axis=axis, resolution=resolution)
    for component, value in profiles.items():
        axis_idx = _axis_index_from_component_indices(indices.get(component), axis)
        coord = _component_axis_coord(component, axis_idx, dummy_spec)
        delay = _numeric_phase_delay(omega, k_num, coord - ref_coord)
        out[component] = np.asarray(value, dtype=np.complex128) * np.exp(
            -1j * omega * delay
        )
    return out


@dataclass(frozen=True)
class _CoordSpec:
    axis: AxisName
    resolution: float


def _modal_power_from_profiles(
    profiles: dict[str, np.ndarray],
    axis: AxisName,
    d_area: float,
    direction_sign: float,
) -> float:
    if axis == "x":
        terms = ("Ey", "Ez", "Hz", "Hy")
    elif axis == "y":
        terms = ("Ez", "Ex", "Hx", "Hz")
    else:
        terms = ("Ex", "Ey", "Hy", "Hx")
    arrays = [np.asarray(profiles.get(name, ()), dtype=np.complex128) for name in terms]
    if any(arr.size == 0 for arr in arrays):
        return 0.0
    a0, a1, b0, b1 = arrays
    flux = np.vdot(b0.reshape(-1), a0.reshape(-1)) - np.vdot(
        b1.reshape(-1), a1.reshape(-1)
    )
    return float(0.5 * direction_sign * np.real(flux * float(d_area)))


def _axis_index_from_component_indices(
    indices: ComponentIndex | None, axis: AxisName
) -> int | None:
    if indices is None:
        return None
    axis_pos = {"x": 2, "y": 1, "z": 0}[axis]
    value = indices[axis_pos]
    return None if isinstance(value, slice) else int(value)


def _component_axis_coord(
    component: str, axis_index: int | None, spec: ModePlaneSpec | _CoordSpec
) -> float:
    if axis_index is None:
        return 0.0
    staggered_along_axis = {
        "x": {"Ex", "Hy", "Hz"},
        "y": {"Ey", "Hx", "Hz"},
        "z": {"Ez", "Hx", "Hy"},
    }
    offset = 1.0 if component in staggered_along_axis[spec.axis] else 0.5
    return (int(axis_index) + offset) * float(spec.resolution)


def _solve_numeric_k_axis(
    omega: float,
    dt: float | None,
    d_axis: float,
    neff: complex,
) -> float:
    neff_r = max(float(np.real(neff)), 1e-30)
    if dt is None:
        return float(omega) * neff_r / 299_792_458.0
    s = 299_792_458.0 * float(dt) / (neff_r * float(d_axis))
    if (not np.isfinite(s)) or s <= 1e-30:
        return float(omega) * neff_r / 299_792_458.0
    rhs = np.sin(0.5 * float(omega) * float(dt)) / s
    k_num = (2.0 / float(d_axis)) * np.arcsin(float(np.clip(rhs, -1.0, 1.0)))
    if np.isfinite(k_num) and k_num > 0.0:
        return float(k_num)
    return float(omega) * neff_r / 299_792_458.0


def _numeric_phase_delay(omega: float, k_num: float, delta_s: float) -> float:
    return float(float(k_num) * float(delta_s) / max(abs(float(omega)), 1e-30))


def _detect_transverse_symmetry_axes(
    eps_profile: np.ndarray,
    threshold: float = 0.995,
) -> tuple[int, ...]:
    eps = np.asarray(np.real(eps_profile), dtype=float)
    symmetric = []
    for axis in range(eps.ndim):
        denom = float(np.sum(np.abs(eps) ** 2))
        corr = (
            0.0
            if denom <= 1e-18
            else float(np.real(np.sum(eps * np.flip(eps, axis=axis))) / denom)
        )
        if corr >= threshold:
            symmetric.append(axis)
    return tuple(symmetric)


def _enforce_componentwise_parity(
    component_map: dict[str, np.ndarray],
    symmetric_axes: tuple[int, ...],
) -> dict[str, np.ndarray]:
    out = {}
    for name, value in component_map.items():
        arr = np.asarray(value, dtype=np.complex128)
        for axis in symmetric_axes:
            if arr.ndim <= axis:
                continue
            flipped = np.flip(arr, axis=axis)
            if flipped.shape != arr.shape:
                continue
            overlap = float(np.real(np.sum(arr * np.conjugate(flipped))))
            parity = 1.0 if overlap >= 0.0 else -1.0
            arr = 0.5 * (arr + parity * flipped)
        out[name] = arr
    return out


def _runtime_oriented_profiles(
    profiles: dict[str, np.ndarray],
    axis: AxisName,
    direction_sign: float,
) -> dict[str, np.ndarray]:
    out = {
        key: np.asarray(value, dtype=np.complex128) for key, value in profiles.items()
    }
    if axis != "y":
        return out
    if direction_sign > 0.0:
        out["Ex"] = -out["Ex"]
        out["Hz"] = -out["Hz"]
    else:
        out["Ez"] = -out["Ez"]
        out["Hx"] = -out["Hx"]
    return out


def _backward_mode_from_forward(
    profiles: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        key: (-value if key.startswith("H") else value.copy())
        for key, value in profiles.items()
    }


def _direction_sign(direction: str) -> float:
    return 1.0 if str(direction).startswith("+") else -1.0
