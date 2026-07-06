"""Gaussian beam field-profile generation internals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from beamz._yee import component_axis_offsets_3d
from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.devices.sources._profiles import FieldProfile3D

Direction3D = Literal["+x", "-x", "+y", "-y", "+z", "-z"]

_AXES = ("x", "y", "z")
_INDEX_AXES = ("z", "y", "x")
_AXIS_TO_VECTOR = {
    "x": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    "y": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    "z": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
}
_TRANSVERSE_AXES = {
    "x": ("y", "z"),
    "y": ("z", "x"),
    "z": ("x", "y"),
}
_FIELD_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _parse_direction(direction: str) -> tuple[str, float]:
    if direction not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
        raise ValueError(f"Unsupported Gaussian beam direction {direction!r}.")
    return direction[1], (1.0 if direction[0] == "+" else -1.0)


def _unit_vector(axis: str) -> np.ndarray:
    return _AXIS_TO_VECTOR[axis].copy()


def _normalize(vector: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 1e-30:
        raise ValueError(f"{name} must have non-zero finite norm.")
    return arr / norm


def _as_xyz(value, *, name: str) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{name} must contain exactly three values.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values.")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _component_field_shape(
    component: str,
    grid_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    offsets = component_axis_offsets_3d(component)
    dims = {"z": int(grid_shape[0]), "y": int(grid_shape[1]), "x": int(grid_shape[2])}
    return tuple(
        max(0, dims[axis] - (1 if float(offsets[axis]) == 0.5 else 0))
        for axis in _INDEX_AXES
    )


@dataclass(frozen=True)
class GaussianBeamProfile:
    """Generate a Gaussian beam as a prepared planar ``FieldProfile3D``."""

    center: tuple[float, float, float]
    size: float | tuple[float, float] | tuple[float, float, float]
    direction: Direction3D
    angle_theta: float
    angle_phi: float
    pol_angle: float
    waist_radius: float
    waist_distance: float
    wavelength: float
    background_index: float = 1.0
    power: float = 1.0

    def __post_init__(self):
        _as_xyz(self.center, name="center")
        for name in (
            "angle_theta",
            "angle_phi",
            "pol_angle",
            "waist_radius",
            "waist_distance",
            "wavelength",
            "background_index",
            "power",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if float(self.waist_radius) <= 0.0:
            raise ValueError("waist_radius must be positive.")
        if float(self.wavelength) <= 0.0:
            raise ValueError("wavelength must be positive.")
        if float(self.background_index) <= 0.0:
            raise ValueError("background_index must be positive.")
        if float(self.power) < 0.0:
            raise ValueError("power must be non-negative.")
        _parse_direction(self.direction)
        self._transverse_extents()

    @property
    def axis(self) -> str:
        axis, _sign = _parse_direction(self.direction)
        return axis

    @property
    def direction_sign(self) -> float:
        _axis, sign = _parse_direction(self.direction)
        return sign

    @property
    def omega(self) -> float:
        return float(2.0 * np.pi * LIGHT_SPEED / float(self.wavelength))

    @property
    def medium_wavenumber(self) -> float:
        return float(2.0 * np.pi * float(self.background_index) / self.wavelength)

    def propagation_unit_vector(self) -> np.ndarray:
        axis = self.axis
        normal = self.direction_sign * _unit_vector(axis)
        t1_axis, t2_axis = _TRANSVERSE_AXES[axis]
        t1 = _unit_vector(t1_axis)
        t2 = _unit_vector(t2_axis)
        theta = float(self.angle_theta)
        phi = float(self.angle_phi)
        return _normalize(
            np.cos(theta) * normal
            + np.sin(theta) * (np.cos(phi) * t1 + np.sin(phi) * t2),
            name="propagation direction",
        )

    def propagation_vector(self) -> np.ndarray:
        return self.medium_wavenumber * self.propagation_unit_vector()

    def electric_unit_vector(self) -> np.ndarray:
        axis = self.axis
        t1_axis, t2_axis = _TRANSVERSE_AXES[axis]
        t1 = _unit_vector(t1_axis)
        t2 = _unit_vector(t2_axis)
        seed = np.cos(float(self.pol_angle)) * t1 + np.sin(float(self.pol_angle)) * t2
        k_hat = self.propagation_unit_vector()
        projected = seed - float(np.dot(seed, k_hat)) * k_hat
        if float(np.linalg.norm(projected)) <= 1e-12:
            projected = np.cross(k_hat, t1)
        return _normalize(projected, name="electric polarization")

    def magnetic_unit_vector(self) -> np.ndarray:
        return _normalize(
            np.cross(self.propagation_unit_vector(), self.electric_unit_vector()),
            name="magnetic polarization",
        )

    def field_profile(
        self,
        *,
        resolution: float,
        grid_shape: tuple[int, int, int],
    ) -> FieldProfile3D:
        """Sample the Gaussian beam on a Yee source plane."""
        resolution = float(resolution)
        if not np.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("resolution must be positive and finite.")
        if len(tuple(grid_shape)) != 3:
            raise ValueError("grid_shape must be a 3D cell shape in (z, y, x) order.")
        grid_shape = tuple(int(v) for v in grid_shape)
        if any(v <= 1 for v in grid_shape):
            raise ValueError("grid_shape dimensions must be greater than one.")

        axis = self.axis
        direction_sign = self.direction_sign
        center_xyz = dict(zip(_AXES, _as_xyz(self.center, name="center"), strict=True))
        transverse_slices = self._transverse_slices(resolution, grid_shape)
        phase_ref_coord = float(center_xyz[axis])
        phase_plane_coord = float(center_xyz[axis])
        k_vector = self.propagation_vector()
        k_axis = float(k_vector[_AXES.index(axis)])
        e_hat = self.electric_unit_vector()
        h_hat = (
            float(self.background_index)
            / float(np.sqrt(MU_0 / EPS_0))
            * np.cross(self.propagation_unit_vector(), e_hat)
        )
        amplitude = self._power_amplitude_scale(
            resolution,
            transverse_slices,
            k_normal_abs=abs(
                float(np.dot(self.propagation_unit_vector(), _unit_vector(axis)))
            ),
        )

        components: dict[str, np.ndarray] = {}
        indices: dict[str, tuple[slice, slice, slice]] = {}
        for component in _FIELD_COMPONENTS:
            index = self._component_index(
                component,
                resolution=resolution,
                grid_shape=grid_shape,
                transverse_slices=transverse_slices,
            )
            coords = self._component_coordinate_arrays(
                component,
                index,
                resolution=resolution,
            )
            scalar = amplitude * self._scalar_profile(
                coords,
                center_xyz=center_xyz,
                phase_ref_coord=phase_ref_coord,
                k_vector=k_vector,
                k_axis=k_axis,
            )
            vector = e_hat if component.startswith("E") else h_hat
            component_axis = component[1].lower()
            components[component] = (
                scalar * float(vector[_AXES.index(component_axis)])
            ).astype(np.complex128, copy=False)
            indices[component] = index

        return FieldProfile3D(
            components=components,
            indices=indices,
            axis=axis,  # type: ignore[arg-type]
            direction_sign=direction_sign,
            omega=self.omega,
            k_axis=k_axis,
            phase_ref_coord=phase_ref_coord,
            phase_plane_coord=phase_plane_coord,
        )

    def _transverse_extents(self) -> tuple[float, float]:
        values = np.asarray(self.size, dtype=np.float64).reshape(-1)
        if values.size == 1:
            out = (float(values[0]), float(values[0]))
        elif values.size == 2:
            out = (float(values[0]), float(values[1]))
        elif values.size == 3:
            extents = dict(zip(_AXES, values, strict=True))
            t1_axis, t2_axis = _TRANSVERSE_AXES[self.axis]
            out = (float(extents[t1_axis]), float(extents[t2_axis]))
        else:
            raise ValueError("size must be a scalar, 2-tuple, or 3-tuple.")
        if any((not np.isfinite(v)) or v <= 0.0 for v in out):
            raise ValueError("size extents must be positive finite values.")
        return out

    def _transverse_slices(
        self,
        resolution: float,
        grid_shape: tuple[int, int, int],
    ) -> dict[str, slice]:
        center_xyz = dict(zip(_AXES, _as_xyz(self.center, name="center"), strict=True))
        extents = self._transverse_extents()
        out: dict[str, slice] = {}
        dims = {"z": grid_shape[0], "y": grid_shape[1], "x": grid_shape[2]}
        for axis_name, extent in zip(_TRANSVERSE_AXES[self.axis], extents, strict=True):
            center = float(center_xyz[axis_name])
            start = int(np.floor((center - 0.5 * extent) / resolution))
            stop = int(np.ceil((center + 0.5 * extent) / resolution))
            start = max(0, min(start, int(dims[axis_name]) - 1))
            stop = max(start + 1, min(stop, int(dims[axis_name])))
            out[axis_name] = slice(start, stop)
        return out

    def _component_index(
        self,
        component: str,
        *,
        resolution: float,
        grid_shape: tuple[int, int, int],
        transverse_slices: dict[str, slice],
    ) -> tuple[slice, slice, slice]:
        center_xyz = dict(zip(_AXES, _as_xyz(self.center, name="center"), strict=True))
        offsets = component_axis_offsets_3d(component)
        field_shape = _component_field_shape(component, grid_shape)
        items: list[int | slice] = []
        for dim, axis_name in enumerate(_INDEX_AXES):
            if axis_name == self.axis:
                raw = int(
                    round(
                        float(center_xyz[axis_name]) / resolution - offsets[axis_name]
                    )
                )
                items.append(max(0, min(raw, int(field_shape[dim]) - 1)))
                continue
            source_slice = transverse_slices[axis_name]
            start = max(0, min(int(source_slice.start or 0), int(field_shape[dim]) - 1))
            stop = max(
                start + 1,
                min(int(source_slice.stop or start + 1), int(field_shape[dim])),
            )
            items.append(slice(start, stop))
        return tuple(items)  # type: ignore[return-value]

    def _component_coordinate_arrays(
        self,
        component: str,
        index: tuple[slice, slice, slice],
        *,
        resolution: float,
    ) -> dict[str, np.ndarray]:
        offsets = component_axis_offsets_3d(component)
        axis_values: dict[str, np.ndarray | float] = {}
        mesh_axes: list[str] = []
        mesh_values: list[np.ndarray] = []
        for axis_name, item in zip(_INDEX_AXES, index, strict=True):
            if isinstance(item, slice):
                values = (
                    np.arange(
                        int(item.start or 0), int(item.stop or 0), dtype=np.float64
                    )
                    + float(offsets[axis_name])
                ) * resolution
                axis_values[axis_name] = values
                mesh_axes.append(axis_name)
                mesh_values.append(values)
            else:
                axis_values[axis_name] = (
                    int(item) + float(offsets[axis_name])
                ) * resolution

        meshes = np.meshgrid(*mesh_values, indexing="ij")
        coords: dict[str, np.ndarray] = {}
        shape = meshes[0].shape if meshes else ()
        for axis_name, values in axis_values.items():
            if axis_name in mesh_axes:
                coords[axis_name] = meshes[mesh_axes.index(axis_name)]
            else:
                coords[axis_name] = np.full(shape, float(values), dtype=np.float64)
        return coords

    def _scalar_profile(
        self,
        coords: dict[str, np.ndarray],
        *,
        center_xyz: dict[str, float],
        phase_ref_coord: float,
        k_vector: np.ndarray,
        k_axis: float,
    ) -> np.ndarray:
        r = np.stack(
            [coords[axis] - float(center_xyz[axis]) for axis in _AXES],
            axis=0,
        )
        e_hat = self.electric_unit_vector()
        v_hat = self.magnetic_unit_vector()
        u_coord = np.tensordot(e_hat, r, axes=(0, 0))
        v_coord = np.tensordot(v_hat, r, axes=(0, 0))
        rho2 = u_coord**2 + v_coord**2
        radius, curvature, gouy = self._beam_radius_curvature_gouy()
        envelope = np.exp(-rho2 / max(radius**2, 1e-300))
        phase = -np.tensordot(k_vector, r, axes=(0, 0))
        phase += float(k_axis) * (coords[self.axis] - float(phase_ref_coord))
        if np.isfinite(curvature):
            phase += -self.medium_wavenumber * rho2 / (2.0 * curvature)
        phase += gouy
        return envelope * np.exp(1j * phase)

    def _beam_radius_curvature_gouy(self) -> tuple[float, float, float]:
        waist = float(self.waist_radius)
        wavelength_medium = float(self.wavelength) / float(self.background_index)
        rayleigh = np.pi * waist**2 / wavelength_medium
        z = float(self.waist_distance)
        radius = waist * np.sqrt(1.0 + (z / rayleigh) ** 2)
        if abs(z) <= 1e-30:
            curvature = np.inf
        else:
            curvature = z * (1.0 + (rayleigh / z) ** 2)
        gouy = float(np.arctan2(z, rayleigh))
        return float(radius), float(curvature), gouy

    def _power_amplitude_scale(
        self,
        resolution: float,
        transverse_slices: dict[str, slice],
        *,
        k_normal_abs: float,
    ) -> float:
        if float(self.power) == 0.0:
            return 0.0
        center_xyz = dict(zip(_AXES, _as_xyz(self.center, name="center"), strict=True))
        t_axes = _TRANSVERSE_AXES[self.axis]
        coords = []
        for axis_name in t_axes:
            item = transverse_slices[axis_name]
            coords.append(
                (np.arange(int(item.start or 0), int(item.stop or 0)) + 0.5)
                * resolution
                - float(center_xyz[axis_name])
            )
        a, b = np.meshgrid(coords[0], coords[1], indexing="ij")
        radius, _curvature, _gouy = self._beam_radius_curvature_gouy()
        envelope2 = np.exp(-2.0 * (a**2 + b**2) / max(radius**2, 1e-300))
        eta = float(np.sqrt(MU_0 / EPS_0)) / float(self.background_index)
        flux = (
            0.5
            * max(float(k_normal_abs), 1e-30)
            / eta
            * float(np.sum(envelope2))
            * float(resolution) ** 2
        )
        if (not np.isfinite(flux)) or flux <= 1e-300:
            return 0.0
        return float(np.sqrt(float(self.power) / flux))
