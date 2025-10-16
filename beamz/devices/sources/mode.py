from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.sources.solve import solve_modes
from beamz.sources.tsfs import TFSFPlaneSource

Direction = Literal["+x", "-x", "+y", "-y", "+z", "-z"]
Axis = Literal[0, 1, 2]
Polarization = Literal["te", "tm", None]


@dataclass(frozen=True)
class Box:
    center: tuple[float, ...]
    size: tuple[float, ...]


def _direction_to_axis(direction: Direction) -> Axis:
    if "x" in direction:
        return 0
    if "y" in direction:
        return 1
    if "z" in direction:
        return 2
    raise ValueError(f"Unknown propagation direction '{direction}'")


def _direction_sign(direction: Direction) -> str:
    return "+" if direction.startswith("+") else "-"


def _ensure_box(
    plane: Box | dict | Sequence[Sequence[float]] | None,
    dims: int,
    default_center: Sequence[float],
    default_size: Sequence[float],
) -> Box:
    if plane is None:
        return Box(tuple(default_center), tuple(default_size))
    if isinstance(plane, Box):
        return plane
    if isinstance(plane, dict):
        center = tuple(float(c) for c in plane.get("center", default_center))
        size = tuple(float(s) for s in plane.get("size", default_size))
    elif isinstance(plane, (list, tuple)) and len(plane) == 2:
        center = tuple(float(c) for c in plane[0])
        size = tuple(float(s) for s in plane[1])
    else:
        raise TypeError("plane must be Box, dict, or (center, size) tuple")

    if len(center) != dims:
        if len(center) == 2 and dims == 3:
            center = (*center, 0.0)
        else:
            raise ValueError("Plane center dimensionality mismatch")
    if len(size) != dims:
        if len(size) == 2 and dims == 3:
            size = (*size, 0.0)
        else:
            raise ValueError("Plane size dimensionality mismatch")
    return Box(center=tuple(center), size=tuple(size))


def _coerce_center(center, dims: int, default_center: Sequence[float]) -> tuple[float, ...]:
    if center is None:
        return tuple(default_center)
    if isinstance(center, (list, tuple)):
        if len(center) != dims:
            raise ValueError(f"center must have length {dims}, got {len(center)}")
        return tuple(float(c) for c in center)
    raise TypeError("center must be a tuple/list of coordinates")


def _coerce_size(width, dims: int, default_size: Sequence[float], axis: Axis) -> tuple[float, ...]:
    if width is None:
        return tuple(default_size)
    if isinstance(width, (int, float)):
        size = list(default_size)
        for idx in range(dims):
            size[idx] = 0.0 if idx == axis else float(width)
        return tuple(size)
    if isinstance(width, (list, tuple)):
        if len(width) != dims:
            raise ValueError(f"width must have length {dims}, got {len(width)}")
        return tuple(float(w) for w in width)
    raise TypeError("width must be a float or a tuple/list of floats")


class ModeSource:
    """Unidirectional Huygens-mode source built on a TFSF plane."""

    def __init__(
        self,
        grid,
        center=None,
        width=None,
        wavelength: float = 1.55e-6,
        direction: Direction = "+x",
        mode: int = 0,
        target_neff: float | None = None,
        pol: Polarization = "tm",
        signal: Sequence[float] | None = None,
    ) -> None:
        if getattr(grid, "permittivity", None) is None:
            raise ValueError("Grid must expose a 'permittivity' array. Did you call Design.rasterize()?")
        self.grid = grid
        self.wavelength = float(wavelength)
        self.omega = 2 * np.pi * LIGHT_SPEED / self.wavelength
        self.direction = direction
        self.mode_index = int(mode)
        if self.mode_index < 0:
            raise ValueError("mode must be non-negative")
        self.target_neff = target_neff
        self.polarization: Polarization = pol

        if signal is None:
            self.signal = np.array([1.0], dtype=float)
        else:
            self.signal = np.asarray(signal, dtype=float)
        self.max_signal_magnitude = float(np.max(np.abs(self.signal))) if self.signal.size else 0.0

        permittivity = np.asarray(self.grid.permittivity, dtype=float)
        if permittivity.ndim == 2:
            dims = 2
            default_center = (self.grid.width / 2, self.grid.height / 2)
            default_size = (0.0, min(self.grid.width, self.grid.height))
            if self.polarization is None:
                self.polarization = "tm"
        elif permittivity.ndim == 3:
            dims = 3
            depth = getattr(self.grid, "depth", 0.0) or 0.0
            default_center = (self.grid.width / 2, self.grid.height / 2, depth / 2)
            default_size = (0.0, self.grid.height, depth if depth > 0 else 1e-6)
        else:
            raise ValueError("Unsupported permittivity dimensionality")

        axis = _direction_to_axis(self.direction)
        plane = None
        if center is not None or width is not None:
            center_tuple = _coerce_center(center, dims, default_center)
            size_tuple = _coerce_size(width, dims, default_size, axis)
            plane = Box(center_tuple, size_tuple)

        self.plane = _ensure_box(plane, dims, default_center, default_size)

        center_tuple = self.plane.center
        if len(center_tuple) == 2:
            center_tuple = (*center_tuple, 0.0)
        self.center = tuple(float(c) for c in center_tuple)
        size_tuple = self.plane.size
        if len(size_tuple) == 2:
            size_tuple = (*size_tuple, 0.0)
        self.plane_size = tuple(float(s) for s in size_tuple)

        self.axis = axis
        if permittivity.ndim == 2 and self.axis != 0:
            raise NotImplementedError("2D ModeSource currently supports propagation along the x-axis only")
        self.signal_phase_shift = 0.0
        self.mode_profiles: list[list[dict[str, complex]]] = []
        self.max_field_amplitude: float | None = None
        self.max_power_density: float | None = None
        self.total_power: float | None = None

        self._mode_cache = None
        self._tsfs: TFSFPlaneSource | None = None
        self._electric_currents: Dict[str, np.ndarray] = {}
        self._magnetic_currents: Dict[str, np.ndarray] = {}
        self._grid_indices: Dict[str, tuple[slice, int]] = {}
        self._coordinate_vectors: Dict[str, np.ndarray] = {}
        self._design = getattr(grid, "design", None)

        self._update_line_endpoints()

    # Compatibility helpers -------------------------------------------------
    def compute_modes_on_fdtd_grid(self, *args, **kwargs):
        del args, kwargs
        return self.compute_modes(force=True)

    def initialize_for_fdtd(self, *args, **kwargs):
        del args, kwargs
        self.compute_modes(force=True)

    # Geometry ---------------------------------------------------------------
    @property
    def design(self):
        return self._design

    @design.setter
    def design(self, value):
        self._design = value

    def _update_line_endpoints(self) -> None:
        axis = self.axis
        if axis == 0:
            self.start = (
                self.center[0],
                self.center[1] - self.plane_size[1] / 2,
                self.center[2],
            )
            self.end = (
                self.center[0],
                self.center[1] + self.plane_size[1] / 2,
                self.center[2],
            )
        elif axis == 1:
            self.start = (
                self.center[0] - self.plane_size[0] / 2,
                self.center[1],
                self.center[2],
            )
            self.end = (
                self.center[0] + self.plane_size[0] / 2,
                self.center[1],
                self.center[2],
            )
        else:
            self.start = (
                self.center[0] - self.plane_size[0] / 2,
                self.center[1] - self.plane_size[1] / 2,
                self.center[2],
            )
            self.end = (
                self.center[0] + self.plane_size[0] / 2,
                self.center[1] + self.plane_size[1] / 2,
                self.center[2],
            )

    # Mode computation ------------------------------------------------------
    def compute_modes(self, force: bool = False):
        if self._mode_cache is not None and not force:
            return self._mode_cache

        permittivity = np.asarray(self.grid.permittivity, dtype=float)
        if permittivity.ndim == 2:
            modes = self._compute_modes_2d(permittivity)
        else:
            raise NotImplementedError("3D mode source support is not implemented yet")

        self._mode_cache = modes
        return modes

    def _compute_modes_2d(self, permittivity: np.ndarray):
        if not hasattr(self.grid, "dx") or not hasattr(self.grid, "dy"):
            raise TypeError("Grid must expose dx and dy properties for mode computation")

        ny, nx = permittivity.shape
        dx = float(getattr(self.grid, "dx"))
        dy = float(getattr(self.grid, "dy"))

        if self.axis not in (0,):
            raise NotImplementedError("2D mode source currently supports propagation along ±x")

        line_data = self._slice_line_y(permittivity, ny, nx, dx, dy)

        eps_profile, coords, grid_indices = line_data

        npml = max(0, min(20, eps_profile.size // 4))
        neff, e_fields, h_fields, _ = solve_modes(
            eps=eps_profile,
            omega=self.omega,
            dL=dy if self.axis == 0 else dx,
            npml=npml,
            m=self.mode_index + 1,
            direction=self.direction,
            filter_pol=self.polarization,
            return_fields=True,
            target_neff=self.target_neff,
        )

        if e_fields.size == 0 or h_fields.size == 0:
            raise RuntimeError("Mode solver did not return field profiles")

        mode_idx = min(self.mode_index, e_fields.shape[0] - 1)
        E = np.squeeze(e_fields[mode_idx])
        H = np.squeeze(h_fields[mode_idx])
        E_cart, H_cart = self._reorder_components(E, H)

        E_cart, H_cart = self._enforce_propagation_direction(E_cart, H_cart)
        self._tsfs = TFSFPlaneSource(electric=E_cart, magnetic=H_cart, axis=self.axis, direction=_direction_sign(self.direction))

        electric_updates = self._tsfs.electric_updates()
        magnetic_updates = self._tsfs.magnetic_updates()

        self._electric_currents = {
            comp: np.atleast_1d(np.asarray(val, dtype=np.complex128))
            for comp, val in electric_updates.items()
            if np.any(val)
        }
        self._magnetic_currents = {
            comp: np.atleast_1d(np.asarray(val, dtype=np.complex128))
            for comp, val in magnetic_updates.items()
            if np.any(val)
        }
        self._coordinate_vectors = {"primary": coords}
        self._grid_indices = self._build_grid_indices(grid_indices)

        poynting = self._tsfs.poynting_density()
        self.max_field_amplitude = float(np.max(np.abs(E_cart)))
        self.max_power_density = float(np.max(np.abs(poynting)))
        cell_length = dy if self.axis == 0 else dx
        self.total_power = float(np.sum(poynting) * cell_length)

        profile = self._build_mode_profile(coords, grid_indices, E_cart, H_cart)
        self.mode_profiles = [profile]
        return [
            {
            "neff": neff[mode_idx],
            "coords": coords,
            "E": E_cart,
            "H": H_cart,
            }
        ]

    def _slice_line_y(self, permittivity, ny, nx, dx, dy):
        y_min = max(0.0, self.center[1] - abs(self.plane_size[1]) / 2)
        y_max = min(self.grid.height, self.center[1] + abs(self.plane_size[1]) / 2)
        y_start = int(np.clip(np.floor(y_min / dy), 0, ny - 1))
        y_end = int(np.clip(np.ceil(y_max / dy), y_start + 1, ny))

        x_idx = int(np.clip(np.round(self.center[0] / dx - 0.5), 0, nx - 1))
        eps_profile = permittivity[y_start:y_end, x_idx]
        coords = (np.arange(y_start, y_end) + 0.5) * dy
        index_info = {
            "y_start": y_start,
            "y_end": y_end,
            "x_index": int(x_idx),
        }
        return np.asarray(eps_profile, dtype=np.float64), coords, index_info

    def _reorder_components(self, E: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if E.ndim == 1:
            E = E[:, np.newaxis]
            H = H[:, np.newaxis]

        if self.axis == 0:
            Ex = E[1]
            Ey = E[2]
            Ez = E[0]
            Hx = H[1]
            Hy = H[2]
            Hz = H[0]
        elif self.axis == 1:
            Ex = E[0]
            Ey = E[2]
            Ez = E[1]
            Hx = H[0]
            Hy = H[2]
            Hz = H[1]
        else:
            Ex = E[0]
            Ey = E[1]
            Ez = E[2]
            Hx = H[0]
            Hy = H[1]
            Hz = H[2]

        E_cart = np.vstack([Ex, Ey, Ez])
        H_cart = np.vstack([Hx, Hy, Hz])
        return E_cart, H_cart

    def _enforce_propagation_direction(self, E: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cross = np.cross(E, np.conjugate(H), axisa=0, axisb=0, axisc=0)
        power = np.real(cross[self.axis])
        avg = np.mean(power)
        sign = 1.0 if self.direction.startswith("+") else -1.0
        if avg * sign < 0:
            H = -H
        return E, H

    def _build_grid_indices(self, index_info: dict):
        y_start = int(index_info["y_start"])
        y_end = int(index_info["y_end"])
        x_idx = int(index_info["x_index"])

        indices: Dict[str, tuple[slice, int]] = {}
        indices["Ez"] = (slice(y_start, y_end), x_idx)
        if y_end - y_start > 1:
            indices["Hy"] = (slice(y_start, y_end - 1), x_idx)
        return indices

    def _build_mode_profile(
        self,
        coords: np.ndarray,
        indices: dict,
        E: np.ndarray,
        H: np.ndarray,
    ) -> list[dict[str, complex]]:
        profile: list[dict[str, complex]] = []
        y_start = indices["y_start"]
        y_end = indices["y_end"]
        x_idx = indices["x_index"]
        y_indices = np.arange(y_start, y_end, dtype=int)
        amplification = 5e6

        for offset, coord in enumerate(coords):
            entry = {
                "x": float(x_idx * getattr(self.grid, "dx")),
                "y": float(coord),
                "z": float(self.center[2]),
                "coord": float(coord),
                "Ex": complex(E[0, offset] * amplification),
                "Ey": complex(E[1, offset] * amplification),
                "Ez": complex(E[2, offset] * amplification),
                "Hx": complex(H[0, offset] * amplification),
                "Hy": complex(H[1, offset] * amplification),
                "Hz": complex(H[2, offset] * amplification),
                "index": int(y_indices[offset]),
            }
            profile.append(entry)
        return profile

    # Injection ----------------------------------------------------------------
    def _get_electric_modulation(self, time_step: int) -> float:
        idx = min(time_step, self.signal.size - 1)
        return float(self.signal[idx])

    def _get_magnetic_modulation(self, time_step: int) -> float:
        if time_step <= 0 or self.signal.size == 0:
            return float(self.signal[0]) if self.signal.size else 0.0
        prev = self.signal[min(time_step - 1, self.signal.size - 1)]
        curr = self.signal[min(time_step, self.signal.size - 1)]
        return float(0.5 * (prev + curr))

    def apply(self, fdtd, time_step: int) -> None:
        if self._tsfs is None:
            self.compute_modes(force=True)
        if self._tsfs is None:
            raise RuntimeError("Mode fields not initialised")

        e_mod = self._get_electric_modulation(time_step)
        h_mod = self._get_magnetic_modulation(time_step)

        dt = float(getattr(fdtd, "dt"))

        ez_entry = self._grid_indices.get("Ez")
        if "Ez" in self._electric_currents and ez_entry is not None:
            y_slice, x_idx = ez_entry
            values = np.asarray(self._electric_currents["Ez"], dtype=np.complex128) * e_mod
            sigma = getattr(fdtd, "sigma", None)
            if sigma is not None:
                sigma_slice = np.asarray(sigma[y_slice, x_idx])
                values = np.where(np.abs(sigma_slice) > 0, 0.0, values)
            eps_slice = fdtd.epsilon_r[y_slice, x_idx]
            fdtd.Ez[y_slice, x_idx] += (dt / (EPS_0 * eps_slice)) * values

        hy_entry = self._grid_indices.get("Hy")
        if "Hy" in self._magnetic_currents and hy_entry is not None:
            y_slice, x_idx = hy_entry
            values = np.asarray(self._prepare_h_component(self._magnetic_currents["Hy"]), dtype=np.complex128) * h_mod
            sigma = getattr(fdtd, "sigma", None)
            if sigma is not None:
                sigma_slice = np.asarray(sigma[y_slice, x_idx])
                values = np.where(np.abs(sigma_slice) > 0, 0.0, values)
            fdtd.Hy[y_slice, x_idx] -= (dt / MU_0) * values

        hx_entry = self._grid_indices.get("Hx")
        if "Hx" in self._magnetic_currents and hx_entry is not None:
            y_slice, x_idx = hx_entry
            values = np.asarray(self._prepare_h_component(self._magnetic_currents["Hx"]), dtype=np.complex128) * h_mod
            sigma = getattr(fdtd, "sigma", None)
            if sigma is not None:
                sigma_slice = np.asarray(sigma[y_slice, x_idx])
                values = np.where(np.abs(sigma_slice) > 0, 0.0, values)
            fdtd.Hx[y_slice, x_idx] -= (dt / MU_0) * values

    def _prepare_h_component(self, values: np.ndarray) -> np.ndarray:
        if values.ndim == 1 and values.size > 1:
            return 0.5 * (values[:-1] + values[1:])
        return values

    # Visualisation -------------------------------------------------------------
    def show(self, component: str = "Ez", figsize=None) -> None:
        self.compute_modes()
        if not self.mode_profiles:
            raise RuntimeError("No mode profile available")

        profile = self.mode_profiles[0]
        coords = np.array([pt["y"] for pt in profile])
        field = np.array([pt.get(component, 0.0) for pt in profile], dtype=np.complex128)

        if field.size == 0:
            raise ValueError(f"Component {component!r} not found in mode profile")

        plt.figure(figsize=figsize)
        plt.plot(coords * 1e6, np.real(field), label="Real")
        plt.plot(coords * 1e6, np.imag(field), label="Imag")
        plt.xlabel("Position (µm)")
        plt.ylabel(component)
        plt.title(f"Mode profile ({component})")
        plt.legend()
        plt.tight_layout()
        plt.show()


__all__ = ["ModeSource"]
