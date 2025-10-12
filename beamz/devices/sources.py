from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from beamz.const import LIGHT_SPEED, µm
from beamz.devices import mode as mode_solver

try:
    from beamz.simulation.meshing import RegularGrid, RegularGrid3D
except ImportError:  # pragma: no cover - during packaging
    RegularGrid = RegularGrid3D = object


@dataclass
class Box:
    center: tuple[float, ...]
    size: tuple[float, ...]


def _direction_to_axis(direction: str) -> int:
    d = direction.strip().lower()
    if "x" in d:
        return 0
    if "y" in d:
        return 1
    if "z" in d:
        return 2
    raise ValueError(f"Unknown propagation direction '{direction}'")


def _ensure_box(plane, dims: int, default_center, default_size) -> Box:
    if plane is None:
        return Box(tuple(default_center), tuple(default_size))
    if isinstance(plane, Box):
        center = tuple(float(c) for c in plane.center)
        size = tuple(float(s) for s in plane.size)
    elif isinstance(plane, dict):
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
    return Box(center, size)


class ModeSource:
    """Visualise eigenmodes from a mesh grid using a Tidy3D-like plane specification."""

    def __init__(
        self,
        grid,
        plane=None,
        wavelength: float = 1.55e-6,
        direction: str = "+x",
        num_modes: int = 3,
        target_neff: float | None = None,
        polarization: str | None = "tm",
    ) -> None:
        if getattr(grid, "permittivity", None) is None:
            raise ValueError("Grid must expose a 'permittivity' array. Did you call Design.rasterize()?")
        self.grid = grid
        self.wavelength = float(wavelength)
        self.omega = 2 * np.pi * LIGHT_SPEED / self.wavelength
        self.direction = direction
        self.num_modes = int(num_modes)
        self.target_neff = target_neff
        self.polarization = polarization

        if self.num_modes <= 0:
            raise ValueError("num_modes must be positive")

        permittivity = np.asarray(self.grid.permittivity, dtype=float)
        if permittivity.ndim == 2:
            dims = 2
            default_center = (self.grid.width / 2, self.grid.height / 2)
            default_size = (0.0, min(self.grid.width, self.grid.height))
        elif permittivity.ndim == 3:
            dims = 3
            default_center = (self.grid.width / 2, self.grid.height / 2, getattr(self.grid, "depth", 0.0) / 2)
            default_size = (0.0, self.grid.height, getattr(self.grid, "depth", 0.0) or 1e-6)
        else:
            raise ValueError("Unsupported permittivity dimensionality")

        box = _ensure_box(plane, dims, default_center, default_size)
        self.plane = box

        self._mode_cache = None
        self._mode_metadata: list[_ModeMetadata] | None = None
        self._mode_type = "1d" if dims == 2 else "2d"

    def compute_modes(self, force: bool = False):
        if self._mode_cache is not None and not force:
            return self._mode_cache

        permittivity = np.asarray(self.grid.permittivity, dtype=float)
        axis = _direction_to_axis(self.direction)

        if permittivity.ndim == 2:
            modes, metadata = self._compute_modes_1d(permittivity, axis)
        elif permittivity.ndim == 3:
            modes, metadata = self._compute_modes_2d(permittivity, axis)
        else:
            raise ValueError("Unsupported permittivity dimensionality")

        self._mode_cache = modes
        self._mode_metadata = metadata
        return modes

    def show(self, modes=None, component="Etot", figsize=None):
        modes = modes or self.compute_modes()
        metadata = self._mode_metadata or []
        if not modes:
            raise RuntimeError("No modes available to visualise")

        if self._mode_type == "1d":
            self._show_1d(modes, metadata, component=component, figsize=figsize)
        else:
            self._show_2d(modes, metadata, component=component, figsize=figsize)

    def _compute_modes_1d(self, permittivity: np.ndarray, axis: int):
        if not isinstance(self.grid, RegularGrid):
            raise TypeError("RegularGrid expected for 1D mode computation")

        ny, nx = permittivity.shape
        dx = getattr(self.grid, "dx", 1.0)
        dy = getattr(self.grid, "dy", 1.0)

        center = self.plane.center
        size = self.plane.size

        if axis == 0:  # propagation along x, transverse coordinate is y
            y_min = max(0.0, center[1] - abs(size[1]) / 2)
            y_max = min(self.grid.height, center[1] + abs(size[1]) / 2)
            y_start = int(np.clip(np.floor(y_min / dy), 0, ny - 1))
            y_end = int(np.clip(np.ceil(y_max / dy), y_start + 1, ny))

            x_idx = int(np.clip(np.round(center[0] / dx - 0.5), 0, nx - 1))
            eps_profile = permittivity[y_start:y_end, x_idx]
            coords = (np.arange(y_start, y_end) + 0.5) * dy
            dL = dy
        else:  # propagation along y, transverse coordinate is x
            x_min = max(0.0, center[0] - abs(size[0]) / 2)
            x_max = min(self.grid.width, center[0] + abs(size[0]) / 2)
            x_start = int(np.clip(np.floor(x_min / dx), 0, nx - 1))
            x_end = int(np.clip(np.ceil(x_max / dx), x_start + 1, nx))

            y_idx = int(np.clip(np.round(center[1] / dy - 0.5), 0, ny - 1))
            eps_profile = permittivity[y_idx, x_start:x_end]
            coords = (np.arange(x_start, x_end) + 0.5) * dx
            dL = dx

        eps_profile = np.asarray(eps_profile, dtype=float)
        if eps_profile.ndim != 1:
            eps_profile = np.squeeze(eps_profile)

        # Mimic modesolver_1d default PML padding for stability
        npml = max(0, min(20, eps_profile.size // 4))

        neff, e_fields, h_fields, _ = mode_solver.solve_modes(
            eps=eps_profile,
            omega=self.omega,
            dL=dL,
            npml=npml,
            m=self.num_modes,
            direction=self.direction,
            filter_pol=self.polarization,
            return_fields=True,
            target_neff=self.target_neff,
        )

        modes = []
        metadata = []
        max_modes = min(self.num_modes, len(neff))
        for idx in range(max_modes):
            Ez = np.squeeze(e_fields[idx][2])
            Ez = Ez if Ez.ndim == 1 else Ez[:, 0]
            modes.append({
                "index": idx,
                "neff": float(np.real(neff[idx])),
                "Ez": Ez,
                "coord": coords,
                "eps": eps_profile,
                "axis": axis,
            })
            metadata.append(_ModeMetadata.from_fields(
                index=idx,
                neff=float(np.real(neff[idx])),
                e_field=e_fields[idx],
                h_field=h_fields[idx],
            ))
        return modes, metadata

    def _compute_modes_2d(self, permittivity: np.ndarray, axis: int):
        if not isinstance(self.grid, RegularGrid3D):
            raise TypeError("RegularGrid3D expected for 2D mode computation")

        nz, ny, nx = permittivity.shape
        dx = getattr(self.grid, "dx", 1.0)
        dy = getattr(self.grid, "dy", 1.0)
        dz = getattr(self.grid, "dz", 1.0)

        center = self.plane.center
        size = self.plane.size

        if axis == 0:  # prop along x, perpendicular plane is y/z
            y_min = max(0.0, center[1] - abs(size[1]) / 2)
            y_max = min(self.grid.height, center[1] + abs(size[1]) / 2)
            z_min = max(0.0, center[2] - abs(size[2]) / 2)
            z_max = min(getattr(self.grid, "depth", dz * nz), center[2] + abs(size[2]) / 2)

            y_start = int(np.clip(np.floor(y_min / dy), 0, ny - 1))
            y_end = int(np.clip(np.ceil(y_max / dy), y_start + 1, ny))
            z_start = int(np.clip(np.floor(z_min / dz), 0, nz - 1))
            z_end = int(np.clip(np.ceil(z_max / dz), z_start + 1, nz))

            eps_slice = permittivity[z_start:z_end, y_start:y_end, :]
            eps_slice = eps_slice.mean(axis=2)
            y_edges = np.linspace(y_start * dy, y_end * dy, y_end - y_start + 1)
            z_edges = np.linspace(z_start * dz, z_end * dz, z_end - z_start + 1)
        elif axis == 1:  # prop along y, plane is x/z
            x_min = max(0.0, center[0] - abs(size[0]) / 2)
            x_max = min(self.grid.width, center[0] + abs(size[0]) / 2)
            z_min = max(0.0, center[2] - abs(size[2]) / 2)
            z_max = min(getattr(self.grid, "depth", dz * nz), center[2] + abs(size[2]) / 2)

            x_start = int(np.clip(np.floor(x_min / dx), 0, nx - 1))
            x_end = int(np.clip(np.ceil(x_max / dx), x_start + 1, nx))
            z_start = int(np.clip(np.floor(z_min / dz), 0, nz - 1))
            z_end = int(np.clip(np.ceil(z_max / dz), z_start + 1, nz))

            eps_slice = permittivity[z_start:z_end, :, x_start:x_end]
            eps_slice = eps_slice.mean(axis=2)
            y_edges = np.linspace(0.0, ny * dy, ny + 1)
            z_edges = np.linspace(z_start * dz, z_end * dz, z_end - z_start + 1)
        else:  # axis == 2 (prop along z)
            x_min = max(0.0, center[0] - abs(size[0]) / 2)
            x_max = min(self.grid.width, center[0] + abs(size[0]) / 2)
            y_min = max(0.0, center[1] - abs(size[1]) / 2)
            y_max = min(self.grid.height, center[1] + abs(size[1]) / 2)

            x_start = int(np.clip(np.floor(x_min / dx), 0, nx - 1))
            x_end = int(np.clip(np.ceil(x_max / dx), x_start + 1, nx))
            y_start = int(np.clip(np.floor(y_min / dy), 0, ny - 1))
            y_end = int(np.clip(np.ceil(y_max / dy), y_start + 1, ny))

            eps_slice = permittivity[:, y_start:y_end, x_start:x_end]
            eps_slice = eps_slice.mean(axis=0)
            y_edges = np.linspace(y_start * dy, y_end * dy, y_end - y_start + 1)
            z_edges = np.linspace(x_start * dx, x_end * dx, x_end - x_start + 1)

        eps_slice = np.transpose(eps_slice, (1, 0))

        tidy_modes = mode_solver.tidy3d_mode_computation_wrapper(
            frequency=LIGHT_SPEED / self.wavelength,
            permittivity_cross_section=eps_slice,
            coords=[y_edges / µm, z_edges / µm],
            direction="+",
            num_modes=self.num_modes,
            precision="double",
        )

        modes = []
        metadata = []
        for idx, mode in enumerate(tidy_modes[: self.num_modes]):
            modes.append({
                "index": idx,
                "neff": float(np.real(mode.neff)),
                "Ez": np.array(mode.Ez),
                "Ey": np.array(mode.Ey),
                "Ex": np.array(mode.Ex),
                "Hy": np.array(mode.Hy),
                "Hx": np.array(mode.Hx),
                "eps": eps_slice,
                "y_edges": y_edges,
                "z_edges": z_edges,
            })
            metadata.append(_ModeMetadata.from_fields(
                index=idx,
                neff=float(np.real(mode.neff)),
                e_field=np.stack([mode.Ex, mode.Ey, mode.Ez], axis=0),
                h_field=np.stack([mode.Hx, mode.Hy, mode.Hz], axis=0),
            ))
        return modes, metadata

    # ------------------------------------------------------------------
    # Plotting utilities
    # ------------------------------------------------------------------
    def _show_1d(self, modes, metadata, component="Etot", figsize=None):
        eps_profile = modes[0]["eps"]
        coord = modes[0]["coord"] / µm
        axis = modes[0]["axis"]
        axis_label = "x" if axis == 0 else "y"

        fig, ax1 = plt.subplots(figsize=figsize or (7, 4))
        ax1.plot(coord, eps_profile, color="black", label="εr")
        ax1.set_xlabel(f"{axis_label} (µm)")
        ax1.set_ylabel("εr", color="black")
        ax1.tick_params(axis="y", labelcolor="black")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        for mode, meta in zip(modes, metadata):
            field = _extract_mode_component_1d(meta, component)
            intensity = np.abs(field) ** 2
            intensity /= np.max(intensity) + 1e-18
            ax2.plot(coord, intensity, label=f"Mode {mode['index']} (neff={mode['neff']:.3f})")

        ax2.set_ylabel(f"|{component}|² (norm)")
        ax2.legend(loc="upper right")
        fig.tight_layout()
        plt.show()


class _ModeMetadata:
    __slots__ = ("index", "neff", "E", "H")

    def __init__(self, index: int, neff: float, E: np.ndarray, H: np.ndarray) -> None:
        self.index = index
        self.neff = neff
        self.E = E
        self.H = H

    @classmethod
    def from_fields(cls, index: int, neff: float, e_field: np.ndarray, h_field: np.ndarray) -> "_ModeMetadata":
        E = np.asarray(e_field, dtype=np.complex128)
        H = np.asarray(h_field, dtype=np.complex128)
        if E.ndim >= 3:
            E = np.squeeze(E)
        if H.ndim >= 3:
            H = np.squeeze(H)
        return cls(index=index, neff=neff, E=E, H=H)

    def component(self, name: str) -> np.ndarray:
        comp = name.lower()
        if comp == "ex":
            return self.E[0]
        if comp == "ey":
            return self.E[1]
        if comp == "ez":
            return self.E[2]
        if comp == "hx":
            return self.H[0]
        if comp == "hy":
            return self.H[1]
        if comp == "hz":
            return self.H[2]
        raise KeyError(name)


def _extract_mode_component_1d(meta: _ModeMetadata, component: str) -> np.ndarray:
    comp = component.lower()
    if comp == "etot":
        return np.sqrt(np.sum(np.abs(meta.E) ** 2, axis=0))
    if comp == "htot":
        return np.sqrt(np.sum(np.abs(meta.H) ** 2, axis=0))
    try:
        return meta.component(component)
    except KeyError as exc:
        raise ValueError(
            f"Unknown component '{component}'. Use Ex, Ey, Ez, Hx, Hy, Hz, Etot, or Htot"
        ) from exc


def _extract_mode_component_2d(meta: _ModeMetadata, component: str) -> np.ndarray:
    comp = component.lower()
    if comp == "etot":
        return np.sqrt(np.sum(np.abs(meta.E) ** 2, axis=0))
    if comp == "htot":
        return np.sqrt(np.sum(np.abs(meta.H) ** 2, axis=0))
    try:
        return meta.component(component)
    except KeyError as exc:
        raise ValueError(
            f"Unknown component '{component}'. Use Ex, Ey, Ez, Hx, Hy, Hz, Etot, or Htot"
        ) from exc

    def _show_2d(self, modes, metadata, component="Etot", figsize=None):
        num_modes = len(modes)
        cols = min(2, num_modes)
        rows = int(np.ceil(num_modes / cols))
        fig, axes = plt.subplots(rows, cols, figsize=figsize or (5 * cols, 4 * rows), constrained_layout=True)
        axes = np.array(axes).reshape(rows, cols)

        for mode, meta, ax in zip(modes, metadata, axes.ravel()):
            field = np.real(_extract_mode_component_2d(meta, component))
            eps = mode["eps"]
            y_edges = mode["y_edges"] / µm
            z_edges = mode["z_edges"] / µm

            extent = (y_edges[0], y_edges[-1], z_edges[0], z_edges[-1])
            ax.imshow(eps.T, origin="lower", extent=extent, cmap="Greys", alpha=0.3, aspect="equal")
            vmax = np.max(np.abs(field)) or 1.0
            im = ax.imshow(field.T / vmax, origin="lower", extent=extent, cmap="RdBu", aspect="equal", vmin=-1, vmax=1)
            ax.set_title(f"Mode {mode['index']} ({component}, neff={mode['neff']:.3f})")
            ax.set_xlabel("y (µm)")
            ax.set_ylabel("z (µm)")
            fig.colorbar(im, ax=ax, shrink=0.8, label="Re(field) (norm)")

        plt.show()

# Deprecated placeholder to maintain backward compatibility
class GaussianSource:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("GaussianSource is deprecated in this module.")