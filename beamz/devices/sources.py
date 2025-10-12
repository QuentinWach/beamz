from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from beamz.const import LIGHT_SPEED, µm
from beamz.devices import mode as mode_solver

try:
    from beamz.simulation.meshing import RegularGrid, RegularGrid3D
except ImportError:  # pragma: no cover - during packaging
    RegularGrid = RegularGrid3D = object


class ModeSource:
    """Visualise eigenmodes from a rasterised permittivity grid.

    Parameters
    ----------
    grid : RegularGrid | RegularGrid3D
        Rasterised design produced via ``Design.rasterize``.
    start, end : tuple[float, float] | tuple[float, float, float]
        Points (in metres) defining a line across the modal cross-section.
        Only the dominant varying axis is used for slicing.
    wavelength : float
        Free-space wavelength in metres.
    num_modes : int, optional
        Number of modes to compute (default 3).
    polarization : str, optional
        Target polarization for 1D solver ("te" or "tm").
    direction : str, optional
        Propagation direction hint for the solver (default "+x").
    """

    def __init__(
        self,
        grid,
        start,
        end,
        wavelength: float,
        num_modes: int = 3,
        polarization: str | None = "te",
        direction: str = "+x",
    ) -> None:
        if getattr(grid, "permittivity", None) is None:
            raise ValueError("Grid must expose a 'permittivity' array. Did you call Design.rasterize()?")
        self.grid = grid
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        self.wavelength = float(wavelength)
        self.omega = 2 * np.pi * LIGHT_SPEED / self.wavelength
        self.num_modes = int(num_modes)
        self.polarization = polarization
        self.direction = direction

        if self.num_modes <= 0:
            raise ValueError("num_modes must be positive")

        if self.start.shape != self.end.shape:
            raise ValueError("start and end points must have the same dimensionality")

        self._mode_cache = None
        self._mode_type = None  # "1d" or "2d"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute_modes(self, force: bool = False):
        """Compute eigenmodes using the stored grid.

        Parameters
        ----------
        force : bool
            Recompute even if cached results exist.
        """
        if self._mode_cache is not None and not force:
            return self._mode_cache

        permittivity = np.asarray(self.grid.permittivity, dtype=float)
        if permittivity.ndim == 2:
            modes = self._compute_modes_1d(permittivity)
            self._mode_type = "1d"
        elif permittivity.ndim == 3:
            modes = self._compute_modes_2d(permittivity)
            self._mode_type = "2d"
        else:
            raise ValueError("Unsupported permittivity dimensionality")

        self._mode_cache = modes
        return modes

    def show(self, modes=None, figsize=None):
        """Display the computed modes using matplotlib."""
        modes = modes or self.compute_modes()
        if not modes:
            raise RuntimeError("No modes available to visualise")

        if self._mode_type == "1d":
            self._show_1d(modes, figsize=figsize)
        elif self._mode_type == "2d":
            self._show_2d(modes, figsize=figsize)
        else:
            raise RuntimeError("Unknown mode type")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_modes_1d(self, permittivity: np.ndarray):
        if not isinstance(self.grid, RegularGrid):
            raise TypeError("RegularGrid expected for 1D mode computation")

        ny, nx = permittivity.shape
        dx = getattr(self.grid, "dx", 1.0)
        dy = getattr(self.grid, "dy", 1.0)

        x_col = float(self.start[0])
        col_idx = int(np.clip(np.round(x_col / dx), 0, nx - 1))
        eps_profile = permittivity[:, col_idx]

        neff, e_fields, h_fields, _ = mode_solver.solve_modes(
            eps=eps_profile,
            omega=self.omega,
            dL=dy,
            m=self.num_modes,
            direction=self.direction,
            filter_pol=self.polarization,
            return_fields=True,
        )

        y_coords = (np.arange(ny) + 0.5) * dy
        modes = []
        max_modes = min(self.num_modes, len(neff))
        for idx in range(max_modes):
            Ez = np.squeeze(e_fields[idx][2])
            Ez = Ez if Ez.ndim == 1 else Ez[:, 0]
            modes.append({
                "index": idx,
                "neff": float(np.real(neff[idx])),
                "Ez": Ez,
                "y": y_coords,
                "eps": eps_profile,
            })
        return modes

    def _compute_modes_2d(self, permittivity: np.ndarray):
        if not isinstance(self.grid, RegularGrid3D):
            raise TypeError("RegularGrid3D expected for 2D mode computation")

        nz, ny, nx = permittivity.shape  # z, y, x order
        dx = getattr(self.grid, "dx", 1.0)
        dy = getattr(self.grid, "dy", 1.0)
        dz = getattr(self.grid, "dz", 1.0)

        x_col = float(self.start[0])
        col_idx = int(np.clip(np.round(x_col / dx), 0, nx - 1))

        eps_slice = permittivity[:, :, col_idx]  # (nz, ny)
        # reorder to (ny, nz) for tidy3d wrapper
        eps_slice = np.transpose(eps_slice, (1, 0))

        y_edges = np.linspace(0.0, ny * dy, ny + 1)
        z_edges = np.linspace(0.0, nz * dz, nz + 1)

        tidy_modes = mode_solver.tidy3d_mode_computation_wrapper(
            frequency=LIGHT_SPEED / self.wavelength,
            permittivity_cross_section=eps_slice,
            coords=[y_edges / µm, z_edges / µm],
            direction="+",
            num_modes=self.num_modes,
            precision="double",
        )

        modes = []
        for idx, mode in enumerate(tidy_modes[: self.num_modes]):
            Ez = np.array(mode.Ez)
            modes.append({
                "index": idx,
                "neff": float(np.real(mode.neff)),
                "Ez": Ez,
                "Ey": np.array(mode.Ey),
                "Ex": np.array(mode.Ex),
                "Hy": np.array(mode.Hy),
                "Hx": np.array(mode.Hx),
                "eps": eps_slice,
                "y_edges": y_edges,
                "z_edges": z_edges,
            })
        return modes

    # ------------------------------------------------------------------
    # Plotting utilities
    # ------------------------------------------------------------------
    def _show_1d(self, modes, figsize=None):
        eps_profile = modes[0]["eps"]
        y_coords = modes[0]["y"] / µm

        fig, ax1 = plt.subplots(figsize=figsize or (7, 4))
        ax1.plot(y_coords, eps_profile, color="black", label="εr")
        ax1.set_xlabel("y (µm)")
        ax1.set_ylabel("εr", color="black")
        ax1.tick_params(axis="y", labelcolor="black")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        for mode in modes:
            Ez = mode["Ez"]
            intensity = np.abs(Ez) ** 2
            intensity /= np.max(intensity) + 1e-18
            ax2.plot(y_coords, intensity, label=f"Mode {mode['index']} (neff={mode['neff']:.3f})")

        ax2.set_ylabel("|Ez|² (norm)")
        ax2.legend(loc="upper right")
        fig.tight_layout()
        plt.show()

    def _show_2d(self, modes, figsize=None):
        num_modes = len(modes)
        cols = min(2, num_modes)
        rows = int(np.ceil(num_modes / cols))
        fig, axes = plt.subplots(rows, cols, figsize=figsize or (5 * cols, 4 * rows), constrained_layout=True)
        axes = np.array(axes).reshape(rows, cols)

        for mode, ax in zip(modes, axes.ravel()):
            Ez = np.real(mode["Ez"])
            eps = mode["eps"]
            y_edges = mode["y_edges"] / µm
            z_edges = mode["z_edges"] / µm

            extent = (y_edges[0], y_edges[-1], z_edges[0], z_edges[-1])
            ax.imshow(eps.T, origin="lower", extent=extent, cmap="Greys", alpha=0.3, aspect="equal")
            vmax = np.max(np.abs(Ez)) or 1.0
            im = ax.imshow(Ez.T / vmax, origin="lower", extent=extent, cmap="RdBu", aspect="equal", vmin=-1, vmax=1)
            ax.set_title(f"Mode {mode['index']} (Re(Ez), neff={mode['neff']:.3f})")
            ax.set_xlabel("y (µm)")
            ax.set_ylabel("z (µm)")
            fig.colorbar(im, ax=ax, shrink=0.8, label="Re(Ez) (norm)")

        plt.show()

# Deprecated placeholder to maintain backward compatibility
class GaussianSource:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("GaussianSource is deprecated in this module.")