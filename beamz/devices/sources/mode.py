import matplotlib.pyplot as plt
import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.devices.core import Device
from beamz.devices.sources.solve import solve_modes

class Box:
    """Axis-aligned plane descriptor used to place the mode source."""

    def __init__(self, center, size):
        coords = tuple(float(c) for c in center)
        spans = tuple(float(s) for s in size)
        self.center = coords
        self.size = spans


def _direction_to_axis(direction):
    """Return axis index for propagation direction string."""
    if "x" in direction: return 0
    if "y" in direction: return 1
    if "z" in direction: return 2
    raise ValueError(f"Unknown propagation direction '{direction}'")


def _direction_sign(direction):
    """Return '+' or '-' depending on propagation sign."""
    text = str(direction)
    symbol = "+"
    if text.startswith("-"): symbol = "-"
    return symbol


def _normal_vector(axis, direction_sign):
    """Return unit normal vector pointing along the propagation direction."""
    vec = np.zeros(3, dtype=np.float64)
    vec[axis] = 1.0 if direction_sign == "+" else -1.0
    return vec


def _surface_currents(electric, magnetic, axis, direction_sign):
    """Return electric and magnetic sheet currents for a plane with normal along the chosen axis."""
    normal = _normal_vector(axis, direction_sign)
    reshape = (3,) + (1,) * (electric.ndim - 1)
    normal_field = normal.reshape(reshape)
    current_electric = np.cross(normal_field, magnetic, axisa=0, axisb=0, axisc=0)
    current_magnetic = -np.cross(normal_field, electric, axisa=0, axisb=0, axisc=0)
    return current_electric, current_magnetic


def _poynting_density(electric, magnetic, axis, direction_sign):
    """Return power density along the propagation axis."""
    cross = np.cross(electric, np.conjugate(magnetic), axisa=0, axisb=0, axisc=0)
    density = 0.5 * np.real(cross[axis])
    sign = 1.0 if direction_sign == "+" else -1.0
    return density * sign


def _ensure_box(plane, dims, default_center, default_size):
    """Return a Box instance built from various input forms."""
    if plane is None: return Box(tuple(default_center), tuple(default_size))
    if isinstance(plane, Box): return plane
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
    return Box(center, size)


def _coerce_center(center, dims, default_center):
    """Coerce a user-provided center to the expected dimensionality."""
    if center is None: return tuple(default_center)
    if isinstance(center, (list, tuple)):
        if len(center) != dims: raise ValueError(f"center must have length {dims}, got {len(center)}")
        return tuple(float(c) for c in center)
    raise TypeError("center must be a tuple/list of coordinates")


def _coerce_size(width, dims, default_size, axis):
    """Coerce user-provided width into a tuple matching the plane dimensionality."""
    if width is None: return tuple(default_size)
    if isinstance(width, (int, float)):
        size = list(default_size)
        for idx in range(dims): size[idx] = 0.0 if idx == axis else float(width)
        return tuple(size)
    if isinstance(width, (list, tuple)):
        if len(width) != dims: raise ValueError(f"width must have length {dims}, got {len(width)}")
        return tuple(float(w) for w in width)
    raise TypeError("width must be a float or a tuple/list of floats")


class ModeSource(Device):
    """Unidirectional Huygens-mode source built on a TFSF plane."""

    def __init__(self, grid, center=None, width=None, wavelength=1.55e-6, direction="+x", mode=0, target_neff=None, pol="tm", signal=None):
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
        self.polarization = pol

        if signal is None:
            raw_signal = np.array([1.0], dtype=float)
        else:
            raw_signal = np.asarray(signal, dtype=float)
        if raw_signal.ndim != 1:
            raise ValueError("signal must be a one-dimensional array of samples")
        self.signal = raw_signal.astype(float, copy=False)
        self._repeat_signal = self.signal.size == 1
        self._analytic_signal = self._build_analytic_signal(self.signal)
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
        self.mode_profiles = []
        self.max_field_amplitude = None
        self.max_power_density = None
        self.total_power = None

        self._mode_cache = None
        self._electric_currents = {}
        self._magnetic_currents = {}
        self._grid_indices = {}
        self._coordinate_vectors = {}
        self._design = getattr(grid, "design", None)

        self._update_line_endpoints()

    # Compatibility helpers -------------------------------------------------
    def compute_modes_on_fdtd_grid(self, *args, **kwargs):
        del args, kwargs
        return self.compute_modes(force=True)

    def initialize_for_fdtd(self, *args, **kwargs):
        del args, kwargs
        self.compute_modes(force=True)

    def _select_fundamental_mode(self, e_fields, neff):
        """Select fundamental mode by analyzing spatial profiles (fewest peaks)."""
        n_modes = e_fields.shape[0]
        mode_scores = []
        for i in range(n_modes):
            Ez = np.squeeze(e_fields[i])[2] if e_fields[i].shape[0] > 2 else np.squeeze(e_fields[i])[0]
            Ez_abs = np.abs(Ez)
            threshold = 0.3 * np.max(Ez_abs)
            Ez_above = Ez_abs > threshold
            boundaries = np.diff(np.concatenate([[False], Ez_above, [False]])) != 0
            n_peaks = np.sum(boundaries) // 2
            central_half = len(Ez) // 4
            max_idx = np.argmax(Ez_abs)
            central_region = slice(max(0, max_idx - central_half), min(len(Ez), max_idx + central_half))
            central_concentration = np.sum(Ez_abs[central_region]) / np.sum(Ez_abs)
            if not np.isfinite(np.real(neff[i])) or np.real(neff[i]) < 1.0 or np.imag(neff[i]) > 0.01:
                continue
            mode_scores.append((n_peaks, 1.0 - central_concentration, -float(np.real(neff[i])), i))
        if not mode_scores:
            return min(self.mode_index, n_modes - 1)
        mode_scores.sort()
        selected_idx = mode_scores[0][3]
        print(f"[ModeSource] Selected mode {selected_idx}: n_peaks={mode_scores[0][0]}, concentration={1.0-mode_scores[0][1]:.3f}, neff={-mode_scores[0][2]:.4f}")
        return selected_idx

    # Geometry ---------------------------------------------------------------
    @property
    def design(self):
        return self._design

    @design.setter
    def design(self, value):
        self._design = value

    def _update_line_endpoints(self):
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
    def compute_modes(self, force=False):
        if self._mode_cache is not None and not force:
            return self._mode_cache

        permittivity = np.asarray(self.grid.permittivity, dtype=float)
        if permittivity.ndim == 2:
            modes = self._compute_modes_2d(permittivity)
        else:
            raise NotImplementedError("3D mode source support is not implemented yet")

        self._mode_cache = modes
        return modes

    def _compute_modes_2d(self, permittivity):
        if not hasattr(self.grid, "dx") or not hasattr(self.grid, "dy"):
            raise TypeError("Grid must expose dx and dy properties for mode computation")

        ny, nx = permittivity.shape
        dx = float(getattr(self.grid, "dx"))
        dy = float(getattr(self.grid, "dy"))

        if self.axis not in (0,):
            raise NotImplementedError("2D mode source currently supports propagation along ±x")

        yee_coords = self._compute_yee_coordinates_2d(ny, nx, dx, dy)
        eps_profile, mode_coords, index_info = self._sample_permittivity_for_mode(permittivity, ny, nx, dx, dy, yee_coords)

        npml = max(0, min(20, eps_profile.size // 4))
        neff, e_fields, h_fields, _ = solve_modes(
            eps=eps_profile,
            omega=self.omega,
            dL=dy if self.axis == 0 else dx,
            npml=npml,
            m=max(5, self.mode_index + 3),
            direction=self.direction,
            filter_pol=self.polarization,
            return_fields=True,
            target_neff=self.target_neff,
        )

        if e_fields.size == 0 or h_fields.size == 0:
            raise RuntimeError("Mode solver did not return field profiles")

        mode_idx = self._select_fundamental_mode(e_fields, neff)
        E = np.squeeze(e_fields[mode_idx])
        H = np.squeeze(h_fields[mode_idx])
        E_cart, H_cart = self._reorder_components(E, H)
        E_cart, H_cart = self._phase_align_fields(E_cart, H_cart)
        E_cart, H_cart = self._enforce_propagation_direction(E_cart, H_cart)
        
        fields_at_yee = self._sample_mode_fields_at_yee_positions(E_cart, H_cart, mode_coords, yee_coords, dy)
        
        direction_sign = _direction_sign(self.direction)
        
        if self.axis != 0:
            raise NotImplementedError("Only ±x propagation handled for 2D mode source")
        
        E_ez = fields_at_yee["E"].get("Ez")
        H_ez = fields_at_yee["H"].get("Ez")
        E_hy = fields_at_yee["E"].get("Hy")
        
        if E_ez is None or H_ez is None:
            raise RuntimeError("Failed to sample mode fields at Ez Yee grid positions")
        
        if E_hy is None:
            raise RuntimeError("Failed to sample E_z mode field at Hy Yee grid positions")
        
        jz_profile = np.asarray(H_ez, dtype=np.complex128).flatten()
        my_profile = np.asarray(E_hy, dtype=np.complex128).flatten()
        
        self._electric_currents = {"Ez": jz_profile}
        self._magnetic_currents = {"Hy": my_profile}
        self._coordinate_vectors = {"primary": mode_coords}
        self._grid_indices = self._build_yee_grid_indices(yee_coords)
        
        alignment_ok = self._verify_yee_grid_alignment()
        if not alignment_ok:
            ez_entry = self._grid_indices.get("Ez")
            hy_entry = self._grid_indices.get("Hy")
            jz_size = self._electric_currents["Ez"].size if "Ez" in self._electric_currents else 0
            my_size = self._magnetic_currents["Hy"].size if "Hy" in self._magnetic_currents else 0
            ez_slice_size = (ez_entry[0].stop - ez_entry[0].start) if ez_entry and isinstance(ez_entry[0], slice) else 0
            hy_slice_size = (hy_entry[0].stop - hy_entry[0].start) if hy_entry and isinstance(hy_entry[0], slice) else 0
            print(f"[DEBUG] J_z size={jz_size}, Ez slice={ez_entry[0] if ez_entry else None} (size={ez_slice_size})")
            print(f"[DEBUG] M_y size={my_size}, Hy slice={hy_entry[0] if hy_entry else None} (size={hy_slice_size})")
            print(f"[DEBUG] E_ez size={E_ez.size if E_ez is not None else 0}, H_ez size={H_ez.size if H_ez is not None else 0}, E_hy size={E_hy.size if E_hy is not None else 0}")
            raise RuntimeError(f"Mode field arrays do not match Yee grid index shapes: J_z size={jz_size}, Ez slice size={ez_slice_size}, M_y size={my_size}, Hy slice size={hy_slice_size}")
        
        poynting = _poynting_density(E_cart, H_cart, self.axis, direction_sign)
        self.max_field_amplitude = float(np.max(np.abs(E_cart)))
        self.max_power_density = float(np.max(np.abs(poynting)))
        cell_length = dy if self.axis == 0 else dx
        self.total_power = float(np.sum(poynting) * cell_length)

        profile = self._build_mode_profile(mode_coords, index_info, E_cart, H_cart)
        self.mode_profiles = [profile]
        return [
            {
            "neff": neff[mode_idx],
            "coords": mode_coords,
            "E": E_cart,
            "H": H_cart,
            }
        ]

    def _compute_yee_coordinates_2d(self, ny, nx, dx, dy):
        """Compute Yee grid coordinates for each field component in 2D TEz mode.
        
        For 2D TEz with +x propagation:
        - Ez: at cell centers (i+1/2, j+1/2) → y coords at (j+1/2)*dy
        - Hy: at (i, j+1/2) → y coords at (j+1/2)*dy (same y as Ez, but shifted x)
        - Hx: at (i+1/2, j) → y coords at j*dy (not used for injection)
        
        Returns:
            dict with keys 'Ez', 'Hy', 'Hx' containing:
            - 'y_coords': array of y coordinates
            - 'y_start': starting y index
            - 'y_end': ending y index (exclusive)
            - 'x_coord': x coordinate for this component
            - 'x_idx': x grid index
        """
        y_min = max(0.0, self.center[1] - abs(self.plane_size[1]) / 2)
        y_max = min(self.grid.height, self.center[1] + abs(self.plane_size[1]) / 2)
        
        y_ez_start = int(np.clip(np.floor(y_min / dy), 0, ny - 1))
        y_ez_end = int(np.clip(np.ceil(y_max / dy), y_ez_start + 1, ny))
        y_ez_coords = (np.arange(y_ez_start, y_ez_end) + 0.5) * dy
        
        y_hy_start = y_ez_start
        y_hy_end = max(y_ez_start, y_ez_end - 1)
        if y_hy_end > y_hy_start:
            y_hy_coords = (np.arange(y_hy_start, y_hy_end) + 0.5) * dy
        else:
            y_hy_coords = np.array([])
        
        y_hx_start = y_ez_start
        y_hx_end = y_ez_end
        y_hx_coords = np.arange(y_hx_start, y_hx_end) * dy
        
        x_ez_idx = int(np.clip(np.round(self.center[0] / dx - 0.5), 0, nx - 1))
        x_ez_coord = (x_ez_idx + 0.5) * dx
        
        x_hy_idx = max(0, x_ez_idx - 1)
        x_hy_coord = x_hy_idx * dx
        
        return {
            "Ez": {
                "y_coords": y_ez_coords,
                "y_start": y_ez_start,
                "y_end": y_ez_end,
                "x_coord": x_ez_coord,
                "x_idx": x_ez_idx,
            },
            "Hy": {
                "y_coords": y_hy_coords,
                "y_start": y_hy_start,
                "y_end": y_hy_end,
                "x_coord": x_hy_coord,
                "x_idx": x_hy_idx,
            },
            "Hx": {
                "y_coords": y_hx_coords,
                "y_start": y_hx_start,
                "y_end": y_hx_end,
                "x_coord": x_ez_coord,
                "x_idx": x_ez_idx,
            },
        }

    def _sample_permittivity_for_mode(self, permittivity, ny, nx, dx, dy, yee_coords):
        """Sample permittivity at positions needed for mode solving.
        
        For 2D +x propagation, sample permittivity along y-axis at source x position.
        The mode solver needs permittivity at the transverse positions where we'll
        evaluate the mode fields. We use Ez coordinates for mode solving since
        both Ez and Hy will be sampled at their respective Yee positions afterward.
        
        Args:
            permittivity: 2D permittivity array
            ny, nx: grid dimensions
            dx, dy: grid spacing
            yee_coords: output from _compute_yee_coordinates_2d()
        
        Returns:
            eps_profile: 1D permittivity profile along y
            coords: y coordinates for mode solver (Ez positions)
            index_info: dict with y_start, y_end, x_index
        """
        ez_info = yee_coords["Ez"]
        y_start = ez_info["y_start"]
        y_end = ez_info["y_end"]
        x_idx = ez_info["x_idx"]
        
        eps_profile = permittivity[y_start:y_end, x_idx]
        coords = ez_info["y_coords"]
        
        index_info = {
            "y_start": y_start,
            "y_end": y_end,
            "x_index": int(x_idx),
        }
        return np.asarray(eps_profile, dtype=np.float64), coords, index_info

    def _sample_mode_fields_at_yee_positions(self, E_mode, H_mode, mode_coords, yee_coords, dy):
        """Map mode solver output to exact Yee grid positions for each field component.
        
        The mode solver returns fields at coordinates mode_coords (Ez y-positions).
        We need to sample these fields at the exact Yee grid positions for Ez and Hy.
        
        For 2D TEz +x propagation:
        - Ez^mode: already at Ez Yee positions (cell centers at (j+1/2)*dy)
        - Hy^mode: need to sample at Hy Yee positions (also at (j+1/2)*dy, but fewer points)
        
        Args:
            E_mode: Electric field from mode solver, shape (3, n_points) where n_points matches mode_coords
            H_mode: Magnetic field from mode solver, shape (3, n_points)
            mode_coords: y coordinates where mode was solved (Ez positions)
            yee_coords: output from _compute_yee_coordinates_2d()
            dy: grid spacing in y direction
        
        Returns:
            dict with keys 'Ez', 'Hy' containing field arrays sampled at Yee positions
        """
        ez_info = yee_coords["Ez"]
        hy_info = yee_coords["Hy"]
        
        ez_y_coords = ez_info["y_coords"]
        hy_y_coords = hy_info["y_coords"]
        
        E_sampled = {}
        H_sampled = {}
        
        if len(ez_y_coords) > 0:
            ez_indices = np.searchsorted(mode_coords, ez_y_coords, side="left")
            ez_indices = np.clip(ez_indices, 0, len(mode_coords) - 1)
            if len(mode_coords) == len(ez_y_coords) and np.allclose(mode_coords, ez_y_coords):
                E_sampled["Ez"] = E_mode[2]
                H_sampled["Ez"] = H_mode[1]
            else:
                E_sampled["Ez"] = E_mode[2, ez_indices]
                H_sampled["Ez"] = H_mode[1, ez_indices]
        
        if len(hy_y_coords) > 0:
            hy_start_idx = hy_info["y_start"] - ez_info["y_start"]
            hy_end_idx = hy_start_idx + len(hy_y_coords)
            if hy_start_idx >= 0 and hy_end_idx <= len(mode_coords):
                E_sampled["Hy"] = E_mode[2, hy_start_idx:hy_end_idx]
                H_sampled["Hy"] = H_mode[1, hy_start_idx:hy_end_idx]
            else:
                hy_indices = np.searchsorted(mode_coords, hy_y_coords, side="left")
                hy_indices = np.clip(hy_indices, 0, len(mode_coords) - 1)
                E_sampled["Hy"] = E_mode[2, hy_indices]
                H_sampled["Hy"] = H_mode[1, hy_indices]
        
        return {"E": E_sampled, "H": H_sampled}

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

    def _reorder_components(self, E, H):
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

    def _phase_align_fields(self, E, H):
        reference = E[2] if E.shape[0] > 2 else E[0]
        if reference.size == 0:
            return E, H
        pivot = int(np.abs(reference).argmax())
        angle = np.angle(reference[pivot])
        if not np.isfinite(angle):
            return E, H
        rotate = np.exp(-1j * angle)
        return E * rotate, H * rotate

    def _enforce_propagation_direction(self, E, H):
        cross = np.cross(E, np.conjugate(H), axisa=0, axisb=0, axisc=0)
        power = np.real(cross[self.axis])
        avg = np.mean(power)
        sign = 1.0 if self.direction.startswith("+") else -1.0
        if avg * sign < 0:
            H = -H
        return E, H

    def _build_yee_grid_indices(self, yee_coords):
        """Build grid indices for each field component accounting for Yee grid staggering.
        
        For 2D TEz +x propagation:
        - Ez: at (i+1/2, j+1/2) → indices (y_slice, x_idx) where x_idx is Ez column
        - Hy: at (i, j+1/2) → indices (y_slice, x_idx) where x_idx may differ due to staggering
        
        Args:
            yee_coords: output from _compute_yee_coordinates_2d()
        
        Returns:
            dict mapping field components to (y_slice, x_idx) tuples
        """
        ez_info = yee_coords["Ez"]
        hy_info = yee_coords["Hy"]
        
        indices = {}
        
        y_ez_start = ez_info["y_start"]
        y_ez_end = ez_info["y_end"]
        x_ez_idx = ez_info["x_idx"]
        indices["Ez"] = (slice(y_ez_start, y_ez_end), x_ez_idx)
        
        y_hy_start = hy_info["y_start"]
        y_hy_end = hy_info["y_end"]
        x_hy_idx = hy_info["x_idx"]
        if y_hy_end > y_hy_start:
            indices["Hy"] = (slice(y_hy_start, y_hy_end), x_hy_idx)
        
        return indices

    def _build_mode_profile(self, coords, indices, E, H):
        profile = []
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
    def apply(self, fdtd, time_step):
        if not self._electric_currents and not self._magnetic_currents: self.compute_modes(force=True)
        if not self._electric_currents and not self._magnetic_currents:
            raise RuntimeError("Mode fields not initialised")

        e_mod = self._get_electric_modulation(time_step)
        h_mod = self._get_magnetic_modulation(time_step)
        normal_cell = self._normal_cell_size(getattr(fdtd, "resolution", None))

        dt = float(getattr(fdtd, "dt"))

        ez_entry = self._grid_indices.get("Ez")
        if "Ez" in self._electric_currents and ez_entry is not None:
            y_slice, x_idx = ez_entry
            values_complex = np.asarray(self._electric_currents["Ez"], dtype=np.complex128) * e_mod
            values = np.real(values_complex) / normal_cell
            sigma = getattr(fdtd, "sigma", None)
            if sigma is not None:
                sigma_slice = np.asarray(sigma[y_slice, x_idx])
                values = np.where(np.abs(sigma_slice) > 0, 0.0, values)
            eps_slice = fdtd.epsilon_r[y_slice, x_idx]
            fdtd.Ez[y_slice, x_idx] += (dt / (EPS_0 * eps_slice)) * values

        hy_entry = self._grid_indices.get("Hy")
        if "Hy" in self._magnetic_currents and hy_entry is not None:
            y_slice, x_idx = hy_entry
            values_complex = np.asarray(self._prepare_h_component(self._magnetic_currents["Hy"]), dtype=np.complex128) * h_mod
            values = np.real(values_complex) / normal_cell
            sigma = getattr(fdtd, "sigma", None)
            if sigma is not None:
                sigma_slice = np.asarray(sigma[y_slice, x_idx])
                values = np.where(np.abs(sigma_slice) > 0, 0.0, values)
            fdtd.Hy[y_slice, x_idx] -= (dt / MU_0) * values

    def _prepare_h_component(self, values):
        if values.ndim == 1 and values.size > 1:
            return 0.5 * (values[:-1] + values[1:])
        return values

    # Visualisation -------------------------------------------------------------
    def show(self, component="Ez", figsize=None):
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

    def get_source_terms(self, fields, t, dt, current_step, resolution, design):
        """Return electric and magnetic sheet currents for soft Huygens injection.
        
        Returns source currents sampled at exact Yee grid positions:
        - J_z = H_y^mode (electric current for Ez update)
        - M_y = E_z^mode (magnetic current for Hy update)
        """
        if not hasattr(fields, "Ez"):
            return {}, {}
        if not self._electric_currents or not self._magnetic_currents:
            self.compute_modes()
        electric_scale = self._get_electric_modulation(current_step)
        magnetic_scale = self._get_magnetic_modulation(current_step)
        if abs(electric_scale) < 1e-15 and abs(magnetic_scale) < 1e-15:
            return {}, {}
        source_j = {}
        source_m = {}
        normal_cell = self._normal_cell_size(resolution)
        ez_entry = self._grid_indices.get("Ez")
        if ez_entry and "Ez" in self._electric_currents and abs(electric_scale) > 0.0:
            y_slice, x_idx = ez_entry
            sheet = np.asarray(self._electric_currents["Ez"], dtype=np.complex128)
            current = np.real(sheet * electric_scale) / normal_cell
            slice_size = y_slice.stop - y_slice.start if isinstance(y_slice, slice) else len(y_slice)
            if current.size != slice_size:
                current = current[:slice_size] if current.size > slice_size else np.pad(current, (0, slice_size - current.size))
            if current.size and np.max(np.abs(current)) > 0.0:
                contribution = -np.asarray(current, dtype=np.float64)
                source_j["Ez"] = (contribution, (y_slice, x_idx))
        hy_entry = self._grid_indices.get("Hy")
        if hy_entry and "Hy" in self._magnetic_currents and abs(magnetic_scale) > 0.0:
            y_slice, x_idx = hy_entry
            sheet = np.asarray(self._magnetic_currents["Hy"], dtype=np.complex128)
            slice_size = y_slice.stop - y_slice.start if isinstance(y_slice, slice) else len(y_slice)
            if sheet.size != slice_size:
                sheet = sheet[:slice_size] if sheet.size > slice_size else np.pad(sheet, (0, slice_size - sheet.size))
            current = np.real(sheet * magnetic_scale) / normal_cell
            if current.size and np.max(np.abs(current)) > 0.0:
                contribution = np.asarray(current, dtype=np.float64)
                source_m["Hy"] = (contribution, (y_slice, x_idx))
        return source_j, source_m

    def _get_electric_modulation(self, time_step):
        return self._sample_signal(time_step)

    def _get_magnetic_modulation(self, time_step):
        if time_step <= 0:
            return self._sample_signal(0)
        prev = self._sample_signal(time_step - 1)
        curr = self._sample_signal(time_step)
        return 0.5 * (prev + curr)

    def _sample_signal(self, idx):
        if idx < self._analytic_signal.size:
            return complex(self._analytic_signal[idx])
        if self._repeat_signal and self._analytic_signal.size:
            return complex(self._analytic_signal[-1])
        return 0.0 + 0.0j

    def _build_analytic_signal(self, samples):
        if samples.size == 0:
            return np.zeros(0, dtype=np.complex128)
        spectrum = np.fft.fft(samples)
        n = samples.size
        h = np.zeros(n, dtype=float)
        if n % 2 == 0:
            h[0] = h[n // 2] = 1.0
            h[1:n // 2] = 2.0
        else:
            h[0] = 1.0
            h[1:(n + 1) // 2] = 2.0
        analytic = np.fft.ifft(spectrum * h)
        return analytic.astype(np.complex128, copy=False)

    def _normal_cell_size(self, resolution):
        if self.axis == 0 and hasattr(self.grid, "dx"):
            return float(getattr(self.grid, "dx"))
        if self.axis == 1 and hasattr(self.grid, "dy"):
            return float(getattr(self.grid, "dy"))
        if self.axis == 2 and hasattr(self.grid, "dz"):
            return float(getattr(self.grid, "dz"))
        if resolution is not None:
            return float(resolution)
        raise ValueError("Unable to determine grid spacing along source normal")

    def _verify_yee_grid_alignment(self):
        """Verify that mode field arrays match Yee grid index shapes."""
        if not self._grid_indices or not self._electric_currents or not self._magnetic_currents:
            return True
        ez_entry = self._grid_indices.get("Ez")
        if ez_entry and "Ez" in self._electric_currents:
            y_slice, x_idx = ez_entry
            jz_size = self._electric_currents["Ez"].size
            slice_size = y_slice.stop - y_slice.start if isinstance(y_slice, slice) else len(y_slice)
            if jz_size != slice_size:
                return False
        hy_entry = self._grid_indices.get("Hy")
        if hy_entry and "Hy" in self._magnetic_currents:
            y_slice, x_idx = hy_entry
            my_size = self._magnetic_currents["Hy"].size
            slice_size = y_slice.stop - y_slice.start if isinstance(y_slice, slice) else len(y_slice)
            if my_size != slice_size:
                return False
        return True


__all__ = ["ModeSource"]
