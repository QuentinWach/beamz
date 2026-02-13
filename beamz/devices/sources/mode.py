import logging

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED, MU_0, µm
from beamz.devices.core import Device
from beamz.devices.sources._injection import build_3d_profiles, inject_3d_fields
from beamz.devices.sources.solve import solve_modes

logger = logging.getLogger(__name__)


class ModeSource(Device):
    """Huygens mode source on Yee grid supporting ±x/±y propagation.

    In 3D, injects all 6 field components (Ex, Ey, Ez, Hx, Hy, Hz) for accurate
    mode injection, accounting for proper Yee grid staggering.
    """

    def __init__(
        self, grid, center, width, wavelength, pol, signal, direction="+x", height=None
    ):
        self.grid = grid
        self.center = (
            center if isinstance(center, (tuple, list)) else (center, grid.height / 2)
        )
        self.width = width
        self.height = height
        self.wavelength = wavelength
        self.pol = pol
        self.signal = signal
        self.direction = direction

        # Storage for all 6 field component profiles (for 3D injection)
        self._Ex_profile = None
        self._Ey_profile = None
        self._Ez_profile = None
        self._Hx_profile = None
        self._Hy_profile = None
        self._Hz_profile = None

        # Indices for each component's injection position
        self._Ex_indices = None
        self._Ey_indices = None
        self._Ez_indices = None
        self._Hx_indices = None
        self._Hy_indices = None
        self._Hz_indices = None

        # Legacy attributes for compatibility and 2D
        self._jz_profile = None
        self._my_profile = None
        self._mz_profile = None
        self._jy_profile = None
        self._jx_profile = None
        self._ez_indices = None
        self._h_indices = None
        self._hz_indices = None
        self._e_indices = None

        self._h_component = None
        self._e_component = None
        self._neff = None
        self._dt_physical = 0.0

    def initialize(self, permittivity, resolution):
        """Compute the mode and set up the source currents for all 6 components in 3D."""
        dx = dy = resolution
        is_3d = permittivity.ndim == 3
        self._resolution = resolution
        self._is_3d = is_3d

        if is_3d:
            nz, ny, nx = permittivity.shape
            dz = resolution
            self._grid_shape = (nz, ny, nx)
            if self.height is None:
                self.height = self.width
        else:
            ny, nx = permittivity.shape
            nz = 1
            self._grid_shape = (ny, nx)
            self.height = None

        axis = "x" if self.direction in ("+x", "-x") else "y"
        self._dt_physical = 0.0

        # 1. Get center index for injection plane
        if axis == "x":
            center_idx = int(np.clip(np.round(self.center[0] / dx - 0.5), 0, nx - 1))
            if self.direction == "+x":
                offset_idx = max(0, center_idx - 1)
            else:
                offset_idx = min(nx - 2, center_idx)

            if is_3d:
                eps_profile = permittivity[:, :, center_idx]
                self._eps_profile_2d = eps_profile
            else:
                eps_profile = permittivity[:, center_idx]
                self._eps_profile_2d = None

        else:  # axis == "y"
            center_idx = int(np.clip(np.round(self.center[1] / dy - 0.5), 0, ny - 1))
            if self.direction == "+y":
                offset_idx = max(0, center_idx - 1)
            else:
                offset_idx = min(ny - 2, center_idx)

            if is_3d:
                eps_profile = permittivity[:, center_idx, :]
                self._eps_profile_2d = eps_profile
            else:
                eps_profile = permittivity[center_idx, :]
                self._eps_profile_2d = None

        # 2. Solve for mode fields
        omega = 2 * np.pi * LIGHT_SPEED / self.wavelength
        dL = dz if is_3d else (dy if axis == "x" else dx)
        neff_val, e_fields, h_fields, _ = solve_modes(
            eps=eps_profile,
            omega=omega,
            dL=dL,
            m=1,
            direction=self.direction,
            filter_pol=self.pol,
            return_fields=True,
        )
        self._neff = neff_val[0]
        E_mode = e_fields[0]
        H_mode = h_fields[0]

        # 3. Extract all 6 components and convert to JAX arrays
        Ex_raw = jnp.asarray(jnp.squeeze(E_mode[0]))
        Ey_raw = jnp.asarray(jnp.squeeze(E_mode[1]))
        Ez_raw = jnp.asarray(jnp.squeeze(E_mode[2]))
        Hx_raw = jnp.asarray(jnp.squeeze(H_mode[0]))
        Hy_raw = jnp.asarray(jnp.squeeze(H_mode[1]))
        Hz_raw = jnp.asarray(jnp.squeeze(H_mode[2]))

        # 4. Phase align all components to dominant field (JAX-compatible)
        if self.pol == "tm":
            ref_field = jnp.where(
                jnp.max(jnp.abs(Ez_raw)) > jnp.max(jnp.abs(Ey_raw)), Ez_raw, Ey_raw
            )
        else:
            ref_field = Ey_raw if axis == "x" else Ex_raw
            ref_field = jnp.where(jnp.max(jnp.abs(ref_field)) < 1e-9, Ez_raw, ref_field)

        idx_max = jnp.argmax(jnp.abs(ref_field))
        phase_ref = jnp.angle(ref_field.flatten()[idx_max])

        Ex_aligned = Ex_raw * jnp.exp(-1j * phase_ref)
        Ey_aligned = Ey_raw * jnp.exp(-1j * phase_ref)
        Ez_aligned = Ez_raw * jnp.exp(-1j * phase_ref)
        Hx_aligned = Hx_raw * jnp.exp(-1j * phase_ref)
        Hy_aligned = Hy_raw * jnp.exp(-1j * phase_ref)
        Hz_aligned = Hz_raw * jnp.exp(-1j * phase_ref)

        # 5. Apply Yee grid staggering and set up indices
        if is_3d:
            self._setup_3d_injection(
                Ex_aligned, Ey_aligned, Ez_aligned,
                Hx_aligned, Hy_aligned, Hz_aligned,
                center_idx, offset_idx, axis,
                nz, ny, nx, resolution,
            )
        else:
            self._setup_2d_injection(
                E_mode, H_mode, center_idx, offset_idx, axis, ny, nx, resolution
            )

        self._compute_dt_physical(axis, is_3d, dx, dy)

    def _setup_3d_injection(
        self, Ex, Ey, Ez, Hx, Hy, Hz,
        center_idx, offset_idx, axis,
        nz, ny, nx, resolution,
    ):
        """Set up full 6-component injection for 3D simulations."""
        profiles, indices, extra = build_3d_profiles(
            Ex, Ey, Ez, Hx, Hy, Hz,
            axis=axis,
            direction=self.direction,
            center=self.center,
            width=self.width,
            height=self.height,
            center_idx=center_idx,
            offset_idx=offset_idx,
            grid_shape=(nz, ny, nx),
            resolution=resolution,
            neff=self._neff,
        )

        # Store profiles on self
        self._Ex_profile = profiles.get("Ex")
        self._Ey_profile = profiles.get("Ey")
        self._Ez_profile = profiles.get("Ez")
        self._Hx_profile = profiles.get("Hx")
        self._Hy_profile = profiles.get("Hy")
        self._Hz_profile = profiles.get("Hz")

        # Store indices on self
        self._Ex_indices = indices.get("Ex")
        self._Ey_indices = indices.get("Ey")
        self._Ez_indices = indices.get("Ez")
        self._Hx_indices = indices.get("Hx")
        self._Hy_indices = indices.get("Hy")
        self._Hz_indices = indices.get("Hz")

        # Store extra metadata
        for key, val in extra.items():
            setattr(self, key, val)

        # Legacy compatibility
        self._jz_profile = self._Hz_profile
        self._my_profile = self._Ez_profile

    def _setup_2d_injection(
        self, E_mode, H_mode, center_idx, offset_idx, axis, ny, nx, resolution
    ):
        """Legacy 2D injection setup using original index-based extraction.

        The mode solver returns fields in a specific order based on propagation axis.
        For 2D (1D eps profile), the output uses propagation_axis=0, giving:
        E_mode = [Ez, Ex, Ey], H_mode = [Hz, Hx, Hy] in tidy3d convention.

        We use index-based extraction with fallback to handle different mode types.
        """
        dir_sign = 1.0 if self.direction.startswith("+") else -1.0
        ETA_0 = np.sqrt(MU_0 / EPS_0)
        Z_phys = ETA_0 / max(np.real(self._neff), 1e-6)

        if axis == "x":
            self._setup_2d_x(
                E_mode, H_mode, center_idx, offset_idx, ny, nx, resolution,
                dir_sign, Z_phys,
            )
        else:
            self._setup_2d_y(
                E_mode, H_mode, center_idx, offset_idx, ny, nx, resolution,
                dir_sign, Z_phys,
            )

    def _setup_2d_x(
        self, E_mode, H_mode, center_idx, offset_idx, ny, nx, resolution,
        dir_sign, Z_phys,
    ):
        """2D injection setup for x-propagation."""
        center_y_idx = int(round(self.center[1] / resolution))
        half_width_idx = int(round((self.width / 2) / resolution))
        y_start = max(0, center_y_idx - half_width_idx)
        y_end = min(ny, center_y_idx + half_width_idx)
        y_slice = slice(y_start, y_end)
        self._y_start = y_start
        self._y_end = y_end

        if self.pol == "tm":
            self._ez_indices = (y_slice, center_idx)
            self._h_indices = (y_slice, offset_idx)
            self._h_component = "Hx"

            Hy_raw = np.squeeze(H_mode[1])
            Ez_raw = np.squeeze(E_mode[2])
            if np.max(np.abs(Hy_raw)) < 1e-9:
                Hy_raw = np.squeeze(H_mode[2])
            if np.max(np.abs(Ez_raw)) < 1e-9:
                Ez_raw = np.squeeze(E_mode[1])

            idx_max = np.argmax(np.abs(Hy_raw))
            phase_ref = np.angle(Hy_raw.flatten()[idx_max])
            Hy_profile = Hy_raw * np.exp(-1j * phase_ref)
            Ez_profile = Ez_raw * np.exp(-1j * phase_ref)

            norm_h, norm_e = np.max(np.abs(Hy_profile)), np.max(np.abs(Ez_profile))
            if norm_h > 1e-12 and norm_e > 1e-12:
                Ez_profile = Ez_profile * (Z_phys / (norm_e / norm_h))

            width_cells = y_end - y_start
            window = self._make_1d_window(width_cells)

            Hy_cropped = np.real(Hy_profile)[y_start:y_end]
            Ez_cropped = np.real(Ez_profile)[y_start:y_end]
            if len(Hy_cropped) == len(window):
                Hy_cropped = Hy_cropped * window
                Ez_cropped = Ez_cropped * window

            self._jz_profile = dir_sign * Hy_cropped
            self._my_profile = dir_sign * Ez_cropped

        else:  # TE
            hz_col = (
                max(0, offset_idx - 1) if self.direction == "+x"
                else min(nx - 2, offset_idx)
            )

            self._hz_indices = (slice(y_start, min(y_end, ny - 1)), hz_col)
            self._e_indices = (slice(y_start, min(y_end, ny - 1)), offset_idx)
            self._e_component = "Ey"

            h_candidates = [np.squeeze(H_mode[i]) for i in range(3)]
            e_candidates = [np.squeeze(E_mode[i]) for i in range(3)]
            h_scores = [float(np.max(np.abs(hc))) for hc in h_candidates]
            e_scores = [float(np.max(np.abs(ec))) for ec in e_candidates]
            Hz_raw = h_candidates[int(np.argmax(h_scores))]
            Ey_raw = e_candidates[int(np.argmax(e_scores))]

            Hz_staggered = 0.5 * (Hz_raw[:-1] + Hz_raw[1:])
            Ey_staggered = 0.5 * (Ey_raw[:-1] + Ey_raw[1:])

            idx_max = np.argmax(np.abs(Hz_staggered))
            phase_ref = np.angle(Hz_staggered.flatten()[idx_max])
            Hz_profile = Hz_staggered * np.exp(-1j * phase_ref)
            Ey_profile = Ey_staggered * np.exp(-1j * phase_ref)

            norm_h, norm_e = np.max(np.abs(Hz_profile)), np.max(np.abs(Ey_profile))
            if norm_h > 1e-12 and norm_e > 1e-12:
                Ey_profile = Ey_profile * (Z_phys / (norm_e / norm_h))

            width_cells = min(y_end, len(Hz_profile)) - y_start
            window = self._make_1d_window(width_cells)

            Hz_cropped = np.real(Hz_profile)[y_start : min(y_end, len(Hz_profile))]
            Ey_cropped = np.real(Ey_profile)[y_start : min(y_end, len(Ey_profile))]
            if len(Hz_cropped) == len(window):
                Hz_cropped = Hz_cropped * window
                Ey_cropped = Ey_cropped * window

            if self.direction == "+x":
                self._jy_profile = Hz_cropped
                self._mz_profile = Ey_cropped
            else:
                self._jy_profile = -Hz_cropped
                self._mz_profile = -Ey_cropped

    def _setup_2d_y(
        self, E_mode, H_mode, center_idx, offset_idx, ny, nx, resolution,
        dir_sign, Z_phys,
    ):
        """2D injection setup for y-propagation."""
        center_x_idx = int(round(self.center[0] / resolution))
        half_width_idx = int(round((self.width / 2) / resolution))
        x_start = max(0, center_x_idx - half_width_idx)
        x_end = min(nx, center_x_idx + half_width_idx)
        x_slice = slice(x_start, x_end)
        self._x_start = x_start
        self._x_end = x_end

        if self.pol == "tm":
            self._ez_indices = (center_idx, x_slice)
            self._h_indices = (offset_idx, x_slice)
            self._h_component = "Hy"

            Hx_raw = np.squeeze(H_mode[1])
            Ez_raw = np.squeeze(E_mode[2])
            if np.max(np.abs(Hx_raw)) < 1e-9:
                Hx_raw = np.squeeze(H_mode[2])
            if np.max(np.abs(Ez_raw)) < 1e-9:
                Ez_raw = np.squeeze(E_mode[1])

            idx_max = np.argmax(np.abs(Hx_raw))
            phase_ref = np.angle(Hx_raw.flatten()[idx_max])
            Hx_profile = Hx_raw * np.exp(-1j * phase_ref)
            Ez_profile = Ez_raw * np.exp(-1j * phase_ref)

            norm_h, norm_e = np.max(np.abs(Hx_profile)), np.max(np.abs(Ez_profile))
            if norm_h > 1e-12 and norm_e > 1e-12:
                Ez_profile = Ez_profile * (Z_phys / (norm_e / norm_h))

            width_cells = x_end - x_start
            window = self._make_1d_window(width_cells)

            Hx_cropped = np.real(Hx_profile)[x_start:x_end]
            Ez_cropped = np.real(Ez_profile)[x_start:x_end]
            if len(Hx_cropped) == len(window):
                Hx_cropped = Hx_cropped * window
                Ez_cropped = Ez_cropped * window

            if self.direction == "+y":
                self._jz_profile = -Hx_cropped
                self._my_profile = Ez_cropped
            else:
                self._jz_profile = Hx_cropped
                self._my_profile = -Ez_cropped

        else:  # TE y-prop
            hz_row = (
                max(0, offset_idx - 1) if self.direction == "+y"
                else min(ny - 2, offset_idx)
            )

            self._hz_indices = (hz_row, slice(x_start, min(x_end, nx - 1)))
            self._e_indices = (offset_idx, slice(x_start, min(x_end, nx - 1)))
            self._e_component = "Ex"

            h_candidates = [np.squeeze(H_mode[i]) for i in range(3)]
            e_candidates = [np.squeeze(E_mode[i]) for i in range(3)]
            h_scores = [float(np.max(np.abs(hc))) for hc in h_candidates]
            e_scores = [float(np.max(np.abs(ec))) for ec in e_candidates]
            Hz_raw = h_candidates[int(np.argmax(h_scores))]
            Ex_raw = e_candidates[int(np.argmax(e_scores))]

            Hz_staggered = 0.5 * (Hz_raw[:-1] + Hz_raw[1:])
            Ex_staggered = 0.5 * (Ex_raw[:-1] + Ex_raw[1:])

            idx_max = np.argmax(np.abs(Hz_staggered))
            phase_ref = np.angle(Hz_staggered.flatten()[idx_max])
            Hz_profile = Hz_staggered * np.exp(-1j * phase_ref)
            Ex_profile = Ex_staggered * np.exp(-1j * phase_ref)

            norm_h, norm_e = np.max(np.abs(Hz_profile)), np.max(np.abs(Ex_profile))
            if norm_h > 1e-12 and norm_e > 1e-12:
                Ex_profile = Ex_profile * (Z_phys / (norm_e / norm_h))

            width_cells = min(x_end, len(Hz_profile)) - x_start
            window = self._make_1d_window(width_cells)

            Hz_cropped = np.real(Hz_profile)[x_start : min(x_end, len(Hz_profile))]
            Ex_cropped = np.real(Ex_profile)[x_start : min(x_end, len(Ex_profile))]
            if len(Hz_cropped) == len(window):
                Hz_cropped = Hz_cropped * window
                Ex_cropped = Ex_cropped * window

            self._jx_profile = dir_sign * Hz_cropped
            self._mz_profile = dir_sign * Ex_cropped

    @staticmethod
    def _make_1d_window(width_cells, alpha=0.3):
        """Create a 1D Tukey window for smooth edges."""
        if width_cells > 2:
            from scipy.signal.windows import tukey
            return tukey(width_cells, alpha=alpha)
        return np.ones(max(1, width_cells))

    def _compute_dt_physical(self, axis, is_3d, dx, dy):
        """Compute physical time shift between E and H injection planes."""
        if self._neff is None:
            return

        coord_e = 0.0
        coord_h = 0.0

        if is_3d:
            if axis == "x":
                if self._Ez_indices is not None:
                    coord_e = (self._Ez_indices[2] + 0.5) * dx
                if self._Hy_indices is not None:
                    coord_h = (self._Hy_indices[2] + 1.0) * dx
            else:
                if self._Ez_indices is not None:
                    coord_e = (self._Ez_indices[1] + 0.5) * dy
                if self._Hx_indices is not None:
                    coord_h = (self._Hx_indices[1] + 1.0) * dy
        else:
            if axis == "x":
                if self.pol == "tm":
                    idx_e = self._ez_indices[1] if self._ez_indices else 0
                    idx_h = self._h_indices[1] if self._h_indices else 0
                else:
                    idx_e = self._e_indices[1] if self._e_indices else 0
                    idx_h = self._hz_indices[1] if self._hz_indices else 0
                coord_e = (idx_e + 0.5) * dx
                coord_h = (idx_h + 1.0) * dx
            else:
                if self.pol == "tm":
                    idx_e = self._ez_indices[0] if self._ez_indices else 0
                    idx_h = self._h_indices[0] if self._h_indices else 0
                else:
                    idx_e = self._e_indices[0] if self._e_indices else 0
                    idx_h = self._hz_indices[0] if self._hz_indices else 0
                coord_e = (idx_e + 0.5) * dy
                coord_h = (idx_h + 1.0) * dy

        self._dt_physical = (
            (coord_e - coord_h) * float(np.real(self._neff)) / LIGHT_SPEED
        )

    def _get_signal_value(self, time, dt):
        """Interpolate signal value at arbitrary time."""
        idx_float = float(time / dt)
        idx_low = int(np.floor(idx_float))
        idx_high = idx_low + 1
        frac = idx_float - idx_low

        if 0 <= idx_low < len(self.signal) - 1:
            return (1.0 - frac) * self.signal[idx_low] + frac * self.signal[idx_high]
        elif idx_low == len(self.signal) - 1:
            return self.signal[idx_low]
        else:
            return 0.0

    def inject(self, fields, t, dt, current_step, resolution, design):
        """Inject source fields into the grid."""
        if self._Ez_profile is None and self._jz_profile is None:
            self.initialize(fields.permittivity, resolution)

        signal_value_e = self._get_signal_value(t + 0.5 * dt, dt)
        signal_value_h = self._get_signal_value(t + 0.5 * dt + self._dt_physical, dt)

        if self._Ex_profile is not None and self._is_3d:
            self._inject_3d(fields, signal_value_e, signal_value_h, dt, resolution)
        else:
            self._inject_2d(fields, signal_value_e, signal_value_h, dt, resolution)

    def _inject_3d(self, fields, signal_e, signal_h, dt, resolution):
        """Inject all 6 components for 3D simulation."""
        profiles = {
            "Ex": self._Ex_profile, "Ey": self._Ey_profile, "Ez": self._Ez_profile,
            "Hx": self._Hx_profile, "Hy": self._Hy_profile, "Hz": self._Hz_profile,
        }
        indices = {
            "Ex": self._Ex_indices, "Ey": self._Ey_indices, "Ez": self._Ez_indices,
            "Hx": self._Hx_indices, "Hy": self._Hy_indices, "Hz": self._Hz_indices,
        }
        inject_3d_fields(fields, profiles, indices, signal_e, signal_h, dt, resolution)

    def _inject_2d(self, fields, signal_e, signal_h, dt, resolution):
        """Legacy 2D injection (dominant components only)."""
        if self.pol == "tm":
            if self._ez_indices is not None and self._jz_profile is not None:
                eps_at_source = fields.permittivity[self._ez_indices]
                jz_term = self._jz_profile * signal_e / resolution
                ez_injection = -jz_term * dt / (EPS_0 * eps_at_source)
                fields.Ez = fields.Ez.at[self._ez_indices].add(ez_injection)

            if self._h_indices is not None and self._my_profile is not None:
                mu_val = getattr(fields, "permeability", None)
                mu_at_source = mu_val[self._h_indices] if mu_val is not None else 1.0
                my_term = self._my_profile * signal_h / resolution
                h_injection = -my_term * dt / (MU_0 * mu_at_source)

                if self._h_component == "Hx":
                    fields.Hx = fields.Hx.at[self._h_indices].add(h_injection)
                else:
                    fields.Hy = fields.Hy.at[self._h_indices].add(h_injection)
        else:  # TE
            if self._e_indices is not None:
                j_profile = (
                    self._jx_profile if self._e_component == "Ex" else self._jy_profile
                )
                if j_profile is not None:
                    eps_at_source = fields.permittivity[self._e_indices]
                    j_term = j_profile * signal_e / resolution
                    e_injection = -j_term * dt / (EPS_0 * eps_at_source)

                    if self._e_component == "Ex":
                        fields.Ex = fields.Ex.at[self._e_indices].add(e_injection)
                    else:
                        fields.Ey = fields.Ey.at[self._e_indices].add(e_injection)

            if self._hz_indices is not None and self._mz_profile is not None:
                mu_val = getattr(fields, "permeability", None)
                mu_at_source = mu_val[self._hz_indices] if mu_val is not None else 1.0
                mz_term = self._mz_profile * signal_h / resolution
                hz_injection = -mz_term * dt / (MU_0 * mu_at_source)
                fields.Hz = fields.Hz.at[self._hz_indices].add(hz_injection)

    def show(self, field=None):
        """Visualize the 2D mode profile (for 3D simulations) or 1D profile (for 2D)."""
        from beamz.visual.source_plots import show_mode_profile

        show_mode_profile(self, field=field)

    def add_to_plot(
        self, ax, facecolor="none", edgecolor="crimson", alpha=0.8, linestyle="-"
    ):
        """Add source visualization to 2D matplotlib plot."""
        from beamz.visual.overlays import add_mode_source_to_plot

        add_mode_source_to_plot(
            self, ax, facecolor=facecolor, edgecolor=edgecolor,
            alpha=alpha, linestyle=linestyle,
        )
