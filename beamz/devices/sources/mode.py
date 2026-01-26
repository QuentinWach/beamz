import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED, MU_0, µm
from beamz.devices.sources.jax_utils import (
    differentiable_phase_alignment,
    jax_tukey_window,
)
from beamz.devices.sources.solve import solve_modes


class ModeSource:
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
        self.height = height  # For 3D: constrains z-direction; None for 2D, defaults to width for 3D
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
            # For 3D, default height to width if not specified
            if self.height is None:
                self.height = self.width
        else:
            ny, nx = permittivity.shape
            nz = 1
            self._grid_shape = (ny, nx)
            # For 2D, height should be None
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

            # Get permittivity slice for mode solver
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
        E_mode = e_fields[0]  # Shape: (3, nz, ny) or (3, ny)
        H_mode = h_fields[0]  # Shape: (3, nz, ny) or (3, ny)

        # 3. Extract all 6 components and convert to JAX arrays
        # Note: Mode solver output order depends on propagation axis
        # For 1D/2D eps with propagation_axis=0: E=[Ez, Ex, Ey], H=[Hz, Hx, Hy]
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

        # Use JAX argmax and angle for phase alignment
        idx_max = jnp.argmax(jnp.abs(ref_field))
        phase_ref = jnp.angle(ref_field.flatten()[idx_max])

        Ex_aligned = Ex_raw * jnp.exp(-1j * phase_ref)
        Ey_aligned = Ey_raw * jnp.exp(-1j * phase_ref)
        Ez_aligned = Ez_raw * jnp.exp(-1j * phase_ref)
        Hx_aligned = Hx_raw * jnp.exp(-1j * phase_ref)
        Hy_aligned = Hy_raw * jnp.exp(-1j * phase_ref)
        Hz_aligned = Hz_raw * jnp.exp(-1j * phase_ref)

        # 5. Apply Yee grid staggering and set up indices for 3D
        if is_3d:
            self._setup_3d_injection(
                Ex_aligned,
                Ey_aligned,
                Ez_aligned,
                Hx_aligned,
                Hy_aligned,
                Hz_aligned,
                center_idx,
                offset_idx,
                axis,
                nz,
                ny,
                nx,
                resolution,
            )
        else:
            # Fall back to 2D injection (legacy code path)
            # Pass raw E_mode and H_mode for proper index-based extraction
            self._setup_2d_injection(
                E_mode, H_mode, center_idx, offset_idx, axis, ny, nx, resolution
            )

        # Compute physical time shift
        self._compute_dt_physical(axis, is_3d, dx, dy)

    def _setup_3d_injection(
        self,
        Ex,
        Ey,
        Ez,
        Hx,
        Hy,
        Hz,
        center_idx,
        offset_idx,
        axis,
        nz,
        ny,
        nx,
        resolution,
    ):
        """Set up full 6-component injection for 3D simulations.

        Yee grid positions (for x-propagation, injection plane perpendicular to x):
        - Ex at (i+0.5, j, k)     -> longitudinal, at half-integer x
        - Ey at (i, j+0.5, k)     -> transverse, at integer x
        - Ez at (i, j, k+0.5)     -> transverse, at integer x
        - Hx at (i, j+0.5, k+0.5) -> longitudinal, at integer x
        - Hy at (i+0.5, j, k+0.5) -> transverse, at half-integer x
        - Hz at (i+0.5, j+0.5, k) -> transverse, at half-integer x
        """
        dir_sign = 1.0 if self.direction.startswith("+") else -1.0

        # Impedance correction factor
        ETA_0 = np.sqrt(MU_0 / EPS_0)
        Z_phys = ETA_0 / max(np.real(self._neff), 1e-6)

        if axis == "x":
            # Mode solver output is on (z, y) grid
            # Apply staggering to each component based on Yee positions

            # E-field components (inject as J currents)
            # Ex: longitudinal, at x+0.5, so use center_idx for E injection
            # Ey: at y+0.5, stagger along y
            # Ez: at z+0.5, stagger along z

            # For transverse E components at integer x (center_idx)
            # Ey at (center_idx, j+0.5, k) - stagger y
            Ey_staggered = 0.5 * (Ey[:, :-1] + Ey[:, 1:]) if Ey.shape[1] > 1 else Ey
            # Ez at (center_idx, j, k+0.5) - stagger z
            Ez_staggered = 0.5 * (Ez[:-1, :] + Ez[1:, :]) if Ez.shape[0] > 1 else Ez

            # Ex is longitudinal - at x+0.5, needs to be at offset position
            # Stagger nothing for Ex along transverse (it's at integer y, k)
            Ex_staggered = Ex

            # H-field components (inject as M currents)
            # Hx: longitudinal, at integer x (center_idx), stagger y and z
            Hx_staggered = Hx
            if Hx.shape[1] > 1:
                Hx_staggered = 0.5 * (Hx_staggered[:, :-1] + Hx_staggered[:, 1:])
            if Hx_staggered.shape[0] > 1:
                Hx_staggered = 0.5 * (Hx_staggered[:-1, :] + Hx_staggered[1:, :])

            # Hy: at x+0.5, stagger z only
            Hy_staggered = 0.5 * (Hy[:-1, :] + Hy[1:, :]) if Hy.shape[0] > 1 else Hy

            # Hz: at x+0.5, stagger y only
            Hz_staggered = 0.5 * (Hz[:, :-1] + Hz[:, 1:]) if Hz.shape[1] > 1 else Hz

            # Set up indices for x-propagation
            # E components at center_idx (integer x), H components at offset_idx (x+0.5)
            nz_ez = Ez_staggered.shape[0]
            ny_ez = Ez_staggered.shape[1]
            nz_ey = Ey_staggered.shape[0]
            ny_ey = Ey_staggered.shape[1]
            nz_ex = Ex_staggered.shape[0]
            ny_ex = Ex_staggered.shape[1]

            nz_hx = Hx_staggered.shape[0]
            ny_hx = Hx_staggered.shape[1]
            nz_hy = Hy_staggered.shape[0]
            ny_hy = Hy_staggered.shape[1]
            nz_hz = Hz_staggered.shape[0]
            ny_hz = Hz_staggered.shape[1]

            # Calculate y-bounds from width parameter to constrain injection region
            center_y_idx = int(round(self.center[1] / resolution))
            half_width_idx = int(round((self.width / 2) / resolution))
            y_start = max(0, center_y_idx - half_width_idx)
            y_end = min(ny, center_y_idx + half_width_idx)
            self._y_start = y_start
            self._y_end = y_end

            # Calculate z-bounds from height parameter to constrain injection region (for 3D)
            center_z_idx = (
                int(round(self.center[2] / resolution))
                if len(self.center) > 2
                else nz // 2
            )
            half_height_idx = int(round((self.height / 2) / resolution))
            z_start = max(0, center_z_idx - half_height_idx)
            z_end = min(nz, center_z_idx + half_height_idx)
            self._z_start = z_start
            self._z_end = z_end

            # Indices: (z_slice, y_slice, x_index) - constrained to width and height regions
            # Ex at offset (longitudinal)
            self._Ex_indices = (
                slice(z_start, min(z_end, nz_ex, nz)),
                slice(y_start, min(y_end, ny_ex, ny)),
                offset_idx,
            )
            # Ey, Ez at center (transverse E)
            self._Ey_indices = (
                slice(z_start, min(z_end, nz_ey, nz)),
                slice(y_start, min(y_end, ny_ey, ny - 1)),
                center_idx,
            )
            self._Ez_indices = (
                slice(z_start, min(z_end, nz_ez, nz - 1)),
                slice(y_start, min(y_end, ny_ez, ny)),
                center_idx,
            )

            # Hx at center (longitudinal) - constrained to width and height regions
            self._Hx_indices = (
                slice(z_start, min(z_end, nz_hx, nz - 1)),
                slice(y_start, min(y_end, ny_hx, ny - 1)),
                center_idx,
            )
            # Hy, Hz at offset (transverse H) - constrained to width and height regions
            self._Hy_indices = (
                slice(z_start, min(z_end, nz_hy, nz - 1)),
                slice(y_start, min(y_end, ny_hy, ny)),
                offset_idx,
            )
            self._Hz_indices = (
                slice(z_start, min(z_end, nz_hz, nz)),
                slice(y_start, min(y_end, ny_hz, ny - 1)),
                offset_idx,
            )

            # Store profiles with direction sign and impedance correction
            # J = n × H (electric current from H)
            # M = -n × E (magnetic current from E)
            # For +x propagation, n = +x_hat

            # Apply impedance correction to E fields (JAX-compatible)
            norm_E = jnp.maximum(
                jnp.max(jnp.abs(Ey_staggered)),
                jnp.maximum(jnp.max(jnp.abs(Ez_staggered)), 1e-12),
            )
            norm_H = jnp.maximum(
                jnp.max(jnp.abs(Hy_staggered)),
                jnp.maximum(jnp.max(jnp.abs(Hz_staggered)), 1e-12),
            )
            current_Z = norm_E / norm_H
            corr = jnp.where(
                (norm_E > 1e-12) & (norm_H > 1e-12), Z_phys / current_Z, 1.0
            )
            Ey_staggered = Ey_staggered * corr
            Ez_staggered = Ez_staggered * corr
            Ex_staggered = Ex_staggered * corr

            # Crop profiles to width (y) and height (z) regions and apply smooth window
            profile_z_start = z_start
            profile_z_end = min(z_end, Ex_staggered.shape[0])
            profile_y_start = y_start
            profile_y_end = min(y_end, Ex_staggered.shape[1])
            height_cells = profile_z_end - profile_z_start
            width_cells = profile_y_end - profile_y_start

            # Create 2D Tukey window for smooth edges in both z and y directions (JAX)
            if height_cells > 2:
                window_z = jax_tukey_window(height_cells, alpha=0.3)
            else:
                window_z = jnp.ones(max(1, height_cells))

            if width_cells > 2:
                window_y = jax_tukey_window(width_cells, alpha=0.3)
            else:
                window_y = jnp.ones(max(1, width_cells))

            # Create 2D window by outer product (JAX)
            window_2d = window_z[:, jnp.newaxis] * window_y[jnp.newaxis, :]

            # Crop and window each profile
            def crop_and_window_2d(profile, z_s, z_e, y_s, y_e, window):
                # Crop in both z and y directions
                cropped = profile[z_s:z_e, y_s:y_e]
                # Match window shape to cropped profile
                if cropped.shape == window.shape:
                    return cropped * window
                elif cropped.size > 0:
                    # Window might be slightly different size due to shape mismatches
                    # Crop or pad window to match cropped profile
                    z_min = min(cropped.shape[0], window.shape[0])
                    y_min = min(cropped.shape[1], window.shape[1])
                    window_cropped = window[:z_min, :y_min]
                    cropped_matched = cropped[:z_min, :y_min]
                    return cropped_matched * window_cropped
                return cropped

            # Store profiles (real part for time-domain injection)
            # Sign convention for TFSF: J_i = dir_sign * H_j, M_i = dir_sign * E_j
            self._Ex_profile = dir_sign * np.real(
                crop_and_window_2d(
                    Ex_staggered,
                    profile_z_start,
                    profile_z_end,
                    profile_y_start,
                    profile_y_end,
                    window_2d,
                )
            )
            self._Ey_profile = dir_sign * np.real(
                crop_and_window_2d(
                    Ey_staggered,
                    profile_z_start,
                    min(profile_z_end, Ey_staggered.shape[0]),
                    profile_y_start,
                    min(profile_y_end, Ey_staggered.shape[1]),
                    window_2d,
                )
            )
            self._Ez_profile = dir_sign * np.real(
                crop_and_window_2d(
                    Ez_staggered,
                    profile_z_start,
                    min(profile_z_end, Ez_staggered.shape[0]),
                    profile_y_start,
                    min(profile_y_end, Ez_staggered.shape[1]),
                    window_2d,
                )
            )
            self._Hx_profile = dir_sign * np.real(
                crop_and_window_2d(
                    Hx_staggered,
                    profile_z_start,
                    min(profile_z_end, Hx_staggered.shape[0]),
                    profile_y_start,
                    min(profile_y_end, Hx_staggered.shape[1]),
                    window_2d,
                )
            )
            self._Hy_profile = dir_sign * np.real(
                crop_and_window_2d(
                    Hy_staggered,
                    profile_z_start,
                    min(profile_z_end, Hy_staggered.shape[0]),
                    profile_y_start,
                    min(profile_y_end, Hy_staggered.shape[1]),
                    window_2d,
                )
            )
            self._Hz_profile = dir_sign * np.real(
                crop_and_window_2d(
                    Hz_staggered,
                    profile_z_start,
                    min(profile_z_end, Hz_staggered.shape[0]),
                    profile_y_start,
                    min(profile_y_end, Hz_staggered.shape[1]),
                    window_2d,
                )
            )

            # Legacy compatibility
            self._h_component = "Hy"
            self._e_component = "Ey"
            self._jz_profile = self._Hz_profile
            self._my_profile = self._Ez_profile

        else:  # axis == "y"
            # Mode solver output is on (z, x) grid

            # For y-propagation:
            # Ex at (i+0.5, j, k) - stagger x
            Ex_staggered = 0.5 * (Ex[:, :-1] + Ex[:, 1:]) if Ex.shape[1] > 1 else Ex
            # Ey longitudinal at y+0.5
            Ey_staggered = Ey
            # Ez at z+0.5 - stagger z
            Ez_staggered = 0.5 * (Ez[:-1, :] + Ez[1:, :]) if Ez.shape[0] > 1 else Ez

            # Hx at y integer, stagger z
            Hx_staggered = 0.5 * (Hx[:-1, :] + Hx[1:, :]) if Hx.shape[0] > 1 else Hx
            # Hy longitudinal
            Hy_staggered = Hy
            if Hy.shape[0] > 1:
                Hy_staggered = 0.5 * (Hy_staggered[:-1, :] + Hy_staggered[1:, :])
            if Hy_staggered.shape[1] > 1:
                Hy_staggered = 0.5 * (Hy_staggered[:, :-1] + Hy_staggered[:, 1:])
            # Hz at y+0.5, stagger x
            Hz_staggered = 0.5 * (Hz[:, :-1] + Hz[:, 1:]) if Hz.shape[1] > 1 else Hz

            nz_ex = Ex_staggered.shape[0]
            nx_ex = Ex_staggered.shape[1]
            nz_ey = Ey_staggered.shape[0]
            nx_ey = Ey_staggered.shape[1]
            nz_ez = Ez_staggered.shape[0]
            nx_ez = Ez_staggered.shape[1]

            nz_hx = Hx_staggered.shape[0]
            nx_hx = Hx_staggered.shape[1]
            nz_hy = Hy_staggered.shape[0]
            nx_hy = Hy_staggered.shape[1]
            nz_hz = Hz_staggered.shape[0]
            nx_hz = Hz_staggered.shape[1]

            nx_grid = self._grid_shape[2]
            nz_grid = self._grid_shape[0]

            # Calculate x-bounds from width parameter (for y-propagation, x is transverse)
            center_x_idx = int(round(self.center[0] / resolution))
            half_width_idx = int(round((self.width / 2) / resolution))
            x_start = max(0, center_x_idx - half_width_idx)
            x_end = min(nx_grid, center_x_idx + half_width_idx)
            self._x_start = x_start
            self._x_end = x_end

            # Calculate z-bounds from height parameter to constrain injection region (for 3D)
            center_z_idx = (
                int(round(self.center[2] / resolution))
                if len(self.center) > 2
                else nz_grid // 2
            )
            half_height_idx = int(round((self.height / 2) / resolution))
            z_start = max(0, center_z_idx - half_height_idx)
            z_end = min(nz_grid, center_z_idx + half_height_idx)
            self._z_start = z_start
            self._z_end = z_end

            # Indices: (z_slice, y_index, x_slice) - constrained to width and height regions
            self._Ex_indices = (
                slice(z_start, min(z_end, nz_ex, nz_grid)),
                center_idx,
                slice(x_start, min(x_end, nx_ex, nx_grid - 1)),
            )
            self._Ey_indices = (
                slice(z_start, min(z_end, nz_ey, nz_grid)),
                offset_idx,
                slice(x_start, min(x_end, nx_ey, nx_grid)),
            )
            self._Ez_indices = (
                slice(z_start, min(z_end, nz_ez, nz_grid - 1)),
                center_idx,
                slice(x_start, min(x_end, nx_ez, nx_grid)),
            )

            self._Hx_indices = (
                slice(z_start, min(z_end, nz_hx, nz_grid - 1)),
                center_idx,
                slice(x_start, min(x_end, nx_hx, nx_grid)),
            )
            self._Hy_indices = (
                slice(z_start, min(z_end, nz_hy, nz_grid - 1)),
                center_idx,
                slice(x_start, min(x_end, nx_hy, nx_grid - 1)),
            )
            self._Hz_indices = (
                slice(z_start, min(z_end, nz_hz, nz_grid)),
                offset_idx,
                slice(x_start, min(x_end, nx_hz, nx_grid - 1)),
            )

            # Impedance correction
            norm_E = max(
                np.max(np.abs(Ex_staggered)), np.max(np.abs(Ez_staggered)), 1e-12
            )
            norm_H = max(
                np.max(np.abs(Hx_staggered)), np.max(np.abs(Hz_staggered)), 1e-12
            )
            if norm_E > 1e-12 and norm_H > 1e-12:
                current_Z = norm_E / norm_H
                corr = Z_phys / current_Z
                Ex_staggered = Ex_staggered * corr
                Ey_staggered = Ey_staggered * corr
                Ez_staggered = Ez_staggered * corr

            # Crop profiles to width (x) and height (z) regions and apply smooth window
            profile_z_start = z_start
            profile_z_end = min(z_end, Ex_staggered.shape[0])
            profile_x_start = x_start
            profile_x_end = min(x_end, Ex_staggered.shape[1])
            height_cells = profile_z_end - profile_z_start
            width_cells = profile_x_end - profile_x_start

            # Create 2D Tukey window for smooth edges in both z and x directions
            from scipy.signal.windows import tukey

            if height_cells > 2:
                window_z = tukey(height_cells, alpha=0.3)
            else:
                window_z = np.ones(max(1, height_cells))

            if width_cells > 2:
                window_x = tukey(width_cells, alpha=0.3)
            else:
                window_x = np.ones(max(1, width_cells))

            # Create 2D window by outer product
            window_2d = window_z[:, np.newaxis] * window_x[np.newaxis, :]

            def crop_and_window_2d(profile, z_s, z_e, x_s, x_e, window):
                # Crop in both z and x directions
                cropped = profile[z_s:z_e, x_s:x_e]
                # Match window shape to cropped profile
                if cropped.shape == window.shape:
                    return cropped * window
                elif cropped.size > 0:
                    # Window might be slightly different size due to shape mismatches
                    # Crop or pad window to match cropped profile
                    z_min = min(cropped.shape[0], window.shape[0])
                    x_min = min(cropped.shape[1], window.shape[1])
                    window_cropped = window[:z_min, :x_min]
                    cropped_matched = cropped[:z_min, :x_min]
                    return cropped_matched * window_cropped
                return cropped

            self._Ex_profile = dir_sign * np.real(
                crop_and_window_2d(
                    Ex_staggered,
                    profile_z_start,
                    profile_z_end,
                    profile_x_start,
                    profile_x_end,
                    window_2d,
                )
            )
            self._Ey_profile = dir_sign * np.real(
                crop_and_window_2d(
                    Ey_staggered,
                    profile_z_start,
                    min(profile_z_end, Ey_staggered.shape[0]),
                    profile_x_start,
                    min(profile_x_end, Ey_staggered.shape[1]),
                    window_2d,
                )
            )
            self._Ez_profile = dir_sign * np.real(
                crop_and_window_2d(
                    Ez_staggered,
                    profile_z_start,
                    min(profile_z_end, Ez_staggered.shape[0]),
                    profile_x_start,
                    min(profile_x_end, Ez_staggered.shape[1]),
                    window_2d,
                )
            )
            self._Hx_profile = dir_sign * np.real(
                crop_and_window_2d(
                    Hx_staggered,
                    profile_z_start,
                    min(profile_z_end, Hx_staggered.shape[0]),
                    profile_x_start,
                    min(profile_x_end, Hx_staggered.shape[1]),
                    window_2d,
                )
            )
            self._Hy_profile = dir_sign * np.real(
                crop_and_window_2d(
                    Hy_staggered,
                    profile_z_start,
                    min(profile_z_end, Hy_staggered.shape[0]),
                    profile_x_start,
                    min(profile_x_end, Hy_staggered.shape[1]),
                    window_2d,
                )
            )
            self._Hz_profile = dir_sign * np.real(
                crop_and_window_2d(
                    Hz_staggered,
                    profile_z_start,
                    min(profile_z_end, Hz_staggered.shape[0]),
                    profile_x_start,
                    min(profile_x_end, Hz_staggered.shape[1]),
                    window_2d,
                )
            )

            self._h_component = "Hx"
            self._e_component = "Ex"
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
            # Calculate y-bounds from width parameter (for x-propagation, y is transverse)
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

                # Extract using indices with fallback (original logic)
                # For TM x-prop: use H_mode[1] and E_mode[2]
                Hy_raw = np.squeeze(H_mode[1])
                Ez_raw = np.squeeze(E_mode[2])
                if np.max(np.abs(Hy_raw)) < 1e-9:
                    Hy_raw = np.squeeze(H_mode[2])
                if np.max(np.abs(Ez_raw)) < 1e-9:
                    Ez_raw = np.squeeze(E_mode[1])

                # Phase align
                idx_max = np.argmax(np.abs(Hy_raw))
                phase_ref = np.angle(Hy_raw.flatten()[idx_max])
                Hy_profile = Hy_raw * np.exp(-1j * phase_ref)
                Ez_profile = Ez_raw * np.exp(-1j * phase_ref)

                # Impedance correction
                norm_h, norm_e = np.max(np.abs(Hy_profile)), np.max(np.abs(Ez_profile))
                if norm_h > 1e-12 and norm_e > 1e-12:
                    corr = Z_phys / (norm_e / norm_h)
                    Ez_profile = Ez_profile * corr

                # Crop to width region and apply smooth window
                width_cells = y_end - y_start
                if width_cells > 2:
                    from scipy.signal.windows import tukey

                    window = tukey(width_cells, alpha=0.3)
                else:
                    window = np.ones(max(1, width_cells))

                Hy_cropped = np.real(Hy_profile)[y_start:y_end]
                Ez_cropped = np.real(Ez_profile)[y_start:y_end]
                if len(Hy_cropped) == len(window):
                    Hy_cropped = Hy_cropped * window
                    Ez_cropped = Ez_cropped * window

                self._jz_profile = dir_sign * Hy_cropped
                self._my_profile = dir_sign * Ez_cropped

            else:  # TE
                hz_col = (
                    max(0, offset_idx - 1)
                    if self.direction == "+x"
                    else min(nx - 2, offset_idx)
                )
                ny_eff = min(ny - 1, y_end - y_start)

                self._hz_indices = (slice(y_start, min(y_end, ny - 1)), hz_col)
                self._e_indices = (slice(y_start, min(y_end, ny - 1)), offset_idx)
                self._e_component = "Ey"

                # Extract with fallback
                h_candidates = [np.squeeze(H_mode[i]) for i in range(3)]
                e_candidates = [np.squeeze(E_mode[i]) for i in range(3)]
                h_scores = [float(np.max(np.abs(hc))) for hc in h_candidates]
                e_scores = [float(np.max(np.abs(ec))) for ec in e_candidates]
                Hz_raw = h_candidates[int(np.argmax(h_scores))]
                Ey_raw = e_candidates[int(np.argmax(e_scores))]

                # Stagger to Yee grid positions
                Hz_staggered = 0.5 * (Hz_raw[:-1] + Hz_raw[1:])
                Ey_staggered = 0.5 * (Ey_raw[:-1] + Ey_raw[1:])

                # Phase align
                idx_max = np.argmax(np.abs(Hz_staggered))
                phase_ref = np.angle(Hz_staggered.flatten()[idx_max])
                Hz_profile = Hz_staggered * np.exp(-1j * phase_ref)
                Ey_profile = Ey_staggered * np.exp(-1j * phase_ref)

                # Impedance correction
                norm_h, norm_e = np.max(np.abs(Hz_profile)), np.max(np.abs(Ey_profile))
                if norm_h > 1e-12 and norm_e > 1e-12:
                    corr = Z_phys / (norm_e / norm_h)
                    Ey_profile = Ey_profile * corr

                # Crop to width region and apply smooth window
                width_cells = min(y_end, len(Hz_profile)) - y_start
                if width_cells > 2:
                    from scipy.signal.windows import tukey

                    window = tukey(width_cells, alpha=0.3)
                else:
                    window = np.ones(max(1, width_cells))

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

        else:  # axis == "y"
            # Calculate x-bounds from width parameter (for y-propagation, x is transverse)
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

                # Extract using indices with fallback
                Hx_raw = np.squeeze(H_mode[1])
                Ez_raw = np.squeeze(E_mode[2])
                if np.max(np.abs(Hx_raw)) < 1e-9:
                    Hx_raw = np.squeeze(H_mode[2])
                if np.max(np.abs(Ez_raw)) < 1e-9:
                    Ez_raw = np.squeeze(E_mode[1])

                # Phase align
                idx_max = np.argmax(np.abs(Hx_raw))
                phase_ref = np.angle(Hx_raw.flatten()[idx_max])
                Hx_profile = Hx_raw * np.exp(-1j * phase_ref)
                Ez_profile = Ez_raw * np.exp(-1j * phase_ref)

                # Impedance correction
                norm_h, norm_e = np.max(np.abs(Hx_profile)), np.max(np.abs(Ez_profile))
                if norm_h > 1e-12 and norm_e > 1e-12:
                    corr = Z_phys / (norm_e / norm_h)
                    Ez_profile = Ez_profile * corr

                # Crop to width region and apply smooth window
                width_cells = x_end - x_start
                if width_cells > 2:
                    from scipy.signal.windows import tukey

                    window = tukey(width_cells, alpha=0.3)
                else:
                    window = np.ones(max(1, width_cells))

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
                    max(0, offset_idx - 1)
                    if self.direction == "+y"
                    else min(ny - 2, offset_idx)
                )
                nx_eff = min(nx - 1, x_end - x_start)

                self._hz_indices = (hz_row, slice(x_start, min(x_end, nx - 1)))
                self._e_indices = (offset_idx, slice(x_start, min(x_end, nx - 1)))
                self._e_component = "Ex"

                # Extract with fallback
                h_candidates = [np.squeeze(H_mode[i]) for i in range(3)]
                e_candidates = [np.squeeze(E_mode[i]) for i in range(3)]
                h_scores = [float(np.max(np.abs(hc))) for hc in h_candidates]
                e_scores = [float(np.max(np.abs(ec))) for ec in e_candidates]
                Hz_raw = h_candidates[int(np.argmax(h_scores))]
                Ex_raw = e_candidates[int(np.argmax(e_scores))]

                # Stagger
                Hz_staggered = 0.5 * (Hz_raw[:-1] + Hz_raw[1:])
                Ex_staggered = 0.5 * (Ex_raw[:-1] + Ex_raw[1:])

                # Phase align
                idx_max = np.argmax(np.abs(Hz_staggered))
                phase_ref = np.angle(Hz_staggered.flatten()[idx_max])
                Hz_profile = Hz_staggered * np.exp(-1j * phase_ref)
                Ex_profile = Ex_staggered * np.exp(-1j * phase_ref)

                # Impedance correction
                norm_h, norm_e = np.max(np.abs(Hz_profile)), np.max(np.abs(Ex_profile))
                if norm_h > 1e-12 and norm_e > 1e-12:
                    corr = Z_phys / (norm_e / norm_h)
                    Ex_profile = Ex_profile * corr

                # Crop to width region and apply smooth window
                width_cells = min(x_end, len(Hz_profile)) - x_start
                if width_cells > 2:
                    from scipy.signal.windows import tukey

                    window = tukey(width_cells, alpha=0.3)
                else:
                    window = np.ones(max(1, width_cells))

                Hz_cropped = np.real(Hz_profile)[x_start : min(x_end, len(Hz_profile))]
                Ex_cropped = np.real(Ex_profile)[x_start : min(x_end, len(Ex_profile))]
                if len(Hz_cropped) == len(window):
                    Hz_cropped = Hz_cropped * window
                    Ex_cropped = Ex_cropped * window

                self._jx_profile = dir_sign * Hz_cropped
                self._mz_profile = dir_sign * Ex_cropped

    def _compute_dt_physical(self, axis, is_3d, dx, dy):
        """Compute physical time shift between E and H injection planes."""
        if self._neff is None:
            return

        coord_e = 0.0
        coord_h = 0.0

        if is_3d:
            # For 3D, use center positions from indices
            if axis == "x":
                if self._Ez_indices is not None:
                    idx_e = self._Ez_indices[2]
                    coord_e = (idx_e + 0.5) * dx
                if self._Hy_indices is not None:
                    idx_h = self._Hy_indices[2]
                    coord_h = (idx_h + 1.0) * dx
            else:
                if self._Ez_indices is not None:
                    idx_e = self._Ez_indices[1]
                    coord_e = (idx_e + 0.5) * dy
                if self._Hx_indices is not None:
                    idx_h = self._Hx_indices[1]
                    coord_h = (idx_h + 1.0) * dy
        else:
            # Legacy 2D calculation
            if axis == "x":
                if self.pol == "tm":
                    idx_e = self._ez_indices[1] if self._ez_indices else 0
                    idx_h = self._h_indices[1] if self._h_indices else 0
                    coord_e = (idx_e + 0.5) * dx
                    coord_h = (idx_h + 1.0) * dx
                else:
                    idx_e = self._e_indices[1] if self._e_indices else 0
                    idx_h = self._hz_indices[1] if self._hz_indices else 0
                    coord_e = (idx_e + 0.5) * dx
                    coord_h = (idx_h + 1.0) * dx
            else:
                if self.pol == "tm":
                    idx_e = self._ez_indices[0] if self._ez_indices else 0
                    idx_h = self._h_indices[0] if self._h_indices else 0
                    coord_e = (idx_e + 0.5) * dy
                    coord_h = (idx_h + 1.0) * dy
                else:
                    idx_e = self._e_indices[0] if self._e_indices else 0
                    idx_h = self._hz_indices[0] if self._hz_indices else 0
                    coord_e = (idx_e + 0.5) * dy
                    coord_h = (idx_h + 1.0) * dy

        self._dt_physical = (
            (coord_e - coord_h) * float(np.real(self._neff)) / LIGHT_SPEED
        )

    def _enforce_propagation_direction(self, E, H, axis):
        """Ensure the mode propagates in the correct direction by checking Poynting vector."""
        S = np.cross(E, np.conjugate(H), axis=0)
        power = float(np.real(np.sum(S[axis])))
        direction_sign = 1.0 if self.direction.startswith("+") else -1.0
        if power * direction_sign < 0:
            H = -H
        return E, H

    def _phase_align(self, field):
        """Align phase so field is mostly real at the peak amplitude."""
        idx_max = np.argmax(np.abs(field))
        phase = np.angle(field[idx_max])
        return field * np.exp(-1j * phase)

    def show(self, field=None):
        """Visualize the 2D mode profile (for 3D simulations) or 1D profile (for 2D)."""
        import matplotlib.pyplot as plt

        if self._Ez_profile is None and self._jz_profile is None:
            if self.grid is not None and hasattr(self.grid, "permittivity"):
                res = getattr(self.grid, "resolution", 0.05e-6)
                self.initialize(self.grid.permittivity, res)
            else:
                print(
                    "[ModeSource] Source not initialized. Call Simulation or initialize manually."
                )
                return

        # Use 3D profiles if available
        if self._Ez_profile is not None:
            profile = self._Ez_profile
            title = "Ez (mode profile)"
        elif self._jz_profile is not None:
            profile = self._jz_profile
            title = "Hz (mode profile)"
        else:
            print("[ModeSource] No profiles available.")
            return

        profile = np.squeeze(profile)

        plt.figure(figsize=(8, 6))
        if profile.ndim == 2:
            im = plt.imshow(
                np.abs(profile), origin="lower", cmap="magma", aspect="auto"
            )
            plt.colorbar(im, label="Absolute Amplitude")
            plt.title(f"Mode Source 2D Profile: {title} (neff={self._neff:.4f})")
            if self.direction in ["+x", "-x"]:
                plt.xlabel("Y-axis")
                plt.ylabel("Z-axis")
            else:
                plt.xlabel("X-axis")
                plt.ylabel("Z-axis")
        else:
            plt.plot(np.abs(profile), "k-")
            plt.title(f"Mode Source 1D Profile: {title} (neff={self._neff:.4f})")
            plt.xlabel("Transverse Coordinate (cells)")
            plt.ylabel("Absolute Amplitude")
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    def _plot_mode_profile_3d(self):
        """Plot all 6 field component profiles for 3D simulation."""
        try:
            import matplotlib.pyplot as plt

            profiles = {
                "Ex": self._Ex_profile,
                "Ey": self._Ey_profile,
                "Ez": self._Ez_profile,
                "Hx": self._Hx_profile,
                "Hy": self._Hy_profile,
                "Hz": self._Hz_profile,
            }

            # Filter out None profiles
            valid_profiles = {k: v for k, v in profiles.items() if v is not None}
            if not valid_profiles:
                return

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for idx, (name, profile) in enumerate(valid_profiles.items()):
                if idx >= 6:
                    break
                ax = axes[idx]
                profile_squeezed = np.squeeze(profile)

                if profile_squeezed.ndim == 2:
                    # Use aspect='equal' to ensure grid cells are square (not stretched)
                    im = ax.imshow(
                        np.abs(profile_squeezed),
                        origin="lower",
                        cmap="magma",
                        aspect="equal",
                    )
                    plt.colorbar(im, ax=ax, label="Amplitude")
                else:
                    ax.plot(np.abs(profile_squeezed), "b-")

                ax.set_title(f"{name} (max={np.max(np.abs(profile_squeezed)):.2e})")
                ax.set_xlabel("Y" if self.direction in ["+x", "-x"] else "X")
                ax.set_ylabel("Z")

            plt.suptitle(
                f"3D Mode Profiles (neff={self._neff:.4f}, dir={self.direction}, pol={self.pol})"
            )
            plt.tight_layout()
            plt.savefig("mode_profile.png", dpi=150, bbox_inches="tight")
            print(f"[ModeSource] 3D mode profiles saved to mode_profile.png")
            plt.close()
        except Exception as e:
            print(f"[ModeSource] Could not plot mode profile: {e}")

    def _plot_mode_profile_2d(self):
        """Plot mode profiles for 2D simulation."""
        try:
            import matplotlib.pyplot as plt

            if self.pol == "tm":
                j_profile = self._jz_profile
                m_profile = self._my_profile
                j_label = "Jz (from H)"
                m_label = "My (from Ez)"
            else:
                j_profile = (
                    self._jy_profile
                    if self._jy_profile is not None
                    else self._jx_profile
                )
                m_profile = self._mz_profile
                j_label = "Jy/Jx (from Hz)"
                m_label = "Mz (from E)"

            if j_profile is None or m_profile is None:
                return

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot(np.abs(np.squeeze(j_profile)), "b-")
            ax1.set_xlabel("Transverse coord")
            ax1.set_ylabel("Amplitude")
            ax1.set_title(f"{j_label} (neff={self._neff:.4f})")
            ax1.grid(True)

            ax2.plot(np.abs(np.squeeze(m_profile)), "r-")
            ax2.set_xlabel("Transverse coord")
            ax2.set_ylabel("Amplitude")
            ax2.set_title(f"{m_label} (dir={self.direction})")
            ax2.grid(True)

            plt.tight_layout()
            plt.savefig("mode_profile.png", dpi=150, bbox_inches="tight")
            print(f"[ModeSource] Mode profile saved to mode_profile.png")
            plt.close()
        except Exception as e:
            print(f"[ModeSource] Could not plot mode profile: {e}")

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
        """Inject source fields into the grid.

        For 3D simulations, injects all 6 field components at their proper
        Yee grid positions for accurate mode injection.
        """
        if self._Ez_profile is None and self._jz_profile is None:
            self.initialize(fields.permittivity, resolution)

        # Timing for E and H injection
        signal_value_e = self._get_signal_value(t + 0.5 * dt, dt)
        signal_value_h = self._get_signal_value(t + 0.5 * dt + self._dt_physical, dt)

        # Check if we're in 3D mode with full 6-component profiles
        if self._Ex_profile is not None and self._is_3d:
            self._inject_3d(fields, signal_value_e, signal_value_h, dt, resolution)
        else:
            self._inject_2d(fields, signal_value_e, signal_value_h, dt, resolution)

    def _inject_3d(self, fields, signal_e, signal_h, dt, resolution):
        """Inject all 6 components for 3D simulation."""
        # Electric current injection: J affects E update
        # ∂E/∂t = (curl H - J) / ε → E += -J * dt / (ε₀ * ε)

        # Magnetic current injection: M affects H update
        # ∂H/∂t = -(curl E + M) / μ → H += -M * dt / (μ₀ * μ)

        # Inject Ex
        if self._Ex_profile is not None and self._Ex_indices is not None:
            try:
                profile = self._Ex_profile
                target_shape = fields.Ex[self._Ex_indices].shape
                profile = self._match_shape(profile, target_shape)
                if profile is not None:
                    eps = fields.permittivity[self._Ex_indices]
                    j_term = self._Hx_profile  # J_x from H_x (cross product)
                    j_term = self._match_shape(j_term, target_shape)
                    if j_term is not None:
                        fields.Ex = fields.Ex.at[self._Ex_indices].add(
                            -j_term * signal_e * dt / (EPS_0 * eps * resolution)
                        )
            except Exception:
                pass

        # Inject Ey
        if self._Ey_profile is not None and self._Ey_indices is not None:
            try:
                target_shape = fields.Ey[self._Ey_indices].shape
                # J_y comes from H_z component (n × H)_y
                j_term = self._Hz_profile
                j_term = self._match_shape(j_term, target_shape)
                if j_term is not None:
                    eps = fields.permittivity[self._Ey_indices]
                    fields.Ey = fields.Ey.at[self._Ey_indices].add(
                        -j_term * signal_e * dt / (EPS_0 * eps * resolution)
                    )
            except Exception:
                pass

        # Inject Ez (primary for TM)
        if self._Ez_profile is not None and self._Ez_indices is not None:
            try:
                target_shape = fields.Ez[self._Ez_indices].shape
                # J_z comes from H_y component (n × H)_z
                j_term = self._Hy_profile
                j_term = self._match_shape(j_term, target_shape)
                if j_term is not None:
                    eps = fields.permittivity[self._Ez_indices]
                    fields.Ez = fields.Ez.at[self._Ez_indices].add(
                        -j_term * signal_e * dt / (EPS_0 * eps * resolution)
                    )
            except Exception:
                pass

        # Inject Hx
        if self._Hx_profile is not None and self._Hx_indices is not None:
            try:
                target_shape = fields.Hx[self._Hx_indices].shape
                # M_x comes from E_x component (-n × E)_x
                m_term = self._Ex_profile
                m_term = self._match_shape(m_term, target_shape)
                if m_term is not None:
                    mu = getattr(fields, "permeability", None)
                    mu_val = mu[self._Hx_indices] if mu is not None else 1.0
                    fields.Hx = fields.Hx.at[self._Hx_indices].add(
                        -m_term * signal_h * dt / (MU_0 * mu_val * resolution)
                    )
            except Exception:
                pass

        # Inject Hy (primary for TM x-prop)
        if self._Hy_profile is not None and self._Hy_indices is not None:
            try:
                target_shape = fields.Hy[self._Hy_indices].shape
                # M_y comes from E_z component (-n × E)_y
                m_term = self._Ez_profile
                m_term = self._match_shape(m_term, target_shape)
                if m_term is not None:
                    mu = getattr(fields, "permeability", None)
                    mu_val = mu[self._Hy_indices] if mu is not None else 1.0
                    fields.Hy = fields.Hy.at[self._Hy_indices].add(
                        -m_term * signal_h * dt / (MU_0 * mu_val * resolution)
                    )
            except Exception:
                pass

        # Inject Hz (primary for TE)
        if self._Hz_profile is not None and self._Hz_indices is not None:
            try:
                target_shape = fields.Hz[self._Hz_indices].shape
                # M_z comes from E_y component (-n × E)_z
                m_term = self._Ey_profile
                m_term = self._match_shape(m_term, target_shape)
                if m_term is not None:
                    mu = getattr(fields, "permeability", None)
                    mu_val = mu[self._Hz_indices] if mu is not None else 1.0
                    fields.Hz = fields.Hz.at[self._Hz_indices].add(
                        -m_term * signal_h * dt / (MU_0 * mu_val * resolution)
                    )
            except Exception:
                pass

    def _match_shape(self, profile, target_shape):
        """Match profile shape to target field shape."""
        if profile is None:
            return None
        profile = np.squeeze(profile)

        # Handle shape mismatch
        if profile.shape == target_shape:
            return profile

        # Try to trim or pad
        if profile.ndim == len(target_shape):
            slices = tuple(
                slice(0, min(profile.shape[i], target_shape[i]))
                for i in range(profile.ndim)
            )
            trimmed = profile[slices]

            if trimmed.shape == target_shape:
                return trimmed

            # Pad if needed
            result = np.zeros(target_shape, dtype=profile.dtype)
            slices_result = tuple(
                slice(0, trimmed.shape[i]) for i in range(trimmed.ndim)
            )
            result[slices_result] = trimmed
            return result

        return None

    def _inject_2d(self, fields, signal_e, signal_h, dt, resolution):
        """Legacy 2D injection (dominant components only)."""
        if self.pol == "tm":
            # TM Injection: Jz -> Ez, My -> Hx/Hy
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
            # TE Injection: Jx/Jy -> Ex/Ey, Mz -> Hz
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

    def add_to_plot(
        self, ax, facecolor="none", edgecolor="crimson", alpha=0.8, linestyle="-"
    ):
        """Add source visualization to 2D matplotlib plot.

        Draws a line at the source position perpendicular to the propagation direction.
        For x-propagation: vertical line at x position
        For y-propagation: horizontal line at y position
        """
        from matplotlib.patches import FancyArrowPatch

        # Get center position
        center = (
            self.center if isinstance(self.center, (tuple, list)) else (self.center, 0)
        )
        if len(center) == 3:
            # 3D center - project to 2D based on direction
            if self.direction in ["+x", "-x"]:
                x_pos = center[0]
                y_pos = center[1]
            else:
                x_pos = center[0]
                y_pos = center[1]
        else:
            x_pos, y_pos = center[0], center[1]

        # Get transverse extent from width or grid
        half_width = self.width / 2 if self.width else 0.5e-6

        # Determine line endpoints based on propagation direction
        if self.direction in ["+x", "-x"]:
            # Vertical line at x position
            y_start = y_pos - half_width
            y_end = y_pos + half_width
            line_x = [x_pos, x_pos]
            line_y = [y_start, y_end]
            # Arrow direction
            arrow_dx = 0.3e-6 if self.direction == "+x" else -0.3e-6
            arrow_dy = 0
        else:  # +y or -y
            # Horizontal line at y position
            x_start = x_pos - half_width
            x_end = x_pos + half_width
            line_x = [x_start, x_end]
            line_y = [y_pos, y_pos]
            # Arrow direction
            arrow_dx = 0
            arrow_dy = 0.3e-6 if self.direction == "+y" else -0.3e-6

        # Draw the source line (thick colored line)
        ax.plot(
            line_x,
            line_y,
            color=edgecolor,
            linewidth=3,
            alpha=alpha,
            solid_capstyle="round",
            label="ModeSource",
        )

        # Draw direction arrow from center
        arrow_length = self.wavelength * 0.5 if hasattr(self, "wavelength") else 0.5e-6
        if self.direction in ["+x", "-x"]:
            arrow_dx = arrow_length if self.direction == "+x" else -arrow_length
            arrow_dy = 0
        else:
            arrow_dx = 0
            arrow_dy = arrow_length if self.direction == "+y" else -arrow_length

        arrow = FancyArrowPatch(
            (x_pos, y_pos),
            (x_pos + arrow_dx, y_pos + arrow_dy),
            arrowstyle="-|>",
            mutation_scale=10,
            color=edgecolor,
            linewidth=2,
            alpha=alpha,
        )
        ax.add_patch(arrow)
