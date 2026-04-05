import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.devices.sources.inject import (
    _get_3d_huygens_terms,
    _inject_3d_e_fields,
    _inject_3d_fields,
    _inject_3d_h_fields,
    _inject_e_component,
    _inject_h_component,
    _match_shape,
)
from beamz.devices.sources.profiles import (
    _axis_index_from_component_indices,
    _backward_3d_mode_from_forward,
    _build_3d_profiles,
    _component_axis_coord,
    _dominant_3d_pair,
    _impedance_match_3d_tangential_pairs,
    _impedance_match_e_profile,
    _make_3d_mode_basis_profiles,
    _modal_overlap_3d_profiles,
    _modal_power_2d,
    _modal_power_3d_from_profiles,
    _normalize_2d_pair_by_power,
    _normalize_3d_profiles_by_flux,
    _numeric_impedance_axis,
    _numeric_phase_delay,
    _parse_direction,
    _project_3d_profiles_to_real,
    _remap_3d_solver_components,
    _select_3d_impedance_index,
    _select_3d_phase_ref,
    _select_core_confined_mode_index,
    _solve_numeric_k_axis,
    _to_real_profile,
)
from beamz.devices.sources.solve import solve_modes
from beamz.devices.sources.windows import (
    _compute_transverse_bounds,
    _crop_and_window_2d,
    _jax_tukey_window,
    _make_tukey_window_2d,
    _scipy_tukey,
    _stagger_both,
    _stagger_half,
)


# ---------------------------------------------------------------------------
# ModeSource class
# ---------------------------------------------------------------------------


class ModeSource:
    """Huygens mode source on Yee grid supporting ±x/±y in 2D and ±x/±y/±z in 3D.

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
        self.pol = str(pol).lower()
        if self.pol not in {"te", "tm"}:
            raise ValueError(f"pol must be 'te' or 'tm', got {pol!r}")
        self.signal = signal
        self.direction, self._direction_axis, self._direction_sign = _parse_direction(
            direction
        )

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
        self._impedance_neff = None
        self._dt_physical = 0.0
        self._launch_dt = None
        self._initialized = False

    def initialize(self, permittivity, resolution, dt=None):
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
            dz = resolution
            self._grid_shape = (ny, nx)
            self.height = None

        axis = self._direction_axis
        if (not is_3d) and axis == "z":
            raise ValueError(
                "direction '+z'/'-z' requires a 3D permittivity grid; received 2D data"
            )
        self._axis = axis
        self._dt_physical = 0.0
        self._launch_dt = dt

        # 1. Get center index for injection plane
        if axis == "x":
            center_idx = int(np.clip(np.round(self.center[0] / dx - 0.5), 0, nx - 1))
            if self.direction == "+x":
                offset_idx = max(0, center_idx - 1)
            else:
                offset_idx = min(nx - 2, center_idx + 1)

            if is_3d:
                eps_profile = permittivity[:, :, center_idx]
                self._eps_profile_2d = eps_profile
            else:
                eps_profile = permittivity[:, center_idx]
                self._eps_profile_2d = None

        elif axis == "y":
            center_idx = int(np.clip(np.round(self.center[1] / dy - 0.5), 0, ny - 1))
            if self.direction == "+y":
                offset_idx = max(0, center_idx - 1)
            else:
                offset_idx = min(ny - 2, center_idx + 1)

            if is_3d:
                eps_profile = permittivity[:, center_idx, :]
                self._eps_profile_2d = eps_profile
            else:
                eps_profile = permittivity[center_idx, :]
                self._eps_profile_2d = None

        else:  # axis == "z" (3D only)
            center_idx = int(np.clip(np.round(self.center[2] / dz - 0.5), 0, nz - 1))
            if self.direction == "+z":
                offset_idx = max(0, center_idx - 1)
            else:
                offset_idx = min(nz - 2, center_idx + 1)

            eps_profile = permittivity[center_idx, :, :]
            self._eps_profile_2d = eps_profile

        # 2. Solve for mode fields
        omega = 2 * np.pi * LIGHT_SPEED / self.wavelength
        dL = dz if is_3d else (dy if axis == "x" else dx)
        solver_direction = self.direction
        if is_3d and axis in {"x", "y"}:
            # 3D x/y cross-sections are solved in a rotated basis; flip +/- here so
            # the resulting mode phase/gauge matches the 2D-corrected launch direction.
            solver_direction = ("-" if self.direction.startswith("+") else "+") + axis

        eps_profile_arr = np.asarray(eps_profile)
        n_local_max = float(
            np.sqrt(max(float(np.max(np.real(eps_profile_arr))), 1e-12))
        )
        # Bias the solver toward the guided branch in large windows where cladding-like
        # continuum modes can otherwise dominate the sort order.
        target_neff = 0.98 * n_local_max

        mode_candidates = 3
        try:
            neff_val, e_fields, h_fields, _ = solve_modes(
                eps=eps_profile,
                omega=omega,
                dL=dL,
                m=mode_candidates,
                direction=solver_direction,
                filter_pol=self.pol,
                target_neff=target_neff,
                return_fields=True,
            )
        except ValueError:
            neff_val, e_fields, h_fields, _ = solve_modes(
                eps=eps_profile,
                omega=omega,
                dL=dL,
                m=1,
                direction=solver_direction,
                filter_pol=self.pol,
                target_neff=target_neff,
                return_fields=True,
            )

        mode_idx = _select_core_confined_mode_index(eps_profile, e_fields, neff_val)
        self._neff = neff_val[mode_idx]
        E_mode = e_fields[mode_idx]
        H_mode = h_fields[mode_idx]

        # 3. Extract all 6 components and convert to JAX arrays
        Ex_raw = jnp.asarray(jnp.squeeze(E_mode[0]))
        Ey_raw = jnp.asarray(jnp.squeeze(E_mode[1]))
        Ez_raw = jnp.asarray(jnp.squeeze(E_mode[2]))
        Hx_raw = jnp.asarray(jnp.squeeze(H_mode[0]))
        Hy_raw = jnp.asarray(jnp.squeeze(H_mode[1]))
        Hz_raw = jnp.asarray(jnp.squeeze(H_mode[2]))

        if is_3d:
            Ex_raw, Ey_raw, Ez_raw, Hx_raw, Hy_raw, Hz_raw = (
                _remap_3d_solver_components(
                    Ex_raw, Ey_raw, Ez_raw, Hx_raw, Hy_raw, Hz_raw, axis
                )
            )

        # 4. Phase-align all components with the same H-based gauge convention as 2D.
        if is_3d:
            ref_field = _select_3d_phase_ref(
                axis, self.pol, Ex_raw, Ey_raw, Ez_raw, Hx_raw, Hy_raw, Hz_raw
            )
        elif self.pol == "tm":
            ex_max = jnp.max(jnp.abs(Ex_raw))
            ey_max = jnp.max(jnp.abs(Ey_raw))
            ez_max = jnp.max(jnp.abs(Ez_raw))
            ref_field = jnp.where(
                ex_max > ey_max,
                jnp.where(ex_max > ez_max, Ex_raw, Ez_raw),
                jnp.where(ey_max > ez_max, Ey_raw, Ez_raw),
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
            self._impedance_neff = _select_3d_impedance_index(
                axis,
                self.pol,
                self._eps_profile_2d,
                Ex_aligned,
                Ey_aligned,
                Ez_aligned,
                Hx_aligned,
                Hy_aligned,
                Hz_aligned,
            )
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
                omega=omega,
                dt=dt,
            )
        else:
            self._impedance_neff = None
            self._setup_2d_injection(
                E_mode, H_mode, center_idx, offset_idx, axis, ny, nx, resolution
            )

        self._compute_dt_physical(axis, is_3d, dx, dy, dz, dt=dt)
        self._initialized = True

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
        omega,
        dt,
    ):
        """Set up full 6-component injection for 3D simulations."""
        profiles, indices, extra = _build_3d_profiles(
            Ex,
            Ey,
            Ez,
            Hx,
            Hy,
            Hz,
            axis=axis,
            direction=self.direction,
            center=self.center,
            width=self.width,
            height=self.height,
            center_idx=center_idx,
            offset_idx=offset_idx,
            grid_shape=(nz, ny, nx),
            resolution=resolution,
            impedance_neff=(
                self._impedance_neff if self._impedance_neff is not None else self._neff
            ),
            omega=omega,
            dt=dt,
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
        """2D injection setup using explicit global component mapping.

        `solve_modes(..., return_fields=True)` returns fields ordered as:
        E_mode = [Ex, Ey, Ez], H_mode = [Hx, Hy, Hz] in global components.
        We pick the physically matching TE/TM pair for the chosen propagation axis.
        """
        dir_sign = 1.0 if self.direction.startswith("+") else -1.0
        ETA_0 = np.sqrt(MU_0 / EPS_0)
        z_target = ETA_0 / max(np.real(self._neff), 1e-6)

        if axis == "x":
            self._setup_2d_x(
                E_mode,
                H_mode,
                center_idx,
                offset_idx,
                ny,
                nx,
                resolution,
                dir_sign,
                z_target,
            )
        else:
            self._setup_2d_y(
                E_mode,
                H_mode,
                center_idx,
                offset_idx,
                ny,
                nx,
                resolution,
                dir_sign,
                z_target,
            )

    def _setup_2d_x(
        self,
        E_mode,
        H_mode,
        center_idx,
        offset_idx,
        ny,
        nx,
        resolution,
        dir_sign,
        z_target,
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

            # +x TM: (Ez, Hy)
            Hy_raw = np.squeeze(H_mode[1])
            Ez_raw = np.squeeze(E_mode[2])

            idx_max = np.argmax(np.abs(Hy_raw))
            phase_ref = np.angle(Hy_raw.flatten()[idx_max])
            Hy_profile = Hy_raw * np.exp(-1j * phase_ref)
            Ez_profile = Ez_raw * np.exp(-1j * phase_ref)
            Ez_profile = _impedance_match_e_profile(Ez_profile, Hy_profile, z_target)

            width_cells = y_end - y_start
            window = self._make_1d_window(width_cells)

            Hy_cropped = Hy_profile[y_start:y_end]
            Ez_cropped = Ez_profile[y_start:y_end]
            if len(Hy_cropped) == len(window):
                Hy_cropped = Hy_cropped * window
                Ez_cropped = Ez_cropped * window

            jz_profile = dir_sign * Hy_cropped
            my_profile = dir_sign * Ez_cropped
            jz_profile, my_profile = _normalize_2d_pair_by_power(
                jz_profile, my_profile, signed_flux_sign=-1.0, dl=resolution
            )
            jz_profile = _to_real_profile(jz_profile)
            my_profile = _to_real_profile(my_profile)
            jz_profile, my_profile = _normalize_2d_pair_by_power(
                jz_profile, my_profile, signed_flux_sign=-1.0, dl=resolution
            )

            self._jz_profile = jz_profile
            self._my_profile = my_profile

        else:  # TE
            hz_col = (
                max(0, offset_idx - 1)
                if self.direction == "+x"
                else min(nx - 2, offset_idx)
            )

            self._hz_indices = (slice(y_start, min(y_end, ny - 1)), hz_col)
            self._e_indices = (slice(y_start, min(y_end, ny - 1)), offset_idx)
            self._e_component = "Ey"

            # +x TE: (Ey, Hz)
            Hz_raw = np.squeeze(H_mode[2])
            Ey_raw = np.squeeze(E_mode[1])

            Hz_staggered = 0.5 * (Hz_raw[:-1] + Hz_raw[1:])
            Ey_staggered = 0.5 * (Ey_raw[:-1] + Ey_raw[1:])

            idx_max = np.argmax(np.abs(Hz_staggered))
            phase_ref = np.angle(Hz_staggered.flatten()[idx_max])
            Hz_profile = Hz_staggered * np.exp(-1j * phase_ref)
            Ey_profile = Ey_staggered * np.exp(-1j * phase_ref)
            Ey_profile = _impedance_match_e_profile(Ey_profile, Hz_profile, z_target)

            width_cells = min(y_end, len(Hz_profile)) - y_start
            window = self._make_1d_window(width_cells)

            Hz_cropped = Hz_profile[y_start : min(y_end, len(Hz_profile))]
            Ey_cropped = Ey_profile[y_start : min(y_end, len(Ey_profile))]
            if len(Hz_cropped) == len(window):
                Hz_cropped = Hz_cropped * window
                Ey_cropped = Ey_cropped * window

            # Relative J/M sign controls propagation handedness for TE in x-propagation.
            jy_profile = dir_sign * Hz_cropped
            mz_profile = -dir_sign * Ey_cropped
            jy_profile, mz_profile = _normalize_2d_pair_by_power(
                jy_profile, mz_profile, signed_flux_sign=1.0, dl=resolution
            )
            jy_profile = _to_real_profile(jy_profile)
            mz_profile = _to_real_profile(mz_profile)
            jy_profile, mz_profile = _normalize_2d_pair_by_power(
                jy_profile, mz_profile, signed_flux_sign=1.0, dl=resolution
            )

            self._jy_profile = jy_profile
            self._mz_profile = mz_profile

    def _setup_2d_y(
        self,
        E_mode,
        H_mode,
        center_idx,
        offset_idx,
        ny,
        nx,
        resolution,
        dir_sign,
        z_target,
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

            # +y TM uses the rotated x-solver basis: (Ez, Hx) <- (Ez, Hy_xbasis)
            Hx_raw = np.squeeze(H_mode[1])
            Ez_raw = np.squeeze(E_mode[2])

            idx_max = np.argmax(np.abs(Hx_raw))
            phase_ref = np.angle(Hx_raw.flatten()[idx_max])
            Hx_profile = Hx_raw * np.exp(-1j * phase_ref)
            Ez_profile = Ez_raw * np.exp(-1j * phase_ref)
            Ez_profile = _impedance_match_e_profile(Ez_profile, Hx_profile, z_target)

            width_cells = x_end - x_start
            window = self._make_1d_window(width_cells)

            Hx_cropped = Hx_profile[x_start:x_end]
            Ez_cropped = Ez_profile[x_start:x_end]
            if len(Hx_cropped) == len(window):
                Hx_cropped = Hx_cropped * window
                Ez_cropped = Ez_cropped * window

            # Relative J/M sign controls propagation handedness for TM in y-propagation.
            jz_profile = dir_sign * Hx_cropped
            my_profile = -dir_sign * Ez_cropped
            jz_profile, my_profile = _normalize_2d_pair_by_power(
                jz_profile, my_profile, signed_flux_sign=1.0, dl=resolution
            )
            jz_profile = _to_real_profile(jz_profile)
            my_profile = _to_real_profile(my_profile)
            jz_profile, my_profile = _normalize_2d_pair_by_power(
                jz_profile, my_profile, signed_flux_sign=1.0, dl=resolution
            )

            self._jz_profile = jz_profile
            self._my_profile = my_profile

        else:  # TE y-prop
            hz_row = (
                max(0, offset_idx - 1)
                if self.direction == "+y"
                else min(ny - 2, offset_idx)
            )

            self._hz_indices = (hz_row, slice(x_start, min(x_end, nx - 1)))
            self._e_indices = (offset_idx, slice(x_start, min(x_end, nx - 1)))
            self._e_component = "Ex"

            # +y TE uses the rotated x-solver basis: (Ex, Hz) <- (Ey_xbasis, Hz_xbasis)
            Hz_raw = np.squeeze(H_mode[2])
            Ex_raw = np.squeeze(E_mode[1])

            Hz_staggered = 0.5 * (Hz_raw[:-1] + Hz_raw[1:])
            Ex_staggered = 0.5 * (Ex_raw[:-1] + Ex_raw[1:])

            idx_max = np.argmax(np.abs(Hz_staggered))
            phase_ref = np.angle(Hz_staggered.flatten()[idx_max])
            Hz_profile = Hz_staggered * np.exp(-1j * phase_ref)
            Ex_profile = Ex_staggered * np.exp(-1j * phase_ref)
            Ex_profile = _impedance_match_e_profile(Ex_profile, Hz_profile, z_target)

            width_cells = min(x_end, len(Hz_profile)) - x_start
            window = self._make_1d_window(width_cells)

            Hz_cropped = Hz_profile[x_start : min(x_end, len(Hz_profile))]
            Ex_cropped = Ex_profile[x_start : min(x_end, len(Ex_profile))]
            if len(Hz_cropped) == len(window):
                Hz_cropped = Hz_cropped * window
                Ex_cropped = Ex_cropped * window

            jx_profile = -dir_sign * Hz_cropped
            mz_profile = -dir_sign * Ex_cropped
            jx_profile, mz_profile = _normalize_2d_pair_by_power(
                jx_profile, mz_profile, signed_flux_sign=-1.0, dl=resolution
            )
            jx_profile = _to_real_profile(jx_profile)
            mz_profile = _to_real_profile(mz_profile)
            jx_profile, mz_profile = _normalize_2d_pair_by_power(
                jx_profile, mz_profile, signed_flux_sign=-1.0, dl=resolution
            )

            self._jx_profile = jx_profile
            self._mz_profile = mz_profile

    @staticmethod
    def _make_1d_window(width_cells, alpha=0.3):
        """Create a 1D Tukey window for smooth edges."""
        if width_cells > 2:
            from scipy.signal.windows import tukey

            return tukey(width_cells, alpha=alpha)
        return np.ones(max(1, width_cells))

    def _compute_dt_physical(self, axis, is_3d, dx, dy, dz=None, dt=None):
        """Compute physical time shift between E and H injection planes."""
        if self._neff is None:
            return
        if dz is None:
            dz = dx

        coord_e = 0.0
        coord_h = 0.0

        if is_3d:
            e_comp, h_comp = _dominant_3d_pair(axis, self.pol)
            e_indices = getattr(self, f"_{e_comp}_indices", None)
            h_indices = getattr(self, f"_{h_comp}_indices", None)
            e_axis_idx = _axis_index_from_component_indices(e_indices, axis)
            h_axis_idx = _axis_index_from_component_indices(h_indices, axis)
            coord_e = _component_axis_coord(e_comp, e_axis_idx, axis, dx, dy, dz)
            coord_h = _component_axis_coord(h_comp, h_axis_idx, axis, dx, dy, dz)
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

        delta_s = float(coord_e - coord_h)
        if is_3d and dt is not None:
            omega = 2 * np.pi * LIGHT_SPEED / self.wavelength
            d_axis = {"x": dx, "y": dy, "z": dz}[axis]
            k_num = _solve_numeric_k_axis(omega, dt, d_axis, self._neff)
            self._dt_physical = _numeric_phase_delay(omega, k_num, delta_s)
        else:
            self._dt_physical = delta_s * float(np.real(self._neff)) / LIGHT_SPEED

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

    def inject_h(self, fields, t, dt, current_step, resolution, design):
        """Inject magnetic current (M) into H-fields after the H update."""
        needs_reinit = (
            (not self._initialized)
            or (self._grid_shape != fields.permittivity.shape)
            or (self._resolution is None)
            or (not np.isclose(self._resolution, resolution))
        )
        if (
            (not needs_reinit)
            and getattr(self, "_is_3d", False)
            and ((self._launch_dt is None) or (not np.isclose(self._launch_dt, dt)))
        ):
            needs_reinit = True
        if needs_reinit:
            self.initialize(fields.permittivity, resolution, dt=dt)

        # M=-n×E is injected on the H update at the standard half-step time.
        signal_value_h = self._get_signal_value(t + 0.5 * dt, dt)

        if self._Ex_profile is not None and self._is_3d:
            self._inject_3d_h(fields, signal_value_h, dt, resolution)
        else:
            self._inject_2d_h(fields, signal_value_h, dt, resolution)

    def inject_e(self, fields, t, dt, current_step, resolution, design):
        """Inject electric current (J) into E-fields after the E update."""
        needs_reinit = (
            (not self._initialized)
            or (self._grid_shape != fields.permittivity.shape)
            or (self._resolution is None)
            or (not np.isclose(self._resolution, resolution))
        )
        if (
            (not needs_reinit)
            and getattr(self, "_is_3d", False)
            and ((self._launch_dt is None) or (not np.isclose(self._launch_dt, dt)))
        ):
            needs_reinit = True
        if needs_reinit:
            self.initialize(fields.permittivity, resolution, dt=dt)

        # J=n×H is evaluated on the E update and needs the physical E/H plane offset.
        # Keep E/H drive samples on the same temporal convention and only apply
        # the physical E/H plane offset correction via _dt_physical.
        signal_time_e = t + 0.5 * dt + self._dt_physical
        signal_value_e = self._get_signal_value(signal_time_e, dt)

        if self._Ex_profile is not None and self._is_3d:
            self._inject_3d_e(fields, signal_value_e, dt, resolution)
        else:
            self._inject_2d_e(fields, signal_value_e, dt, resolution)

    def inject(self, fields, t, dt, current_step, resolution, design):
        """Inject source fields (calls inject_h + inject_e for backward compatibility)."""
        self.inject_h(fields, t, dt, current_step, resolution, design)
        self.inject_e(fields, t, dt, current_step, resolution, design)

    # -- 3D injection (split) ------------------------------------------

    def _get_3d_profiles_and_indices(self):
        profiles = {
            "Ex": self._Ex_profile,
            "Ey": self._Ey_profile,
            "Ez": self._Ez_profile,
            "Hx": self._Hx_profile,
            "Hy": self._Hy_profile,
            "Hz": self._Hz_profile,
        }
        indices = {
            "Ex": self._Ex_indices,
            "Ey": self._Ey_indices,
            "Ez": self._Ez_indices,
            "Hx": self._Hx_indices,
            "Hy": self._Hy_indices,
            "Hz": self._Hz_indices,
        }
        return profiles, indices

    def _inject_3d_h(self, fields, signal_h, dt, resolution):
        """Inject H-field components for 3D Huygens source."""
        profiles, indices = self._get_3d_profiles_and_indices()
        _inject_3d_h_fields(
            fields, profiles, indices, signal_h, dt, resolution, self._axis, self.pol
        )

    def _inject_3d_e(self, fields, signal_e, dt, resolution):
        """Inject E-field components for 3D Huygens source."""
        profiles, indices = self._get_3d_profiles_and_indices()
        _inject_3d_e_fields(
            fields, profiles, indices, signal_e, dt, resolution, self._axis, self.pol
        )

    # -- 2D injection (split, with corrected signs) ---------------------

    def _inject_2d_h(self, fields, signal_h, dt, resolution):
        """Inject magnetic current into H-fields for 2D (after H update)."""
        if self.pol == "tm":
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
            if self._hz_indices is not None and self._mz_profile is not None:
                mu_val = getattr(fields, "permeability", None)
                mu_at_source = mu_val[self._hz_indices] if mu_val is not None else 1.0
                mz_term = self._mz_profile * signal_h / resolution
                hz_injection = +mz_term * dt / (MU_0 * mu_at_source)
                fields.Hz = fields.Hz.at[self._hz_indices].add(hz_injection)

    def _inject_2d_e(self, fields, signal_e, dt, resolution):
        """Inject electric current into E-fields for 2D (after E update)."""
        if self.pol == "tm":
            if self._ez_indices is not None and self._jz_profile is not None:
                eps_at_source = fields.permittivity[self._ez_indices]
                jz_term = self._jz_profile * signal_e / resolution
                ez_injection = +jz_term * dt / (EPS_0 * eps_at_source)
                fields.Ez = fields.Ez.at[self._ez_indices].add(ez_injection)
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
            self,
            ax,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linestyle=linestyle,
        )
