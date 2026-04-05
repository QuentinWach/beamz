import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.devices.sources import apply as apply_helpers
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
from beamz.devices.sources import setup as setup_helpers
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
        setup_helpers.initialize(self, permittivity, resolution, dt=dt)

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
        setup_helpers.setup_3d(
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
            omega=omega,
            dt=dt,
        )

    def _setup_2d_injection(
        self, E_mode, H_mode, center_idx, offset_idx, axis, ny, nx, resolution
    ):
        """2D injection setup using explicit global component mapping.

        `solve_modes(..., return_fields=True)` returns fields ordered as:
        E_mode = [Ex, Ey, Ez], H_mode = [Hx, Hy, Hz] in global components.
        We pick the physically matching TE/TM pair for the chosen propagation axis.
        """
        setup_helpers.setup_2d(
            self, E_mode, H_mode, center_idx, offset_idx, axis, ny, nx, resolution
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
        setup_helpers.setup_2d_x(
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
        )

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
        setup_helpers.setup_2d_y(
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
        )

    @staticmethod
    def _make_1d_window(width_cells, alpha=0.3):
        """Create a 1D Tukey window for smooth edges."""
        return setup_helpers.make_1d_window(width_cells, alpha=alpha)

    def _compute_dt_physical(self, axis, is_3d, dx, dy, dz=None, dt=None):
        """Compute physical time shift between E and H injection planes."""
        return setup_helpers.compute_dt_physical(
            self, axis, is_3d, dx, dy, dz=dz, dt=dt
        )

    def _get_signal_value(self, time, dt):
        """Interpolate signal value at arbitrary time."""
        return apply_helpers.get_signal_value(self, time, dt)

    def inject_h(self, fields, t, dt, current_step, resolution, design):
        """Inject magnetic current (M) into H-fields after the H update."""
        del current_step, design
        apply_helpers.inject_h(self, fields, t, dt, resolution)

    def inject_e(self, fields, t, dt, current_step, resolution, design):
        """Inject electric current (J) into E-fields after the E update."""
        del current_step, design
        apply_helpers.inject_e(self, fields, t, dt, resolution)

    def inject(self, fields, t, dt, current_step, resolution, design):
        """Inject source fields (calls inject_h + inject_e for backward compatibility)."""
        apply_helpers.inject(self, fields, t, dt, current_step, resolution, design)

    # -- 3D injection (split) ------------------------------------------

    def _get_3d_profiles_and_indices(self):
        return apply_helpers.get_3d_profiles_and_indices(self)

    def _inject_3d_h(self, fields, signal_h, dt, resolution):
        """Inject H-field components for 3D Huygens source."""
        apply_helpers.inject_3d_h(self, fields, signal_h, dt, resolution)

    def _inject_3d_e(self, fields, signal_e, dt, resolution):
        """Inject E-field components for 3D Huygens source."""
        apply_helpers.inject_3d_e(self, fields, signal_e, dt, resolution)

    # -- 2D injection (split, with corrected signs) ---------------------

    def _inject_2d_h(self, fields, signal_h, dt, resolution):
        """Inject magnetic current into H-fields for 2D (after H update)."""
        apply_helpers.inject_2d_h(self, fields, signal_h, dt, resolution)

    def _inject_2d_e(self, fields, signal_e, dt, resolution):
        """Inject electric current into E-fields for 2D (after E update)."""
        apply_helpers.inject_2d_e(self, fields, signal_e, dt, resolution)

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
