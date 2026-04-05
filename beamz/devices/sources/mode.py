import warnings

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

    # -- 2D injection (split, with corrected signs) ---------------------

    def _inject_2d_h(self, fields, signal_h, dt, resolution):
        """Inject magnetic current into H-fields for 2D (after H update)."""
        apply_helpers.inject_2d_h(self, fields, signal_h, dt, resolution)

    def _inject_2d_e(self, fields, signal_e, dt, resolution):
        """Inject electric current into E-fields for 2D (after E update)."""
        apply_helpers.inject_2d_e(self, fields, signal_e, dt, resolution)

    def profile_data(self, field=None):
        """Return the current source profile as plotting-ready data."""
        from beamz.visual.data import Slice2D, Trace1D

        if self._Ez_profile is None and self._jz_profile is None:
            if self.grid is not None and hasattr(self.grid, "permittivity"):
                resolution = getattr(self.grid, "resolution", 0.05e-6)
                self.initialize(self.grid.permittivity, resolution)
            else:
                raise ValueError(
                    "Mode source is not initialized and no grid permittivity is available."
                )

        choices = {
            "ez": ("Ez", self._Ez_profile),
            "hz": ("Hz", self._jz_profile),
            "jz": ("Hz", self._jz_profile),
        }
        key = None if field is None else str(field).strip().lower()
        if key is None:
            label, profile = ("Ez", self._Ez_profile)
            if profile is None:
                label, profile = ("Hz", self._jz_profile)
        elif key in choices:
            label, profile = choices[key]
        else:
            raise ValueError("field must be one of None, 'Ez', 'Hz', or 'Jz'.")

        if profile is None:
            raise ValueError(f"No profile data available for field '{field}'.")

        profile = np.squeeze(np.asarray(profile))
        title = f"{label} mode profile"
        if self._neff is not None:
            title = f"{title} (neff={self._neff:.4f})"

        if profile.ndim == 2:
            if self.direction in {"+x", "-x"}:
                plane, x_label, y_label = "yz", "Y index", "Z index"
            else:
                plane, x_label, y_label = "xz", "X index", "Z index"
            height, width = profile.shape
            return Slice2D(
                values=profile,
                extent=(0.0, float(max(width - 1, 1)), 0.0, float(max(height - 1, 1))),
                value_label="Amplitude",
                plane=plane,
                title=title,
                x_label=x_label,
                y_label=y_label,
            )

        return Trace1D(
            values=profile.reshape(-1),
            coords=np.arange(profile.size, dtype=float),
            coord_label="index",
            value_label="Amplitude",
            title=title,
        )

    def show(self, field=None):
        """Visualize the 2D mode profile (for 3D simulations) or 1D profile (for 2D)."""
        import matplotlib.pyplot as plt
        from beamz.visual.data import Slice2D

        try:
            plot_data = self.profile_data(field=field)
        except ValueError as exc:
            warnings.warn(str(exc), stacklevel=2)
            return None
        if isinstance(plot_data, Slice2D):
            plot_data.plot(cmap="magma", abs_value=True, aspect="auto")
        else:
            ax = plot_data.plot(color="k", abs_value=True)
            ax.grid(True)
        plt.tight_layout()
        plt.show()
        return plot_data

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
