from dataclasses import replace

from beamz.devices.sources import apply as apply_helpers
from beamz.devices.sources import mode_visual
from beamz.devices.sources.profiles_common import (
    _axis_index_from_component_indices,
    _component_axis_coord,
    _dominant_3d_pair,
    _impedance_match_3d_tangential_pairs,
    _impedance_match_e_profile,
    _numeric_impedance_axis,
    _numeric_phase_delay,
    _parse_direction,
    _remap_3d_solver_components,
    _select_3d_impedance_index,
    _select_3d_phase_ref,
    _select_core_confined_mode_index,
    _solve_numeric_k_axis,
)
from beamz.devices.sources.profiles_basis import (
    _backward_3d_mode_from_forward,
    _make_3d_mode_basis_profiles,
    _modal_overlap_3d_profiles,
    _modal_power_2d,
    _modal_power_3d_from_profiles,
    _normalize_2d_pair_by_power,
    _normalize_3d_profiles_by_flux,
    _project_3d_profiles_to_real,
)
from beamz.devices.sources import setup as setup_helpers
from beamz.devices.sources.spec import ModeSourceSpec, build_mode_source_spec
from beamz.devices.sources.state import ModeSourceState
from beamz.devices.sources.solve import solve_modes


# ---------------------------------------------------------------------------
# ModeSource class
# ---------------------------------------------------------------------------


_MODE_SPEC_FIELDS = frozenset(ModeSourceSpec.__dataclass_fields__.keys())
_MODE_SPEC_MAP = {
    "_direction_axis": "direction_axis",
    "_direction_sign": "direction_sign",
}
_MODE_STATE_MAP = {
    "_Ex_profile": "Ex_profile",
    "_Ey_profile": "Ey_profile",
    "_Ez_profile": "Ez_profile",
    "_Hx_profile": "Hx_profile",
    "_Hy_profile": "Hy_profile",
    "_Hz_profile": "Hz_profile",
    "_Ex_indices": "Ex_indices",
    "_Ey_indices": "Ey_indices",
    "_Ez_indices": "Ez_indices",
    "_Hx_indices": "Hx_indices",
    "_Hy_indices": "Hy_indices",
    "_Hz_indices": "Hz_indices",
    "_jz_profile": "jz_profile",
    "_my_profile": "my_profile",
    "_mz_profile": "mz_profile",
    "_jy_profile": "jy_profile",
    "_jx_profile": "jx_profile",
    "_ez_indices": "ez_indices",
    "_h_indices": "h_indices",
    "_hz_indices": "hz_indices",
    "_e_indices": "e_indices",
    "_h_component": "h_component",
    "_e_component": "e_component",
    "_neff": "neff",
    "_impedance_neff": "impedance_neff",
    "_dt_physical": "dt_physical",
    "_launch_dt": "launch_dt",
    "_initialized": "initialized",
    "_resolution": "resolution",
    "_is_3d": "is_3d",
    "_grid_shape": "grid_shape",
    "_eps_profile_2d": "eps_profile_2d",
    "_axis": "axis",
    "_transverse_start": "transverse_start",
    "_transverse_end": "transverse_end",
    "_x_start": "x_start",
    "_x_end": "x_end",
    "_y_start": "y_start",
    "_y_end": "y_end",
    "_z_start": "z_start",
    "_z_end": "z_end",
}


class ModeSource:
    """Huygens mode source on Yee grid supporting ±x/±y in 2D and ±x/±y/±z in 3D.

    In 3D, injects all 6 field components (Ex, Ey, Ez, Hx, Hy, Hz) for accurate
    mode injection, accounting for proper Yee grid staggering.
    """

    def __init__(
        self, grid, center, width, wavelength, pol, signal, direction="+x", height=None
    ):
        object.__setattr__(self, "grid", grid)
        object.__setattr__(
            self,
            "spec",
            build_mode_source_spec(
                grid=grid,
                center=center,
                width=width,
                wavelength=wavelength,
                pol=pol,
                signal=signal,
                direction=direction,
                height=height,
            ),
        )
        object.__setattr__(self, "state", ModeSourceState())

    def __getattr__(self, name):
        spec = self.__dict__.get("spec")
        if spec is not None:
            mapped = _MODE_SPEC_MAP.get(name, name)
            if hasattr(spec, mapped):
                return getattr(spec, mapped)
        state = self.__dict__.get("state")
        if state is not None and name in _MODE_STATE_MAP:
            return getattr(state, _MODE_STATE_MAP[name])
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in {"grid", "spec", "state"}:
            object.__setattr__(self, name, value)
            return
        if name in _MODE_SPEC_MAP and "spec" in self.__dict__:
            object.__setattr__(self, "spec", replace(self.spec, **{_MODE_SPEC_MAP[name]: value}))
            return
        if name in _MODE_SPEC_FIELDS and "spec" in self.__dict__:
            object.__setattr__(self, "spec", replace(self.spec, **{name: value}))
            return
        if name in _MODE_STATE_MAP and "state" in self.__dict__:
            setattr(self.state, _MODE_STATE_MAP[name], value)
            return
        object.__setattr__(self, name, value)

    def with_spec(self, spec=None, /, **changes):
        base_spec = self.spec if spec is None else spec
        if not isinstance(base_spec, ModeSourceSpec):
            raise TypeError("with_spec expects a ModeSourceSpec or spec field updates")
        if changes:
            base_spec = replace(base_spec, **changes)
        new = object.__new__(type(self))
        object.__setattr__(new, "grid", self.grid)
        object.__setattr__(new, "spec", base_spec)
        object.__setattr__(new, "state", ModeSourceState())
        return new

    def to_dict(self):
        return self.spec.to_dict()

    @classmethod
    def from_dict(cls, data, *, grid=None):
        return cls.from_spec(ModeSourceSpec.from_dict(data), grid=grid)

    @classmethod
    def from_spec(cls, spec, *, grid=None):
        if not isinstance(spec, ModeSourceSpec):
            raise TypeError("from_spec expects a ModeSourceSpec")
        new = object.__new__(cls)
        object.__setattr__(new, "grid", grid)
        object.__setattr__(new, "spec", spec)
        object.__setattr__(new, "state", ModeSourceState())
        return new

    def initialize(self, permittivity, resolution, dt=None):
        """Compute the mode and set up the source currents for all 6 components in 3D."""
        setup_helpers.initialize_mode_state(
            self.spec,
            self.state,
            permittivity,
            resolution,
            dt=dt,
        )

    def _compute_dt_physical(self, axis, is_3d, dx, dy, dz=None, dt=None):
        """Compute physical time shift between E and H injection planes."""
        return setup_helpers.compute_dt_physical(
            self, axis, is_3d, dx, dy, dz=dz, dt=dt
        )

    def _get_signal_value(self, time, dt):
        """Interpolate signal value at arbitrary time."""
        return setup_helpers.sample_signal(self.spec, time, dt)

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
        return mode_visual.profile_data(self, field=field)

    def show(self, field=None):
        """Visualize the 2D mode profile (for 3D simulations) or 1D profile (for 2D)."""
        return mode_visual.show(self, field=field)

    def add_to_plot(
        self, ax, facecolor="none", edgecolor="crimson", alpha=0.8, linestyle="-"
    ):
        """Add source visualization to 2D matplotlib plot."""
        mode_visual.add_to_plot(
            self,
            ax,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linestyle=linestyle,
        )
