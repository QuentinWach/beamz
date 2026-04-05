from dataclasses import replace
from typing import Callable, Optional

import numpy as np

from beamz.devices.monitors import dft as dft_helpers
from beamz.devices.monitors import geom as geom_helpers
from beamz.devices.monitors import live as live_helpers
from beamz.devices.monitors import record as record_helpers
from beamz.devices.monitors import store as store_helpers
from beamz.devices.monitors.spec import MonitorSpec, build_monitor_spec
from beamz.devices.monitors.state import MonitorRecorder, create_monitor_state


_SPEC_FIELDS = frozenset(MonitorSpec.__dataclass_fields__.keys())
_STATE_FIELDS = frozenset(MonitorRecorder.__dataclass_fields__.keys())


class Monitor:
    def __init__(
        self,
        design=None,
        start=(0, 0),
        end=None,
        plane_normal=None,
        plane_position=0,
        size=None,
        record_fields=True,
        accumulate_power=True,
        live_update=False,
        record_interval=1,
        max_history_steps=None,
        dft_frequencies=None,
        dft_t_start=0.0,
        dft_t_end=None,
        dft_enabled=False,
        dft_components=None,
        dft_record_every_step=True,
        dft_record_interval=None,
        dft_window="rect",
        objective_function: Optional[Callable[["Monitor"], float]] = None,
        name: Optional[str] = None,
        frequency_points=None,
        frequency_record_interval=1,
    ):
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "_live_state", live_helpers.create_state())
        object.__setattr__(self, "update_interval", 10)
        object.__setattr__(self, "objective_function", objective_function)
        object.__setattr__(
            self,
            "spec",
            build_monitor_spec(
                design=design,
                start=start,
                end=end,
                plane_normal=plane_normal,
                plane_position=plane_position,
                size=size,
                record_fields=record_fields,
                accumulate_power=accumulate_power,
                live_update=live_update,
                record_interval=record_interval,
                max_history_steps=max_history_steps,
                dft_frequencies=dft_frequencies,
                dft_t_start=dft_t_start,
                dft_t_end=dft_t_end,
                dft_enabled=dft_enabled,
                dft_components=dft_components,
                dft_record_every_step=dft_record_every_step,
                dft_record_interval=dft_record_interval,
                dft_window=dft_window,
                name=name,
                frequency_points=frequency_points,
                frequency_record_interval=frequency_record_interval,
            ),
        )
        object.__setattr__(self, "state", create_monitor_state(self.spec))

    def __getattr__(self, name):
        spec = self.__dict__.get("spec")
        if spec is not None and hasattr(spec, name):
            return getattr(spec, name)
        state = self.__dict__.get("state")
        if state is not None and hasattr(state, name):
            return getattr(state, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in {"design", "spec", "state", "_live_state", "update_interval", "objective_function"}:
            object.__setattr__(self, name, value)
            return
        if name in _SPEC_FIELDS and "spec" in self.__dict__:
            object.__setattr__(self, "spec", replace(self.spec, **{name: value}))
            return
        if name in _STATE_FIELDS and "state" in self.__dict__:
            setattr(self.state, name, value)
            return
        object.__setattr__(self, name, value)

    def with_spec(self, spec=None, /, **changes):
        base_spec = self.spec if spec is None else spec
        if not isinstance(base_spec, MonitorSpec):
            raise TypeError("with_spec expects a MonitorSpec or spec field updates")
        if changes:
            base_spec = replace(base_spec, **changes)
        new = object.__new__(type(self))
        object.__setattr__(new, "design", self.design)
        object.__setattr__(new, "_live_state", live_helpers.create_state())
        object.__setattr__(new, "update_interval", self.update_interval)
        object.__setattr__(new, "objective_function", self.objective_function)
        object.__setattr__(new, "spec", base_spec)
        object.__setattr__(new, "state", create_monitor_state(base_spec))
        return new

    def to_dict(self):
        return self.spec.to_dict()

    @classmethod
    def from_dict(cls, data, *, design=None, objective_function=None):
        return cls.from_spec(
            MonitorSpec.from_dict(data),
            design=design,
            objective_function=objective_function,
        )

    @classmethod
    def from_spec(cls, spec, *, design=None, objective_function=None):
        if not isinstance(spec, MonitorSpec):
            raise TypeError("from_spec expects a MonitorSpec")
        new = object.__new__(cls)
        object.__setattr__(new, "design", design)
        object.__setattr__(new, "_live_state", live_helpers.create_state())
        object.__setattr__(new, "update_interval", 10)
        object.__setattr__(new, "objective_function", objective_function)
        object.__setattr__(new, "spec", spec)
        object.__setattr__(new, "state", create_monitor_state(spec))
        return new

    def evaluate_objective(self) -> Optional[float]:
        """Evaluate the objective function associated with this monitor, if any."""
        return store_helpers.evaluate_objective(self)

    def _determine_3d_mode(self, start, end, design):
        """Determine if this should be a 3D monitor based on inputs."""
        return geom_helpers.determine_3d_mode(start, end, design)

    def _init_2d_monitor(self, start, end):
        """Initialize 2D line monitor."""
        self.spec = build_monitor_spec(
            design=self.design,
            start=start,
            end=end,
            record_fields=self.should_record_fields,
            accumulate_power=self.accumulate_power,
            live_update=self.live_update,
            record_interval=self.record_interval,
            max_history_steps=self.max_history_steps,
            dft_frequencies=self.dft_frequencies,
            dft_t_start=self.dft_t_start,
            dft_t_end=self.dft_t_end,
            dft_enabled=self.dft_enabled,
            dft_components=self.dft_components,
            dft_record_every_step=self.dft_record_every_step,
            dft_record_interval=self.dft_record_interval,
            dft_window=self.dft_window,
            name=self.name,
            frequency_points=self.frequency_points,
            frequency_record_interval=self.frequency_record_interval,
        )

    def _init_3d_monitor(self, start, end, plane_normal, plane_position, size):
        """Initialize 3D plane monitor from two points or plane definition."""
        self.spec = build_monitor_spec(
            design=self.design,
            start=start,
            end=end,
            plane_normal=plane_normal,
            plane_position=plane_position,
            size=size,
            record_fields=self.should_record_fields,
            accumulate_power=self.accumulate_power,
            live_update=self.live_update,
            record_interval=self.record_interval,
            max_history_steps=self.max_history_steps,
            dft_frequencies=self.dft_frequencies,
            dft_t_start=self.dft_t_start,
            dft_t_end=self.dft_t_end,
            dft_enabled=self.dft_enabled,
            dft_components=self.dft_components,
            dft_record_every_step=self.dft_record_every_step,
            dft_record_interval=self.dft_record_interval,
            dft_window=self.dft_window,
            name=self.name,
            frequency_points=self.frequency_points,
            frequency_record_interval=self.frequency_record_interval,
        )

    def _generate_plane_vertices(self):
        """Generate vertices for the monitor plane for 3D visualization."""
        return geom_helpers.generate_plane_vertices(self)

    def _get_plane_center(self):
        """Get center position of 3D plane monitor."""
        return geom_helpers.get_plane_center(self)

    def get_grid_points_2d(self, dx, dy):
        """Get grid points for 2D line monitor."""
        return geom_helpers.grid_points_2d(self, dx, dy)

    def get_grid_slice_3d(self, dx, dy, dz, field_shape):
        """Get grid slice for 3D plane monitor.
        Returns (z_idx, y_idx, x_idx) consistent with simulation array order (z, y, x).
        One of these will be an integer, the other two will be slice objects.
        """
        return geom_helpers.grid_slice_3d(self, dx, dy, dz, field_shape)

    def should_record(self, step):
        """Check if this step should be recorded based on interval."""
        return record_helpers.should_record(self, step)

    def _dft_should_accumulate(self, step, t):
        return dft_helpers.should_accumulate(self, step, t)

    def _dft_weight(self, t):
        return dft_helpers.weight(self, t)

    def _init_or_get_dft_accum(self, component, npoints):
        return dft_helpers.init_or_get_accum(self, component, npoints)

    def _dft_current_phase(self, t):
        return dft_helpers.current_phase(self, t)

    def _update_dft(self, t, component_vectors):
        dft_helpers.update(self, t, component_vectors)

    def reset_dft(self):
        dft_helpers.reset(self)

    def get_dft_frequencies(self):
        return dft_helpers.get_frequencies(self)

    def get_dft_component(self, component: str):
        return dft_helpers.get_component(self, component)

    def record_fields_2d(
        self, Ez, Hx, Hy, t, dx, dy, step=0, Ex=None, Ey=None, Hz=None
    ):
        """Record 2D field data."""
        record_helpers.record_fields_2d(
            self, Ez, Hx, Hy, t, dx, dy, step=step, Ex=Ex, Ey=Ey, Hz=Hz
        )

    def record_fields_3d(self, Ex, Ey, Ez, Hx, Hy, Hz, t, dx, dy, dz, step=0):
        """Record 3D field data from plane slice."""
        record_helpers.record_fields_3d(
            self, Ex, Ey, Ez, Hx, Hy, Hz, t, dx, dy, dz, step=step
        )

    def record_fields(self, *args, **kwargs):
        """Generic field recording method that delegates to 2D or 3D."""
        record_helpers.record_fields(self, *args, **kwargs)

    def _calculate_power_2d(self, Ez_values, Hx_values, Hy_values, t, dx, dy):
        """Calculate Poynting vector and power for 2D fields.

        Power is computed as the integral of the Poynting vector magnitude
        over the monitor line, properly normalized by grid cell area.
        """
        record_helpers.calculate_power_2d(
            self, Ez_values, Hx_values, Hy_values, t, dx, dy
        )

    def _calculate_power_3d(self, Ex, Ey, Ez, Hx, Hy, Hz, t, dx, dy):
        """Calculate Poynting vector and power for 3D fields.

        Power is computed as the integral of the Poynting vector magnitude
        over the monitor plane, properly normalized by grid cell area.
        """
        record_helpers.calculate_power_3d(self, Ex, Ey, Ez, Hx, Hy, Hz, t, dx, dy)

    def _manage_memory(self):
        """Manage memory by limiting stored history."""
        record_helpers.manage_memory(self)

    def _setup_live_plot_2d(self, field_component):
        """Setup live plotting for 2D monitor."""
        live_helpers.setup_plot_2d(self, field_component)

    def _setup_live_plot_3d(self, field_component):
        """Setup live plotting for 3D monitor."""
        live_helpers.setup_plot_3d(self, field_component)

    def _update_live_plot_2d(self):
        """Update live plot for 2D monitor."""
        live_helpers.update_plot_2d(self)

    def _update_live_plot_3d(self):
        """Update live plot for 3D monitor."""
        live_helpers.update_plot_3d(self)

    def get_field_statistics(self):
        """Get statistical information about recorded fields."""
        return record_helpers.field_statistics(self)

    def save_data(self, filename, format="npz"):
        """Save recorded data to file."""
        store_helpers.save_data(self, filename, format=format)

    def load_data(self, filename):
        """Load data from file."""
        store_helpers.load_data(self, filename)

    def add_to_plot(
        self, ax, facecolor="none", edgecolor="navy", alpha=1, linestyle="-"
    ):
        """Add monitor visualization to 2D plot."""
        geom_helpers.add_to_plot(
            self,
            ax,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linestyle=linestyle,
        )

    def to_polygon(self):
        """Convert monitor to a polygon for 3D visualization."""
        return geom_helpers.to_polygon(self)

    def get_field_at_time(self, field="Ez", time_value=None, time_index=None):
        """Get field data at a specific time.

        Args:
            field: Field component to retrieve
            time_value: Specific time value (will find closest)
            time_index: Specific time index

        Returns:
            Field data array
        """
        return record_helpers.field_at_time(
            self, field=field, time_value=time_value, time_index=time_index
        )

    def field_snapshot(self, field="Ez", time_value=None, time_index=None):
        """Return a plotting-ready field snapshot as Trace1D or Slice2D."""
        return store_helpers.field_snapshot(
            self, field=field, time_value=time_value, time_index=time_index
        )

    def get_power_statistics(self):
        """Get power statistics from recorded data.

        Returns:
            Dictionary with power statistics
        """
        return record_helpers.power_statistics(self)

    def power_trace(self, *, db_scale=False):
        """Return monitor power history as a Trace1D."""
        return store_helpers.power_trace(self, db_scale=db_scale)

    def get_signed_flux_trace(self, normal_direction, field_pair=None):
        """Return signed directional flux trace from recorded field components.

        For 2D monitors:
          - default +x/-x uses (Ez, Hy) with Sx = -Re(Ez * conj(Hy))
          - default +y/-y uses (Ez, Hx) with Sy = +Re(Ez * conj(Hx))
        """
        return record_helpers.signed_flux_trace(
            self, normal_direction, field_pair=field_pair
        )

    def flux_trace(self, normal_direction, field_pair=None):
        """Return signed directional flux history as a Trace1D."""
        return store_helpers.flux_trace(
            self, normal_direction, field_pair=field_pair
        )

    def __str__(self):
        return store_helpers.describe(self)

    def copy(self):
        """Create a deep copy of the Monitor."""
        return store_helpers.copy_monitor(self)
