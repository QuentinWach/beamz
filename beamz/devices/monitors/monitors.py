import logging
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np

from beamz.devices.monitors import dft as dft_helpers
from beamz.devices.monitors import geom as geom_helpers
from beamz.devices.monitors import record as record_helpers

logger = logging.getLogger(__name__)


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
        self.design = design
        self.should_record_fields = record_fields
        self.accumulate_power = accumulate_power
        self.live_update = live_update
        self.record_interval = record_interval
        self.max_history_steps = max_history_steps
        self.dft_enabled = bool(dft_enabled)
        self.dft_record_every_step = bool(dft_record_every_step)
        if dft_record_interval is None:
            self.dft_record_interval = (
                1 if self.dft_record_every_step else max(1, int(record_interval))
            )
        else:
            self.dft_record_interval = max(1, int(dft_record_interval))
        self.dft_t_start = float(dft_t_start) if dft_t_start is not None else 0.0
        self.dft_t_end = None if dft_t_end is None else float(dft_t_end)
        self.dft_window = str(dft_window).lower()
        if self.dft_window in {"none", "rectangular"}:
            self.dft_window = "rect"
        if self.dft_window not in {"rect", "hann"}:
            raise ValueError(
                f"dft_window must be one of ['rect', 'hann'], got {dft_window!r}"
            )
        if dft_frequencies is None:
            self.dft_frequencies = np.array([], dtype=float)
        else:
            self.dft_frequencies = np.atleast_1d(
                np.asarray(dft_frequencies, dtype=float)
            )
        self.dft_components = (
            tuple(str(c) for c in dft_components)
            if dft_components is not None
            else None
        )
        if frequency_points is None:
            freq_arr = np.zeros((0,), dtype=np.float64)
        else:
            freq_arr = np.asarray(frequency_points, dtype=np.float64).ravel()
            if freq_arr.ndim != 1:
                raise ValueError(
                    "frequency_points must be a 1D sequence of frequencies in Hz"
                )
            if np.any(freq_arr < 0.0):
                raise ValueError(
                    "frequency_points must be non-negative frequencies in Hz"
                )
        self.frequency_points = freq_arr
        self.frequency_record_interval = max(1, int(frequency_record_interval))
        self.accumulate_frequency = bool(freq_arr.size > 0)
        self.frequency_flux_spectrum = np.zeros(freq_arr.shape, dtype=np.complex64)
        self.objective_function = objective_function
        self.objective_value: Optional[float] = None
        self.name = name

        # Determine if this is a 3D monitor based on input parameters
        self.is_3d = self._determine_3d_mode(start, end, design)

        # Initialize field storage
        if self.is_3d:
            # 3D fields: Ex, Ey, Ez, Hx, Hy, Hz
            self.fields = {
                "Ex": [],
                "Ey": [],
                "Ez": [],
                "Hx": [],
                "Hy": [],
                "Hz": [],
                "t": [],
            }
        else:
            # 2D fields: Ez, Hx, Hy and optional TE set Ex, Ey, Hz
            self.fields = {
                "Ex": [],
                "Ey": [],
                "Ez": [],
                "Hx": [],
                "Hy": [],
                "Hz": [],
                "t": [],
            }

        # Power and energy storage
        self.power_accumulated = None
        self.energy_history = []
        self.power_history = []
        self.power_timestamps = []
        self.power_accumulation_count = 0
        # Recording control
        self.step_count = 0
        self.last_record_step = -1
        # Live visualization
        self.live_fig = None
        self.live_axes = None
        self.live_plots = {}
        self.update_interval = (
            10  # Update every N records (faster updates for visibility)
        )
        # DFT accumulators: component -> complex[nf, npoints], plus scalar weight sum.
        self._dft_accum = {}
        self._dft_weight_sum = np.zeros(self.dft_frequencies.size, dtype=float)
        self._dft_sample_count = 0
        self._dft_phase = np.ones(self.dft_frequencies.size, dtype=np.complex128)
        self._dft_last_t = None
        self._dft_last_dt = None
        self._dft_last_rot = None

        if self.is_3d:
            self._init_3d_monitor(start, end, plane_normal, plane_position, size)
        else:
            self._init_2d_monitor(start, end)

    def evaluate_objective(self) -> Optional[float]:
        """Evaluate the objective function associated with this monitor, if any."""
        if self.objective_function is None:
            return None
        try:
            value = self.objective_function(self)
        except Exception as exc:
            print(f"Warning: monitor objective evaluation failed: {exc}")
            return None
        if value is None:
            return None
        try:
            self.objective_value = float(value)
        except (TypeError, ValueError):
            print(f"Warning: monitor objective returned non-numeric value: {value}")
            return None
        return self.objective_value

    def _determine_3d_mode(self, start, end, design):
        """Determine if this should be a 3D monitor based on inputs."""
        return geom_helpers.determine_3d_mode(start, end, design)

    def _init_2d_monitor(self, start, end):
        """Initialize 2D line monitor."""
        geom_helpers.init_2d_monitor(self, start, end)

    def _init_3d_monitor(self, start, end, plane_normal, plane_position, size):
        """Initialize 3D plane monitor from two points or plane definition."""
        geom_helpers.init_3d_monitor(self, start, end, plane_normal, plane_position, size)

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

    def start_live_visualization(self, field_component="Ez"):
        """Start live field visualization."""
        if not self.live_update:
            self.live_update = True
        if self.is_3d:
            self._setup_live_plot_3d(field_component)
        else:
            self._setup_live_plot_2d(field_component)

    def _setup_live_plot_2d(self, field_component):
        """Setup live plotting for 2D monitor."""
        self.live_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # Field amplitude plot
        ax1.set_title(f"{field_component} along monitor line")
        ax1.set_xlabel("Position along line")
        ax1.set_ylabel(f"{field_component} amplitude")
        self.live_plots["field_line"] = ax1.plot([], [], "b-")[0]
        # Power history plot
        ax2.set_title("Power vs Time")
        ax2.set_xlabel("Time step")
        ax2.set_ylabel("Total power")
        self.live_plots["power_time"] = ax2.plot([], [], "r-")[0]
        plt.tight_layout()
        plt.ion()
        plt.show()

    def _setup_live_plot_3d(self, field_component):
        """Setup live plotting for 3D monitor."""
        self.live_fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        # Field magnitude plot
        ax1.set_title(f"{field_component} magnitude on plane")
        self.live_plots["field_2d"] = ax1.imshow(
            np.zeros((10, 10)), cmap="RdBu", animated=True
        )
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        # Power history
        ax2.set_title("Power vs Time")
        ax2.set_xlabel("Time step")
        ax2.set_ylabel("Total power")
        self.live_plots["power_time"] = ax2.plot([], [], "r-")[0]
        # Field profile along X
        ax3.set_title(f"{field_component} along X (center)")
        ax3.set_xlabel("X position")
        ax3.set_ylabel(f"{field_component} amplitude")
        self.live_plots["field_x"] = ax3.plot([], [], "b-")[0]
        # Field profile along Y
        ax4.set_title(f"{field_component} along Y (center)")
        ax4.set_xlabel("Y position")
        ax4.set_ylabel(f"{field_component} amplitude")
        self.live_plots["field_y"] = ax4.plot([], [], "g-")[0]
        plt.tight_layout()
        plt.ion()
        plt.show()

    def _update_live_plot_2d(self):
        """Update live plot for 2D monitor."""
        if self.live_fig is None or not self.fields["t"]:
            return
        try:
            # Update field line plot
            latest_field = self.fields["Ez"][-1]
            x_pos = range(len(latest_field))
            self.live_plots["field_line"].set_data(x_pos, latest_field)
            # Update power history
            self.live_plots["power_time"].set_data(
                range(len(self.power_history)), self.power_history
            )
            # Rescale axes
            for ax in self.live_fig.axes:
                ax.relim()
                ax.autoscale_view()
            self.live_fig.canvas.draw()
            self.live_fig.canvas.flush_events()
        except Exception:
            logger.debug("Failed to update 2D monitor live plot.", exc_info=True)

    def _update_live_plot_3d(self):
        """Update live plot for 3D monitor."""
        if self.live_fig is None or not self.fields["t"]:
            return
        try:
            # Get latest field data
            latest_field = self.fields["Ez"][-1]  # Default to Ez
            # Update 2D field plot
            self.live_plots["field_2d"].set_array(latest_field)
            self.live_plots["field_2d"].set_clim(
                vmin=np.min(latest_field), vmax=np.max(latest_field)
            )
            # Update power history
            self.live_plots["power_time"].set_data(
                range(len(self.power_history)), self.power_history
            )
            # Update field profiles
            center_y = latest_field.shape[0] // 2
            center_x = latest_field.shape[1] // 2
            self.live_plots["field_x"].set_data(
                range(latest_field.shape[1]), latest_field[center_y, :]
            )
            self.live_plots["field_y"].set_data(
                range(latest_field.shape[0]), latest_field[:, center_x]
            )
            # Rescale axes
            for ax in self.live_fig.axes[1:]:  # Skip imshow axis
                ax.relim()
                ax.autoscale_view()
            self.live_fig.canvas.draw()
            self.live_fig.canvas.flush_events()
        except Exception:
            logger.debug("Failed to update 3D monitor live plot.", exc_info=True)

    def get_field_statistics(self):
        """Get statistical information about recorded fields."""
        return record_helpers.field_statistics(self)

    def save_data(self, filename, format="npz"):
        """Save recorded data to file."""
        if format == "npz":
            np.savez(
                filename,
                fields=self.fields,
                power_history=self.power_history,
                power_timestamps=self.power_timestamps,
                frequency_points=self.frequency_points,
                frequency_flux_spectrum=self.frequency_flux_spectrum,
                monitor_info={"type": self.monitor_type, "is_3d": self.is_3d},
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load_data(self, filename):
        """Load data from file."""
        data = np.load(filename, allow_pickle=True)
        self.fields = data["fields"].item()
        self.power_history = list(data["power_history"])
        if "power_timestamps" in data:
            self.power_timestamps = list(data["power_timestamps"])
        else:
            self.power_timestamps = list(range(len(self.power_history)))
        if "frequency_points" in data:
            self.frequency_points = np.asarray(
                data["frequency_points"], dtype=np.float64
            )
        if "frequency_flux_spectrum" in data:
            self.frequency_flux_spectrum = np.asarray(
                data["frequency_flux_spectrum"], dtype=np.complex64
            )

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

    def plot_fields(self, **kwargs):
        """Plot field data from the monitor. Delegates to visual.monitor_plots."""
        from beamz.visual.monitor_plots import plot_monitor_fields

        return plot_monitor_fields(self, **kwargs)

    def plot_power(self, **kwargs):
        """Plot power history from the monitor. Delegates to visual.monitor_plots."""
        from beamz.visual.monitor_plots import plot_monitor_power

        return plot_monitor_power(self, **kwargs)

    def animate_fields(self, **kwargs):
        """Create an animation of field evolution. Delegates to visual.monitor_plots."""
        from beamz.visual.monitor_plots import animate_monitor_fields

        return animate_monitor_fields(self, **kwargs)

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

    def get_power_statistics(self):
        """Get power statistics from recorded data.

        Returns:
            Dictionary with power statistics
        """
        return record_helpers.power_statistics(self)

    def get_signed_flux_trace(self, normal_direction, field_pair=None):
        """Return signed directional flux trace from recorded field components.

        For 2D monitors:
          - default +x/-x uses (Ez, Hy) with Sx = -Re(Ez * conj(Hy))
          - default +y/-y uses (Ez, Hx) with Sy = +Re(Ez * conj(Hx))
        """
        return record_helpers.signed_flux_trace(
            self, normal_direction, field_pair=field_pair
        )

    def __str__(self):
        if not self.fields["t"]:
            return f"Monitor: {self.monitor_type} ({'3D' if self.is_3d else '2D'}), 0 records"
        stats = self.get_field_statistics()
        return f"Monitor: {stats['monitor_type']} ({'3D' if stats['is_3d'] else '2D'}), {stats['total_records']} records"

    def copy(self):
        """Create a deep copy of the Monitor."""
        if self.is_3d:
            # 3D monitor
            if hasattr(self, "end") and self.end is not None:
                # Defined by start and end points
                return Monitor(
                    design=self.design,  # Reference to same design is okay
                    start=self.start,
                    end=self.end,
                    record_fields=self.should_record_fields,
                    accumulate_power=self.accumulate_power,
                    live_update=self.live_update,
                    record_interval=self.record_interval,
                    max_history_steps=self.max_history_steps,
                    dft_frequencies=self.dft_frequencies.copy(),
                    dft_t_start=self.dft_t_start,
                    dft_t_end=self.dft_t_end,
                    dft_enabled=self.dft_enabled,
                    dft_components=self.dft_components,
                    dft_record_every_step=self.dft_record_every_step,
                    dft_record_interval=self.dft_record_interval,
                    dft_window=self.dft_window,
                )
            else:
                # Defined by plane normal and position
                return Monitor(
                    design=self.design,  # Reference to same design is okay
                    start=self.start,
                    plane_normal=self.plane_normal,
                    plane_position=self.plane_position,
                    size=self.size,
                    record_fields=self.should_record_fields,
                    accumulate_power=self.accumulate_power,
                    live_update=self.live_update,
                    record_interval=self.record_interval,
                    max_history_steps=self.max_history_steps,
                    dft_frequencies=self.dft_frequencies.copy(),
                    dft_t_start=self.dft_t_start,
                    dft_t_end=self.dft_t_end,
                    dft_enabled=self.dft_enabled,
                    dft_components=self.dft_components,
                    dft_record_every_step=self.dft_record_every_step,
                    dft_record_interval=self.dft_record_interval,
                    dft_window=self.dft_window,
                )
        else:
            # 2D monitor
            return Monitor(
                design=self.design,  # Reference to same design is okay
                start=self.start,
                end=self.end,
                record_fields=self.should_record_fields,
                accumulate_power=self.accumulate_power,
                live_update=self.live_update,
                record_interval=self.record_interval,
                max_history_steps=self.max_history_steps,
                dft_frequencies=self.dft_frequencies.copy(),
                dft_t_start=self.dft_t_start,
                dft_t_end=self.dft_t_end,
                dft_enabled=self.dft_enabled,
                dft_components=self.dft_components,
                dft_record_every_step=self.dft_record_every_step,
                dft_record_interval=self.dft_record_interval,
                dft_window=self.dft_window,
            )
