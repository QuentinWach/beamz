from typing import Callable, Optional

import numpy as np

from beamz.devices.monitors import dft as dft_helpers
from beamz.devices.monitors import geom as geom_helpers
from beamz.devices.monitors import live as live_helpers
from beamz.devices.monitors import record as record_helpers
from beamz.devices.monitors import store as store_helpers


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
        return store_helpers.evaluate_objective(self)

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
        live_helpers.start_visualization(self, field_component=field_component)

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

    def plot_fields(self, **kwargs):
        """Plot field data from the monitor. Delegates to visual.monitor_plots."""
        return store_helpers.plot_fields(self, **kwargs)

    def plot_power(self, **kwargs):
        """Plot power history from the monitor. Delegates to visual.monitor_plots."""
        return store_helpers.plot_power(self, **kwargs)

    def animate_fields(self, **kwargs):
        """Create an animation of field evolution. Delegates to visual.monitor_plots."""
        return store_helpers.animate_fields(self, **kwargs)

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
        return store_helpers.describe(self)

    def copy(self):
        """Create a deep copy of the Monitor."""
        return store_helpers.copy_monitor(self)
