from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle as MatplotlibRectangle


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
        objective_function: Optional[Callable[["Monitor"], float]] = None,
        name: Optional[str] = None,
    ):
        self.design = design
        self.should_record_fields = record_fields
        self.accumulate_power = accumulate_power
        self.live_update = live_update
        self.record_interval = record_interval
        self.max_history_steps = max_history_steps
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
            # 2D fields: Ez, Hx, Hy
            self.fields = {"Ez": [], "Hx": [], "Hy": [], "t": []}

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
        # If end is provided and has 3 coordinates, it's 3D
        if end is not None and len(end) == 3:
            return True
        # If start has 3 coordinates, it's 3D
        if len(start) == 3:
            return True
        # For 2D monitors with start/end (line monitors), stay in 2D mode
        # even if the design supports 3D - this handles the common case where
        # users create line monitors in 2D simulations
        if end is not None and len(start) == 2 and len(end) == 2:
            return False
        # If design is 3D and not explicitly using 2D start/end, default to 3D monitor
        if design and hasattr(design, "is_3d") and design.is_3d:
            return True
        return False

    def _init_2d_monitor(self, start, end):
        """Initialize 2D line monitor."""
        if end is None:
            end = start
        self.start = start
        self.end = end
        self.position = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        self.monitor_type = "line"

    def _init_3d_monitor(self, start, end, plane_normal, plane_position, size):
        """Initialize 3D plane monitor from two points or plane definition."""
        # Ensure start is 3D
        if len(start) == 2:
            start = (start[0], start[1], 0.0)
        self.start = start

        if end is not None:
            # Monitor defined by two corner points - create a plane
            if len(end) == 2:
                end = (end[0], end[1], start[2])  # Same z as start
            self.end = end

            # Auto-detect plane normal if not explicitly provided
            if plane_normal is None:
                dx = abs(end[0] - start[0])
                dy = abs(end[1] - start[1])
                dz = abs(end[2] - start[2])

                # The normal is the axis with the smallest (ideally zero) extent
                dims = [dx, dy, dz]
                min_dim_idx = np.argmin(dims)
                if min_dim_idx == 0:
                    self.plane_normal = "x"
                elif min_dim_idx == 1:
                    self.plane_normal = "y"
                else:
                    self.plane_normal = "z"
            else:
                self.plane_normal = plane_normal

            # Set size and position based on detected/provided normal
            if self.plane_normal == "x":
                self.size = (abs(end[1] - start[1]), abs(end[2] - start[2]))
                self.plane_position = start[0]
            elif self.plane_normal == "y":
                self.size = (abs(end[0] - start[0]), abs(end[2] - start[2]))
                self.plane_position = start[1]
            else:  # z
                self.size = (abs(end[0] - start[0]), abs(end[1] - start[1]))
                self.plane_position = start[2]

            # Ensure start is the bottom-left corner of the ROI
            self.start = (
                min(start[0], end[0]),
                min(start[1], end[1]),
                min(start[2], end[2]),
            )

        else:
            # Monitor defined by plane normal and position (legacy mode)
            self.end = None
            self.plane_normal = plane_normal or "z"  # Default to xy plane

            # Extract plane_position from start if not explicitly provided
            if plane_position == 0 and start is not None and len(start) >= 3:
                if self.plane_normal == "z":
                    self.plane_position = start[2]
                elif self.plane_normal == "y":
                    self.plane_position = start[1]
                elif self.plane_normal == "x":
                    self.plane_position = start[0]
                else:
                    self.plane_position = plane_position
            else:
                self.plane_position = plane_position

            # Determine plane dimensions
            if size is None:
                # Use design dimensions if available
                if self.design:
                    if self.plane_normal == "z":
                        size = (self.design.width, self.design.height)
                    elif self.plane_normal == "y":
                        size = (
                            self.design.width,
                            self.design.depth or self.design.width,
                        )
                    else:  # x normal
                        size = (
                            self.design.height,
                            self.design.depth or self.design.height,
                        )
                else:
                    size = (1e-6, 1e-6)  # Default 1μm x 1μm
            self.size = size

        self.monitor_type = "plane"
        self.position = self._get_plane_center()

        # Generate vertices for 3D visualization
        self.vertices = self._generate_plane_vertices()

    def _generate_plane_vertices(self):
        """Generate vertices for the monitor plane for 3D visualization."""
        if self.plane_normal == "z" or (hasattr(self, "end") and self.end is not None):
            # xy plane at fixed z
            x_min, y_min = self.start[0], self.start[1]
            x_max = x_min + self.size[0]
            y_max = y_min + self.size[1]
            z = self.plane_position

            vertices = [
                (x_min, y_min, z),  # Bottom left
                (x_max, y_min, z),  # Bottom right
                (x_max, y_max, z),  # Top right
                (x_min, y_max, z),  # Top left
            ]

        elif self.plane_normal == "y":
            # xz plane at fixed y
            x_min, z_min = self.start[0], self.start[2]
            x_max = x_min + self.size[0]
            z_max = z_min + self.size[1]
            y = self.plane_position

            vertices = [
                (x_min, y, z_min),  # Bottom left
                (x_max, y, z_min),  # Bottom right
                (x_max, y, z_max),  # Top right
                (x_min, y, z_max),  # Top left
            ]

        else:  # x normal
            # yz plane at fixed x
            y_min, z_min = self.start[1], self.start[2]
            y_max = y_min + self.size[0]
            z_max = z_min + self.size[1]
            x = self.plane_position

            vertices = [
                (x, y_min, z_min),  # Bottom left
                (x, y_max, z_min),  # Bottom right
                (x, y_max, z_max),  # Top right
                (x, y_min, z_max),  # Top left
            ]

        return vertices

    def _get_plane_center(self):
        """Get center position of 3D plane monitor."""
        if self.plane_normal == "z" or (hasattr(self, "end") and self.end is not None):
            return (
                self.start[0] + self.size[0] / 2,
                self.start[1] + self.size[1] / 2,
                self.plane_position,
            )
        elif self.plane_normal == "y":
            return (
                self.start[0] + self.size[0] / 2,
                self.plane_position,
                self.start[2] + self.size[1] / 2,
            )
        else:
            return (
                self.plane_position,
                self.start[1] + self.size[0] / 2,
                self.start[2] + self.size[1] / 2,
            )

    def get_grid_points_2d(self, dx, dy):
        """Get grid points for 2D line monitor."""
        start_x_grid = int(round(self.start[0] / dx))
        start_y_grid = int(round(self.start[1] / dy))
        end_x_grid = int(round(self.end[0] / dx))
        end_y_grid = int(round(self.end[1] / dy))

        if abs(end_x_grid - start_x_grid) > abs(end_y_grid - start_y_grid):
            num_points = abs(end_x_grid - start_x_grid) + 1
            x_indices = np.linspace(start_x_grid, end_x_grid, num_points, dtype=int)
            y_indices = np.linspace(start_y_grid, end_y_grid, num_points, dtype=int)
        else:
            num_points = abs(end_y_grid - start_y_grid) + 1
            x_indices = np.linspace(start_x_grid, end_x_grid, num_points, dtype=int)
            y_indices = np.linspace(start_y_grid, end_y_grid, num_points, dtype=int)

        return list(zip(x_indices, y_indices))

    def get_grid_slice_3d(self, dx, dy, dz, field_shape):
        """Get grid slice for 3D plane monitor.
        Returns (z_idx, y_idx, x_idx) consistent with simulation array order (z, y, x).
        One of these will be an integer, the other two will be slice objects.
        """
        # Derive base grid counts from either design or field_shape
        if self.design:
            base_nx = max(1, int(round((getattr(self.design, "width", 0.0)) / dx)))
            base_ny = max(1, int(round((getattr(self.design, "height", 0.0)) / dy)))
            base_nz = max(
                1, int(round((getattr(self.design, "depth", 0.0) or 0.0) / dz))
            )
        else:
            base_nz, base_ny, base_nx = field_shape

        if self.plane_normal == "z":
            # xy plane at fixed z
            z_idx = int(round(self.plane_position / dz))
            x_start = int(round(self.start[0] / dx))
            x_end = int(round((self.start[0] + self.size[0]) / dx))
            y_start = int(round(self.start[1] / dy))
            y_end = int(round((self.start[1] + self.size[1]) / dy))
            return z_idx, slice(y_start, y_end), slice(x_start, x_end)

        elif self.plane_normal == "y":
            # xz plane at fixed y
            y_idx = int(round(self.plane_position / dy))
            x_start = int(round(self.start[0] / dx))
            x_end = int(round((self.start[0] + self.size[0]) / dx))
            z_start = int(round(self.start[2] / dz))
            z_end = int(round((self.start[2] + self.size[1]) / dz))
            return slice(z_start, z_end), y_idx, slice(x_start, x_end)
        else:  # x normal
            # yz plane at fixed x
            x_idx = int(round(self.plane_position / dx))
            y_start = int(round(self.start[1] / dy))
            y_end = int(round((self.start[1] + self.size[0]) / dy))
            z_start = int(round(self.start[2] / dz))
            z_end = int(round((self.start[2] + self.size[1]) / dz))
            return slice(z_start, z_end), slice(y_start, y_end), x_idx

    def should_record(self, step):
        """Check if this step should be recorded based on interval."""
        return (step - self.last_record_step) >= self.record_interval

    def record_fields_2d(self, Ez, Hx, Hy, t, dx, dy, step=0):
        """Record 2D field data."""
        if not self.should_record(step):
            return
        grid_points = self.get_grid_points_2d(dx, dy)
        Ez_values, Hx_values, Hy_values = [], [], []
        for x_idx, y_idx in grid_points:
            # Ez values
            if 0 <= y_idx < Ez.shape[0] and 0 <= x_idx < Ez.shape[1]:
                val = Ez[y_idx, x_idx]
                Ez_values.append(complex(val) if np.iscomplexobj(val) else float(val))
            else:
                Ez_values.append(0.0)
            # Hx values
            if 0 <= y_idx < Hx.shape[0] and 0 <= x_idx < Hx.shape[1]:
                val = Hx[y_idx, x_idx]
                Hx_values.append(complex(val) if np.iscomplexobj(val) else float(val))
            else:
                Hx_values.append(0.0)
            # Hy values
            if 0 <= y_idx < Hy.shape[0] and 0 <= x_idx < Hy.shape[1]:
                val = Hy[y_idx, x_idx]
                Hy_values.append(complex(val) if np.iscomplexobj(val) else float(val))
            else:
                Hy_values.append(0.0)

        if self.should_record_fields:
            self.fields["Ez"].append(Ez_values)
            self.fields["Hx"].append(Hx_values)
            self.fields["Hy"].append(Hy_values)
            self.fields["t"].append(t)

        if self.accumulate_power:
            self._calculate_power_2d(Ez_values, Hx_values, Hy_values, t, dx, dy)

        self.last_record_step = step
        self._manage_memory()

        if self.live_update and (len(self.fields["t"]) % self.update_interval == 0):
            self._update_live_plot_2d()

    def record_fields_3d(self, Ex, Ey, Ez, Hx, Hy, Hz, t, dx, dy, dz, step=0):
        """Record 3D field data from plane slice."""
        if not self.should_record(step):
            return

        def slice_field(arr):
            # Returns (z_slice, y_slice, x_slice) consistent with (z, y, x) order
            z_idx, y_idx, x_idx = self.get_grid_slice_3d(dx, dy, dz, arr.shape)

            nz, ny, nx = arr.shape

            # Helper to clamp indices to the actual array shape (handles Yee staggering)
            def clamp(idx, limit):
                if isinstance(idx, int):
                    return min(max(0, idx), limit - 1)
                else:
                    start = max(
                        0, min(idx.start if idx.start is not None else 0, limit - 1)
                    )
                    stop = max(
                        start, min(idx.stop if idx.stop is not None else limit, limit)
                    )
                    return slice(start, stop)

            # Extract and copy the 2D slice
            return arr[clamp(z_idx, nz), clamp(y_idx, ny), clamp(x_idx, nx)].copy()

        Ex_slice = slice_field(Ex)
        Ey_slice = slice_field(Ey)
        Ez_slice = slice_field(Ez)
        Hx_slice = slice_field(Hx)
        Hy_slice = slice_field(Hy)
        Hz_slice = slice_field(Hz)

        # Align to common overlapping region to account for Yee staggering differences between components
        min_dim0 = min(
            Ex_slice.shape[0],
            Ey_slice.shape[0],
            Ez_slice.shape[0],
            Hx_slice.shape[0],
            Hy_slice.shape[0],
            Hz_slice.shape[0],
        )
        min_dim1 = min(
            Ex_slice.shape[1],
            Ey_slice.shape[1],
            Ez_slice.shape[1],
            Hx_slice.shape[1],
            Hy_slice.shape[1],
            Hz_slice.shape[1],
        )

        Ex_slice = Ex_slice[:min_dim0, :min_dim1]
        Ey_slice = Ey_slice[:min_dim0, :min_dim1]
        Ez_slice = Ez_slice[:min_dim0, :min_dim1]
        Hx_slice = Hx_slice[:min_dim0, :min_dim1]
        Hy_slice = Hy_slice[:min_dim0, :min_dim1]
        Hz_slice = Hz_slice[:min_dim0, :min_dim1]

        # print(f"● Monitor record step {step}: Ez_slice max={np.max(np.abs(Ez_slice)):.2e}")
        # print(f"● Monitor record step {step}: Ez_slice max={np.max(np.abs(Ez_slice)):.2e}")

        if self.should_record_fields:
            self.fields["Ex"].append(Ex_slice)
            self.fields["Ey"].append(Ey_slice)
            self.fields["Ez"].append(Ez_slice)
            self.fields["Hx"].append(Hx_slice)
            self.fields["Hy"].append(Hy_slice)
            self.fields["Hz"].append(Hz_slice)
            self.fields["t"].append(t)

        if self.accumulate_power:
            self._calculate_power_3d(
                Ex_slice, Ey_slice, Ez_slice, Hx_slice, Hy_slice, Hz_slice, t, dx, dy
            )

        self.last_record_step = step
        self._manage_memory()

        if self.live_update and (len(self.fields["t"]) % self.update_interval == 0):
            self._update_live_plot_3d()

    def record_fields(self, *args, **kwargs):
        """Generic field recording method that delegates to 2D or 3D."""
        if self.is_3d and len(args) >= 6:
            # 3D: Ex, Ey, Ez, Hx, Hy, Hz, t, dx, dy, dz, step
            self.record_fields_3d(*args, **kwargs)
        else:
            # 2D: Ez, Hx, Hy, t, dx, dy, step
            self.record_fields_2d(*args, **kwargs)

    def _calculate_power_2d(self, Ez_values, Hx_values, Hy_values, t, dx, dy):
        """Calculate Poynting vector and power for 2D fields.

        Power is computed as the integral of the Poynting vector magnitude
        over the monitor line, properly normalized by grid cell area.
        """
        Ez_array = np.array(Ez_values)
        Hx_array = np.array(Hx_values)
        Hy_array = np.array(Hy_values)
        # Poynting vector S = E × H (units: W/m²)
        Sx = -Ez_array * Hy_array
        Sy = Ez_array * Hx_array
        # Power magnitude per grid point
        power_mag = np.sqrt(Sx**2 + Sy**2)
        # Total power = integral over monitor area (multiply by cell area for proper units)
        total_power = np.sum(power_mag) * dx * dy
        if self.power_accumulated is None:
            self.power_accumulated = power_mag
        else:
            self.power_accumulated += power_mag
        self.power_history.append(total_power)
        self.power_timestamps.append(float(t))
        self.power_accumulation_count += 1

    def _calculate_power_3d(self, Ex, Ey, Ez, Hx, Hy, Hz, t, dx, dy):
        """Calculate Poynting vector and power for 3D fields.

        Power is computed as the integral of the Poynting vector magnitude
        over the monitor plane, properly normalized by grid cell area.
        """
        # Poynting vector S = E × H (units: W/m²)
        Sx = Ey * Hz - Ez * Hy
        Sy = Ez * Hx - Ex * Hz
        Sz = Ex * Hy - Ey * Hx
        # Power magnitude per grid point
        power_mag = np.sqrt(Sx**2 + Sy**2 + Sz**2)
        # Total power = integral over monitor area (multiply by cell area for proper units)
        total_power = np.sum(power_mag) * dx * dy
        if self.power_accumulated is None:
            self.power_accumulated = power_mag.copy()
        else:
            self.power_accumulated += power_mag
        self.power_history.append(total_power)
        self.power_timestamps.append(float(t))
        self.power_accumulation_count += 1

    def _manage_memory(self):
        """Manage memory by limiting stored history."""
        if self.max_history_steps is None:
            return
        for field_name in self.fields:
            if len(self.fields[field_name]) > self.max_history_steps:
                # Remove oldest entries
                excess = len(self.fields[field_name]) - self.max_history_steps
                self.fields[field_name] = self.fields[field_name][excess:]
        # Also limit power history
        if len(self.power_history) > self.max_history_steps:
            excess = len(self.power_history) - self.max_history_steps
            self.power_history = self.power_history[excess:]
            self.power_timestamps = self.power_timestamps[excess:]

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
        except:
            pass  # Ignore plotting errors

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
        except:
            pass  # Ignore plotting errors

    def get_field_statistics(self):
        """Get statistical information about recorded fields."""
        if not self.fields["t"]:
            return {}
        stats = {
            "total_records": len(self.fields["t"]),
            "time_span": (
                self.fields["t"][-1] - self.fields["t"][0]
                if len(self.fields["t"]) > 1
                else 0
            ),
            "avg_power": np.mean(self.power_history) if self.power_history else 0,
            "max_power": np.max(self.power_history) if self.power_history else 0,
            "monitor_type": self.monitor_type,
            "is_3d": self.is_3d,
        }
        if self.is_3d:
            stats["plane_normal"] = self.plane_normal
            stats["plane_position"] = self.plane_position
            stats["plane_size"] = self.size
        else:
            stats["line_start"] = self.start
            stats["line_end"] = self.end
        return stats

    def save_data(self, filename, format="npz"):
        """Save recorded data to file."""
        if format == "npz":
            np.savez(
                filename,
                fields=self.fields,
                power_history=self.power_history,
                power_timestamps=self.power_timestamps,
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

    def add_to_plot(
        self, ax, facecolor="none", edgecolor="navy", alpha=1, linestyle="-"
    ):
        """Add monitor visualization to 2D plot."""
        if self.monitor_type == "line":
            # For line monitors, use edgecolor if facecolor is none
            color = edgecolor if facecolor == "none" else facecolor
            ax.plot(
                (self.start[0], self.end[0]),
                (self.start[1], self.end[1]),
                lw=4,
                color=color,
                label="Monitor",
                alpha=alpha,
            )
            ax.plot(
                (self.start[0], self.end[0]),
                (self.start[1], self.end[1]),
                lw=1,
                color=edgecolor,
                linestyle=linestyle,
            )
        else:
            # For 3D plane monitors, show projection on 2D plot
            if self.plane_normal == "z" or (
                hasattr(self, "end") and self.end is not None
            ):
                # xy plane - show as rectangle
                rect = MatplotlibRectangle(
                    (self.start[0], self.start[1]),
                    self.size[0],
                    self.size[1],
                    fill=(facecolor != "none"),
                    facecolor=facecolor,
                    alpha=alpha * 0.3,
                    edgecolor=edgecolor,
                    linestyle=linestyle,
                    linewidth=2,
                )
                ax.add_patch(rect)
                ax.text(
                    self.position[0],
                    self.position[1],
                    "Monitor\n(3D plane)",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=edgecolor,
                )

    def to_polygon(self):
        """Convert monitor to a polygon for 3D visualization."""
        if not hasattr(self, "vertices") or not self.vertices:
            return None

        # Import here to avoid circular imports
        # Create a polygon with the monitor vertices
        # Use a semi-transparent material for visualization
        from beamz.design.materials import Material
        from beamz.design.structures import Polygon

        monitor_material = Material(
            permittivity=1.0, permeability=1.0, conductivity=0.0
        )

        # Create polygon with monitor vertices
        polygon = Polygon(
            vertices=self.vertices,
            material=monitor_material,
            color="rgba(0,0,255,0.3)",  # Semi-transparent blue
            depth=0.001,  # Very thin for visualization
        )

        return polygon

    def plot_fields(self, field="Ez", figsize=(10, 6), time_index=-1):
        """Plot field data from the monitor.

        Args:
            field: Field component to plot ('Ez', 'Ex', 'Ey', 'Hx', 'Hy', 'Hz')
            figsize: Figure size tuple
            time_index: Time index to plot (-1 for latest)

        Returns:
            (fig, ax) tuple
        """
        if not self.fields["t"]:
            print("No field data recorded.")
            return None, None

        if field not in self.fields:
            print(
                f"Field '{field}' not available. Available fields: {list(self.fields.keys())}"
            )
            return None, None

        if not self.fields[field]:
            print(f"No data for field '{field}'.")
            return None, None

        fig, ax = plt.subplots(figsize=figsize)

        if self.monitor_type == "line":
            # 2D line monitor - plot field along the line
            field_data = self.fields[field][time_index]
            x_pos = range(len(field_data))
            ax.plot(x_pos, field_data, "b-", linewidth=2)
            ax.set_xlabel("Position along monitor line")
            ax.set_ylabel(f"{field} amplitude")
            ax.set_title(f'{field} at t = {self.fields["t"][time_index]:.2e} s')
            ax.grid(True, alpha=0.3)

        else:
            # 3D plane monitor - show 2D field distribution
            field_data = self.fields[field][time_index]
            im = ax.imshow(field_data, cmap="RdBu", origin="lower", aspect="auto")
            plt.colorbar(im, ax=ax, label=f"{field} amplitude")
            ax.set_xlabel("X index")
            ax.set_ylabel("Y index")
            ax.set_title(f'{field} at t = {self.fields["t"][time_index]:.2e} s')

        plt.tight_layout()
        return fig, ax

    def plot_power(self, figsize=(10, 6), log_scale=False, db_scale=False):
        """Plot power history from the monitor.

        Args:
            figsize: Figure size tuple
            log_scale: Use logarithmic scale
            db_scale: Use dB scale (10*log10)

        Returns:
            (fig, ax) tuple
        """
        if not self.power_history:
            print("No power data recorded.")
            return None, None

        fig, ax = plt.subplots(figsize=figsize)

        time_steps = range(len(self.power_history))
        power_data = np.array(self.power_history)

        if db_scale:
            # Convert to dB scale
            power_data = 10 * np.log10(np.maximum(power_data, 1e-12))  # Avoid log(0)
            ax.plot(time_steps, power_data, "r-", linewidth=2)
            ax.set_ylabel("Power (dB)")
        elif log_scale:
            ax.semilogy(time_steps, power_data, "r-", linewidth=2)
            ax.set_ylabel("Power (log scale)")
        else:
            ax.plot(time_steps, power_data, "r-", linewidth=2)
            ax.set_ylabel("Power")

        ax.set_xlabel("Time step")
        ax.set_title("Power vs Time")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax

    def animate_fields(
        self, field="Ez", figsize=(8, 6), interval=100, save_filename=None
    ):
        """Create an animation of field evolution.

        Args:
            field: Field component to animate
            figsize: Figure size tuple
            interval: Animation interval in milliseconds
            save_filename: Optional filename to save animation

        Returns:
            Animation object
        """
        if not self.fields["t"] or field not in self.fields:
            print(f"No data available for field '{field}'.")
            return None

        from matplotlib.animation import FuncAnimation

        fig, ax = plt.subplots(figsize=figsize)

        if self.monitor_type == "line":
            # Line plot animation
            (line,) = ax.plot([], [], "b-", linewidth=2)
            ax.set_xlabel("Position along monitor line")
            ax.set_ylabel(f"{field} amplitude")

            # Set fixed limits based on data range
            all_data = np.concatenate(self.fields[field])
            ax.set_xlim(0, len(self.fields[field][0]))
            ax.set_ylim(np.min(all_data), np.max(all_data))

            def animate(frame):
                field_data = self.fields[field][frame]
                x_pos = range(len(field_data))
                line.set_data(x_pos, field_data)
                ax.set_title(f'{field} at t = {self.fields["t"][frame]:.2e} s')
                return (line,)

        else:
            # 2D plot animation
            field_data = self.fields[field][0]
            im = ax.imshow(
                field_data, cmap="RdBu", origin="lower", aspect="auto", animated=True
            )
            plt.colorbar(im, ax=ax, label=f"{field} amplitude")
            ax.set_xlabel("X index")
            ax.set_ylabel("Y index")

            # Set fixed color limits
            all_data = np.array(self.fields[field])
            vmin, vmax = np.min(all_data), np.max(all_data)
            im.set_clim(vmin, vmax)

            def animate(frame):
                field_data = self.fields[field][frame]
                im.set_array(field_data)
                ax.set_title(f'{field} at t = {self.fields["t"][frame]:.2e} s')
                return [im]

        anim = FuncAnimation(
            fig,
            animate,
            frames=len(self.fields["t"]),
            interval=interval,
            blit=True,
            repeat=True,
        )

        if save_filename:
            anim.save(save_filename, writer="pillow", fps=1000 // interval)
            print(f"Animation saved to {save_filename}")

        plt.tight_layout()
        return anim

    def get_field_at_time(self, field="Ez", time_value=None, time_index=None):
        """Get field data at a specific time.

        Args:
            field: Field component to retrieve
            time_value: Specific time value (will find closest)
            time_index: Specific time index

        Returns:
            Field data array
        """
        if not self.fields["t"] or field not in self.fields:
            return None

        if time_index is not None:
            if 0 <= time_index < len(self.fields[field]):
                return self.fields[field][time_index]
            else:
                return None

        if time_value is not None:
            # Find closest time index
            times = np.array(self.fields["t"])
            time_index = np.argmin(np.abs(times - time_value))
            return self.fields[field][time_index]

        # Return latest data
        return self.fields[field][-1] if self.fields[field] else None

    def get_power_statistics(self):
        """Get power statistics from recorded data.

        Returns:
            Dictionary with power statistics
        """
        if not self.power_history:
            return {}

        power_array = np.array(self.power_history)
        return {
            "mean_power": np.mean(power_array),
            "max_power": np.max(power_array),
            "min_power": np.min(power_array),
            "std_power": np.std(power_array),
            "total_energy": np.sum(power_array),
            "peak_to_average_ratio": (
                np.max(power_array) / np.mean(power_array)
                if np.mean(power_array) > 0
                else 0
            ),
        }

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
            )
