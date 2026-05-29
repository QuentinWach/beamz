import logging
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from beamz.const import µm
from beamz.devices._placement import (
    line_region_points,
    plane_region_slices,
    snap_axis_aligned_line_region,
    snap_plane_region,
)
from beamz.devices._runtime import RuntimeStateProxy
from beamz.devices.ports import Port, _normalize_direction, _normalize_polarization
from beamz.shared_kernels import (
    full_tm_xy_component_to_centered_grid,
    is_full_tm_xy_lattice,
    monitor_dft_sample_scale,
    monitor_dft_should_accumulate,
    monitor_dft_window_weight,
    monitor_records_on_step,
    poynting_flux_2d,
    poynting_flux_3d,
)

logger = logging.getLogger(__name__)


def _normalize_center_size(center, size):
    center = tuple(float(v) for v in center)
    size = tuple(float(v) for v in size)
    if len(center) == 2:
        center = (center[0], center[1], 0.0)
    if len(size) == 2:
        size = (size[0], size[1], 0.0)
    if len(center) != 3 or len(size) != 3:
        raise ValueError("center and size must be 2D or 3D coordinate tuples.")
    return center, size


def _plane_start_end_from_center_size(center, size):
    center, size = _normalize_center_size(center, size)
    abs_size = np.abs(np.asarray(size, dtype=float))
    normal_index = int(np.argmin(abs_size))
    lower = [center[i] - 0.5 * size[i] for i in range(3)]
    upper = [center[i] + 0.5 * size[i] for i in range(3)]
    lower[normal_index] = center[normal_index]
    upper[normal_index] = center[normal_index]
    return tuple(lower), tuple(upper)


def _interp_complex_1d(
    values: np.ndarray,
    src_coords: np.ndarray,
    dst_coords: np.ndarray,
) -> np.ndarray:
    src = np.asarray(src_coords, dtype=np.float64).reshape(-1)
    dst = np.asarray(dst_coords, dtype=np.float64).reshape(-1)
    arr = np.asarray(values, dtype=np.complex128).reshape(src.size, -1)
    out = np.empty((dst.size, arr.shape[1]), dtype=np.complex128)
    for col in range(arr.shape[1]):
        out[:, col] = np.interp(dst, src, arr[:, col].real) + 1j * np.interp(
            dst, src, arr[:, col].imag
        )
    return out


def _plane_axes_for_normal_3d(axis: str) -> tuple[str, str]:
    axis = str(axis).lower()
    mapping = {
        "x": ("z", "y"),
        "y": ("z", "x"),
        "z": ("y", "x"),
    }
    try:
        return mapping[axis]
    except KeyError as exc:
        raise ValueError(f"Unsupported plane normal {axis!r}.") from exc


def _line_normal_2d(start, end) -> tuple[str, float] | None:
    """Return the signed Cartesian normal for an axis-aligned 2D monitor line."""
    if start is None or end is None:
        return None
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    dx = x1 - x0
    dy = y1 - y0
    tol = 1e-12 * max(abs(dx), abs(dy), 1.0)
    if abs(dx) <= tol and abs(dy) > tol:
        return "x", 1.0 if dy >= 0.0 else -1.0
    if abs(dy) <= tol and abs(dx) > tol:
        return "y", -1.0 if dx >= 0.0 else 1.0
    return None


def _line_integral_scale_2d(normal_axis: str, dx: float, dy: float) -> float:
    """Return the 2D line element for a monitor normal to `normal_axis`."""
    axis = str(normal_axis).lower()
    if axis == "x":
        return float(dy)
    if axis == "y":
        return float(dx)
    return float(0.5 * (float(dx) + float(dy)))


@dataclass
class _MonitorState:
    """Mutable runtime state kept separate from Monitor configuration."""

    fields: dict[str, list]
    power_spectrum: np.ndarray
    _frequency_flux_spectrum_legacy: np.ndarray | None
    objective_value: Optional[float]
    power_accumulated: np.ndarray | None
    energy_history: list
    power_history: list
    power_timestamps: list
    power_accumulation_count: int
    step_count: int
    last_record_step: int
    _dft_accum: dict
    _dft_weight_sum: np.ndarray
    _dft_sample_count: int
    _dft_phase: np.ndarray
    _dft_last_t: float | None
    _dft_last_dt: float | None
    _dft_last_rot: np.ndarray | None
    _dft_base_dt: float | None

    @classmethod
    def create(
        cls, *, dft_frequencies: np.ndarray, power_spectrum_frequencies: np.ndarray
    ):
        fields = {
            "Ex": [],
            "Ey": [],
            "Ez": [],
            "Hx": [],
            "Hy": [],
            "Hz": [],
            "t": [],
        }
        return cls(
            fields=fields,
            power_spectrum=np.zeros(
                power_spectrum_frequencies.shape, dtype=np.complex64
            ),
            _frequency_flux_spectrum_legacy=None,
            objective_value=None,
            power_accumulated=None,
            energy_history=[],
            power_history=[],
            power_timestamps=[],
            power_accumulation_count=0,
            step_count=0,
            last_record_step=-1,
            _dft_accum={},
            _dft_weight_sum=np.zeros(dft_frequencies.size, dtype=float),
            _dft_sample_count=0,
            _dft_phase=np.ones(dft_frequencies.size, dtype=np.complex128),
            _dft_last_t=None,
            _dft_last_dt=None,
            _dft_last_rot=None,
            _dft_base_dt=None,
        )


class Monitor(RuntimeStateProxy):
    _RUNTIME_ATTRS = {
        "fields",
        "power_spectrum",
        "_frequency_flux_spectrum_legacy",
        "objective_value",
        "power_accumulated",
        "energy_history",
        "power_history",
        "power_timestamps",
        "power_accumulation_count",
        "step_count",
        "last_record_step",
        "_dft_accum",
        "_dft_weight_sum",
        "_dft_sample_count",
        "_dft_phase",
        "_dft_last_t",
        "_dft_last_dt",
        "_dft_last_rot",
        "_dft_base_dt",
    }

    @property
    def frequency_points(self):
        warnings.warn(
            "Monitor.frequency_points is deprecated; use "
            "Monitor.power_spectrum_frequencies instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.power_spectrum_frequencies

    @frequency_points.setter
    def frequency_points(self, value):
        warnings.warn(
            "Monitor.frequency_points is deprecated; use "
            "Monitor.power_spectrum_frequencies instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.power_spectrum_frequencies = np.asarray(value, dtype=np.float64).ravel()
        self.accumulate_frequency = bool(self.power_spectrum_frequencies.size > 0)

    @property
    def frequency_flux_spectrum(self):
        warnings.warn(
            "Monitor.frequency_flux_spectrum is deprecated; use "
            "Monitor.power_spectrum for the time-domain power spectrum, or "
            "Monitor.get_dft_flux() for phasor DFT flux.",
            DeprecationWarning,
            stacklevel=2,
        )
        legacy = getattr(self, "_frequency_flux_spectrum_legacy", None)
        if legacy is not None:
            return legacy
        return self.power_spectrum

    @frequency_flux_spectrum.setter
    def frequency_flux_spectrum(self, value):
        warnings.warn(
            "Monitor.frequency_flux_spectrum is deprecated; use "
            "Monitor.power_spectrum instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        arr = np.asarray(value, dtype=np.complex64)
        self.power_spectrum = arr
        self._frequency_flux_spectrum_legacy = arr

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
        dft_normalization="native",
        dft_length_unit=µm,
        objective_function: Optional[Callable[["Monitor"], float]] = None,
        name: Optional[str] = None,
        frequency_points=None,
        frequency_record_interval=1,
        power_spectrum_frequencies=None,
        power_spectrum_record_interval=None,
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
        self.dft_normalization = str(dft_normalization).lower()
        if self.dft_normalization not in {"native", "physical"}:
            raise ValueError(
                "dft_normalization must be one of ['native', 'physical'], "
                f"got {dft_normalization!r}"
            )
        self.dft_length_unit = float(dft_length_unit)
        if not np.isfinite(self.dft_length_unit) or self.dft_length_unit <= 0.0:
            raise ValueError(
                "dft_length_unit must be a positive finite length in meters, "
                f"got {dft_length_unit!r}"
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
        if power_spectrum_frequencies is not None and frequency_points is not None:
            raise ValueError(
                "Use either power_spectrum_frequencies or deprecated frequency_points, "
                "not both."
            )
        if frequency_points is not None:
            warnings.warn(
                "Monitor(frequency_points=...) is deprecated; use "
                "Monitor(power_spectrum_frequencies=...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            power_spectrum_frequencies = frequency_points
        if power_spectrum_frequencies is None:
            freq_arr = np.zeros((0,), dtype=np.float64)
        else:
            freq_arr = np.asarray(power_spectrum_frequencies, dtype=np.float64).ravel()
            if freq_arr.ndim != 1:
                raise ValueError(
                    "power_spectrum_frequencies must be a 1D sequence of "
                    "frequencies in Hz"
                )
            if np.any(freq_arr < 0.0):
                raise ValueError(
                    "power_spectrum_frequencies must be non-negative frequencies in Hz"
                )
        self.power_spectrum_frequencies = freq_arr
        if power_spectrum_record_interval is None:
            power_spectrum_record_interval = frequency_record_interval
        self.power_spectrum_record_interval = max(
            1, int(power_spectrum_record_interval)
        )
        self.frequency_record_interval = self.power_spectrum_record_interval
        self.accumulate_frequency = bool(freq_arr.size > 0)
        self.objective_function = objective_function
        self.name = name
        self._snapped_region = None
        self._snap_signature = None
        self._resolution = None
        self._field_shape = None

        # Determine if this is a 3D monitor based on input parameters
        self.is_3d = self._determine_3d_mode(start, end, design)

        # Runtime recording state is kept separate from the monitor spec.
        self._state = _MonitorState.create(
            dft_frequencies=self.dft_frequencies,
            power_spectrum_frequencies=self.power_spectrum_frequencies,
        )

        self.update_interval = 10

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
        region = self._snapped_region
        if region is not None:
            start = region.start
            end = region.end
            x_min, y_min, z_min = (float(v) for v in start)
            x_max, y_max, z_max = (float(v) for v in end)
            if self.plane_normal == "z":
                z = float(region.plane_coord)
                return [
                    (x_min, y_min, z),
                    (x_max, y_min, z),
                    (x_max, y_max, z),
                    (x_min, y_max, z),
                ]
            if self.plane_normal == "y":
                y = float(region.plane_coord)
                return [
                    (x_min, y, z_min),
                    (x_max, y, z_min),
                    (x_max, y, z_max),
                    (x_min, y, z_max),
                ]
            x = float(region.plane_coord)
            return [
                (x, y_min, z_min),
                (x, y_max, z_min),
                (x, y_max, z_max),
                (x, y_min, z_max),
            ]

        if self.plane_normal == "z":
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
        region = self._snapped_region
        if region is not None:
            return tuple(float(v) for v in region.center)

        if self.plane_normal == "z":
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
        self._resolution = float(dx)
        self._field_shape = None
        snapped = self.get_snapped_region(dx=dx, dy=dy)
        if snapped is not None:
            return line_region_points(snapped)

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

    def _line_sample_coords_2d(self, dx, dy):
        snapped = self.get_snapped_region(dx=dx, dy=dy)
        if snapped is None:
            return None
        if snapped.normal_axis == "x":
            interval = snapped.axis_interval("y")
            if interval is None:
                return None
            y = (
                np.arange(int(interval.start), int(interval.stop), dtype=np.float64)
                + 0.5
            ) * float(dy)
            if self.end is not None:
                x_coord = 0.5 * (float(self.start[0]) + float(self.end[0]))
            else:
                x_coord = float(self.start[0])
            x = np.full(y.shape, x_coord, dtype=np.float64)
            return x, y
        interval = snapped.axis_interval("x")
        if interval is None:
            return None
        x = (
            np.arange(int(interval.start), int(interval.stop), dtype=np.float64) + 0.5
        ) * float(dx)
        if self.end is not None:
            y_coord = 0.5 * (float(self.start[1]) + float(self.end[1]))
        else:
            y_coord = float(self.start[1])
        y = np.full(x.shape, y_coord, dtype=np.float64)
        return x, y

    @staticmethod
    def _sample_component_line_2d(arr, x_coords, y_coords, x_targets, y_targets):
        field = np.asarray(arr, dtype=np.complex128)
        if field.ndim != 2:
            raise ValueError(f"Expected 2D field array, got shape {field.shape}")
        x_src = np.asarray(x_coords, dtype=np.float64).reshape(-1)
        y_src = np.asarray(y_coords, dtype=np.float64).reshape(-1)
        x_tgt = np.asarray(x_targets, dtype=np.float64).reshape(-1)
        y_tgt = np.asarray(y_targets, dtype=np.float64).reshape(-1)
        row_interp = _interp_complex_1d(field.T, x_src, x_tgt)
        samples = _interp_complex_1d(row_interp.T, y_src, y_tgt).reshape(
            y_tgt.size, x_tgt.size
        )
        if x_tgt.size == y_tgt.size:
            return np.diag(samples)
        if x_tgt.size == 1:
            return samples[:, 0]
        if y_tgt.size == 1:
            return samples[0, :]
        raise ValueError(
            "Cannot collapse 2D monitor interpolation with varying x and y targets: "
            f"x={x_tgt.size}, y={y_tgt.size}"
        )

    def get_grid_slice_3d(self, dx, dy, dz, field_shape):
        """Get grid slice for 3D plane monitor.
        Returns (z_idx, y_idx, x_idx) consistent with simulation array order (z, y, x).
        One of these will be an integer, the other two will be slice objects.
        """
        self._resolution = float(dx)
        self._field_shape = tuple(int(v) for v in field_shape)
        snapped = self.get_snapped_region(dx=dx, dy=dy, dz=dz, field_shape=field_shape)
        if snapped is not None:
            return plane_region_slices(snapped)

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

    @staticmethod
    def _uniform_axis_centers(lower: float, upper: float, count: int) -> np.ndarray:
        n = max(0, int(count))
        if n <= 0:
            return np.zeros((0,), dtype=np.float64)
        lo = float(lower)
        hi = float(upper)
        if n == 1:
            return np.asarray([0.5 * (lo + hi)], dtype=np.float64)
        step = (hi - lo) / float(n)
        return lo + (np.arange(n, dtype=np.float64) + 0.5) * step

    def _analysis_plane_bounds_3d(self) -> tuple[dict[str, tuple[float, float]], float]:
        if not self.is_3d:
            raise ValueError("3D analysis-plane bounds requested on a 2D monitor.")
        axis = str(self.plane_normal).lower()
        if axis == "x":
            bounds = {
                "y": (
                    float(self.start[1]),
                    float(self.start[1]) + float(self.size[0]),
                ),
                "z": (
                    float(self.start[2]),
                    float(self.start[2]) + float(self.size[1]),
                ),
            }
            plane_pos = float(self.plane_position)
        elif axis == "y":
            bounds = {
                "x": (
                    float(self.start[0]),
                    float(self.start[0]) + float(self.size[0]),
                ),
                "z": (
                    float(self.start[2]),
                    float(self.start[2]) + float(self.size[1]),
                ),
            }
            plane_pos = float(self.plane_position)
        else:
            bounds = {
                "x": (
                    float(self.start[0]),
                    float(self.start[0]) + float(self.size[0]),
                ),
                "y": (
                    float(self.start[1]),
                    float(self.start[1]) + float(self.size[1]),
                ),
            }
            plane_pos = float(self.plane_position)
        return bounds, plane_pos

    @staticmethod
    def _base_grid_shape_from_fields_3d(*arrays) -> tuple[int, int, int]:
        shapes = [tuple(int(v) for v in np.asarray(arr).shape) for arr in arrays]
        if not shapes:
            raise ValueError("At least one 3D field array is required.")
        return tuple(max(shape[axis] for shape in shapes) for axis in range(3))

    def get_analysis_plane_coords_3d(
        self,
        *,
        dx: float,
        dy: float,
        dz: float,
        field_shape: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        snapped = self.get_snapped_region(
            dx=dx,
            dy=dy,
            dz=dz,
            field_shape=field_shape,
        )
        if snapped is None:
            raise ValueError("3D analysis plane requires a snapped monitor region.")
        axis0, axis1 = _plane_axes_for_normal_3d(self.plane_normal)
        interval0 = snapped.axis_interval(axis0)
        interval1 = snapped.axis_interval(axis1)
        if interval0 is None or interval1 is None:
            raise ValueError(
                f"Monitor '{self.name}' is missing tangential intervals for analysis-plane sampling."
            )
        bounds, _ = self._analysis_plane_bounds_3d()
        lo0, hi0 = bounds[axis0]
        lo1, hi1 = bounds[axis1]
        return (
            self._uniform_axis_centers(
                lo0, hi0, int(interval0.stop) - int(interval0.start)
            ),
            self._uniform_axis_centers(
                lo1, hi1, int(interval1.stop) - int(interval1.start)
            ),
        )

    def _sample_component_plane_3d(
        self,
        component: str,
        arr: np.ndarray,
        *,
        dx: float,
        dy: float,
        dz: float,
        base_shape: tuple[int, int, int],
        target0: np.ndarray,
        target1: np.ndarray,
    ) -> np.ndarray:
        from beamz.simulation.yee import component_coordinates_3d_um

        field = np.asarray(arr, dtype=np.complex128)
        axis = str(self.plane_normal).lower()
        axis0, axis1 = _plane_axes_for_normal_3d(axis)
        bounds, plane_pos = self._analysis_plane_bounds_3d()
        coords_um = component_coordinates_3d_um(
            component,
            tuple(int(v) for v in base_shape),
            float(dx / µm),
        )
        src0 = np.asarray(coords_um[axis0], dtype=np.float64) * float(µm)
        src1 = np.asarray(coords_um[axis1], dtype=np.float64) * float(µm)
        srcn = np.asarray(coords_um[axis], dtype=np.float64) * float(µm)
        axis_order = {"z": 0, "y": 1, "x": 2}
        moved = np.moveaxis(
            field,
            [axis_order[axis0], axis_order[axis1], axis_order[axis]],
            [0, 1, 2],
        )
        if moved.shape != (src0.size, src1.size, srcn.size):
            raise ValueError(
                f"Unexpected 3D component layout for '{component}': "
                f"{moved.shape} vs {(src0.size, src1.size, srcn.size)}"
            )

        lo0, hi0 = bounds[axis0]
        lo1, hi1 = bounds[axis1]
        steps = {"x": float(dx), "y": float(dy), "z": float(dz)}
        pad0 = 0.5 * steps[axis0]
        pad1 = 0.5 * steps[axis1]
        idx0_lo = max(0, int(np.searchsorted(src0, lo0 - pad0, side="left")) - 1)
        idx0_hi = min(
            src0.size, int(np.searchsorted(src0, hi0 + pad0, side="right")) + 1
        )
        idx1_lo = max(0, int(np.searchsorted(src1, lo1 - pad1, side="left")) - 1)
        idx1_hi = min(
            src1.size, int(np.searchsorted(src1, hi1 + pad1, side="right")) + 1
        )
        src0_local = src0[idx0_lo:idx0_hi]
        src1_local = src1[idx1_lo:idx1_hi]
        moved = moved[idx0_lo:idx0_hi, idx1_lo:idx1_hi, :]

        flat = moved.reshape(src0_local.size * src1_local.size, srcn.size).T
        normal_interp = _interp_complex_1d(
            flat,
            srcn,
            np.asarray([plane_pos], dtype=np.float64),
        )[0].reshape(src0_local.size, src1_local.size)
        mid = _interp_complex_1d(
            normal_interp.T, src1_local, np.asarray(target1, dtype=np.float64)
        ).T
        out = _interp_complex_1d(mid, src0_local, np.asarray(target0, dtype=np.float64))
        return np.asarray(out, dtype=np.complex128)

    def get_snapped_region(self, *, dx, dy, dz=None, field_shape=None):
        """Return a canonical snapped region for sampling and diagnostics."""
        if self.is_3d:
            if dz is None or field_shape is None:
                return self._snapped_region
            signature = ("3d", float(dx), float(dy), float(dz), tuple(field_shape))
            if self._snap_signature != signature:
                self._snapped_region = snap_plane_region(
                    start=tuple(float(v) for v in self.start),
                    end=(
                        None
                        if getattr(self, "end", None) is None
                        else tuple(float(v) for v in self.end)
                    ),
                    plane_normal=str(getattr(self, "plane_normal", "z")).lower(),
                    size=(
                        None
                        if getattr(self, "end", None) is not None
                        else tuple(float(v) for v in getattr(self, "size", (0.0, 0.0)))
                    ),
                    dx=float(dx),
                    dy=float(dy),
                    dz=float(dz),
                    shape=tuple(int(v) for v in field_shape),
                )
                self._snap_signature = signature
                self.position = tuple(float(v) for v in self._snapped_region.center)
                self.vertices = self._generate_plane_vertices()
            return self._snapped_region

        if self.end is None:
            return None
        signature = ("2d", float(dx), float(dy))
        if self._snap_signature != signature:
            shape = None
            if self.design is not None:
                shape = (
                    max(
                        1,
                        int(
                            round(
                                float(getattr(self.design, "height", 0.0)) / float(dy)
                            )
                        ),
                    ),
                    max(
                        1,
                        int(
                            round(float(getattr(self.design, "width", 0.0)) / float(dx))
                        ),
                    ),
                )
            if shape is not None:
                self._snapped_region = snap_axis_aligned_line_region(
                    tuple(float(v) for v in self.start),
                    tuple(float(v) for v in self.end),
                    float(dx),
                    float(dy),
                    shape,
                )
            else:
                self._snapped_region = None
            self._snap_signature = signature
        return self._snapped_region

    def should_record(self, step):
        """Check if this step should be recorded based on interval."""
        return bool(monitor_records_on_step(step, self.record_interval))

    def _dft_should_accumulate(self, step, t):
        return bool(
            monitor_dft_should_accumulate(
                bool(self.dft_enabled and self.dft_frequencies.size > 0),
                step,
                t,
                self.dft_t_start,
                np.inf if self.dft_t_end is None else self.dft_t_end,
                self.dft_record_interval,
            )
        )

    def _dft_weight(self, t):
        return float(
            monitor_dft_window_weight(
                t,
                self.dft_t_start,
                np.inf if self.dft_t_end is None else self.dft_t_end,
                self.dft_window == "hann",
            )
        )

    def _init_or_get_dft_accum(self, component, npoints):
        arr = self._dft_accum.get(component)
        shape = (self.dft_frequencies.size, int(npoints))
        if arr is None or arr.shape != shape:
            arr = np.zeros(shape, dtype=np.complex128)
            self._dft_accum[component] = arr
        return arr

    def _dft_current_phase(self, t):
        t_now = float(t)
        if self._dft_last_t is None:
            self._dft_phase = np.exp(1j * 2.0 * np.pi * self.dft_frequencies * t_now)
            self._dft_last_t = t_now
            return self._dft_phase
        dt = t_now - float(self._dft_last_t)
        if abs(dt) > 0.0:
            if (
                self._dft_last_dt is None
                or self._dft_last_rot is None
                or abs(dt - float(self._dft_last_dt)) > 1e-18
            ):
                self._dft_last_dt = dt
                self._dft_last_rot = np.exp(
                    1j * 2.0 * np.pi * self.dft_frequencies * dt
                )
            self._dft_phase = self._dft_phase * self._dft_last_rot
        self._dft_last_t = t_now
        return self._dft_phase

    def _dft_physical_base_dt(self, t, step):
        base_dt = self._dft_base_dt
        if base_dt is not None and float(base_dt) > 0.0:
            return float(base_dt)
        step_idx = int(step) if step is not None else -1
        if step_idx >= 0:
            elapsed = float(t) - float(self.dft_t_start)
            if elapsed > 0.0:
                return elapsed / float(step_idx + 1)
        return None

    def _dft_sample_scale(self, t, step):
        w = float(self._dft_weight(t))
        if w <= 0.0:
            return 0.0
        base_dt = self._dft_physical_base_dt(t, step)
        if self.dft_normalization == "physical" and (base_dt is None or base_dt <= 0.0):
            return 0.0
        return float(
            monitor_dft_sample_scale(
                w,
                normalization_code=1 if self.dft_normalization == "physical" else 0,
                base_dt=1.0 if base_dt is None else float(base_dt),
                record_interval=float(self.dft_record_interval),
                length_unit=float(self.dft_length_unit),
            )
        )

    @staticmethod
    def _reshape_dft_component(
        accum: np.ndarray,
        nfreq: int,
        component: str,
    ) -> np.ndarray:
        arr = np.asarray(accum, dtype=np.complex128)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            if arr.shape[0] == nfreq:
                return arr[:, None]
            if nfreq == 1:
                return arr.reshape(1, -1)
            raise ValueError(
                "Cannot infer DFT frequency axis for component "
                f"'{component}': shape={arr.shape}, nfreq={nfreq}"
            )
        if arr.shape[0] != nfreq:
            raise ValueError(
                "DFT accumulator must use frequency on axis 0 for component "
                f"'{component}': shape={arr.shape}, nfreq={nfreq}"
            )
        return arr.reshape(nfreq, -1)

    def _update_dft(self, t, component_vectors, *, step=None):
        if not component_vectors:
            return
        sample_scale = float(self._dft_sample_scale(t, step))
        if sample_scale <= 0.0:
            return
        phase = self._dft_current_phase(t)
        w = float(self._dft_weight(t))
        self._dft_weight_sum = self._dft_weight_sum + w
        self._dft_sample_count += 1
        for comp, vec in component_vectors.items():
            arr = np.asarray(vec, dtype=np.complex128).reshape(-1)
            accum = self._init_or_get_dft_accum(comp, arr.size)
            accum += (sample_scale * phase)[:, None] * arr[None, :]
            self._dft_accum[comp] = accum

    def reset_dft(self):
        self._dft_accum = {}
        self._dft_weight_sum = np.zeros(self.dft_frequencies.size, dtype=float)
        self._dft_sample_count = 0
        self._dft_phase = np.ones(self.dft_frequencies.size, dtype=np.complex128)
        self._dft_last_t = None
        self._dft_last_dt = None
        self._dft_last_rot = None

    def get_dft_frequencies(self):
        return np.asarray(self.dft_frequencies, dtype=float)

    def get_dft_component(self, component: str):
        comp = str(component)
        if comp not in self._dft_accum:
            raise ValueError(f"No DFT data recorded for component '{comp}'.")
        nfreq = int(self.dft_frequencies.size)
        if nfreq <= 0:
            raise ValueError(
                f"Monitor '{self.name}' has no configured DFT frequencies."
            )
        accum = self._reshape_dft_component(self._dft_accum[comp], nfreq, comp)
        if self.dft_normalization == "physical":
            return accum
        scale = np.maximum(
            np.asarray(self._dft_weight_sum, dtype=float), 1e-18
        ).reshape(nfreq, 1)
        return (2.0 / scale) * accum

    def _get_dft_component_for_flux(self, component: str):
        values = np.asarray(self.get_dft_component(component), dtype=np.complex128)
        if not str(component).startswith("H"):
            return values
        dt = self._dft_base_dt
        if dt is None or float(dt) == 0.0:
            return values
        phase = np.exp(
            -1j
            * np.pi
            * np.asarray(self.dft_frequencies, dtype=float).reshape(-1, 1)
            * float(dt)
        )
        return values * phase

    def _normal_axis_and_sign(self) -> tuple[str, float]:
        if self.is_3d:
            return str(getattr(self, "plane_normal", "z")).lower(), 1.0
        line_normal = _line_normal_2d(
            getattr(self, "start", None),
            getattr(self, "end", None),
        )
        snapped = self.get_snapped_region(
            dx=self._resolution or 1.0,
            dy=self._resolution or 1.0,
        )
        if snapped is not None:
            axis = str(snapped.normal_axis).lower()
            sign = (
                float(line_normal[1])
                if line_normal is not None and line_normal[0] == axis
                else 1.0
            )
            return axis, sign
        if line_normal is not None:
            return line_normal
        return "x", 1.0

    def get_dft_flux(self) -> np.ndarray:
        """Return phasor flux 0.5 Re integral n . (E(w) x H*(w)).

        2D line monitors integrate over `dl`; 3D plane monitors integrate over
        `dA`.
        """
        dx = float(self._resolution or 1.0)
        axis, sign = self._normal_axis_and_sign()
        if self.is_3d:
            measure = dx * dx
            ex = np.asarray(self._get_dft_component_for_flux("Ex"), dtype=np.complex128)
            ey = np.asarray(self._get_dft_component_for_flux("Ey"), dtype=np.complex128)
            ez = np.asarray(self._get_dft_component_for_flux("Ez"), dtype=np.complex128)
            hx = np.asarray(self._get_dft_component_for_flux("Hx"), dtype=np.complex128)
            hy = np.asarray(self._get_dft_component_for_flux("Hy"), dtype=np.complex128)
            hz = np.asarray(self._get_dft_component_for_flux("Hz"), dtype=np.complex128)
            sx = ey * np.conjugate(hz) - ez * np.conjugate(hy)
            sy = ez * np.conjugate(hx) - ex * np.conjugate(hz)
            sz = ex * np.conjugate(hy) - ey * np.conjugate(hx)
            component = {"x": sx, "y": sy, "z": sz}.get(axis, sz)
        else:
            measure = _line_integral_scale_2d(axis, dx, dx)
            ez = np.asarray(self._get_dft_component_for_flux("Ez"), dtype=np.complex128)
            if axis == "x":
                hy = np.asarray(
                    self._get_dft_component_for_flux("Hy"), dtype=np.complex128
                )
                component = -ez * np.conjugate(hy)
            else:
                hx = np.asarray(
                    self._get_dft_component_for_flux("Hx"), dtype=np.complex128
                )
                component = ez * np.conjugate(hx)
        return 0.5 * np.real(np.sum(sign * component, axis=1)) * measure

    def record_fields_2d(
        self, Ez, Hx, Hy, t, dx, dy, step=0, Ex=None, Ey=None, Hz=None
    ):
        """Record 2D field data."""
        do_record = self.should_record(step)
        do_dft = self._dft_should_accumulate(step, t)
        if not do_record and not do_dft:
            return
        grid_points = self.get_grid_points_2d(dx, dy)
        line_coords = self._line_sample_coords_2d(dx, dy)
        full_tm_xy = is_full_tm_xy_lattice(Ez, Hx, Hy)
        if full_tm_xy and line_coords is not None:
            from beamz.simulation.yee import tm_xy_full_component_coordinates_2d_um

            grid_shape = (int(Ez.shape[0]) - 1, int(Ez.shape[1]) - 1)
            x_targets, y_targets = line_coords

            def sample_full_tm(component, arr):
                if self.dft_normalization == "physical":
                    centered = np.asarray(
                        full_tm_xy_component_to_centered_grid(component, arr),
                        dtype=np.complex128,
                    )
                    centered_x = (
                        np.arange(centered.shape[1], dtype=np.float64) + 0.5
                    ) * float(dx)
                    centered_y = (
                        np.arange(centered.shape[0], dtype=np.float64) + 0.5
                    ) * float(dy)
                    return self._sample_component_line_2d(
                        centered,
                        centered_x,
                        centered_y,
                        x_targets,
                        y_targets,
                    )
                coords = tm_xy_full_component_coordinates_2d_um(
                    component, grid_shape, float(dx)
                )
                return self._sample_component_line_2d(
                    arr,
                    coords["x"],
                    coords["y"],
                    x_targets,
                    y_targets,
                )

            Ez_values = sample_full_tm("Ez", Ez).tolist()
            Hx_values = sample_full_tm("Hx", Hx).tolist()
            Hy_values = sample_full_tm("Hy", Hy).tolist()
        else:
            Ez_values, Hx_values, Hy_values = [], [], []
            for x_idx, y_idx in grid_points:
                if 0 <= y_idx < Ez.shape[0] and 0 <= x_idx < Ez.shape[1]:
                    val = Ez[y_idx, x_idx]
                    Ez_values.append(
                        complex(val) if np.iscomplexobj(val) else float(val)
                    )
                else:
                    Ez_values.append(0.0)
                if 0 <= y_idx < Hx.shape[0] and 0 <= x_idx < Hx.shape[1]:
                    val = Hx[y_idx, x_idx]
                    Hx_values.append(
                        complex(val) if np.iscomplexobj(val) else float(val)
                    )
                else:
                    Hx_values.append(0.0)
                if 0 <= y_idx < Hy.shape[0] and 0 <= x_idx < Hy.shape[1]:
                    val = Hy[y_idx, x_idx]
                    Hy_values.append(
                        complex(val) if np.iscomplexobj(val) else float(val)
                    )
                else:
                    Hy_values.append(0.0)

        def sample_xy_component(component, arr):
            if arr is None:
                return [0.0] * len(grid_points)
            if line_coords is None:
                return None
            from beamz.simulation.yee import component_coordinates_2d_um

            if Ex is not None and Ey is not None:
                grid_shape = (int(Ex.shape[0]), int(Ey.shape[1]))
            elif Ey is not None and Hz is not None:
                grid_shape = (int(Hz.shape[0]) + 1, int(Ey.shape[1]))
            elif Ex is not None and Hz is not None:
                grid_shape = (int(Ex.shape[0]), int(Hz.shape[1]) + 1)
            elif full_tm_xy:
                grid_shape = (int(Ez.shape[0]) - 1, int(Ez.shape[1]) - 1)
            else:
                return None
            coords = component_coordinates_2d_um(component, grid_shape, float(dx), "xy")
            x_targets, y_targets = line_coords
            try:
                return self._sample_component_line_2d(
                    arr,
                    coords["x"],
                    coords["y"],
                    x_targets,
                    y_targets,
                ).tolist()
            except Exception:
                return None

        sampled_ex = sample_xy_component("Ex", Ex)
        sampled_ey = sample_xy_component("Ey", Ey)
        sampled_hz = sample_xy_component("Hz", Hz)

        Ex_values, Ey_values, Hz_values = [], [], []
        if sampled_ex is not None and sampled_ey is not None and sampled_hz is not None:
            Ex_values = sampled_ex
            Ey_values = sampled_ey
            Hz_values = sampled_hz
        else:
            for x_idx, y_idx in grid_points:
                if (
                    Ex is not None
                    and 0 <= y_idx < Ex.shape[0]
                    and 0 <= x_idx < Ex.shape[1]
                ):
                    val = Ex[y_idx, x_idx]
                    Ex_values.append(
                        complex(val) if np.iscomplexobj(val) else float(val)
                    )
                else:
                    Ex_values.append(0.0)
                if (
                    Ey is not None
                    and 0 <= y_idx < Ey.shape[0]
                    and 0 <= x_idx < Ey.shape[1]
                ):
                    val = Ey[y_idx, x_idx]
                    Ey_values.append(
                        complex(val) if np.iscomplexobj(val) else float(val)
                    )
                else:
                    Ey_values.append(0.0)
                if (
                    Hz is not None
                    and 0 <= y_idx < Hz.shape[0]
                    and 0 <= x_idx < Hz.shape[1]
                ):
                    val = Hz[y_idx, x_idx]
                    Hz_values.append(
                        complex(val) if np.iscomplexobj(val) else float(val)
                    )
                else:
                    Hz_values.append(0.0)

        if do_record and self.should_record_fields:
            self.fields["Ex"].append(Ex_values)
            self.fields["Ey"].append(Ey_values)
            self.fields["Ez"].append(Ez_values)
            self.fields["Hx"].append(Hx_values)
            self.fields["Hy"].append(Hy_values)
            self.fields["Hz"].append(Hz_values)
            self.fields["t"].append(t)

        if do_dft:
            dft_components = self.dft_components or ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
            vectors = {}
            if "Ex" in dft_components:
                vectors["Ex"] = Ex_values
            if "Ey" in dft_components:
                vectors["Ey"] = Ey_values
            if "Ez" in dft_components:
                vectors["Ez"] = Ez_values
            if "Hx" in dft_components:
                vectors["Hx"] = Hx_values
            if "Hy" in dft_components:
                vectors["Hy"] = Hy_values
            if "Hz" in dft_components:
                vectors["Hz"] = Hz_values
            self._update_dft(t, vectors, step=step)

        if do_record and self.accumulate_power:
            self._calculate_power_2d(Ez_values, Hx_values, Hy_values, t, dx, dy)

        if do_record:
            self.last_record_step = step
        self._manage_memory()

    def record_fields_3d(self, Ex, Ey, Ez, Hx, Hy, Hz, t, dx, dy, dz, step=0):
        """Record 3D field data from plane slice."""
        do_record = self.should_record(step)
        do_dft = self._dft_should_accumulate(step, t)
        if not do_record and not do_dft:
            return
        base_shape = self._base_grid_shape_from_fields_3d(Ex, Ey, Ez, Hx, Hy, Hz)
        target0, target1 = self.get_analysis_plane_coords_3d(
            dx=dx,
            dy=dy,
            dz=dz,
            field_shape=base_shape,
        )
        Ex_slice = self._sample_component_plane_3d(
            "Ex",
            Ex,
            dx=dx,
            dy=dy,
            dz=dz,
            base_shape=base_shape,
            target0=target0,
            target1=target1,
        )
        Ey_slice = self._sample_component_plane_3d(
            "Ey",
            Ey,
            dx=dx,
            dy=dy,
            dz=dz,
            base_shape=base_shape,
            target0=target0,
            target1=target1,
        )
        Ez_slice = self._sample_component_plane_3d(
            "Ez",
            Ez,
            dx=dx,
            dy=dy,
            dz=dz,
            base_shape=base_shape,
            target0=target0,
            target1=target1,
        )
        Hx_slice = self._sample_component_plane_3d(
            "Hx",
            Hx,
            dx=dx,
            dy=dy,
            dz=dz,
            base_shape=base_shape,
            target0=target0,
            target1=target1,
        )
        Hy_slice = self._sample_component_plane_3d(
            "Hy",
            Hy,
            dx=dx,
            dy=dy,
            dz=dz,
            base_shape=base_shape,
            target0=target0,
            target1=target1,
        )
        Hz_slice = self._sample_component_plane_3d(
            "Hz",
            Hz,
            dx=dx,
            dy=dy,
            dz=dz,
            base_shape=base_shape,
            target0=target0,
            target1=target1,
        )

        if do_record and self.should_record_fields:
            self.fields["Ex"].append(Ex_slice)
            self.fields["Ey"].append(Ey_slice)
            self.fields["Ez"].append(Ez_slice)
            self.fields["Hx"].append(Hx_slice)
            self.fields["Hy"].append(Hy_slice)
            self.fields["Hz"].append(Hz_slice)
            self.fields["t"].append(t)

        if do_dft:
            dft_components = self.dft_components or ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
            vectors = {}
            if "Ex" in dft_components:
                vectors["Ex"] = Ex_slice.reshape(-1)
            if "Ey" in dft_components:
                vectors["Ey"] = Ey_slice.reshape(-1)
            if "Ez" in dft_components:
                vectors["Ez"] = Ez_slice.reshape(-1)
            if "Hx" in dft_components:
                vectors["Hx"] = Hx_slice.reshape(-1)
            if "Hy" in dft_components:
                vectors["Hy"] = Hy_slice.reshape(-1)
            if "Hz" in dft_components:
                vectors["Hz"] = Hz_slice.reshape(-1)
            self._update_dft(t, vectors, step=step)

        if do_record and self.accumulate_power:
            d0 = float(np.mean(np.diff(target0))) if target0.size > 1 else float(dx)
            d1 = float(np.mean(np.diff(target1))) if target1.size > 1 else float(dy)
            self._calculate_power_3d(
                Ex_slice, Ey_slice, Ez_slice, Hx_slice, Hy_slice, Hz_slice, t, d0, d1
            )

        if do_record:
            self.last_record_step = step
        self._manage_memory()

    def record_fields(self, *args, **kwargs):
        """Generic field recording method that delegates to 2D or 3D."""
        if self.is_3d and len(args) >= 6:
            # 3D: Ex, Ey, Ez, Hx, Hy, Hz, t, dx, dy, dz, step
            self.record_fields_3d(*args, **kwargs)
        else:
            # 2D: Ez, Hx, Hy, t, dx, dy, step
            self.record_fields_2d(*args, **kwargs)

    def _calculate_power_2d(self, Ez_values, Hx_values, Hy_values, t, dx, dy):
        """Calculate signed normal Poynting flux for 2D fields.

        Power history stores the 2D line integral
        P(t) = integral n . (E(t) x H(t)) dl.
        """
        Ez_array = np.array(Ez_values)
        Hx_array = np.array(Hx_values)
        Hy_array = np.array(Hy_values)
        axis, sign = self._normal_axis_and_sign()
        flux = np.asarray(
            poynting_flux_2d(Ez_array, Hx_array, Hy_array, axis, sign),
            dtype=np.complex128,
        )
        flux = np.real_if_close(flux)
        total_power = np.sum(flux) * _line_integral_scale_2d(axis, dx, dy)
        if self.power_accumulated is None:
            self.power_accumulated = flux
        else:
            self.power_accumulated += flux
        self.power_history.append(total_power)
        self.power_timestamps.append(float(t))
        self.power_accumulation_count += 1

    def _calculate_power_3d(self, Ex, Ey, Ez, Hx, Hy, Hz, t, dx, dy):
        """Calculate signed normal Poynting flux for 3D fields.

        Power history stores P(t) = integral n . (E(t) x H(t)) dA.
        """
        axis, sign = self._normal_axis_and_sign()
        flux = np.asarray(
            poynting_flux_3d(Ex, Ey, Ez, Hx, Hy, Hz, axis, sign),
            dtype=np.complex128,
        )
        flux = np.real_if_close(flux)
        total_power = np.sum(flux) * dx * dy
        if self.power_accumulated is None:
            self.power_accumulated = flux.copy()
        else:
            self.power_accumulated += flux
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
        """Display the latest recorded field with matplotlib.

        This compatibility hook no longer installs a persistent live updater. Use
        ``Simulation.animate(...)`` for streamed simulation updates.
        """
        return self.show(field=field_component)

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
                power_spectrum_frequencies=self.power_spectrum_frequencies,
                power_spectrum=self.power_spectrum,
                frequency_points=self.power_spectrum_frequencies,
                frequency_flux_spectrum=(
                    self._frequency_flux_spectrum_legacy
                    if self._frequency_flux_spectrum_legacy is not None
                    else self.power_spectrum
                ),
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
        if "power_spectrum_frequencies" in data:
            self.power_spectrum_frequencies = np.asarray(
                data["power_spectrum_frequencies"], dtype=np.float64
            )
        elif "frequency_points" in data:
            self.power_spectrum_frequencies = np.asarray(
                data["frequency_points"], dtype=np.float64
            )
        self.accumulate_frequency = bool(self.power_spectrum_frequencies.size > 0)
        if "power_spectrum" in data:
            self.power_spectrum = np.asarray(data["power_spectrum"], dtype=np.complex64)
        elif "frequency_flux_spectrum" in data:
            self.power_spectrum = np.asarray(
                data["frequency_flux_spectrum"], dtype=np.complex64
            )
        if "frequency_flux_spectrum" in data:
            self._frequency_flux_spectrum_legacy = np.asarray(
                data["frequency_flux_spectrum"], dtype=np.complex64
            )

    def to_plot_data(
        self, *, facecolor="none", edgecolor="navy", alpha=1.0, linestyle="-"
    ):
        """Return a renderer-agnostic monitor payload."""
        from beamz.visual.data import monitor_plot_data

        return monitor_plot_data(
            self,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linestyle=linestyle,
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

    def field_plot_data(self, **kwargs):
        """Return monitor-field data for manual plotting."""
        from beamz.visual.data import monitor_field_plot_data

        return monitor_field_plot_data(self, **kwargs)

    def power_plot_data(self, **kwargs):
        """Return monitor-power data for manual plotting."""
        from beamz.visual.data import monitor_power_plot_data

        return monitor_power_plot_data(self, **kwargs)

    def plot(self, **kwargs):
        """Plot recorded monitor field data using the matplotlib backend."""
        from beamz.visual.mpl import plot_monitor_field

        kwargs.setdefault("show", False)
        return plot_monitor_field(self, **kwargs)

    def show(self, **kwargs):
        """Display recorded monitor field data using the matplotlib backend."""
        kwargs.setdefault("show", True)
        return self.plot(**kwargs)

    def plot_fields(self, **kwargs):
        """Plot recorded monitor field data."""
        return self.plot(**kwargs)

    def plot_power(self, **kwargs):
        """Plot monitor power history."""
        from beamz.visual.mpl import plot_monitor_power

        kwargs.setdefault("show", False)
        return plot_monitor_power(self, **kwargs)

    def show_power(self, **kwargs):
        """Display monitor power history using the matplotlib backend."""
        kwargs.setdefault("show", True)
        return self.plot_power(**kwargs)

    def to_xarray(self):
        """Return recorded monitor data as an xarray Dataset."""
        from beamz.data.xarray import monitor_dataset

        return monitor_dataset(self)

    def animate_fields(self, **kwargs):
        """Animate recorded monitor field data using matplotlib."""
        from beamz.visual.mpl import animate_monitor_fields

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

    def get_signed_flux_trace(self, normal_direction, field_pair=None):
        """Return signed directional flux trace from recorded field components.

        For 2D monitors:
          - default +x/-x uses (Ez, Hy) with Sx = -Re(Ez * conj(Hy))
          - default +y/-y uses (Ez, Hx) with Sy = +Re(Ez * conj(Hx))
        """
        direction = str(normal_direction).lower()
        if direction not in {"+x", "-x", "+y", "-y"}:
            raise ValueError(
                f"normal_direction must be one of ['+x','-x','+y','-y'], got {normal_direction!r}"
            )
        axis = direction[1]
        dir_sign = 1.0 if direction.startswith("+") else -1.0

        if field_pair is None:
            if axis == "x":
                e_comp, h_comp, base_sign = "Ez", "Hy", -1.0
            else:
                e_comp, h_comp, base_sign = "Ez", "Hx", 1.0
        else:
            if len(field_pair) != 2:
                raise ValueError("field_pair must be a tuple like ('Ez', 'Hy').")
            e_comp, h_comp = field_pair
            base_sign = 1.0

        if e_comp not in self.fields or h_comp not in self.fields:
            raise ValueError(
                f"Requested components ({e_comp}, {h_comp}) are not recorded by this monitor."
            )
        if not self.fields[e_comp] or not self.fields[h_comp]:
            raise ValueError(
                f"No recorded data for ({e_comp}, {h_comp}) on monitor '{self.name}'."
            )

        e_arr = np.asarray(self.fields[e_comp], dtype=np.complex128)
        h_arr = np.asarray(self.fields[h_comp], dtype=np.complex128)
        if e_arr.ndim == 1:
            e_arr = e_arr[:, None]
        if h_arr.ndim == 1:
            h_arr = h_arr[:, None]

        n_t = min(e_arr.shape[0], h_arr.shape[0])
        n_p = min(e_arr.shape[1], h_arr.shape[1])
        signed_density = base_sign * np.real(
            e_arr[:n_t, :n_p] * np.conjugate(h_arr[:n_t, :n_p])
        )
        return dir_sign * np.sum(signed_density, axis=1)

    def __str__(self):
        if not self.fields["t"]:
            return f"Monitor: {self.monitor_type} ({'3D' if self.is_3d else '2D'}), 0 records"
        stats = self.get_field_statistics()
        return f"Monitor: {stats['monitor_type']} ({'3D' if stats['is_3d'] else '2D'}), {stats['total_records']} records"

    def copy(self, *, update=None):
        """Create a configuration copy of the Monitor."""
        import copy

        copied = copy.deepcopy(self)
        if update:
            for key, value in dict(update).items():
                setattr(copied, key, value)
        return copied

    def shifted(self, offset):
        copied = self.copy()
        offset = tuple(float(v) for v in offset)
        if getattr(copied, "start", None) is not None:
            copied.start = tuple(
                a + b for a, b in zip(copied.start, offset, strict=False)
            )
        if getattr(copied, "end", None) is not None:
            copied.end = tuple(a + b for a, b in zip(copied.end, offset, strict=False))
        if getattr(copied, "position", None) is not None:
            copied.position = tuple(
                a + b for a, b in zip(copied.position, offset, strict=False)
            )
        if getattr(copied, "center", None) is not None:
            copied.center = tuple(
                a + b for a, b in zip(copied.center, offset, strict=False)
            )
        if getattr(copied, "vertices", None):
            copied.vertices = [
                tuple(a + b for a, b in zip(vertex, offset, strict=False))
                for vertex in copied.vertices
            ]
        if getattr(copied, "plane_normal", None) in {"x", "y", "z"}:
            idx = {"x": 0, "y": 1, "z": 2}[copied.plane_normal]
            copied.plane_position = float(copied.plane_position) + offset[idx]
        return copied


class FieldMonitor(Monitor):
    """Frequency-domain field monitor specified by center, size, and frequencies."""

    def __init__(
        self,
        *,
        center,
        size,
        freqs,
        name=None,
        components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        **kwargs,
    ):
        start, end = _plane_start_end_from_center_size(center, size)
        self.center = tuple(float(v) for v in center)
        self.size_spec = tuple(float(v) for v in size)
        super().__init__(
            start=start,
            end=end,
            name=name,
            record_fields=False,
            accumulate_power=False,
            dft_enabled=True,
            dft_frequencies=np.asarray(freqs, dtype=float),
            dft_components=tuple(components),
            **kwargs,
        )


class FluxMonitor(Monitor):
    """Frequency-domain flux monitor specified by center, size, and frequencies."""

    def __init__(self, *, center, size, freqs, name=None, **kwargs):
        start, end = _plane_start_end_from_center_size(center, size)
        self.center = tuple(float(v) for v in center)
        self.size_spec = tuple(float(v) for v in size)
        super().__init__(
            start=start,
            end=end,
            name=name,
            record_fields=False,
            accumulate_power=False,
            dft_enabled=True,
            dft_frequencies=np.asarray(freqs, dtype=float),
            dft_components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
            **kwargs,
        )


class ModeMonitor(Monitor):
    """A monitor carrying modal-port metadata.

    `direction` is the direction of a wave entering the simulated device at this
    port. Pass `ModeMonitor` objects directly to modal S-parameter extraction to
    avoid manually selecting `plus`/`minus` incident and scattered waves.
    """

    def __init__(
        self,
        *args,
        center=None,
        size=None,
        freqs=None,
        mode_spec=None,
        direction,
        polarization,
        mode_index=0,
        reference_monitor=None,
        projection_direction=None,
        dft_components=None,
        dft_enabled=None,
        **kwargs,
    ):
        if center is not None or size is not None:
            if center is None or size is None:
                raise ValueError("ModeMonitor requires both center and size.")
            start, end = _plane_start_end_from_center_size(center, size)
            kwargs.setdefault("start", start)
            kwargs.setdefault("end", end)
            self.center = tuple(float(v) for v in center)
            self.size_spec = tuple(float(v) for v in size)
        if freqs is not None:
            kwargs.setdefault("dft_frequencies", np.asarray(freqs, dtype=float))
        self.mode_spec = mode_spec
        if mode_spec is not None and mode_index == 0:
            mode_index = int(getattr(mode_spec, "mode_index", 0))
        self.direction = _normalize_direction(direction)
        self.polarization = _normalize_polarization(polarization)
        self.mode_index = int(mode_index)
        self.reference_monitor = reference_monitor
        self.projection_direction = (
            None
            if projection_direction is None
            else _normalize_direction(projection_direction)
        )
        if dft_enabled is None:
            dft_enabled = kwargs.get("dft_frequencies") is not None
        if dft_components is None:
            # Keep the first-class monitor robust across 2D/3D and TE/TM. Modal
            # extraction will only consume the tangential components it needs.
            dft_components = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        super().__init__(
            *args,
            dft_enabled=bool(dft_enabled),
            dft_components=dft_components,
            **kwargs,
        )

    def to_port(self, *, name=None) -> Port:
        port_name = self.name if name is None else name
        if not port_name:
            raise ValueError("ModeMonitor must have a name to be used as a Port.")
        return Port(
            name=str(port_name),
            monitor=self,
            direction=self.direction,
            polarization=self.polarization,
            mode_index=self.mode_index,
            reference_monitor=self.reference_monitor,
            projection_direction=self.projection_direction,
        )

    def to_portspec_dict(self) -> dict:
        return self.to_port().to_portspec_dict()
