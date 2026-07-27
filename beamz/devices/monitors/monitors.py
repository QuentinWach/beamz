from dataclasses import dataclass, field, replace

import numpy as np

from beamz._helpers import positive_integer
from beamz.devices._immutable import (
    finite_tuple,
    immutable_snapshot,
    nonnegative_finite_extents,
    readonly_1d_array,
)
from beamz.devices._placement import (
    line_region_points,
    plane_region_slices,
    snap_axis_aligned_line_region,
    snap_plane_region,
)
from beamz.devices.modes.specs import ModeSpec
from beamz.lattice import yee_plane_coordinates_3d


def _normalize_center_size(center, size):
    center = finite_tuple(center, name="Monitor center")
    size = nonnegative_finite_extents(size, name="Monitor size")
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


@dataclass(frozen=True, kw_only=True, eq=False)
class _Monitor:
    """Shared immutable geometry and frequency configuration for monitors."""

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    freqs: np.ndarray
    name: str | None = None
    interval: int = 1

    def __post_init__(self) -> None:
        center, size = _normalize_center_size(self.center, self.size)
        frequencies = readonly_1d_array(self.freqs, dtype=float)
        if np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
            raise ValueError("Monitor frequencies must be finite and positive.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "freqs", frequencies)
        object.__setattr__(self, "name", None if self.name is None else str(self.name))
        object.__setattr__(
            self, "interval", positive_integer(self.interval, name="Monitor interval")
        )

    @property
    def size_spec(self):
        return self.size

    @property
    def start(self):
        return _plane_start_end_from_center_size(self.center, self.size)[0]

    @property
    def end(self):
        return _plane_start_end_from_center_size(self.center, self.size)[1]

    @property
    def plane_normal(self):
        return ("x", "y", "z")[int(np.argmin(np.abs(np.asarray(self.size))))]

    @property
    def plane_position(self):
        return self.center[{"x": 0, "y": 1, "z": 2}[self.plane_normal]]

    @property
    def position(self):
        return self.center

    @property
    def is_3d(self):
        return int(np.count_nonzero(np.asarray(self.size) > 0.0)) >= 2

    def updated_copy(self, **changes):
        return replace(self, **changes)

    def _comparison_key(self):
        return (
            type(self),
            self.center,
            self.size,
            tuple(float(freq) for freq in self.freqs),
            self.name,
            self.interval,
            getattr(self, "fields", None),
            getattr(self, "components", None),
            getattr(self, "region", None),
            getattr(self, "mode_spec", None),
        )

    def __eq__(self, other):
        return (
            isinstance(other, _Monitor)
            and self._comparison_key() == other._comparison_key()
        )

    def __hash__(self):
        return hash(self._comparison_key())

    def get_grid_points_2d(self, dx, dy):
        """Return integer grid points for a 2D line monitor.

        Parameters
        ----------
        dx, dy : float
            Cell spacing along x and y in metres.
        """
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

        return list(zip(x_indices, y_indices, strict=True))

    def _line_sample_coords_2d(self, dx, dy, field_shape=None):
        snapped = self.get_snapped_region(dx=dx, dy=dy, field_shape=field_shape)
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

    def get_grid_slice_3d(self, dx, dy, dz, field_shape):
        """Return array indices for a 3D plane monitor.

        Parameters
        ----------
        dx, dy, dz : float
            Cell spacing along each Cartesian axis in metres.
        field_shape : tuple of int
            Base field shape in ``(z, y, x)`` order.

        Returns
        -------
        tuple
            ``(z_index, y_index, x_index)`` in array order. The plane-normal
            entry is an integer and both tangential entries are slices.
        """
        snapped = self.get_snapped_region(dx=dx, dy=dy, dz=dz, field_shape=field_shape)
        if snapped is not None:
            return plane_region_slices(snapped)

        if self.plane_normal == "z":
            # xy plane at fixed z
            z_idx = int(round(self.plane_position / dz))
            x_start = int(round(self.start[0] / dx))
            x_end = int(round(self.end[0] / dx))
            y_start = int(round(self.start[1] / dy))
            y_end = int(round(self.end[1] / dy))
            return z_idx, slice(y_start, y_end), slice(x_start, x_end)

        elif self.plane_normal == "y":
            # xz plane at fixed y
            y_idx = int(round(self.plane_position / dy))
            x_start = int(round(self.start[0] / dx))
            x_end = int(round(self.end[0] / dx))
            z_start = int(round(self.start[2] / dz))
            z_end = int(round(self.end[2] / dz))
            return slice(z_start, z_end), y_idx, slice(x_start, x_end)
        else:  # x normal
            # yz plane at fixed x
            x_idx = int(round(self.plane_position / dx))
            y_start = int(round(self.start[1] / dy))
            y_end = int(round(self.end[1] / dy))
            z_start = int(round(self.start[2] / dz))
            z_end = int(round(self.end[2] / dz))
            return slice(z_start, z_end), slice(y_start, y_end), x_idx

    def get_analysis_plane_coords_3d(
        self,
        *,
        dx: float,
        dy: float,
        dz: float,
        field_shape: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return canonical tangential analysis coordinates in metres."""
        snapped = self.get_snapped_region(
            dx=dx,
            dy=dy,
            dz=dz,
            field_shape=field_shape,
        )
        if snapped is None:
            raise ValueError("3D analysis plane requires a snapped monitor region.")
        return yee_plane_coordinates_3d(
            self.center,
            self.size,
            self.plane_normal,
            snapped,
        )

    def get_snapped_region(self, *, dx, dy, dz=None, field_shape=None):
        """Return the canonical snapped region for monitor sampling.

        Parameters
        ----------
        dx, dy : float
            Cell spacing along x and y in metres.
        dz : float, optional
            Cell spacing along z in metres; required for 3D monitors.
        field_shape : tuple of int, optional
            Base field shape in ``(z, y, x)`` order; required for 3D monitors.
        """
        if field_shape is not None and len(tuple(field_shape)) == 3:
            if dz is None or field_shape is None:
                return None
            return snap_plane_region(
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

        if field_shape is not None:
            shape = tuple(int(v) for v in field_shape)
        else:
            return None
        return snap_axis_aligned_line_region(
            tuple(float(v) for v in self.start),
            tuple(float(v) for v in self.end),
            float(dx),
            float(dy),
            shape,
        )

    def get_dft_frequencies(self):
        """Return DFT frequencies in hertz."""
        return np.asarray(self.freqs, dtype=float)

    def shifted(self, offset):
        """Return a translated monitor."""
        delta = tuple(float(value) for value in offset)
        if len(delta) != 3:
            raise ValueError("Monitor offsets must contain three values.")
        return replace(
            self,
            center=tuple(
                value + shift for value, shift in zip(self.center, delta, strict=True)
            ),
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class FieldMonitor(_Monitor):
    """Record selected frequency-domain field components on a plane or line."""

    fields: tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

    def __post_init__(self) -> None:
        if not np.any(
            np.isclose(
                tuple(float(value) for value in self.size),
                0.0,
                rtol=0.0,
                atol=1e-15,
            )
        ):
            raise ValueError("FieldMonitor size must contain at least one zero extent.")
        super().__post_init__()
        if self.freqs.size == 0:
            raise ValueError("FieldMonitor requires at least one frequency.")
        fields = tuple(str(component) for component in self.fields)
        invalid = set(fields) - {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
        if not fields or invalid:
            raise ValueError(f"Unsupported FieldMonitor fields: {sorted(invalid)}")
        object.__setattr__(self, "fields", fields)

    @property
    def dft_components(self):
        return self.fields


@dataclass(frozen=True, init=False, eq=False)
class FieldRecorder(_Monitor):
    """Record time-domain field snapshots on the domain or a center/size plane."""

    freqs: np.ndarray = field(init=False, repr=False)
    components: tuple[str, ...]
    region: str = field(init=False)

    def __init__(
        self,
        components=("Ez",),
        interval=10,
        *,
        name="fields",
        center=None,
        size=None,
    ):
        components = tuple(str(component) for component in components)
        invalid = set(components) - {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
        if not components or invalid:
            raise ValueError(f"Unsupported FieldRecorder components: {sorted(invalid)}")
        if (center is None) != (size is None):
            raise ValueError("FieldRecorder requires both center and size.")
        region = "domain" if center is None else "slice"
        _Monitor.__init__(
            self,
            center=(0.0, 0.0, 0.0) if center is None else center,
            size=(0.0, 0.0, 0.0) if size is None else size,
            freqs=np.zeros((0,), dtype=float),
            name=name,
            interval=interval,
        )
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "region", region)

    @property
    def dft_components(self):
        return None

    def updated_copy(self, **changes):
        values = {
            "components": self.components,
            "interval": self.interval,
            "name": self.name,
            "center": None if self.region == "domain" else self.center,
            "size": None if self.region == "domain" else self.size,
        }
        values.update(changes)
        return type(self)(**values)

    def shifted(self, offset):
        if self.region == "domain":
            return self
        delta = tuple(float(value) for value in offset)
        return self.updated_copy(
            center=tuple(
                value + shift for value, shift in zip(self.center, delta, strict=True)
            )
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class FluxMonitor(_Monitor):
    """Record frequency-domain power flux through a center/size plane."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.freqs.size == 0:
            raise ValueError("FluxMonitor requires at least one frequency.")
        if np.count_nonzero(np.isclose(self.size, 0.0, rtol=0.0, atol=1e-15)) != 1:
            raise ValueError("FluxMonitor size must contain exactly one zero extent.")

    @property
    def dft_components(self):
        return {
            "x": ("Ey", "Ez", "Hy", "Hz"),
            "y": ("Ex", "Ez", "Hx", "Hz"),
            "z": ("Ex", "Ey", "Hx", "Hy"),
        }[self.plane_normal]


@dataclass(frozen=True, kw_only=True, eq=False)
class ModeMonitor(_Monitor):
    """Record modal amplitudes on a center/size plane."""

    mode_spec: ModeSpec = field(default_factory=ModeSpec)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.freqs.size == 0:
            raise ValueError("ModeMonitor requires at least one frequency.")
        if np.count_nonzero(np.isclose(self.size, 0.0, rtol=0.0, atol=1e-15)) != 1:
            raise ValueError("ModeMonitor size must contain exactly one zero extent.")
        mode_spec = immutable_snapshot(self.mode_spec)
        if not isinstance(mode_spec, ModeSpec):
            raise TypeError("ModeMonitor mode_spec must be a ModeSpec.")
        object.__setattr__(self, "mode_spec", mode_spec)

    @property
    def dft_components(self):
        return ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
