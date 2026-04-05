import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0
from beamz.devices.sources.spec import GaussianSourceSpec, build_gaussian_source_spec
from beamz.devices.sources.state import GaussianSourceState


_GAUSSIAN_SPEC_FIELDS = frozenset(GaussianSourceSpec.__dataclass_fields__.keys())
_GAUSSIAN_STATE_MAP = {
    "_spatial_profile_ez": "spatial_profile_ez",
    "_grid_indices": "grid_indices",
}


class GaussianSource:
    """Gaussian spatial source for FDTD simulations.

    Injects a Gaussian spatial profile into the Ez field (and other E components in 3D).
    Useful for dipole-like excitations.

    Supports JAX differentiability through position and width parameters.
    """

    def __init__(self, position, width, signal):
        """Initialize the Gaussian source.

        Args:
            position: (x, y) for 2D or (x, y, z) for 3D - center of Gaussian
            width: Standard deviation of Gaussian profile
            signal: Time-dependent signal function s(t) or array
        """
        object.__setattr__(
            self,
            "spec",
            build_gaussian_source_spec(position=position, width=width, signal=signal),
        )
        object.__setattr__(self, "state", GaussianSourceState())

    def __getattr__(self, name):
        spec = self.__dict__.get("spec")
        if spec is not None and hasattr(spec, name):
            return getattr(spec, name)
        state = self.__dict__.get("state")
        if state is not None and name in _GAUSSIAN_STATE_MAP:
            return getattr(state, _GAUSSIAN_STATE_MAP[name])
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in {"spec", "state"}:
            object.__setattr__(self, name, value)
            return
        if name in _GAUSSIAN_STATE_MAP and "state" in self.__dict__:
            setattr(self.state, _GAUSSIAN_STATE_MAP[name], value)
            return
        object.__setattr__(self, name, value)

    def _get_signal_value(self, time, dt):
        """Interpolate signal value at arbitrary time (JAX-compatible)."""
        # Handle JAX/numpy array signal
        if isinstance(self.signal, (jnp.ndarray, np.ndarray)):
            signal_arr = jnp.asarray(self.signal)
            idx_float = time / dt
            idx_low = jnp.floor(idx_float).astype(jnp.int32)
            idx_high = idx_low + 1
            frac = idx_float - jnp.floor(idx_float)

            signal_len = signal_arr.shape[0]
            # Clamp indices to valid range
            idx_low_safe = jnp.clip(idx_low, 0, signal_len - 1)
            idx_high_safe = jnp.clip(idx_high, 0, signal_len - 1)

            # Interpolate
            interp_val = (1.0 - frac) * signal_arr[idx_low_safe] + frac * signal_arr[
                idx_high_safe
            ]

            # Return 0 if out of range (except at last valid index)
            in_range = (idx_low >= 0) & (idx_low < signal_len - 1)
            at_end = idx_low == signal_len - 1
            return jnp.where(
                in_range, interp_val, jnp.where(at_end, signal_arr[idx_low_safe], 0.0)
            )
        # Handle list signal (convert to JAX)
        elif isinstance(self.signal, list):
            self.signal = jnp.asarray(self.signal)
            return self._get_signal_value(time, dt)
        # Handle callable signal
        elif callable(self.signal):
            return self.signal(time)
        else:
            return 0.0

    def inject(self, fields, t, dt, current_step, resolution, design):
        """Inject source fields directly into the simulation grid before the FDTD update step."""
        is_3d = len(self.position) >= 3 if hasattr(self.position, "__len__") else False

        if self._spatial_profile_ez is None:
            self._init_spatial_profile(fields.Ez.shape, resolution, is_3d)

        signal_val = self._get_signal_value(t + 0.5 * dt, dt)
        eps_region = fields.permittivity[self._grid_indices]

        term = self._spatial_profile_ez * signal_val
        injection = -term * dt / (EPS_0 * eps_region)
        fields.Ez = fields.Ez.at[self._grid_indices].add(injection)

    def _init_spatial_profile(self, ez_shape, resolution, is_3d):
        """Compute the spatial Gaussian profile and grid indices (called once)."""
        sigma_grid = self.width / resolution
        radius_grid = int(np.ceil(4 * sigma_grid))

        if is_3d:
            x0, y0, z0 = self.position
            nz, ny, nx = ez_shape
            cx, cy, cz = (int(round(c / resolution)) for c in (x0, y0, z0))

            x_start, x_end = max(0, cx - radius_grid), min(nx, cx + radius_grid + 1)
            y_start, y_end = max(0, cy - radius_grid), min(ny, cy + radius_grid + 1)
            z_start, z_end = max(0, cz - radius_grid), min(nz, cz + radius_grid + 1)

            self._grid_indices = (
                slice(z_start, z_end),
                slice(y_start, y_end),
                slice(x_start, x_end),
            )

            x_coords = (jnp.arange(x_start, x_end) + 0.5) * resolution
            y_coords = (jnp.arange(y_start, y_end) + 0.5) * resolution
            z_coords = (jnp.arange(z_start, z_end) + 0.5) * resolution
            Z, Y, X = jnp.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
            dist_sq = (X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2
        else:
            x0, y0 = self.position
            ny, nx = ez_shape
            cx, cy = int(round(x0 / resolution)), int(round(y0 / resolution))

            x_start, x_end = max(0, cx - radius_grid), min(nx, cx + radius_grid + 1)
            y_start, y_end = max(0, cy - radius_grid), min(ny, cy + radius_grid + 1)

            self._grid_indices = (slice(y_start, y_end), slice(x_start, x_end))

            x_coords = (jnp.arange(x_start, x_end) + 0.5) * resolution
            y_coords = (jnp.arange(y_start, y_end) + 0.5) * resolution
            X, Y = jnp.meshgrid(x_coords, y_coords, indexing="xy")
            dist_sq = (X - x0) ** 2 + (Y - y0) ** 2

        self._spatial_profile_ez = jnp.exp(-dist_sq / (2 * self.width**2))

    def add_to_plot(
        self, ax, facecolor="none", edgecolor="orange", alpha=0.8, linestyle="-"
    ):
        """Add source visualization to 2D matplotlib plot."""
        from beamz.visual.overlays import add_gaussian_source_to_plot

        add_gaussian_source_to_plot(
            self,
            ax,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linestyle=linestyle,
        )
