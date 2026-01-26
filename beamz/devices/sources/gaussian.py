import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0


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
        self.position = position
        self.width = width
        # Convert signal to JAX array if it's a numpy array
        if isinstance(signal, np.ndarray):
            self.signal = jnp.asarray(signal)
        else:
            self.signal = signal
        self._spatial_profile_ez = None
        self._grid_indices = None

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
        dx = dy = resolution

        # Check dimensionality from position length (more reliable than fields when meshing is 2D for 3D design)
        position_len = len(self.position) if hasattr(self.position, "__len__") else 1
        if position_len == 3:
            self._inject_3d(fields, t, dt, resolution)
        else:
            self._inject_2d(fields, t, dt, resolution)

    def _inject_2d(self, fields, t, dt, resolution):
        """Inject into 2D grid (Ez component)."""
        ny, nx = fields.Ez.shape

        # Initialize spatial profile if needed (do this once)
        if self._spatial_profile_ez is None:
            x0, y0 = self.position

            # Determine bounding box for Gaussian (e.g., +/- 4 sigma) to save computation
            # Convert position to grid indices (these are static, computed once)
            sigma_grid = self.width / resolution
            radius_grid = int(np.ceil(4 * sigma_grid))

            cx = int(round(x0 / resolution))
            cy = int(round(y0 / resolution))

            # Define ROI limits
            x_start = max(0, cx - radius_grid)
            x_end = min(nx, cx + radius_grid + 1)
            y_start = max(0, cy - radius_grid)
            y_end = min(ny, cy + radius_grid + 1)

            self._grid_indices = (slice(y_start, y_end), slice(x_start, x_end))

            # Generate coordinate grids for the ROI using JAX
            x_coords = (jnp.arange(x_start, x_end) + 0.5) * resolution
            y_coords = (jnp.arange(y_start, y_end) + 0.5) * resolution

            X, Y = jnp.meshgrid(x_coords, y_coords, indexing="xy")

            # Compute Gaussian using JAX (differentiable w.r.t. position and width)
            dist_sq = (X - x0) ** 2 + (Y - y0) ** 2
            profile = jnp.exp(-dist_sq / (2 * self.width**2))
            self._spatial_profile_ez = profile

        # Get signal value
        # Inject at t + 0.5 dt because Ez is updated at n+1 from n
        # Soft source: J_z is added.
        # Ez_new = Ez_old + ... - dt/eps * J_z
        # We want to add to Ez.
        # Typically soft source adds to E directly or J term.
        # ModeSource adds: fields.Ez += injection
        # injection = -jz * dt / (eps * eps0)
        # Here we treat GaussianSource as a J_z source.

        signal_val = self._get_signal_value(t + 0.5 * dt, dt)

        # Get permittivity in the region
        eps_region = fields.permittivity[self._grid_indices]

        # Calculate injection term
        # J(x, t) = Profile(x) * s(t)
        # Update: E += -J * dt / (eps * eps0)
        # We can absorb the negative sign into the signal definition if we want E to follow signal
        # But strictly J is current.
        # Let's just add it as a "forcing function" to E.
        # If we want E ~ Signal, we might just add Profile * Signal * Scaling
        # Let's follow ModeSource physics: inject current J.

        term = self._spatial_profile_ez * signal_val
        injection = -term * dt / (EPS_0 * eps_region)

        # Inject using JAX functional update
        fields.Ez = fields.Ez.at[self._grid_indices].add(injection)

    def _inject_3d(self, fields, t, dt, resolution):
        """Inject into 3D grid (Ez component, could be expanded)."""
        # Similar to 2D but with Z coordinate
        # Only implementing Ez injection for dipole-like behavior along Z
        nz, ny, nx = fields.Ez.shape

        if self._spatial_profile_ez is None:
            x0, y0, z0 = self.position

            sigma_grid = self.width / resolution
            radius_grid = int(np.ceil(4 * sigma_grid))

            cx = int(round(x0 / resolution))
            cy = int(round(y0 / resolution))
            cz = int(round(z0 / resolution))

            x_start, x_end = max(0, cx - radius_grid), min(nx, cx + radius_grid + 1)
            y_start, y_end = max(0, cy - radius_grid), min(ny, cy + radius_grid + 1)
            z_start, z_end = max(0, cz - radius_grid), min(nz, cz + radius_grid + 1)

            self._grid_indices = (
                slice(z_start, z_end),
                slice(y_start, y_end),
                slice(x_start, x_end),
            )

            # Generate coordinate grids using JAX
            x_coords = (jnp.arange(x_start, x_end) + 0.5) * resolution
            y_coords = (jnp.arange(y_start, y_end) + 0.5) * resolution
            z_coords = (jnp.arange(z_start, z_end) + 0.5) * resolution

            Z, Y, X = jnp.meshgrid(z_coords, y_coords, x_coords, indexing="ij")

            # Compute Gaussian using JAX (differentiable w.r.t. position and width)
            dist_sq = (X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2
            self._spatial_profile_ez = jnp.exp(-dist_sq / (2 * self.width**2))

        signal_val = self._get_signal_value(t + 0.5 * dt, dt)
        eps_region = fields.permittivity[self._grid_indices]

        term = self._spatial_profile_ez * signal_val
        injection = -term * dt / (EPS_0 * eps_region)

        # Inject using JAX functional update
        fields.Ez = fields.Ez.at[self._grid_indices].add(injection)

    def add_to_plot(
        self, ax, facecolor="none", edgecolor="orange", alpha=0.8, linestyle="-"
    ):
        """Add source visualization to 2D matplotlib plot.

        Draws a circle at the source position with radius proportional to the width.
        """
        from matplotlib.patches import Circle

        # Get position (2D or 3D)
        if len(self.position) >= 2:
            x_pos, y_pos = self.position[0], self.position[1]
        else:
            x_pos, y_pos = self.position[0], 0

        # Draw circle with radius = width (standard deviation)
        circle = Circle(
            (x_pos, y_pos),
            radius=self.width,
            facecolor="none",
            edgecolor=edgecolor,
            linewidth=2,
            alpha=alpha,
            linestyle=linestyle,
            label="GaussianSource",
        )
        ax.add_patch(circle)

        # Draw a small filled circle at center
        center_dot = Circle(
            (x_pos, y_pos),
            radius=self.width * 0.1,
            facecolor=edgecolor,
            edgecolor="none",
            alpha=alpha,
        )
        ax.add_patch(center_dot)
