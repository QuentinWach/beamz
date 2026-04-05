from dataclasses import replace

import jax.numpy as jnp

from beamz.const import EPS_0
from beamz.devices.sources import setup as setup_helpers
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
            signal: Time-dependent 1D sampled signal array
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

    def with_spec(self, spec=None, /, **changes):
        base_spec = self.spec if spec is None else spec
        if not isinstance(base_spec, GaussianSourceSpec):
            raise TypeError("with_spec expects a GaussianSourceSpec or spec field updates")
        if changes:
            base_spec = replace(base_spec, **changes)
        new = object.__new__(type(self))
        object.__setattr__(new, "spec", base_spec)
        object.__setattr__(new, "state", GaussianSourceState())
        return new

    def to_dict(self):
        return self.spec.to_dict()

    @classmethod
    def from_dict(cls, data):
        return cls.from_spec(GaussianSourceSpec.from_dict(data))

    @classmethod
    def from_spec(cls, spec):
        if not isinstance(spec, GaussianSourceSpec):
            raise TypeError("from_spec expects a GaussianSourceSpec")
        new = object.__new__(cls)
        object.__setattr__(new, "spec", spec)
        object.__setattr__(new, "state", GaussianSourceState())
        return new

    def _get_signal_value(self, time, dt):
        """Interpolate signal value at arbitrary time (JAX-compatible)."""
        signal_arr = jnp.asarray(self.signal)
        idx_float = time / dt
        idx_low = jnp.floor(idx_float).astype(jnp.int32)
        idx_high = idx_low + 1
        frac = idx_float - jnp.floor(idx_float)

        signal_len = signal_arr.shape[0]
        idx_low_safe = jnp.clip(idx_low, 0, signal_len - 1)
        idx_high_safe = jnp.clip(idx_high, 0, signal_len - 1)
        interp_val = (1.0 - frac) * signal_arr[idx_low_safe] + frac * signal_arr[
            idx_high_safe
        ]
        in_range = (idx_low >= 0) & (idx_low < signal_len - 1)
        at_end = idx_low == signal_len - 1
        return jnp.where(
            in_range, interp_val, jnp.where(at_end, signal_arr[idx_low_safe], 0.0)
        )

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
        del is_3d
        setup_helpers.initialize_gaussian_state(self.spec, self.state, ez_shape, resolution)

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
