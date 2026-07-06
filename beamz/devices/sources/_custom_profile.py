"""Experimental prepared-field-profile source internals."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from beamz.devices.sources._planar_tfsf import (
    _ModeSource3DResidual,
    compute_discrete_3d_e_phasor_residuals,
    compute_discrete_3d_h_phasor_residuals,
)
from beamz.devices.sources._profiles import FieldProfile3D
from beamz.devices.sources._time import (
    _analytic_signal_quadrature,
    _interpolate_time_signal,
    _sample_waveform,
)


class _CustomFieldProfileSource:
    """Experimental planar TF/SF source backed by a prepared 3D field profile."""

    def __init__(
        self,
        *,
        profile: FieldProfile3D,
        signal,
        power: float = 1.0,
        signal_quadrature=None,
        max_shift: int = 1,
    ):
        if not isinstance(profile, FieldProfile3D):
            raise TypeError("profile must be a FieldProfile3D instance")
        power_value = float(power)
        if not np.isfinite(power_value) or power_value < 0.0:
            raise ValueError(
                "Custom field profile source power must be a non-negative finite "
                f"value, got {power!r}."
            )
        self.profile = self._scaled_profile(profile, power_value)
        self.signal = jnp.asarray(signal) if isinstance(signal, np.ndarray) else signal
        self.signal_quadrature = signal_quadrature
        self.power = power_value
        self.max_shift = int(max(1, max_shift))
        self._signal_quadrature = None
        self._signal_quadrature_signature = None

    @staticmethod
    def _scaled_profile(profile: FieldProfile3D, power: float) -> FieldProfile3D:
        if power == 1.0:
            return profile
        scale = float(np.sqrt(power))
        return replace(
            profile,
            components={
                name: np.asarray(value, dtype=np.complex128) * scale
                for name, value in profile.components.items()
            },
        )

    def _get_signal_value(self, time, dt):
        if isinstance(self.signal, (jnp.ndarray, np.ndarray)):
            return _interpolate_time_signal(np.asarray(self.signal), time, dt)
        if isinstance(self.signal, list):
            self.signal = jnp.asarray(self.signal)
            return self._get_signal_value(time, dt)
        if callable(self.signal):
            return float(self.signal(float(time)))
        return 0.0

    def _get_signal_quadrature(self):
        explicit = self.signal_quadrature
        if explicit is not None:
            return np.asarray(explicit, dtype=np.float64)
        if not isinstance(self.signal, (jnp.ndarray, np.ndarray)):
            return np.zeros((0,), dtype=np.float64)
        signal = np.asarray(self.signal, dtype=np.float64)
        signature = (signal.shape, signal.dtype.str, signal.tobytes())
        if (
            self._signal_quadrature is None
            or self._signal_quadrature_signature != signature
        ):
            self._signal_quadrature = _analytic_signal_quadrature(signal)
            self._signal_quadrature_signature = signature
        return self._signal_quadrature

    def _get_signal_quadrature_value(self, time, dt):
        explicit = self.signal_quadrature
        if callable(explicit):
            return float(explicit(float(time)))
        return _interpolate_time_signal(self._get_signal_quadrature(), time, dt)

    def _compute_discrete_3d_phasor_residuals(
        self,
        fields,
        *,
        dt: float,
        resolution: float,
    ) -> tuple[_ModeSource3DResidual, ...]:
        """Return compact phasor residuals for the prepared field profile."""
        return (
            *compute_discrete_3d_h_phasor_residuals(
                self.profile,
                fields,
                resolution=float(resolution),
                max_shift=self.max_shift,
                dt=float(dt),
            ),
            *compute_discrete_3d_e_phasor_residuals(
                self.profile,
                fields,
                resolution=float(resolution),
                max_shift=self.max_shift,
                dt=float(dt),
            ),
        )

    def compile_source_specs(
        self,
        *,
        fields,
        dt: float,
        num_steps: int,
        t0: float,
        resolution: float,
        total_steps: int | None = None,
    ):
        """Emit compiled phasor source specs through the planar TF/SF engine."""
        from beamz.devices.sources.compiler import _append_phasor_source_specs

        waveform = _sample_waveform(
            self._get_signal_value,
            t0=t0,
            dt=dt,
            num_steps=num_steps,
            offset_fn=lambda t, dt_: t,
            total_steps=total_steps,
        )
        quadrature_waveform = _sample_waveform(
            self._get_signal_quadrature_value,
            t0=t0,
            dt=dt,
            num_steps=num_steps,
            offset_fn=lambda t, dt_: t,
            total_steps=total_steps,
        )

        specs = []
        for residual in self._compute_discrete_3d_phasor_residuals(
            fields,
            dt=float(dt),
            resolution=float(resolution),
        ):
            component = residual.component
            index = residual.index
            target = np.asarray(getattr(fields, component)[index])
            _append_phasor_source_specs(
                specs,
                component=component,
                timing=residual.timing,
                index=index,
                profile=np.asarray(residual.residual, dtype=np.complex128),
                target=target,
                dt=1.0,
                scale_denom=np.asarray(1.0, dtype=np.float64),
                waveform=waveform,
                quadrature_waveform=quadrature_waveform,
                target_shape=tuple(getattr(fields, component).shape),
            )
        return tuple(specs)
