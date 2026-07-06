from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beamz.const import LIGHT_SPEED
from beamz.devices._runtime import RuntimeStateProxy
from beamz.devices.sources._custom_profile import _CustomFieldProfileSource
from beamz.devices.sources._gaussian_beam import GaussianBeamProfile
from beamz.devices.sources._profiles import FieldProfile3D
from beamz.devices.sources._time import (
    _analytic_signal_quadrature,
    _interpolate_time_signal,
)


@dataclass
class _GaussianBeamSourceState:
    """Mutable prepared state for GaussianBeamSource."""

    _field_profile: FieldProfile3D | None = None
    _field_profile_signature: tuple | None = None
    _sampled_waveform: tuple[np.ndarray, np.ndarray] | None = None
    _sampled_time: tuple[float, float] | None = None


class GaussianBeamSource(RuntimeStateProxy):
    """Public Gaussian beam source backed by planar launched-side TF/SF residuals."""

    _RUNTIME_ATTRS = frozenset(_GaussianBeamSourceState.__annotations__)

    def __init__(
        self,
        *,
        center,
        size,
        source_time,
        direction="-z",
        angle_theta=0.0,
        angle_phi=0.0,
        pol_angle=0.0,
        waist_radius=None,
        waist_distance=0.0,
        power=1.0,
        background_index=1.0,
        wavelength=None,
        max_shift=1,
    ):
        self.center = tuple(float(v) for v in center)
        if len(self.center) != 3:
            raise ValueError("GaussianBeamSource center must be a 3D coordinate.")
        self.size = size
        self.source_time = source_time
        self.direction = str(direction)
        self.angle_theta = float(angle_theta)
        self.angle_phi = float(angle_phi)
        self.pol_angle = float(pol_angle)
        self.waist_radius = waist_radius
        self.waist_distance = float(waist_distance)
        power_value = float(power)
        if not np.isfinite(power_value) or power_value < 0.0:
            raise ValueError(
                "GaussianBeamSource power must be a non-negative finite value, "
                f"got {power!r}."
            )
        self.power = power_value
        self.background_index = float(background_index)
        self.wavelength = None if wavelength is None else float(wavelength)
        self.max_shift = int(max(1, max_shift))
        self._state = _GaussianBeamSourceState()

    def shifted(self, offset):
        copied = self.copy()
        copied.center = tuple(
            a + b for a, b in zip(self.center, tuple(offset), strict=True)
        )
        return copied

    def copy(self, *, update=None):
        import copy

        copied = copy.deepcopy(self)
        if update:
            for key, value in dict(update).items():
                setattr(copied, key, value)
        copied._state = _GaussianBeamSourceState()
        return copied

    def source_spectrum(self, freqs, *, normalize: bool = True) -> np.ndarray | None:
        source_time = self.source_time
        if source_time is None:
            return None
        freq_arr = np.asarray(freqs, dtype=float)
        if normalize and hasattr(source_time, "dft_normalization_spectrum"):
            return np.asarray(
                source_time.dft_normalization_spectrum(freq_arr),
                dtype=np.complex128,
            )
        if hasattr(source_time, "spectrum"):
            try:
                return np.asarray(
                    source_time.spectrum(freq_arr, normalize=normalize),
                    dtype=np.complex128,
                )
            except TypeError:
                return np.asarray(source_time.spectrum(freq_arr), dtype=np.complex128)
        return None

    def _wavelength(self) -> float:
        if self.wavelength is not None:
            return float(self.wavelength)
        freq0 = getattr(self.source_time, "freq0", None)
        if freq0 is None:
            raise ValueError(
                "GaussianBeamSource requires wavelength=... when source_time has no "
                "freq0 attribute."
            )
        freq0 = float(freq0)
        if not np.isfinite(freq0) or freq0 <= 0.0:
            raise ValueError("GaussianBeamSource source_time.freq0 must be positive.")
        return float(LIGHT_SPEED / freq0)

    def _waist_radius(self) -> float:
        if self.waist_radius is not None:
            return float(self.waist_radius)
        values = np.asarray(self.size, dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError("GaussianBeamSource size must not be empty.")
        return 0.25 * float(np.min(values))

    def _field_profile_for_fields(self, fields, *, resolution: float) -> FieldProfile3D:
        permittivity = np.asarray(getattr(fields, "permittivity"))
        if permittivity.ndim != 3:
            raise ValueError(
                "GaussianBeamSource currently supports 3D simulations only."
            )
        grid_shape = tuple(int(v) for v in permittivity.shape)
        signature = (
            grid_shape,
            float(resolution),
            self.center,
            repr(self.size),
            self.direction,
            float(self.angle_theta),
            float(self.angle_phi),
            float(self.pol_angle),
            float(self._waist_radius()),
            float(self.waist_distance),
            float(self._wavelength()),
            float(self.background_index),
            float(self.power),
        )
        if (
            self._field_profile is not None
            and self._field_profile_signature == signature
        ):
            return self._field_profile
        generator = GaussianBeamProfile(
            center=self.center,
            size=self.size,
            direction=self.direction,
            angle_theta=self.angle_theta,
            angle_phi=self.angle_phi,
            pol_angle=self.pol_angle,
            waist_radius=self._waist_radius(),
            waist_distance=self.waist_distance,
            wavelength=self._wavelength(),
            background_index=self.background_index,
            power=self.power,
        )
        self._field_profile = generator.field_profile(
            resolution=float(resolution),
            grid_shape=grid_shape,
        )
        self._field_profile_signature = signature
        return self._field_profile

    def _sample_source_time(
        self,
        *,
        t0: float,
        dt: float,
        num_steps: int,
        total_steps: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = int(total_steps if total_steps is not None else num_steps)
        time = float(t0) + np.arange(n, dtype=np.float64) * float(dt)
        source_time = self.source_time
        if hasattr(source_time, "sample"):
            signal, quadrature = source_time.sample(time)
        elif callable(source_time):
            signal = np.asarray([float(source_time(float(t))) for t in time])
            quadrature = np.zeros_like(signal)
        else:
            signal = np.asarray(source_time, dtype=np.float64).reshape(-1)
            if signal.size != n:
                padded = np.zeros((n,), dtype=np.float64)
                padded[: min(n, int(signal.size))] = signal[:n]
                signal = padded
            quadrature = _analytic_signal_quadrature(signal)

        signal = np.asarray(signal, dtype=np.float64).reshape(-1)
        quadrature = np.asarray(quadrature, dtype=np.float64).reshape(-1)
        self._sampled_waveform = (signal, quadrature)
        self._sampled_time = (float(t0), float(dt))
        return signal, quadrature

    def _sampled_value(self, time, dt, part: int):
        if self._sampled_waveform is None:
            return 0.0
        t0, sample_dt = self._sampled_time or (0.0, float(dt))
        return _interpolate_time_signal(
            self._sampled_waveform[int(part)],
            float(time) - float(t0),
            sample_dt,
        )

    def _get_signal_value(self, time, dt):
        return self._sampled_value(time, dt, 0)

    def _get_signal_quadrature_value(self, time, dt):
        return self._sampled_value(time, dt, 1)

    def _profile_source(
        self,
        fields,
        *,
        dt: float,
        resolution: float,
        num_steps: int,
        t0: float,
        total_steps: int | None = None,
    ) -> _CustomFieldProfileSource:
        signal, quadrature = self._sample_source_time(
            t0=t0,
            dt=dt,
            num_steps=num_steps,
            total_steps=total_steps,
        )
        return _CustomFieldProfileSource(
            profile=self._field_profile_for_fields(fields, resolution=resolution),
            signal=signal,
            signal_quadrature=quadrature,
            power=1.0,
            max_shift=self.max_shift,
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
        return self._profile_source(
            fields,
            dt=dt,
            resolution=resolution,
            num_steps=num_steps,
            t0=t0,
            total_steps=total_steps,
        ).compile_source_specs(
            fields=fields,
            dt=dt,
            num_steps=num_steps,
            t0=t0,
            resolution=resolution,
            total_steps=total_steps,
        )
