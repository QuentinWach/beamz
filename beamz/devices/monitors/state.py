from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from beamz.devices.monitors.spec import MonitorSpec


@dataclass(slots=True)
class MonitorRecorder:
    fields: dict[str, list]
    power_accumulated: np.ndarray | None
    energy_history: list
    power_history: list
    power_timestamps: list
    power_accumulation_count: int
    step_count: int
    last_record_step: int
    frequency_flux_spectrum: np.ndarray
    objective_value: float | None
    _dft_accum: dict = field(default_factory=dict)
    _dft_weight_sum: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=float))
    _dft_sample_count: int = 0
    _dft_phase: np.ndarray = field(default_factory=lambda: np.ones((0,), dtype=np.complex128))
    _dft_last_t: float | None = None
    _dft_last_dt: float | None = None
    _dft_last_rot: np.ndarray | None = None

    @classmethod
    def create(cls, spec) -> "MonitorRecorder":
        return create_monitor_state(spec)


def create_monitor_state(spec) -> MonitorRecorder:
    if not isinstance(spec, MonitorSpec):
        raise TypeError("create_monitor_state expects a MonitorSpec")
    fields = {
        "Ex": [],
        "Ey": [],
        "Ez": [],
        "Hx": [],
        "Hy": [],
        "Hz": [],
        "t": [],
    }
    nfreq = int(spec.dft_frequencies.size)
    return MonitorRecorder(
        fields=fields,
        power_accumulated=None,
        energy_history=[],
        power_history=[],
        power_timestamps=[],
        power_accumulation_count=0,
        step_count=0,
        last_record_step=-1,
        frequency_flux_spectrum=np.zeros(spec.frequency_points.shape, dtype=np.complex64),
        objective_value=None,
        _dft_accum={},
        _dft_weight_sum=np.zeros(nfreq, dtype=float),
        _dft_sample_count=0,
        _dft_phase=np.ones(nfreq, dtype=np.complex128),
        _dft_last_t=None,
        _dft_last_dt=None,
        _dft_last_rot=None,
    )


def monitor_state_for(spec, *, monitor=None, state=None):
    if isinstance(state, MonitorRecorder):
        return state
    monitor_state = getattr(monitor, "state", None)
    if isinstance(monitor_state, MonitorRecorder):
        return monitor_state
    return create_monitor_state(spec)
