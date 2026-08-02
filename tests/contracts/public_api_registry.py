"""Executable inventory of BeamZ's top-level public API.

The registry deliberately lives outside a test module so contract tests can share
one authoritative classification instead of maintaining parallel name lists.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

import beamz as bz


@dataclass(frozen=True)
class PublicConfigCase:
    """A constructible public configuration type and its smallest valid example."""

    name: str
    factory: Callable[[], Any]


CONSTANT_EXPORTS = (
    "LIGHT_SPEED",
    "VAC_PERMITTIVITY",
    "VAC_PERMEABILITY",
    "EPS_0",
    "MU_0",
    "um",
    "nm",
    "µm",
    "μm",
    "inf",
)

MODULE_EXPORTS = ("design", "optimization")

FUNCTION_EXPORTS = (
    "ramped_cosine",
    "display_status",
    "create_plain_progress",
    "get_si_scale_and_label",
    "calc_optimal_fdtd_params",
    "dxdt",
)

RUNTIME_EXPORTS = (
    "ModeData",
    "MonitorResults",
    "SimulationResults",
    "SimulationRun",
    "SimulationState",
    "SParameterResult",
)

CONFIGURATION_CASES = (
    PublicConfigCase("Material", bz.Material),
    PublicConfigCase("Design", lambda: bz.Design(width=2.0, height=2.0)),
    PublicConfigCase("Box", bz.Box),
    PublicConfigCase("Rectangle", bz.Rectangle),
    PublicConfigCase("Circle", bz.Circle),
    PublicConfigCase("Ring", bz.Ring),
    PublicConfigCase("CircularBend", bz.CircularBend),
    PublicConfigCase(
        "Polygon",
        lambda: bz.Polygon(vertices=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
    ),
    PublicConfigCase("Taper", bz.Taper),
    PublicConfigCase("Sphere", bz.Sphere),
    PublicConfigCase(
        "ModeSource",
        lambda: bz.ModeSource(
            center=(0.0, 0.0, 0.0),
            size=(0.0, 1.0, 1.0),
            source_time=bz.GaussianPulse(2e14, 2e13),
            direction="+",
        ),
    ),
    PublicConfigCase(
        "GaussianSource",
        lambda: bz.GaussianSource(position=(0.0, 0.0), width=1.0, signal=np.ones(2)),
    ),
    PublicConfigCase(
        "GaussianBeamSource",
        lambda: bz.GaussianBeamSource(
            center=(0.0, 0.0, 0.0),
            size=(1.0, 1.0),
            source_time=np.ones(2),
            wavelength=1.55,
        ),
    ),
    PublicConfigCase(
        "CustomSource",
        lambda: bz.CustomSource(
            component="Ez",
            timing="e",
            index=(slice(None), slice(None)),
            coeff=np.ones((2, 2)),
            waveform=np.ones(2),
            target_shape=(2, 2),
        ),
    ),
    PublicConfigCase(
        "FieldMonitor",
        lambda: bz.FieldMonitor(
            center=(0.0, 0.0, 0.0),
            size=(0.0, 1.0, 1.0),
            freqs=np.array([2e14]),
        ),
    ),
    PublicConfigCase("FieldRecorder", bz.FieldRecorder),
    PublicConfigCase(
        "FluxMonitor",
        lambda: bz.FluxMonitor(
            center=(0.0, 0.0, 0.0),
            size=(0.0, 1.0, 1.0),
            freqs=np.array([2e14]),
        ),
    ),
    PublicConfigCase(
        "ModeMonitor",
        lambda: bz.ModeMonitor(
            center=(0.0, 0.0, 0.0),
            size=(0.0, 1.0, 1.0),
            freqs=np.array([2e14]),
        ),
    ),
    PublicConfigCase(
        "Port",
        lambda: bz.Port(
            center=(0.0, 0.0, 0.0),
            size=(0.0, 1.0, 1.0),
            name="port",
            direction="+",
        ),
    ),
    PublicConfigCase(
        "Simulation",
        lambda: bz.Simulation(
            design=bz.Design(width=2.0, height=2.0),
            time=np.array([0.0, 1e-15]),
        ),
    ),
    PublicConfigCase("GridSpec", bz.GridSpec),
    PublicConfigCase("GaussianPulse", lambda: bz.GaussianPulse(2e14, 2e13)),
    PublicConfigCase("SampledSignal", lambda: bz.SampledSignal(np.ones(2), dt=1e-15)),
    PublicConfigCase("ModeSpec", bz.ModeSpec),
    PublicConfigCase("PML", bz.PML),
    PublicConfigCase("PEC", bz.PEC),
    PublicConfigCase("Absorber", bz.Absorber),
)


def registered_export_names() -> tuple[str, ...]:
    """Return every classified top-level export, preserving registry order."""

    return (
        *CONSTANT_EXPORTS,
        *MODULE_EXPORTS,
        *FUNCTION_EXPORTS,
        *RUNTIME_EXPORTS,
        *(case.name for case in CONFIGURATION_CASES),
    )
