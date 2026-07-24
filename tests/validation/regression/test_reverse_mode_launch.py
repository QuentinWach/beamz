"""Reciprocal transmission evidence and a locked reverse-launch limitation."""

from __future__ import annotations

import numpy as np
import pytest

from tests.modal_sparam_physical_case import (
    StraightWaveguideSParamConfig,
    run_straight_waveguide_sparam_case,
)


@pytest.fixture(scope="module")
def reciprocal_straight_waveguide_results():
    """Run the smallest two source excitations once for this module."""
    config = StraightWaveguideSParamConfig()
    forward = run_straight_waveguide_sparam_case(
        resolution_ppw=10,
        cfg=config,
        launch_direction="+",
    )
    reverse = run_straight_waveguide_sparam_case(
        resolution_ppw=10,
        cfg=config,
        launch_direction="-",
    )
    center = int(np.argmin(np.abs(forward.wavelengths_um - config.wavelength_um)))
    return config, forward, reverse, center


@pytest.mark.simulation
@pytest.mark.compiled
def test_straight_waveguide_complex_transmission_is_reciprocal(
    reciprocal_straight_waveguide_results,
    validation_metrics,
):
    """Opposite source excitations produce the same de-embedded transmission."""
    config, forward, reverse, center = reciprocal_straight_waveguide_results
    s21 = complex(forward.s21_by_monitor["far"][center])
    s12 = complex(reverse.s21_by_monitor["far"][center])
    reciprocity_residual = abs(s21 - s12)

    validation_metrics.check(
        "complex transmission reciprocity |S21-S12|",
        measured=reciprocity_residual,
        reference=0.0,
        tolerance="sparameter_reciprocity",
        resolution="10 ppw",
        metadata={
            "device": "symmetric straight slab waveguide",
            "wavelength_um": config.wavelength_um,
            "reference_plane_separation_um": abs(
                forward.monitor_x_um["far"] - forward.monitor_x_um["o1"]
            ),
            "forward_transmission": {
                "real": float(np.real(s21)),
                "imag": float(np.imag(s21)),
            },
            "reverse_transmission": {
                "real": float(np.real(s12)),
                "imag": float(np.imag(s12)),
            },
        },
    )


@pytest.mark.simulation
@pytest.mark.compiled
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Reverse 2-D ModeSource currently leaks about -10.5 dB into the "
        "counter-propagating branch; keep this locked until leapfrog phase "
        "compensation is made direction-symmetric."
    ),
)
def test_reverse_mode_launch_has_low_spurious_counterpropagation(
    reciprocal_straight_waveguide_results,
):
    """Future fix target: reverse-source reflection should match the forward case."""
    _config, _forward, reverse, center = reciprocal_straight_waveguide_results
    reverse_reflection_db = 20.0 * np.log10(max(float(abs(reverse.s11[center])), 1e-12))
    assert reverse_reflection_db < -35.0
