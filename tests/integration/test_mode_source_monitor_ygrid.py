from __future__ import annotations

import numpy as np
import pytest

from beamz.analysis import mode_projection as mp
from beamz.devices.modes.fields import (
    _modal_overlap,
    _modal_power,
    _normalize_profiles,
)

_COMPONENTS_3D = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _unit_flux_x_mode(mask):
    e = np.zeros((4, 5), dtype=np.complex128)
    e[mask] = 1.0
    zeros = np.zeros_like(e)
    forward = {
        "Ex": zeros.copy(),
        "Ey": e.copy(),
        "Ez": zeros.copy(),
        "Hx": zeros.copy(),
        "Hy": zeros.copy(),
        "Hz": e.copy(),
    }
    _normalize_profiles(
        forward,
        axis="x",
        measure=0.25,
        direction_sign=1.0,
    )
    backward = {
        name: (-value if name.startswith("H") else value.copy())
        for name, value in forward.items()
    }
    return forward, backward


def _projection_from_basis(forward, backward):
    return {
        "components": ("Ey", "Ez", "Hy", "Hz"),
        "axis": "x",
        "d_area": 0.25,
        "direction_sign": 1.0,
        "mode_components": forward,
        "mode_components_bwd": backward,
        "overlap_matrix": np.asarray(
            [
                [
                    _modal_overlap(forward, forward, "x", 0.25),
                    _modal_overlap(forward, backward, "x", 0.25),
                ],
                [
                    _modal_overlap(backward, forward, "x", 0.25),
                    _modal_overlap(backward, backward, "x", 0.25),
                ],
            ],
            dtype=np.complex128,
        ),
    }


@pytest.mark.unit
def test_grouped_3d_modal_projection_recovers_exact_multimode_coefficients():
    fwd0, bwd0 = _unit_flux_x_mode((slice(0, 2), slice(0, 2)))
    fwd1, bwd1 = _unit_flux_x_mode((slice(2, 4), slice(3, 5)))
    coeff_true = (
        (0.8 - 0.2j, -0.1 + 0.3j),
        (-0.4 + 0.5j, 0.25 + 0.15j),
    )
    field = {}
    for component in _COMPONENTS_3D:
        field[component] = (
            coeff_true[0][0] * fwd0[component]
            + coeff_true[0][1] * bwd0[component]
            + coeff_true[1][0] * fwd1[component]
            + coeff_true[1][1] * bwd1[component]
        )

    coeffs, residual, condition, diagnostics = mp._project_modal_coefficients_3d_group(
        field,
        (
            _projection_from_basis(fwd0, bwd0),
            _projection_from_basis(fwd1, bwd1),
        ),
    )

    for actual, expected in zip(coeffs, coeff_true, strict=True):
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-12)
    assert residual < 1e-12
    assert condition < 10.0
    assert diagnostics["residual_balanced"] < 1e-12
    expected_power = sum(
        abs(forward) ** 2 - abs(backward) ** 2 for forward, backward in coeff_true
    )
    assert diagnostics["projected_signed_power"] == pytest.approx(expected_power)


@pytest.mark.unit
def test_signed_flux_subtracts_backward_modal_power():
    forward, backward = _unit_flux_x_mode((slice(0, 4), slice(0, 5)))
    a_forward = 1.1 - 0.2j
    a_backward = 0.3 + 0.1j
    field = {
        component: a_forward * forward[component] + a_backward * backward[component]
        for component in _COMPONENTS_3D
    }

    flux = _modal_power(
        field,
        axis="x",
        measure=0.25,
        direction_sign=1.0,
    )

    assert flux == pytest.approx(abs(a_forward) ** 2 - abs(a_backward) ** 2)
    assert flux < abs(a_forward) ** 2
