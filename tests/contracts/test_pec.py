import pytest

from beamz import PEC

pytestmark = pytest.mark.unit


def test_pec_thickness_is_zero_for_api_parity():
    assert PEC().thickness == pytest.approx(0.0)


def test_pec_all_edges_resolve_by_dimensionality():
    pec = PEC(edges="all")

    assert pec._get_edges_for_dimensionality(False) == [
        "left",
        "right",
        "top",
        "bottom",
    ]
    assert pec._get_edges_for_dimensionality(True) == [
        "left",
        "right",
        "top",
        "bottom",
        "front",
        "back",
    ]


def test_pec_explicit_edges_are_preserved():
    pec = PEC(edges=["left", "front"])
    assert pec._get_edges_for_dimensionality(True) == ["left", "front"]
