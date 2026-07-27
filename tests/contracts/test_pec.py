import pytest

from beamz import PEC, PML, Absorber

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


@pytest.mark.parametrize("boundary_type", [PEC, PML, Absorber])
def test_boundary_edges_are_normalized_and_validated(boundary_type):
    boundary = boundary_type(edges=["LEFT", "Top"])
    assert boundary.edges == ("left", "top")

    with pytest.raises(ValueError, match="Unsupported boundary edges.*lef"):
        boundary_type(edges=["lef"])


@pytest.mark.parametrize("boundary_type", [PEC, PML, Absorber])
def test_out_of_plane_boundary_edges_require_3d(boundary_type):
    boundary = boundary_type(edges=["front"])

    assert boundary._get_edges_for_dimensionality(True) == ["front"]
    with pytest.raises(ValueError, match="only available in 3D"):
        boundary._get_edges_for_dimensionality(False)
