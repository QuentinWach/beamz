"""Contracts for the canonical complete Yee representation and PEC masks."""

from __future__ import annotations

import numpy as np

from beamz.devices._boundary_compile import (
    compile_metallic_masks,
    resolve_metallic_edges,
)
from beamz.devices.boundaries import PEC, PML, normalize_boundaries
from beamz.lattice import component_shape_3d
from tests.utils import compiled_grid


def _fields(shape=(4, 5, 6)):
    return compiled_grid(
        np.ones(shape, dtype=np.float32),
        np.zeros(shape, dtype=np.float32),
        np.ones(shape, dtype=np.float32),
        resolution=1.0,
    )


def test_complete_yee_shapes_retain_every_domain_wall():
    fields = _fields()
    expected = {
        "Ex": (5, 6, 6),
        "Ey": (5, 5, 7),
        "Ez": (4, 6, 7),
        "Hx": (4, 5, 7),
        "Hy": (4, 6, 6),
        "Hz": (5, 5, 6),
    }

    for component, shape in expected.items():
        assert component_shape_3d(component, (4, 5, 6)) == shape
        assert getattr(fields, component).shape == shape


def test_default_boundary_is_explicit_six_sided_pec():
    boundaries = normalize_boundaries([])

    assert boundaries == (PEC(),)
    assert resolve_metallic_edges(boundaries, is_3d=True) == {
        "left",
        "right",
        "bottom",
        "top",
        "front",
        "back",
    }


def test_complete_yee_masks_cover_both_sides_of_every_pec_axis():
    fields = _fields()
    masks = compile_metallic_masks(
        fields.component_shapes, fields.material_grid.shape, [PEC()]
    )

    # Tangential E and normal H samples are constrained on both faces of each axis.
    for component in ("Ex", "Ey", "Hz"):
        assert np.asarray(masks[component])[0].all()
        assert np.asarray(masks[component])[-1].all()
    for component in ("Ex", "Ez", "Hy"):
        assert np.asarray(masks[component])[:, 0].all()
        assert np.asarray(masks[component])[:, -1].all()
    for component in ("Ey", "Ez", "Hx"):
        assert np.asarray(masks[component])[:, :, 0].all()
        assert np.asarray(masks[component])[:, :, -1].all()


def test_partial_pec_only_masks_selected_faces():
    fields = _fields()
    masks = compile_metallic_masks(
        fields.component_shapes,
        fields.material_grid.shape,
        [PEC(edges=("left", "top"))],
    )

    assert np.asarray(masks["Ez"])[:, :, 0].all()
    assert np.asarray(masks["Ez"])[:, -1, :].all()
    assert not np.asarray(masks["Ez"])[:, :-1, -1].any()
    assert not np.asarray(masks["Ez"])[:, 0, 1:].any()


def test_absorbing_boundary_opens_matching_pec_faces():
    boundaries = [PEC(), PML(edges=("left", "right"))]

    assert resolve_metallic_edges(boundaries, is_3d=True) == {
        "bottom",
        "top",
        "front",
        "back",
    }
