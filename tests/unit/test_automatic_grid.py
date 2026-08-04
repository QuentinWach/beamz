from __future__ import annotations

import numpy as np
import pytest

import beamz as bz
from beamz.design.grid import RectilinearGrid
from beamz.design.mesher import _GradedMesher


def _ring_design():
    wavelength = 1.55 * bz.um
    width, height = 16.0 * bz.um, 13.0 * bz.um
    waveguide_width = 0.6 * bz.um
    gap = 0.25 * bz.um
    radius = 4.0 * bz.um
    bus_y = 2.5 * bz.um
    outer_radius = radius + 0.5 * waveguide_width
    ring_center = (
        0.5 * width,
        bus_y + 0.5 * waveguide_width + gap + outer_radius,
    )
    core = bz.Material(permittivity=2.04**2)
    design = bz.Design(
        width=width,
        height=height,
        background=bz.Material(permittivity=1.444**2),
    )
    design += bz.Rectangle(
        position=(0.0, bus_y - 0.5 * waveguide_width),
        width=width,
        height=waveguide_width,
        material=core,
    )
    design += bz.Ring(
        position=ring_center,
        inner_radius=radius - 0.5 * waveguide_width,
        outer_radius=outer_radius,
        material=core,
    )
    return design, wavelength, bus_y, ring_center, outer_radius, gap


def test_geometry_aware_grid_resolves_ring_bus_and_coupling_gap():
    design, wavelength, bus_y, ring_center, outer_radius, gap = _ring_design()
    spec = bz.GridSpec.auto(
        wavelength=wavelength,
        min_steps_per_wvl=16,
        min_feature_cells=6,
        max_scale=1.2,
    )

    grid = spec.realize(design)
    report = grid.quality_report()
    core_limit = wavelength / (2.04 * 16)
    bus_top = bus_y + 0.3 * bz.um
    ring_bottom = ring_center[1] - outer_radius

    assert report.satisfies_max_scale(1.2 * (1.0 + 1e-12), active_axes=("x", "y"))
    assert np.max(grid.cell_widths("x")) <= core_limit * (1.0 + 1e-12)
    assert np.any(np.isclose(grid.y_edges, bus_top, rtol=1e-12, atol=0.0))
    assert np.any(np.isclose(grid.y_edges, ring_bottom, rtol=1e-12, atol=0.0))
    gap_cells = np.count_nonzero(
        (grid.centers("y") > bus_top) & (grid.centers("y") < ring_bottom)
    )
    assert gap_cells >= 6
    assert np.max(
        grid.cell_widths("y")[
            (grid.centers("y") > ring_center[1] - outer_radius)
            & (grid.centers("y") < ring_center[1] + outer_radius)
        ]
    ) <= core_limit * (1.0 + 1e-12)
    assert ring_bottom - bus_top == pytest.approx(gap)


def test_mesh_override_refines_selected_axis_and_preserves_snapping_point():
    design = bz.Design(width=4.0, height=3.0)
    spec = bz.GridSpec.auto(
        wavelength=2.0,
        min_steps_per_wvl=4,
        max_scale=1.25,
        overrides=(
            bz.MeshOverride(center=(2.0, 1.5), size=(1.0, 1.0), dl=(0.05, None)),
        ),
        snapping_points=((1.25, None),),
    )

    grid = spec.realize(design)

    selected = (grid.centers("x") > 1.5) & (grid.centers("x") < 2.5)
    assert np.max(grid.cell_widths("x")[selected]) <= 0.05 * (1.0 + 1e-12)
    assert np.any(grid.x_edges == 1.25)
    assert grid.axis_quality("x").max_adjacent_ratio <= 1.25 * (1.0 + 1e-12)
    assert np.allclose(grid.cell_widths("y"), grid.cell_widths("y")[0])


def test_grid_spec_auto_is_nonuniform_and_uniform_is_explicit():
    design = bz.Design(width=2.0, height=1.0)

    automatic_spec = bz.GridSpec.auto(wavelength=1.0)
    automatic = automatic_spec.realize(design)
    uniform = bz.GridSpec.uniform(0.1).realize(design)

    assert uniform.metric_kind_for(("x", "y")) == "isotropic_uniform"
    assert uniform.shape[2] == 1
    assert automatic_spec.is_automatic
    assert isinstance(automatic, bz.RectilinearGrid)
    assert not hasattr(bz.GridSpec, "graded")
    with pytest.raises(ValueError, match="GridSpec.auto requires wavelength"):
        bz.GridSpec().realize(design)


def test_uniform_realization_preserves_requested_isotropic_spacing():
    design = bz.Design(width=1.0, height=0.8)

    grid = bz.GridSpec.uniform(0.3).realize(design)

    assert grid.extent == pytest.approx((1.2, 0.9, 1.0))
    assert grid.metric_kind_for(("x", "y")) == "isotropic_uniform"
    np.testing.assert_allclose(grid.cell_widths("x"), 0.3)
    np.testing.assert_allclose(grid.cell_widths("y"), 0.3)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"max_cells_per_axis": 0},
        {"max_cells_per_axis": 1.5},
        {"max_total_cells": True},
    ),
)
def test_grid_budget_limits_must_be_positive_integers(kwargs):
    with pytest.raises(ValueError, match="positive integer or None"):
        bz.GridSpec.auto(wavelength=1.0, **kwargs)


def test_grid_budget_rejects_runaway_refinement_before_rasterization():
    design = bz.Design(width=4.0, height=4.0)
    spec = bz.GridSpec.auto(
        wavelength=1.0,
        max_cells_per_axis=20,
        max_total_cells=200,
    )

    with pytest.raises(ValueError, match="Grid budget exceeded") as error:
        spec.realize(design)

    assert "estimated setup storage" in str(error.value)
    assert "raise dl_min" in str(error.value)


def test_uniform_budget_fails_before_edge_allocation(monkeypatch):
    design = bz.Design(width=1.0, height=1.0)
    spec = bz.GridSpec.uniform(1e-9, max_cells_per_axis=100)

    def unexpected_allocation(*_args, **_kwargs):
        raise AssertionError("uniform edge allocation must not start")

    monkeypatch.setattr(RectilinearGrid, "uniform", unexpected_allocation)

    with pytest.raises(ValueError, match="uniform spacing preflight") as error:
        spec.realize(design)

    assert "smallest requested spacing" in str(error.value)
    assert "x=1,000,000,000" in str(error.value)


def test_graded_total_budget_fails_before_axis_allocation(monkeypatch):
    design = bz.Design(width=4.0, height=4.0)
    spec = bz.GridSpec.auto(
        wavelength=1.0,
        max_cells_per_axis=None,
        max_total_cells=200,
    )

    def unexpected_allocation(*_args, **_kwargs):
        raise AssertionError("graded edge allocation must not start")

    monkeypatch.setattr(_GradedMesher, "make_axis_edges", unexpected_allocation)

    with pytest.raises(ValueError, match="lower-bound preflight") as error:
        spec.realize(design)

    assert "predicted active shape" in str(error.value)
    assert "total limit" in str(error.value)


def test_explicit_resolution_takes_precedence_over_automatic_fields():
    design = bz.Design(width=2.0, height=1.0)
    spec = bz.GridSpec(resolution=0.2, wavelength=1.55)

    grid = spec.realize(design)

    assert not spec.is_automatic
    assert np.allclose(grid.cell_widths("x"), 0.2)
    assert np.allclose(grid.cell_widths("y"), 0.2)


def test_centered_design_geometry_overrides_and_snapping_are_translated():
    material = bz.Material(permittivity=4.0)
    design = bz.Design(background=bz.Material(permittivity=1.0))
    design += bz.Box(
        center=(0.0, 0.0, 0.0),
        size=(2.0 * bz.um, 2.0 * bz.um, 1.0 * bz.um),
        material=material,
    )
    symmetric = bz.GridSpec.auto(wavelength=2.0 * bz.um).realize(design)
    spec = bz.GridSpec.auto(
        wavelength=2.0 * bz.um,
        overrides=(
            bz.MeshOverride(
                center=(-1.5 * bz.um, 0.0),
                size=(0.2 * bz.um, 1.0 * bz.um),
                dl=0.02 * bz.um,
            ),
        ),
        snapping_points=((1.5 * bz.um, None),),
    )

    grid = spec.realize(design)

    assert np.allclose(symmetric.cell_widths("x"), symmetric.cell_widths("x")[::-1])
    assert np.any(np.isclose(grid.x_edges, 0.4 * bz.um, rtol=1e-12, atol=0.0))
    assert np.any(np.isclose(grid.x_edges, 0.6 * bz.um, rtol=1e-12, atol=0.0))
    assert np.any(np.isclose(grid.x_edges, 3.5 * bz.um, rtol=1e-12, atol=0.0))


@pytest.mark.parametrize(
    "structure",
    [
        bz.Taper(
            position=(2.0, 5.0),
            input_width=1.0,
            output_width=0.1,
            length=4.0,
            material=bz.Material(permittivity=1.0),
        ),
        bz.Polygon(
            vertices=((2.0, 4.95), (6.0, 4.95), (6.0, 5.05), (2.0, 5.05)),
            material=bz.Material(permittivity=1.0),
        ),
    ],
)
def test_min_feature_cells_resolves_tapers_and_explicit_polygons(structure):
    design = bz.Design(
        width=10.0,
        height=10.0,
        background=bz.Material(permittivity=1.0),
        structures=(structure,),
    )
    grid = bz.GridSpec.auto(
        wavelength=100.0,
        min_steps_per_wvl=10,
        min_steps_per_sim_size=10,
        min_feature_cells=10,
    ).realize(design)

    output_cells = np.count_nonzero(
        (grid.centers("y") > 4.95) & (grid.centers("y") < 5.05)
    )

    assert output_cells >= 10


def test_polygon_neck_refinement_stays_local_and_axis_aware():
    material = bz.Material(permittivity=1.0)
    dumbbell = bz.Polygon(
        vertices=(
            (10.0, 30.0),
            (40.0, 30.0),
            (40.0, 49.95),
            (60.0, 49.95),
            (60.0, 30.0),
            (90.0, 30.0),
            (90.0, 70.0),
            (60.0, 70.0),
            (60.0, 50.05),
            (40.0, 50.05),
            (40.0, 70.0),
            (10.0, 70.0),
        ),
        material=material,
    )
    design = bz.Design(
        width=100.0,
        height=100.0,
        background=material,
        structures=(dumbbell,),
    )

    grid = bz.GridSpec.auto(
        wavelength=100.0,
        min_steps_per_wvl=10,
        min_steps_per_sim_size=10,
        min_feature_cells=10,
    ).realize(design)

    neck_cells = np.count_nonzero(
        (grid.centers("y") > 49.95) & (grid.centers("y") < 50.05)
    )
    fine_y_cells = grid.centers("y")[grid.cell_widths("y") < 0.02]
    assert neck_cells >= 10
    assert fine_y_cells.size < 40
    assert np.all((fine_y_cells > 49.8) & (fine_y_cells < 50.2))
    assert grid.shape[0] < 100


@pytest.mark.parametrize(
    ("polygon", "axis", "lower", "upper"),
    (
        (
            bz.Polygon(
                vertices=(
                    (20.0, 20.0),
                    (80.0, 20.0),
                    (80.0, 90.0),
                    (50.05, 90.0),
                    (50.05, 40.0),
                    (49.95, 40.0),
                    (49.95, 90.0),
                    (20.0, 90.0),
                ),
                material=bz.Material(permittivity=1.0),
            ),
            "x",
            49.95,
            50.05,
        ),
        (
            bz.Polygon(
                vertices=(
                    (20.0, 20.0),
                    (80.0, 20.0),
                    (80.0, 30.0),
                    (30.0, 30.0),
                    (30.0, 90.0),
                    (20.0, 90.0),
                ),
                material=bz.Material(permittivity=1.0),
            ),
            "x",
            20.0,
            30.0,
        ),
        (
            bz.Polygon(
                vertices=((20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (20.0, 80.0)),
                interiors=(
                    (
                        (20.1, 20.1),
                        (79.9, 20.1),
                        (79.9, 79.9),
                        (20.1, 79.9),
                    ),
                ),
                material=bz.Material(permittivity=1.0),
            ),
            "y",
            20.0,
            20.1,
        ),
    ),
    ids=("u-slot", "l-arm", "polygon-ring-wall"),
)
def test_polygon_local_features_resolve_slots_arms_and_hole_walls(
    polygon, axis, lower, upper
):
    material = bz.Material(permittivity=1.0)
    grid = bz.GridSpec.auto(
        wavelength=100.0,
        min_steps_per_wvl=10,
        min_steps_per_sim_size=10,
        min_feature_cells=10,
    ).realize(
        bz.Design(
            width=100.0,
            height=100.0,
            background=material,
            structures=(polygon,),
        )
    )

    feature_cells = np.count_nonzero(
        (grid.centers(axis) > lower) & (grid.centers(axis) < upper)
    )
    assert feature_cells >= 10


def test_polygon_feature_target_is_invariant_to_curve_tessellation():
    material = bz.Material(permittivity=1.0)
    counts = []
    minimum_spacings = []
    for points in (32, 64, 128, 256, 512):
        ellipse = bz.Circle(
            position=(2.0, 2.0),
            radius=1.0,
            points=points,
            material=material,
        ).scale(1.0, 0.8)
        design = bz.Design(
            width=4.0,
            height=4.0,
            background=material,
            structures=(ellipse,),
        )

        grid = bz.GridSpec.auto(
            wavelength=10.0,
            min_steps_per_wvl=10,
            min_steps_per_sim_size=10,
            min_feature_cells=10,
        ).realize(design)
        counts.append(grid.shape[:2])
        minimum_spacings.append(grid.minimum_spacing)

    assert max(x for x, _y in counts) - min(x for x, _y in counts) <= 2
    assert max(y for _x, y in counts) - min(y for _x, y in counts) <= 2
    assert max(minimum_spacings) / min(minimum_spacings) < 1.02


@pytest.mark.parametrize(
    ("structure", "dimensions"),
    (
        (
            bz.Circle(
                position=(5.0, 5.0),
                radius=0.05,
                material=bz.Material(permittivity=1.0),
            ),
            2,
        ),
        (
            bz.Sphere(
                position=(5.0, 5.0, 5.0),
                radius=0.05,
                material=bz.Material(permittivity=1.0),
            ),
            3,
        ),
    ),
)
def test_min_feature_cells_resolves_circular_primitives(structure, dimensions):
    material = bz.Material(permittivity=1.0)
    design = bz.Design(
        width=10.0,
        height=10.0,
        depth=10.0 if dimensions == 3 else 0.0,
        background=material,
        structures=(structure,),
    )

    grid = bz.GridSpec.auto(
        wavelength=100.0,
        min_steps_per_wvl=10,
        min_steps_per_sim_size=10,
        min_feature_cells=10,
    ).realize(design)

    for axis in "xyz"[:dimensions]:
        feature_cells = np.count_nonzero(
            (grid.centers(axis) > 4.95) & (grid.centers(axis) < 5.05)
        )
        assert feature_cells >= 10
