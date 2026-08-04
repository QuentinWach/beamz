from __future__ import annotations

import numpy as np
import pytest

import beamz as bz


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
    assert automatic_spec.nonuniform
    assert isinstance(automatic, bz.RectilinearGrid)
    assert not hasattr(bz.GridSpec, "graded")
    with pytest.raises(ValueError, match="GridSpec.auto requires wavelength"):
        bz.GridSpec(nonuniform=True).realize(design)
