"""Lean cross-engine reference cases.

The self-contained dipole case runs in every checkout.  The Y-branch case uses
an optional external fixture when reference data is available beside BeamZ,
without vendoring a multi-megabyte mesh into this project.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import label

import beamz as bz
import beamz.design.raster as raster
from beamz.design import MaterialGrid
from beamz.design.raster.importers import from_mesh

NM = 1e-9
SILICON_INDEX = 3.47
SILICA_INDEX = 1.44

pytestmark = pytest.mark.integration


def _reference_data_directory() -> Path:
    configured = os.environ.get("BEAMZ_REFERENCE_DATA")
    if configured:
        return Path(configured).expanduser()
    repository = Path(__file__).resolve().parents[2]
    return repository.parent / "beamz-reference-data"


@pytest.mark.compiled
@pytest.mark.smoke
def test_dipole_geometry_rasterizes_and_advances():
    """Exercise a 200 x 200 x 100 nm builder/dipole smoke case."""

    scene = raster.Scene(
        (
            raster.Material(SILICA_INDEX**2),
            raster.Material(SILICON_INDEX**2),
        ),
        (
            raster.Object(
                raster.Box(
                    (70 * NM, 70 * NM, 30 * NM),
                    (130 * NM, 130 * NM, 70 * NM),
                ),
                material_id=1,
            ),
        ),
    )
    result = raster.rasterize(
        scene,
        raster.Grid.uniform(
            (0.0, 0.0, 0.0),
            (200 * NM, 200 * NM, 100 * NM),
            (10, 10, 5),
        ),
        options=raster.RasterOptions(
            quality="reference",
            smoothing="farjadpour_diagonal",
        ),
    )
    material_grid = MaterialGrid.from_raster_result(result, dimensions=3)

    dt = 0.95 * 20 * NM / (bz.LIGHT_SPEED * np.sqrt(3.0))
    signal = np.asarray((0.0, 1.0, 0.5, 0.0, 0.0, 0.0))
    simulation = bz.Simulation(
        material_grid=material_grid,
        sources=(
            bz.GaussianSource(
                position=(100 * NM, 100 * NM, 50 * NM),
                width=20 * NM,
                signal=signal,
            ),
        ),
        boundaries=(bz.PEC(edges="all"),),
        time=np.arange(signal.size) * dt,
        normalize_source=None,
    )
    run = simulation.advance(num_steps=4, progress=False)

    assert material_grid.uses_direct_yee_materials
    assert np.isclose(material_grid.permittivity.min(), SILICA_INDEX**2)
    assert np.isclose(material_grid.permittivity.max(), SILICON_INDEX**2)
    fields = tuple(
        np.asarray(getattr(run.state, name))
        for name in ("ex", "ey", "ez", "hx", "hy", "hz")
    )
    assert all(np.isfinite(field).all() for field in fields)
    assert sum(float(np.vdot(field, field).real) for field in fields) > 0.0


@pytest.mark.slow
@pytest.mark.characterization
def test_external_y_branch_crosses_rasterizer_engine_boundary():
    """Exercise an external Y-branch mesh when reference data is present."""

    path = _reference_data_directory() / "y_branch" / "mesh.msh"
    if not path.is_file():
        pytest.skip(
            "reference fixture unavailable; set BEAMZ_REFERENCE_DATA to its data directory"
        )
    pytest.importorskip("meshio")

    scene = from_mesh(
        path,
        materials={
            "core": raster.Material(SILICON_INDEX**2),
            "SiO2": raster.Material(SILICA_INDEX**2),
        },
        background=raster.Material(SILICA_INDEX**2),
        unit_scale=NM,
    )
    core = next(
        object_
        for object_ in scene.objects
        if scene.materials[object_.material_id].epsilon_r[0] > 10.0
    )
    core_scene = raster.Scene(
        (
            raster.Material(SILICA_INDEX**2),
            raster.Material(SILICON_INDEX**2),
        ),
        (raster.Object(core.geometry, material_id=1),),
    )
    cross_section = raster.rasterize(
        core_scene,
        raster.Grid.uniform(
            (-7.6e-6, -3.2e-6, 100 * NM),
            (7.6e-6, 3.2e-6, 120 * NM),
            (152, 64, 1),
        ),
        options=raster.RasterOptions(quality="balanced"),
    )
    epsilon = np.asarray(cross_section.tensors["epsilon"])[0, 0]
    low = SILICA_INDEX**2
    high = SILICON_INDEX**2
    fractions = np.clip((epsilon - low) / (high - low), 0.0, 1.0)
    _, connected_components = label(fractions > 0.5)
    core_report = raster.inspect_mesh(
        core.geometry.vertices,
        core.geometry.triangles,
    )
    used_vertices = core.geometry.vertices[np.unique(core.geometry.triangles)]
    core_thickness = float(np.ptp(used_vertices[:, 2]))
    expected_area = abs(core_report.signed_volume) / core_thickness
    rasterized_area = float(fractions.sum()) * (100 * NM) ** 2

    compiled_scene = raster.compile_scene(scene)
    result = compiled_scene.rasterize(
        raster.Grid.uniform(
            (-9e-6, -4.5e-6, -3e-6),
            (9e-6, 4.5e-6, 2e-6),
            (36, 18, 10),
        ),
        options=raster.RasterOptions(quality="fast"),
    )
    material_grid = MaterialGrid.from_raster_result(result, dimensions=3)
    program = bz.Simulation(material_grid=material_grid, run_time=1e-15).compile(
        num_steps=1
    )

    assert len(scene.objects) == 2
    assert connected_components == 1
    assert rasterized_area == pytest.approx(expected_area, rel=0.01)
    assert np.isclose(material_grid.permittivity.min(), SILICA_INDEX**2)
    assert material_grid.permittivity.max() > SILICA_INDEX**2
    assert program.grid.eps_x.shape == material_grid.yee_materials["eps_x"].shape
