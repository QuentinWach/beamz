import numpy as np

from beamz import Circle, Design, Material, Polygon, Rectangle
from beamz.design.core import _normalize_aa_config
from beamz.design.meshing import RegularGrid, RegularGrid3D, create_mesh


def _build_circle_design():
    design = Design(width=10.0, height=10.0, material=Material(permittivity=1.0))
    design += Circle(
        position=(5.0, 5.0),
        radius=2.65,
        material=Material(permittivity=12.0),
    )
    return design


def test_default_antialias_matches_legacy_grid_64(monkeypatch):
    monkeypatch.setenv("BEAMZ_RASTER_CACHE", "0")
    monkeypatch.setenv("BEAMZ_RASTER_TIMING", "0")

    design = _build_circle_design()
    default_grid = design.rasterize(resolution=0.5, force_recompute=True)
    default_explicit_grid = design.rasterize(
        resolution=0.5,
        force_recompute=True,
        aa_mode="legacy_grid",
        aa_samples=64,
        aa_seed=0,
    )
    np.testing.assert_allclose(
        default_grid.permittivity, default_explicit_grid.permittivity
    )


def test_default_antialias_preserves_circle_symmetry(monkeypatch):
    monkeypatch.setenv("BEAMZ_RASTER_CACHE", "0")
    monkeypatch.setenv("BEAMZ_RASTER_TIMING", "0")

    design = Design(width=6.0, height=6.0, material=Material(permittivity=1.0))
    design += Circle(
        position=(3.0, 3.0),
        radius=1.35,
        material=Material(permittivity=12.0),
    )
    grid = design.rasterize(resolution=0.2, force_recompute=True)

    np.testing.assert_allclose(grid.permittivity, np.fliplr(grid.permittivity))
    np.testing.assert_allclose(grid.permittivity, np.flipud(grid.permittivity))
    np.testing.assert_allclose(grid.permittivity, np.rot90(grid.permittivity, 2))


def test_stratified_jitter_is_deterministic_for_fixed_seed(monkeypatch):
    monkeypatch.setenv("BEAMZ_RASTER_CACHE", "0")
    monkeypatch.setenv("BEAMZ_RASTER_TIMING", "0")

    design = _build_circle_design()
    grid_a = design.rasterize(
        resolution=0.5,
        force_recompute=True,
        aa_mode="stratified_jitter",
        aa_samples=16,
        aa_seed=1234,
    )
    grid_b = design.rasterize(
        resolution=0.5,
        force_recompute=True,
        aa_mode="stratified_jitter",
        aa_samples=16,
        aa_seed=1234,
    )
    np.testing.assert_allclose(grid_a.permittivity, grid_b.permittivity)


def test_switching_antialias_mode_recomputes_grid(monkeypatch):
    monkeypatch.setenv("BEAMZ_RASTER_CACHE", "0")
    monkeypatch.setenv("BEAMZ_RASTER_TIMING", "0")

    design = _build_circle_design()
    legacy_grid = design.rasterize(
        resolution=0.5,
        force_recompute=True,
        aa_mode="legacy_grid",
        aa_samples=9,
        aa_seed=0,
    )
    jitter_grid = design.rasterize(
        resolution=0.5,
        aa_mode="stratified_jitter",
        aa_samples=16,
        aa_seed=1234,
    )

    assert jitter_grid is not legacy_grid
    assert not np.allclose(legacy_grid.permittivity, jitter_grid.permittivity)


def test_legacy_grid_single_sample_is_centered(monkeypatch):
    monkeypatch.setenv("BEAMZ_RASTER_CACHE", "0")
    monkeypatch.setenv("BEAMZ_RASTER_TIMING", "0")

    design = _build_circle_design()
    grid = RegularGrid(
        design,
        resolution=0.5,
        aa_mode="legacy_grid",
        aa_samples=1,
        aa_seed=0,
    )

    sample_dx, sample_dy = grid._build_supersample_offsets_xy(cell_size=0.5)
    np.testing.assert_allclose(sample_dx, np.array([0.0]))
    np.testing.assert_allclose(sample_dy, np.array([0.0]))


def test_axis_aligned_rectangle_uses_exact_area_fraction(monkeypatch):
    monkeypatch.setenv("BEAMZ_RASTER_CACHE", "0")
    monkeypatch.setenv("BEAMZ_RASTER_TIMING", "0")

    design = Design(width=2.0, height=1.0, material=Material(permittivity=1.0))
    design += Rectangle(
        position=(0.5, 0.0),
        width=1.0,
        height=1.0,
        material=Material(permittivity=13.0),
    )
    grid = design.rasterize(
        resolution=1.0,
        force_recompute=True,
        aa_mode="legacy_grid",
        aa_samples=64,
    )

    np.testing.assert_allclose(grid.permittivity[0], np.array([7.0, 7.0]))


def test_polygon_uses_exact_area_fraction(monkeypatch):
    monkeypatch.setenv("BEAMZ_RASTER_CACHE", "0")
    monkeypatch.setenv("BEAMZ_RASTER_TIMING", "0")

    design = Design(width=1.0, height=1.0, material=Material(permittivity=1.0))
    design += Polygon(
        vertices=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        material=Material(permittivity=5.0),
    )
    grid = design.rasterize(
        resolution=1.0,
        force_recompute=True,
        aa_mode="legacy_grid",
        aa_samples=64,
    )

    np.testing.assert_allclose(grid.permittivity, np.array([[3.0]]))


def test_create_mesh_ignores_resolution_z_for_2d_design():
    design = _build_circle_design()
    grid = create_mesh(
        design,
        resolution=0.5,
        auto_select=False,
        force_3d=False,
        resolution_z=0.1,
    )
    assert isinstance(grid, RegularGrid)


def test_explicit_grid_type_cache_signature_distinguishes_2d_and_3d(monkeypatch):
    monkeypatch.setenv("BEAMZ_RASTER_CACHE", "0")
    monkeypatch.setenv("BEAMZ_RASTER_TIMING", "0")

    design = _build_circle_design()
    grid_3d = design.rasterize(
        resolution=0.5,
        grid_type=RegularGrid3D,
        force_recompute=True,
        aa_mode="legacy_grid",
        aa_samples=1,
        aa_seed=0,
    )
    grid_2d = design.rasterize(
        resolution=0.5,
        grid_type=RegularGrid,
        aa_mode="legacy_grid",
        aa_samples=1,
        aa_seed=0,
    )

    assert isinstance(grid_3d, RegularGrid3D)
    assert isinstance(grid_2d, RegularGrid)
    assert grid_2d is not grid_3d


def test_stratified_jitter_per_cell_scramble_is_deterministic_and_decorrelated(
    monkeypatch,
):
    monkeypatch.setenv("BEAMZ_RASTER_CACHE", "0")
    monkeypatch.setenv("BEAMZ_RASTER_TIMING", "0")

    design = _build_circle_design()
    grid = RegularGrid(
        design,
        resolution=0.5,
        aa_mode="stratified_jitter",
        aa_samples=16,
        aa_seed=1234,
    )
    base_dx, base_dy = grid._build_supersample_offsets_xy(cell_size=0.5)

    cell_a_dx, cell_a_dy = grid._scramble_offsets_xy_for_cell(
        base_dx, base_dy, cell_size=0.5, cell_i=7, cell_j=11
    )
    cell_a2_dx, cell_a2_dy = grid._scramble_offsets_xy_for_cell(
        base_dx, base_dy, cell_size=0.5, cell_i=7, cell_j=11
    )
    cell_b_dx, cell_b_dy = grid._scramble_offsets_xy_for_cell(
        base_dx, base_dy, cell_size=0.5, cell_i=7, cell_j=12
    )

    np.testing.assert_allclose(cell_a_dx, cell_a2_dx)
    np.testing.assert_allclose(cell_a_dy, cell_a2_dy)
    assert not (np.allclose(cell_a_dx, cell_b_dx) and np.allclose(cell_a_dy, cell_b_dy))


def test_normalize_aa_config_tracks_scramble_policy():
    default_cfg = _normalize_aa_config({})
    jitter_cfg = _normalize_aa_config({"aa_mode": "stratified_jitter"})
    legacy_cfg = _normalize_aa_config({"aa_mode": "legacy_grid"})

    assert default_cfg["mode"] == "legacy_grid"
    assert default_cfg["scramble"] == "none"
    assert jitter_cfg["scramble"] == "cp_cell_v1"
    assert legacy_cfg["scramble"] == "none"
