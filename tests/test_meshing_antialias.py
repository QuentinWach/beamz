import numpy as np
import pytest

from beamz import Circle, CustomMaterial, Design, Material, Polygon, Rectangle
from beamz.design.core import _normalize_aa_config
from beamz.design.meshing import (
    MaterialGrids,
    RegularGrid,
    RegularGrid3D,
    create_mesh,
)


def _build_circle_design():
    design = Design(width=10.0, height=10.0, material=Material(permittivity=1.0))
    design += Circle(
        position=(5.0, 5.0),
        radius=2.65,
        material=Material(permittivity=12.0),
    )
    return design


@pytest.mark.parametrize(
    "grid_name",
    ("permittivity_grid", "permeability_grid", "conductivity_grid"),
)
def test_grid_backed_custom_material_requires_bounds(grid_name):
    with pytest.raises(ValueError, match="requires bounds"):
        CustomMaterial(**{grid_name: np.full((2, 2), 12.0)})


def test_custom_material_maximum_covers_sampled_permittivity():
    with pytest.raises(ValueError, match="must include sampled"):
        CustomMaterial(
            permittivity_grid=np.full((2, 2), 12.0),
            bounds=((0.0, 1.0), (0.0, 1.0)),
            max_permittivity=11.0,
        )


def test_custom_material_evaluation_failure_aborts_rasterization(monkeypatch):
    monkeypatch.setenv("BEAMZ_RASTER_TIMING", "0")

    def broken_permittivity(x, y):
        raise RuntimeError("material model failed")

    material = CustomMaterial(
        permittivity_func=broken_permittivity,
        cache_key="broken",
        max_permittivity=12.0,
    )
    design = Design(width=1.0, height=1.0).with_structure(
        Rectangle(
            position=(0.0, 0.0),
            width=1.0,
            height=1.0,
            material=material,
        )
    )

    with pytest.raises(RuntimeError, match="material model failed"):
        design.rasterize(resolution=0.5, force_recompute=True)


def test_raster_progress_is_opt_in(capsys):
    _build_circle_design().rasterize(resolution=0.5, progress=True)

    assert "Rasterizing structures" in capsys.readouterr().out


def test_2d_grid_for_3d_design_uses_runtime_warning():
    design = Design(width=1.0, height=1.0, depth=1.0, material=Material(1.0))

    with pytest.warns(RuntimeWarning, match="RegularGrid3D"):
        RegularGrid(design, resolution=0.5)


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
    assert not grid.permittivity.flags.writeable
    with pytest.raises(AttributeError, match="grids are immutable"):
        grid.resolution = 1.0
    with pytest.raises(ValueError):
        grid.permittivity[0, 0] = 1.0
    original = np.array(grid.permittivity, copy=True)
    changed = grid.updated_copy(permittivity=np.full(grid.shape, 2.0))
    np.testing.assert_allclose(changed.permittivity, 2.0)
    assert not changed.permittivity.flags.writeable
    np.testing.assert_array_equal(grid.permittivity, original)


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


def test_polygon_shift_keeps_z_field_and_vertices_consistent():
    poly = Polygon(
        vertices=[(0.0, 0.0, 0.2), (1.0, 0.0, 0.2), (0.0, 1.0, 0.2)],
        depth=0.3,
        z=0.2,
        material=Material(permittivity=5.0),
    )

    shifted = poly.shift(2.0, 3.0, 4.0)

    assert shifted.z == 4.2
    np.testing.assert_allclose([vertex[2] for vertex in shifted.vertices], 4.2)


def test_polygon_2d_vertices_inherit_explicit_z_and_bounding_box():
    poly = Polygon(
        vertices=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        depth=0.3,
        z=2.2,
        material=Material(permittivity=5.0),
    )

    assert poly.z == 2.2
    np.testing.assert_allclose([vertex[2] for vertex in poly.vertices], 2.2)
    np.testing.assert_allclose(poly.get_bounding_box(), (0.0, 0.0, 2.2, 1.0, 1.0, 2.5))


def test_polygon_with_explicit_z_rasterizes_in_requested_3d_layer(monkeypatch):
    monkeypatch.setenv("BEAMZ_RASTER_CACHE", "0")
    monkeypatch.setenv("BEAMZ_RASTER_TIMING", "0")
    design = Design(
        width=1.0,
        height=1.0,
        depth=3.0,
        material=Material(permittivity=1.0),
    )
    design += Polygon(
        vertices=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        z=2.0,
        depth=0.5,
        material=Material(permittivity=5.0),
    )

    grid = RegularGrid3D(
        design,
        resolution_xy=0.5,
        resolution_z=0.5,
        aa_mode="legacy_grid",
        aa_samples=1,
    )

    np.testing.assert_allclose(grid.permittivity[:4], 1.0)
    np.testing.assert_allclose(grid.permittivity[4], 5.0)
    np.testing.assert_allclose(grid.permittivity[5], 1.0)


def test_sidewalled_rectangle_point_membership_varies_with_z():
    rect = Rectangle(
        position=(0.0, 0.0, 0.0),
        width=1.0,
        height=1.0,
        depth=1.0,
        material=Material(permittivity=12.0),
        sidewall_angle=10.0,
        width_to_z=0.0,
    )

    assert rect.point_in_polygon(0.98, 0.5, 0.05)
    assert not rect.point_in_polygon(0.98, 0.5, 0.95)


def test_sidewalled_rectangle_rasterizes_narrower_at_top(monkeypatch):
    monkeypatch.setenv("BEAMZ_RASTER_CACHE", "0")
    monkeypatch.setenv("BEAMZ_RASTER_TIMING", "0")

    design = Design(
        width=2.0,
        height=2.0,
        depth=1.2,
        material=Material(permittivity=1.0),
    )
    design += Rectangle(
        position=(0.5, 0.5, 0.1),
        width=1.0,
        height=1.0,
        depth=1.0,
        material=Material(permittivity=9.0),
        sidewall_angle=10.0,
        width_to_z=0.0,
    )
    grid = design.rasterize(
        resolution=0.1,
        force_recompute=True,
        aa_mode="legacy_grid",
        aa_samples=16,
    )

    filled_bottom = np.count_nonzero(grid.permittivity[1] > 2.0)
    filled_top = np.count_nonzero(grid.permittivity[-2] > 2.0)
    assert filled_top < filled_bottom


def test_aligned_3d_rectangle_uses_direct_region_fill(monkeypatch):
    monkeypatch.setenv("BEAMZ_RASTER_CACHE", "0")
    monkeypatch.setenv("BEAMZ_RASTER_TIMING", "0")
    monkeypatch.setenv("BEAMZ_RASTER_EXACT_3D", "1")

    def fail_masked_region(*args, **kwargs):
        raise AssertionError("aligned rectangles should not allocate masked regions")

    monkeypatch.setattr(MaterialGrids, "set_masked_region", fail_masked_region)
    monkeypatch.setattr(MaterialGrids, "blend_masked_region", fail_masked_region)

    design = Design(
        width=2.0,
        height=2.0,
        depth=2.0,
        material=Material(permittivity=1.0),
    )
    design += Rectangle(
        position=(0.5, 0.5, 0.5),
        width=1.0,
        height=1.0,
        depth=1.0,
        material=Material(permittivity=9.0),
    )

    grid = design.rasterize(
        resolution=0.5,
        force_recompute=True,
        aa_mode="legacy_grid",
        aa_samples=1,
    )

    expected = np.ones((4, 4, 4), dtype=np.float32)
    expected[1:3, 1:3, 1:3] = 9.0
    np.testing.assert_allclose(grid.permittivity, expected)


def test_exact_3d_coverage_samples_custom_material_at_voxel_centers(monkeypatch):
    monkeypatch.setenv("BEAMZ_RASTER_TIMING", "0")
    material = CustomMaterial(
        permittivity_func=lambda x, y, z: 2.0 + x + y + z,
        cache_key="xyz-gradient",
        max_permittivity=5.0,
    )
    design = Design(width=1.0, height=1.0, depth=1.0).with_structure(
        Rectangle(
            position=(0.0, 0.0, 0.0),
            width=1.0,
            height=1.0,
            depth=1.0,
            material=material,
        )
    )

    grid = RegularGrid3D(design, resolution_xy=0.5, resolution_z=0.5)

    z, y, x = np.meshgrid(*([np.array([0.25, 0.75])] * 3), indexing="ij")
    np.testing.assert_allclose(grid.permittivity, 2.0 + x + y + z)


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
