import numpy as np

from beamz import Circle, Design, Material


def _build_circle_design():
    design = Design(width=10.0, height=10.0, material=Material(permittivity=1.0))
    design += Circle(
        position=(5.0, 5.0),
        radius=2.65,
        material=Material(permittivity=12.0),
    )
    return design


def test_default_antialias_matches_stratified_jitter_64(monkeypatch):
    monkeypatch.setenv("BEAMZ_RASTER_CACHE", "0")
    monkeypatch.setenv("BEAMZ_RASTER_TIMING", "0")

    design = _build_circle_design()
    default_grid = design.rasterize(resolution=0.5, force_recompute=True)
    default_explicit_grid = design.rasterize(
        resolution=0.5,
        force_recompute=True,
        aa_mode="stratified_jitter",
        aa_samples=64,
        aa_seed=0,
    )
    np.testing.assert_allclose(
        default_grid.permittivity, default_explicit_grid.permittivity
    )


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
