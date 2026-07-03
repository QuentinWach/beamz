from types import SimpleNamespace

import numpy as np

from beamz.design.core import _build_grid_from_cached_arrays, _save_grid_to_cache
from beamz.design.meshing import MaterialGrids
from beamz.simulation.boundaries import initialize_tm_2d_xy_state
from beamz.simulation.fields import Fields


def test_material_grids_keep_default_thermal_channels_scalar():
    shape = (2, 3, 4)
    grids = MaterialGrids(shape)
    default_props = (12.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 300.0)

    grids.fill_all(default_props)
    grids.set_at((0, 0, 0), default_props)
    grids.set_region((slice(None), slice(None), slice(None)), default_props)

    assert grids.permittivity.shape == shape
    assert grids.permeability.shape == shape
    assert np.asarray(grids.conductivity).shape == ()
    assert np.asarray(grids.k).shape == ()
    assert np.asarray(grids.rho).shape == ()
    assert np.asarray(grids.cp).shape == ()
    assert np.asarray(grids.dn_dT).shape == ()
    assert np.asarray(grids.T0).shape == ()


def test_material_grids_materialize_nondefault_thermal_channel():
    shape = (2, 3, 4)
    grids = MaterialGrids(shape)
    thermal_props = (12.0, 1.0, 0.0, 2.5, 0.0, 0.0, 0.0, 300.0)

    grids.set_at((1, 2, 3), thermal_props)

    assert grids.k.shape == shape
    assert grids.k[1, 2, 3] == 2.5
    assert np.asarray(grids.rho).shape == ()


def test_raster_cache_stores_compact_default_channels(tmp_path):
    shape = (1, 2, 3)
    grid = SimpleNamespace(
        permittivity=np.ones(shape),
        permeability=np.ones(shape),
        conductivity=np.asarray(0.0),
        k=0.0,
        rho=0.0,
        cp=0.0,
        dn_dT=0.0,
        T0=300.0,
    )
    cache_path = tmp_path / "grid.npz"

    _save_grid_to_cache(grid, cache_path)

    with np.load(cache_path) as arrays:
        assert arrays["permittivity"].shape == shape
        assert arrays["conductivity"].shape == ()
        assert arrays["k"].shape == ()
        assert arrays["rho"].shape == ()
        assert arrays["cp"].shape == ()
        assert arrays["dn_dT"].shape == ()
        assert arrays["T0"].shape == ()


def test_raster_cache_load_preserves_compact_default_channels(tmp_path):
    shape = (1, 2, 3)
    cache_path = tmp_path / "grid.npz"
    np.savez_compressed(
        cache_path,
        permittivity=np.ones(shape),
        permeability=np.ones(shape),
        conductivity=np.asarray(0.0),
        k=np.asarray(0.0),
        rho=np.asarray(0.0),
        cp=np.asarray(0.0),
        dn_dT=np.asarray(0.0),
        T0=np.asarray(300.0),
    )

    with np.load(cache_path) as arrays:
        grid = _build_grid_from_cached_arrays(
            design_obj=SimpleNamespace(width=3.0, height=2.0, depth=1.0),
            resolution=1.0,
            resolution_z=1.0,
            grid_kind="3d",
            arrays=arrays,
        )

    assert grid.shape == shape
    assert grid.permittivity.shape == shape
    assert np.asarray(grid.conductivity).shape == ()
    assert np.asarray(grid.k).shape == ()
    assert np.asarray(grid.T0).shape == ()


def test_tm_xy_state_accepts_scalar_zero_conductivity():
    shape = (3, 4)
    fields = Fields(
        permittivity=np.ones(shape, dtype=np.float32),
        conductivity=np.asarray(0.0, dtype=np.float32),
        permeability=np.ones(shape, dtype=np.float32),
        resolution=0.1,
        plane_2d="xy",
    )

    state = initialize_tm_2d_xy_state(fields)

    assert np.asarray(fields.conductivity).shape == ()
    assert np.asarray(fields.total_conductivity).shape == ()
    assert np.asarray(state.sig_z_region).shape == ()
    assert np.asarray(state.sigma_m_hx).shape == ()
    assert np.asarray(state.sigma_m_hy).shape == ()
