from types import SimpleNamespace

import numpy as np

from beamz.design.core import _build_grid_from_cached_arrays, _save_grid_to_cache
from beamz.design.meshing import MaterialGrids
from beamz.simulation.boundaries import initialize_tm_2d_xy_state
from beamz.simulation.fields import Fields


def test_material_grids_keep_default_conductivity_scalar():
    shape = (2, 3, 4)
    grids = MaterialGrids(shape)
    default_props = (12.0, 1.0, 0.0)

    grids.fill_all(default_props)
    grids.set_at((0, 0, 0), default_props)
    grids.set_region((slice(None), slice(None), slice(None)), default_props)

    assert grids.permittivity.shape == shape
    assert grids.permittivity.dtype == np.float32
    assert np.asarray(grids.permeability).shape == ()
    assert np.asarray(grids.permeability).dtype == np.float32
    assert np.asarray(grids.conductivity).shape == ()
    assert not hasattr(grids, "k")
    assert not hasattr(grids, "rho")
    assert not hasattr(grids, "cp")
    assert not hasattr(grids, "dn_dT")
    assert not hasattr(grids, "T0")


def test_material_grids_materialize_nondefault_conductivity_channel():
    shape = (2, 3, 4)
    grids = MaterialGrids(shape)
    conductive_props = (12.0, 1.0, 2.5)

    grids.set_at((1, 2, 3), conductive_props)

    assert grids.conductivity.shape == shape
    assert grids.conductivity.dtype == np.float32
    assert grids.conductivity[1, 2, 3] == 2.5


def test_material_grids_materialize_nondefault_permeability_channel():
    shape = (2, 3, 4)
    grids = MaterialGrids(shape)
    magnetic_props = (12.0, 1.5, 0.0)

    grids.set_at((1, 2, 3), magnetic_props)

    assert grids.permeability.shape == shape
    assert grids.permeability.dtype == np.float32
    assert grids.permeability[1, 2, 3] == 1.5


def test_raster_cache_stores_compact_default_channels(tmp_path):
    shape = (1, 2, 3)
    grid = SimpleNamespace(
        permittivity=np.ones(shape),
        permeability=np.asarray(1.0, dtype=np.float32),
        conductivity=np.asarray(0.0),
    )
    cache_path = tmp_path / "grid.npz"

    _save_grid_to_cache(grid, cache_path)

    with np.load(cache_path) as arrays:
        assert arrays["permittivity"].shape == shape
        assert arrays["permeability"].shape == ()
        assert arrays["conductivity"].shape == ()
        assert set(arrays.files) == {"permittivity", "permeability", "conductivity"}


def test_raster_cache_load_preserves_compact_default_channels(tmp_path):
    shape = (1, 2, 3)
    cache_path = tmp_path / "grid.npz"
    np.savez_compressed(
        cache_path,
        permittivity=np.ones(shape),
        permeability=np.asarray(1.0, dtype=np.float32),
        conductivity=np.asarray(0.0),
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
    assert np.asarray(grid.permeability).shape == ()
    assert np.asarray(grid.conductivity).shape == ()
    assert not hasattr(grid, "k")
    assert not hasattr(grid, "rho")
    assert not hasattr(grid, "cp")
    assert not hasattr(grid, "dn_dT")
    assert not hasattr(grid, "T0")


def test_tm_xy_state_accepts_scalar_zero_conductivity():
    shape = (3, 4)
    fields = Fields(
        permittivity=np.ones(shape, dtype=np.float32),
        conductivity=np.asarray(0.0, dtype=np.float32),
        permeability=np.asarray(1.0, dtype=np.float32),
        resolution=0.1,
        plane_2d="xy",
    )

    state = initialize_tm_2d_xy_state(fields)

    assert np.asarray(fields.conductivity).shape == ()
    assert np.asarray(fields.permeability).shape == ()
    assert np.asarray(fields.mu_hx).shape == ()
    assert np.asarray(fields.mu_tm_hx).shape == ()
    assert np.asarray(fields.total_conductivity).shape == ()
    assert np.asarray(fields.sigma_m_hx).shape == ()
    assert np.asarray(fields.sigma_m_hy).shape == ()
    assert np.asarray(fields.sigma_m_hz).shape == ()
    assert np.asarray(state.sig_z_region).shape == ()
    assert np.asarray(state.sigma_m_hx).shape == ()
    assert np.asarray(state.sigma_m_hy).shape == ()
