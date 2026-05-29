from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from beamz.simulation.boundaries import (
    PEC,
    create_metallic_boundary_masks,
    full_pec_curl_e_to_h_2d_xy,
    full_pec_curl_h_to_e_2d_xy,
    full_pec_update_e_from_h_3d,
    full_pec_update_h_from_e_3d,
    has_full_pec_2d_xy,
    initialize_full_pec_3d_state,
    initialize_tm_2d_xy_state,
    normalize_boundaries,
    pec_curl_e_to_h_3d,
    pec_curl_h_to_e_3d,
)
from beamz.simulation.fields import Fields
from beamz.simulation.yee import (
    sample_voxel_grid_at_component_2d,
    sample_voxel_grid_at_tm_xy_full_component_2d,
)


def _uniform_3d_fields(n: int = 4) -> Fields:
    permittivity = np.ones((n, n, n), dtype=np.float32)
    conductivity = np.zeros((n, n, n), dtype=np.float32)
    permeability = np.ones((n, n, n), dtype=np.float32)
    fields = Fields(
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=permeability,
        resolution=1.0,
    )
    fields.boundaries = normalize_boundaries([], is_3d=True)
    fields.set_metallic_masks(
        create_metallic_boundary_masks(fields, boundaries=[], is_3d=True)
    )
    return fields


def test_3d_pec_update_keeps_constrained_ex_planes_zero():
    fields = _uniform_3d_fields()
    dt = 1e-3

    # Create nonzero curl contributions that touch Ex boundary planes via both
    # dHz/dy and dHy/dz terms.
    hz = np.zeros(fields.Hz.shape, dtype=np.float32)
    hz[:, 1:, :] = 1.0
    hy = np.zeros(fields.Hy.shape, dtype=np.float32)
    hy[1:, :, :] = 1.0

    fields.Hz = hz
    fields.Hy = hy

    fields.update_e(dt)
    ex = np.asarray(fields.Ex)

    np.testing.assert_allclose(ex[0, :, :], 0.0)
    np.testing.assert_allclose(ex[:, 0, :], 0.0)
    assert np.max(np.abs(ex[-1, 1:, :])) > 0.0
    assert np.max(np.abs(ex[1:, -1, :])) > 0.0
    assert np.max(np.abs(ex[1:-1, 1:-1, :])) > 0.0


def test_3d_default_pec_masks_zero_expected_stored_planes():
    fields = _uniform_3d_fields()
    masks = create_metallic_boundary_masks(fields, boundaries=[], is_3d=True)

    for name, mask in masks.items():
        assert mask.shape == getattr(fields, name).shape

    assert np.all(np.asarray(masks["Ex"])[:, 0, :])
    assert np.all(np.asarray(masks["Ex"])[0, :, :])
    assert not np.any(np.asarray(masks["Ex"])[1:, -1, :])
    assert not np.any(np.asarray(masks["Ex"])[-1, 1:, :])

    assert np.all(np.asarray(masks["Ey"])[:, :, 0])
    assert np.all(np.asarray(masks["Ey"])[0, :, :])
    assert not np.any(np.asarray(masks["Ey"])[1:, :, -1])
    assert not np.any(np.asarray(masks["Ey"])[-1, 1:, 1:])

    assert np.all(np.asarray(masks["Ez"])[:, 0, :])
    assert np.all(np.asarray(masks["Ez"])[:, :, 0])
    assert not np.any(np.asarray(masks["Ez"])[1:, -1, 1:])
    assert not np.any(np.asarray(masks["Ez"])[1:, 1:, -1])


def test_empty_boundary_list_resolves_to_explicit_pec():
    resolved = normalize_boundaries([], is_3d=True)
    assert len(resolved) == 1
    assert isinstance(resolved[0], PEC)
    assert resolved[0]._get_edges_for_dimensionality(True) == [
        "left",
        "right",
        "top",
        "bottom",
        "front",
        "back",
    ]


def test_pec_curl_h_to_e_3d_keeps_boundary_planes_zero():
    fields = _uniform_3d_fields()

    hz = np.zeros(fields.Hz.shape, dtype=np.float32)
    hz[:, 1:, :] = 1.0
    hy = np.zeros(fields.Hy.shape, dtype=np.float32)
    hy[1:, :, :] = 0.25
    hx = np.zeros(fields.Hx.shape, dtype=np.float32)

    curl_hx, curl_hy, curl_hz = pec_curl_h_to_e_3d(
        hx,
        hy,
        hz,
        resolution=1.0,
        ex_shape=fields.Ex.shape,
        ey_shape=fields.Ey.shape,
        ez_shape=fields.Ez.shape,
    )

    assert np.max(np.abs(np.asarray(curl_hx)[:, 0, :])) > 0.0
    assert np.max(np.abs(np.asarray(curl_hx)[:, -1, :])) > 0.0
    assert np.max(np.abs(np.asarray(curl_hx)[1:-1, 1:-1, :])) > 0.0


def test_pec_curl_e_to_h_3d_matches_h_shapes():
    fields = _uniform_3d_fields()

    curl_ex, curl_ey, curl_ez = pec_curl_e_to_h_3d(
        fields.Ex,
        fields.Ey,
        fields.Ez,
        resolution=1.0,
        hx_shape=fields.Hx.shape,
        hy_shape=fields.Hy.shape,
        hz_shape=fields.Hz.shape,
    )

    assert curl_ex.shape == fields.Hx.shape
    assert curl_ey.shape == fields.Hy.shape
    assert curl_ez.shape == fields.Hz.shape


def test_initialize_full_pec_3d_state_adds_missing_high_planes():
    fields = _uniform_3d_fields()
    state = initialize_full_pec_3d_state(fields)

    assert state.Ex.shape == tuple(v + 1 for v in fields.Ex.shape)
    assert state.Ey.shape == tuple(v + 1 for v in fields.Ey.shape)
    assert state.Ez.shape == tuple(v + 1 for v in fields.Ez.shape)
    assert state.Hx.shape == tuple(v + 1 for v in fields.Hx.shape)
    assert state.Hy.shape == tuple(v + 1 for v in fields.Hy.shape)
    assert state.Hz.shape == tuple(v + 1 for v in fields.Hz.shape)

    np.testing.assert_allclose(np.asarray(state.Ex)[-1, :, :], 0.0)
    np.testing.assert_allclose(np.asarray(state.Ex)[:, -1, :], 0.0)
    np.testing.assert_allclose(np.asarray(state.Ey)[-1, :, :], 0.0)
    np.testing.assert_allclose(np.asarray(state.Ey)[:, :, -1], 0.0)
    np.testing.assert_allclose(np.asarray(state.Ez)[:, -1, :], 0.0)
    np.testing.assert_allclose(np.asarray(state.Ez)[:, :, -1], 0.0)


def test_full_pec_material_regions_sample_centered_e_positions():
    permittivity = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    conductivity = np.zeros_like(permittivity)
    permeability = np.ones_like(permittivity)
    fields = Fields(
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=permeability,
        resolution=1.0,
    )
    state = initialize_full_pec_3d_state(fields)

    expected_ex = 0.5 * (
        permittivity[np.ix_([1, 2, 3], [1, 2, 3], [0, 1, 2, 3])]
        + permittivity[np.ix_([1, 2, 3], [1, 2, 3], [1, 2, 3, 3])]
    )
    expected_ey = 0.5 * (
        permittivity[np.ix_([1, 2, 3], [0, 1, 2, 3], [1, 2, 3])]
        + permittivity[np.ix_([1, 2, 3], [1, 2, 3, 3], [1, 2, 3])]
    )
    expected_ez = 0.5 * (
        permittivity[np.ix_([0, 1, 2, 3], [1, 2, 3], [1, 2, 3])]
        + permittivity[np.ix_([1, 2, 3, 3], [1, 2, 3], [1, 2, 3])]
    )

    np.testing.assert_array_equal(np.asarray(state.eps_x_region), expected_ex)
    np.testing.assert_array_equal(np.asarray(state.eps_y_region), expected_ey)
    np.testing.assert_array_equal(np.asarray(state.eps_z_region), expected_ez)


def test_xy_2d_component_sampling_matches_expected_owned_voxels():
    grid = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)

    ex = sample_voxel_grid_at_component_2d(grid, "Ex", "xy")
    ey = sample_voxel_grid_at_component_2d(grid, "Ey", "xy")
    ez = sample_voxel_grid_at_component_2d(grid, "Ez", "xy")
    hz = sample_voxel_grid_at_component_2d(grid, "Hz", "xy")

    np.testing.assert_array_equal(np.asarray(ex), grid[:, :-1])
    np.testing.assert_array_equal(np.asarray(ey), grid[:-1, :])
    np.testing.assert_array_equal(
        np.asarray(ez), np.pad(grid, ((0, 1), (0, 1)), mode="edge")
    )
    np.testing.assert_array_equal(np.asarray(hz), grid[:-1, :-1])


def test_xy_2d_full_pec_state_adds_missing_h_boundary_edges():
    permittivity = np.ones((4, 5), dtype=np.float32)
    conductivity = np.zeros_like(permittivity)
    permeability = np.ones_like(permittivity)
    fields = Fields(
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=permeability,
        resolution=1.0,
        plane_2d="xy",
    )
    fields.boundaries = normalize_boundaries([], is_3d=False)

    state = initialize_tm_2d_xy_state(fields)
    assert state.Ez.shape == fields.Ez.shape
    assert state.Hx.shape == fields.Hx.shape
    assert state.Hy.shape == fields.Hy.shape
    assert has_full_pec_2d_xy(fields.boundaries, "xy")
    np.testing.assert_allclose(np.asarray(state.Hx)[:, 0], 0.0)
    np.testing.assert_allclose(np.asarray(state.Hx)[:, -1], 0.0)
    np.testing.assert_allclose(np.asarray(state.Hy)[0, :], 0.0)
    np.testing.assert_allclose(np.asarray(state.Hy)[-1, :], 0.0)


def test_2d_tm_pec_update_keeps_constrained_h_edges_zero():
    permittivity = np.ones((4, 5), dtype=np.float32)
    conductivity = np.zeros_like(permittivity)
    permeability = np.ones_like(permittivity)
    fields = Fields(
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=permeability,
        resolution=1.0,
        plane_2d="xy",
    )
    fields.boundaries = normalize_boundaries([], is_3d=False)

    fields.Ez = jnp.zeros(fields.Ez.shape, dtype=jnp.float32)
    fields.Ez = fields.Ez.at[1:4, 1:5].set(1.0)

    fields.update_h(dt=1e-3)
    state = fields.ensure_tm_xy_state()

    np.testing.assert_allclose(np.asarray(state.Hx)[:, 0], 0.0)
    np.testing.assert_allclose(np.asarray(state.Hx)[:, -1], 0.0)
    np.testing.assert_allclose(np.asarray(state.Hy)[0, :], 0.0)
    np.testing.assert_allclose(np.asarray(state.Hy)[-1, :], 0.0)
    assert np.max(np.abs(np.asarray(state.Hx)[:, 1:-1])) > 0.0
    assert np.max(np.abs(np.asarray(state.Hy)[1:-1, :])) > 0.0


def test_xy_2d_full_state_sampling_uses_physical_tmz_h_locations():
    grid = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)

    ez = sample_voxel_grid_at_tm_xy_full_component_2d(grid, "Ez")
    hx = sample_voxel_grid_at_tm_xy_full_component_2d(grid, "Hx")
    hy = sample_voxel_grid_at_tm_xy_full_component_2d(grid, "Hy")

    np.testing.assert_array_equal(
        np.asarray(ez), grid[[0, 1, 2, 3, 3]][:, [0, 1, 2, 3, 4, 4]]
    )
    np.testing.assert_array_equal(np.asarray(hx), grid[:, [0, 1, 2, 3, 4, 4]])
    np.testing.assert_array_equal(np.asarray(hy), grid[[0, 1, 2, 3, 3], :])


def test_xy_2d_full_pec_curls_match_full_shapes():
    permittivity = np.ones((4, 5), dtype=np.float32)
    conductivity = np.zeros_like(permittivity)
    permeability = np.ones_like(permittivity)
    fields = Fields(
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=permeability,
        resolution=1.0,
        plane_2d="xy",
    )
    state = initialize_tm_2d_xy_state(fields)

    curl_hx, curl_hy = full_pec_curl_e_to_h_2d_xy(
        state.Ez,
        1.0,
        state.Hx.shape,
        state.Hy.shape,
    )
    assert curl_hx.shape == state.Hx.shape
    assert curl_hy.shape == state.Hy.shape

    curl_hz = full_pec_curl_h_to_e_2d_xy(
        state.Hx,
        state.Hy,
        1.0,
        state.Ez.shape,
    )
    assert curl_hz.shape == state.Ez.shape


def test_full_pec_updates_match_full_shapes():
    fields = _uniform_3d_fields()
    state = initialize_full_pec_3d_state(fields)

    hx, hy, hz = full_pec_update_h_from_e_3d(
        state.Ex,
        state.Ey,
        state.Ez,
        state.Hx,
        state.Hy,
        state.Hz,
        1.0,
        h_decay_x=jnp.ones_like(state.Hx),
        h_source_x=jnp.ones_like(state.Hx),
        h_decay_y=jnp.ones_like(state.Hy),
        h_source_y=jnp.ones_like(state.Hy),
        h_decay_z=jnp.ones_like(state.Hz),
        h_source_z=jnp.ones_like(state.Hz),
        hx_mask=state.masks["Hx"],
        hy_mask=state.masks["Hy"],
        hz_mask=state.masks["Hz"],
    )
    assert hx.shape == state.Hx.shape
    assert hy.shape == state.Hy.shape
    assert hz.shape == state.Hz.shape

    ex, ey, ez = full_pec_update_e_from_h_3d(
        state.Hx,
        state.Hy,
        state.Hz,
        state.Ex,
        state.Ey,
        state.Ez,
        1.0,
        e_decay_x=jnp.ones_like(state.Ex),
        e_source_x=jnp.ones_like(state.Ex),
        e_decay_y=jnp.ones_like(state.Ey),
        e_source_y=jnp.ones_like(state.Ey),
        e_decay_z=jnp.ones_like(state.Ez),
        e_source_z=jnp.ones_like(state.Ez),
        ex_mask=state.masks["Ex"],
        ey_mask=state.masks["Ey"],
        ez_mask=state.masks["Ez"],
    )
    assert ex.shape == state.Ex.shape
    assert ey.shape == state.Ey.shape
    assert ez.shape == state.Ez.shape
