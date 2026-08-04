from __future__ import annotations

import numpy as np

from beamz import RectilinearGrid
from beamz.lattice import (
    canonical_component_2d,
    component_coordinates_2d_um,
    component_coordinates_3d_um,
    component_coordinates_rectilinear,
    component_material_at,
    component_shape_2d,
    component_shape_3d,
    coordinates_in_public_frame,
    material_for_component,
    public_component_2d,
)
from tests.utils import compiled_grid


def test_component_shape_3d_uses_one_complete_yee_support():
    grid_shape = (24, 24, 24)
    assert component_shape_3d("Ex", grid_shape) == (25, 25, 24)
    assert component_shape_3d("Ey", grid_shape) == (25, 24, 25)
    assert component_shape_3d("Ez", grid_shape) == (24, 25, 25)
    assert component_shape_3d("Hx", grid_shape) == (24, 24, 25)
    assert component_shape_3d("Hy", grid_shape) == (24, 25, 24)
    assert component_shape_3d("Hz", grid_shape) == (25, 24, 24)


def test_component_coordinates_3d_follow_standard_yee_offsets():
    grid_shape = (24, 24, 24)
    dx_um = 0.125

    ex = component_coordinates_3d_um("Ex", grid_shape, dx_um)
    ey = component_coordinates_3d_um("Ey", grid_shape, dx_um)
    ez = component_coordinates_3d_um("Ez", grid_shape, dx_um)
    hx = component_coordinates_3d_um("Hx", grid_shape, dx_um)

    np.testing.assert_allclose(ex["x"][0], 0.0625)
    np.testing.assert_allclose(ex["x"][-1], 2.9375)
    np.testing.assert_allclose(ey["y"][0], 0.0625)
    np.testing.assert_allclose(ey["y"][-1], 2.9375)
    np.testing.assert_allclose(ez["z"][0], 0.0625)
    np.testing.assert_allclose(ez["z"][-1], 2.9375)

    np.testing.assert_allclose(ex["z"][0], 0.0)
    np.testing.assert_allclose(ex["y"][0], 0.0)
    np.testing.assert_allclose(ey["x"][0], 0.0)
    np.testing.assert_allclose(ey["z"][0], 0.0)
    np.testing.assert_allclose(ez["x"][0], 0.0)
    np.testing.assert_allclose(ez["y"][0], 0.0)

    np.testing.assert_allclose(hx["x"][0], 0.0)
    np.testing.assert_allclose(hx["y"][0], 0.0625)
    np.testing.assert_allclose(hx["z"][0], 0.0625)

    # Integer-aligned axes retain both domain walls in the complete representation.
    np.testing.assert_allclose(ex["y"][-1], 3.0)
    np.testing.assert_allclose(ey["x"][-1], 3.0)
    np.testing.assert_allclose(ez["x"][-1], 3.0)
    np.testing.assert_allclose(hx["x"][-1], 3.0)


def test_rectilinear_component_coordinates_use_exact_edges_and_centers():
    grid = RectilinearGrid(
        np.asarray([0.0, 1.0, 3.0]),
        np.asarray([0.0, 2.0, 5.0]),
        np.asarray([0.0, 4.0, 6.0]),
    )

    ex = component_coordinates_rectilinear("Ex", grid)
    hx = component_coordinates_rectilinear("Hx", grid)
    hy_2d = component_coordinates_rectilinear("Hy", grid, plane="xy")

    np.testing.assert_allclose(ex["x"], [0.5, 2.0])
    np.testing.assert_allclose(ex["y"], grid.y_edges)
    np.testing.assert_allclose(hx["z"], [2.0, 5.0])
    np.testing.assert_allclose(hx["y"], [1.0, 3.5])
    np.testing.assert_allclose(hy_2d["x"], [0.5, 2.0])
    np.testing.assert_allclose(hy_2d["y"], grid.y_edges)


def test_component_coordinates_translate_from_solver_local_to_public_frame():
    coordinates = {"x": np.asarray([0.0, 0.2, 1.0]), "y": np.asarray([0.0, 1.0])}

    public = coordinates_in_public_frame(coordinates, (-2.0, -3.0, -4.0))

    np.testing.assert_allclose(public["x"], [2.0, 2.2, 3.0])
    np.testing.assert_allclose(public["y"], [3.0, 4.0])


def test_component_coordinates_2d_follow_standard_xy_offsets():
    grid_shape = (24, 24)
    dx_um = 0.125

    assert component_shape_2d("Ez", grid_shape, "xy") == (25, 25)

    ez = component_coordinates_2d_um("Ez", grid_shape, dx_um, "xy")

    np.testing.assert_allclose(ez["y"][0], 0.0)
    np.testing.assert_allclose(ez["x"][0], 0.0)


def test_canonical_2d_coordinates_follow_physical_tmz_offsets():
    grid_shape = (24, 24)
    dx_um = 0.125

    assert component_shape_2d("Ez", grid_shape, "xy") == (25, 25)
    assert component_shape_2d("Hx", grid_shape, "xy") == (24, 25)
    assert component_shape_2d("Hy", grid_shape, "xy") == (25, 24)

    ez = component_coordinates_2d_um("Ez", grid_shape, dx_um, "xy")
    hx = component_coordinates_2d_um("Hx", grid_shape, dx_um, "xy")
    hy = component_coordinates_2d_um("Hy", grid_shape, dx_um, "xy")

    np.testing.assert_allclose(ez["y"][0], 0.0)
    np.testing.assert_allclose(ez["x"][0], 0.0)
    np.testing.assert_allclose(hx["y"][0], 0.0625)
    np.testing.assert_allclose(hx["x"][0], 0.0)
    np.testing.assert_allclose(hy["y"][0], 0.0)
    np.testing.assert_allclose(hy["x"][0], 0.0625)


def test_public_2d_planes_map_to_one_canonical_tmxy_support():
    grid_shape = (5, 7)
    expected = {
        "xy": {"Ez": "Ez", "Hx": "Hx", "Hy": "Hy"},
        "yz": {"Ex": "Ez", "Hy": "Hx", "Hz": "Hy"},
        "xz": {"Ey": "Ez", "Hx": "Hx", "Hz": "Hy"},
    }
    canonical_shapes = {"Ez": (6, 8), "Hx": (5, 8), "Hy": (6, 7)}
    for plane, mapping in expected.items():
        for public, canonical in mapping.items():
            assert canonical_component_2d(public, plane) == canonical
            assert (
                component_shape_2d(public, grid_shape, plane)
                == canonical_shapes[canonical]
            )
            roundtrip, sign = public_component_2d(canonical, plane)
            assert roundtrip == public
            assert sign == (-1.0 if (plane, public) == ("xz", "Ey") else 1.0)


def test_public_2d_planes_map_to_one_canonical_texy_support():
    grid_shape = (5, 7)
    expected = {
        "xy": {"Ex": "Ex", "Ey": "Ey", "Hz": "Hz"},
        "yz": {"Ey": "Ex", "Ez": "Ey", "Hx": "Hz"},
        "xz": {"Ex": "Ex", "Ez": "Ey", "Hy": "Hz"},
    }
    canonical_shapes = {"Ex": (6, 7), "Ey": (5, 8), "Hz": (5, 7)}
    for plane, mapping in expected.items():
        for public, canonical in mapping.items():
            assert canonical_component_2d(public, plane, "te") == canonical
            assert (
                component_shape_2d(public, grid_shape, plane, "te")
                == canonical_shapes[canonical]
            )
            roundtrip, sign = public_component_2d(canonical, plane, "te")
            assert roundtrip == public
            assert sign == (-1.0 if (plane, public) == ("xz", "Hy") else 1.0)


def test_xy_generic_h_coordinates_follow_native_tmz_offsets():
    assert component_shape_2d("Hx", (24, 24), "xy") == (24, 25)
    hy = component_coordinates_2d_um("Hy", (24, 24), 0.125, "xy")
    np.testing.assert_allclose(hy["y"][0], 0.0)
    np.testing.assert_allclose(hy["x"][0], 0.0625)


def test_fields_expose_component_material_arrays_for_3d():
    grid = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5) + 1.0
    fields = compiled_grid(
        permittivity=grid,
        conductivity=np.zeros_like(grid),
        permeability=2.0 * grid,
        resolution=0.1,
    )

    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        material = material_for_component(fields.materials, component)
        assert material.shape == getattr(fields, component).shape

    assert material_for_component(fields.materials, "Ex") is fields.eps_x
    assert material_for_component(fields.materials, "Ey") is fields.eps_y
    assert material_for_component(fields.materials, "Ez") is fields.eps_z
    assert material_for_component(fields.materials, "Hx") is fields.mu_hx
    assert material_for_component(fields.materials, "Hy") is fields.mu_hy
    assert material_for_component(fields.materials, "Hz") is fields.mu_hz


def test_fields_expose_component_material_arrays_for_xy_2d():
    grid = np.arange(4 * 5, dtype=np.float32).reshape(4, 5) + 1.0
    fields = compiled_grid(
        permittivity=grid,
        conductivity=np.zeros_like(grid),
        permeability=2.0 * grid,
        resolution=0.1,
        plane_2d="xy",
    )

    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        material = material_for_component(fields.materials, component)
        assert material.shape == getattr(fields, component).shape

    np.testing.assert_array_equal(
        material_for_component(fields.materials, "Ez"), fields.eps_z
    )
    np.testing.assert_array_equal(
        material_for_component(fields.materials, "Hx"), fields.mu_hx
    )
    np.testing.assert_array_equal(
        material_for_component(fields.materials, "Hy"), fields.mu_hy
    )


def test_component_material_sampling_uses_fields_component_materials():
    grid = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5) + 1.0
    fields = compiled_grid(
        permittivity=grid,
        conductivity=np.zeros_like(grid),
        permeability=2.0 * grid,
        resolution=0.1,
    )
    e_index = (slice(None), slice(None), slice(1, 3))
    h_index = (slice(None), slice(1, 3), slice(None))

    np.testing.assert_array_equal(
        component_material_at(fields, "Ex", e_index), fields.eps_x[e_index]
    )
    np.testing.assert_array_equal(
        component_material_at(fields, "Hx", h_index), fields.mu_hx[h_index]
    )
