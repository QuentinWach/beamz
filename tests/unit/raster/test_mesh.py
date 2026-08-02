from __future__ import annotations

import numpy as np
import pytest

import beamz as bz
import beamz.design.raster as raster
from beamz.design import MaterialGrid
from beamz.design.raster.importers import from_mesh, from_mesh_arrays, repair_mesh


def tetrahedron(scale: float = 1.0):
    vertices = scale * np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
    )
    triangles = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.uint32)
    return vertices, triangles


def cube():
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    triangles = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.uint32,
    )
    return vertices, triangles


@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e6])
def test_inspection_and_compilation_share_scale_aware_validity(scale):
    vertices, triangles = tetrahedron(scale)
    report = raster.inspect_mesh(vertices, triangles)
    assert report.valid_for_rasterization

    scene = from_mesh_arrays(vertices, triangles, material=raster.Material(4))
    raster.compile_scene(scene)


def test_open_mesh_is_reported_and_rejected():
    vertices, triangles = tetrahedron()
    report = raster.inspect_mesh(vertices, triangles[:-1])
    assert not report.valid_for_rasterization
    assert report.boundary_edges > 0

    scene = from_mesh_arrays(vertices, triangles[:-1], material=raster.Material(4))
    with pytest.raises(ValueError, match="invalid mesh"):
        raster.compile_scene(scene)


def test_raw_mesh_rasterizes():
    vertices, triangles = tetrahedron()
    result = raster.rasterize(
        from_mesh_arrays(
            vertices,
            triangles,
            material=bz.Material(permittivity=4),
        ),
        raster.Grid.uniform((0, 0, 0), (1, 1, 1), (4, 4, 4)),
        options=raster.RasterOptions(quality="fast"),
    )
    assert result.tensors["epsilon"][0].max() == 4


@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
def test_closed_mesh_rasterization_is_scale_invariant(scale):
    vertices, triangles = cube()
    result = raster.rasterize(
        from_mesh_arrays(vertices * scale, triangles, material=raster.Material(4)),
        raster.Grid.uniform((0, 0, 0), (scale, scale, scale), (1, 1, 1)),
        options=raster.RasterOptions(quality="reference"),
    )

    assert result.tensors["epsilon"][0, 0, 0, 0] == 4


def test_stl_import_round_trip(tmp_path):
    meshio = pytest.importorskip("meshio")
    vertices, triangles = cube()
    path = tmp_path / "cube.stl"
    meshio.write(path, meshio.Mesh(vertices, [("triangle", triangles)]), binary=True)

    result = raster.rasterize(
        from_mesh(path, material=raster.Material(3)),
        raster.Grid.uniform((0, 0, 0), (1, 1, 1), (2, 2, 2)),
        options=raster.RasterOptions(quality="fast"),
    )
    assert result.tensors["epsilon"][0].min() == 3


def test_gmsh_physical_region_import(tmp_path):
    meshio = pytest.importorskip("meshio")
    vertices, tetra = tetrahedron()
    path = tmp_path / "tetra.msh"
    meshio.write(
        path,
        meshio.Mesh(
            vertices,
            [("tetra", np.array([[0, 1, 2, 3]], dtype=np.int32))],
            cell_data={"gmsh:physical": [np.array([7], dtype=np.int32)]},
            field_data={"core": np.array([7, 3], dtype=np.int32)},
        ),
        file_format="gmsh22",
    )
    scene = from_mesh(path, materials={"core": raster.Material(6)})
    assert len(scene.objects) == 1
    raster.compile_scene(scene)


def test_multi_region_gmsh_enters_simulation_without_losing_materials(tmp_path):
    meshio = pytest.importorskip("meshio")
    first_vertices, _ = tetrahedron(1e-6)
    second_vertices = first_vertices + np.array((2e-6, 0.0, 0.0))
    vertices = np.vstack((first_vertices, second_vertices))
    tetrahedra = np.array(((0, 1, 2, 3), (4, 5, 6, 7)), dtype=np.int32)
    path = tmp_path / "regions.msh"
    meshio.write(
        path,
        meshio.Mesh(
            vertices,
            [("tetra", tetrahedra)],
            cell_data={"gmsh:physical": [np.array((7, 8), dtype=np.int32)]},
            field_data={
                "core": np.array((7, 3), dtype=np.int32),
                "cladding": np.array((8, 3), dtype=np.int32),
            },
        ),
        file_format="gmsh22",
    )
    scene = from_mesh(
        path,
        materials={
            "core": raster.Material(12.0),
            "cladding": raster.Material(2.25),
        },
    )
    result = raster.rasterize(
        scene,
        raster.Grid.uniform((0, 0, 0), (3e-6, 1e-6, 1e-6), (6, 2, 2)),
        options=raster.RasterOptions(quality="reference"),
    )
    material_grid = MaterialGrid.from_raster_result(result, dimensions=3)
    simulation = bz.Simulation(material_grid=material_grid, run_time=4e-15)

    assert len(scene.objects) == 2
    assert set(object_.material_id for object_ in scene.objects) == {1, 2}
    assert np.max(material_grid.permittivity) > 6.0
    assert simulation.to_request().materials is material_grid
    simulation.compile()


def test_mesh_reference_quality_converges_to_tetrahedron_volume_fraction():
    vertices, triangles = tetrahedron()
    scene = from_mesh_arrays(vertices, triangles, material=raster.Material(4.0))
    grid = raster.Grid.uniform((0, 0, 0), (1, 1, 1), (1, 1, 1))
    values = []
    for quality in ("fast", "balanced", "reference"):
        result = raster.rasterize(
            scene,
            grid,
            options=raster.RasterOptions(quality=quality),
        )
        values.append(float(result.tensors["epsilon"][0, 0, 0, 0]))

    errors = [abs(value - 1.5) for value in values]
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1e-3


def test_mesh_triangle_order_does_not_change_raster_result():
    vertices, triangles = cube()
    grid = raster.Grid.uniform((0, 0, 0), (1, 1, 1), (3, 3, 3))
    first = raster.rasterize(
        from_mesh_arrays(vertices, triangles, material=raster.Material(4.0)),
        grid,
    )
    second = raster.rasterize(
        from_mesh_arrays(vertices, triangles[::-1], material=raster.Material(4.0)),
        grid,
    )

    for name in first.tensors:
        np.testing.assert_array_equal(first.tensors[name], second.tensors[name])
    for name in first.yee_tensors:
        np.testing.assert_array_equal(first.yee_tensors[name], second.yee_tensors[name])


def test_repair_returns_an_auditable_report():
    pytest.importorskip("trimesh")
    vertices, triangles = tetrahedron()
    result = repair_mesh(vertices, np.vstack((triangles, triangles[0])))

    assert result.report.removed_duplicate_triangles == 1
    assert result.report.valid_for_rasterization
    raster.compile_scene(
        from_mesh_arrays(
            result.mesh.vertices,
            result.mesh.triangles,
            material=raster.Material(2),
        )
    )
