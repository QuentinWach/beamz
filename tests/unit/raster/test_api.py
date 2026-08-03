from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import beamz
import beamz.design.raster as raster


def box_scene(epsilon: float = 4.0) -> raster.Scene:
    return raster.Scene(
        (raster.Material(), raster.Material(epsilon, 2.0, 3.0)),
        (raster.Object(raster.Box((0.25, 0.0, 0.0), (0.75, 1.0, 1.0)), 1),),
    )


def test_raster_and_design_material_are_one_public_type():
    assert raster.Material is beamz.Material
    material = beamz.Material(epsilon_r=(2.0, 3.0, 4.0))
    assert material.permittivity == (2.0, 3.0, 4.0)
    assert material.epsilon_r[:3] == (2.0, 3.0, 4.0)


def test_uniform_result_uses_solver_facing_yee_shapes():
    result = raster.rasterize(
        box_scene(), raster.Grid.uniform((0, 0, 0), (1, 1, 1), (2, 3, 4))
    )

    assert result.tensors["epsilon"].shape == (1, 4, 3, 2)
    assert result.yee_tensors["epsilon_ex"].shape == (1, 5, 4, 2)
    assert result.yee_tensors["epsilon_ey"].shape == (1, 5, 3, 3)
    assert result.yee_tensors["epsilon_ez"].shape == (1, 4, 4, 3)
    assert result.yee_tensors["mu_hx"].shape == (1, 4, 3, 3)
    assert result.yee_tensors["mu_hy"].shape == (1, 4, 4, 2)
    assert result.yee_tensors["mu_hz"].shape == (1, 5, 3, 2)


def test_scene_has_design_like_rasterize_convenience():
    scene = raster.Scene((raster.Material(epsilon_r=2.0),))
    grid = raster.Grid.uniform((0, 0, 0), (1, 1, 1), (2, 2, 2))

    direct = scene.rasterize(grid)
    functional = raster.rasterize(scene, grid)

    np.testing.assert_array_equal(
        direct.tensors["epsilon"],
        functional.tensors["epsilon"],
    )


def test_nonuniform_grid_is_supported_and_immutable():
    grid = raster.Grid(
        np.array([0.0, 0.2, 1.0]),
        np.array([0.0, 0.7, 1.0]),
        np.array([0.0, 1.0]),
    )
    result = raster.rasterize(box_scene(), grid)

    assert not grid.is_uniform
    assert not result.is_uniform
    assert result.tensors["epsilon"].shape == (1, 1, 2, 2)
    with pytest.raises(ValueError):
        result.tensors["epsilon"][0, 0, 0, 0] = 9
    with pytest.raises(ValueError):
        result.grid_edges[0][0] = -1
    with pytest.raises(TypeError):
        result.yee_tensors["extra"] = np.array(1)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.scene_hash = "changed"  # type: ignore[misc]


def test_two_dimensional_tm_omits_unused_components():
    result = raster.rasterize(
        box_scene(),
        raster.Grid.uniform((0, 0, 0), (1, 1, 1), (2, 2, 1)),
        options=raster.RasterOptions(components="two_dimensional_tm"),
    )

    assert set(result.yee_tensors) == {
        "epsilon_ez",
        "conductivity_ez",
        "mu_hx",
        "mu_hy",
    }
    assert result.tensors["epsilon"].shape == (1, 1, 2, 2)


def test_two_dimensional_te_omits_unused_components():
    result = raster.rasterize(
        box_scene(),
        raster.Grid.uniform((0, 0, 0), (1, 1, 1), (2, 2, 1)),
        options=raster.RasterOptions(components="two_dimensional_te"),
    )

    assert set(result.yee_tensors) == {
        "epsilon_ex",
        "epsilon_ey",
        "conductivity_ex",
        "conductivity_ey",
        "mu_hz",
    }
    assert result.tensors["epsilon"].shape == (1, 1, 2, 2)


def test_compiled_scene_reuse_and_cache_recovery(tmp_path):
    compiled = raster.compile_scene(box_scene())
    grid = raster.Grid.uniform((0, 0, 0), (1, 1, 1), (2, 2, 2))

    first = compiled.rasterize(grid, cache_directory=tmp_path)
    second = compiled.rasterize(grid, cache_directory=tmp_path)
    assert not first.cache_hit
    assert second.cache_hit
    np.testing.assert_array_equal(first.tensors["epsilon"], second.tensors["epsilon"])

    cache_file = next(tmp_path.glob("*.npz"))
    cache_file.write_bytes(b"broken")
    recovered = compiled.rasterize(grid, cache_directory=tmp_path)
    assert not recovered.cache_hit
    np.testing.assert_array_equal(
        first.tensors["epsilon"], recovered.tensors["epsilon"]
    )


def test_cache_round_trip_preserves_tensor_materials_and_smoothing(tmp_path):
    scene = raster.Scene(
        (
            raster.Material(
                epsilon_r=((3.0, 0.2, 0.0), (0.2, 2.0, 0.1), (0.0, 0.1, 1.5))
            ),
        )
    )
    compiled = raster.compile_scene(scene)
    grid = raster.Grid([0.0, 0.3, 1.0], [0.0, 1.0], [0.0, 0.4, 1.0])
    options = raster.RasterOptions(smoothing="farjadpour_full")

    first = compiled.rasterize(grid, options=options, cache_directory=tmp_path)
    cached = compiled.rasterize(grid, options=options, cache_directory=tmp_path)

    assert cached.cache_hit
    assert cached.smoothing == "farjadpour_full"
    assert not cached.is_uniform
    for name in first.tensors:
        np.testing.assert_array_equal(first.tensors[name], cached.tensors[name])
    for name in first.yee_tensors:
        np.testing.assert_array_equal(first.yee_tensors[name], cached.yee_tensors[name])
    with np.load(next(tmp_path.glob("*.npz")), allow_pickle=False) as payload:
        names = set(payload.files)
    assert "yee__epsilon_ex" in names
    assert "yee__conductivity_ez" in names
    assert not names.intersection(
        {"yee__eps_x", "yee__eps_y", "yee__eps_z", "yee__sig_x", "yee__sig_y"}
    )


def test_scene_hash_and_cache_include_materials_and_nonuniform_edges(tmp_path):
    first_scene = raster.compile_scene(box_scene(4))
    second_scene = raster.compile_scene(box_scene(5))
    assert first_scene.hash != second_scene.hash

    first_grid = raster.Grid([0, 0.4, 1], [0, 1], [0, 1])
    second_grid = raster.Grid([0, 0.5, 1], [0, 1], [0, 1])
    first_scene.rasterize(first_grid, cache_directory=tmp_path)
    first_scene.rasterize(second_grid, cache_directory=tmp_path)
    assert len(tuple(tmp_path.glob("*.npz"))) == 2


def test_invalid_public_options_fail_early():
    with pytest.raises(ValueError, match="quality"):
        raster.RasterOptions(quality="custom")
    with pytest.raises(ValueError, match="smoothing"):
        raster.RasterOptions(smoothing="electrostatic_cell")
    with pytest.raises(ValueError, match="components"):
        raster.RasterOptions(components="ex")


@pytest.mark.parametrize(
    "geometry",
    [
        raster.Sphere((0.5, 0.5, 0.5), 0.3),
        raster.Cylinder((0.5, 0.5), 0.3, 0.2, 0.8),
        raster.ExtrudedPolygon(
            raster.Polygon(((0.2, 0.2), (0.8, 0.2), (0.5, 0.8))),
            0.2,
            0.8,
        ),
    ],
)
def test_supported_geometry_rasterizes(geometry):
    scene = raster.Scene(
        (raster.Material(), raster.Material(3)),
        (raster.Object(geometry, 1),),
    )
    result = raster.rasterize(
        scene,
        raster.Grid.uniform((0, 0, 0), (1, 1, 1), (4, 4, 4)),
        options=raster.RasterOptions(quality="fast"),
    )
    assert result.tensors["epsilon"][0].max() > 1.0


def test_material_accepts_scalar_diagonal_and_symmetric_tensor():
    scalar = raster.Material(4.0)
    diagonal = raster.Material((2.0, 3.0, 4.0))
    tensor = raster.Material(((3.0, 0.2, 0.0), (0.2, 2.0, 0.1), (0.0, 0.1, 1.5)))

    assert scalar.epsilon_r == (4.0, 4.0, 4.0, 0.0, 0.0, 0.0)
    assert diagonal.epsilon_r[:3] == (2.0, 3.0, 4.0)
    assert tensor.epsilon_r[3:] == (0.2, 0.0, 0.1)
    with pytest.raises(ValueError, match="symmetric"):
        raster.Material(((2.0, 1.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)))
    with pytest.raises(ValueError, match="positive definite"):
        raster.Material(((1.0, 2.0, 0.0), (2.0, 1.0, 0.0), (0.0, 0.0, 1.0)))


def test_diagonal_material_uses_three_component_tensor_storage():
    scene = raster.Scene(
        (raster.Material((2.0, 3.0, 4.0)),),
        (),
    )
    result = raster.rasterize(
        scene,
        raster.Grid.uniform((0, 0, 0), (1, 1, 1), (2, 2, 2)),
    )

    assert result.tensors["epsilon"].shape == (3, 2, 2, 2)
    np.testing.assert_array_equal(result.tensors["epsilon"][:, 0, 0, 0], (2, 3, 4))


def test_intrinsic_full_tensors_are_retained_at_their_field_supports():
    epsilon = (3.0, 2.0, 1.5, 0.2, 0.0, 0.1)
    mu = (2.0, 1.5, 1.2, 0.1, 0.0, 0.0)
    conductivity = (1.0, 0.5, 0.2, 0.1, 0.0, 0.0)
    result = raster.rasterize(
        raster.Scene((raster.Material(epsilon, mu, conductivity),)),
        raster.Grid.uniform((0, 0, 0), (1, 1, 1), (2, 3, 4)),
    )

    for name in ("epsilon_ex", "epsilon_ey", "epsilon_ez"):
        values = result.yee_tensors[name]
        assert values.shape[0] == 6
        np.testing.assert_allclose(values[:, 0, 0, 0], epsilon)
    for name in ("conductivity_ex", "conductivity_ey", "conductivity_ez"):
        values = result.yee_tensors[name]
        assert values.shape[0] == 6
        np.testing.assert_allclose(values[:, 0, 0, 0], conductivity)
    for name in ("mu_hx", "mu_hy", "mu_hz"):
        values = result.yee_tensors[name]
        assert values.shape[0] == 6
        np.testing.assert_allclose(values[:, 0, 0, 0], mu)


def test_full_farjadpour_produces_off_diagonal_cell_tensor():
    scene = raster.Scene(
        (raster.Material(), raster.Material(4.0)),
        (
            raster.Object(
                raster.ExtrudedPolygon(
                    raster.Polygon(((-1.0, -1.0), (2.0, -1.0), (-1.0, 2.0))),
                    -10.0,
                    10.0,
                ),
                1,
            ),
        ),
    )
    result = raster.rasterize(
        scene,
        raster.Grid.uniform((0, 0, 0), (1, 1, 1), (1, 1, 1)),
        options=raster.RasterOptions(smoothing="farjadpour_full", quality="reference"),
    )

    assert result.smoothing == "farjadpour_full"
    assert abs(float(result.tensors["epsilon"][3, 0, 0, 0])) > 1e-3
    assert result.diagnostics["smoothed_samples"] > 0


def test_multisided_interface_falls_back_with_public_diagnostic():
    scene = raster.Scene(
        (raster.Material(), raster.Material(4.0)),
        (
            raster.Object(
                raster.ExtrudedPolygon(
                    raster.Polygon(((-0.2, 0.3), (0.7, -0.2), (1.2, 0.7), (0.3, 1.2))),
                    -10.0,
                    10.0,
                ),
                1,
            ),
        ),
    )
    result = raster.rasterize(
        scene,
        raster.Grid.uniform((0, 0, 0), (1, 1, 1), (1, 1, 1)),
        options=raster.RasterOptions(smoothing="farjadpour_full"),
    )

    assert result.diagnostics["fallback_multiple_orientations"] > 0
    assert result.diagnostics["ambiguous_interface_samples"] > 0


def test_oblique_farjadpour_tensor_matches_layered_medium_solution():
    # x + y = 1 bisects the cell. The exact effective eigenvalue is harmonic
    # normal to the interface and arithmetic along both tangent directions.
    scene = raster.Scene(
        (raster.Material(), raster.Material(4.0)),
        (
            raster.Object(
                raster.ExtrudedPolygon(
                    raster.Polygon(((-1, -1), (2, -1), (-1, 2))),
                    -1,
                    2,
                ),
                1,
            ),
        ),
    )
    result = raster.rasterize(
        scene,
        raster.Grid.uniform((0, 0, 0), (1, 1, 1), (1, 1, 1)),
        options=raster.RasterOptions(smoothing="farjadpour_full"),
    )

    packed = result.tensors["epsilon"][:, 0, 0, 0]
    matrix = np.array(
        (
            (packed[0], packed[3], packed[4]),
            (packed[3], packed[1], packed[5]),
            (packed[4], packed[5], packed[2]),
        )
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(matrix),
        (1.6, 2.5, 2.5),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(matrix[:2, :2], ((2.05, -0.45), (-0.45, 2.05)))
    for name in ("epsilon_ex", "epsilon_ey", "epsilon_ez"):
        tensor = result.yee_tensors[name]
        assert tensor.shape[0] == 6
        assert np.any(np.abs(tensor[3]) > 1e-3)
        assert not tensor.flags.writeable


def test_farjadpour_tensor_matches_three_four_five_normal_oracle():
    """Reproduce an independent n=(0.6, 0.8, 0) tensor oracle."""

    # 0.6 x + 0.8 y = 0.7 bisects the unit cell by central symmetry.  The
    # oversized polygon keeps every boundary except that interface outside it.
    scene = raster.Scene(
        (raster.Material(), raster.Material(4.0)),
        (
            raster.Object(
                raster.ExtrudedPolygon(
                    raster.Polygon(((-10, -10), (10, -10), (10, -6.625), (-10, 8.375))),
                    -10,
                    10,
                ),
                1,
            ),
        ),
    )
    result = raster.rasterize(
        scene,
        raster.Grid.uniform((0, 0, 0), (1, 1, 1), (1, 1, 1)),
        options=raster.RasterOptions(smoothing="farjadpour_full"),
    )

    packed = result.tensors["epsilon"][:, 0, 0, 0]
    np.testing.assert_allclose(
        packed,
        (2.176, 1.924, 2.5, -0.432, 0.0, 0.0),
        rtol=1e-6,
        atol=1e-6,
    )
    assert result.diagnostics["smoothed_samples"] > 0
    assert result.diagnostics["ambiguous_interface_samples"] == 0


def test_tapered_farjadpour_result_is_invariant_to_length_units():
    points = ((0.173, 0.193), (0.817, 0.223), (0.791, 0.843), (0.201, 0.779))

    def solve(scale: float):
        polygon = raster.Polygon(tuple((scale * x, scale * y) for x, y in points))
        scene = raster.Scene(
            (raster.Material(), raster.Material(4.0)),
            (
                raster.Object(
                    raster.TaperedExtrudedPolygon(
                        polygon,
                        0.041 * scale,
                        0.937 * scale,
                        sidewall_angle_degrees=17.3,
                        width_to_z=0.37,
                    ),
                    1,
                ),
            ),
        )
        return raster.rasterize(
            scene,
            raster.Grid.uniform((0, 0, 0), (scale, scale, scale), (5, 6, 7)),
            options=raster.RasterOptions(
                quality="reference", smoothing="farjadpour_full"
            ),
        )

    normalized = solve(1.0)
    metres = solve(1e-6)
    for name in normalized.tensors:
        np.testing.assert_array_equal(normalized.tensors[name], metres.tensors[name])
    for name in normalized.yee_tensors:
        np.testing.assert_array_equal(
            normalized.yee_tensors[name], metres.yee_tensors[name]
        )
    non_timing = {
        name for name in normalized.diagnostics if not name.endswith("_seconds")
    }
    assert {name: normalized.diagnostics[name] for name in non_timing} == {
        name: metres.diagnostics[name] for name in non_timing
    }
