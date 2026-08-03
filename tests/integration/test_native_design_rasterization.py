from __future__ import annotations

import numpy as np
import pytest

import beamz as bz
from beamz.design import MaterialGrid
from beamz.design.raster import Grid, Material, RasterOptions, Scene, rasterize


def design_2d() -> bz.Design:
    return bz.Design(
        width=1e-6,
        height=1e-6,
        structures=(
            bz.Rectangle(
                position=(0.0, 0.0),
                width=0.5e-6,
                height=1e-6,
                material=bz.Material(
                    permittivity=4.0,
                    permeability=1.0,
                    conductivity=3.0,
                ),
            ),
        ),
    )


def design_3d() -> bz.Design:
    return bz.Design(
        width=1e-6,
        height=1e-6,
        depth=1e-6,
        structures=(
            bz.Box(
                center=(0.5e-6, 0.5e-6, 0.5e-6),
                size=(0.5e-6, 1e-6, 1e-6),
                material=bz.Material(
                    permittivity=4.0,
                    permeability=1.0,
                    conductivity=3.0,
                ),
            ),
        ),
    )


def test_design_rasterize_returns_direct_2d_solver_grid():
    grid = design_2d().rasterize(0.25e-6)

    assert isinstance(grid, MaterialGrid)
    assert grid.shape == (4, 4)
    assert grid.permittivity.shape == (4, 4)


def test_material_grid_rejects_nonpositive_resolution():
    values = np.ones((2, 2))

    with pytest.raises(ValueError, match="resolution must be finite and positive"):
        MaterialGrid(values, values, values, 0.0, values.shape)


def test_design_rasterize_returns_direct_3d_solver_grid():
    grid = design_3d().rasterize(0.25e-6)

    assert isinstance(grid, MaterialGrid)
    assert grid.shape == (4, 4, 4)
    assert grid.permittivity.shape == (4, 4, 4)


def test_design_tensor_material_uses_same_native_path_as_imported_scene():
    material = bz.Material(permittivity=(2.0, 3.0, 4.0))
    design = bz.Design(width=1.0, height=1.0, depth=1.0, background=material)

    material_grid = design.rasterize(0.5)

    assert material_grid.tensors["epsilon"].shape == (3, 2, 2, 2)
    np.testing.assert_allclose(material_grid.tensors["epsilon"][:, 0, 0, 0], (2, 3, 4))
    np.testing.assert_allclose(material_grid.yee_materials["eps_x"], 2.0)
    np.testing.assert_allclose(material_grid.yee_materials["eps_y"], 3.0)
    np.testing.assert_allclose(material_grid.yee_materials["eps_z"], 4.0)


@pytest.mark.parametrize("design", [design_2d(), design_3d()])
def test_default_scalar_volume_compilation_preserves_cell_colocation(design):
    material_grid = design.rasterize(0.25e-6)
    simulation = bz.Simulation(
        design=design,
        resolution=0.25e-6,
        run_time=1e-15,
    )
    compiled = simulation.compile().grid

    np.testing.assert_array_equal(compiled.permittivity, material_grid.permittivity)
    np.testing.assert_array_equal(compiled.permeability, material_grid.permeability)
    assert not material_grid.uses_direct_yee_materials


def test_simulation_conversion_rejects_nonunit_permeability():
    result = rasterize(
        Scene((Material(mu_r=2.0),)),
        Grid.uniform((0, 0, 0), (1, 1, 1), (2, 2, 2)),
    )

    with pytest.raises(ValueError, match="unit permeability"):
        MaterialGrid.from_raster_result(result)


def test_material_grid_rejects_nonuniform_raster_results():
    result = rasterize(
        Scene((Material(),)),
        Grid([0.0, 0.2, 1.0], [0.0, 1.0], [0.0, 1.0]),
    )

    with pytest.raises(ValueError, match="uniform spacing"):
        MaterialGrid.from_raster_result(result)


def test_material_grid_infers_two_dimensional_tm_output():
    result = rasterize(
        Scene((Material(),)),
        Grid.uniform((0, 0, 0), (1, 1, 1), (2, 2, 1)),
        options=RasterOptions(components="two_dimensional_tm"),
    )
    grid = MaterialGrid.from_raster_result(result)

    assert grid.shape == (2, 2)
    assert grid.permittivity.shape == (2, 2)


def test_material_grid_infers_two_dimensional_te_output():
    result = rasterize(
        Scene((Material(),)),
        Grid.uniform((0, 0, 0), (1, 1, 1), (2, 2, 1)),
        options=RasterOptions(components="two_dimensional_te"),
    )
    grid = MaterialGrid.from_raster_result(result, polarization="te")

    assert grid.shape == (2, 2)
    assert grid.polarization == "te"
    assert set(grid.yee_materials) == {
        "eps_x",
        "eps_y",
        "sig_x",
        "sig_y",
        "mu_hz",
    }


def test_full_one_cell_raster_remains_3d_unless_2d_is_explicit():
    result = rasterize(
        Scene((Material(epsilon_r=(2.0, 3.0, 4.0)),)),
        Grid.uniform((0, 0, 0), (1, 1, 0.5), (2, 2, 1)),
    )

    inferred = MaterialGrid.from_raster_result(result)
    tm = MaterialGrid.from_raster_result(result, dimensions=2, polarization="tm")
    te = MaterialGrid.from_raster_result(result, dimensions=2, polarization="te")

    assert inferred.shape == (1, 2, 2)
    assert inferred.polarization is None
    assert set(inferred.yee_materials) == {
        "eps_x",
        "eps_y",
        "eps_z",
        "sig_x",
        "sig_y",
        "sig_z",
        "mu_hx",
        "mu_hy",
        "mu_hz",
    }
    assert set(tm.yee_materials) == {"eps_z", "sig_z", "mu_hx", "mu_hy"}
    assert set(te.yee_materials) == {"eps_x", "eps_y", "sig_x", "sig_y", "mu_hz"}


def test_diagonal_tensor_result_preserves_native_yee_coefficients():
    result = rasterize(
        Scene((Material(epsilon_r=(2.0, 3.0, 4.0)),)),
        Grid.uniform((0, 0, 0), (1, 1, 1), (2, 2, 2)),
    )

    grid = MaterialGrid.from_raster_result(result, dimensions=3)

    for target, source, axis in (
        ("eps_x", "epsilon_ex", 0),
        ("eps_y", "epsilon_ey", 1),
        ("eps_z", "epsilon_ez", 2),
    ):
        np.testing.assert_array_equal(
            grid.yee_materials[target], result.yee_tensors[source][axis]
        )
    assert np.all(grid.yee_materials["eps_x"] == 2.0)
    assert np.all(grid.yee_materials["eps_y"] == 3.0)
    assert np.all(grid.yee_materials["eps_z"] == 4.0)


def test_raster_result_material_grid_enters_simulation_without_resampling():
    result = rasterize(
        Scene((Material(epsilon_r=(2.0, 3.0, 4.0)),)),
        Grid.uniform((2.0, 3.0, 4.0), (3.0, 4.0, 5.0), (2, 2, 2)),
    )
    material_grid = MaterialGrid.from_raster_result(result, dimensions=3)
    simulation = bz.Simulation(material_grid=material_grid, run_time=2e-9)

    assert simulation.material_grid is material_grid
    assert simulation.resolution == 0.5
    assert simulation.size == (1.0, 1.0, 1.0)
    assert simulation.coordinate_offset == (-2.0, -3.0, -4.0)
    assert simulation.to_request().materials is material_grid
    compiled = simulation.compile().grid
    for name, values in material_grid.yee_materials.items():
        np.testing.assert_array_equal(getattr(compiled, name), values)


@pytest.mark.parametrize(
    ("shape", "dimensions", "components", "conductivity_name"),
    (
        ((4, 4, 1), 2, "two_dimensional_tm", "sig_z"),
        ((4, 4, 4), 3, "all", "sig_x"),
    ),
)
@pytest.mark.parametrize(
    ("formulation", "adds_conductivity"),
    (("sponge", True), ("cpml", False)),
)
def test_direct_yee_conductivity_composes_with_pml(
    shape,
    dimensions,
    components,
    conductivity_name,
    formulation,
    adds_conductivity,
):
    result = rasterize(
        Scene(
            (
                Material(
                    epsilon_r=(2.0, 3.0, 4.0),
                    conductivity=(1.0, 2.0, 3.0),
                ),
            )
        ),
        Grid.uniform((0, 0, 0), (1, 1, 1), shape),
        options=RasterOptions(components=components),
    )
    material_grid = MaterialGrid.from_raster_result(result, dimensions=dimensions)

    compiled = (
        bz.Simulation(
            material_grid=material_grid,
            boundaries=[bz.PML(thickness=0.25, sigma_max=6.0, formulation=formulation)],
            run_time=2e-9,
        )
        .compile()
        .grid
    )

    physical = np.asarray(material_grid.yee_materials[conductivity_name])
    with_pml = np.asarray(getattr(compiled, conductivity_name))
    np.testing.assert_array_equal(with_pml.shape, physical.shape)
    assert np.all(with_pml >= physical)
    assert bool(np.any(with_pml > physical)) is adds_conductivity


def test_imported_scene_enters_simulation_without_manual_conversion():
    scene = Scene((Material(epsilon_r=3.0),))
    raster_grid = Grid.uniform((1.0, 2.0, 0.0), (2.0, 3.0, 0.5), (2, 2, 1))

    simulation = bz.Simulation(
        scene=scene,
        raster_grid=raster_grid,
        run_time=2e-9,
    )

    assert simulation.material_grid is not None
    assert simulation.size == (1.0, 1.0, 0.0)
    assert simulation.coordinate_offset == (-1.0, -2.0, 0.0)
    np.testing.assert_allclose(simulation.material_grid.permittivity, 3.0)
    np.testing.assert_array_equal(
        simulation.compile().grid.eps_z,
        simulation.material_grid.yee_materials["eps_z"],
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"scene": Scene((Material(),))},
        {"raster_grid": Grid.uniform((0, 0, 0), (1, 1, 1), (1, 1, 1))},
    ),
)
def test_scene_and_raster_grid_must_be_supplied_together(kwargs):
    with pytest.raises(ValueError, match="scene and raster_grid together"):
        bz.Simulation(run_time=1e-9, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (
            {
                "scene": object(),
                "raster_grid": Grid.uniform((0, 0, 0), (1, 1, 1), (1, 1, 1)),
            },
            "scene must",
        ),
        ({"scene": Scene((Material(),)), "raster_grid": object()}, "raster_grid must"),
        (
            {
                "scene": Scene((Material(),)),
                "raster_grid": Grid.uniform((0, 0, 0), (1, 1, 1), (1, 1, 1)),
                "raster_options": object(),
            },
            "raster_options must",
        ),
    ),
)
def test_imported_scene_simulation_validates_native_types(kwargs, message):
    with pytest.raises(TypeError, match=message):
        bz.Simulation(run_time=1e-9, **kwargs)


def test_scene_rejects_conflicting_material_sources():
    scene = Scene((Material(),))
    raster_grid = Grid.uniform((0, 0, 0), (1, 1, 1), (2, 2, 2))

    with pytest.raises(ValueError, match="exactly one material source"):
        bz.Simulation(
            design=design_3d(),
            scene=scene,
            raster_grid=raster_grid,
            run_time=1e-9,
        )


def test_two_dimensional_raster_grid_enters_tm_simulation():
    result = rasterize(
        Scene((Material(epsilon_r=3.0),)),
        Grid.uniform((1.0, 2.0, 0.0), (2.0, 3.0, 0.5), (2, 2, 1)),
        options=RasterOptions(components="two_dimensional_tm"),
    )
    material_grid = MaterialGrid.from_raster_result(result, dimensions=2)
    simulation = bz.Simulation(material_grid=material_grid, run_time=2e-9)

    assert simulation.size == (1.0, 1.0, 0.0)
    assert simulation.coordinate_offset == (-1.0, -2.0, 0.0)
    compiled = simulation.compile().grid
    for name, values in material_grid.yee_materials.items():
        np.testing.assert_array_equal(getattr(compiled, name), values)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"design": design_3d()}, "either design"),
        ({"background": bz.Material(2.0)}, "background"),
        ({"grid_spec": bz.GridSpec.uniform(0.5)}, "grid_spec"),
        ({"raster_options": RasterOptions()}, "raster_options"),
        ({"domain": (2.0, 1.0, 1.0)}, "must match"),
    ],
)
def test_material_grid_simulation_rejects_conflicting_geometry_controls(
    kwargs, message
):
    result = rasterize(
        Scene((Material(),)),
        Grid.uniform((0, 0, 0), (1, 1, 1), (2, 2, 2)),
    )
    material_grid = MaterialGrid.from_raster_result(result, dimensions=3)

    with pytest.raises(ValueError, match=message):
        bz.Simulation(material_grid=material_grid, run_time=2e-9, **kwargs)


def test_full_tensor_and_full_farjadpour_require_future_engine():
    tensor_scene = Scene(
        (Material(epsilon_r=((3.0, 0.2, 0.0), (0.2, 2.0, 0.0), (0.0, 0.0, 1.0))),)
    )
    grid = Grid.uniform((0, 0, 0), (1, 1, 1), (2, 2, 2))
    with pytest.raises(ValueError, match="silently discard"):
        MaterialGrid.from_raster_result(rasterize(tensor_scene, grid))

    full = rasterize(
        Scene((Material(),)),
        grid,
        options=RasterOptions(smoothing="farjadpour_full"),
    )
    with pytest.raises(ValueError, match="full off-diagonal Farjadpour"):
        MaterialGrid.from_raster_result(full)


def test_simulation_selects_explicit_diagonal_farjadpour_policy():
    simulation = bz.Simulation(
        design=design_3d(),
        resolution=0.25e-6,
        run_time=1e-15,
        raster_options=RasterOptions(smoothing="farjadpour_diagonal"),
    )

    material_grid = simulation.to_request().materials
    compiled = simulation.compile().grid

    assert material_grid.smoothing == "farjadpour_diagonal"
    assert set(material_grid.yee_materials) == {
        "eps_x",
        "eps_y",
        "eps_z",
        "sig_x",
        "sig_y",
        "sig_z",
        "mu_hx",
        "mu_hy",
        "mu_hz",
    }
    assert material_grid.uses_direct_yee_materials
    for name, values in material_grid.yee_materials.items():
        np.testing.assert_array_equal(getattr(compiled, name), values)


def test_design_rasterize_has_no_engine_selection_switch():
    with pytest.raises(TypeError, match="unexpected keyword argument 'engine'"):
        design_2d().rasterize(0.25e-6, engine="legacy")


def test_native_circle_rasterization_preserves_symmetry():
    design = bz.Design(width=6.0, height=6.0, material=bz.Material(1.0))
    design += bz.Circle(
        position=(3.0, 3.0),
        radius=1.35,
        material=bz.Material(12.0),
    )

    epsilon = design.rasterize(0.2, quality="reference").permittivity

    np.testing.assert_allclose(epsilon, np.fliplr(epsilon), atol=1e-6)
    np.testing.assert_allclose(epsilon, np.flipud(epsilon), atol=1e-6)


def test_native_polygon_respects_explicit_vertical_placement():
    design = bz.Design(width=1.0, height=1.0, depth=3.0, material=bz.Material(1.0))
    design += bz.Polygon(
        vertices=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        z=2.0,
        depth=0.5,
        material=bz.Material(5.0),
    )

    epsilon = design.rasterize(0.5).permittivity

    np.testing.assert_allclose(epsilon[:4], 1.0)
    np.testing.assert_allclose(epsilon[4], 5.0)
    np.testing.assert_allclose(epsilon[5], 1.0)


def test_native_sidewall_rasterization_narrows_toward_top():
    design = bz.Design(width=2.0, height=2.0, depth=1.2, material=bz.Material(1.0))
    design += bz.Rectangle(
        position=(0.5, 0.5, 0.1),
        width=1.0,
        height=1.0,
        depth=1.0,
        material=bz.Material(9.0),
        sidewall_angle=10.0,
        width_to_z=0.0,
    )

    epsilon = design.rasterize(0.1, quality="reference").permittivity

    assert np.count_nonzero(epsilon[-2] > 2.0) < np.count_nonzero(epsilon[1] > 2.0)
