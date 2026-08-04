from __future__ import annotations

import numpy as np
import pytest

import beamz as bz
from beamz.design import MaterialGrid
from beamz.design.raster import (
    ExtrudedPolygon,
    Grid,
    Material,
    Object,
    Polygon,
    RasterOptions,
    Scene,
    rasterize,
)
from beamz.simulation.model import ShardingLayout
from beamz.simulation.sharding import _lower_coefficients


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


@pytest.mark.parametrize(
    ("yee_tensors", "message"),
    (
        ({"unknown": np.ones((6, 2, 2))}, "Unknown Yee permittivity tensors"),
        ({"eps_x": np.ones((2, 3, 2))}, "expected compact"),
    ),
)
def test_material_grid_validates_full_yee_tensor_contract(yee_tensors, message):
    values = np.ones((2, 2))

    with pytest.raises(ValueError, match=message):
        MaterialGrid(
            values,
            values,
            values,
            1.0,
            values.shape,
            yee_tensors=yee_tensors,
        )


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
def test_default_farjadpour_compilation_preserves_native_colocation(design):
    material_grid = design.rasterize(0.25e-6)
    simulation = bz.Simulation(
        design=design,
        resolution=0.25e-6,
        run_time=1e-15,
    )
    compiled = simulation.compile().grid

    np.testing.assert_array_equal(compiled.permittivity, material_grid.permittivity)
    np.testing.assert_array_equal(compiled.permeability, material_grid.permeability)
    assert material_grid.smoothing == "farjadpour_diagonal"
    assert material_grid.uses_direct_yee_materials


def test_simulation_conversion_rejects_nonunit_permeability():
    result = rasterize(
        Scene((Material(mu_r=2.0),)),
        Grid.uniform((0, 0, 0), (1, 1, 1), (2, 2, 2)),
    )

    with pytest.raises(ValueError, match="unit permeability"):
        MaterialGrid.from_raster_result(result)


@pytest.mark.parametrize(
    ("epsilon", "conductivity", "dimensions", "message"),
    (
        (
            (3.0, 2.0, 1.5),
            (1.0, 0.5, 0.2, 0.1, 0.0, 0.0),
            3,
            "off-diagonal conductivity",
        ),
        ((3.0, 2.0, 1.5, 0.0, 0.1, 0.0), 0.0, 2, "without xz or yz"),
        ((3.0, 2.0, 1.5, 0.2, 0.0, 0.0), (1.0, 0.5, 0.2), 3, "zero conductivity"),
    ),
)
def test_full_tensor_conversion_rejects_unsupported_material_couplings(
    epsilon, conductivity, dimensions, message
):
    shape = (2, 2, 2 if dimensions == 3 else 1)
    result = rasterize(
        Scene((Material(epsilon_r=epsilon, conductivity=conductivity),)),
        Grid.uniform((0, 0, 0), (1, 1, 1), shape),
    )

    with pytest.raises(ValueError, match=message):
        MaterialGrid.from_raster_result(result, dimensions=dimensions)


def test_material_grid_preserves_nonuniform_raster_edges():
    result = rasterize(
        Scene((Material(),)),
        Grid([0.0, 0.2, 1.0], [0.0, 1.0], [0.0, 1.0]),
    )

    material_grid = MaterialGrid.from_raster_result(result)

    np.testing.assert_array_equal(material_grid.grid.x_edges, [0.0, 0.2, 1.0])
    assert material_grid.grid.metric_kind == "rectilinear"
    assert material_grid.resolution == pytest.approx(0.2)
    assert material_grid.uses_direct_yee_materials


def test_design_rasterize_accepts_realized_rectilinear_grid():
    design = design_2d()
    grid = Grid(
        [0.0, 0.2e-6, 0.5e-6, 1.0e-6],
        [0.0, 0.25e-6, 0.6e-6, 1.0e-6],
        [0.0, 1.0],
    )

    material_grid = design.rasterize(grid)

    assert material_grid.grid == grid
    assert material_grid.metric_kind == "rectilinear"
    assert material_grid.uses_direct_yee_materials


def test_uniform_grid_spec_preview_matches_simulation_and_rasterization():
    design = bz.Design(width=1.0e-6, height=0.8e-6)
    spec = bz.GridSpec.uniform(0.3e-6)
    preview = spec.realize(design)

    simulation = bz.Simulation(
        design=design,
        grid_spec=spec,
        time=np.asarray([0.0, 1e-16]),
    )
    direct = design.rasterize(preview)

    assert preview == simulation.grid
    assert preview == simulation._material_grid().grid
    assert preview == direct.grid
    assert preview.extent == pytest.approx((1.2e-6, 0.9e-6, 1.0))


def test_automatic_grid_subpixel_ring_recovers_analytic_material_area():
    n_clad, n_core = 1.44, 2.0
    inner_radius, outer_radius = 0.75e-6, 1.05e-6
    design = bz.Design(
        width=3.0e-6,
        height=3.0e-6,
        background=bz.Material(permittivity=n_clad**2),
        structures=(
            bz.Ring(
                position=(1.5e-6, 1.5e-6),
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                material=bz.Material(permittivity=n_core**2),
            ),
        ),
    )
    grid = bz.GridSpec.auto(
        wavelength=1.55e-6,
        min_steps_per_wvl=14,
        min_feature_cells=6,
        max_scale=1.15,
    ).realize(design)

    material_grid = design.rasterize(
        grid,
        quality="balanced",
        smoothing="farjadpour_full",
        polarization="tm",
    )
    epsilon = np.asarray(material_grid.permittivity)
    fill_fraction = (epsilon - n_clad**2) / (n_core**2 - n_clad**2)
    cell_areas = np.diff(grid.y_edges)[:, None] * np.diff(grid.x_edges)[None, :]
    raster_area = float(np.sum(fill_fraction * cell_areas))
    analytic_area = float(np.pi * (outer_radius**2 - inner_radius**2))
    partial_cells = (fill_fraction > 1e-5) & (fill_fraction < 1.0 - 1e-5)

    assert np.count_nonzero(partial_cells) > 100
    assert raster_area == pytest.approx(analytic_area, rel=2e-3)
    assert grid.quality_report().satisfies_max_scale(
        1.15 * (1.0 + 1e-12), active_axes=("x", "y")
    )


def test_simulation_realizes_and_rasterizes_geometry_aware_grid_spec():
    design = design_2d()
    grid_spec = bz.GridSpec.auto(
        wavelength=1.55e-6,
        min_steps_per_wvl=10,
        max_scale=1.2,
    )
    simulation = bz.Simulation(
        design=design,
        grid_spec=grid_spec,
        run_time=2e-15,
        sources=[
            bz.GaussianSource(
                position=(0.25e-6, 0.5e-6),
                width=0.2e-6,
                signal=np.ones(8),
            )
        ],
        monitors=[bz.FieldRecorder(("Ez",), interval=2, name="fields")],
    )

    material_grid = simulation._material_grid()
    program = simulation.compile()

    assert isinstance(simulation.resolution, float)
    assert material_grid.grid == simulation.grid
    assert material_grid.metric_kind == "rectilinear"
    assert program.config.metric_kind == "rectilinear"
    expected_dt = grid_spec.resolve_time_step(simulation.grid, dims=2)
    assert simulation.dt == pytest.approx(expected_dt)


def test_automatic_grid_simulation_copy_and_result_metadata_keep_exact_grid():
    simulation = bz.Simulation(
        design=design_2d(),
        grid_spec=bz.GridSpec.auto(wavelength=1.55e-6, max_scale=1.2),
        run_time=2e-15,
        sources=[],
        monitors=[],
    )

    copied = simulation.updated_copy(monitors=())
    program = copied.compile()
    results = bz.SimulationResults.from_run(
        copied,
        runtime_fields=program.grid,
        monitor_results={},
    )

    assert copied.grid == simulation.grid
    assert copied.resolution == pytest.approx(simulation.grid.minimum_spacing)
    assert results.metadata.resolution == pytest.approx(copied.resolution)
    assert results.metadata.grid == copied.grid


def test_automatic_grid_keeps_rectilinear_mesh_for_mode_devices():
    monitor = bz.ModeMonitor(
        center=(0.5e-6, 0.5e-6, 0.0),
        size=(0.0, 0.5e-6, 1.0),
        freqs=[2.0e14],
        mode_spec=bz.ModeSpec(polarization="tm"),
        name="mode",
    )
    simulation = bz.Simulation(
        design=design_2d(),
        grid_spec=bz.GridSpec.auto(wavelength=1.55e-6),
        run_time=2e-15,
        sources=[],
        monitors=[monitor],
    )

    assert simulation.grid.metric_kind_for(("x", "y")) == "rectilinear"
    assert simulation._material_grid().grid == simulation.grid
    simulation.compile()


def test_compiler_builds_separable_staggered_metrics_for_rectilinear_grid():
    result = rasterize(
        Scene((Material(),)),
        Grid([0.0, 0.2, 1.0], [0.0, 0.5, 1.0], [0.0, 1.0]),
        options=RasterOptions(components="two_dimensional_tm"),
    )
    material_grid = MaterialGrid.from_raster_result(result)
    simulation = bz.Simulation(
        material_grid=material_grid,
        time=np.asarray([0.0, 1e-16]),
    )

    program = simulation.compile()

    assert program.config.metric_kind == "rectilinear"
    np.testing.assert_allclose(program.metrics.e_to_h_x, [5.0, 1.25])
    np.testing.assert_allclose(program.metrics.h_to_e_x, [5.0, 2.0, 1.25])
    np.testing.assert_allclose(program.metrics.e_to_h_y, [2.0, 2.0])
    assert program.metrics.e_to_h_z.size == 0


def test_uniform_compiler_keeps_metric_leaves_empty_for_scalar_fast_path():
    simulation = bz.Simulation(
        design=design_2d(),
        resolution=0.25e-6,
        time=np.asarray([0.0, 1e-16]),
    )

    program = simulation.compile()

    assert program.config.metric_kind == "isotropic_uniform"
    assert all(value.size == 0 for value in program.metrics)


@pytest.mark.parametrize("polarization", ["tm", "te"])
@pytest.mark.parametrize("cpml", [False, True])
def test_rectilinear_2d_update_kernel_executes_for_both_polarizations(
    polarization, cpml
):
    result = rasterize(
        Scene((Material(),)),
        Grid([0.0, 0.2, 0.6, 1.0], [0.0, 0.3, 1.0], [0.0, 1.0]),
        options=RasterOptions(components=f"two_dimensional_{polarization}"),
    )
    material_grid = MaterialGrid.from_raster_result(result, polarization=polarization)
    simulation = bz.Simulation(
        material_grid=material_grid,
        polarization=polarization,
        time=np.asarray([0.0, 1e-16]),
        boundaries=[bz.PML(thickness=0.2, formulation="cpml")] if cpml else None,
    )
    state = simulation.initial_state()
    state = state._replace(
        ez=np.ones_like(state.ez) if polarization == "tm" else state.ez,
        ex=np.ones_like(state.ex) if polarization == "te" else state.ex,
    )

    advanced = simulation.step(state)

    assert int(advanced.current_step) == 1
    assert all(
        np.isfinite(np.asarray(value)).all()
        for value in (
            advanced.ex,
            advanced.ey,
            advanced.ez,
            advanced.hx,
            advanced.hy,
            advanced.hz,
        )
    )


@pytest.mark.parametrize(
    ("grid", "metric_kind", "cpml"),
    [
        (
            Grid(
                [0.0, 0.2, 1.0],
                [0.0, 0.4, 1.0],
                [0.0, 0.3, 1.0],
            ),
            "rectilinear",
            False,
        ),
        (
            Grid.from_spacing((2, 2, 2), (0.2, 0.3, 0.4)),
            "axis_uniform",
            False,
        ),
        (
            Grid(
                [0.0, 0.2, 1.0],
                [0.0, 0.4, 1.0],
                [0.0, 0.3, 1.0],
            ),
            "rectilinear",
            True,
        ),
    ],
)
def test_nonisotropic_3d_update_kernel_executes(grid, metric_kind, cpml):
    material_grid = MaterialGrid.from_raster_result(
        rasterize(Scene((Material(),)), grid), dimensions=3
    )
    simulation = bz.Simulation(
        material_grid=material_grid,
        time=np.asarray([0.0, 1e-16]),
        boundaries=[bz.PML(thickness=0.2, formulation="cpml")] if cpml else None,
    )
    program = simulation.compile()
    state = simulation.initial_state()._replace(ez=np.ones_like(program.grid.Ez))

    advanced = simulation.step(state)

    assert program.config.metric_kind == metric_kind
    assert int(advanced.current_step) == 1
    assert all(
        np.isfinite(np.asarray(value)).all()
        for value in (
            advanced.ex,
            advanced.ey,
            advanced.ez,
            advanced.hx,
            advanced.hy,
            advanced.hz,
        )
    )


def test_rectilinear_monitors_compile_local_line_and_face_weights():
    grid_2d = Grid([0.0, 0.2, 1.0], [0.0, 0.3, 1.0], [0.0, 1.0])
    material_2d = MaterialGrid.from_raster_result(
        rasterize(
            Scene((Material(),)),
            grid_2d,
            options=RasterOptions(components="two_dimensional_tm"),
        )
    )
    line = bz.FieldMonitor(
        center=(0.1, 0.5, 0.0),
        size=(0.0, 1.0, 0.0),
        freqs=np.asarray([1.0]),
    )
    line_spec = (
        bz.Simulation(
            material_grid=material_2d,
            monitors=[line],
            time=np.asarray([0.0, 1e-16]),
        )
        .compile()
        .monitors[0]
    )

    np.testing.assert_allclose(line_spec.integration_weights, [0.3, 0.7])

    grid_3d = Grid(
        [0.0, 0.2, 1.0],
        [0.0, 0.3, 1.0],
        [0.0, 0.4, 1.0],
    )
    material_3d = MaterialGrid.from_raster_result(
        rasterize(Scene((Material(),)), grid_3d), dimensions=3
    )
    face = bz.FieldMonitor(
        center=(0.5, 0.5, 0.2),
        size=(1.0, 1.0, 0.0),
        freqs=np.asarray([1.0]),
    )
    face_spec = (
        bz.Simulation(
            material_grid=material_3d,
            monitors=[face],
            time=np.asarray([0.0, 1e-16]),
        )
        .compile()
        .monitors[0]
    )

    np.testing.assert_allclose(face_spec.integration_weights, [0.06, 0.24, 0.14, 0.56])
    assert face_spec.sample_region is not None
    assert face_spec.sample_region.plane_coord == pytest.approx(0.2)
    interval_x = face_spec.sample_region.axis_interval("x")
    interval_y = face_spec.sample_region.axis_interval("y")
    assert interval_x is not None and interval_y is not None
    assert interval_x.edges is None and interval_y.edges is None
    assert (interval_x.start, interval_x.stop) == (0, 2)
    assert (interval_y.start, interval_y.stop) == (0, 2)
    assert (interval_x.lower, interval_x.upper) == pytest.approx((0.0, 1.0))
    assert (interval_y.lower, interval_y.upper) == pytest.approx((0.0, 1.0))


def test_centered_design_monitor_uses_normalized_grid_once():
    monitor = bz.FieldMonitor(
        center=(0.0, 0.0, 0.0),
        size=(1.0, 1.0, 0.0),
        freqs=np.asarray([1.0]),
    )
    simulation = bz.Simulation(
        domain=(1.0, 1.0, 1.0),
        design=bz.Design(background=bz.Material()),
        monitors=[monitor],
        resolution=0.25,
        time=np.asarray([0.0, 1e-16]),
    )

    program = simulation.compile()

    assert simulation.coordinate_offset == (0.5, 0.5, 0.5)
    assert program.grid.geometry.origin == (0.0, 0.0, 0.0)
    assert program.grid.geometry.maximum == (1.0, 1.0, 1.0)
    assert program.monitors[0].dft_point_count == 16


def test_rectilinear_mode_monitor_compiles_with_exact_integration_weights():
    grid = Grid(
        [0.0, 0.2, 1.0],
        [0.0, 0.3, 1.0],
        [0.0, 0.4, 1.0],
    )
    material_grid = MaterialGrid.from_raster_result(
        rasterize(Scene((Material(),)), grid), dimensions=3
    )
    monitor = bz.ModeMonitor(
        center=(0.5, 0.5, 0.4),
        size=(1.0, 1.0, 0.0),
        freqs=np.asarray([1.0]),
    )

    simulation = bz.Simulation(
        material_grid=material_grid,
        monitors=[monitor],
        time=np.asarray([0.0, 1e-16]),
    )

    program = simulation.compile()

    assert program.config.metric_kind == "rectilinear"
    expected = np.outer(grid.cell_widths("y"), grid.cell_widths("x")).reshape(-1)
    np.testing.assert_allclose(program.monitors[0].integration_weights, expected)


def test_rectilinear_2d_mode_source_and_monitor_compile_on_auto_grid():
    wavelength = 1.55 * bz.um
    frequency = bz.LIGHT_SPEED / wavelength
    design = bz.Design(
        width=2.0 * bz.um,
        height=1.5 * bz.um,
        material=bz.Material(permittivity=1.0),
    ).with_structure(
        bz.Rectangle(
            position=(0.0, 0.55 * bz.um),
            width=2.0 * bz.um,
            height=0.4 * bz.um,
            material=bz.Material(permittivity=4.0),
        )
    )
    mode_spec = bz.ModeSpec(polarization="tm", target_neff=1.8)
    source = bz.ModeSource(
        center=(0.4 * bz.um, 0.75 * bz.um, 0.0),
        size=(0.0, 1.2 * bz.um, 1.0),
        source_time=bz.GaussianPulse(freq0=frequency, fwidth=0.1 * frequency),
        direction="+",
        mode_spec=mode_spec,
    )
    monitor = bz.ModeMonitor(
        center=(1.4 * bz.um, 0.75 * bz.um, 0.0),
        size=(0.0, 1.2 * bz.um, 1.0),
        freqs=[frequency],
        mode_spec=mode_spec,
        name="mode",
    )
    simulation = bz.Simulation(
        design=design,
        grid_spec=bz.GridSpec.auto(
            wavelength=wavelength,
            min_steps_per_wvl=8,
            max_scale=1.2,
        ),
        run_time=1e-15,
        sources=[source],
        monitors=[monitor],
    )

    program = simulation.compile()

    assert simulation.grid.metric_kind_for(("x", "y")) == "rectilinear"
    assert program.config.metric_kind == "rectilinear"
    assert {spec.component for spec in program.sources} == {"Ez", "Hy"}
    assert np.asarray(program.monitors[0].integration_weights).size > 1


@pytest.mark.parametrize(
    ("plane", "expected_offset"),
    (
        ("xy", (-2.0, -3.0, -4.0)),
        ("xz", (-2.0, -4.0, -3.0)),
        ("yz", (-4.0, -2.0, -3.0)),
    ),
)
def test_rectilinear_gaussian_source_uses_physical_edges_and_local_frame(
    plane, expected_offset
):
    grid = Grid(
        [2.0, 2.2, 3.0],
        [3.0, 3.3, 4.0],
        [4.0, 5.0],
    )
    material_grid = MaterialGrid.from_raster_result(
        rasterize(
            Scene((Material(),)),
            grid,
            options=RasterOptions(components="two_dimensional_tm"),
        )
    )
    source = bz.GaussianSource(
        position=(2.2, 3.3),
        width=0.02,
        signal=np.asarray([1.0, 0.0]),
    )
    simulation = bz.Simulation(
        material_grid=material_grid,
        sources=[source],
        plane_2d=plane,
        time=np.asarray([0.0, 1e-16]),
    )

    program = simulation.compile()
    compiled_source = program.sources[0]

    assert program.grid.geometry.origin == (0.0, 0.0, 0.0)
    assert simulation.coordinate_offset == expected_offset
    np.testing.assert_allclose(simulation.sources[0].position, (0.2, 0.3))
    assert compiled_source.slab_starts == (1, 1)
    assert compiled_source.slab_sizes == (1, 1)


@pytest.mark.parametrize(
    ("plane", "polarization", "component", "expected_axes", "expected_offset"),
    (
        ("xy", "tm", "Ez", ("x", "y"), (-2.0, -3.0, -4.0)),
        ("xz", "tm", "Ey", ("x", "z"), (-2.0, -4.0, -3.0)),
        ("yz", "tm", "Ex", ("y", "z"), (-4.0, -2.0, -3.0)),
        ("xy", "te", "Hz", ("x", "y"), (-2.0, -3.0, -4.0)),
        ("xz", "te", "Hy", ("x", "z"), (-2.0, -4.0, -3.0)),
        ("yz", "te", "Hx", ("y", "z"), (-4.0, -2.0, -3.0)),
    ),
)
def test_imported_grid_results_keep_local_metadata_and_public_coordinates(
    plane, polarization, component, expected_axes, expected_offset
):
    grid = Grid(
        [2.0, 2.2, 3.0],
        [3.0, 3.3, 4.0],
        [4.0, 5.0],
    )
    material_grid = MaterialGrid.from_raster_result(
        rasterize(
            Scene((Material(),)),
            grid,
            options=RasterOptions(components=f"two_dimensional_{polarization}"),
        ),
        polarization=polarization,
    )
    recorder = bz.FieldRecorder((component,), interval=1, name="fields")
    simulation = bz.Simulation(
        material_grid=material_grid,
        monitors=[recorder],
        plane_2d=plane,
        polarization=polarization,
        time=np.asarray([0.0, 1e-16]),
    )
    program = simulation.compile()
    from beamz.lattice import component_coordinates_rectilinear

    local_coordinates = component_coordinates_rectilinear(
        component,
        program.grid.geometry,
        plane=plane,
        polarization=polarization,
    )
    monitor_result = bz.MonitorResults(
        monitor=recorder,
        fields={
            component: np.zeros(
                (
                    1,
                    len(local_coordinates[expected_axes[1]]),
                    len(local_coordinates[expected_axes[0]]),
                ),
                dtype=np.float32,
            )
        },
        power_history=np.empty(0),
        power_timestamps=np.empty(0),
        power_spectrum=np.empty(0, dtype=np.complex64),
        field_times=np.asarray([1e-16]),
        field_steps=np.asarray([1]),
    )

    results = bz.SimulationResults.from_run(
        simulation,
        runtime_fields=program.grid,
        monitor_results={"fields": monitor_result},
    )
    dataset = bz.analysis.to_xarray(results)

    from beamz.analysis.plotting import _coord_extent_um

    assert results.metadata.grid == program.grid.geometry
    assert results.metadata.grid.origin == (0.0, 0.0, 0.0)
    assert results.metadata.coordinate_offset == expected_offset
    assert dataset.attrs["coordinate_frame"] == "public"
    expected_x = (
        grid.x_edges
        if polarization == "tm"
        else 0.5 * (grid.x_edges[:-1] + grid.x_edges[1:])
    )
    expected_y = (
        grid.y_edges
        if polarization == "tm"
        else 0.5 * (grid.y_edges[:-1] + grid.y_edges[1:])
    )
    np.testing.assert_allclose(dataset.coords[expected_axes[0]], expected_x)
    np.testing.assert_allclose(dataset.coords[expected_axes[1]], expected_y)
    assert _coord_extent_um(
        dataset[component], sim=results.metadata
    ) == _coord_extent_um(dataset[component], sim=None)


@pytest.mark.parametrize(
    ("plane", "expected_offset"),
    (
        ("xy", (1.0, 2.0, 0.0)),
        ("xz", (1.0, 0.0, 2.0)),
        ("yz", (0.0, 1.0, 2.0)),
    ),
)
@pytest.mark.parametrize("polarization", ("tm", "te"))
def test_centered_2d_devices_round_trip_through_plane_aware_offsets(
    plane, expected_offset, polarization
):
    signal = np.asarray([1.0, 0.0])
    simulation = bz.Simulation(
        domain=(2.0, 4.0),
        sources=[bz.GaussianSource(position=(0.0, 0.0), width=0.1, signal=signal)],
        plane_2d=plane,
        polarization=polarization,
        time=np.asarray([0.0, 1e-16]),
    )

    assert simulation.coordinate_offset == expected_offset
    assert simulation.sources[0].position == (1.0, 2.0)

    replacement = bz.GaussianSource(position=(0.25, -0.5), width=0.1, signal=signal)
    updated = simulation.updated_copy(sources=[replacement])
    assert updated.coordinate_offset == expected_offset
    assert updated.sources[0].position == (1.25, 1.5)

    changed_plane = simulation.updated_copy(plane_2d="xz")
    assert changed_plane.coordinate_offset == (1.0, 0.0, 2.0)
    assert changed_plane.sources[0].position == (1.0, 2.0)


def test_rectilinear_cpml_depth_is_graded_in_physical_distance():
    grid = Grid(
        [0.0, 0.1, 0.3, 0.6, 1.0],
        [0.0, 0.4, 1.0],
        [0.0, 1.0],
    )
    material_grid = MaterialGrid.from_raster_result(
        rasterize(
            Scene((Material(),)),
            grid,
            options=RasterOptions(components="two_dimensional_tm"),
        )
    )
    simulation = bz.Simulation(
        material_grid=material_grid,
        boundaries=[
            bz.PML(
                edges=("left",),
                thickness=0.25,
                sigma_max=10.0,
                formulation="cpml",
            )
        ],
        time=np.asarray([0.0, 1e-16]),
    )

    profile = np.asarray(simulation.compile().grid.pml_data["tm_xy_cpml"]["Hy_x_sigma"])

    np.testing.assert_allclose(profile[:, 0], 10.0 * 0.8**3, atol=1e-7)
    np.testing.assert_allclose(profile[:, 1], 10.0 * 0.2**3, atol=1e-7)
    np.testing.assert_allclose(profile[:, 2:], 0.0)


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


def test_full_tensor_and_full_farjadpour_compile_onto_constitutive_supports():
    tensor_scene = Scene(
        (Material(epsilon_r=((3.0, 0.2, 0.0), (0.2, 2.0, 0.0), (0.0, 0.0, 1.0))),)
    )
    grid = Grid.uniform((0, 0, 0), (1, 1, 1), (2, 2, 2))
    intrinsic = MaterialGrid.from_raster_result(
        rasterize(
            tensor_scene,
            grid,
            options=RasterOptions(smoothing="farjadpour_full"),
        )
    )
    assert intrinsic.uses_full_permittivity

    full = rasterize(
        Scene(
            (Material(), Material(4.0)),
            (
                Object(
                    ExtrudedPolygon(
                        Polygon(((-1.0, -1.0), (2.0, -1.0), (-1.0, 2.0))),
                        -1.0,
                        2.0,
                    ),
                    1,
                ),
            ),
        ),
        grid,
        options=RasterOptions(smoothing="farjadpour_full"),
    )
    material_grid = MaterialGrid.from_raster_result(full)
    simulation = bz.Simulation(
        material_grid=material_grid,
        boundaries=[bz.PML(thickness=0.5, formulation="cpml")],
        run_time=2e-9,
    )
    program = simulation.compile()
    state = simulation.advance().state

    assert material_grid.uses_full_permittivity
    assert program.boundary.cpml.enabled
    assert program.coefficients.e_inverse_diagonal_x.shape == (3, 3, 2)
    assert program.coefficients.e_inverse_diagonal_y.shape == (3, 2, 3)
    assert program.coefficients.e_inverse_diagonal_z.shape == (2, 3, 3)
    assert program.coefficients.e_inverse_offdiagonal.shape == (3, 3, 3, 3)
    assert all(np.all(np.isfinite(getattr(state, name))) for name in ("ex", "ey", "ez"))

    logical_shapes = program.sharding.layout.logical_shapes
    padded_shapes = {
        name: (shape[0] + 1, *shape[1:]) for name, shape in logical_shapes.items()
    }
    lowered = _lower_coefficients(
        program.coefficients,
        ShardingLayout(True, "z", 0, 2, "cpu", logical_shapes, padded_shapes),
    )
    assert lowered.e_inverse_diagonal_x.shape == padded_shapes["Ex"]
    assert lowered.e_inverse_offdiagonal.shape == (4, 3, 3, 3)

    with pytest.raises(ValueError, match="use CPML"):
        bz.Simulation(
            material_grid=material_grid,
            boundaries=[bz.PML(thickness=0.5, formulation="sponge")],
            run_time=2e-9,
        ).compile()


def test_rectilinear_full_tensor_update_executes_combined_kernel():
    result = rasterize(
        Scene(
            (
                Material(
                    epsilon_r=(
                        (3.0, 0.2, 0.0),
                        (0.2, 2.0, 0.0),
                        (0.0, 0.0, 1.0),
                    )
                ),
            )
        ),
        Grid(
            [0.0, 0.2, 1.0],
            [0.0, 0.4, 1.0],
            [0.0, 0.3, 1.0],
        ),
        options=RasterOptions(smoothing="farjadpour_full"),
    )
    material_grid = MaterialGrid.from_raster_result(result)
    simulation = bz.Simulation(
        material_grid=material_grid,
        time=np.asarray([0.0, 1e-16]),
    )
    state = simulation.initial_state()
    state = state._replace(hx=np.ones_like(state.hx))

    program = simulation.compile()
    advanced = simulation.step(state)

    assert material_grid.uses_full_permittivity
    assert program.config.metric_kind == "rectilinear"
    assert program.coefficients.e_inverse_offdiagonal.size > 0
    assert all(
        np.isfinite(getattr(advanced, name)).all() for name in ("ex", "ey", "ez")
    )


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
