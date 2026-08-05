import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import beamz as bz
from beamz.analysis.plotting import extract_axis_aligned_slice, plot_field_view
from beamz.design import MaterialGrid
from beamz.design.raster import Grid, Material, Scene, rasterize


def test_generic_slice_and_field_primitives_preserve_axis_coordinates():
    values = np.arange(4 * 5 * 6).reshape(4, 5, 6)
    section = extract_axis_aligned_slice(
        values,
        axis="y",
        step=0.5,
        position=1.0,
        lengths={"x": 3.0, "y": 2.5, "z": 2.0},
    )

    np.testing.assert_array_equal(section.values, values[:, 2, :])
    assert (section.vertical, section.horizontal) == ("z", "x")
    assert section.extent == (0.0, 3.0, 0.0, 2.0)

    fig, ax = plt.subplots()
    try:
        image, view = plot_field_view(ax, section.values * (1.0 + 1.0j), val="abs^2")
        np.testing.assert_allclose(image.get_array(), 2.0 * section.values**2)
        assert view.magnitude and view.power
    finally:
        plt.close(fig)


def test_3d_simulation_plot_uses_tidy_layout_cross_sections():
    design = bz.Design(background=bz.Material(2.25))
    design += bz.Box(
        center=(0.0, 0.0, 0.0),
        size=(1.0 * bz.um, 1.0 * bz.um, 0.4 * bz.um),
        material=bz.Material(12.0),
    )
    source = bz.ModeSource(
        center=(-1.0 * bz.um, 0.0, 0.0),
        size=(0.0, 1.0 * bz.um, 0.8 * bz.um),
        source_time=bz.GaussianPulse(
            freq0=bz.LIGHT_SPEED / (1.55 * bz.um), fwidth=2e13
        ),
        direction="+",
    )
    monitor = bz.FluxMonitor(
        center=(1.0 * bz.um, 0.0, 0.0),
        size=(0.0, 1.0 * bz.um, 0.8 * bz.um),
        freqs=[bz.LIGHT_SPEED / (1.55 * bz.um)],
        name="flux",
    )
    sim = bz.Simulation(
        domain=(4.0 * bz.um, 4.0 * bz.um, 2.0 * bz.um),
        design=design,
        sources=[source],
        monitors=[monitor],
        boundaries=[bz.PML(thickness=0.5 * bz.um)],
        resolution=0.5 * bz.um,
        time=np.array([0.0, 1e-15]),
    )

    with pytest.warns(RuntimeWarning, match="PML material varies"):
        fig, axes = sim.plot(z=0.0, y=0.0, show=False)

    try:
        assert len(axes) == 2
        assert len(fig.axes) == 2
        assert len(axes[0].images) == 1
        assert len(axes[1].images) == 1
        assert len(axes[0].patches) >= 4
        assert len(axes[1].patches) >= 4
        assert len(axes[0].lines) >= 2
        assert len(axes[1].lines) >= 2
        assert axes[0].get_xlim() == (-2.0, 2.0)
        assert axes[1].get_xlim() == (-2.0, 2.0)
        assert axes[0].get_ylim() == (-2.0, 2.0)
        assert axes[1].get_ylim() == (-1.0, 1.0)
        assert axes[0].get_xlabel() == "x (um)"
        assert axes[1].get_ylabel() == "z (um)"
    finally:
        plt.close(fig)

    fig, axes = sim.view3d(show=False)

    try:
        assert axes[0].get_title() == "cross section at z=0.00 (um)"
        assert axes[1].get_title() == "cross section at y=0.00 (um)"
    finally:
        plt.close(fig)


def test_3d_simulation_plot_clips_x_pml_overlay_to_active_vertical_span():
    sim = bz.Simulation(
        domain=(4.0 * bz.um, 4.0 * bz.um, 2.0 * bz.um),
        design=bz.Design(background=bz.Material(2.25)),
        boundaries=[bz.PML(thickness=0.5 * bz.um)],
        resolution=0.5 * bz.um,
        time=np.array([0.0, 1e-15]),
    )

    fig, axes = sim.plot(z=0.0, y=0.0, show=False)

    try:
        xy_pml = [patch for patch in axes[0].patches if patch.get_hatch() == "xx"]
        xz_pml = [patch for patch in axes[1].patches if patch.get_hatch() == "xx"]

        assert len(xy_pml) == 4
        assert len(xz_pml) == 4

        xy_left = min(
            (patch for patch in xy_pml if patch.get_width() == pytest.approx(0.5)),
            key=lambda patch: patch.get_x(),
        )
        xz_left = min(
            (patch for patch in xz_pml if patch.get_width() == pytest.approx(0.5)),
            key=lambda patch: patch.get_x(),
        )

        assert xy_left.get_x() == pytest.approx(-2.0)
        assert xy_left.get_y() == pytest.approx(-1.5)
        assert xy_left.get_height() == pytest.approx(3.0)
        assert xz_left.get_x() == pytest.approx(-2.0)
        assert xz_left.get_y() == pytest.approx(-0.5)
        assert xz_left.get_height() == pytest.approx(1.0)
    finally:
        plt.close(fig)


def test_3d_plot_keeps_square_limits_for_source_added_by_copy_update():
    sim0 = bz.Simulation(
        domain=(4.0 * bz.um, 4.0 * bz.um, 2.0 * bz.um),
        design=bz.Design(background=bz.Material(2.25)),
        sources=[],
        monitors=[],
        boundaries=[bz.PML(thickness=0.5 * bz.um)],
        resolution=0.5 * bz.um,
        time=np.array([0.0, 1e-15]),
    )
    source = bz.ModeSource(
        center=(-1.0 * bz.um, 0.0, 0.0),
        size=(0.0, 1.0 * bz.um, 0.8 * bz.um),
        source_time=bz.GaussianPulse(
            freq0=bz.LIGHT_SPEED / (1.55 * bz.um), fwidth=2e13
        ),
        direction="+",
    )

    sim = sim0.updated_copy(sources=[source])
    fig, axes = sim.plot(z=0.0, y=0.0, show=False)

    try:
        assert sim.sources[0].center == pytest.approx(
            (1.0 * bz.um, 2.0 * bz.um, 1.0 * bz.um)
        )
        assert source.center == pytest.approx((-1.0 * bz.um, 0.0, 0.0))
        assert axes[0].get_xlim() == pytest.approx((-2.0, 2.0))
        assert axes[0].get_ylim() == pytest.approx((-2.0, 2.0))
        source_line = next(
            line for line in axes[0].lines if line.get_color() == "#66bb6a"
        )
        np.testing.assert_allclose(source_line.get_xdata(), [-1.0, -1.0])
    finally:
        plt.close(fig)


def test_monitor_dft_field_plot_restores_tidy_plane_view():
    freq0 = 2.0e14
    design = bz.Design(background=bz.Material(2.25))
    design += bz.Box(
        center=(0.0, 0.0, 0.0),
        size=(1.0 * bz.um, 1.0 * bz.um, 0.4 * bz.um),
        material=bz.Material(12.0),
    )
    monitor = bz.FieldMonitor(
        center=(0.0, 0.0, 0.0),
        size=(4.0 * bz.um, 4.0 * bz.um, 0.0),
        freqs=[freq0],
        fields=("Ex", "Ey", "Ez"),
        name="field",
    )
    sim = bz.Simulation(
        domain=(4.0 * bz.um, 4.0 * bz.um, 2.0 * bz.um),
        design=design,
        sources=[],
        monitors=[monitor],
        # A non-integral cell count makes the staggered monitor plane one
        # sample larger per in-plane axis than the material grid. This mirrors
        # the modal_sources_monitors notebook regression.
        resolution=0.45 * bz.um,
        time=np.array([0.0, 1e-15]),
    )
    monitor = sim.monitors[0]
    fields = sim.compile().grid
    component_shapes = tuple(
        tuple(int(v) for v in getattr(fields, name).shape)
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    )
    monitor_base_shape = tuple(
        max(shape[axis] for shape in component_shapes) for axis in range(3)
    )
    coords0, coords1 = monitor.get_analysis_plane_coords_3d(
        dx=sim.resolution,
        dy=sim.resolution,
        dz=sim.resolution,
        field_shape=monitor_base_shape,
    )
    npoints = int(coords0.size * coords1.size)
    monitor_result = bz.MonitorResults(
        monitor=monitor,
        fields={},
        power_history=np.asarray([], dtype=float),
        power_timestamps=np.asarray([], dtype=float),
        power_spectrum=np.asarray([], dtype=np.complex64),
        dft_fields={
            "Ex": np.full((1, npoints), 10.0 / bz.um, dtype=np.complex128),
            "Ey": np.zeros((1, npoints), dtype=np.complex128),
            "Ez": np.zeros((1, npoints), dtype=np.complex128),
        },
        dft_frequencies=np.asarray([freq0]),
        dft_weight_sum=np.array([2.0]),
        dft_base_dt=0.0,
        resolution=float(sim.resolution),
    )
    results = bz.SimulationResults.from_run(
        sim,
        runtime_fields=fields,
        monitor_results={"field": monitor_result},
    )

    fig, ax = results.plot_field(
        field_monitor_name="field",
        field_name="E",
        val="abs^2",
        f=freq0,
        vmin=0,
        vmax=3000,
        cmap="magma",
        figsize=(6, 5),
        show=False,
    )

    try:
        image = ax.images[0].get_array()
        assert image.shape == (coords0.size, coords1.size)
        assert np.nanmax(image) == 100.0
        assert ax.get_xlabel() == "x (um)"
        assert ax.get_ylabel() == "y (um)"
        assert ax.get_xlim() == (-2.0, 2.0)
        assert ax.get_ylim() == (-2.0, 2.0)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ("normal", "center", "size", "expected_x_edges", "expected_y_edges"),
    (
        ("x", (0.1, 0.5, 0.5), (0.0, 1.0, 1.0), (0.0, 0.3, 1.0), (0.0, 0.4, 1.0)),
        ("y", (0.5, 0.15, 0.5), (1.0, 0.0, 1.0), (0.0, 0.2, 1.0), (0.0, 0.4, 1.0)),
        ("z", (0.5, 0.5, 0.2), (1.0, 1.0, 0.0), (0.0, 0.2, 1.0), (0.0, 0.3, 1.0)),
    ),
)
def test_monitor_dft_field_plot_uses_exact_nonuniform_plane_coordinates(
    normal, center, size, expected_x_edges, expected_y_edges
):
    grid = Grid(
        np.asarray([0.0, 0.2, 1.0]) * bz.um,
        np.asarray([0.0, 0.3, 1.0]) * bz.um,
        np.asarray([0.0, 0.4, 1.0]) * bz.um,
    )
    material_grid = MaterialGrid.from_raster_result(
        rasterize(Scene((Material(),)), grid), dimensions=3
    )
    monitor = bz.FieldMonitor(
        center=tuple(value * bz.um for value in center),
        size=tuple(value * bz.um for value in size),
        freqs=[1.0],
        fields=("Ez",),
        name="field",
    )
    simulation = bz.Simulation(
        material_grid=material_grid,
        monitors=[monitor],
        time=np.asarray([0.0, 1e-16]),
    )
    program = simulation.compile()
    spec = program.monitors[0]
    from beamz.simulation.results import material_region_for_monitor

    result = bz.MonitorResults(
        monitor=simulation.monitors[0],
        fields={},
        power_history=np.empty(0),
        power_timestamps=np.empty(0),
        power_spectrum=np.empty(0, dtype=np.complex64),
        dft_fields={"Ez": np.ones((1, spec.dft_point_count), dtype=np.complex128)},
        dft_frequencies=np.asarray([1.0]),
        dft_weight_sum=np.ones(1),
        resolution=simulation.resolution,
        sample_region=spec.sample_region,
        material_region=material_region_for_monitor(
            simulation,
            simulation.monitors[0],
            runtime_fields=program.grid,
        ),
    )
    results = bz.SimulationResults.from_run(
        simulation,
        runtime_fields=program.grid,
        monitor_results={"field": result},
    )

    fig, ax = results.plot_field(
        field_monitor_name="field",
        field_name="Ez",
        show=False,
    )

    try:
        assert not ax.images
        meshes = [
            collection
            for collection in ax.collections
            if hasattr(collection, "get_coordinates")
        ]
        assert len(meshes) == 2
        assert len(ax.collections) == 4
        for mesh in meshes:
            assert mesh.get_array().shape[:2] == (2, 2)
            coordinates = mesh.get_coordinates()
            np.testing.assert_allclose(coordinates[0, :, 0], expected_x_edges)
            np.testing.assert_allclose(coordinates[:, 0, 1], expected_y_edges)
        assert ax.get_xlim() == pytest.approx((0.0, 1.0))
        assert ax.get_ylim() == pytest.approx((0.0, 1.0))
    finally:
        plt.close(fig)


def test_flux_result_is_finite_for_notebook_style_line_plot():
    freqs = np.linspace(1.9e14, 2.1e14, 5)
    ldas = np.linspace(1.26, 1.36, freqs.size)
    monitor = bz.FluxMonitor(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 2.0 * bz.um, 2.0 * bz.um),
        freqs=freqs,
        name="flux",
    )
    npoints = 4
    result = bz.MonitorResults(
        monitor=monitor,
        fields={},
        power_history=np.asarray([], dtype=float),
        power_timestamps=np.asarray([], dtype=float),
        power_spectrum=np.asarray([], dtype=np.complex64),
        dft_fields={
            "Ex": np.zeros((freqs.size, npoints), dtype=np.complex128),
            "Ey": np.ones((freqs.size, npoints), dtype=np.complex128),
            "Ez": np.zeros((freqs.size, npoints), dtype=np.complex128),
            "Hx": np.zeros((freqs.size, npoints), dtype=np.complex128),
            "Hy": np.zeros((freqs.size, npoints), dtype=np.complex128),
            "Hz": np.ones((freqs.size, npoints), dtype=np.complex128),
        },
        dft_frequencies=freqs,
        dft_weight_sum=np.full(freqs.size, 2.0),
        dft_base_dt=0.0,
        resolution=1.0 * bz.um,
        power_scale=bz.um**2,
    )

    fig, ax = plt.subplots()
    try:
        flux_db = 10 * np.log10(result.flux)
        lines = ax.plot(ldas, flux_db, lw=3)
        ydata = np.asarray(lines[0].get_ydata(), dtype=float)
        assert len(lines) == 1
        assert ydata.shape == ldas.shape
        assert np.all(np.isfinite(ydata))
    finally:
        plt.close(fig)
