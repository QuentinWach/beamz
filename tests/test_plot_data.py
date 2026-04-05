import numpy as np

from beamz import Circle, Design, Material
from beamz.devices.monitors.monitors import Monitor
from beamz.devices.sources.mode import ModeSource
from beamz.simulation.core import Simulation
from beamz.visual.animation import animate_manual_field
from beamz.visual.runner import VizConfig, _update_live_display
from beamz.visual.data import Slice2D, Trace1D


def test_slice2d_plot_scales_extent_labels():
    sl = Slice2D(
        values=np.ones((2, 3)),
        extent=(0.0, 4e-6, 0.0, 2e-6),
        value_label="permittivity",
        plane="xy",
        title="slice",
    )

    ax = sl.plot(colorbar=False)

    assert ax.get_xlabel() == "x (µm)"
    assert ax.get_ylabel() == "y (µm)"
    assert ax.get_title() == "slice"


def test_trace1d_plot_scales_coordinate_labels():
    trace = Trace1D(
        values=np.array([0.0, 1.0, 0.0]),
        coords=np.array([0.0, 1e-15, 2e-15]),
        coord_label="time",
        value_label="amplitude",
        title="trace",
    )

    ax = trace.plot()

    assert ax.get_xlabel() == "time (fs)"
    assert ax.get_ylabel() == "amplitude"
    assert ax.get_title() == "trace"


def test_design_slice2d_returns_rasterized_permittivity(monkeypatch):
    monkeypatch.setenv("BEAMZ_RASTER_CACHE", "0")
    monkeypatch.setenv("BEAMZ_RASTER_TIMING", "0")

    design = Design(width=2.0, height=2.0, material=Material(permittivity=1.0))
    design += Circle(
        position=(1.0, 1.0),
        radius=0.5,
        material=Material(permittivity=12.0),
    )

    sl = design.slice2d(resolution=0.25)

    assert isinstance(sl, Slice2D)
    assert sl.values.shape == (8, 8)
    assert sl.extent == (0.0, 2.0, 0.0, 2.0)
    assert np.max(sl.values) > 1.0


def test_monitor_field_snapshot_returns_trace_or_slice():
    line = Monitor(start=(0.0, 0.0), end=(2.0, 0.0), record_fields=True)
    line.fields["Ez"] = [np.array([0.0, 1.0, 0.5])]
    line.fields["t"] = [1e-15]
    trace = line.field_snapshot("Ez")
    assert isinstance(trace, Trace1D)
    assert trace.coords.shape == (3,)

    plane = Monitor(
        start=(0.0, 0.0, 0.1),
        end=(2.0, 1.0, 0.1),
        record_fields=True,
        name="plane",
    )
    plane.fields["Ez"] = [np.ones((4, 8))]
    plane.fields["t"] = [2e-15]
    sl = plane.field_snapshot("Ez")
    assert isinstance(sl, Slice2D)
    assert sl.extent == (0.0, 2.0, 0.0, 1.0)


def test_monitor_power_trace_and_flux_trace_return_trace_data():
    mon = Monitor(start=(0.0, 0.0), end=(1.0, 0.0), record_fields=True)
    mon.power_history = [1.0, 2.0, 4.0]
    mon.power_timestamps = [0.0, 1e-15, 2e-15]
    power = mon.power_trace(db_scale=True)
    assert isinstance(power, Trace1D)
    assert power.value_label == "Power (dB)"

    mon.fields["Ez"] = [np.array([1.0]), np.array([2.0])]
    mon.fields["Hy"] = [np.array([0.5]), np.array([0.25])]
    mon.fields["t"] = [0.0, 1e-15]
    flux = mon.flux_trace("+x")
    assert isinstance(flux, Trace1D)
    assert flux.coords.shape == (2,)


def test_regular_grid_slice2d_returns_plot_data(monkeypatch):
    monkeypatch.setenv("BEAMZ_RASTER_CACHE", "0")
    monkeypatch.setenv("BEAMZ_RASTER_TIMING", "0")

    design = Design(width=2.0, height=2.0, material=Material(permittivity=1.0))
    design += Circle(
        position=(1.0, 1.0),
        radius=0.5,
        material=Material(permittivity=12.0),
    )
    grid = design.rasterize(resolution=0.25)

    sl = grid.slice2d(field="permittivity")
    assert isinstance(sl, Slice2D)
    assert sl.extent == (0.0, 2.0, 0.0, 2.0)


def test_simulation_monitor_trace_returns_trace_data():
    mon = Monitor(start=(0.0, 0.0), end=(1.0, 0.0), record_fields=True, name="m")
    mon.fields["Ez"] = [np.array([0.0, 2.0]), np.array([1.0, 3.0])]
    mon.fields["t"] = [0.0, 1e-15]

    sim = Simulation.__new__(Simulation)
    sim.dt = 1e-15
    sim.time = np.array([0.0, 1e-15])

    trace = sim.monitor_trace(mon, field_component="Ez", reduction="mean")
    assert isinstance(trace, Trace1D)
    np.testing.assert_allclose(trace.values, np.array([1.0, 2.0]))


def test_mode_source_profile_data_returns_plot_data():
    src = ModeSource.__new__(ModeSource)
    src.grid = None
    src.direction = "+x"
    src._neff = 2.1
    src._Ez_profile = np.ones((4, 6))
    src._jz_profile = None

    sl = src.profile_data()
    assert isinstance(sl, Slice2D)
    assert sl.plane == "yz"

    src._Ez_profile = None
    src._jz_profile = np.array([0.0, 1.0, 0.5])
    trace = src.profile_data("Hz")
    assert isinstance(trace, Trace1D)
    np.testing.assert_allclose(trace.values, np.array([0.0, 1.0, 0.5]))


def test_animate_manual_field_marks_closed_figure():
    import matplotlib.pyplot as plt

    ctx = animate_manual_field(
        np.ones((4, 4)),
        pause=0.0,
        title="Ez at t = 1.00e-15 s",
    )
    plt.close(ctx["fig"])

    ctx = animate_manual_field(
        np.zeros((4, 4)),
        context=ctx,
        pause=0.0,
        title="Ez at t = 2.00e-15 s",
    )

    assert ctx["closed"] is True


def test_runner_disables_live_updates_after_window_close(monkeypatch):
    sim = Simulation.__new__(Simulation)
    sim.is_3d = False
    sim.fields = type(
        "Fields",
        (),
        {
            "Ez": np.ones((4, 4)),
            "available_components": lambda self: ["Ez"],
        },
    )()
    sim.design = type("Design", (), {"width": 2.0, "height": 1.0})()
    sim.boundaries = []
    sim.plane_2d = "xy"
    sim.t = 1e-15
    sim.current_step = 1
    sim.num_steps = 10

    cfg = VizConfig(animate_live="Ez")

    def fake_animate_manual_field(*args, **kwargs):
        return {"closed": True}

    monkeypatch.setattr(
        "beamz.visual.animation.animate_manual_field", fake_animate_manual_field
    )

    ctx = _update_live_display(
        sim,
        cfg,
        active_monitor=None,
        use_jupyter=False,
        jupyter_animator=None,
        viz_context=None,
    )

    assert ctx["closed"] is True
    assert cfg.animate_live is None
