import numpy as np

from beamz import Circle, Design, Material
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
