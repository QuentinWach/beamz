from contextlib import AbstractContextManager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import LinearSegmentedColormap

import beamz as bz
from beamz.analysis import video as video_module
from beamz.devices._placement import snap_plane_region_grid
from beamz.simulation.results import FieldMetadata, SimulationMetadata


def _results(*, names=("fields",)):
    frames = np.asarray(
        [
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 1.0]],
            [[0.0, 2.0, 0.0], [-2.0, 0.0, 2.0]],
            [[0.0, 3.0, 0.0], [-3.0, 0.0, 3.0]],
        ],
        dtype=np.float32,
    )
    metadata = SimulationMetadata(
        dt=0.25e-15,
        resolution=0.5 * bz.um,
        is_3d=False,
        plane_2d="xy",
        coordinate_offset=(0.0, 0.0, 0.0),
        time=np.arange(4) * 0.25e-15,
        width=1.5 * bz.um,
        height=1.0 * bz.um,
        depth=0.0,
        fields=FieldMetadata((2, 3), {"Ez": (2, 3)}),
    )
    monitors = {}
    for name in names:
        monitors[name] = bz.MonitorResults(
            monitor=bz.FieldRecorder(("Ez",), interval=1, name=name),
            fields={"Ez": frames},
            power_history=np.empty(0),
            power_timestamps=np.empty(0),
            power_spectrum=np.empty(0, dtype=np.complex64),
            field_times=np.asarray([0.25e-15, 0.5e-15, 0.75e-15]),
            field_steps=np.asarray([1, 2, 3]),
        )
    return bz.SimulationResults(metadata, monitors)


def _results_3d():
    frames = np.arange(2 * 2 * 4 * 5, dtype=np.float32).reshape(2, 2, 4, 5)
    component_shapes = {
        "Ex": (3, 4, 4),
        "Ey": (3, 3, 5),
        "Ez": (2, 4, 5),
        "Hx": (2, 3, 5),
        "Hy": (2, 4, 4),
        "Hz": (3, 3, 4),
    }
    metadata = SimulationMetadata(
        dt=0.25e-15,
        resolution=0.5 * bz.um,
        is_3d=True,
        plane_2d="xy",
        coordinate_offset=(0.0, 0.0, 0.0),
        time=np.arange(3) * 0.25e-15,
        width=2.0 * bz.um,
        height=1.5 * bz.um,
        depth=1.0 * bz.um,
        fields=FieldMetadata((2, 3, 4), component_shapes),
    )
    recording = bz.MonitorResults(
        monitor=bz.FieldRecorder(("Ez",), interval=1, name="volume"),
        fields={"Ez": frames},
        power_history=np.empty(0),
        power_timestamps=np.empty(0),
        power_spectrum=np.empty(0, dtype=np.complex64),
        field_times=np.asarray([0.25e-15, 0.5e-15]),
        field_steps=np.asarray([1, 2]),
    )
    return bz.SimulationResults(metadata, {"volume": recording})


class _FakeWriter(AbstractContextManager):
    captured = []

    def __init__(self, fps, codec, bitrate, extra_args):
        self.fps = fps
        self.codec = codec
        self.bitrate = bitrate
        self.extra_args = extra_args

    @classmethod
    def isAvailable(cls):
        return True

    def saving(self, fig, filename, dpi):
        self.fig = fig
        self.filename = filename
        self.dpi = dpi
        self.pixel_size = tuple(
            int(round(value * dpi)) for value in fig.get_size_inches()
        )
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def grab_frame(self):
        image = self.fig.axes[0].images[0]
        self.captured.append(
            {
                "filename": self.filename,
                "fps": self.fps,
                "dpi": self.dpi,
                "codec": self.codec,
                "extra_args": self.extra_args,
                "pixel_size": self.pixel_size,
                "field": np.asarray(image.get_array()).copy(),
                "clim": image.get_clim(),
                "title": self.fig.axes[0].get_title(),
            }
        )


def test_save_field_video_writes_every_recorded_frame_with_global_limits(
    monkeypatch,
):
    _FakeWriter.captured = []
    monkeypatch.setattr(
        video_module,
        "_mpl_types",
        lambda: (plt, _FakeWriter, LinearSegmentedColormap),
    )

    output = bz.analysis.save_field_video(
        _results(),
        "fields.mp4",
        field="Ez",
        fps=12,
        dpi=90,
        cmap="RdBu",
        cmap_limits="global",
        interpolation="nearest",
        colorbar=False,
    )

    assert output == Path("fields.mp4")
    assert len(_FakeWriter.captured) == 3
    assert [frame["fps"] for frame in _FakeWriter.captured] == [12, 12, 12]
    assert [frame["dpi"] for frame in _FakeWriter.captured] == [90, 90, 90]
    assert [frame["codec"] for frame in _FakeWriter.captured] == [
        "libx264",
        "libx264",
        "libx264",
    ]
    assert _FakeWriter.captured[0]["pixel_size"] == (630, 496)
    assert all(size % 2 == 0 for size in _FakeWriter.captured[0]["pixel_size"])
    assert _FakeWriter.captured[0]["extra_args"] == [
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
    ]
    assert [frame["clim"] for frame in _FakeWriter.captured] == [
        (-3.0, 3.0),
        (-3.0, 3.0),
        (-3.0, 3.0),
    ]
    np.testing.assert_array_equal(
        _FakeWriter.captured[-1]["field"],
        _results()["fields"].fields["Ez"][-1],
    )
    assert _FakeWriter.captured[-1]["title"] == "Ez at t = 0.75 fs"


def test_save_field_video_requires_monitor_name_when_recorders_are_ambiguous():
    with pytest.raises(ValueError, match="pass monitor_name"):
        bz.analysis.save_field_video(_results(names=("a", "b")), "fields.mp4")


def test_save_field_video_selects_a_full_domain_3d_slice(monkeypatch):
    _FakeWriter.captured = []
    monkeypatch.setattr(
        video_module,
        "_mpl_types",
        lambda: (plt, _FakeWriter, LinearSegmentedColormap),
    )
    results = _results_3d()

    bz.analysis.save_field_video(
        results,
        "volume.mp4",
        field="Ez",
        plane="y",
        index=1,
        cmap="RdBu",
        cmap_limits="global",
        colorbar=False,
    )

    assert len(_FakeWriter.captured) == 2
    np.testing.assert_array_equal(
        _FakeWriter.captured[1]["field"],
        results["volume"].fields["Ez"][1, :, 1, :],
    )
    assert _FakeWriter.captured[1]["title"].endswith(", y=0.5 um")


@pytest.mark.parametrize("retain_compiled_region", (True, False))
def test_slice_video_uses_exact_rectilinear_extents_and_snapped_position(
    retain_compiled_region,
):
    grid = bz.RectilinearGrid(
        np.asarray([0.0, 0.2, 1.0]) * bz.um,
        np.asarray([0.0, 0.3, 1.0]) * bz.um,
        np.asarray([0.0, 0.1, 0.4, 1.0]) * bz.um,
    )
    monitor = bz.FieldRecorder(
        ("Ez",),
        interval=1,
        name="slice",
        center=(0.5 * bz.um, 0.26 * bz.um, 0.5 * bz.um),
        size=(1.0 * bz.um, 0.0, 1.0 * bz.um),
    )
    region = snap_plane_region_grid(
        center=monitor.center,
        size=monitor.size,
        plane_normal="y",
        grid=grid,
    )
    metadata = SimulationMetadata(
        dt=0.25e-15,
        resolution=0.1 * bz.um,
        is_3d=True,
        plane_2d="xy",
        coordinate_offset=(-2.0 * bz.um, -3.0 * bz.um, -4.0 * bz.um),
        time=np.asarray([0.0, 0.25e-15]),
        width=1.0 * bz.um,
        height=1.0 * bz.um,
        depth=1.0 * bz.um,
        fields=FieldMetadata((3, 2, 2), {"Ez": (3, 3, 3)}),
        grid=grid,
    )
    recording = bz.MonitorResults(
        monitor=monitor,
        fields={"Ez": np.zeros((1, 3, 2), dtype=np.float32)},
        power_history=np.empty(0),
        power_timestamps=np.empty(0),
        power_spectrum=np.empty(0, dtype=np.complex64),
        field_times=np.asarray([0.25e-15]),
        field_steps=np.asarray([1]),
        sample_region=region if retain_compiled_region else None,
    )
    results = bz.SimulationResults(metadata, {"slice": recording})

    data = video_module._video_data(
        results, recording, field="Ez", plane="z", index=None
    )

    assert data.extent == pytest.approx((2.0, 3.0, 4.0, 5.0))
    assert data.horizontal == "x"
    assert data.vertical == "z"
    assert data.slice_label == "y=3.15 um"


def test_save_field_video_rejects_conflicting_color_limits():
    with pytest.raises(ValueError, match="either cmap_limits='global' or vmin/vmax"):
        bz.analysis.save_field_video(
            _results(),
            "fields.mp4",
            cmap_limits="global",
            vmin=-1.0,
        )
