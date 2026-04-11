import numpy as np
from pathlib import Path

from beamz.simulation.snapshots import run_with_snapshots
from beamz.visual.data import signal_plot_data


class _DummyDesign:
    def __init__(self):
        self.width = 4.0
        self.height = 3.0
        self.depth = 0.0
        self.is_3d = False
        self.structures = []

    def copy(self):
        return self

    def unify_polygons(self):
        return None


class _DummyFields:
    def __init__(self):
        self.Ez = np.ones((2, 3))

    def available_components(self):
        return ["Ez"]


class _DummySim:
    def __init__(self):
        self.fields = _DummyFields()
        self.design = _DummyDesign()
        self.sources = []
        self.monitors = []
        self.boundaries = []
        self.plane_2d = "xy"
        self.current_step = 0
        self.num_steps = 1
        self.t = 0.0

    def step(self):
        if self.current_step >= self.num_steps:
            return False
        self.current_step += 1
        self.t = 1.5e-15
        return True


def test_run_with_snapshots_collects_layout_and_field_payload():
    seen = []
    sim = _DummySim()

    results = run_with_snapshots(
        sim,
        snapshot_field="Ez",
        snapshot_interval=1,
        snapshot_callback=seen.append,
        store_snapshots=True,
    )

    assert len(seen) == 1
    snapshot = seen[0]
    assert snapshot["field_name"] == "Ez"
    assert snapshot["units"] == "V/µm"
    assert snapshot["layout"]["design"]["width"] == 4.0
    assert np.allclose(snapshot["field"], np.ones((2, 3)) * 1e-6)

    assert results is not None
    assert len(results.snapshots) == 1
    assert results["snapshots"][0]["step"] == 1


def test_signal_plot_data_scales_picoseconds():
    payload = signal_plot_data(np.array([0.0, 1.0]), np.array([0.0, 2.0e-12]))

    assert payload["time_unit"] == "ps"
    assert np.allclose(payload["t_scaled"], np.array([0.0, 2.0]))


def test_beamz_source_tree_contains_no_matplotlib_imports():
    root = Path(__file__).resolve().parents[1] / "beamz"
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text()
        if "import matplotlib" in text or "from matplotlib" in text:
            offenders.append(path.relative_to(root).as_posix())

    assert offenders == []
