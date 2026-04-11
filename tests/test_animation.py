import numpy as np
from pathlib import Path

from beamz import (
    LIGHT_SPEED,
    PML,
    Design,
    GaussianSource,
    Material,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    um,
)
from beamz.visual.data import signal_plot_data


def _make_snapshot_sim():
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=2, safety_factor=0.95, points_per_wavelength=8
    )
    domain = 4.0 * wl
    steps = 12
    t = np.arange(0, steps * dt, dt)
    freq = LIGHT_SPEED / wl
    signal = ramped_cosine(
        t,
        amplitude=1.0,
        frequency=freq,
        ramp_duration=2 / freq,
        t_max=t[-1] * 0.4,
    )
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    source = GaussianSource(position=(domain / 2, domain / 2), width=wl / 6, signal=signal)
    return Simulation(
        design=design,
        sources=[source],
        boundaries=[PML(thickness=1.0 * wl)],
        time=t,
        resolution=dx,
    )


def test_simulation_run_collects_snapshot_layout_and_field_payload():
    seen = []
    sim = _make_snapshot_sim()

    results = sim.run(
        snapshot_field="Ez",
        snapshot_interval=4,
        snapshot_callback=seen.append,
        store_snapshots=True,
        progress=False,
    )

    assert len(seen) == 3
    snapshot = seen[0]
    assert snapshot["field_name"] == "Ez"
    assert snapshot["units"] == "V/µm"
    assert snapshot["layout"]["design"]["width"] == sim.design.width
    assert np.max(np.abs(snapshot["field"])) > 0.0

    assert results is not None
    assert len(results.snapshots) == 3
    assert results["snapshots"][0]["step"] == 4


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
