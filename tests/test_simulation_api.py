from contextlib import contextmanager

import numpy as np
import pytest

import beamz.simulation.core as simulation_core
from beamz import (
    Design,
    GaussianSource,
    Material,
    Monitor,
    MonitorResults,
    Simulation,
    SimulationResults,
    um,
)

pytestmark = pytest.mark.unit


def _time_axis():
    return np.array([0.0, 1e-16, 2e-16], dtype=float)


def test_simulation_accepts_explicit_sources_and_monitors():
    design = Design(width=4 * um, height=4 * um, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(2 * um, 2 * um), width=0.2 * um, signal=[1.0, 0.0]
    )
    monitor = Monitor(start=(1 * um, 1 * um), end=(1 * um, 3 * um), name="m1")

    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[monitor],
        time=_time_axis(),
        resolution=0.2 * um,
    )

    assert sim.sources == [source]
    assert sim.monitors == [monitor]


def test_simulation_accepts_rasterized_2d_grid_as_design():
    design = Design(width=2 * um, height=2 * um, material=Material(permittivity=1.0))
    grid = design.rasterize(resolution=0.5 * um)
    source = GaussianSource(
        position=(1 * um, 1 * um), width=0.2 * um, signal=[1.0, 0.0]
    )

    sim = Simulation(
        design=grid,
        sources=[source],
        monitors=[],
        time=_time_axis(),
        resolution=0.5 * um,
    )

    assert sim.domain == (2 * um, 2 * um, 0.0)
    assert not sim.is_3d


def test_simulation_rejects_duplicate_named_monitors():
    design = Design(width=4 * um, height=4 * um, material=Material(permittivity=1.0))
    m1 = Monitor(start=(1 * um, 1 * um), end=(1 * um, 3 * um), name="port")
    m2 = Monitor(start=(2 * um, 1 * um), end=(2 * um, 3 * um), name="port")

    with pytest.raises(ValueError, match="duplicate.*port|port.*duplicate"):
        Simulation(
            design=design,
            monitors=[m1, m2],
            time=_time_axis(),
            resolution=0.2 * um,
        )


def test_simulation_setup_device_cpu_runs_compiled():
    design = Design(width=2 * um, height=2 * um, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(1 * um, 1 * um), width=0.2 * um, signal=[1.0, 0.0, 0.0]
    )
    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[],
        time=np.array([0.0, 1e-16, 2e-16, 3e-16], dtype=float),
        resolution=0.5 * um,
        setup_device="cpu",
    )

    assert sim.setup_device_policy == "cpu"
    assert sim.setup_device_resolved in {"cpu", "default"}
    sim.run_compiled(num_steps=1, progress=False)


def test_simulation_compile_uses_resolved_setup_device_context(monkeypatch):
    design = Design(width=2 * um, height=2 * um, material=Material(permittivity=1.0))
    sim = Simulation(
        design=design,
        sources=[],
        monitors=[],
        time=np.array([0.0, 1e-16, 2e-16], dtype=float),
        resolution=0.5 * um,
    )
    sim.setup_device_resolved = "cpu"
    active = {"value": False}
    program = object()

    @contextmanager
    def fake_setup_context(resolved_device):
        assert resolved_device == "cpu"
        active["value"] = True
        try:
            yield
        finally:
            active["value"] = False

    def fake_compile_simulation(**_kwargs):
        assert active["value"]
        return program

    monkeypatch.setattr(
        simulation_core,
        "_resolved_setup_device_context",
        fake_setup_context,
    )
    monkeypatch.setattr(simulation_core, "compile_simulation", fake_compile_simulation)

    assert sim.compile(num_steps=1) is program


def test_simulation_rejects_invalid_setup_device():
    design = Design(width=2 * um, height=2 * um, material=Material(permittivity=1.0))

    with pytest.raises(ValueError, match="setup_device"):
        Simulation(
            design=design,
            sources=[],
            monitors=[],
            time=_time_axis(),
            resolution=0.5 * um,
            setup_device="quantum",
        )


def test_design_rejects_device_objects():
    design = Design(width=4 * um, height=4 * um, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(2 * um, 2 * um), width=0.2 * um, signal=[1.0, 0.0]
    )
    monitor = Monitor(start=(1 * um, 1 * um), end=(1 * um, 3 * um), name="m1")

    try:
        design += source
    except TypeError as exc:
        assert "Simulation(sources=[...])" in str(exc)
    else:
        raise AssertionError("Expected Design to reject source objects")

    try:
        design += monitor
    except TypeError as exc:
        assert "Simulation(monitors=[...])" in str(exc)
    else:
        raise AssertionError("Expected Design to reject monitor objects")


def test_run_compiled_returns_simulation_results_with_backward_compatible_mapping():
    design = Design(width=4 * um, height=4 * um, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(2 * um, 2 * um), width=0.2 * um, signal=[1.0, 0.0, 0.0]
    )
    monitor = Monitor(start=(1 * um, 1 * um), end=(1 * um, 3 * um), name="m1")

    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[monitor],
        time=np.array([0.0, 1e-16, 2e-16, 3e-16], dtype=float),
        resolution=0.2 * um,
    )
    result = sim.run_compiled(progress=False)

    assert isinstance(result, SimulationResults)
    assert result["monitors"] == [monitor]
    assert "monitor_results" in result
    assert isinstance(result.monitor_results["m1"], MonitorResults)
