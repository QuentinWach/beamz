import numpy as np
import pytest

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


def _time_axis():
    return np.array([0.0, 1e-16, 2e-16], dtype=float)


def test_simulation_normalizes_sources_and_monitors_from_explicit_args():
    design = Design(width=4 * um, height=4 * um, material=Material(permittivity=1.0))
    source = GaussianSource(position=(2 * um, 2 * um), width=0.2 * um, signal=[1.0, 0.0])
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
    assert sim.devices == [source, monitor]


def test_simulation_normalizes_legacy_devices_into_sources_and_monitors():
    design = Design(width=4 * um, height=4 * um, material=Material(permittivity=1.0))
    source = GaussianSource(position=(2 * um, 2 * um), width=0.2 * um, signal=[1.0, 0.0])
    monitor = Monitor(start=(1 * um, 1 * um), end=(1 * um, 3 * um), name="m1")

    sim = Simulation(
        design=design,
        devices=[source, monitor],
        time=_time_axis(),
        resolution=0.2 * um,
    )

    assert sim.sources == [source]
    assert sim.monitors == [monitor]
    assert sim.devices == [source, monitor]


def test_simulation_falls_back_to_design_devices_with_deprecation_warning():
    design = Design(width=4 * um, height=4 * um, material=Material(permittivity=1.0))
    source = GaussianSource(position=(2 * um, 2 * um), width=0.2 * um, signal=[1.0, 0.0])
    monitor = Monitor(start=(1 * um, 1 * um), end=(1 * um, 3 * um), name="m1")
    design += source
    design += monitor

    with pytest.deprecated_call(match="Passing sources and monitors via Design is deprecated"):
        sim = Simulation(
            design=design,
            time=_time_axis(),
            resolution=0.2 * um,
        )

    assert sim.sources == [source]
    assert sim.monitors == [monitor]
    assert sim.devices == [source, monitor]


def test_run_compiled_returns_simulation_results_with_backward_compatible_mapping():
    design = Design(width=4 * um, height=4 * um, material=Material(permittivity=1.0))
    source = GaussianSource(position=(2 * um, 2 * um), width=0.2 * um, signal=[1.0, 0.0, 0.0])
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
