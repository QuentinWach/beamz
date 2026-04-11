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


def test_simulation_accepts_explicit_sources_and_monitors():
    design = Design(width=4 * um, height=4 * um, material=Material(permittivity=1.0))
    source = GaussianSource(position=(2 * um, 2 * um), width=0.2 * um, signal=[1.0, 0.0])
    monitor = Monitor(start=(1 * um, 1 * um), end=(1 * um, 3 * um), name="m1")

    sim = Simulation(
        design=design,
        sources=[source], monitors=[monitor],
        time=_time_axis(),
        resolution=0.2 * um,
    )

    assert sim.sources == [source]
    assert sim.monitors == [monitor]


def test_simulation_rejects_duplicate_named_monitors():
    design = Design(width=4 * um, height=4 * um, material=Material(permittivity=1.0))
    m1 = Monitor(start=(1 * um, 1 * um), end=(1 * um, 3 * um), name="port")
    m2 = Monitor(start=(2 * um, 1 * um), end=(2 * um, 3 * um), name="port")

    with pytest.raises(ValueError, match="Simulation\\._normalize_specs.*port"):
        Simulation(
            design=design,
            monitors=[m1, m2],
            time=_time_axis(),
            resolution=0.2 * um,
        )


def test_design_rejects_device_objects():
    design = Design(width=4 * um, height=4 * um, material=Material(permittivity=1.0))
    source = GaussianSource(position=(2 * um, 2 * um), width=0.2 * um, signal=[1.0, 0.0])
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


def test_monitor_runtime_state_is_kept_in_private_container():
    monitor = Monitor(start=(1 * um, 1 * um), end=(1 * um, 3 * um), name="m1")

    assert "_state" in monitor.__dict__
    assert "power_history" not in monitor.__dict__
    assert "fields" not in monitor.__dict__

    monitor.power_history = [1.0, 2.0]
    monitor.fields["t"] = [0.0]

    assert monitor._state.power_history == [1.0, 2.0]
    assert monitor._state.fields["t"] == [0.0]


def test_source_runtime_state_is_kept_in_private_container():
    source = GaussianSource(position=(2 * um, 2 * um), width=0.2 * um, signal=[1.0, 0.0])

    assert "_state" in source.__dict__
    assert "_spatial_profile_ez" not in source.__dict__
    assert "_grid_indices" not in source.__dict__

    source._grid_indices = (slice(0, 1), slice(0, 1))
    assert source._state._grid_indices == (slice(0, 1), slice(0, 1))
