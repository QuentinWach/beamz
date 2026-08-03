import gc
import weakref
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, dataclass, is_dataclass, replace
from inspect import signature

import numpy as np
import pytest

import beamz.simulation.api as simulation_core
import beamz.simulation.execute as execution_runtime
from beamz import (
    AutoTermination,
    Design,
    FieldMonitor,
    FieldRecorder,
    GaussianBeamSource,
    GaussianPulse,
    GaussianSource,
    GridSpec,
    Material,
    ModeSource,
    MonitorResults,
    RunTermination,
    Simulation,
    SimulationResults,
    SimulationRun,
    um,
)
from beamz.simulation.execute import execution_cache
from beamz.simulation.model import SimulationRequest, SimulationState
from beamz.simulation.results import FieldMetadata, SimulationMetadata

pytestmark = pytest.mark.unit


def _time_axis():
    return np.array([0.0, 1e-16, 2e-16], dtype=float)


def _simulation(**changes):
    values = {
        "design": Design(width=2 * um, height=2 * um, material=Material(1.0)),
        "time": _time_axis(),
        "resolution": 0.5 * um,
    }
    values.update(changes)
    return Simulation(**values)


def test_simulation_rejects_invalid_plane_and_resolution():
    assert _simulation(plane_2d="XY").plane_2d == "xy"
    with pytest.raises(ValueError, match="plane_2d"):
        _simulation(plane_2d="xxy")

    for resolution in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="resolution.*positive and finite"):
            _simulation(resolution=resolution)
        with pytest.raises(ValueError, match="resolution.*positive and finite"):
            _simulation(grid_spec=GridSpec.uniform(resolution))


def test_simulation_rejects_conflicting_or_invalid_time_specifications():
    with pytest.raises(ValueError, match="only one of time.*run_time"):
        _simulation(run_time=1e-15)

    for run_time in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="run_time.*positive and finite"):
            _simulation(time=None, run_time=run_time)


def test_auto_termination_is_an_immutable_validated_configuration():
    spec = AutoTermination(
        field_decay=2e-5,
        monitor_change=None,
        chunk_steps=np.int64(128),
        min_steps=64,
        monitor_names=["output"],
    )

    assert is_dataclass(spec)
    assert spec.field_decay == pytest.approx(2e-5)
    assert spec.chunk_steps == 128
    assert spec.monitor_names == ("output",)
    with pytest.raises(FrozenInstanceError):
        spec.chunk_steps = 32

    invalid = (
        {"field_decay": -1.0},
        {"monitor_change": np.inf},
        {"source_decay": np.nan},
        {"chunk_steps": 0},
        {"min_steps": -1},
        {"consecutive_checks": 0},
        {"growth_factor": 1.0},
        {"growth_checks": 0},
        {"monitor_names": ("m", "m")},
        {"field_decay": 0.0, "monitor_change": None},
    )
    for values in invalid:
        with pytest.raises(ValueError):
            AutoTermination(**values)


def test_run_termination_validates_reason_and_result_attachment():
    fields = FieldMetadata(grid_shape=(1, 1), component_shapes={"Ez": (1, 1)})
    metadata = SimulationMetadata(
        dt=1.0,
        resolution=1.0,
        is_3d=False,
        plane_2d="xy",
        coordinate_offset=(0.0, 0.0, 0.0),
        time=np.array([0.0, 1.0]),
        width=1.0,
        height=1.0,
        depth=0.0,
        fields=fields,
    )
    report = RunTermination(
        reason="converged",
        steps=12,
        time=1.2e-15,
        converged=True,
        field_decay=8e-6,
        consecutive_checks=3,
    )

    assert (
        SimulationResults(metadata=metadata, termination=report).termination is report
    )
    with pytest.raises(ValueError, match="reason"):
        RunTermination(reason="unknown", steps=0, time=0.0, converged=False)
    with pytest.raises(TypeError, match="termination"):
        SimulationResults(metadata=metadata, termination=object())


def test_run_without_progress_is_silent(capsys):
    sim = _simulation()
    sim.clear_compiled_cache()

    sim.run(progress=False)

    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_run_can_stop_after_consecutive_field_decay_checks():
    sim = _simulation(time=np.arange(20, dtype=float) * 1e-16)

    result = sim.run(
        termination=AutoTermination(
            chunk_steps=2,
            consecutive_checks=2,
            monitor_change=None,
        )
    )

    assert result.termination is not None
    assert result.termination.reason == "converged"
    assert result.termination.converged
    assert result.termination.steps == 4
    assert result.termination.field_decay == pytest.approx(0.0)
    assert result.termination.source_decay == pytest.approx(0.0)


def test_run_monitor_stability_requires_a_previous_check():
    monitor = FieldMonitor(
        center=(um, um, 0.0),
        size=(0.0, um, 0.0),
        freqs=(2e14,),
        name="output",
    )
    sim = _simulation(
        time=np.arange(20, dtype=float) * 1e-16,
        monitors=(monitor,),
    )

    result = sim.run(
        termination=AutoTermination(
            field_decay=0.0,
            monitor_change=1e-8,
            chunk_steps=2,
            consecutive_checks=2,
            monitor_names=("output",),
        )
    )

    assert result.termination is not None
    assert result.termination.reason == "converged"
    assert result.termination.steps == 6
    assert result.termination.monitor_change == pytest.approx(0.0)
    np.testing.assert_allclose(result.monitors["output"].dft_weight_sum, [20.0])


def test_run_never_converges_while_a_source_remains_active():
    time = np.arange(12, dtype=float) * 1e-16
    source = GaussianSource(
        position=(um, um), width=0.2 * um, signal=np.ones(time.size)
    )
    sim = _simulation(time=time, sources=(source,))

    result = sim.run(
        termination=AutoTermination(
            chunk_steps=2,
            consecutive_checks=2,
            monitor_change=None,
            field_decay=1.0,
        )
    )

    assert result.termination is not None
    assert result.termination.reason == "time_limit"
    assert not result.termination.converged
    assert result.termination.steps == time.size


def test_run_reports_nonfinite_fields(monkeypatch):
    sim = _simulation(time=np.arange(6, dtype=float) * 1e-16)
    monkeypatch.setattr(
        execution_runtime,
        "_field_diagnostics",
        lambda state, plan: (np.nan, np.inf, False),
    )

    result = sim.run(
        termination=AutoTermination(
            chunk_steps=2,
            consecutive_checks=2,
            monitor_change=None,
        )
    )

    assert result.termination is not None
    assert result.termination.reason == "nonfinite"
    assert not result.termination.converged
    assert result.termination.steps == 2


def test_run_reports_consecutive_post_source_energy_growth(monkeypatch):
    sim = _simulation(time=np.arange(8, dtype=float) * 1e-16)
    energies = iter((1.0, 2.0, 4.0))

    def increasing_energy(state, plan):
        del state, plan
        energy = next(energies)
        return energy, energy, True

    monkeypatch.setattr(execution_runtime, "_field_diagnostics", increasing_energy)

    result = sim.run(
        termination=AutoTermination(
            field_decay=1e-12,
            monitor_change=None,
            chunk_steps=1,
            consecutive_checks=2,
            growth_factor=1.1,
            growth_checks=2,
        )
    )

    assert result.termination is not None
    assert result.termination.reason == "diverged"
    assert not result.termination.converged
    assert result.termination.steps == 3
    assert result.termination.energy == pytest.approx(4.0)


def test_run_rejects_invalid_automatic_termination_inputs():
    sim = _simulation()
    sim.clear_compiled_cache()
    with pytest.raises(TypeError, match="AutoTermination"):
        sim.run(termination=object())
    with pytest.raises(ValueError, match="frequency-domain"):
        sim.run(
            termination=AutoTermination(
                monitor_names=("missing",),
                chunk_steps=1,
            )
        )


@pytest.mark.parametrize("interval", (0, -3, 1.9, True))
def test_monitor_interval_must_be_a_positive_integer(interval):
    with pytest.raises(ValueError, match="Monitor interval.*positive integer"):
        FieldRecorder(interval=interval)

    assert FieldRecorder(interval=np.int64(2)).interval == 2


def test_simulation_is_a_deeply_immutable_configuration_value():
    source = GaussianSource(position=(um, um), width=0.2 * um, signal=[1.0, 0.0])
    monitor = FieldMonitor(
        center=(0.5 * um, 0.0, 0.0),
        size=(um, 0.0, 0.0),
        freqs=(2e14,),
        fields=("Ez",),
        name="m",
    )
    sim = _simulation(sources=[source], monitors=[monitor])

    assert is_dataclass(sim)
    assert sim.sources == (source,)
    assert sim.monitors == (monitor,)
    assert isinstance(sim.boundaries, tuple)
    assert not sim.time.flags.writeable
    assert not hasattr(sim, "engine")
    assert not hasattr(sim, "fields")
    assert not hasattr(sim, "runtime")
    with pytest.raises(FrozenInstanceError):
        sim.sources = ()
    with pytest.raises(FrozenInstanceError):
        sim.resolution = 1.0


def test_updated_copy_returns_a_new_validated_simulation():
    sim = _simulation()
    changed = sim.updated_copy(time=[0.0, 2e-16, 4e-16, 6e-16])

    assert changed is not sim
    assert changed.dt == pytest.approx(2e-16)
    assert changed.num_steps == 4
    assert sim.dt == pytest.approx(1e-16)
    with pytest.raises(ValueError, match="uniformly spaced"):
        sim.updated_copy(time=[0.0, 1e-16, 3e-16])


def test_updated_copy_preserves_run_time_without_passing_a_conflicting_time_grid():
    sim = Simulation(
        design=Design(width=2 * um, height=2 * um, material=Material(1.0)),
        resolution=0.5 * um,
        run_time=1e-14,
    )

    changed = sim.updated_copy(sources=())

    assert changed.run_time == pytest.approx(sim.run_time)
    np.testing.assert_allclose(changed.time, sim.time)

    explicit_time = sim.updated_copy(time=_time_axis())
    assert explicit_time.run_time is None
    np.testing.assert_array_equal(explicit_time.time, _time_axis())


def test_source_normalization_selection_is_validated_and_replaceable():
    source = GaussianSource(position=(um, um), width=0.2 * um, signal=[1.0, 0.0])
    sim = _simulation(sources=[source])

    assert sim.normalize_source == 0
    assert sim.updated_copy(normalize_source=None).normalize_source is None
    with pytest.raises(ValueError, match="normalize_source"):
        sim.updated_copy(normalize_source=1)


def test_explicit_step_returns_new_runtime_state():
    source = GaussianSource(position=(um, um), width=0.2 * um, signal=[1.0, 0.0, 0.0])
    sim = _simulation(sources=(source,))
    initial = sim.initial_state()
    direct = SimulationState.initial(sim.compile().grid, t=float(sim.time[0]))
    advanced = sim.step(initial)
    changed = sim.updated_copy(sources=(source.updated_copy(width=0.3 * um),))
    advanced_again = changed.step(advanced)

    assert isinstance(initial, SimulationState)
    assert direct._fields == initial._fields
    assert initial.current_step == 0
    assert advanced.current_step == 1
    assert advanced_again.current_step == 2
    assert not hasattr(sim, "current_step")


def test_constructor_defers_discretization(monkeypatch):
    simulation_core._MATERIAL_GRID_CACHE.clear()
    calls = {"count": 0}
    original = simulation_core.build_material_grid

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(simulation_core, "build_material_grid", counted)
    sim = _simulation()

    assert calls["count"] == 0
    request = sim.to_request(num_steps=1)
    assert request.run.num_steps == 1
    assert calls["count"] == 1
    sim.to_request(num_steps=1)
    assert calls["count"] == 1


def test_simulation_rejects_duplicate_named_monitors():
    m1 = FieldMonitor(
        center=(0.5 * um, 0.0, 0.0), size=(um, 0.0, 0.0), freqs=(2e14,), name="port"
    )
    m2 = FieldMonitor(
        center=(0.5 * um, um, 0.0), size=(um, 0.0, 0.0), freqs=(2e14,), name="port"
    )
    with pytest.raises(ValueError, match="(?i)duplicate.*port|port.*duplicate"):
        _simulation(monitors=(m1, m2))


def test_simulation_rejects_arbitrary_mutable_devices():
    @dataclass
    class MutableSource:
        amplitude: float

    source = MutableSource(amplitude=1.0)

    with pytest.raises(TypeError, match="canonical immutable device types"):
        _simulation(sources=(source,))

    source.amplitude = 2.0
    with pytest.raises(TypeError, match="MutableSource"):
        _simulation(sources=(source,))


def test_canonical_sources_reject_or_snapshot_mutable_nested_values():
    @dataclass
    class MutablePulse:
        amplitude: float = 1.0

    with pytest.raises(TypeError, match="MutablePulse must be a frozen dataclass"):
        GaussianBeamSource(
            center=(0.0, 0.0, 0.0),
            size=(um, um),
            source_time=MutablePulse(),
            wavelength=1.55 * um,
        )

    @dataclass
    class MutableModeSpec:
        num_modes: int = 1

    with pytest.raises(TypeError, match="MutableModeSpec must be a frozen dataclass"):
        ModeSource(
            center=(0.0, 0.0, 0.0),
            size=(0.0, um, um),
            source_time=GaussianPulse(freq0=2e14, fwidth=2e13),
            direction="+",
            mode_spec=MutableModeSpec(),
        )

    waveform = np.array([1.0, 0.0], dtype=float)
    source = GaussianBeamSource(
        center=(0.0, 0.0, 0.0),
        size=(um, um),
        source_time=waveform,
        wavelength=1.55 * um,
    )
    sim = _simulation(sources=(source,))
    request = sim.to_request(num_steps=1)
    original_hash = hash(sim)

    waveform[0] = 9.0

    np.testing.assert_allclose(sim.sources[0].source_time, [1.0, 0.0])
    assert not sim.sources[0].source_time.flags.writeable
    assert hash(sim) == original_hash
    assert request.sources[0] is sim.sources[0]


def test_setup_device_policy_is_immutable_and_replaceable():
    sim = _simulation(setup_device="cpu")
    assert sim.setup_device_policy == "cpu"
    assert sim.setup_device_resolved in {"cpu", "default"}
    changed = sim.updated_copy(setup_device="default")
    assert changed.setup_device_policy == "default"
    assert sim.setup_device_policy == "cpu"


def test_compile_uses_resolved_setup_device_context(monkeypatch):
    sim = _simulation(setup_device="cpu")
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

    def fake_compile(request):
        assert active["value"]
        assert isinstance(request, SimulationRequest)
        return program

    sim.clear_compiled_cache()
    monkeypatch.setattr(
        simulation_core, "_resolved_setup_device_context", fake_setup_context
    )
    monkeypatch.setattr(simulation_core, "compile_simulation", fake_compile)

    assert sim.compile(num_steps=1) is program


def test_design_rejects_device_objects():
    design = Design(width=4 * um, height=4 * um, material=Material(1.0))
    source = GaussianSource(
        position=(2 * um, 2 * um), width=0.2 * um, signal=[1.0, 0.0]
    )
    beam = GaussianBeamSource(
        center=(0.0, 0.0, 0.0),
        size=(um, um),
        source_time=np.ones(2),
        direction="-z",
        waist_radius=0.4 * um,
        wavelength=1.55 * um,
    )
    monitor = FieldMonitor(
        center=(um, 2 * um, 0.0), size=(0.0, 2 * um, 0.0), freqs=(2e14,), name="m"
    )

    for item in (source, beam, monitor):
        with pytest.raises(TypeError):
            design.with_structure(item)


def test_run_returns_detached_immutable_results():
    source = GaussianSource(position=(um, um), width=0.2 * um, signal=[1.0, 0.0, 0.0])
    monitor = FieldRecorder(
        components=("Ez",),
        interval=1,
        center=(0.5 * um, um, 0.0),
        size=(0.0, um, 0.0),
        name="m",
    )
    sim = _simulation(sources=(source,), monitors=(monitor,))
    result = sim.run(progress=False)

    assert isinstance(result, SimulationResults)
    assert not isinstance(result, SimulationRun)
    assert isinstance(result.monitors["m"], MonitorResults)
    assert not hasattr(result, "monitor_results")
    assert not hasattr(result, "final_state")
    assert not hasattr(result, "state")
    assert not hasattr(result, "fields")
    assert len(result.monitors["m"].fields["Ez"]) > 0
    assert not hasattr(result, "simulation")
    assert result.metadata.fields.materials is None
    with pytest.raises(ValueError):
        result.metadata.time[0] = 1.0
    with pytest.raises(TypeError):
        result.monitors["new"] = result.monitors["m"]

    ref = weakref.ref(sim)
    del sim
    gc.collect()
    assert ref() is None


def test_continuation_preserves_state_by_default_and_donation_is_explicit():
    design = Design(
        width=2 * um,
        height=2 * um,
        depth=um,
        material=Material(1.0),
    )
    sim = _simulation(design=design)
    state = sim.initial_state()
    original_ez = np.asarray(state.ez).copy()

    first = sim.advance(num_steps=1, state=state, progress=False)
    second = sim.advance(num_steps=1, state=state, progress=False)

    assert isinstance(first, SimulationRun)
    assert isinstance(first.results, SimulationResults)
    assert int(state.current_step) == 0
    np.testing.assert_array_equal(state.ez, original_ez)
    np.testing.assert_array_equal(first.state.ez, second.state.ez)
    program = sim.compile(num_steps=1)
    cache = execution_cache(program)
    preserved_scan = cache.compiled_scan
    assert callable(preserved_scan)
    assert cache.compiled_scan_donating is None

    donated = sim.advance(
        num_steps=1,
        state=first.state,
        progress=False,
        donate_state=True,
    )
    assert int(donated.state.current_step) == 2
    assert cache.compiled_scan is preserved_scan
    assert callable(cache.compiled_scan_donating)


def test_advance_rejects_steps_outside_remaining_time_grid():
    recorder = FieldRecorder(
        components=("Ez",),
        interval=1,
        center=(um, um, 0.0),
        size=(0.0, um, 0.0),
        name="m",
    )
    sim = _simulation(time=np.array([0.0, 1e-16]), monitors=(recorder,))

    for num_steps in (0, -1, 3):
        with pytest.raises(ValueError, match=r"num_steps must be in \[1, 2\]"):
            sim.advance(num_steps=num_steps, progress=False)

    state = sim.step()
    with pytest.raises(ValueError, match=r"num_steps must be in \[1, 1\]"):
        sim.advance(num_steps=2, state=state, progress=False)

    run = sim.advance(num_steps=1, state=state, progress=False)
    completed = run.state
    assert int(completed.current_step) == 2
    np.testing.assert_array_equal(run.results.monitors["m"].field_steps, [1, 2])
    with pytest.raises(ValueError, match=r"num_steps must be in \[1, 0\]"):
        sim.advance(state=completed, progress=False)
    with pytest.raises(ValueError, match="outside the simulation time grid"):
        sim.advance(num_steps=1, state=completed._replace(current_step=3))


def test_simulation_metadata_copies_time_and_validates_nested_values():
    time = np.array([0.0, 0.25])
    fields = FieldMetadata(grid_shape=(2, 2), component_shapes={"Ez": (2, 2)})
    metadata = SimulationMetadata(
        dt=0.25,
        resolution=1.0,
        is_3d=False,
        plane_2d="XY",
        coordinate_offset=[0, 0, 0],
        time=time,
        width=2,
        height=2,
        depth=0,
        fields=fields,
    )

    time[0] = 9.0
    np.testing.assert_allclose(metadata.time, [0.0, 0.25])
    assert not metadata.time.flags.writeable
    assert metadata.plane_2d == "xy"
    assert metadata.coordinate_offset == (0.0, 0.0, 0.0)
    equivalent = replace(metadata, time=np.array([0.0, 0.25]))
    changed = replace(metadata, time=np.array([0.0, 0.5]))
    assert metadata == equivalent
    assert hash(metadata) == hash(equivalent)
    assert metadata != changed

    with pytest.raises(TypeError, match="fields must be FieldMetadata"):
        _ = SimulationMetadata(
            dt=0.25,
            resolution=1.0,
            is_3d=False,
            plane_2d="xy",
            coordinate_offset=(0, 0, 0),
            time=[0.0],
            width=2,
            height=2,
            depth=0,
            fields=object(),
        )
    with pytest.raises(TypeError, match="metadata must be SimulationMetadata"):
        SimulationResults(metadata=object())
    with pytest.raises(TypeError, match="values must be MonitorResults"):
        SimulationResults(metadata=metadata, monitors={"bad": object()})

    source = GaussianSource(position=(um, um), width=0.2 * um, signal=[1.0, 0.0])
    results = SimulationResults(
        metadata=metadata,
        sources=(source,),
        source_launch_powers=(0.987,),
    )
    assert results.launched_power() == pytest.approx(0.987)
    with pytest.raises(ValueError, match="invalid for 1 sources"):
        results.launched_power(source=1)


def test_run_api_has_no_live_snapshot_callback():
    assert "snapshot_callback" not in signature(Simulation.run).parameters
    assert "snapshot_callback" not in signature(Simulation.advance).parameters
    assert "state" not in signature(Simulation.run).parameters
    assert "num_steps" not in signature(Simulation.run).parameters
    assert "donate_state" not in signature(Simulation.run).parameters
    assert not hasattr(Simulation, "run_compiled")


def test_invalid_setup_device_is_rejected():
    with pytest.raises(ValueError, match="setup_device"):
        _simulation(setup_device="quantum")
