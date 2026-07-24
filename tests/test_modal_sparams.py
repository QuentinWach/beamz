import numpy as np
import pytest

from beamz import (
    Design,
    FieldMonitor,
    FieldRecorder,
    FluxMonitor,
    Material,
    ModeMonitor,
    ModeSpec,
    Port,
    Simulation,
)
from beamz.analysis import s_parameters
from beamz.analysis import sparameters as sp
from beamz.analysis.data import AnalysisData
from beamz.simulation.results import (
    FieldMetadata,
    MonitorResults,
    SimulationMetadata,
    SimulationResults,
)


def _simulation(*, width=2.0, height=2.0, depth=0.0, monitors=(), sources=()):
    return Simulation(
        design=Design(
            width=width,
            height=height,
            depth=depth,
            background=Material(1.0),
        ),
        monitors=monitors,
        sources=sources,
        resolution=1.0,
        time=np.array([0.0, 0.25]),
    )


def _analysis_contract(*, is_3d=False):
    shape = (1, 1, 1) if is_3d else (1, 1)
    metadata = SimulationMetadata(
        dt=0.25,
        resolution=1.0,
        is_3d=is_3d,
        plane_2d="xy",
        coordinate_offset=(0.0, 0.0, 0.0),
        time=np.array([0.0, 0.25]),
        width=1.0,
        height=1.0,
        depth=float(is_3d),
        fields=FieldMetadata(shape, {}),
    )
    return AnalysisData(metadata, {}, None, (), None)


def _empty_monitor_result(monitor):
    return MonitorResults(
        monitor=monitor,
        fields={},
        power_history=np.empty(0),
        power_timestamps=np.empty(0),
        power_spectrum=np.empty(0, dtype=np.complex64),
    )


def _port(
    *,
    name,
    monitor_name=None,
    direction="+x",
    polarization="tm",
    mode_index=0,
):
    axis = str(direction)[-1]
    size = tuple(0.0 if value == axis else 1.0 for value in "xyz")
    return Port(
        center=(0.0, 0.0, 0.0),
        size=size,
        name=name,
        monitor_name=monitor_name,
        direction=str(direction)[0],
        mode_spec=ModeSpec(mode_index=mode_index, polarization=polarization),
    )


def _run_sparameters(waves, ports, frequencies, output_ports, monkeypatch, **kwargs):
    inputs = {"analysis": _analysis_contract()}

    def fake_extract(
        sim_arg,
        ports,
        frequencies,
        min_incident_db=-40.0,
        return_power=True,
        mode_strategy="per_frequency",
    ):
        del sim_arg, ports, min_incident_db, return_power, mode_strategy
        return waves

    monkeypatch.setattr(sp, "_extract_port_waves_dft", fake_extract)
    return s_parameters(
        inputs,
        source_port=kwargs.pop("source_port", "o1"),
        ports=ports,
        output_ports=output_ports,
        frequencies=frequencies,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("direction", "incident", "scattered"),
    [
        ("+x", "plus", "minus"),
        ("-x", "minus", "plus"),
        ("+y", "minus", "plus"),
        ("-y", "plus", "minus"),
    ],
)
def test_2d_wave_selectors_follow_physical_port_direction(
    direction, incident, scattered
):
    port = _port(name="port", direction=direction, polarization="tm")
    assert sp._wave_selectors(port, is_3d=False) == (incident, scattered)


def test_s_parameters_returns_typed_result_and_matrix(monkeypatch):
    freqs = np.array([191e12, 193e12, 195e12], dtype=float)
    ports = [
        _port(name="o1", monitor_name="o1", direction="+x", polarization="tm"),
        _port(name="o2", monitor_name="o2", direction="+x", polarization="tm"),
    ]
    waves = {
        "o1": {
            "a_plus": np.ones(3, dtype=np.complex128),
            "a_minus": np.array([0.01, 0.015, 0.02], dtype=np.complex128),
            "a_incident": np.ones(3, dtype=np.complex128),
        },
        "o2": {
            "a_plus": np.zeros(3, dtype=np.complex128),
            "a_minus": np.array([0.98, 0.975, 0.97], dtype=np.complex128),
        },
    }

    result = _run_sparameters(waves, ports, freqs, ["o1", "o2"], monkeypatch)

    assert isinstance(result, sp.SParameterResult)
    assert set(result.s_matrix) == {("o1", "o1"), ("o2", "o1")}
    np.testing.assert_allclose(result.frequencies, freqs)
    np.testing.assert_allclose(result.s_matrix[("o2", "o1")], waves["o2"]["a_minus"])
    assert np.nanmax(result.diagnostics["power_sum"]) <= 1.02


def test_s_parameters_mmi_balance_and_loss_visibility(monkeypatch):
    freqs = np.array([192e12, 193.5e12, 195e12], dtype=float)
    ports = [
        _port(name="o1", monitor_name="o1", direction="+x", polarization="tm"),
        _port(name="o2", monitor_name="o2", direction="+x", polarization="tm"),
        _port(name="o3", monitor_name="o3", direction="+x", polarization="tm"),
    ]
    low_loss = {
        "o1": {
            "a_plus": np.ones(3),
            "a_minus": 0.05 * np.ones(3),
            "a_incident": np.ones(3),
        },  # fmt: skip
        "o2": {"a_plus": np.zeros(3), "a_minus": np.array([0.69, 0.705, 0.695])},
        "o3": {"a_plus": np.zeros(3), "a_minus": np.array([0.695, 0.700, 0.692])},
    }
    high_loss = {
        "o1": {
            "a_plus": np.ones(3),
            "a_minus": 0.05 * np.ones(3),
            "a_incident": np.ones(3),
        },  # fmt: skip
        "o2": {"a_plus": np.zeros(3), "a_minus": np.array([0.52, 0.50, 0.49])},
        "o3": {"a_plus": np.zeros(3), "a_minus": np.array([0.50, 0.49, 0.48])},
    }

    low = _run_sparameters(low_loss, ports, freqs, ["o1", "o2", "o3"], monkeypatch)
    high = _run_sparameters(high_loss, ports, freqs, ["o1", "o2", "o3"], monkeypatch)

    idx = 1
    s21_db = 20 * np.log10(max(abs(low.s_matrix[("o2", "o1")][idx]), 1e-12))
    s31_db = 20 * np.log10(max(abs(low.s_matrix[("o3", "o1")][idx]), 1e-12))
    assert abs(s21_db - s31_db) < 0.5
    assert np.nanmean(high.diagnostics["power_sum"]) < np.nanmean(low.diagnostics["power_sum"])  # fmt: skip
    assert np.nanmean(high.diagnostics["loss_est"]) > np.nanmean(low.diagnostics["loss_est"])  # fmt: skip


def test_s_parameters_results_ignore_live_simulation_and_monitor_mutation(monkeypatch):
    freqs = np.array([1.0], dtype=float)
    sim = _simulation(width=1.0, height=1.0)

    monitor = FieldMonitor(
        center=(0.5, 0.0, 0.0),
        size=(1.0, 0.0, 0.0),
        name="o1",
        freqs=freqs,
        fields=("Ez",),
    )
    monitor_result = MonitorResults(
        monitor=monitor,
        fields={},
        power_history=np.asarray([], dtype=float),
        power_timestamps=np.asarray([], dtype=float),
        power_spectrum=np.asarray([], dtype=np.complex64),
        dft_fields={"Ez": np.array([[2.0 + 0.0j]])},
        dft_frequencies=freqs,
        dft_weight_sum=np.array([2.0]),
        dft_base_dt=sim.dt,
        resolution=1.0,
    )
    results = SimulationResults.from_run(
        sim,
        runtime_fields=sim.compile().grid,
        monitor_results={"o1": monitor_result},
    )

    monitor = monitor.updated_copy(freqs=np.array([9.0]))

    def fake_extract(sim_arg, ports, frequencies, **kwargs):
        del ports, kwargs
        data = sp.analysis_inputs(sim_arg)["o1"]
        assert data.dt == 0.25
        np.testing.assert_allclose(frequencies, freqs)
        np.testing.assert_allclose(data.field("Ez"), [[2.0 + 0.0j]])
        return {
            "o1": {
                "a_plus": np.ones(1, dtype=np.complex128),
                "a_minus": np.array([0.25], dtype=np.complex128),
                "a_incident": np.ones(1, dtype=np.complex128),
            },
        }

    monkeypatch.setattr(sp, "_extract_port_waves_dft", fake_extract)
    result = s_parameters(
        results,
        source_port="o1",
        ports=[_port(name="o1", monitor_name="o1", direction="+x", polarization="tm")],
        output_ports=["o1"],
    )

    np.testing.assert_allclose(result.s_matrix[("o1", "o1")], [0.25])


def test_results_analysis_snapshot_keeps_compact_field_shapes():
    sim = _simulation(width=4.0, height=3.0)
    monitor = FieldMonitor(
        center=(0.5, 0.0, 0.0),
        size=(1.0, 0.0, 0.0),
        name="o1",
        freqs=np.array([1.0]),
    )

    grid = sim.compile().grid
    results = SimulationResults.from_run(
        sim,
        runtime_fields=grid,
        monitor_results={"o1": _empty_monitor_result(monitor)},
    )
    fields = results.metadata.fields

    assert fields.grid_shape == (3, 4)
    assert fields.component_shapes["Ex"] == grid.Ex.shape
    assert fields.component_shapes["Hz"] == grid.Hz.shape
    assert fields.materials is None

    full_results = SimulationResults.from_run(
        sim,
        runtime_fields=grid,
        monitor_results={"o1": _empty_monitor_result(monitor)},
        store_full_materials=True,
    )
    np.testing.assert_allclose(
        full_results.metadata.fields.materials.permittivity,
        grid.permittivity,
    )
    assert not full_results.metadata.fields.materials.permittivity.flags.writeable


def test_3d_mode_results_store_a_thin_material_region_by_default():
    shape = (12, 14, 16)
    sim = _simulation(width=16.0, height=14.0, depth=12.0)
    monitor = ModeMonitor(
        center=(8.0, 7.0, 6.0),
        size=(0.0, 6.0, 6.0),
        freqs=[1.0],
        mode_spec=ModeSpec(polarization="te"),
        name="mode",
    )

    results = SimulationResults.from_run(
        sim,
        runtime_fields=sim.compile().grid,
        monitor_results={"mode": _empty_monitor_result(monitor)},
    )
    region = results.monitors["mode"].material_region

    assert region.full_shape == shape
    assert region.permittivity.shape[:2] == shape[:2]
    assert region.permittivity.shape[2] <= 5
    assert region.permittivity.size < np.prod(shape)
    assert not region.permittivity.flags.writeable


def test_monitor_results_snapshot_copies_power_arrays():
    monitor = FluxMonitor(
        center=(0.5, 0.0, 0.0),
        size=(1.0, 0.0, 1.0),
        name="o1",
        freqs=np.array([1.0]),
    )
    power_history = np.array([1.0, 2.0])
    power_timestamps = np.array([0.0, 0.25])
    power_spectrum = np.array([3.0 + 0.0j], dtype=np.complex64)

    result = MonitorResults(
        monitor=monitor,
        fields={},
        power_history=power_history,
        power_timestamps=power_timestamps,
        power_spectrum=power_spectrum,
    )
    power_history[0] = 9.0
    power_timestamps[0] = 9.0
    power_spectrum[0] = 9.0 + 0.0j

    np.testing.assert_allclose(result.power_history, [1.0, 2.0])
    np.testing.assert_allclose(result.power_timestamps, [0.0, 0.25])
    np.testing.assert_allclose(result.power_spectrum, [3.0 + 0.0j])


def test_results_from_run_copies_saved_fields_and_times():
    sim = _simulation(width=1.0, height=1.0)
    field = np.ones((1, 2, 2), dtype=np.float32)
    times = np.array([0.0])
    steps = np.array([1])

    recorder = MonitorResults(
        monitor=FieldRecorder(("Ez",), 1),
        fields={"Ez": field},
        field_times=times,
        field_steps=steps,
        power_history=np.empty(0),
        power_timestamps=np.empty(0),
        power_spectrum=np.empty(0, dtype=np.complex64),
    )
    results = SimulationResults.from_run(
        sim,
        runtime_fields=sim.compile().grid,
        monitor_results={"fields": recorder},
    )
    field[0, 0, 0] = 9.0
    times[0] = 9.0
    steps[0] = 9

    np.testing.assert_allclose(
        results.monitor("fields").fields["Ez"], np.ones((1, 2, 2))
    )
    np.testing.assert_allclose(results.monitor("fields").field_times, [0.0])
    np.testing.assert_allclose(results.monitor("fields").field_steps, [1])
    with pytest.raises(TypeError):
        results.monitor("fields").fields["new"] = np.zeros((1, 2, 2))
    with pytest.raises(ValueError, match="read-only"):
        results.monitor("fields").fields["Ez"].flat[0] = 0.0


def test_results_metadata_does_not_copy_unneeded_live_state():
    sim = _simulation()
    monitor = FieldMonitor(
        center=(0.5, 0.0, 0.0),
        size=(1.0, 0.0, 0.0),
        name="o1",
        freqs=np.array([1.0]),
    )

    results = SimulationResults.from_run(
        sim,
        runtime_fields=sim.compile().grid,
        monitor_results={"o1": _empty_monitor_result(monitor)},
    )
    assert not hasattr(results.metadata, "sources")
    assert not hasattr(results.metadata, "monitors")
