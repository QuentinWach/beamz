import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    ModeMonitor,
    Monitor,
    Port,
    PortSpec,
    Simulation,
    calc_optimal_fdtd_params,
    design,
    dxdt,
)
from beamz.devices.sources.mode import _modal_overlap_3d_profiles
from beamz.simulation.yee import component_shape_3d


def test_dxdt_alias_matches_calc_optimal_fdtd_params():
    wavelength = 1.55e-6
    kwargs = dict(n_max=2.0, dims=2, safety_factor=0.95, points_per_wavelength=8)
    dx_alias, dt_alias = dxdt(wavelength, **kwargs)
    dx_ref, dt_ref = calc_optimal_fdtd_params(wavelength, **kwargs)
    assert dx_alias == dx_ref
    assert dt_alias == dt_ref


@pytest.mark.skipif(
    importlib.util.find_spec("gdsfactory") is None,
    reason="gdsfactory not installed",
)
def test_gdsf_loader_mmi1x2_returns_materialized_design_and_ports():
    loaded_design, ports = design.io.gdsf.load(
        "mmi1x2", layer=(1, 0), n_core=2.0, n_clad=1.44, padding=2.0
    )

    assert loaded_design.width > 0
    assert loaded_design.height > 0
    assert len(loaded_design.structures) > 1  # background + imported polygons

    core_structures = loaded_design.structures[1:]
    assert core_structures
    assert all(
        getattr(struct, "material", None) is not None for struct in core_structures
    )

    assert {"o1", "o2", "o3"}.issubset(ports.keys())
    for port_name in ("o1", "o2", "o3"):
        port = ports[port_name]
        assert "center" in port and len(port["center"]) == 2
        assert port["width"] > 0
        assert port["direction"] in {"+x", "-x", "+y", "-y"}
        cx, cy = port["center"]
        assert 0 <= cx <= loaded_design.width
        assert 0 <= cy <= loaded_design.height


def test_get_S_matrix_proxy_raises_and_points_to_modal_api():
    sim = Simulation.__new__(Simulation)
    with pytest.raises(RuntimeError, match="get_S_matrix_modal"):
        sim.get_S_matrix(input_ports=["o1"], output_ports=["o1"], source_port="o1")
    with pytest.raises(RuntimeError, match="get_s_matrix_modal"):
        sim.get_s_matrix(input_ports=["o1"], output_ports=["o1"], source_port="o1")


def test_port_abstraction_derives_wave_selectors_from_incoming_direction():
    source = Port(name="o1", monitor="m1", direction="+x", polarization="tm")
    output = Port(name="o2", monitor="m2", direction="-x", polarization="tm")
    cross = Port(name="o3", monitor="m3", direction="+y", polarization="tm")

    specs = Simulation._normalize_portspecs([source, output, cross])

    assert specs["o1"].monitor_name == "m1"
    assert specs["o1"].direction == "+x"
    assert specs["o1"].incident_wave == "minus"
    assert specs["o1"].scattered_wave == "plus"
    assert specs["o2"].monitor_name == "m2"
    assert specs["o2"].direction == "-x"
    assert specs["o2"].incident_wave == "minus"
    assert specs["o2"].scattered_wave == "plus"
    assert specs["o3"].monitor_name == "m3"
    assert specs["o3"].direction == "+y"
    assert specs["o3"].incident_wave == "minus"
    assert specs["o3"].scattered_wave == "plus"


def test_mode_monitor_is_first_class_port_metadata():
    freqs = np.array([1.0], dtype=float)
    mon = ModeMonitor(
        start=(0.0, 0.0),
        end=(0.0, 1.0),
        name="o2",
        direction="-x",
        polarization="tm",
        dft_frequencies=freqs,
        record_fields=False,
    )

    assert mon.dft_enabled
    assert set(mon.dft_components) == {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
    spec = Simulation._normalize_portspecs([mon])["o2"]
    assert spec.monitor_name == "o2"
    assert spec.direction == "-x"
    assert spec.incident_wave == "minus"
    assert spec.scattered_wave == "plus"


def test_get_s_matrix_modal_dft_accepts_mode_monitor_ports(monkeypatch):
    freqs = np.array([1.0], dtype=float)
    src = ModeMonitor(
        start=(0.0, 0.0),
        end=(0.0, 1.0),
        name="o1",
        direction="+x",
        polarization="tm",
        reference_monitor="o1_ref",
        dft_frequencies=freqs,
        record_fields=False,
    )
    out = ModeMonitor(
        start=(1.0, 0.0),
        end=(1.0, 1.0),
        name="o2",
        direction="-x",
        polarization="tm",
        dft_frequencies=freqs,
        record_fields=False,
    )

    waves = {
        "o1": {
            "a_plus": np.array([0.05], dtype=np.complex128),
            "a_minus": np.array([0.01], dtype=np.complex128),
            "a_incident_plus": np.array([0.0], dtype=np.complex128),
            "a_incident_minus": np.array([2.0], dtype=np.complex128),
        },
        "o2": {
            "a_plus": np.array([1.4], dtype=np.complex128),
            "a_minus": np.array([0.2], dtype=np.complex128),
        },
    }

    def fake_extract(
        self,
        ports,
        frequencies,
        min_incident_db=-40.0,
        return_power=True,
        mode_strategy="per_frequency",
    ):
        del self, min_incident_db, return_power, mode_strategy
        np.testing.assert_allclose(frequencies, freqs)
        by_name = {p.name: p for p in ports}
        assert by_name["o1"].reference_monitor == "o1_ref"
        assert by_name["o1"].incident_wave == "minus"
        assert by_name["o1"].scattered_wave == "plus"
        assert by_name["o2"].incident_wave == "minus"
        assert by_name["o2"].scattered_wave == "plus"
        return waves

    sim = Simulation.__new__(Simulation)
    sim.sources = []
    sim.monitors = [src, Monitor(start=(0.0, 0.0), end=(0.0, 1.0), name="o1_ref"), out]
    sim.is_3d = False
    sim.plane_2d = "xy"
    monkeypatch.setattr(Simulation, "extract_port_waves_dft", fake_extract)

    result = sim.get_S_matrix_modal_dft(
        source_port=src,
        ports=[src, out],
        output_ports=[out],
        frequencies=freqs,
        as_sax=False,
        return_diagnostics=True,
    )

    np.testing.assert_allclose(
        result["s_matrix"][("o2", "o1")],
        np.array([0.7], dtype=np.complex128),
        rtol=1e-12,
        atol=1e-12,
    )
    assert result["diagnostics"]["source_reference_normalization"] == {
        "enabled": True,
        "monitor": "o1_ref",
        "incident_wave": "minus",
        "scattered_wave": "plus",
    }


def test_extract_port_waves_modal_coefficients_synthetic(monkeypatch):
    freqs = np.array([1.0, 2.0], dtype=float)
    mode_matrix = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    pinv = np.linalg.pinv(mode_matrix)

    a_src = np.array([1.5 + 0.1j, 1.6 - 0.2j])
    b_src = np.array([0.1 - 0.02j, 0.05 + 0.03j])
    a_ref = np.array([2.0 + 0.0j, 2.1 + 0.0j])
    b_ref = np.zeros_like(a_ref)
    a_out = np.array([0.02 + 0.01j, -0.01 + 0.02j])
    b_out = np.array([0.7 - 0.2j, 0.6 + 0.1j])

    spectral_map = {
        ("m_src", "Ez"): (a_src + b_src)[:, None],
        ("m_src", "Hy"): (a_src - b_src)[:, None],
        ("m_ref", "Ez"): (a_ref + b_ref)[:, None],
        ("m_ref", "Hy"): (a_ref - b_ref)[:, None],
        ("m_out", "Ez"): (a_out + b_out)[:, None],
        ("m_out", "Hy"): (a_out - b_out)[:, None],
    }

    def fake_sample(self, monitor, component, frequencies=None, window="hann"):
        assert frequencies is not None
        return (
            np.asarray(frequencies, dtype=float),
            spectral_map[(monitor.name, component)],
        )

    def fake_projection(
        self,
        spec,
        monitor,
        frequency,
        cache,
        mode_pad_cells=6,
        previous_projection=None,
    ):
        del spec, monitor, frequency, cache, mode_pad_cells, previous_projection
        return {"e_component": "Ez", "h_component": "Hy", "pinv": pinv}

    monkeypatch.setattr(Simulation, "_sample_monitor_component_spectrum", fake_sample)
    monkeypatch.setattr(Simulation, "_build_port_projection", fake_projection)

    sim = Simulation.__new__(Simulation)
    sim.is_3d = False
    sim.plane_2d = "xy"
    sim.resolution = 1.0
    sim.sources = []
    sim.monitors = [
        Monitor(start=(0.0, 0.0), end=(0.0, 1.0), name="m_src"),
        Monitor(start=(0.0, 0.0), end=(0.0, 1.0), name="m_ref"),
        Monitor(start=(1.0, 0.0), end=(1.0, 1.0), name="m_out"),
    ]

    ports = [
        PortSpec(
            name="o1",
            monitor_name="m_src",
            reference_monitor="m_ref",
            direction="+x",
            polarization="tm",
        ),
        PortSpec(name="o2", monitor_name="m_out", direction="+x", polarization="tm"),
    ]
    waves = sim.extract_port_waves(
        ports=ports, frequencies=freqs, mode_strategy="per_frequency"
    )

    np.testing.assert_allclose(waves["o1"]["a_plus"], a_src, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(waves["o1"]["a_minus"], b_src, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(waves["o1"]["a_incident"], a_ref, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(waves["o2"]["a_plus"], a_out, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(waves["o2"]["a_minus"], b_out, rtol=1e-9, atol=1e-9)


def test_get_S_matrix_modal_column_keys_and_shapes(monkeypatch):
    sim = Simulation.__new__(Simulation)
    sim.sources = []
    sim.monitors = []
    sim.is_3d = False
    sim.plane_2d = "xy"

    freqs = np.array([1.0, 2.0, 3.0], dtype=float)
    waves = {
        "o1": {
            "a_plus": np.ones_like(freqs, dtype=np.complex128),
            "a_minus": 0.1 * np.ones_like(freqs, dtype=np.complex128),
            "a_incident": 2.0 * np.ones_like(freqs, dtype=np.complex128),
        },
        "o2": {
            "a_plus": np.zeros_like(freqs, dtype=np.complex128),
            "a_minus": 1.0j * np.ones_like(freqs, dtype=np.complex128),
        },
    }

    def fake_extract(
        self,
        ports,
        frequencies,
        mode_strategy="per_frequency",
        window="hann",
        return_power=True,
    ):
        return waves

    monkeypatch.setattr(Simulation, "extract_port_waves", fake_extract)
    result = sim.get_S_matrix_modal(
        source_port="o1",
        ports=[
            PortSpec(
                name="o1",
                monitor_name="o1",
                reference_monitor="o1_ref",
                direction="+x",
                polarization="tm",
            ),
            PortSpec(name="o2", monitor_name="o2", direction="+x", polarization="tm"),
        ],
        output_ports=["o1", "o2"],
        frequencies=freqs,
        as_sax=False,
        return_diagnostics=True,
    )

    s_matrix = result["s_matrix"]
    assert set(s_matrix.keys()) == {("o1", "o1"), ("o2", "o1")}
    assert s_matrix[("o1", "o1")].shape == freqs.shape
    assert s_matrix[("o2", "o1")].shape == freqs.shape
    assert result["diagnostics"]["power_sum"].shape == freqs.shape


def test_cw_demod_recovers_complex_amplitude():
    n = 2048
    dt = 1e-15
    k_bin = 37
    freq = k_bin / (n * dt)
    amp = 0.65
    phase = 0.42
    t = np.arange(n) * dt
    trace = amp * np.cos(2 * np.pi * freq * t + phase)

    mon = Monitor(start=(0.0, 0.0), end=(0.0, 1.0), name="m", record_fields=True)
    mon.fields["Ez"] = [[v] for v in trace]
    mon.fields["t"] = list(t)

    sim = Simulation.__new__(Simulation)
    demod = sim._demodulate_monitor_component(
        mon,
        "Ez",
        frequency=freq,
        t_start=None,
        avg_cycles=30,
        window="none",
    )
    assert demod.shape == (1,)
    assert np.isclose(np.abs(demod[0]), amp, rtol=0.03)
    phase_err = np.angle(demod[0] / (amp * np.exp(-1j * phase)))
    assert abs(phase_err) < 0.08


def test_extract_port_waves_cw_modal_coefficients_synthetic(monkeypatch):
    mode_matrix = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    pinv = np.linalg.pinv(mode_matrix)

    a_src = 1.5 + 0.1j
    b_src = 0.1 - 0.02j
    a_ref = 2.0 + 0.0j
    b_ref = 0.0 + 0.0j
    a_out = 0.02 + 0.01j
    b_out = 0.7 - 0.2j

    demod_map = {
        ("m_src", "Ez"): np.array([a_src + b_src], dtype=np.complex128),
        ("m_src", "Hy"): np.array([a_src - b_src], dtype=np.complex128),
        ("m_ref", "Ez"): np.array([a_ref + b_ref], dtype=np.complex128),
        ("m_ref", "Hy"): np.array([a_ref - b_ref], dtype=np.complex128),
        ("m_out", "Ez"): np.array([a_out + b_out], dtype=np.complex128),
        ("m_out", "Hy"): np.array([a_out - b_out], dtype=np.complex128),
    }

    def fake_demod(
        self, monitor, component, frequency, t_start=None, avg_cycles=12, window="hann"
    ):
        return demod_map[(monitor.name, component)]

    def fake_projection(self, spec, monitor, frequency, cache, mode_pad_cells=6):
        return {"e_component": "Ez", "h_component": "Hy", "pinv": pinv}

    monkeypatch.setattr(Simulation, "_demodulate_monitor_component", fake_demod)
    monkeypatch.setattr(Simulation, "_build_port_projection", fake_projection)

    sim = Simulation.__new__(Simulation)
    sim.is_3d = False
    sim.plane_2d = "xy"
    sim.resolution = 1.0
    sim.sources = []
    sim.monitors = [
        Monitor(start=(0.0, 0.0), end=(0.0, 1.0), name="m_src"),
        Monitor(start=(0.0, 0.0), end=(0.0, 1.0), name="m_ref"),
        Monitor(start=(1.0, 0.0), end=(1.0, 1.0), name="m_out"),
    ]
    ports = [
        PortSpec(
            name="o1",
            monitor_name="m_src",
            reference_monitor="m_ref",
            direction="+x",
            polarization="tm",
        ),
        PortSpec(name="o2", monitor_name="m_out", direction="+x", polarization="tm"),
    ]
    waves = sim.extract_port_waves_cw(
        ports=ports,
        frequency=200e12,
        steady_start_time=None,
        avg_cycles=12,
        window="hann",
        mode_strategy="per_frequency",
    )
    assert np.isclose(waves["o1"]["a_plus"], a_src)
    assert np.isclose(waves["o1"]["a_minus"], b_src)
    assert np.isclose(waves["o1"]["a_incident"], a_ref)
    assert np.isclose(waves["o2"]["a_plus"], a_out)
    assert np.isclose(waves["o2"]["a_minus"], b_out)


def test_get_S_matrix_modal_cw_shapes_and_keys(monkeypatch):
    sim = Simulation.__new__(Simulation)
    sim.sources = []
    sim.monitors = []
    sim.is_3d = False
    sim.plane_2d = "xy"
    waves = {
        "o1": {"a_plus": 1.0 + 0j, "a_minus": 0.1 + 0j, "a_incident": 2.0 + 0j},
        "o2": {"a_plus": 0.0 + 0j, "a_minus": 0.0 + 1.0j},
    }

    def fake_extract(
        self,
        ports,
        frequency,
        steady_start_time=None,
        avg_cycles=12,
        window="hann",
        mode_strategy="per_frequency",
        return_power=True,
    ):
        return waves

    monkeypatch.setattr(Simulation, "extract_port_waves_cw", fake_extract)
    result = sim.get_S_matrix_modal_cw(
        source_port="o1",
        ports=[
            PortSpec(
                name="o1",
                monitor_name="o1",
                reference_monitor="o1_ref",
                direction="+x",
                polarization="tm",
            ),
            PortSpec(name="o2", monitor_name="o2", direction="+x", polarization="tm"),
        ],
        output_ports=["o1", "o2"],
        frequency=193.4e12,
        as_sax=False,
        return_diagnostics=True,
    )
    s_matrix = result["s_matrix"]
    assert set(s_matrix.keys()) == {("o1", "o1"), ("o2", "o1")}
    assert np.iscomplexobj(s_matrix[("o1", "o1")])
    assert isinstance(result["diagnostics"]["power_sum"], float)


def test_monitor_dft_accum_recovers_known_sinusoid():
    n = 2048
    dt = 1e-15
    k_bin = 31
    freq = k_bin / (n * dt)
    amp = 0.55
    phase = 0.35
    t = np.arange(n, dtype=float) * dt

    mon = Monitor(
        start=(0.0, 0.0),
        end=(0.0, 0.0),
        name="m_dft",
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=np.array([freq], dtype=float),
        dft_t_start=float(t[0]),
        dft_t_end=float(t[-1]),
        dft_components=("Ez",),
        dft_window="rect",
    )
    for i, ti in enumerate(t):
        sample = amp * np.cos(2 * np.pi * freq * ti + phase)
        mon.record_fields_2d(
            Ez=np.array([[sample]], dtype=float),
            Hx=np.zeros((1, 1), dtype=float),
            Hy=np.zeros((1, 1), dtype=float),
            t=float(ti),
            dx=1.0,
            dy=1.0,
            step=i,
        )

    recovered = mon.get_dft_component("Ez")
    assert recovered.shape == (1, 1)
    z = recovered[0, 0]
    assert np.isclose(np.abs(z), amp, rtol=0.03)
    phase_err = np.angle(z / (amp * np.exp(-1j * phase)))
    assert abs(phase_err) < 0.08


def test_monitor_dft_accum_preserves_absolute_phase_for_delayed_start():
    n = 2048
    dt = 1e-15
    k_bin = 19
    freq = k_bin / (n * dt)
    amp = 0.42
    phase = -0.27
    t0 = 137.0 * dt
    t = t0 + np.arange(n, dtype=float) * dt

    mon = Monitor(
        start=(0.0, 0.0),
        end=(0.0, 0.0),
        name="m_dft_offset",
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=np.array([freq], dtype=float),
        dft_t_start=float(t[0]),
        dft_t_end=float(t[-1]),
        dft_components=("Ez",),
        dft_window="rect",
    )
    for i, ti in enumerate(t):
        sample = amp * np.cos(2 * np.pi * freq * ti + phase)
        mon.record_fields_2d(
            Ez=np.array([[sample]], dtype=float),
            Hx=np.zeros((1, 1), dtype=float),
            Hy=np.zeros((1, 1), dtype=float),
            t=float(ti),
            dx=1.0,
            dy=1.0,
            step=i,
        )

    recovered = mon.get_dft_component("Ez")
    assert recovered.shape == (1, 1)
    z = recovered[0, 0]
    assert np.isclose(np.abs(z), amp, rtol=0.03)
    phase_err = np.angle(z / (amp * np.exp(-1j * phase)))
    assert abs(phase_err) < 0.08


def test_monitor_get_dft_component_returns_canonical_matrix_shape():
    mon = Monitor(
        start=(0.0, 0.0),
        end=(0.0, 0.0),
        name="m_shape",
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=np.array([1.0, 2.0], dtype=float),
        dft_components=("Ez",),
    )
    raw = (np.arange(12, dtype=float) + 1j * np.arange(12, dtype=float)).reshape(
        2, 2, 3
    )
    mon._dft_accum["Ez"] = raw
    mon._dft_weight_sum = np.ones((2,), dtype=float)

    dft = mon.get_dft_component("Ez")
    assert dft.shape == (2, 6)
    np.testing.assert_allclose(dft, 2.0 * raw.reshape(2, 6), rtol=1e-12, atol=1e-12)


def test_monitor_get_dft_component_raises_for_invalid_frequency_axis():
    mon = Monitor(
        start=(0.0, 0.0),
        end=(0.0, 0.0),
        name="m_bad_shape",
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=np.array([1.0, 2.0], dtype=float),
        dft_components=("Ez",),
    )
    mon._dft_accum["Ez"] = np.zeros((3, 4), dtype=np.complex128)
    mon._dft_weight_sum = np.ones((2,), dtype=float)

    with pytest.raises(ValueError, match="axis 0"):
        mon.get_dft_component("Ez")


def test_monitor_get_dft_component_physical_mode_returns_raw_accumulator():
    mon = Monitor(
        start=(0.0, 0.0),
        end=(0.0, 0.0),
        name="m_physical_shape",
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=np.array([1.0, 2.0], dtype=float),
        dft_components=("Ez",),
        dft_normalization="physical",
    )
    raw = (np.arange(12, dtype=float) + 1j * np.arange(12, dtype=float)).reshape(
        2, 2, 3
    )
    mon._dft_accum["Ez"] = raw
    mon._dft_weight_sum = np.array([17.0, 29.0], dtype=float)

    dft = mon.get_dft_component("Ez")
    assert dft.shape == (2, 6)
    np.testing.assert_allclose(dft, raw.reshape(2, 6), rtol=1e-12, atol=1e-12)


def test_monitor_power_history_records_signed_normal_flux():
    forward = Monitor(start=(0.0, 0.0), end=(0.0, 2.0), record_fields=False)
    forward._calculate_power_2d(
        [2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0],
        [-3.0, -3.0, -3.0],
        t=0.0,
        dx=2.0,
        dy=3.0,
    )
    assert forward.power_history[-1] == pytest.approx(54.0)

    reverse = Monitor(start=(0.0, 2.0), end=(0.0, 0.0), record_fields=False)
    reverse._calculate_power_2d(
        [2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0],
        [-3.0, -3.0, -3.0],
        t=0.0,
        dx=2.0,
        dy=3.0,
    )
    assert reverse.power_history[-1] == pytest.approx(-54.0)


def test_monitor_get_dft_flux_uses_phasor_poynting_product():
    mon = Monitor(
        start=(0.0, 0.0),
        end=(0.0, 1.0),
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=np.array([1.0], dtype=float),
        dft_components=("Ez", "Hy"),
        dft_normalization="physical",
    )
    mon._resolution = 2.0
    ez = np.array([[2.0 + 1.0j, 1.0 - 0.5j]], dtype=np.complex128)
    hy = np.array([[-3.0 + 0.5j, -2.0 - 1.0j]], dtype=np.complex128)
    mon._dft_accum["Ez"] = ez
    mon._dft_accum["Hy"] = hy
    mon._dft_weight_sum = np.ones((1,), dtype=float)

    expected = 0.5 * np.real(np.sum(-ez * np.conjugate(hy), axis=1)) * 2.0
    np.testing.assert_allclose(mon.get_dft_flux(), expected, rtol=1e-12, atol=1e-12)


def test_monitor_get_dft_flux_phase_aligns_leapfrog_h_phasor():
    mon = Monitor(
        start=(0.0, 0.0),
        end=(0.0, 1.0),
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=np.array([0.5], dtype=float),
        dft_components=("Ez", "Hy"),
        dft_normalization="physical",
    )
    mon._resolution = 1.0
    mon._dft_base_dt = 1.0
    mon._dft_accum["Ez"] = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    mon._dft_accum["Hy"] = np.array([[1.0j]], dtype=np.complex128)
    mon._dft_weight_sum = np.ones((1,), dtype=float)

    np.testing.assert_allclose(mon.get_dft_flux(), [-0.5], rtol=1e-12, atol=1e-12)


def test_monitor_get_dft_flux_3d_uses_analysis_plane_sample_area():
    dx = 0.03569482288828337e-6
    mon = Monitor(
        start=(1.0e-6, 0.0, 0.0),
        end=(1.0e-6, 1.4e-6, 0.644e-6),
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=np.array([1.0], dtype=float),
        dft_components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        dft_normalization="physical",
        name="flux3d",
    )
    mon._resolution = dx
    target0, target1 = mon.get_analysis_plane_coords_3d(
        dx=dx,
        dy=dx,
        dz=dx,
        field_shape=(56, 384, 384),
    )
    npts = int(target0.size * target1.size)
    area = mon._dft_sample_area_3d

    zeros = np.zeros((1, npts), dtype=np.complex128)
    mon._dft_accum["Ex"] = zeros.copy()
    mon._dft_accum["Ey"] = np.full((1, npts), 2.0 + 0.0j, dtype=np.complex128)
    mon._dft_accum["Ez"] = zeros.copy()
    mon._dft_accum["Hx"] = zeros.copy()
    mon._dft_accum["Hy"] = zeros.copy()
    mon._dft_accum["Hz"] = np.full((1, npts), 3.0 + 0.0j, dtype=np.complex128)
    mon._dft_weight_sum = np.ones((1,), dtype=float)

    expected = 0.5 * 2.0 * 3.0 * npts * area
    wrong_dx_squared = expected * (dx * dx / area)

    np.testing.assert_allclose(mon.get_dft_flux(), [expected], rtol=1e-12, atol=1e-12)
    assert abs(float(mon.get_dft_flux()[0]) - wrong_dx_squared) / expected > 0.05


def test_monitor_frequency_points_aliases_are_deprecated():
    with pytest.warns(DeprecationWarning, match="frequency_points"):
        mon = Monitor(
            start=(0.0, 0.0),
            end=(0.0, 1.0),
            frequency_points=[1.0, 2.0],
            record_fields=False,
        )
    np.testing.assert_allclose(mon.power_spectrum_frequencies, [1.0, 2.0])

    with pytest.warns(DeprecationWarning, match="frequency_points"):
        np.testing.assert_allclose(mon.frequency_points, [1.0, 2.0])

    mon.power_spectrum = np.array([1.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex64)
    with pytest.warns(DeprecationWarning, match="frequency_flux_spectrum"):
        np.testing.assert_allclose(mon.frequency_flux_spectrum, mon.power_spectrum)


def test_monitor_physical_dft_accum_matches_direct_sum():
    n = 512
    dt = 1e-15
    k_bin = 23
    freq = k_bin / (n * dt)
    amp = 0.37
    phase = -0.21
    times = (np.arange(n, dtype=float) + 1.0) * dt

    mon = Monitor(
        start=(0.0, 0.0),
        end=(0.0, 0.0),
        name="m_physical_dft",
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=np.array([freq], dtype=float),
        dft_components=("Ez",),
        dft_window="rect",
        dft_normalization="physical",
        dft_length_unit=1e-6,
    )
    mon._dft_base_dt = dt

    for i, ti in enumerate(times):
        sample = amp * np.cos(2 * np.pi * freq * ti + phase)
        mon.record_fields_2d(
            Ez=np.array([[sample]], dtype=float),
            Hx=np.zeros((1, 1), dtype=float),
            Hy=np.zeros((1, 1), dtype=float),
            t=float(ti),
            dx=1.0,
            dy=1.0,
            step=i,
        )

    recovered = mon.get_dft_component("Ez")
    assert recovered.shape == (1, 1)
    scale = (dt * LIGHT_SPEED / 1e-6) / np.sqrt(2.0 * np.pi)
    expected = scale * np.sum(
        amp
        * np.cos(2 * np.pi * freq * times + phase)
        * np.exp(1j * 2.0 * np.pi * freq * times)
    )
    np.testing.assert_allclose(
        recovered[0, 0],
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_resample_complex_matrix_flattens_trailing_spatial_dims():
    freq_src = np.array([1.0, 2.0], dtype=float)
    freq_dst = np.array([1.0, 2.0], dtype=float)
    src = (np.arange(12, dtype=float) + 1j * np.arange(12, dtype=float)).reshape(
        2, 2, 3
    )

    out = Simulation._resample_complex_matrix(freq_src, src, freq_dst)
    assert out.shape == (2, 6)
    np.testing.assert_allclose(out, src.reshape(2, 6), rtol=1e-12, atol=1e-12)


def test_monitor_projection_phase_uses_yee_half_step_h_lag():
    dt = 2.0e-15
    freqs = np.array([1.0e12, 4.0e12], dtype=float)

    np.testing.assert_allclose(
        Simulation._monitor_projection_phase("Ez", freqs, dt),
        np.ones_like(freqs, dtype=np.complex128),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        Simulation._monitor_projection_phase("Hy", freqs, dt),
        np.exp(-1j * np.pi * freqs * dt),
        rtol=1e-12,
        atol=1e-12,
    )


def test_modal_projection_spatial_phase_advances_e_to_h_reference_plane():
    freqs = np.array([1.0e12, 4.0e12], dtype=float)
    plane_delay_s = 0.25e-15

    np.testing.assert_allclose(
        Simulation._modal_projection_spatial_phase("Ez", freqs, plane_delay_s),
        np.exp(1j * 2.0 * np.pi * freqs * plane_delay_s),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        Simulation._modal_projection_spatial_phase("Hy", freqs, plane_delay_s),
        np.ones_like(freqs, dtype=np.complex128),
        rtol=1e-12,
        atol=1e-12,
    )


def test_modal_projection_plane_delay_matches_2d_yee_half_cell_delay():
    sim = Simulation.__new__(Simulation)
    sim.is_3d = False
    sim.resolution = 80e-9
    spec = PortSpec(name="o1", monitor_name="m", direction="+x", polarization="tm")
    delay = sim._modal_projection_plane_delay_s(spec, 193.4e12, 1.8)
    assert delay == pytest.approx(0.5 * sim.resolution * 1.8 / LIGHT_SPEED)

    spec_back = PortSpec(name="o2", monitor_name="m", direction="-x", polarization="tm")
    delay_back = sim._modal_projection_plane_delay_s(spec_back, 193.4e12, 1.8)
    assert delay_back == pytest.approx(-delay)


def test_modal_projection_plane_delay_is_zero_for_3d_colocated_monitors():
    sim = Simulation.__new__(Simulation)
    sim.is_3d = True
    sim.resolution = 80e-9
    sim.dt = 0.1e-15
    spec = PortSpec(name="o1", monitor_name="m", direction="+y", polarization="te")

    delay = sim._modal_projection_plane_delay_s(spec, 193.4e12, 2.4)

    assert delay == pytest.approx(0.0)


def test_sample_monitor_component_dft_applies_yee_phase_to_h_only():
    dt = 2.0e-15
    freqs = np.array([1.0e12, 3.0e12], dtype=float)
    sim = Simulation.__new__(Simulation)
    sim.dt = dt
    mon = Monitor(
        start=(0.0, 0.0),
        end=(0.0, 0.0),
        name="m_yee_phase",
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=freqs,
        dft_components=("Ez", "Hy"),
        dft_normalization="physical",
    )
    raw_e = np.array([[1.0 + 0.25j], [0.5 - 0.75j]], dtype=np.complex128)
    raw_h = np.array([[0.25 - 1.0j], [-0.75 + 0.5j]], dtype=np.complex128)
    mon._dft_accum["Ez"] = raw_e
    mon._dft_accum["Hy"] = raw_h
    mon._dft_weight_sum = np.ones(freqs.shape, dtype=float)

    _, sampled_e = sim._sample_monitor_component_dft(mon, "Ez", freqs)
    _, sampled_h = sim._sample_monitor_component_dft(mon, "Hy", freqs)

    np.testing.assert_allclose(sampled_e, raw_e, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        sampled_h,
        raw_h * np.exp(-1j * np.pi * freqs * dt)[:, None],
        rtol=1e-12,
        atol=1e-12,
    )


def test_project_modal_coefficients_3d_recovers_forward_backward_modes():
    mode_components = {
        "Ey": np.array([1.0 + 0.0j, 0.5 + 0.2j], dtype=np.complex128),
        "Ez": np.array([0.2 + 0.1j, -0.1 + 0.3j], dtype=np.complex128),
        "Hy": np.array([0.6 - 0.1j, -0.2 + 0.2j], dtype=np.complex128),
        "Hz": np.array([0.4 + 0.3j, 0.1 - 0.2j], dtype=np.complex128),
        "Ex": np.zeros((2,), dtype=np.complex128),
        "Hx": np.zeros((2,), dtype=np.complex128),
    }
    components = ("Ey", "Ez", "Hy", "Hz")
    fwd_vec = np.concatenate([mode_components[c] for c in components])
    bwd_vec = np.concatenate(
        [
            (-mode_components[c] if c.startswith("H") else mode_components[c])
            for c in components
        ]
    )
    mode_matrix = np.column_stack([fwd_vec, bwd_vec])
    projection = {
        "components": components,
        "axis": "x",
        "d_area": 1.0,
        "mode_components": mode_components,
        "pinv": np.linalg.pinv(mode_matrix),
    }

    a_p, a_m = Simulation._project_modal_coefficients_3d(mode_components, projection)
    np.testing.assert_allclose(a_p, 1.0 + 0.0j, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(a_m, 0.0 + 0.0j, rtol=1e-10, atol=1e-10)

    a_true = 0.7 - 0.3j
    b_true = 0.25 + 0.15j
    mixed_fields = {}
    for k, v in mode_components.items():
        if k.startswith("H"):
            mixed_fields[k] = a_true * v - b_true * v
        else:
            mixed_fields[k] = a_true * v + b_true * v
    a_p2, a_m2 = Simulation._project_modal_coefficients_3d(mixed_fields, projection)
    np.testing.assert_allclose(a_p2, a_true, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(a_m2, b_true, rtol=1e-10, atol=1e-10)
    mixed_vec = np.concatenate([mixed_fields[c] for c in components])
    coeff_pinv = projection["pinv"] @ mixed_vec
    np.testing.assert_allclose(coeff_pinv[0], a_true, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(coeff_pinv[1], b_true, rtol=1e-10, atol=1e-10)


def test_project_modal_coefficients_3d_is_linear_in_field_amplitude():
    components = ("Ey", "Ez", "Hy", "Hz")
    base = {
        "Ey": np.array([1.0 + 0.0j], dtype=np.complex128),
        "Ez": np.array([0.5 + 0.2j], dtype=np.complex128),
        "Hy": np.array([0.3 - 0.1j], dtype=np.complex128),
        "Hz": np.array([0.4 + 0.05j], dtype=np.complex128),
    }
    fwd_vec = np.concatenate([base[c] for c in components])
    bwd_vec = np.concatenate(
        [(-base[c] if c.startswith("H") else base[c]) for c in components]
    )
    projection = {
        "components": components,
        "axis": "x",
        "d_area": 1.0,
        "mode_components": {
            "Ex": np.zeros((1,), dtype=np.complex128),
            "Ey": base["Ey"].copy(),
            "Ez": base["Ez"].copy(),
            "Hx": np.zeros((1,), dtype=np.complex128),
            "Hy": base["Hy"].copy(),
            "Hz": base["Hz"].copy(),
        },
        "pinv": np.linalg.pinv(np.column_stack([fwd_vec, bwd_vec])),
    }

    scale = 2.75 - 0.4j
    scaled = {k: scale * v for k, v in base.items()}
    a_p, a_m = Simulation._project_modal_coefficients_3d(scaled, projection)
    np.testing.assert_allclose(a_p, scale, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(a_m, 0.0 + 0.0j, rtol=1e-10, atol=1e-10)


def test_project_modal_coefficients_3d_uses_rhs_orientation_for_complex_basis():
    mode_components = {
        "Ex": np.zeros((2,), dtype=np.complex128),
        "Ey": np.array([1.0 + 0.2j, 0.3 - 0.1j], dtype=np.complex128),
        "Ez": np.array([0.2 + 0.1j, -0.1 + 0.4j], dtype=np.complex128),
        "Hx": np.zeros((2,), dtype=np.complex128),
        "Hy": np.array([0.05 - 0.3j, 0.2 + 0.1j], dtype=np.complex128),
        "Hz": np.array([0.9 - 0.1j, -0.2 + 0.5j], dtype=np.complex128),
    }
    mode_components_bwd = {
        name: (-arr if name.startswith("H") else arr.copy())
        for name, arr in mode_components.items()
    }
    overlap = np.asarray(
        [
            [
                _modal_overlap_3d_profiles(mode_components, mode_components, "x", 0.1),
                _modal_overlap_3d_profiles(
                    mode_components, mode_components_bwd, "x", 0.1
                ),
            ],
            [
                _modal_overlap_3d_profiles(
                    mode_components_bwd, mode_components, "x", 0.1
                ),
                _modal_overlap_3d_profiles(
                    mode_components_bwd, mode_components_bwd, "x", 0.1
                ),
            ],
        ],
        dtype=np.complex128,
    )
    projection = {
        "components": ("Ey", "Ez", "Hy", "Hz"),
        "axis": "x",
        "d_area": 0.1,
        "mode_components": mode_components,
        "mode_components_bwd": mode_components_bwd,
        "overlap_matrix": overlap,
        "pinv": np.zeros((2, 0), dtype=np.complex128),
    }
    a_true = 0.7 + 0.1j
    b_true = -0.2 + 0.3j
    mixed_fields = {
        name: a_true * mode_components[name] + b_true * mode_components_bwd[name]
        for name in mode_components
    }

    a_p, a_m = Simulation._project_modal_coefficients_3d(mixed_fields, projection)

    np.testing.assert_allclose(a_p, a_true, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(a_m, b_true, rtol=1e-10, atol=1e-10)


def test_interpolate_plane_matrix_2d_preserves_complex_affine_plane():
    src0 = np.array([0.0, 1.0, 2.0], dtype=float)
    src1 = np.array([0.0, 2.0, 4.0], dtype=float)
    dst0 = np.array([0.25, 1.5], dtype=float)
    dst1 = np.array([0.5, 3.0], dtype=float)

    def plane(c0, c1):
        return 2.0 + 3.0 * c0 - 0.5 * c1 + 1j * (-1.0 + 0.25 * c0 + 2.0 * c1)

    values = plane(src0[:, None], src1[None, :])
    expected = plane(dst0[:, None], dst1[None, :])

    actual = Simulation._interpolate_plane_matrix_2d(values, src0, src1, dst0, dst1)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_colocate_field_components_to_projection_3d_respects_yee_offsets():
    sim = Simulation.__new__(Simulation)
    sim.is_3d = True
    sim.resolution = 1.0
    grid_shape = (5, 5, 5)
    field_arrays = {
        comp: np.zeros(component_shape_3d(comp, grid_shape), dtype=np.float32)
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    field_arrays["permittivity"] = np.ones(grid_shape, dtype=np.float32)
    sim.fields = type("F", (), field_arrays)()

    class DummyXMonitor:
        name = "yee_x"

        @staticmethod
        def get_grid_slice_3d(dx, dy, dz, field_shape):
            del dx, dy, dz
            normal_index = min(1, int(field_shape[2]) - 1)
            return slice(1, 4), slice(1, 4), normal_index

    def plane(c0, c1):
        return 0.25 + 0.75 * c0 - 0.4 * c1 + 1j * (1.5 - 0.2 * c0 + 0.6 * c1)

    monitor = DummyXMonitor()
    target0 = np.array([1.5, 2.0, 2.5], dtype=float)
    target1 = np.array([1.5, 2.0, 2.5], dtype=float)
    projection = {
        "axis": "x",
        "analysis_coords0": target0,
        "analysis_coords1": target1,
    }
    field_components = {}
    for comp in ("Ey", "Ez", "Hy", "Hz"):
        src0, src1 = sim._monitor_component_plane_coords_3d(monitor, comp, "x")
        field_components[comp] = plane(src0[:, None], src1[None, :]).reshape(-1)

    colocated = sim._colocate_field_components_to_projection_3d(
        monitor,
        field_components,
        projection,
    )

    expected = plane(target0[:, None], target1[None, :]).reshape(-1)
    for comp in ("Ey", "Ez", "Hy", "Hz"):
        np.testing.assert_allclose(colocated[comp], expected, rtol=1e-13, atol=1e-13)


def test_build_port_projection_3d_staggers_solver_fields_to_yee_lattices(
    monkeypatch,
):
    import beamz.simulation.core as core_mod

    sim = Simulation.__new__(Simulation)
    sim.is_3d = True
    sim.resolution = 1.0
    grid_shape = (4, 4, 4)
    field_arrays = {
        comp: np.zeros(component_shape_3d(comp, grid_shape), dtype=np.float32)
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    zz, yy, xx = np.meshgrid(
        np.arange(grid_shape[0]),
        np.arange(grid_shape[1]),
        np.arange(grid_shape[2]),
        indexing="ij",
    )
    field_arrays["permittivity"] = (1.0 + 0.8 * ((zz + 2 * yy + 3 * xx) % 5)).astype(
        np.float32
    )
    sim.fields = type("F", (), field_arrays)()

    class DummyXMonitor:
        name = "m_yee_projection"

        @staticmethod
        def get_grid_slice_3d(dx, dy, dz, field_shape):
            del dx, dy, dz
            normal_index = min(1, int(field_shape[2]) - 1)
            return slice(0, 4), slice(0, 4), normal_index

        @staticmethod
        def get_dft_component(component):
            del component
            return np.zeros((1, 9), dtype=np.complex128)

    z, y = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    ex = 10.0 + z + 0.1 * y + 0.03 * z * y
    ey = 20.0 + 2.0 * z - 0.3 * y + 0.11 * z * y
    ez = 30.0 - 0.4 * z + 1.5 * y - 0.07 * z * y
    hx = 40.0 + 0.8 * z + 0.6 * y + 0.05 * z * y
    hy = 50.0 + 1.2 * z - 0.2 * y + 0.09 * z * y
    hz = 60.0 - 0.5 * z + 0.4 * y + 0.13 * z * y
    e_mode = np.stack([ex, ey, ez]).astype(np.complex128)
    h_mode = np.stack([hx, hy, hz]).astype(np.complex128)

    def fake_solve_modes(
        eps,
        omega,
        dL,
        m,
        direction,
        filter_pol,
        target_neff,
        return_fields,
    ):
        del eps, omega, dL, m, direction, filter_pol, target_neff, return_fields
        return np.array([2.0]), [e_mode], [h_mode], None

    monkeypatch.setattr(core_mod, "solve_modes", fake_solve_modes)

    projection = sim._build_port_projection(
        spec=PortSpec(
            name="p",
            monitor_name="m_yee_projection",
            direction="+x",
            polarization="te",
        ),
        monitor=DummyXMonitor(),
        frequency=1.0,
        cache={},
    )

    grids = projection["mode_component_grids"]
    expected = {
        "Ey": 0.5 * (ey[:, :-1] + ey[:, 1:])[:3, :3],
        "Ez": 0.5 * (ez[:-1, :] + ez[1:, :])[:3, :3],
        "Hy": 0.5 * (hy[:-1, :] + hy[1:, :])[:3, :3],
        "Hz": 0.5 * (hz[:, :-1] + hz[:, 1:])[:3, :3],
    }
    for comp, values in expected.items():
        np.testing.assert_allclose(grids[comp], values, rtol=1e-13, atol=1e-13)


def test_build_port_projection_3d_interpolates_cropped_yee_profiles(monkeypatch):
    import beamz.simulation.core as core_mod

    sim = Simulation.__new__(Simulation)
    sim.is_3d = True
    sim.resolution = 1.0
    grid_shape = (5, 5, 5)
    field_arrays = {
        comp: np.zeros(component_shape_3d(comp, grid_shape), dtype=np.float32)
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    zz, yy, xx = np.meshgrid(
        np.arange(grid_shape[0]),
        np.arange(grid_shape[1]),
        np.arange(grid_shape[2]),
        indexing="ij",
    )
    field_arrays["permittivity"] = (1.0 + 0.8 * ((zz + 2 * yy + 3 * xx) % 5)).astype(
        np.float32
    )
    sim.fields = type("F", (), field_arrays)()

    class DummyXMonitor:
        name = "m_cropped_yee_projection"

        @staticmethod
        def get_grid_slice_3d(dx, dy, dz, field_shape):
            del dx, dy, dz
            normal_index = min(1, int(field_shape[2]) - 1)
            return slice(0, 5), slice(0, 5), normal_index

        @staticmethod
        def get_dft_component(component):
            del component
            return np.zeros((1, 16), dtype=np.complex128)

    z, y = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    ex = 10.0 + z + 0.1 * y + 0.03 * z * y
    ey = 20.0 + 2.0 * z - 0.3 * y + 0.11 * z * y
    ez = 30.0 - 0.4 * z + 1.5 * y - 0.07 * z * y
    hx = 40.0 + 0.8 * z + 0.6 * y + 0.05 * z * y
    hy = 50.0 + 1.2 * z - 0.2 * y + 0.09 * z * y
    hz = 60.0 - 0.5 * z + 0.4 * y + 0.13 * z * y
    e_mode = np.stack([ex, ey, ez]).astype(np.complex128)
    h_mode = np.stack([hx, hy, hz]).astype(np.complex128)

    def fake_solve_modes(
        eps,
        omega,
        dL,
        m,
        direction,
        filter_pol,
        target_neff,
        return_fields,
    ):
        del eps, omega, dL, m, direction, filter_pol, target_neff, return_fields
        return np.array([2.0]), [e_mode], [h_mode], None

    monkeypatch.setattr(core_mod, "solve_modes", fake_solve_modes)

    projection = sim._build_port_projection(
        spec=PortSpec(
            name="p",
            monitor_name="m_cropped_yee_projection",
            direction="+x",
            polarization="te",
        ),
        monitor=DummyXMonitor(),
        frequency=1.0,
        cache={},
    )

    assert projection["mode_component_grids"]["Ey"].shape == (4, 4)


def test_build_port_projection_3d_modemonitor_uses_discrete_contract(monkeypatch):
    import beamz.simulation.core as core_mod

    sim = Simulation.__new__(Simulation)
    sim.is_3d = True
    sim.resolution = 1.0
    sim.dt = 0.1
    grid_shape = (4, 4, 4)
    sim.fields = type(
        "F",
        (),
        {
            **{
                comp: np.zeros(component_shape_3d(comp, grid_shape), dtype=float)
                for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
            },
            "permittivity": np.ones(grid_shape, dtype=float),
        },
    )()

    def fail_solve_modes(**kwargs):
        del kwargs
        raise AssertionError("legacy solve_modes should not be used")

    captured = {}

    def fake_solve_beamz_mode_plane(**kwargs):
        captured.update(kwargs)
        profile_shape = (2, 2)
        zeros = np.zeros(profile_shape, dtype=np.complex128)
        forward = {
            "Ex": zeros.copy(),
            "Ey": np.ones(profile_shape, dtype=np.complex128),
            "Ez": zeros.copy(),
            "Hx": zeros.copy(),
            "Hy": zeros.copy(),
            "Hz": np.ones(profile_shape, dtype=np.complex128),
        }
        backward = {
            name: (-value if name.startswith("H") else value.copy())
            for name, value in forward.items()
        }
        indices = {
            "Ex": (slice(1, 3), slice(1, 3), 1),
            "Ey": (slice(1, 3), slice(1, 3), 1),
            "Ez": (slice(1, 3), slice(1, 3), 1),
            "Hx": (slice(1, 3), slice(1, 3), 1),
            "Hy": (slice(1, 3), slice(1, 3), 1),
            "Hz": (slice(1, 3), slice(1, 3), 1),
        }
        return SimpleNamespace(
            neff=2.25,
            profiles=forward,
            backward_profiles=backward,
            component_indices=indices,
        )

    monkeypatch.setattr(core_mod, "solve_modes", fail_solve_modes)
    monkeypatch.setattr(
        core_mod,
        "solve_beamz_mode_plane",
        fake_solve_beamz_mode_plane,
    )

    class ModeMonitor:
        name = "m_discrete_contract"
        center = (1.5, 1.5, 1.5)
        size_spec = (0.0, 2.0, 2.0)
        mode_spec = SimpleNamespace(num_modes=4, target_neff=2.25)

        @staticmethod
        def get_grid_slice_3d(dx, dy, dz, field_shape):
            del dx, dy, dz, field_shape
            return slice(1, 3), slice(1, 3), 1

        @staticmethod
        def get_analysis_plane_coords_3d(*, dx, dy, dz, field_shape):
            del dx, dy, dz, field_shape
            return np.asarray([1.25, 2.25]), np.asarray([1.25, 2.25])

        @staticmethod
        def get_snapped_region(**kwargs):
            del kwargs
            return None

        @staticmethod
        def get_dft_component(component):
            del component
            return np.zeros((1, 4), dtype=np.complex128)

    projection = sim._build_port_projection(
        spec=PortSpec(
            name="p",
            monitor_name="m_discrete_contract",
            direction="+x",
            polarization="te",
        ),
        monitor=ModeMonitor(),
        frequency=1.0,
        cache={},
    )

    assert captured["direction"] == "+x"
    assert captured["solver_direction"] == "+x"
    assert captured["transverse_axes"] == ("z", "y")
    assert captured["plane_index"] == 1
    assert captured["offset_index"] == 0
    assert captured["num_modes"] == 4
    assert captured["target_neff"] == 2.25
    assert captured["aperture_pad_cells"] == 0
    assert captured["aperture_window_alpha"] == 0.0
    assert captured["component_permittivity"]["Ey"].shape == (4, 3, 4)
    assert captured["component_permeability"]["Hz"].shape == (4, 3, 3)
    assert projection["discrete_contract"] == "micromode.beamz.DiscreteMode/v1"
    assert projection["components"] == ("Ey", "Ez", "Hy", "Hz")
    assert np.real(projection["mode_components"]["Hz"][0]) < 0.0
    assert np.real(projection["mode_components_bwd"]["Hz"][0]) > 0.0


def test_project_modal_coefficients_3d_group_recovers_multiple_modes():
    components = ("Ey", "Ez", "Hy", "Hz")

    def _mode(ey, hz):
        return {
            "Ey": np.asarray(ey, dtype=np.complex128),
            "Ez": np.zeros((2,), dtype=np.complex128),
            "Hy": np.zeros((2,), dtype=np.complex128),
            "Hz": np.asarray(hz, dtype=np.complex128),
        }

    def _backward(mode):
        return {
            name: (-value if name.startswith("H") else value.copy())
            for name, value in mode.items()
        }

    mode0 = _mode([1.0, 0.0], [1.0, 0.0])
    mode1 = _mode([0.0, 1.0], [0.0, 1.0])
    mode0_bwd = _backward(mode0)
    mode1_bwd = _backward(mode1)
    projections = [
        {
            "components": components,
            "axis": "x",
            "d_area": 1.0,
            "direction_sign": 1.0,
            "mode_components": mode0,
            "mode_components_bwd": mode0_bwd,
        },
        {
            "components": components,
            "axis": "x",
            "d_area": 1.0,
            "direction_sign": 1.0,
            "mode_components": mode1,
            "mode_components_bwd": mode1_bwd,
        },
    ]
    coeff_true = [
        (0.7 - 0.2j, -0.1 + 0.05j),
        (0.25 + 0.4j, 0.15 - 0.3j),
    ]
    field = {}
    for name in components:
        field[name] = (
            coeff_true[0][0] * mode0[name]
            + coeff_true[0][1] * mode0_bwd[name]
            + coeff_true[1][0] * mode1[name]
            + coeff_true[1][1] * mode1_bwd[name]
        )

    coeff, residual, cond, diagnostics = (
        Simulation._project_modal_coefficients_3d_group(
            field,
            projections,
        )
    )

    assert cond < 10.0
    assert residual < 1e-12
    assert diagnostics["residual_e"] < 1e-12
    assert diagnostics["residual_h"] < 1e-12
    for actual, expected in zip(coeff, coeff_true):
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-10, atol=1e-10)


def test_project_modal_coefficients_3d_uses_overlap_space_when_ill_conditioned(
    monkeypatch,
):
    import beamz.simulation.core as core_mod

    mode_components = {
        "Ex": np.zeros((1,), dtype=np.complex128),
        "Ey": np.array([1.0 + 0.0j], dtype=np.complex128),
        "Ez": np.array([0.25 + 0.1j], dtype=np.complex128),
        "Hx": np.zeros((1,), dtype=np.complex128),
        "Hy": np.array([0.4 - 0.1j], dtype=np.complex128),
        "Hz": np.array([0.3 + 0.05j], dtype=np.complex128),
    }
    mode_components_bwd = {
        name: (-arr if name.startswith("H") else arr)
        for name, arr in mode_components.items()
    }
    overlap = np.array([[1.0, 1.0 - 1e-12], [1.0 - 1e-12, 1.0]], dtype=np.complex128)

    def fake_overlap(field, mode, axis, d_area):
        del field, axis, d_area
        if mode is mode_components:
            return overlap[0, 0]
        if mode is mode_components_bwd:
            return overlap[1, 0]
        raise AssertionError("unexpected mode basis")

    monkeypatch.setattr(core_mod, "_modal_overlap_3d_profiles", fake_overlap)

    projection = {
        "components": ("Ey", "Ez", "Hy", "Hz"),
        "axis": "x",
        "d_area": 1.0,
        "mode_components": mode_components,
        "mode_components_bwd": mode_components_bwd,
        "overlap_matrix": overlap,
        "pinv": np.zeros((2, 0), dtype=np.complex128),
    }

    a_p, a_m = Simulation._project_modal_coefficients_3d(mode_components, projection)
    np.testing.assert_allclose(a_p, 1.0 + 0.0j, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(a_m, 0.0 + 0.0j, rtol=1e-4, atol=1e-4)


def test_build_port_projection_3d_builds_power_orthogonal_basis(monkeypatch):
    import beamz.simulation.core as core_mod

    sim = Simulation.__new__(Simulation)
    sim.is_3d = True
    sim.resolution = 1.0
    sim.fields = type(
        "F",
        (),
        {
            "Ex": np.zeros((3, 3, 3), dtype=float),
            "Ey": np.zeros((3, 3, 3), dtype=float),
            "Ez": np.zeros((3, 3, 3), dtype=float),
            "Hx": np.zeros((3, 3, 3), dtype=float),
            "Hy": np.zeros((3, 3, 3), dtype=float),
            "Hz": np.zeros((3, 3, 3), dtype=float),
            "permittivity": np.ones((3, 3, 3), dtype=float),
        },
    )()

    def fake_profile(self, monitor, axis, pad_cells):
        return np.ones((2, 2), dtype=np.complex128), np.arange(4, dtype=int), 1.0

    monkeypatch.setattr(Simulation, "_monitor_profile_slice", fake_profile)

    def fake_solve_modes(
        eps,
        omega,
        dL,
        m,
        direction,
        filter_pol,
        target_neff,
        return_fields,
    ):
        del eps, omega, dL, m, filter_pol, target_neff, return_fields
        ex = np.array([[1.0, 0.5], [0.3, 0.1]], dtype=np.complex128)
        ey = np.array([[0.6, -0.1], [0.2, 0.4]], dtype=np.complex128)
        ez = np.array([[0.2, 0.3], [0.4, -0.2]], dtype=np.complex128)
        hx = np.array([[0.1, 0.0], [0.2, 0.3]], dtype=np.complex128)
        hy = np.array([[0.5, 0.2], [0.1, 0.4]], dtype=np.complex128)
        hz = np.array([[0.4, 0.1], [0.3, 0.2]], dtype=np.complex128)
        if direction == "+x":
            hx = -hx
            hy = -hy
            hz = -hz
        return (
            np.array([2.1], dtype=float),
            [(ex, ey, ez)],
            [(hx, hy, hz)],
            None,
        )

    monkeypatch.setattr(core_mod, "solve_modes", fake_solve_modes)

    class DummyMonitor:
        name = "m3d"

        @staticmethod
        def get_grid_slice_3d(dx, dy, dz, field_shape):
            del dx, dy, dz, field_shape
            return 1, slice(0, 2), slice(0, 2)

        @staticmethod
        def get_dft_component(component):
            del component
            return np.zeros((1, 4), dtype=np.complex128)

    spec = PortSpec(
        name="p1",
        monitor_name="m3d",
        direction="+x",
        polarization="tm",
        mode_index=0,
    )
    projection = sim._build_port_projection(
        spec=spec,
        monitor=DummyMonitor(),
        frequency=1.0,
        cache={},
    )
    overlap = np.asarray(projection["overlap_matrix"], dtype=np.complex128)
    assert overlap.shape == (2, 2)
    assert np.isfinite(overlap).all()
    np.testing.assert_allclose(np.abs(np.diag(overlap)), np.ones(2), atol=1e-12)
    assert projection["d_area"] == pytest.approx(sim.resolution**2)

    a_fwd = Simulation._project_modal_coefficients_3d(
        projection["mode_components"], projection
    )
    a_bwd = Simulation._project_modal_coefficients_3d(
        projection["mode_components_bwd"], projection
    )
    np.testing.assert_allclose(
        np.asarray(a_fwd, dtype=np.complex128),
        np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(a_bwd, dtype=np.complex128),
        np.asarray([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128),
        rtol=1e-9,
        atol=1e-9,
    )


def test_build_port_projection_3d_x_branch_matches_mode_source_solver_convention(
    monkeypatch,
):
    import beamz.simulation.core as core_mod

    sim = Simulation.__new__(Simulation)
    sim.is_3d = True
    sim.resolution = 1.0
    sim.fields = type(
        "F",
        (),
        {
            "Ex": np.zeros((3, 3, 3), dtype=float),
            "Ey": np.zeros((3, 3, 3), dtype=float),
            "Ez": np.zeros((3, 3, 3), dtype=float),
            "Hx": np.zeros((3, 3, 3), dtype=float),
            "Hy": np.zeros((3, 3, 3), dtype=float),
            "Hz": np.zeros((3, 3, 3), dtype=float),
            "permittivity": np.ones((3, 3, 3), dtype=float),
        },
    )()

    def fake_profile(self, monitor, axis, pad_cells):
        del self, monitor, axis, pad_cells
        return np.ones((2, 2), dtype=np.complex128), np.arange(4, dtype=int), 1.0

    monkeypatch.setattr(Simulation, "_monitor_profile_slice", fake_profile)

    seen_directions = []

    def fake_solve_modes(
        eps,
        omega,
        dL,
        m,
        direction,
        filter_pol,
        target_neff,
        return_fields,
    ):
        del eps, omega, dL, m, filter_pol, target_neff, return_fields
        seen_directions.append(direction)
        ex = np.ones((2, 2), dtype=np.complex128)
        ey = np.ones((2, 2), dtype=np.complex128)
        ez = np.ones((2, 2), dtype=np.complex128)
        hx = np.ones((2, 2), dtype=np.complex128)
        hy = np.ones((2, 2), dtype=np.complex128)
        hz = np.ones((2, 2), dtype=np.complex128)
        return (
            np.array([2.1], dtype=float),
            [(ex, ey, ez)],
            [(hx, hy, hz)],
            None,
        )

    monkeypatch.setattr(core_mod, "solve_modes", fake_solve_modes)

    class DummyMonitor:
        name = "m3d_x"

        @staticmethod
        def get_grid_slice_3d(dx, dy, dz, field_shape):
            del dx, dy, dz, field_shape
            return 1, slice(0, 2), slice(0, 2)

        @staticmethod
        def get_dft_component(component):
            del component
            return np.zeros((1, 4), dtype=np.complex128)

    sim._build_port_projection(
        spec=PortSpec(
            name="p1",
            monitor_name="m3d_x",
            direction="+x",
            polarization="tm",
            mode_index=0,
        ),
        monitor=DummyMonitor(),
        frequency=1.0,
        cache={},
    )

    assert seen_directions[:2] == ["-x", "+x"]


@pytest.mark.parametrize(
    ("direction", "expected_directions"),
    [
        ("+y", ["-y", "+y"]),
        ("-y", ["+y", "-y"]),
    ],
)
def test_build_port_projection_3d_y_branch_matches_mode_source_solver_convention(
    monkeypatch, direction, expected_directions
):
    import beamz.simulation.core as core_mod

    sim = Simulation.__new__(Simulation)
    sim.is_3d = True
    sim.resolution = 1.0
    sim.fields = type(
        "F",
        (),
        {
            "Ex": np.zeros((3, 3, 3), dtype=float),
            "Ey": np.zeros((3, 3, 3), dtype=float),
            "Ez": np.zeros((3, 3, 3), dtype=float),
            "Hx": np.zeros((3, 3, 3), dtype=float),
            "Hy": np.zeros((3, 3, 3), dtype=float),
            "Hz": np.zeros((3, 3, 3), dtype=float),
            "permittivity": np.ones((3, 3, 3), dtype=float),
        },
    )()

    def fake_profile(self, monitor, axis, pad_cells):
        del self, monitor, axis, pad_cells
        return np.ones((2, 2), dtype=np.complex128), np.arange(4, dtype=int), 1.0

    monkeypatch.setattr(Simulation, "_monitor_profile_slice", fake_profile)

    seen_directions = []

    def fake_solve_modes(
        eps,
        omega,
        dL,
        m,
        direction,
        filter_pol,
        target_neff,
        return_fields,
    ):
        del eps, omega, dL, m, filter_pol, target_neff, return_fields
        seen_directions.append(direction)
        ex = np.ones((2, 2), dtype=np.complex128)
        ey = np.ones((2, 2), dtype=np.complex128)
        ez = np.ones((2, 2), dtype=np.complex128)
        hx = np.ones((2, 2), dtype=np.complex128)
        hy = np.ones((2, 2), dtype=np.complex128)
        hz = np.ones((2, 2), dtype=np.complex128)
        return (
            np.array([2.1], dtype=float),
            [(ex, ey, ez)],
            [(hx, hy, hz)],
            None,
        )

    monkeypatch.setattr(core_mod, "solve_modes", fake_solve_modes)

    class DummyMonitor:
        name = "m3d_y"

        @staticmethod
        def get_grid_slice_3d(dx, dy, dz, field_shape):
            del dx, dy, dz, field_shape
            return slice(0, 2), 1, slice(0, 2)

        @staticmethod
        def get_dft_component(component):
            del component
            return np.zeros((1, 4), dtype=np.complex128)

    sim._build_port_projection(
        spec=PortSpec(
            name="p1",
            monitor_name="m3d_y",
            direction=direction,
            polarization="te",
            mode_index=0,
        ),
        monitor=DummyMonitor(),
        frequency=1.0,
        cache={},
    )

    assert seen_directions[:2] == expected_directions


def test_build_port_projection_3d_uses_solved_backward_mode_components(monkeypatch):
    import beamz.simulation.core as core_mod

    sim = Simulation.__new__(Simulation)
    sim.is_3d = True
    sim.resolution = 1.0
    sim.fields = type(
        "F",
        (),
        {
            "Ex": np.zeros((3, 3, 3), dtype=float),
            "Ey": np.zeros((3, 3, 3), dtype=float),
            "Ez": np.zeros((3, 3, 3), dtype=float),
            "Hx": np.zeros((3, 3, 3), dtype=float),
            "Hy": np.zeros((3, 3, 3), dtype=float),
            "Hz": np.zeros((3, 3, 3), dtype=float),
            "permittivity": np.ones((3, 3, 3), dtype=float),
        },
    )()

    def fake_profile(self, monitor, axis, pad_cells):
        del self, monitor, axis, pad_cells
        return np.ones((2, 2), dtype=np.complex128), np.arange(4, dtype=int), 1.0

    monkeypatch.setattr(Simulation, "_monitor_profile_slice", fake_profile)
    monkeypatch.setattr(core_mod, "_detect_transverse_symmetry_axes", lambda eps: ())
    monkeypatch.setattr(
        core_mod,
        "_make_3d_mode_basis_profiles",
        lambda profiles, axis, d_area=1.0, direction_sign=1.0: (dict(profiles), {}),
    )
    monkeypatch.setattr(
        core_mod,
        "_normalize_3d_profiles_by_flux",
        lambda profiles, axis, d_area=1.0, direction_sign=1.0: dict(profiles),
    )

    def fake_solve_modes(
        eps,
        omega,
        dL,
        m,
        direction,
        filter_pol,
        target_neff,
        return_fields,
    ):
        del eps, omega, dL, m, filter_pol, target_neff, return_fields
        if direction == "+y":
            ex = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
            ey = np.zeros((2, 2), dtype=np.complex128)
            ez = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
            hx = np.array([[3.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
            hy = np.zeros((2, 2), dtype=np.complex128)
            hz = np.array([[4.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
            neff = 2.1
        else:
            ex = np.array([[5.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
            ey = np.zeros((2, 2), dtype=np.complex128)
            ez = np.array([[6.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
            hx = np.array([[7.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
            hy = np.zeros((2, 2), dtype=np.complex128)
            hz = np.array([[8.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
            neff = 2.0
        return (
            np.array([neff], dtype=float),
            [(ex, ey, ez)],
            [(hx, hy, hz)],
            None,
        )

    monkeypatch.setattr(core_mod, "solve_modes", fake_solve_modes)

    class DummyMonitor:
        name = "m3d_bwd"

        @staticmethod
        def get_grid_slice_3d(dx, dy, dz, field_shape):
            del dx, dy, dz, field_shape
            return slice(0, 2), 1, slice(0, 2)

        @staticmethod
        def get_dft_component(component):
            del component
            return np.zeros((1, 4), dtype=np.complex128)

    projection = sim._build_port_projection(
        spec=PortSpec(
            name="p1",
            monitor_name="m3d_bwd",
            direction="+y",
            polarization="te",
            mode_index=0,
        ),
        monitor=DummyMonitor(),
        frequency=1.0,
        cache={},
    )

    np.testing.assert_allclose(
        projection["mode_components"]["Ex"],
        np.array([2.5], dtype=np.complex128),
    )
    np.testing.assert_allclose(
        projection["mode_components_bwd"]["Ex"],
        np.array([0.5], dtype=np.complex128),
    )
    assert not np.array_equal(
        projection["mode_components_bwd"]["Ex"],
        projection["mode_components"]["Ex"],
    )


def test_build_port_projection_3d_uses_previous_projection_to_lock_forward_branch(
    monkeypatch,
):
    import beamz.simulation.core as core_mod

    sim = Simulation.__new__(Simulation)
    sim.is_3d = True
    sim.resolution = 1.0
    sim.fields = type(
        "F",
        (),
        {
            "Ex": np.zeros((3, 3, 3), dtype=float),
            "Ey": np.zeros((3, 3, 3), dtype=float),
            "Ez": np.zeros((3, 3, 3), dtype=float),
            "Hx": np.zeros((3, 3, 3), dtype=float),
            "Hy": np.zeros((3, 3, 3), dtype=float),
            "Hz": np.zeros((3, 3, 3), dtype=float),
            "permittivity": np.ones((3, 3, 3), dtype=float),
        },
    )()

    def fake_profile(self, monitor, axis, pad_cells):
        del self, monitor, axis, pad_cells
        return np.ones((2, 2), dtype=np.complex128), np.arange(4, dtype=int), 1.0

    monkeypatch.setattr(Simulation, "_monitor_profile_slice", fake_profile)
    monkeypatch.setattr(core_mod, "_detect_transverse_symmetry_axes", lambda eps: ())
    monkeypatch.setattr(
        core_mod,
        "_make_3d_mode_basis_profiles",
        lambda profiles, axis, d_area=1.0, direction_sign=1.0: (dict(profiles), {}),
    )
    monkeypatch.setattr(core_mod, "_select_core_confined_mode_index", lambda *args: 0)

    a = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    b = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    z = np.zeros((2, 2), dtype=np.complex128)

    def fake_solve_modes(
        eps,
        omega,
        dL,
        m,
        direction,
        filter_pol,
        target_neff,
        return_fields,
    ):
        del eps, omega, dL, m, filter_pol, target_neff, return_fields
        if direction == "+y":
            return (
                np.array([2.3, 1.8], dtype=float),
                [(a, z, a), (b, z, b)],
                [(a, z, a), (b, z, b)],
                None,
            )
        return (
            np.array([1.8], dtype=float),
            [(b, z, b)],
            [(b, z, b)],
            None,
        )

    monkeypatch.setattr(core_mod, "solve_modes", fake_solve_modes)

    class DummyMonitor:
        name = "m3d_y_seeded"

        @staticmethod
        def get_grid_slice_3d(dx, dy, dz, field_shape):
            del dx, dy, dz, field_shape
            return slice(0, 2), 1, slice(0, 2)

        @staticmethod
        def get_dft_component(component):
            del component
            return np.zeros((1, 4), dtype=np.complex128)

    previous_projection = {
        "mode_neff": 1.8,
        "mode_parity": (1.0, 1.0),
        "mode_components": {
            "Ex": b.reshape(-1),
            "Ez": b.reshape(-1),
            "Hx": b.reshape(-1),
            "Hz": b.reshape(-1),
        },
        "mode_component_grids": {
            "Ex": b,
            "Ez": b,
            "Hx": b,
            "Hz": b,
        },
    }

    projection = sim._build_port_projection(
        spec=PortSpec(
            name="pseed",
            monitor_name="m3d_y_seeded",
            direction="+y",
            polarization="te",
            mode_index=0,
        ),
        monitor=DummyMonitor(),
        frequency=1.0,
        cache={},
        previous_projection=previous_projection,
    )

    assert projection["mode_neff"] == pytest.approx(1.8)


def test_projection_seed_key_distinguishes_ports_and_monitors():
    class DummyMonitor:
        def __init__(self, name):
            self.name = name

        @staticmethod
        def get_dft_component(component):
            del component
            return np.zeros((1, 4), dtype=np.complex128)

    parts = {
        "projection_components": ("Ex", "Hz"),
        "projection_components_3d": ("Ex", "Ez", "Hx", "Hz"),
    }
    spec_a = PortSpec(
        name="o1",
        monitor_name="m_shared",
        direction="+y",
        polarization="te",
        mode_index=0,
    )
    spec_b = PortSpec(
        name="o2",
        monitor_name="m_shared",
        direction="+y",
        polarization="te",
        mode_index=0,
    )

    key_a = Simulation._projection_seed_key(
        spec_a, DummyMonitor("m_a"), parts, is_3d=True
    )
    key_b = Simulation._projection_seed_key(
        spec_b, DummyMonitor("m_a"), parts, is_3d=True
    )
    key_c = Simulation._projection_seed_key(
        spec_a, DummyMonitor("m_b"), parts, is_3d=True
    )

    assert key_a != key_b
    assert key_a != key_c


def test_build_port_projection_3d_preserves_requested_mode_index_without_seed(
    monkeypatch,
):
    import beamz.simulation.core as core_mod

    sim = Simulation.__new__(Simulation)
    sim.is_3d = True
    sim.resolution = 1.0
    sim.fields = type(
        "F",
        (),
        {
            "Ex": np.zeros((3, 3, 3), dtype=float),
            "Ey": np.zeros((3, 3, 3), dtype=float),
            "Ez": np.zeros((3, 3, 3), dtype=float),
            "Hx": np.zeros((3, 3, 3), dtype=float),
            "Hy": np.zeros((3, 3, 3), dtype=float),
            "Hz": np.zeros((3, 3, 3), dtype=float),
            "permittivity": np.ones((3, 3, 3), dtype=float),
        },
    )()

    def fake_profile(self, monitor, axis, pad_cells):
        del self, monitor, axis, pad_cells
        return np.ones((2, 2), dtype=np.complex128), np.arange(4, dtype=int), 1.0

    monkeypatch.setattr(Simulation, "_monitor_profile_slice", fake_profile)
    monkeypatch.setattr(core_mod, "_detect_transverse_symmetry_axes", lambda eps: ())
    monkeypatch.setattr(
        core_mod,
        "_make_3d_mode_basis_profiles",
        lambda profiles, axis, d_area=1.0, direction_sign=1.0: (dict(profiles), {}),
    )
    monkeypatch.setattr(core_mod, "_select_core_confined_mode_index", lambda *args: 0)

    first = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    second = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    z = np.zeros((2, 2), dtype=np.complex128)

    def fake_solve_modes(
        eps,
        omega,
        dL,
        m,
        direction,
        filter_pol,
        target_neff,
        return_fields,
    ):
        del eps, omega, dL, m, direction, filter_pol, target_neff, return_fields
        return (
            np.array([2.3, 1.8], dtype=float),
            [(first, z, first), (second, z, second)],
            [(first, z, first), (second, z, second)],
            None,
        )

    monkeypatch.setattr(core_mod, "solve_modes", fake_solve_modes)

    class DummyMonitor:
        name = "m3d_mode1"

        @staticmethod
        def get_grid_slice_3d(dx, dy, dz, field_shape):
            del dx, dy, dz, field_shape
            return slice(0, 2), 1, slice(0, 2)

        @staticmethod
        def get_dft_component(component):
            del component
            return np.zeros((1, 4), dtype=np.complex128)

    projection = sim._build_port_projection(
        spec=PortSpec(
            name="p_mode1",
            monitor_name="m3d_mode1",
            direction="+y",
            polarization="te",
            mode_index=1,
        ),
        monitor=DummyMonitor(),
        frequency=1.0,
        cache={},
    )

    np.testing.assert_allclose(
        projection["mode_components"]["Ex"],
        np.array([1.0], dtype=np.complex128),
    )
    assert projection["mode_neff"] == pytest.approx(1.8)


def test_build_port_projection_2d_preserves_requested_mode_index_without_seed(
    monkeypatch,
):
    import beamz.simulation.core as core_mod

    sim = Simulation.__new__(Simulation)
    sim.is_3d = False
    sim.plane_2d = "xy"
    sim.resolution = 1.0
    sim.fields = type(
        "F",
        (),
        {"permittivity": np.ones((5, 5), dtype=float)},
    )()

    def fake_profile(self, monitor, axis, pad_cells):
        del self, monitor, axis, pad_cells
        return np.ones((5,), dtype=np.complex128), np.arange(5, dtype=int), 1.0

    monkeypatch.setattr(Simulation, "_monitor_profile_slice", fake_profile)
    monkeypatch.setattr(core_mod, "_select_core_confined_mode_index", lambda *args: 0)

    e0 = np.zeros((3, 5), dtype=np.complex128)
    e1 = np.zeros((3, 5), dtype=np.complex128)
    h0 = np.zeros((3, 5), dtype=np.complex128)
    h1 = np.zeros((3, 5), dtype=np.complex128)
    e0[0, 0] = 1.0
    e1[0, 1] = 3.0
    h0[2, 0] = 1.0
    h1[2, 1] = 3.0

    def fake_solve_modes(
        eps,
        omega,
        dL,
        m,
        direction,
        filter_pol,
        target_neff,
        return_fields,
    ):
        del eps, omega, dL, m, direction, filter_pol, target_neff, return_fields
        return (
            np.array([2.3, 1.8], dtype=float),
            [e0, e1],
            [h0, h1],
            None,
        )

    monkeypatch.setattr(core_mod, "solve_modes", fake_solve_modes)

    monitor = Monitor(start=(0.0, 0.0), end=(0.0, 1.0), name="m2d_mode1")
    projection = sim._build_port_projection(
        spec=PortSpec(
            name="p2d_mode1",
            monitor_name="m2d_mode1",
            direction="+y",
            polarization="te",
            mode_index=1,
        ),
        monitor=monitor,
        frequency=1.0,
        cache={},
    )

    assert projection["mode_neff"] == pytest.approx(1.8)
    assert int(np.argmax(np.abs(projection["mode_matrix"][:5, 0]))) == 1
    assert np.count_nonzero(np.abs(projection["mode_matrix"][:5, 0]) > 1e-12) == 1


def test_extract_port_waves_dft_modal_coefficients_synthetic(monkeypatch):
    freqs = np.array([1.0, 2.0], dtype=float)
    mode_matrix = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    pinv = np.linalg.pinv(mode_matrix)

    a_src = np.array([1.2 + 0.2j, 1.1 - 0.1j])
    b_src = np.array([0.08 - 0.02j, 0.03 + 0.01j])
    a_ref = np.array([1.4 + 0.0j, 1.5 + 0.0j])
    b_ref = np.zeros_like(a_ref)
    a_out = np.array([0.05 + 0.01j, -0.02 + 0.03j])
    b_out = np.array([0.72 - 0.1j, 0.69 + 0.08j])

    dft_map = {
        ("m_src", "Ez"): (a_src + b_src)[:, None],
        ("m_src", "Hy"): (a_src - b_src)[:, None],
        ("m_ref", "Ez"): (a_ref + b_ref)[:, None],
        ("m_ref", "Hy"): (a_ref - b_ref)[:, None],
        ("m_out", "Ez"): (a_out + b_out)[:, None],
        ("m_out", "Hy"): (a_out - b_out)[:, None],
    }

    def fake_sample(self, monitor, component, frequencies):
        assert np.allclose(frequencies, freqs)
        return np.asarray(frequencies, dtype=float), dft_map[(monitor.name, component)]

    projection_frequencies = []

    def fake_projection(
        self,
        spec,
        monitor,
        frequency,
        cache,
        mode_pad_cells=6,
        previous_projection=None,
    ):
        del previous_projection
        projection_frequencies.append(float(frequency))
        return {
            "e_component": "Ez",
            "h_component": "Hy",
            "mode_matrix": mode_matrix,
            "pinv": pinv,
            "condition_number": 1.0,
            "mode_neff": 2.0,
        }

    monkeypatch.setattr(Simulation, "_sample_monitor_component_dft", fake_sample)
    monkeypatch.setattr(Simulation, "_build_port_projection", fake_projection)

    sim = Simulation.__new__(Simulation)
    sim.is_3d = False
    sim.plane_2d = "xy"
    sim.resolution = 1.0
    sim.sources = []
    sim.monitors = [
        Monitor(
            start=(0.0, 0.0),
            end=(0.0, 1.0),
            name="m_src",
            dft_enabled=True,
            dft_frequencies=freqs,
        ),
        Monitor(
            start=(0.0, 0.0),
            end=(0.0, 1.0),
            name="m_ref",
            dft_enabled=True,
            dft_frequencies=freqs,
        ),
        Monitor(
            start=(1.0, 0.0),
            end=(1.0, 1.0),
            name="m_out",
            dft_enabled=True,
            dft_frequencies=freqs,
        ),
    ]

    ports = [
        PortSpec(
            name="o1",
            monitor_name="m_src",
            reference_monitor="m_ref",
            direction="+x",
            polarization="tm",
        ),
        PortSpec(name="o2", monitor_name="m_out", direction="+x", polarization="tm"),
    ]
    waves = sim.extract_port_waves_dft(ports=ports, frequencies=freqs)
    np.testing.assert_allclose(waves["o1"]["a_plus"], a_src, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(waves["o1"]["a_minus"], b_src, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(waves["o1"]["a_incident"], a_ref, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(waves["o2"]["a_plus"], a_out, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(waves["o2"]["a_minus"], b_out, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        waves["o1"]["projection_residual"],
        np.zeros_like(freqs),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        waves["o1"]["reference_projection_residual"],
        np.zeros_like(freqs),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        waves["o2"]["projection_residual"],
        np.zeros_like(freqs),
        rtol=1e-12,
        atol=1e-12,
    )

    projection_frequencies.clear()
    sim.extract_port_waves_dft(
        ports=ports,
        frequencies=freqs,
        mode_strategy="single",
    )
    assert projection_frequencies
    np.testing.assert_allclose(
        projection_frequencies,
        np.full(len(projection_frequencies), np.median(freqs)),
    )


def test_get_S_matrix_modal_dft_keys_shapes_and_valid_mask(monkeypatch):
    sim = Simulation.__new__(Simulation)
    sim.sources = []
    sim.monitors = []
    sim.is_3d = False
    sim.plane_2d = "xy"
    freqs = np.array([1.0, 2.0, 3.0], dtype=float)
    waves = {
        "o1": {
            "a_plus": np.ones_like(freqs, dtype=np.complex128),
            "a_minus": 0.1 * np.ones_like(freqs, dtype=np.complex128),
            "a_incident": np.array([1.0, 1e-6, 1.0], dtype=np.complex128),
            "condition_number": np.array([2.0, 2.0, 2.0], dtype=float),
        },
        "o2": {
            "a_plus": np.zeros_like(freqs, dtype=np.complex128),
            "a_minus": 0.7j * np.ones_like(freqs, dtype=np.complex128),
            "condition_number": np.array([3.0, 3.0, 3.0], dtype=float),
        },
    }

    def fake_extract(
        self,
        ports,
        frequencies,
        min_incident_db=-40.0,
        return_power=True,
        mode_strategy="per_frequency",
    ):
        assert np.allclose(frequencies, freqs)
        return waves

    monkeypatch.setattr(Simulation, "extract_port_waves_dft", fake_extract)
    result = sim.get_S_matrix_modal_dft(
        source_port="o1",
        ports=[
            PortSpec(
                name="o1",
                monitor_name="o1",
                reference_monitor="o1_ref",
                direction="+x",
                polarization="tm",
            ),
            PortSpec(name="o2", monitor_name="o2", direction="+x", polarization="tm"),
        ],
        output_ports=["o1", "o2"],
        frequencies=freqs,
        as_sax=False,
        return_diagnostics=True,
        min_incident_db=-40.0,
    )

    s_matrix = result["s_matrix"]
    diag = result["diagnostics"]
    assert set(s_matrix.keys()) == {("o1", "o1"), ("o2", "o1")}
    assert s_matrix[("o1", "o1")].shape == freqs.shape
    assert s_matrix[("o2", "o1")].shape == freqs.shape
    assert diag["valid_mask"].shape == freqs.shape
    assert np.array_equal(diag["valid_mask"], np.array([True, False, True]))
    assert s_matrix[("o2", "o1")][1] == 0.0j
    assert np.isnan(diag["power_sum"][1])


def test_get_S_matrix_modal_dft_respects_wave_selectors(monkeypatch):
    sim = Simulation.__new__(Simulation)
    sim.sources = []
    sim.monitors = []
    sim.is_3d = False
    sim.plane_2d = "xy"
    freqs = np.array([1.0, 2.0], dtype=float)
    waves = {
        "src": {
            "a_plus": np.array([2.0, 2.0], dtype=np.complex128),
            "a_minus": np.array([0.4, 0.4], dtype=np.complex128),
            "a_incident_plus": np.array([1.5, 1.5], dtype=np.complex128),
            "a_incident_minus": np.array([0.2, 0.2], dtype=np.complex128),
            "condition_number": np.array([2.0, 2.0], dtype=float),
        },
        "out": {
            "a_plus": np.array([0.9, 0.8], dtype=np.complex128),
            "a_minus": np.array([0.1, 0.1], dtype=np.complex128),
            "condition_number": np.array([3.0, 3.0], dtype=float),
        },
    }

    def fake_extract(
        self,
        ports,
        frequencies,
        min_incident_db=-40.0,
        return_power=True,
        mode_strategy="per_frequency",
    ):
        del ports, min_incident_db, return_power, mode_strategy
        assert np.allclose(frequencies, freqs)
        return waves

    monkeypatch.setattr(Simulation, "extract_port_waves_dft", fake_extract)
    result = sim.get_S_matrix_modal_dft(
        source_port="src",
        ports=[
            PortSpec(
                name="src",
                monitor_name="src_m",
                direction="+x",
                polarization="tm",
                reference_monitor="src_ref",
                incident_wave="plus",
            ),
            PortSpec(
                name="out",
                monitor_name="out_m",
                direction="+x",
                polarization="tm",
                scattered_wave="plus",
            ),
        ],
        output_ports=["out"],
        frequencies=freqs,
        as_sax=False,
        return_diagnostics=True,
        min_incident_db=-80.0,
    )
    np.testing.assert_allclose(
        result["s_matrix"][("out", "src")],
        np.array([0.6, 0.5333333333333333], dtype=np.complex128),
        rtol=1e-12,
        atol=1e-12,
    )
    out_check = result["diagnostics"]["monitor_flux_checks"]["out"]
    assert out_check["incident_wave"] == "plus"
    assert out_check["scattered_wave"] == "plus"
    np.testing.assert_allclose(out_check["P_selected"], np.array([0.81, 0.64]))
    np.testing.assert_allclose(out_check["P_rejected"], np.array([0.01, 0.01]))
    np.testing.assert_allclose(out_check["P_selected_modal_net"], np.array([0.8, 0.63]))


def test_get_S_matrix_modal_dft_auto_selectors_prefer_band_dominant_source_wave(
    monkeypatch,
):
    sim = Simulation.__new__(Simulation)
    sim.sources = []
    sim.monitors = []
    sim.is_3d = False
    sim.plane_2d = "xy"
    freqs = np.array([1.0, 2.0], dtype=float)
    waves = {
        "src": {
            "a_plus": np.array([0.2, 1.0], dtype=np.complex128),
            "a_minus": np.array([2.0, 0.2], dtype=np.complex128),
            "condition_number": np.array([2.0, 2.0], dtype=float),
        },
        "out": {
            "a_plus": np.array([0.4, 0.4], dtype=np.complex128),
            "a_minus": np.array([0.5, 0.5], dtype=np.complex128),
            "condition_number": np.array([3.0, 3.0], dtype=float),
        },
    }

    def fake_extract(
        self,
        ports,
        frequencies,
        min_incident_db=-40.0,
        return_power=True,
        mode_strategy="per_frequency",
    ):
        del ports, min_incident_db, return_power, mode_strategy
        assert np.allclose(frequencies, freqs)
        return waves

    monkeypatch.setattr(Simulation, "extract_port_waves_dft", fake_extract)
    result = sim.get_S_matrix_modal_dft(
        source_port="src",
        ports=[
            PortSpec(
                name="src",
                monitor_name="src_m",
                direction="+x",
                polarization="tm",
                incident_wave="auto",
                scattered_wave="auto",
            ),
            PortSpec(
                name="out",
                monitor_name="out_m",
                direction="+x",
                polarization="tm",
                scattered_wave="minus",
            ),
        ],
        output_ports=["out"],
        frequencies=freqs,
        as_sax=False,
        return_diagnostics=True,
        min_incident_db=-80.0,
    )
    np.testing.assert_allclose(
        result["s_matrix"][("out", "src")],
        np.array([0.25, 2.5], dtype=np.complex128),
        rtol=1e-12,
        atol=1e-12,
    )
    out_check = result["diagnostics"]["monitor_flux_checks"]["out"]
    assert out_check["incident_wave"] == "plus"
    assert out_check["scattered_wave"] == "minus"
    np.testing.assert_allclose(out_check["P_selected"], np.array([0.25, 0.25]))
    np.testing.assert_allclose(out_check["P_rejected"], np.array([0.16, 0.16]))
    np.testing.assert_allclose(
        out_check["P_selected_modal_net"], np.array([0.09, 0.09])
    )


def test_normalize_portspecs_rejects_invalid_wave_selector():
    with pytest.raises(ValueError, match="incident_wave"):
        Simulation._normalize_portspecs(
            [
                {
                    "name": "o1",
                    "monitor_name": "m1",
                    "direction": "+x",
                    "polarization": "tm",
                    "incident_wave": "invalid",
                }
            ]
        )


def test_colocate_monitor_component_matrix_3d_interpolates_to_canonical_plane():
    sim = Simulation.__new__(Simulation)
    sim.is_3d = True
    sim.resolution = 1.0
    sim.fields = type(
        "F",
        (),
        {
            "permittivity": np.ones((5, 6, 7), dtype=float),
            "Ex": np.zeros((5, 6, 6), dtype=float),
            "Ey": np.zeros((5, 5, 7), dtype=float),
            "Ez": np.zeros((4, 6, 7), dtype=float),
            "Hx": np.zeros((4, 5, 7), dtype=float),
            "Hy": np.zeros((4, 6, 6), dtype=float),
            "Hz": np.zeros((5, 5, 6), dtype=float),
        },
    )()

    monitor = Monitor(
        start=(1.2, 2.8, 1.1),
        end=(5.8, 2.8, 4.4),
        name="mcanon",
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=np.array([1.0], dtype=float),
        dft_components=("Ex",),
        dft_window="none",
        dft_record_every_step=True,
    )

    src0, src1 = sim._monitor_component_plane_coords_3d(monitor, "Ex", axis="y")
    dst0, dst1 = sim._monitor_analysis_plane_3d(monitor, axis="y")
    expected0 = 1.1 + (np.arange(dst0.size, dtype=float) + 0.5) * (
        (4.4 - 1.1) / float(dst0.size)
    )
    expected1 = 1.2 + (np.arange(dst1.size, dtype=float) + 0.5) * (
        (5.8 - 1.2) / float(dst1.size)
    )
    np.testing.assert_allclose(dst0, expected0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(dst1, expected1, atol=1e-12, rtol=0.0)
    src_plane = src0[:, None] + src1[None, :]
    colocated = sim._colocate_monitor_component_matrix_3d(
        monitor,
        "Ex",
        src_plane.reshape(1, -1),
        axis="y",
        target0=dst0,
        target1=dst1,
    )
    dst0_eff = np.clip(dst0, np.min(src0), np.max(src0))
    dst1_eff = np.clip(dst1, np.min(src1), np.max(src1))
    expected = (dst0_eff[:, None] + dst1_eff[None, :]).reshape(1, -1)
    np.testing.assert_allclose(colocated, expected, rtol=1e-9, atol=1e-9)
