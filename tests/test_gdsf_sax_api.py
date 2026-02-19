import importlib.util

import numpy as np
import pytest

from beamz import (
    Monitor,
    PortSpec,
    Simulation,
    calc_optimal_fdtd_params,
    design,
    dxdt,
)


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
    assert all(getattr(struct, "material", None) is not None for struct in core_structures)

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
        return np.asarray(frequencies, dtype=float), spectral_map[(monitor.name, component)]

    def fake_projection(self, spec, monitor, frequency, cache, mode_pad_cells=6):
        return {"e_component": "Ez", "h_component": "Hy", "pinv": pinv}

    monkeypatch.setattr(Simulation, "_sample_monitor_component_spectrum", fake_sample)
    monkeypatch.setattr(Simulation, "_build_port_projection", fake_projection)

    sim = Simulation.__new__(Simulation)
    sim.is_3d = False
    sim.plane_2d = "xy"
    sim.resolution = 1.0
    sim.devices = [
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
    waves = sim.extract_port_waves(ports=ports, frequencies=freqs, mode_strategy="per_frequency")

    np.testing.assert_allclose(waves["o1"]["a_plus"], a_src, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(waves["o1"]["a_minus"], b_src, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(waves["o1"]["a_incident"], a_ref, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(waves["o2"]["a_plus"], a_out, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(waves["o2"]["a_minus"], b_out, rtol=1e-9, atol=1e-9)


def test_get_S_matrix_modal_column_keys_and_shapes(monkeypatch):
    sim = Simulation.__new__(Simulation)
    sim.devices = []
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

    def fake_extract(self, ports, frequencies, mode_strategy="per_frequency", window="hann", return_power=True):
        return waves

    monkeypatch.setattr(Simulation, "extract_port_waves", fake_extract)
    result = sim.get_S_matrix_modal(
        source_port="o1",
        ports=[
            PortSpec(name="o1", monitor_name="o1", direction="+x", polarization="tm"),
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
    phase_err = np.angle(demod[0] / (amp * np.exp(1j * phase)))
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
    sim.devices = [
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
    sim.devices = []
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
            PortSpec(name="o1", monitor_name="o1", direction="+x", polarization="tm"),
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
    phase_err = np.angle(z / (amp * np.exp(1j * phase)))
    assert abs(phase_err) < 0.08


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

    def fake_projection(self, spec, monitor, frequency, cache, mode_pad_cells=6):
        return {
            "e_component": "Ez",
            "h_component": "Hy",
            "pinv": pinv,
            "condition_number": 1.0,
        }

    monkeypatch.setattr(Simulation, "_sample_monitor_component_dft", fake_sample)
    monkeypatch.setattr(Simulation, "_build_port_projection", fake_projection)

    sim = Simulation.__new__(Simulation)
    sim.is_3d = False
    sim.plane_2d = "xy"
    sim.resolution = 1.0
    sim.devices = [
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


def test_get_S_matrix_modal_dft_keys_shapes_and_valid_mask(monkeypatch):
    sim = Simulation.__new__(Simulation)
    sim.devices = []
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

    def fake_extract(self, ports, frequencies, min_incident_db=-40.0, return_power=True):
        assert np.allclose(frequencies, freqs)
        return waves

    monkeypatch.setattr(Simulation, "extract_port_waves_dft", fake_extract)
    result = sim.get_S_matrix_modal_dft(
        source_port="o1",
        ports=[
            PortSpec(name="o1", monitor_name="o1", direction="+x", polarization="tm"),
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
