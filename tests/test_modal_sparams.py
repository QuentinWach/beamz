import numpy as np

from beamz import PortSpec, Simulation


def _run_modal_cw_column(waves, ports, frequency, output_ports):
    sim = Simulation.__new__(Simulation)
    sim.devices = []
    sim.is_3d = False
    sim.plane_2d = "xy"

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

    sim.extract_port_waves_cw = fake_extract.__get__(sim, Simulation)
    return sim.get_S_matrix_modal_cw(
        source_port="o1",
        ports=ports,
        output_ports=output_ports,
        frequency=frequency,
        as_sax=False,
        return_diagnostics=True,
    )


def _run_modal_dft_column(
    waves, ports, frequencies, output_ports, min_incident_db=-40.0
):
    sim = Simulation.__new__(Simulation)
    sim.devices = []
    sim.is_3d = False
    sim.plane_2d = "xy"

    def fake_extract(
        self,
        ports,
        frequencies,
        min_incident_db=-40.0,
        return_power=True,
    ):
        return waves

    sim.extract_port_waves_dft = fake_extract.__get__(sim, Simulation)
    return sim.get_S_matrix_modal_dft(
        source_port="o1",
        ports=ports,
        output_ports=output_ports,
        frequencies=frequencies,
        as_sax=False,
        return_diagnostics=True,
        min_incident_db=min_incident_db,
    )


def test_cw_straight_waveguide_tm_power_bound():
    ports = [
        PortSpec(name="o1", monitor_name="o1", direction="+x", polarization="tm"),
        PortSpec(name="o2", monitor_name="o2", direction="+x", polarization="tm"),
    ]
    waves = {
        "o1": {
            "a_plus": 1.0 + 0.0j,
            "a_minus": 0.01 + 0.0j,
            "a_incident": 1.0 + 0.0j,
        },
        "o2": {
            "a_plus": 0.0 + 0.0j,
            "a_minus": 0.99 * np.exp(1j * 0.1),
        },
    }
    result = _run_modal_cw_column(
        waves, ports, frequency=193e12, output_ports=["o1", "o2"]
    )
    s = result["s_matrix"]
    assert abs(s[("o2", "o1")]) > 0.95
    assert abs(s[("o1", "o1")]) < 0.05
    assert result["diagnostics"]["power_sum"] <= 1.02


def test_cw_straight_waveguide_te_power_bound():
    ports = [
        PortSpec(name="o1", monitor_name="o1", direction="+x", polarization="te"),
        PortSpec(name="o2", monitor_name="o2", direction="+x", polarization="te"),
    ]
    waves = {
        "o1": {
            "a_plus": 1.0 + 0.0j,
            "a_minus": 0.02 + 0.0j,
            "a_incident": 1.0 + 0.0j,
        },
        "o2": {
            "a_plus": 0.0 + 0.0j,
            "a_minus": 0.97 * np.exp(-1j * 0.05),
        },
    }
    result = _run_modal_cw_column(
        waves, ports, frequency=193e12, output_ports=["o1", "o2"]
    )
    s = result["s_matrix"]
    assert abs(s[("o2", "o1")]) > 0.9
    assert abs(s[("o1", "o1")]) < 0.08
    assert result["diagnostics"]["power_sum"] <= 1.02


def test_cw_mmi_split_balance_tm():
    ports = [
        PortSpec(name="o1", monitor_name="o1", direction="+x", polarization="tm"),
        PortSpec(name="o2", monitor_name="o2", direction="+x", polarization="tm"),
        PortSpec(name="o3", monitor_name="o3", direction="+x", polarization="tm"),
    ]
    waves = {
        "o1": {
            "a_plus": 1.0 + 0.0j,
            "a_minus": 0.06 + 0.0j,
            "a_incident": 1.0 + 0.0j,
        },
        "o2": {
            "a_plus": 0.0 + 0.0j,
            "a_minus": 0.70 + 0.0j,
        },
        "o3": {
            "a_plus": 0.0 + 0.0j,
            "a_minus": 0.69 + 0.0j,
        },
    }
    result = _run_modal_cw_column(
        waves, ports, frequency=193e12, output_ports=["o1", "o2", "o3"]
    )
    s = result["s_matrix"]
    s21_db = 20 * np.log10(max(abs(s[("o2", "o1")]), 1e-12))
    s31_db = 20 * np.log10(max(abs(s[("o3", "o1")]), 1e-12))
    assert abs(s21_db - s31_db) < 0.5


def test_cw_loss_visibility_with_absorption():
    ports = [
        PortSpec(name="o1", monitor_name="o1", direction="+x", polarization="tm"),
        PortSpec(name="o2", monitor_name="o2", direction="+x", polarization="tm"),
    ]
    waves_low_loss = {
        "o1": {
            "a_plus": 1.0 + 0.0j,
            "a_minus": 0.05 + 0.0j,
            "a_incident": 1.0 + 0.0j,
        },
        "o2": {
            "a_plus": 0.0 + 0.0j,
            "a_minus": 0.92 + 0.0j,
        },
    }
    waves_high_loss = {
        "o1": {
            "a_plus": 1.0 + 0.0j,
            "a_minus": 0.05 + 0.0j,
            "a_incident": 1.0 + 0.0j,
        },
        "o2": {
            "a_plus": 0.0 + 0.0j,
            "a_minus": 0.72 + 0.0j,
        },
    }
    low = _run_modal_cw_column(
        waves_low_loss, ports, frequency=193e12, output_ports=["o1", "o2"]
    )
    high = _run_modal_cw_column(
        waves_high_loss, ports, frequency=193e12, output_ports=["o1", "o2"]
    )
    assert high["diagnostics"]["power_sum"] < low["diagnostics"]["power_sum"]
    assert high["diagnostics"]["loss_est"] > low["diagnostics"]["loss_est"]


def test_dft_matches_cw_on_straight_waveguide_subset():
    freqs = np.array([191e12, 193e12, 195e12], dtype=float)
    ports = [
        PortSpec(name="o1", monitor_name="o1", direction="+x", polarization="tm"),
        PortSpec(name="o2", monitor_name="o2", direction="+x", polarization="tm"),
    ]
    waves = {
        "o1": {
            "a_plus": np.ones(3, dtype=np.complex128),
            "a_minus": np.array([0.01, 0.015, 0.02], dtype=np.complex128),
            "a_incident": np.ones(3, dtype=np.complex128),
        },
        "o2": {
            "a_plus": np.zeros(3, dtype=np.complex128),
            "a_minus": np.array(
                [
                    0.98 * np.exp(1j * 0.02),
                    0.975 * np.exp(1j * 0.04),
                    0.97 * np.exp(1j * 0.06),
                ],
                dtype=np.complex128,
            ),
        },
    }
    dft = _run_modal_dft_column(waves, ports, freqs, output_ports=["o1", "o2"])
    for i, f in enumerate(freqs):
        waves_i = {
            "o1": {
                "a_plus": waves["o1"]["a_plus"][i],
                "a_minus": waves["o1"]["a_minus"][i],
                "a_incident": waves["o1"]["a_incident"][i],
            },
            "o2": {
                "a_plus": waves["o2"]["a_plus"][i],
                "a_minus": waves["o2"]["a_minus"][i],
            },
        }
        cw = _run_modal_cw_column(
            waves_i, ports, frequency=float(f), output_ports=["o1", "o2"]
        )
        assert np.isclose(
            dft["s_matrix"][("o1", "o1")][i],
            cw["s_matrix"][("o1", "o1")],
            rtol=1e-12,
            atol=1e-12,
        )
        assert np.isclose(
            dft["s_matrix"][("o2", "o1")][i],
            cw["s_matrix"][("o2", "o1")],
            rtol=1e-12,
            atol=1e-12,
        )


def test_dft_mmi_balance_tm():
    freqs = np.array([192e12, 193.5e12, 195e12], dtype=float)
    ports = [
        PortSpec(name="o1", monitor_name="o1", direction="+x", polarization="tm"),
        PortSpec(name="o2", monitor_name="o2", direction="+x", polarization="tm"),
        PortSpec(name="o3", monitor_name="o3", direction="+x", polarization="tm"),
    ]
    waves = {
        "o1": {
            "a_plus": np.ones(3, dtype=np.complex128),
            "a_minus": 0.05 * np.ones(3, dtype=np.complex128),
            "a_incident": np.ones(3, dtype=np.complex128),
        },
        "o2": {
            "a_plus": np.zeros(3, dtype=np.complex128),
            "a_minus": np.array([0.69, 0.705, 0.695], dtype=np.complex128),
        },
        "o3": {
            "a_plus": np.zeros(3, dtype=np.complex128),
            "a_minus": np.array([0.695, 0.700, 0.692], dtype=np.complex128),
        },
    }
    result = _run_modal_dft_column(waves, ports, freqs, output_ports=["o1", "o2", "o3"])
    s = result["s_matrix"]
    idx = 1
    s21_db = 20 * np.log10(max(abs(s[("o2", "o1")][idx]), 1e-12))
    s31_db = 20 * np.log10(max(abs(s[("o3", "o1")][idx]), 1e-12))
    assert abs(s21_db - s31_db) < 0.5


def test_dft_loss_visibility_with_absorption():
    freqs = np.array([193e12, 194e12], dtype=float)
    ports = [
        PortSpec(name="o1", monitor_name="o1", direction="+x", polarization="tm"),
        PortSpec(name="o2", monitor_name="o2", direction="+x", polarization="tm"),
    ]
    low = {
        "o1": {
            "a_plus": np.ones(2, dtype=np.complex128),
            "a_minus": 0.05 * np.ones(2, dtype=np.complex128),
            "a_incident": np.ones(2, dtype=np.complex128),
        },
        "o2": {
            "a_plus": np.zeros(2, dtype=np.complex128),
            "a_minus": np.array([0.92, 0.91], dtype=np.complex128),
        },
    }
    high = {
        "o1": {
            "a_plus": np.ones(2, dtype=np.complex128),
            "a_minus": 0.05 * np.ones(2, dtype=np.complex128),
            "a_incident": np.ones(2, dtype=np.complex128),
        },
        "o2": {
            "a_plus": np.zeros(2, dtype=np.complex128),
            "a_minus": np.array([0.72, 0.70], dtype=np.complex128),
        },
    }
    low_result = _run_modal_dft_column(low, ports, freqs, output_ports=["o1", "o2"])
    high_result = _run_modal_dft_column(high, ports, freqs, output_ports=["o1", "o2"])
    assert np.nanmean(high_result["diagnostics"]["power_sum"]) < np.nanmean(
        low_result["diagnostics"]["power_sum"]
    )
    assert np.nanmean(high_result["diagnostics"]["loss_est"]) > np.nanmean(
        low_result["diagnostics"]["loss_est"]
    )
