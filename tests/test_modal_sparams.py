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
    result = _run_modal_cw_column(waves, ports, frequency=193e12, output_ports=["o1", "o2"])
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
    result = _run_modal_cw_column(waves, ports, frequency=193e12, output_ports=["o1", "o2"])
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
    low = _run_modal_cw_column(waves_low_loss, ports, frequency=193e12, output_ports=["o1", "o2"])
    high = _run_modal_cw_column(waves_high_loss, ports, frequency=193e12, output_ports=["o1", "o2"])
    assert high["diagnostics"]["power_sum"] < low["diagnostics"]["power_sum"]
    assert high["diagnostics"]["loss_est"] > low["diagnostics"]["loss_est"]
