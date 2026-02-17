import importlib.util

import numpy as np
import pytest

from beamz import Monitor, Simulation, calc_optimal_fdtd_params, design, dxdt


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


def test_get_S_matrix_fft_column_extraction():
    n = 1024
    dt = 1e-15
    k_bin = 20
    freq = k_bin / (n * dt)
    phase = 0.7
    amplitude = 0.4
    time = np.arange(n) * dt

    trace_in = np.cos(2 * np.pi * freq * time)
    trace_out = amplitude * np.cos(2 * np.pi * freq * time + phase)

    mon_in = Monitor(start=(0.0, 0.0), end=(0.0, 1.0), name="o1", record_fields=True)
    mon_out = Monitor(start=(1.0, 0.0), end=(1.0, 1.0), name="o2", record_fields=True)
    mon_in.fields["Ez"] = [[v] for v in trace_in]
    mon_out.fields["Ez"] = [[v] for v in trace_out]
    mon_in.fields["t"] = list(time)
    mon_out.fields["t"] = list(time)

    sim = Simulation.__new__(Simulation)
    sim.devices = [mon_in, mon_out]
    sim.dt = dt
    sim.time = time

    s_matrix = sim.get_S_matrix(
        input_ports=["o1", "o2"],
        output_ports=["o2"],
        source_port="o1",
        frequencies=np.array([freq]),
        as_sax=False,
    )
    s21 = s_matrix[("o2", "o1")]
    assert s21.shape == (1,)

    expected = amplitude * np.exp(1j * phase)
    assert np.isclose(np.abs(s21[0]), np.abs(expected), rtol=0.05)
    phase_error = np.angle(s21[0] / expected)
    assert abs(phase_error) < 0.1
