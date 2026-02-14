import numpy as np

from beamz import (
    Design,
    Material,
    Rectangle,
    StaticThermalConfig,
    ThermalBoundaryProfile,
    ThermalScenario,
    ThermalSource,
)


def _r2(x, y):
    coeffs = np.polyfit(x, y, 1)
    y_fit = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def test_mzi_heater_sweep_monotonic_and_near_linear():
    W, H = 30e-6, 12e-6
    design = Design(
        width=W, height=H, material=Material(permittivity=1.0, k=1.38, T0=300.0)
    )

    substrate = Rectangle(
        position=(0.0, 0.0),
        width=W,
        height=7.0e-6,
        material=Material(permittivity=3.45**2, k=130.0, dn_dT=1.86e-4, T0=300.0),
    )
    top_oxide = Rectangle(
        position=(0.0, 7.0e-6),
        width=W,
        height=3.0e-6,
        material=Material(permittivity=1.44**2, k=1.38, dn_dT=1.0e-5, T0=300.0),
    )
    arm_core = Rectangle(
        position=(8e-6, 8.2e-6),
        width=14e-6,
        height=0.22e-6,
        material=Material(permittivity=3.48**2, k=130.0, dn_dT=1.86e-4, T0=300.0),
    )
    heater = Rectangle(
        position=(8e-6, 10.0e-6),
        width=14e-6,
        height=0.3e-6,
        material=Material(permittivity=1.0, k=25.0, T0=300.0),
    )
    for struct in [substrate, top_oxide, arm_core, heater]:
        design += struct

    powers = np.linspace(0.0, 0.05, 8)
    scenario_base = ThermalScenario(
        extrusion_depth_m=150e-6,
        boundary_profile=ThermalBoundaryProfile.photonic_chip(
            sink_thickness_m=0.2e-6,
            sink_temperature_k=300.0,
            top_h_w_m2_k=10.0,
            ambient_temp_k=300.0,
        ),
    )
    result = design.sweep_mzi_heater(
        resolution=0.25e-6,
        powers_w=powers,
        heater=ThermalSource(region=heater),
        optical_region=arm_core,
        arm_length_m=2.0e-3,
        wavelength_m=1.55e-6,
        group_index=4.2,
        scenario_base=scenario_base,
        config=StaticThermalConfig(max_iters=5000, tol=1e-6),
    )

    assert np.all(np.diff(result.delta_t_eff_k) >= -1e-9)
    assert np.all(np.diff(result.delta_n_eff) >= -1e-12)
    assert np.all(np.diff(result.delta_phi_rad) >= -1e-9)

    # In this low-power regime, tuning should be close to linear.
    assert _r2(result.power_w, result.delta_phi_rad) > 0.98

    assert np.isfinite(result.p_pi_w)
    assert result.p_pi_w > 0.0
    assert result.p_pi_w < 0.5
