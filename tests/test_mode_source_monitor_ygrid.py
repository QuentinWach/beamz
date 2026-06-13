from __future__ import annotations

import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    Design,
    Material,
    ModeSource,
    Monitor,
    Simulation,
    calc_optimal_fdtd_params,
)
from beamz.devices.sources.mode import (
    _make_3d_mode_basis_profiles,
    _modal_overlap_3d_profiles,
)
from beamz.simulation.yee import component_coordinates_3d_um
from tests.utils import TEST_WAVELENGTH

_COMPONENTS_3D = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _build_uniform_3d_source_case(direction: str, pol: str):
    wavelength = TEST_WAVELENGTH
    n_bg = 1.5
    long_span = 2.4 * wavelength
    transverse_span = 1.8 * wavelength
    axis = direction[1]
    spans = {
        "x": (long_span, transverse_span, transverse_span),
        "y": (transverse_span, long_span, transverse_span),
        "z": (transverse_span, transverse_span, long_span),
    }[axis]
    design = Design(
        width=spans[0],
        height=spans[1],
        depth=spans[2],
        material=Material(n_bg**2),
    )
    dx, dt = calc_optimal_fdtd_params(
        wavelength,
        n_bg,
        dims=3,
        safety_factor=0.9,
        points_per_wavelength=6,
        width=design.width,
        height=design.height,
        depth=design.depth,
    )
    freq = LIGHT_SPEED / wavelength
    omega = 2.0 * np.pi * freq
    time = np.arange(0.0, 6.0 / freq, dt)
    grid = design.rasterize(resolution=dx)
    source = ModeSource(
        grid=grid,
        center=(0.5 * design.width, 0.5 * design.height, 0.5 * design.depth),
        width=1.2 * wavelength,
        height=1.2 * wavelength,
        wavelength=wavelength,
        pol=pol,
        signal=np.cos(omega * time),
        signal_quadrature=np.sin(omega * time),
        direction=direction,
    )
    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[],
        boundaries=[],
        time=time[:4],
        resolution=dx,
    )
    source.initialize(sim.fields.permittivity, dx, dt=dt)
    return source, sim, dt, freq


def _assert_reconstructs(full, masked, delta):
    delta_norm = 0.0
    for component, delta_arr in delta.items():
        reconstructed = np.asarray(masked[component]) + np.asarray(delta_arr)
        target = np.asarray(full[component])
        err = float(np.linalg.norm(reconstructed - target))
        ref = max(float(np.linalg.norm(target)), float(np.linalg.norm(delta_arr)), 1.0)
        assert err / ref < 2e-6, component
        delta_norm += float(np.linalg.norm(delta_arr))
    assert delta_norm > 1e-9


@pytest.mark.component
@pytest.mark.parametrize(
    ("direction", "pol"),
    [("+x", "tm"), ("+y", "te"), ("+z", "tm")],
)
def test_split_3d_mode_source_residual_reconstructs_launched_incident_update(
    direction, pol
):
    source, sim, dt, freq = _build_uniform_3d_source_case(direction, pol)
    t_e = 2.25 / freq
    t_h = t_e - 0.5 * dt

    full_prev = source._build_incident_3d_state(
        sim.fields, t_e=t_e, t_h=t_h, dt=dt, masked=False
    )
    masked_prev = source._build_incident_3d_state(
        sim.fields, t_e=t_e, t_h=t_h, dt=dt, masked=True
    )

    h_full = source._advance_incident_h_3d(sim.fields, full_prev, dt)
    h_target = source._mask_incident_3d_state_to_launched_side(h_full)
    h_masked = source._advance_incident_h_3d(sim.fields, masked_prev, dt)
    h_delta = source._compute_discrete_3d_h_delta(sim.fields, t=t_e, dt=dt)
    _assert_reconstructs(h_target, h_masked, h_delta)

    e_full = source._advance_incident_e_3d(sim.fields, full_prev, h_full, dt)
    e_target = source._mask_incident_3d_state_to_launched_side(e_full)
    e_masked = source._advance_incident_e_3d(sim.fields, masked_prev, h_target, dt)
    e_delta = source._compute_discrete_3d_e_delta(sim.fields, t=t_e, dt=dt)
    _assert_reconstructs(e_target, e_masked, e_delta)


def _affine_value(x, y, z, dx):
    xn = np.asarray(x, dtype=np.float64) / float(dx)
    yn = np.asarray(y, dtype=np.float64) / float(dx)
    zn = np.asarray(z, dtype=np.float64) / float(dx)
    return (1.3 + 0.7 * xn - 0.2 * yn + 0.4 * zn) + 1j * (
        -0.6 + 0.11 * xn + 0.23 * yn - 0.31 * zn
    )


def _component_affine_array(component: str, grid_shape, dx):
    coords_um = component_coordinates_3d_um(component, grid_shape, dx / 1e-6)
    z = np.asarray(coords_um["z"], dtype=np.float64) * 1e-6
    y = np.asarray(coords_um["y"], dtype=np.float64) * 1e-6
    x = np.asarray(coords_um["x"], dtype=np.float64) * 1e-6
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return _affine_value(xx, yy, zz, dx)


def _monitor_for_axis(axis: str, dx: float) -> Monitor:
    if axis == "x":
        start = (3.25 * dx, 1.25 * dx, 1.25 * dx)
        end = (3.25 * dx, 5.75 * dx, 4.75 * dx)
    elif axis == "y":
        start = (1.25 * dx, 3.25 * dx, 1.25 * dx)
        end = (6.75 * dx, 3.25 * dx, 4.75 * dx)
    else:
        start = (1.25 * dx, 1.25 * dx, 3.25 * dx)
        end = (6.75 * dx, 5.75 * dx, 3.25 * dx)
    return Monitor(
        start=start,
        end=end,
        name=f"{axis}_affine",
        record_fields=True,
        accumulate_power=False,
    )


def _expected_monitor_affine(monitor: Monitor, axis: str, target0, target1, dx):
    aa, bb = np.meshgrid(target0, target1, indexing="ij")
    if axis == "x":
        x = np.full_like(aa, float(monitor.plane_position))
        z, y = aa, bb
    elif axis == "y":
        y = np.full_like(aa, float(monitor.plane_position))
        z, x = aa, bb
    else:
        z = np.full_like(aa, float(monitor.plane_position))
        y, x = aa, bb
    return _affine_value(x, y, z, dx)


@pytest.mark.unit
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_3d_monitor_samples_all_yee_components_on_common_affine_plane(axis):
    dx = 0.1e-6
    grid_shape = (7, 8, 9)
    arrays = {
        component: _component_affine_array(component, grid_shape, dx)
        for component in _COMPONENTS_3D
    }
    monitor = _monitor_for_axis(axis, dx)

    monitor.record_fields_3d(
        arrays["Ex"],
        arrays["Ey"],
        arrays["Ez"],
        arrays["Hx"],
        arrays["Hy"],
        arrays["Hz"],
        t=0.0,
        dx=dx,
        dy=dx,
        dz=dx,
        step=0,
    )

    target0, target1 = monitor.get_analysis_plane_coords_3d(
        dx=dx,
        dy=dx,
        dz=dx,
        field_shape=grid_shape,
    )
    expected = _expected_monitor_affine(monitor, axis, target0, target1, dx)
    for component in _COMPONENTS_3D:
        sampled = np.asarray(monitor.fields[component][0], dtype=np.complex128)
        np.testing.assert_allclose(sampled, expected, rtol=1e-12, atol=1e-12)


def _multi_frequency_real_trace(amplitudes, frequencies, times, *, delay=0.0):
    amp = np.asarray(amplitudes, dtype=np.complex128)
    freq = np.asarray(frequencies, dtype=float)
    t = np.asarray(times, dtype=float)
    phase = np.exp(-1j * 2.0 * np.pi * freq[:, None] * (t[None, :] + float(delay)))
    return np.real(np.sum(amp[:, None] * phase, axis=0))


@pytest.mark.unit
def test_monitor_dft_fft_and_cw_recover_same_yee_phasors():
    freqs = np.asarray([3.0, 7.0], dtype=float)
    dt = 1.0 / 256.0
    n = 1024
    times = np.arange(n, dtype=float) * dt
    e_amp = np.asarray([1.2 - 0.4j, -0.7 + 0.9j], dtype=np.complex128)
    h_amp = np.asarray([0.3 + 0.8j, 1.1 - 0.2j], dtype=np.complex128)
    e_trace = _multi_frequency_real_trace(e_amp, freqs, times)
    h_trace = _multi_frequency_real_trace(h_amp, freqs, times, delay=-0.5 * dt)

    sim = Simulation.__new__(Simulation)
    sim.dt = dt

    mon_dft = Monitor(
        start=(0.0, 0.0),
        end=(0.0, 1.0),
        name="dft",
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=freqs,
        dft_components=("Ez", "Hy"),
        dft_window="none",
    )
    for step, (time, e_val, h_val) in enumerate(zip(times, e_trace, h_trace)):
        mon_dft._update_dft(
            float(time),
            {"Ez": [float(e_val)], "Hy": [float(h_val)]},
            step=step,
        )
    _, e_dft = sim._sample_monitor_component_dft(mon_dft, "Ez", freqs)
    _, h_dft = sim._sample_monitor_component_dft(mon_dft, "Hy", freqs)

    mon_fft = Monitor(start=(0.0, 0.0), end=(0.0, 1.0), name="fft")
    mon_fft.fields["Ez"] = [[float(v)] for v in e_trace]
    mon_fft.fields["Hy"] = [[float(v)] for v in h_trace]
    mon_fft.fields["t"] = list(times)
    _, e_fft = sim._sample_monitor_component_spectrum(
        mon_fft, "Ez", frequencies=freqs, window="none"
    )
    _, h_fft = sim._sample_monitor_component_spectrum(
        mon_fft, "Hy", frequencies=freqs, window="none"
    )
    fft_scale = 0.5 * float(n)

    e_cw = np.asarray(
        [
            sim._demodulate_monitor_component(
                mon_fft,
                "Ez",
                frequency=freq,
                t_start=None,
                avg_cycles=None,
                window="none",
            )[0]
            for freq in freqs
        ],
        dtype=np.complex128,
    )
    h_cw = np.asarray(
        [
            sim._demodulate_monitor_component(
                mon_fft,
                "Hy",
                frequency=freq,
                t_start=None,
                avg_cycles=None,
                window="none",
            )[0]
            for freq in freqs
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(e_dft[:, 0], e_amp, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h_dft[:, 0], h_amp, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(e_fft[:, 0] / fft_scale, e_amp, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h_fft[:, 0] / fft_scale, h_amp, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(e_cw, e_amp, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h_cw, h_amp, rtol=1e-12, atol=1e-12)


def _unit_flux_x_mode(mask):
    e = np.zeros((4, 5), dtype=np.complex128)
    e[mask] = 1.0
    zeros = np.zeros_like(e)
    return _make_3d_mode_basis_profiles(
        {
            "Ex": zeros,
            "Ey": e,
            "Ez": zeros,
            "Hx": zeros,
            "Hy": zeros,
            "Hz": e.copy(),
        },
        axis="x",
        d_area=0.25,
        direction_sign=1.0,
    )


def _projection_from_basis(forward, backward):
    return {
        "components": ("Ey", "Ez", "Hy", "Hz"),
        "axis": "x",
        "d_area": 0.25,
        "direction_sign": 1.0,
        "mode_components": forward,
        "mode_components_bwd": backward,
        "overlap_matrix": np.asarray(
            [
                [
                    _modal_overlap_3d_profiles(forward, forward, "x", 0.25),
                    _modal_overlap_3d_profiles(forward, backward, "x", 0.25),
                ],
                [
                    _modal_overlap_3d_profiles(backward, forward, "x", 0.25),
                    _modal_overlap_3d_profiles(backward, backward, "x", 0.25),
                ],
            ],
            dtype=np.complex128,
        ),
    }


@pytest.mark.unit
def test_grouped_3d_modal_projection_recovers_exact_multimode_coefficients():
    fwd0, bwd0 = _unit_flux_x_mode((slice(0, 2), slice(0, 2)))
    fwd1, bwd1 = _unit_flux_x_mode((slice(2, 4), slice(3, 5)))
    coeff_true = (
        (0.8 - 0.2j, -0.1 + 0.3j),
        (-0.4 + 0.5j, 0.25 + 0.15j),
    )
    field = {}
    for component in _COMPONENTS_3D:
        field[component] = (
            coeff_true[0][0] * fwd0[component]
            + coeff_true[0][1] * bwd0[component]
            + coeff_true[1][0] * fwd1[component]
            + coeff_true[1][1] * bwd1[component]
        )

    coeffs, residual, condition, diagnostics = (
        Simulation._project_modal_coefficients_3d_group(
            field,
            (
                _projection_from_basis(fwd0, bwd0),
                _projection_from_basis(fwd1, bwd1),
            ),
        )
    )

    for actual, expected in zip(coeffs, coeff_true, strict=True):
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-12)
    assert residual < 1e-12
    assert condition < 10.0
    assert diagnostics["residual_balanced"] < 1e-12
