import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    PML,
    Box,
    Design,
    FluxMonitor,
    GaussianPulse,
    GridSpec,
    Material,
    ModeMonitor,
    ModeSource,
    ModeSpec,
    Port,
    Rectangle,
    SampledSignal,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
)
from beamz.analysis import sparameters as sp
from tests.utils import TEST_WAVELENGTH

pytestmark = [pytest.mark.integration, pytest.mark.simulation]


@pytest.mark.compiled
@pytest.mark.parametrize("grid_kind", ("uniform", "rectilinear"))
@pytest.mark.parametrize("direction", ("+", "-"))
def test_3d_mode_source_suppresses_counterpropagating_power_on_both_grids(
    grid_kind,
    direction,
):
    wavelength = 1.55e-6
    frequency = LIGHT_SPEED / wavelength
    core_index = 3.48
    # Exact grid multiples keep the requested and realized symmetry planes equal.
    domain = (4.06e-6, 3.01e-6, 2.52e-6)
    source_x = -0.8e-6
    plane_size = (0.0, 1.5e-6, 1.2e-6)
    design = Design(background=Material(permittivity=1.0))
    design += Box(
        center=(0.0, 0.0, 0.0),
        # Cell-aligned dimensions keep the rasterized cross-section exactly
        # mirror symmetric on both uniform and auto grids.
        size=(np.inf, 0.49e-6, 0.28e-6),
        material=Material(permittivity=core_index**2),
    )
    grid_spec = (
        GridSpec.uniform(0.07e-6)
        if grid_kind == "uniform"
        else GridSpec.auto(
            wavelength=wavelength,
            min_steps_per_wvl=6,
            max_scale=1.25,
        )
    )
    source = ModeSource(
        center=(source_x, 0.0, 0.0),
        size=plane_size,
        direction=direction,
        source_time=GaussianPulse(
            freq0=frequency,
            fwidth=0.1 * frequency,
            offset=0.5,
        ),
        mode_spec=ModeSpec(num_modes=1, target_neff=0.98 * core_index),
    )
    negative_monitor = FluxMonitor(
        center=(source_x - 0.45e-6, 0.0, 0.0),
        size=plane_size,
        freqs=[frequency],
        name="negative",
    )
    positive_monitor = FluxMonitor(
        center=(source_x + 0.45e-6, 0.0, 0.0),
        size=plane_size,
        freqs=[frequency],
        name="positive",
    )
    near_port = Port(
        center=(source_x + 0.45e-6, 0.0, 0.0),
        size=plane_size,
        name="near_mode",
        direction="+",
        mode_spec=source.mode_spec,
    )
    far_port = Port(
        center=(source_x + 1.6e-6, 0.0, 0.0),
        size=plane_size,
        name="far_mode",
        direction="+",
        mode_spec=source.mode_spec,
    )
    monitors = [
        negative_monitor,
        positive_monitor,
        near_port.to_monitor([frequency]),
        far_port.to_monitor([frequency]),
    ]
    simulation = Simulation(
        domain=domain,
        grid_spec=grid_spec,
        design=design,
        sources=[source],
        monitors=monitors,
        boundaries=[PML(thickness=0.45e-6, formulation="cpml")],
        run_time=20.0 / frequency,
    )

    program = simulation.compile()
    if grid_kind == "uniform":
        assert program.grid.geometry.metric_kind == "isotropic_uniform"
    else:
        assert program.grid.geometry.metric_kind != "isotropic_uniform"
    launch_powers = {
        float(spec.launched_power)
        for spec in program.sources
        if spec.launched_power is not None
    }
    assert sorted(launch_powers) == pytest.approx([source.power], rel=1e-6)
    for spec in program.sources:
        if spec.component not in {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}:
            continue
        full = np.zeros(
            getattr(program.grid, spec.component).shape,
            dtype=np.asarray(spec.coeff).dtype,
        )
        full[spec.index] = spec.coeff
        peak = float(np.max(np.abs(full)))
        for transverse_axis in (0, 1):
            np.testing.assert_allclose(
                np.abs(full),
                np.flip(np.abs(full), axis=transverse_axis),
                rtol=2e-5,
                atol=max(2e-6 * peak, 1e-15),
            )

    result = simulation.run(progress=False)
    launched_name = "positive" if direction == "+" else "negative"
    counter_name = "negative" if direction == "+" else "positive"
    counter_power = abs(float(result[counter_name].flux[0]))
    forward_power = abs(float(result[launched_name].flux[0]))

    assert forward_power > 0.0
    assert counter_power / forward_power < 0.05
    if direction == "+":
        waves = sp._extract_port_waves_dft(
            result,
            ports=[near_port, far_port],
            frequencies=[frequency],
            return_power=True,
        )
        near_modal_power = float(np.asarray(waves[near_port.name]["P_minus"])[0])
        far_modal_power = float(np.asarray(waves[far_port.name]["P_minus"])[0])
        assert far_modal_power == pytest.approx(near_modal_power, rel=0.02)
        assert near_modal_power == pytest.approx(
            abs(float(result[near_port.name].flux[0])), rel=0.02
        )


@pytest.mark.compiled
def test_centered_3d_mode_source_has_a_dominant_forward_branch():
    """A centered 3D straight-guide ModeSource should be strongly directional."""
    wavelength = TEST_WAVELENGTH
    n_core = 2.0
    n_clad = 1.0
    guide_width = 0.6 * wavelength
    span = guide_width * 2.5
    long_span = 6.0 * wavelength
    transverse_span = 2.4 * wavelength

    design = Design(
        width=long_span,
        height=transverse_span,
        depth=transverse_span,
        material=Material(permittivity=n_clad**2),
    )
    design += Rectangle(
        position=(
            0.0,
            transverse_span / 2 - guide_width / 2,
            transverse_span / 2 - guide_width / 2,
        ),
        width=long_span,
        height=guide_width,
        depth=guide_width,
        material=Material(permittivity=n_core**2),
    )

    dx, dt = calc_optimal_fdtd_params(
        wavelength,
        n_core,
        dims=3,
        safety_factor=0.9,
        points_per_wavelength=8,
        width=design.width,
        height=design.height,
        depth=design.depth,
    )
    freq = LIGHT_SPEED / wavelength
    t_total = 8.0 / freq
    time = np.arange(0.0, t_total, dt)
    signal = ramped_cosine(
        time,
        amplitude=1.0,
        frequency=freq,
        ramp_duration=1.0 / freq,
        t_max=t_total,
    )

    center = (design.width / 2, design.height / 2, design.depth / 2)
    source = ModeSource(
        center=(1.2 * wavelength, center[1], center[2]),
        size=(0.0, span, span),
        source_time=SampledSignal(signal, dt=dt, freq0=freq),
        direction="+",
        mode_spec=ModeSpec(polarization="te"),
    )
    monitor = ModeMonitor(
        center=(1.8 * wavelength, center[1], center[2]),
        size=(0.0, span, span),
        freqs=np.array([freq]),
        mode_spec=ModeSpec(polarization="te"),
        name="o1",
    )

    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[monitor],
        boundaries=[PML(thickness=0.8 * wavelength)],
        time=time,
        resolution=dx,
    )
    result = sim.run(progress=False)

    waves = sp._extract_port_waves_dft(
        result,
        ports=[
            Port(
                center=monitor.center,
                size=monitor.size,
                name="o1",
                direction="+",
                mode_spec=ModeSpec(polarization="te"),
            )
        ],
        frequencies=np.array([freq]),
    )
    a_plus = complex(waves["o1"]["a_plus"][0])
    a_minus = complex(waves["o1"]["a_minus"][0])
    major = max(abs(a_plus), abs(a_minus))
    minor = min(abs(a_plus), abs(a_minus))
    reflection_db = 20.0 * np.log10(max(minor / max(major, 1e-18), 1e-12))
    dominance_db = 20.0 * np.log10(max(abs(a_minus), 1e-18) / max(abs(a_plus), 1e-18))

    assert dominance_db >= 15.0, (
        "Expected the +x source-port decomposition to identify the minus branch "
        f"as incident, got dominance={dominance_db:.2f} dB "
        f"(a_plus={a_plus}, a_minus={a_minus})."
    )
    assert reflection_db <= -15.0, (
        "Expected low raw source-port reflection in a centered 3D straight guide, "
        f"got {reflection_db:.2f} dB "
        f"(a_plus={a_plus}, a_minus={a_minus})."
    )

    mode_data = result.mode("o1")
    forward = abs(complex(mode_data.amps.sel(direction="+").values[0, 0]))
    backward = abs(complex(mode_data.amps.sel(direction="-").values[0, 0]))
    assert forward / max(backward, 1e-18) >= 10 ** (15.0 / 20.0)
    assert np.all(np.isfinite(mode_data.flux))
    np.testing.assert_allclose(mode_data.flux, result["o1"].flux)


@pytest.mark.compiled
def test_centered_3d_mode_source_reports_calibrated_launch_power():
    """Compiler calibration and downstream flux should be finite and positive."""
    wavelength = TEST_WAVELENGTH
    n_core = 2.0
    n_clad = 1.0
    guide_width = 0.6 * wavelength
    span = guide_width * 2.5
    long_span = 7.0 * wavelength
    transverse_span = 2.4 * wavelength

    design = Design(
        width=long_span,
        height=transverse_span,
        depth=transverse_span,
        material=Material(permittivity=n_clad**2),
    )
    design += Rectangle(
        position=(
            0.0,
            transverse_span / 2 - guide_width / 2,
            transverse_span / 2 - guide_width / 2,
        ),
        width=long_span,
        height=guide_width,
        depth=guide_width,
        material=Material(permittivity=n_core**2),
    )

    dx, dt = calc_optimal_fdtd_params(
        wavelength,
        n_core,
        dims=3,
        safety_factor=0.9,
        points_per_wavelength=8,
        width=design.width,
        height=design.height,
        depth=design.depth,
    )
    freq = LIGHT_SPEED / wavelength
    period = 1.0 / freq
    t_total = 20.0 * period
    time = np.arange(0.0, t_total, dt)
    signal = ramped_cosine(
        time,
        amplitude=1.0,
        frequency=freq,
        ramp_duration=2.0 * period,
        t_max=2.0 * t_total,
    )

    center = (design.width / 2, design.height / 2, design.depth / 2)
    source = ModeSource(
        center=(1.5 * wavelength, center[1], center[2]),
        size=(0.0, span, span),
        source_time=SampledSignal(signal, dt=dt, freq0=freq),
        direction="+",
        mode_spec=ModeSpec(polarization="te"),
    )
    monitor = FluxMonitor(
        center=(3.5 * wavelength, center[1], center[2]),
        size=(0.0, span, span),
        freqs=np.array([freq]),
        name="flux",
    )

    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[monitor],
        boundaries=[PML(thickness=0.8 * wavelength)],
        time=time,
        resolution=dx,
    )
    result = sim.run(progress=False)

    flux = float(np.asarray(result.monitors["flux"].flux)[0])

    assert result.launched_power() == pytest.approx(source.power, rel=1e-6)
    assert np.isfinite(flux)
    assert 0.0 < flux <= 1.1 * source.power
