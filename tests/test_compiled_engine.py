import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    PML,
    Design,
    GaussianSource,
    Material,
    ModeSource,
    Monitor,
    Rectangle,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    um,
)
from beamz.devices.monitors.compiler import CompiledMonitorSpec, compile_monitor_specs
from beamz.devices.sources.compiler import compile_source_specs
from beamz.design.spec import DesignSpec
from beamz.simulation.compiled import (
    CompiledRunConfig,
    CompiledSimulation,
    EngineState,
    MonitorState,
)


@pytest.fixture
def small_sim_params():
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=2, safety_factor=0.95, points_per_wavelength=10
    )
    domain = 5.0 * wl
    steps = 120
    t = np.arange(0, steps * dt, dt)
    freq = LIGHT_SPEED / wl
    signal = ramped_cosine(
        t,
        amplitude=1.0,
        frequency=freq,
        ramp_duration=2 / freq,
        t_max=t[-1] * 0.4,
    )
    return wl, dx, dt, domain, steps, t, signal


def test_run_compiled_matches_python_step_path(small_sim_params):
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))

    source_a = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    source_b = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )

    sim_python = Simulation(
        design=design.copy(),
        devices=[source_a],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )
    sim_compiled = Simulation(
        design=design.copy(),
        devices=[source_b],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    while sim_python.step():
        pass
    sim_compiled.run_compiled(progress=False)

    ez_python = np.asarray(sim_python.fields.Ez)
    ez_compiled = np.asarray(sim_compiled.fields.Ez)

    assert sim_compiled.current_step == sim_python.current_step
    assert np.allclose(ez_compiled, ez_python, rtol=2e-3, atol=2e-4)


def test_compiled_monitor_power_is_populated(small_sim_params):
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))

    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    monitor = Monitor(
        start=(domain * 0.35, domain * 0.35),
        end=(domain * 0.35, domain * 0.65),
        record_interval=3,
    )

    sim = Simulation(
        design=design,
        devices=[source, monitor],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    sim.run_compiled(progress=False)

    assert len(monitor.power_history) > 0
    assert len(monitor.power_timestamps) == len(monitor.power_history)
    assert np.isfinite(np.asarray(monitor.power_history)).all()


def test_source_compilation_accepts_simulation_specs_only(small_sim_params):
    wl, dx, _dt, domain, steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    sim = Simulation(
        design=design,
        devices=[source],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    source_specs = compile_source_specs(
        devices=sim.spec.devices,
        source_states=None,
        fields=sim.fields,
        dt=sim.dt,
        resolution=sim.resolution,
        num_steps=steps,
        t0=float(sim.time[0]),
    )

    assert len(source_specs) == 1
    assert source_specs[0].component == "Ez"
    assert source_specs[0].waveform.shape == (steps,)
    assert source_specs[0].coeff.ndim == 2


def test_monitor_compilation_accepts_simulation_specs_only(small_sim_params):
    wl, dx, _dt, domain, steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    monitor = Monitor(
        start=(domain * 0.35, domain * 0.35),
        end=(domain * 0.35, domain * 0.65),
        record_interval=3,
        frequency_points=[LIGHT_SPEED / wl],
        frequency_record_interval=1,
    )
    sim = Simulation(
        design=design,
        devices=[source, monitor],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    monitor_specs, max_records = compile_monitor_specs(
        devices=sim.spec.devices,
        monitor_states=None,
        fields=sim.fields,
        resolution=sim.resolution,
        num_steps=steps,
        dt=sim.dt,
    )

    assert len(monitor_specs) == 1
    assert monitor_specs[0].name == (monitor.spec.name or "monitor_0")
    assert monitor_specs[0].freq_count == 1
    assert max_records == int(np.ceil(steps / monitor.spec.record_interval))


def test_grid_backed_simulation_reuses_live_material_grid(small_sim_params):
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    grid = design.rasterize(dx)
    grid.permittivity[:] = 2.5
    grid.permittivity[0, 0] = 7.0

    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    sim = Simulation(
        design=grid,
        devices=[source],
        boundaries=[],
        time=t,
        resolution=dx,
    )

    assert isinstance(sim.spec.design, DesignSpec)
    assert np.allclose(sim.fields.permittivity, grid.permittivity)
    assert sim.fields.permittivity[0, 0] == pytest.approx(7.0)


def test_compiled_monitor_accumulates_across_chunks(small_sim_params):
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))

    source_a = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    monitor_a = Monitor(
        start=(domain * 0.35, domain * 0.35),
        end=(domain * 0.35, domain * 0.65),
        record_interval=3,
    )
    sim_full = Simulation(
        design=design.copy(),
        devices=[source_a, monitor_a],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    source_b = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    monitor_b = Monitor(
        start=(domain * 0.35, domain * 0.35),
        end=(domain * 0.35, domain * 0.65),
        record_interval=3,
    )
    sim_chunked = Simulation(
        design=design.copy(),
        devices=[source_b, monitor_b],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    sim_full.run_compiled(num_steps=40, progress=False)
    sim_chunked.run_compiled(
        num_steps=40,
        record_interval=10,  # force chunked execution path
        record_fields=["Ez"],
        progress=False,
    )

    p_full = np.asarray(monitor_a.power_history)
    p_chunked = np.asarray(monitor_b.power_history)
    t_full = np.asarray(monitor_a.power_timestamps)
    t_chunked = np.asarray(monitor_b.power_timestamps)

    assert p_full.size > 0
    assert p_chunked.size == p_full.size
    assert t_chunked.size == t_full.size
    assert np.allclose(p_chunked, p_full, rtol=5e-3, atol=5e-5)
    assert np.allclose(t_chunked, t_full, rtol=0.0, atol=0.0)


def test_compiled_frequency_monitor_matches_direct_sum(small_sim_params):
    wl, dx, dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))

    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    freq = LIGHT_SPEED / wl
    monitor = Monitor(
        start=(domain * 0.35, domain * 0.35),
        end=(domain * 0.35, domain * 0.65),
        record_interval=1,
        frequency_points=[freq],
        frequency_record_interval=1,
    )

    sim = Simulation(
        design=design,
        devices=[source, monitor],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )
    sim.run_compiled(num_steps=60, progress=False)

    assert monitor.frequency_flux_spectrum.shape == (1,)
    assert np.isfinite(monitor.frequency_flux_spectrum).all()

    power = np.asarray(monitor.power_history, dtype=np.float64)
    ts = np.asarray(monitor.power_timestamps, dtype=np.float64)
    direct = np.sum(power * np.exp(-1j * 2.0 * np.pi * freq * ts)) * dt
    assert np.allclose(
        monitor.frequency_flux_spectrum[0],
        direct,
        rtol=5e-3,
        atol=5e-6,
    )


def test_compiled_frequency_monitor_accumulates_across_chunks(small_sim_params):
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    freqs = [LIGHT_SPEED / wl, 1.1 * LIGHT_SPEED / wl]

    source_a = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    monitor_a = Monitor(
        start=(domain * 0.35, domain * 0.35),
        end=(domain * 0.35, domain * 0.65),
        record_interval=2,
        frequency_points=freqs,
        frequency_record_interval=1,
    )
    sim_full = Simulation(
        design=design.copy(),
        devices=[source_a, monitor_a],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    source_b = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    monitor_b = Monitor(
        start=(domain * 0.35, domain * 0.35),
        end=(domain * 0.35, domain * 0.65),
        record_interval=2,
        frequency_points=freqs,
        frequency_record_interval=1,
    )
    sim_chunked = Simulation(
        design=design.copy(),
        devices=[source_b, monitor_b],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    sim_full.run_compiled(num_steps=50, progress=False)
    sim_chunked.run_compiled(
        num_steps=50,
        record_interval=10,
        record_fields=["Ez"],
        progress=False,
    )

    s_full = np.asarray(monitor_a.frequency_flux_spectrum)
    s_chunked = np.asarray(monitor_b.frequency_flux_spectrum)
    assert s_full.shape == (2,)
    assert s_chunked.shape == s_full.shape
    assert np.allclose(s_chunked, s_full, rtol=5e-3, atol=5e-6)


def test_compiled_frequency_monitor_3d_populated():
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=3, safety_factor=0.95, points_per_wavelength=6
    )
    domain = 2.0 * wl
    depth = 1.5 * wl
    t = np.arange(0, 24 * dt, dt)
    freq = LIGHT_SPEED / wl
    signal = ramped_cosine(
        t,
        amplitude=1.0,
        frequency=freq,
        ramp_duration=2 / freq,
        t_max=t[-1] * 0.6,
    )

    design = Design(
        width=domain,
        height=domain,
        depth=depth,
        material=Material(permittivity=1.0),
    )
    source = GaussianSource(
        position=(domain * 0.45, domain * 0.5, depth * 0.5),
        width=wl / 5,
        signal=signal,
    )
    monitor = Monitor(
        design=design,
        start=(domain * 0.65, domain * 0.2, depth * 0.2),
        plane_normal="x",
        plane_position=domain * 0.65,
        size=(domain * 0.6, depth * 0.6),
        record_interval=2,
        frequency_points=[freq],
        frequency_record_interval=1,
        record_fields=False,
    )
    sim = Simulation(
        design=design,
        devices=[source, monitor],
        boundaries=[PML(thickness=0.6 * wl, edges="all")],
        time=t,
        resolution=dx,
    )
    sim.run_compiled(num_steps=12, progress=False)

    spec = np.asarray(monitor.frequency_flux_spectrum)
    assert spec.shape == (1,)
    assert np.isfinite(spec).all()
    assert len(monitor.power_history) > 0
    assert np.isfinite(np.asarray(monitor.power_history)).all()


def test_compiled_dft_component_monitor_populated(small_sim_params):
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    freq = LIGHT_SPEED / wl
    monitor = Monitor(
        start=(domain * 0.35, domain * 0.35),
        end=(domain * 0.35, domain * 0.65),
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=[freq],
        dft_components=("Ez", "Hy"),
        dft_t_start=0.0,
        dft_t_end=float(t[-1]),
        dft_window="rect",
        dft_record_every_step=True,
        record_interval=2,
    )
    sim = Simulation(
        design=design,
        devices=[source, monitor],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )
    sim.run_compiled(num_steps=60, progress=False)

    ez_dft = np.asarray(monitor.get_dft_component("Ez"))
    hy_dft = np.asarray(monitor.get_dft_component("Hy"))
    assert ez_dft.shape[0] == 1
    assert hy_dft.shape == ez_dft.shape
    assert ez_dft.shape[1] > 0
    assert np.isfinite(ez_dft).all()
    assert np.isfinite(hy_dft).all()
    assert np.max(np.abs(ez_dft)) > 0.0
    assert np.max(np.abs(hy_dft)) > 0.0


def test_compiled_static_monitor_dft_uses_current_sample_phase():
    program = CompiledSimulation.__new__(CompiledSimulation)
    program.config = CompiledRunConfig(
        resolution=1.0,
        dt=1.0,
        num_steps=1,
        plane_2d="xy",
        is_3d=False,
    )
    program.monitor_specs = (
        CompiledMonitorSpec(
            name="m",
            monitor_index=0,
            is_3d=False,
            record_interval=1,
            accumulate_power=False,
            power_scale=1.0,
            accumulate_frequency=True,
            freq_record_interval=1,
            freq_count=1,
            freq_hz=jnp.asarray([1.0], dtype=jnp.float32),
            freq_rot_re=jnp.asarray([0.0], dtype=jnp.float32),
            freq_rot_im=jnp.asarray([-1.0], dtype=jnp.float32),
            dft_enabled=True,
            dft_record_interval=1,
            dft_t_start=0.0,
            dft_t_end=1.0,
            dft_window_code=0,
            dft_point_count=1,
            dft_component_mask=jnp.asarray([0, 0, 1, 0, 0, 0], dtype=jnp.float32),
            x_ex=jnp.asarray([0], dtype=jnp.int32),
            y_ex=jnp.asarray([0], dtype=jnp.int32),
            valid_ex=jnp.asarray([0.0], dtype=jnp.float32),
            x_ey=jnp.asarray([0], dtype=jnp.int32),
            y_ey=jnp.asarray([0], dtype=jnp.int32),
            valid_ey=jnp.asarray([0.0], dtype=jnp.float32),
            x_ez=jnp.asarray([0], dtype=jnp.int32),
            y_ez=jnp.asarray([0], dtype=jnp.int32),
            valid_ez=jnp.asarray([1.0], dtype=jnp.float32),
            x_hx=jnp.asarray([0], dtype=jnp.int32),
            y_hx=jnp.asarray([0], dtype=jnp.int32),
            valid_hx=jnp.asarray([0.0], dtype=jnp.float32),
            x_hy=jnp.asarray([0], dtype=jnp.int32),
            y_hy=jnp.asarray([0], dtype=jnp.int32),
            valid_hy=jnp.asarray([0.0], dtype=jnp.float32),
            x_hz=jnp.asarray([0], dtype=jnp.int32),
            y_hz=jnp.asarray([0], dtype=jnp.int32),
            valid_hz=jnp.asarray([0.0], dtype=jnp.float32),
        ),
    )

    monitor_state = MonitorState(
        powers=jnp.zeros((1, 1), dtype=jnp.float32),
        timestamps=jnp.zeros((1, 1), dtype=jnp.float32),
        counts=jnp.zeros((1,), dtype=jnp.int32),
        freq_flux_re=jnp.zeros((1, 1), dtype=jnp.float32),
        freq_flux_im=jnp.zeros((1, 1), dtype=jnp.float32),
        freq_phase_re=jnp.ones((1, 1), dtype=jnp.float32),
        freq_phase_im=jnp.zeros((1, 1), dtype=jnp.float32),
        dft_vec_re=jnp.zeros((1, 6, 1, 1), dtype=jnp.float32),
        dft_vec_im=jnp.zeros((1, 6, 1, 1), dtype=jnp.float32),
        dft_weight_sum=jnp.zeros((1, 1), dtype=jnp.float32),
    )

    updated = program._update_monitors(
        monitor_state,
        abs_step=jnp.asarray(0, dtype=jnp.int32),
        t_phys=jnp.asarray(0.0, dtype=jnp.float32),
        dt_scalar=jnp.asarray(1.0, dtype=jnp.float32),
        ex=jnp.zeros((1, 1), dtype=jnp.float32),
        ey=jnp.zeros((1, 1), dtype=jnp.float32),
        ez=jnp.asarray([[2.0]], dtype=jnp.float32),
        hx=jnp.zeros((1, 1), dtype=jnp.float32),
        hy=jnp.zeros((1, 1), dtype=jnp.float32),
        hz=jnp.zeros((1, 1), dtype=jnp.float32),
        monitors_2d=program.monitor_specs,
    )

    np.testing.assert_allclose(
        updated.dft_vec_re[0, 2, 0, 0], 2.0, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        updated.dft_vec_im[0, 2, 0, 0], 0.0, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(updated.freq_phase_re[0, 0], 0.0, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(updated.freq_phase_im[0, 0], -1.0, rtol=1e-7, atol=1e-7)


def test_compiled_program_compiles_once(small_sim_params):
    _wl, _dx, _dt, _domain, _steps, _t, _signal = small_sim_params

    wl = 1.2 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=2, safety_factor=0.95, points_per_wavelength=8
    )
    t = np.arange(0, 40 * dt, dt)
    design = Design(width=4 * wl, height=4 * wl, material=Material(permittivity=1.0))

    sim = Simulation(
        design=design,
        devices=[],
        boundaries=[PML(thickness=1.0 * wl)],
        time=t,
        resolution=dx,
    )

    program = sim.compile(num_steps=20)
    assert program.compile_count == 0

    eng0 = EngineState(
        ex=sim.fields.Ex,
        ey=sim.fields.Ey,
        ez=sim.fields.Ez,
        hx=sim.fields.Hx,
        hy=sim.fields.Hy,
        hz=sim.fields.Hz,
        t=jnp.asarray(sim.t, dtype=jnp.float32),
        current_step=jnp.asarray(sim.current_step, dtype=jnp.int32),
    )
    mon0 = MonitorState(
        powers=jnp.zeros((0, 0), dtype=jnp.float32),
        timestamps=jnp.zeros((0, 0), dtype=jnp.float32),
        counts=jnp.zeros((0,), dtype=jnp.int32),
        freq_flux_re=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_flux_im=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_phase_re=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_phase_im=jnp.zeros((0, 0), dtype=jnp.float32),
        dft_vec_re=jnp.zeros((0, 0, 0, 0), dtype=jnp.float32),
        dft_vec_im=jnp.zeros((0, 0, 0, 0), dtype=jnp.float32),
        dft_weight_sum=jnp.zeros((0, 0), dtype=jnp.float32),
    )

    eng1, _, _ = program.run(eng0, mon0)
    assert program.compile_count == 1

    # Recreate states since donation invalidates buffers.
    eng1_input = EngineState(
        ex=eng1.ex,
        ey=eng1.ey,
        ez=eng1.ez,
        hx=eng1.hx,
        hy=eng1.hy,
        hz=eng1.hz,
        t=eng1.t,
        current_step=eng1.current_step,
    )
    mon1 = MonitorState(
        powers=jnp.zeros((0, 0), dtype=jnp.float32),
        timestamps=jnp.zeros((0, 0), dtype=jnp.float32),
        counts=jnp.zeros((0,), dtype=jnp.int32),
        freq_flux_re=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_flux_im=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_phase_re=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_phase_im=jnp.zeros((0, 0), dtype=jnp.float32),
        dft_vec_re=jnp.zeros((0, 0, 0, 0), dtype=jnp.float32),
        dft_vec_im=jnp.zeros((0, 0, 0, 0), dtype=jnp.float32),
        dft_weight_sum=jnp.zeros((0, 0), dtype=jnp.float32),
    )
    program.run(eng1_input, mon1)
    assert program.compile_count == 1


def test_compiled_jaxpr_has_no_host_callbacks(small_sim_params):
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )

    sim = Simulation(
        design=design,
        devices=[source],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    program = sim.compile(num_steps=8)
    eng0 = EngineState(
        ex=sim.fields.Ex,
        ey=sim.fields.Ey,
        ez=sim.fields.Ez,
        hx=sim.fields.Hx,
        hy=sim.fields.Hy,
        hz=sim.fields.Hz,
        t=jnp.asarray(sim.t, dtype=jnp.float32),
        current_step=jnp.asarray(sim.current_step, dtype=jnp.int32),
    )
    mon0 = MonitorState(
        powers=jnp.zeros((0, 0), dtype=jnp.float32),
        timestamps=jnp.zeros((0, 0), dtype=jnp.float32),
        counts=jnp.zeros((0,), dtype=jnp.int32),
        freq_flux_re=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_flux_im=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_phase_re=jnp.zeros((0, 0), dtype=jnp.float32),
        freq_phase_im=jnp.zeros((0, 0), dtype=jnp.float32),
        dft_vec_re=jnp.zeros((0, 0, 0, 0), dtype=jnp.float32),
        dft_vec_im=jnp.zeros((0, 0, 0, 0), dtype=jnp.float32),
        dft_weight_sum=jnp.zeros((0, 0), dtype=jnp.float32),
    )

    program._build_scan()
    jaxpr = jax.make_jaxpr(program._compiled_scan)(
        eng0, mon0, program._update_coefficients()
    )
    assert "host_callback" not in str(jaxpr).lower()


def test_compile_mode_source_builds_e_and_h_specs():
    wl = 1.55 * um
    n_core = 2.0
    n_clad = 1.45
    dx, dt = calc_optimal_fdtd_params(
        wl, n_core, dims=2, safety_factor=0.95, points_per_wavelength=10
    )

    width = 8 * wl
    height = 5 * wl
    wg_w = 0.8 * wl

    design = Design(
        width=width, height=height, material=Material(permittivity=n_clad**2)
    )
    design += Rectangle(
        position=(width / 2, height / 2),
        width=width,
        height=wg_w,
        material=Material(permittivity=n_core**2),
    )

    t = np.arange(0, 80 * dt, dt)
    freq = LIGHT_SPEED / wl
    signal = ramped_cosine(
        t,
        amplitude=0.1,
        frequency=freq,
        ramp_duration=2 / freq,
        t_max=t[-1] * 0.5,
    )

    source = ModeSource(
        grid=design.rasterize(resolution=dx),
        center=(2 * wl, height / 2),
        width=2.0 * wg_w,
        wavelength=wl,
        pol="tm",
        signal=signal,
        direction="+x",
    )

    sim = Simulation(
        design=design,
        devices=[source],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    program = sim.compile(num_steps=20)
    assert any(spec.timing == "h" for spec in program.source_specs)
    assert any(spec.timing == "e" for spec in program.source_specs)

    sim.run_compiled(num_steps=20, progress=False)
    assert np.isfinite(np.asarray(sim.fields.Ez)).all()


def test_cache_reuse_across_equal_chunks(small_sim_params):
    """Equal-sized chunks should reuse the same compiled program (compile_count == 1)."""
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )

    sim = Simulation(
        design=design,
        devices=[source],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    # Run with record_interval to force multiple equal-sized chunks.
    chunk_size = 30
    sim.run_compiled(num_steps=90, record_interval=chunk_size, progress=False)

    # The program should have been compiled only once (all chunks are size 30).
    assert sim._compiled_program is not None
    assert sim._compiled_program.compile_count == 1


def test_waveform_absolute_indexing_correctness(small_sim_params):
    """Chunked execution with absolute waveform indexing should match single-shot."""
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))

    source_a = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    source_b = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )

    sim_single = Simulation(
        design=design.copy(),
        devices=[source_a],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )
    sim_chunked = Simulation(
        design=design.copy(),
        devices=[source_b],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    # Single-shot: run all 120 steps at once.
    sim_single.run_compiled(num_steps=120, progress=False)

    # Chunked: run 4 chunks of 30 steps each.
    sim_chunked.run_compiled(num_steps=120, record_interval=30, progress=False)

    ez_single = np.asarray(sim_single.fields.Ez)
    ez_chunked = np.asarray(sim_chunked.fields.Ez)

    assert sim_single.current_step == sim_chunked.current_step
    assert np.allclose(ez_single, ez_chunked, rtol=1e-5, atol=1e-6)
