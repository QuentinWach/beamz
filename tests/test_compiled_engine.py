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
from beamz.devices import COMPILED_DEVICE_REGISTRY
from beamz.simulation.compiled import EngineState, MonitorState
from beamz.simulation.plans import build_compilation_plan


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
    coeffs = program._update_coefficients()
    if getattr(program, "_compiled_mode", "general") in {
        "engine_only",
        "engine_plus_sources",
    }:
        jaxpr = jax.make_jaxpr(program._compiled_scan)(eng0, coeffs)
    else:
        jaxpr = jax.make_jaxpr(program._compiled_scan)(eng0, mon0, coeffs)
    assert "host_callback" not in str(jaxpr).lower()


def test_compilation_plan_classifies_families(small_sim_params):
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    monitor = Monitor(
        start=(domain * 0.35, domain * 0.35),
        end=(domain * 0.35, domain * 0.65),
        record_interval=2,
    )

    sim = Simulation(
        design=design,
        devices=[source, monitor],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    plan = build_compilation_plan(
        sim,
        backend_platform="cpu",
        num_steps=16,
        loop_kind="scan",
        source_single_slab_dense=False,
        temporal_block_steps=1,
    )

    assert plan.key.boundary_family == "pml"
    assert plan.key.source_family == "single_slab"
    assert plan.key.monitor_family == "power_only"
    assert plan.key.kernel_family == "engine_plus_sources_and_monitors"


def test_compilation_plan_key_varies_with_num_steps(small_sim_params):
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    sim = Simulation(
        design=design,
        devices=[source],
        boundaries=[],
        time=t,
        resolution=dx,
    )

    plan_a = build_compilation_plan(
        sim,
        backend_platform="cpu",
        num_steps=8,
        loop_kind="scan",
        source_single_slab_dense=False,
        temporal_block_steps=1,
    )
    plan_b = build_compilation_plan(
        sim,
        backend_platform="cpu",
        num_steps=12,
        loop_kind="scan",
        source_single_slab_dense=False,
        temporal_block_steps=1,
    )

    assert plan_a.key != plan_b.key


def test_compilation_plan_detects_uniform_scalar_coefficients():
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=3, safety_factor=0.95, points_per_wavelength=8
    )
    t = np.arange(0, 10 * dt, dt)
    design = Design(
        width=4 * wl,
        height=4 * wl,
        depth=4 * wl,
        material=Material(permittivity=1.0),
    )
    sim = Simulation(
        design=design,
        devices=[],
        boundaries=[],
        time=t,
        resolution=dx,
    )

    plan = build_compilation_plan(
        sim,
        backend_platform="cpu",
        num_steps=8,
        loop_kind="scan",
        source_single_slab_dense=False,
        temporal_block_steps=1,
    )

    assert plan.key.dimension_family == "3d"
    assert plan.key.boundary_family == "none"
    assert plan.key.coefficient_layout_family == "uniform_scalar"
    assert plan.key.kernel_family == "engine_only"


def test_compiled_device_registry_covers_shipped_device_types():
    source_type_names = {cls.__name__ for cls in COMPILED_DEVICE_REGISTRY.source_types}
    monitor_type_names = {cls.__name__ for cls in COMPILED_DEVICE_REGISTRY.monitor_types}

    assert source_type_names == {"GaussianSource", "ModeSource"}
    assert monitor_type_names == {"Monitor"}


def test_run_compiled_rejects_unsupported_device_type(small_sim_params):
    class DummyLegacySource:
        def inject(self, fields, t, dt, current_step, resolution, design):
            del fields, t, dt, current_step, resolution, design

    wl, dx, _dt, domain, _steps, t, _signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    sim = Simulation(
        design=design,
        devices=[DummyLegacySource()],
        boundaries=[],
        time=t,
        resolution=dx,
    )

    with pytest.raises(NotImplementedError, match="DummyLegacySource"):
        sim.compile(num_steps=8)


def test_run_compiled_matches_python_in_3d_uniform_no_boundary():
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=3, safety_factor=0.95, points_per_wavelength=8
    )
    steps = 12
    t = np.arange(0, steps * dt, dt)
    design = Design(
        width=4 * wl,
        height=4 * wl,
        depth=4 * wl,
        material=Material(permittivity=1.0),
    )
    freq = LIGHT_SPEED / wl
    signal = ramped_cosine(
        t,
        amplitude=1.0,
        frequency=freq,
        ramp_duration=2 / freq,
        t_max=t[-1] * 0.5,
    )
    source_a = GaussianSource(
        position=(2 * wl, 2 * wl, 2 * wl),
        width=wl / 5,
        signal=signal,
    )
    source_b = GaussianSource(
        position=(2 * wl, 2 * wl, 2 * wl),
        width=wl / 5,
        signal=signal,
    )
    sim_python = Simulation(
        design=design.copy(),
        devices=[source_a],
        boundaries=[],
        time=t,
        resolution=dx,
    )
    sim_compiled = Simulation(
        design=design.copy(),
        devices=[source_b],
        boundaries=[],
        time=t,
        resolution=dx,
    )

    while sim_python.step():
        pass
    sim_compiled.run_compiled(progress=False)

    assert np.allclose(
        np.asarray(sim_compiled.fields.Ez),
        np.asarray(sim_python.fields.Ez),
        rtol=3e-3,
        atol=3e-4,
    )


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
