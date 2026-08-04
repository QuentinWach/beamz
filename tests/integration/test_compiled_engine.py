from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    PEC,
    PML,
    Box,
    Design,
    FieldMonitor,
    FieldRecorder,
    GaussianSource,
    Material,
    ModeSource,
    ModeSpec,
    Rectangle,
    SampledSignal,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    um,
)
from beamz.const import EPS_0, MU_0
from beamz.devices._boundary_compile import compile_metallic_masks
from beamz.devices.monitors.compiler import CompiledMonitorSpec
from beamz.devices.sources.compiler import (
    _analytic_subband_waveforms,
    _sample_waveform,
)
from beamz.devices.sources.specs import CustomSource
from beamz.devices.sources.time import (
    analytic_signal_quadrature,
    partition_weights_by_frequency,
)
from beamz.simulation import observe as monitor_runtime
from beamz.simulation.execute import (
    build_program_scan,
    execution_cache,
    initial_program_state,
    run_program,
)
from beamz.simulation.model import (
    RunConfig,
    SimulationState,
)

pytestmark = [pytest.mark.compiled, pytest.mark.component]


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


def _monitor_result(result, name="monitor_0"):
    assert result is not None
    assert result.results.monitors is not None
    return result.results.monitors[name]


def _fields_for_sim(sim: Simulation):
    return sim.compile().grid


def _monitor_state(**values) -> SimulationState:
    """Build a minimal unified state for isolated monitor-kernel tests."""
    scalar_field = jnp.zeros((1, 1), dtype=jnp.float32)
    return SimulationState(
        ex=scalar_field,
        ey=scalar_field,
        ez=scalar_field,
        hx=scalar_field,
        hy=scalar_field,
        hz=scalar_field,
        cpml_psi_h_terms=(),
        cpml_psi_e_terms=(),
        recorded_fields=(),
        recorded_steps=(),
        recorded_times=(),
        recorded_counts=(),
        t=jnp.asarray(0.0, dtype=jnp.float32),
        current_step=jnp.asarray(0, dtype=jnp.int32),
        **values,
    )


def _material_values(fields, name, index, target_shape):
    value = np.asarray(getattr(fields, name))
    if value.ndim == 0:
        value = np.broadcast_to(value, target_shape)
    return np.asarray(value[index], dtype=np.float32)


def test_compile_validates_loop_configuration(monkeypatch):
    simulation = Simulation(
        domain=(1.0, 1.0),
        resolution=0.5,
        time=np.array((0.0, 1e-10)),
    )

    with pytest.raises(ValueError, match="num_steps must be > 0"):
        simulation.compile(num_steps=0)

    monkeypatch.setenv("BEAMZ_COMPILED_LOOP_KIND", "fori")
    assert simulation.compile(num_steps=1).config.loop_kind == "fori_loop"

    monkeypatch.setenv("BEAMZ_COMPILED_LOOP_KIND", "invalid")
    with pytest.raises(ValueError, match="Invalid BEAMZ_COMPILED_LOOP_KIND"):
        simulation.compile(num_steps=1)


def test_advance_supports_3d_custom_current_source():
    class _CurrentSource:
        def __init__(self, signal, voxel_indices, voxel_weights):
            self.signal = signal
            self._voxel_weights = np.asarray(voxel_weights, dtype=np.float32)
            z_idx = np.asarray([idx[0] for idx in voxel_indices], dtype=np.int32)
            y_idx = np.asarray([idx[1] for idx in voxel_indices], dtype=np.int32)
            x_idx = np.asarray([idx[2] for idx in voxel_indices], dtype=np.int32)
            self._indices = (z_idx, y_idx, x_idx)

        def get_source_terms(self, fields, t, dt, current_step, resolution, design):
            del fields, current_step, resolution, design
            signal_value = float(self.signal(float(t) + 0.5 * float(dt)))
            values = -self._voxel_weights * signal_value
            return {"Ez": (values, self._indices)}, {}

        def to_custom_spec(self, sim):
            fields = _fields_for_sim(sim)
            target_shape = tuple(fields.Ez.shape)
            eps_region = _material_values(fields, "eps_z", self._indices, target_shape)
            sig_region = _material_values(fields, "sig_z", self._indices, target_shape)
            dt = sim.dt
            denom = 1.0 + sig_region * (float(dt) / (2.0 * EPS_0 * eps_region))
            source_coeff = (float(dt) / (EPS_0 * eps_region)) / denom
            coeff = -self._voxel_weights * source_coeff
            waveform = _sample_waveform(
                lambda t_sample, _dt: self.signal(float(t_sample)),
                t0=float(sim.time[0]),
                dt=dt,
                num_steps=sim.num_steps,
                offset_fn=lambda t, dt_: t + 0.5 * dt_,
                total_steps=sim.num_steps,
            )
            return CustomSource(
                component="Ez",
                timing="e",
                index=self._indices,
                coeff=coeff,
                waveform=waveform,
                target_shape=target_shape,
            )

    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=3, safety_factor=0.95, points_per_wavelength=8
    )
    steps = 18
    t = np.arange(0, steps * dt, dt)
    freq = LIGHT_SPEED / wl
    center = 1.25 * wl
    sigma_t = 0.45 / freq

    def signal(t_s: float) -> float:
        envelope = np.exp(-0.5 * ((float(t_s) - center / LIGHT_SPEED) / sigma_t) ** 2)
        carrier = np.cos(2.0 * np.pi * freq * float(t_s))
        return float(envelope * carrier)

    domain = 2.4 * wl
    design = Design(
        width=domain,
        height=domain,
        depth=domain,
        material=Material(permittivity=1.0),
    )
    design += Rectangle(
        position=(0.8 * wl, 0.8 * wl, 0.8 * wl),
        width=0.6 * wl,
        height=0.7 * wl,
        depth=0.5 * wl,
        material=Material(permittivity=2.5),
    )

    grid_n = int(round(domain / dx))
    source_center = np.array([grid_n // 2, grid_n // 2, grid_n // 2], dtype=np.int32)
    voxel_indices = []
    voxel_weights = []
    for dz in (-1, 0):
        for dy in (-1, 0):
            for dx_idx in (-1, 0):
                idx = tuple(
                    (
                        source_center + np.array([dz, dy, dx_idx], dtype=np.int32)
                    ).tolist()
                )
                voxel_indices.append(idx)
                voxel_weights.append(1.0 / 8.0)
    source_b = _CurrentSource(signal, voxel_indices, voxel_weights)
    sim_compiled = Simulation(
        design=design.copy(),
        sources=[],
        boundaries=[PEC(edges="all")],
        time=t,
        resolution=dx,
    )
    sim_compiled = sim_compiled.updated_copy(
        sources=(source_b.to_custom_spec(sim_compiled),)
    )

    program = sim_compiled.compile(num_steps=len(t))
    result = sim_compiled.advance(progress=False)

    assert program.boundary.metallic.ez_mask.shape == result.state.ez.shape

    assert result.state.current_step == len(t)
    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        arr = np.asarray(getattr(result.state, component.lower()))
        assert arr.size > 0
        assert np.isfinite(arr).all()
    assert float(np.max(np.abs(np.asarray(result.state.ez)))) > 0.0


def test_advance_supports_2d_cpml_small_case(small_sim_params):
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))

    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    sim = Simulation(
        design=design,
        sources=[source],
        boundaries=[PML(thickness=1.0 * wl, formulation="cpml")],
        time=t,
        resolution=dx,
    )

    result = sim.advance(progress=False)

    for component in ("Ez", "Hx", "Hy"):
        arr = np.asarray(getattr(result.state, component.lower()))
        assert arr.size > 0
        assert np.isfinite(arr).all()


def test_advance_supports_3d_cpml_small_case():
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=3, safety_factor=0.95, points_per_wavelength=8
    )
    domain = 3.0 * wl
    steps = 36
    t = np.arange(0, steps * dt, dt)
    freq = LIGHT_SPEED / wl
    signal = ramped_cosine(
        t,
        amplitude=1.0,
        frequency=freq,
        ramp_duration=2 / freq,
        t_max=t[-1] * 0.35,
    )

    design = Design(
        width=domain,
        height=domain,
        depth=domain,
        material=Material(permittivity=1.0),
    )
    source = GaussianSource(
        position=(domain / 2, domain / 2, domain / 2),
        width=wl / 5,
        signal=signal,
    )
    sim = Simulation(
        design=design,
        sources=[source],
        boundaries=[PML(thickness=1.0 * wl, formulation="cpml")],
        time=t,
        resolution=dx,
    )

    result = sim.advance(progress=False)

    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        arr = np.asarray(getattr(result.state, component.lower()))
        assert arr.size > 0
        assert np.isfinite(arr).all()


def test_split_3d_cpml_boundaries_preserve_identity_kappa_in_compiled_terms():
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=3, safety_factor=0.95, points_per_wavelength=8
    )
    time = np.arange(0, 2 * dt, dt)
    design = Design(
        width=6.0 * wl,
        height=6.0 * wl,
        depth=6.0 * wl,
        material=Material(permittivity=1.0),
    )
    sim = Simulation(
        design=design,
        sources=[],
        boundaries=[
            PML(
                edges=["left", "right", "top", "bottom"],
                thickness=1.0 * wl,
                formulation="cpml",
            ),
            PML(
                edges=["front", "back"],
                thickness=1.0 * wl,
                formulation="cpml",
            ),
        ],
        time=time,
        resolution=dx,
    )
    program = sim.compile(num_steps=1)

    cy = _fields_for_sim(sim).permittivity.shape[1] // 2
    cx = _fields_for_sim(sim).permittivity.shape[2] // 2
    e_term = program.boundary.cpml.e_terms[4]
    h_term = program.boundary.cpml.h_terms[3]

    assert np.asarray(sim.pml_data["kappa_x"], dtype=np.float64)[
        0, 0, cx
    ] == pytest.approx(1.0)
    assert np.asarray(sim.pml_data["kappa_y"], dtype=np.float64)[
        0, cy, 0
    ] == pytest.approx(1.0)
    assert e_term.slab.low + e_term.slab.high < _fields_for_sim(sim).Ez.shape[2]
    assert h_term.slab.low + h_term.slab.high < _fields_for_sim(sim).Hy.shape[2]


def test_compiled_3d_metallic_masks_are_preserved_in_boundary_plan():
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=3, safety_factor=0.95, points_per_wavelength=8
    )
    edges = frozenset({"front", "bottom", "left"})
    sim = Simulation(
        design=Design(
            width=2.0 * wl,
            height=2.0 * wl,
            depth=2.0 * wl,
            material=Material(permittivity=1.0),
        ),
        sources=[],
        boundaries=[PEC(edges=list(edges))],
        time=np.arange(0, 2 * dt, dt),
        resolution=dx,
    )
    program = sim.compile(num_steps=1)
    fields = _fields_for_sim(sim)
    expected = compile_metallic_masks(
        fields.component_shapes,
        fields.material_grid.shape,
        sim.boundaries,
    )

    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        actual = getattr(program.boundary.metallic, f"{component.lower()}_mask")
        np.testing.assert_array_equal(
            np.asarray(actual), np.asarray(expected[component])
        )


def test_compiled_3d_cpml_uses_material_coefficients():
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=3, safety_factor=0.95, points_per_wavelength=8
    )
    design = Design(
        width=3.0 * wl,
        height=2.0 * wl,
        depth=2.0 * wl,
        material=Material(permittivity=1.0),
    )
    sim = Simulation(
        design=design,
        sources=[],
        boundaries=[PML(edges="all", thickness=0.5 * wl, formulation="cpml")],
        time=np.arange(0, 2 * dt, dt),
        resolution=dx,
    )

    program = sim.compile(num_steps=1)
    coefficients = program.coefficients

    assert program.boundary.cpml.enabled
    assert coefficients.e_decay_x.shape == (0, 0, 0)
    assert coefficients.e_source_x.shape == (0, 0, 0)
    assert coefficients.h_decay_x.shape == (0, 0, 0)
    assert coefficients.h_source_x.shape == (0, 0, 0)
    assert coefficients.e_conductivity_x.shape == ()
    assert coefficients.e_conductivity_y.shape == ()
    assert coefficients.e_conductivity_z.shape == ()
    assert float(coefficients.e_conductivity_x) == 0.0
    assert float(coefficients.e_conductivity_y) == 0.0
    assert float(coefficients.e_conductivity_z) == 0.0
    np.testing.assert_array_equal(
        np.asarray(coefficients.e_permittivity_x),
        np.asarray(_fields_for_sim(sim).eps_x),
    )
    np.testing.assert_array_equal(
        np.asarray(coefficients.e_permittivity_y),
        np.asarray(_fields_for_sim(sim).eps_y),
    )
    np.testing.assert_array_equal(
        np.asarray(coefficients.e_permittivity_z),
        np.asarray(_fields_for_sim(sim).eps_z),
    )
    assert program.boundary.metallic.ex_mask.shape == tuple(
        _fields_for_sim(sim).Ex.shape
    )
    assert program.boundary.metallic.hx_mask.shape == tuple(
        _fields_for_sim(sim).Hx.shape
    )
    assert not bool(np.asarray(program.boundary.metallic.ex_mask).any())
    assert not bool(np.asarray(program.boundary.metallic.hx_mask).any())
    assert program.sharding.layout.logical_shapes["Ex"] == tuple(
        _fields_for_sim(sim).Ex.shape
    )
    assert program.sharding.layout.logical_shapes["Hx"] == tuple(
        _fields_for_sim(sim).Hx.shape
    )
    assert coefficients.h_sigma_m_x.shape == ()
    assert coefficients.h_sigma_m_y.shape == ()
    assert coefficients.h_sigma_m_z.shape == ()
    assert float(coefficients.h_sigma_m_x) == 0.0
    assert float(coefficients.h_sigma_m_y) == 0.0
    assert float(coefficients.h_sigma_m_z) == 0.0
    sim.advance(num_steps=1, progress=False)


def test_compiled_3d_field_recorder_uses_logical_component_shape():
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=3, safety_factor=0.95, points_per_wavelength=8
    )
    design = Design(
        width=3.0 * wl,
        height=2.0 * wl,
        depth=2.0 * wl,
        material=Material(permittivity=1.0),
    )
    sim = Simulation(
        design=design,
        sources=[],
        monitors=[
            FieldRecorder(("Ez",), interval=1, name="frames"),
            FieldRecorder(
                ("Ez", "Hx"),
                interval=1,
                name="plane",
                center=(design.width / 2, design.height / 2, design.depth / 2),
                size=(design.width, design.height, 0.0),
            ),
        ],
        boundaries=[PML(edges="all", thickness=0.5 * wl, formulation="cpml")],
        time=np.arange(0, 2 * dt, dt),
        resolution=dx,
    )

    program = sim.compile(num_steps=2)
    result = sim.advance(num_steps=2, progress=False)
    frames = result.results.monitor("frames")
    plane = result.results.monitor("plane")
    assert not hasattr(result.results, "snapshots")

    assert program.coefficients.e_source_z.shape == (0, 0, 0)
    assert program.sharding.layout.logical_shapes["Ez"] == tuple(
        _fields_for_sim(sim).Ez.shape
    )
    assert frames.fields["Ez"].shape == (2, *_fields_for_sim(sim).Ez.shape)
    assert frames.field_steps.tolist() == [1, 2]
    plane_shape = sim.monitors[1].get_analysis_plane_coords_3d(
        dx=dx,
        dy=dx,
        dz=dx,
        field_shape=tuple(
            max(v)
            for v in zip(*program.sharding.layout.logical_shapes.values(), strict=True)
        ),
    )
    expected_plane_shape = tuple(np.asarray(axis).size for axis in plane_shape)
    assert plane.fields["Ez"].shape == (2, *expected_plane_shape)
    assert plane.fields["Hx"].shape == (2, *expected_plane_shape)
    assert plane.sample_region == program.monitors[1].sample_region
    assert plane.sample_region is not None


def test_compiled_3d_sponge_pml_uses_material_coefficients():
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=3, safety_factor=0.95, points_per_wavelength=8
    )
    design = Design(
        width=2.0 * wl,
        height=2.0 * wl,
        depth=1.5 * wl,
        material=Material(permittivity=1.0),
    )
    sim = Simulation(
        design=design,
        sources=[],
        boundaries=[PML(edges="all", thickness=0.5 * wl)],
        time=np.arange(0, 2 * dt, dt),
        resolution=dx,
    )

    program = sim.compile(num_steps=1)
    assert not program.boundary.cpml.enabled
    coefficients = program.coefficients
    assert coefficients.e_decay_x.shape == (0, 0, 0)
    assert coefficients.e_source_x.shape == (0, 0, 0)
    assert coefficients.h_decay_x.shape == (0, 0, 0)
    assert coefficients.h_source_x.shape == (0, 0, 0)
    np.testing.assert_array_equal(
        np.asarray(coefficients.e_conductivity_x),
        np.asarray(_fields_for_sim(sim).sig_x),
    )
    np.testing.assert_array_equal(
        np.asarray(coefficients.e_permittivity_x),
        np.asarray(_fields_for_sim(sim).eps_x),
    )
    np.testing.assert_array_equal(
        np.asarray(coefficients.h_sigma_m_x),
        np.asarray(_fields_for_sim(sim).sigma_m_hx),
    )

    sim.advance(num_steps=1, progress=False)


def test_simulation_memory_estimate_reports_fields_and_compiled_coefficients():
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=3, safety_factor=0.95, points_per_wavelength=8
    )
    design = Design(
        width=2.0 * wl,
        height=1.5 * wl,
        depth=1.0 * wl,
        material=Material(permittivity=1.0),
    )
    sim = Simulation(
        design=design,
        sources=[],
        boundaries=[PML(edges="all", thickness=0.25 * wl, formulation="cpml")],
        time=np.arange(0, 2 * dt, dt),
        resolution=dx,
    )

    report = sim.memory_estimate(include_compiled=True, num_steps=1)

    field_bytes = sum(
        int(np.asarray(getattr(_fields_for_sim(sim), name)).nbytes)
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    )
    assert report["totals_by_category"]["yee_fields"] == field_bytes
    compiled_names = {entry["name"] for entry in report["compiled"]["entries"]}
    referenced_names = {
        entry["name"] for entry in report["compiled"]["referenced_inputs"]["entries"]
    }
    assert not any(
        key.startswith("use_") and key.endswith("_3d_e_coefficients")
        for key in report["compiled"]["config"]
    )
    assert "e_permittivity_x" not in compiled_names
    assert "e_permittivity_x" in referenced_names
    assert report["compiled"]["referenced_inputs"]["total_bytes"] > 0
    assert (
        report["total_with_compiled_bytes"]
        == report["total_bytes"] + report["compiled"]["total_bytes"]
    )


def test_compiled_uses_material_coefficients_for_3d_loss():
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=3, safety_factor=0.95, points_per_wavelength=8
    )
    design = Design(
        width=3.0 * wl,
        height=3.0 * wl,
        depth=3.0 * wl,
        material=Material(permittivity=1.0),
    )
    design += Box(
        center=(1.5 * wl, 1.5 * wl, 1.5 * wl),
        size=(0.5 * wl, 0.5 * wl, 0.5 * wl),
        material=Material(permittivity=2.0, conductivity=5.0),
    )
    sim = Simulation(
        design=design,
        sources=[],
        boundaries=[PML(edges="all", thickness=0.5 * wl)],
        time=np.arange(0, 2 * dt, dt),
        resolution=dx,
    )

    program = sim.compile(num_steps=1)
    coefficients = program.coefficients

    assert coefficients.e_decay_x.shape == (0, 0, 0)
    assert coefficients.e_source_x.shape == (0, 0, 0)
    assert coefficients.h_decay_x.shape == (0, 0, 0)
    assert coefficients.h_source_x.shape == (0, 0, 0)
    np.testing.assert_array_equal(
        np.asarray(coefficients.e_conductivity_x),
        np.asarray(_fields_for_sim(sim).sig_x),
    )
    np.testing.assert_array_equal(
        np.asarray(coefficients.e_permittivity_x),
        np.asarray(_fields_for_sim(sim).eps_x),
    )
    np.testing.assert_array_equal(
        np.asarray(coefficients.h_sigma_m_x),
        np.asarray(_fields_for_sim(sim).sigma_m_hx),
    )

    sim.advance(num_steps=1, progress=False)


def test_compiled_3d_cpml_profiles_match_expected_x_boundary_embedding():
    wl = 1.55 * um
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=3, safety_factor=0.95, points_per_wavelength=8
    )
    time = np.arange(0, 2 * dt, dt)
    thickness = 1.0 * wl
    design = Design(
        width=4.0 * wl,
        height=2.0 * wl,
        depth=2.0 * wl,
        material=Material(permittivity=1.0),
    )
    sim = Simulation(
        design=design,
        sources=[],
        boundaries=[
            PML(
                edges="all",
                thickness=thickness,
                formulation="cpml",
                kappa_max=4.0,
                alpha_max=300.0,
            )
        ],
        time=time,
        resolution=dx,
    )
    program = sim.compile(num_steps=1)

    pml_cells = int(round(thickness / dx))

    def expected_profile(count: int, *, sample_kind: str):
        boundary = sim.boundaries[0]
        eta = np.sqrt(MU_0 / EPS_0)
        sigma_max = -(
            (boundary.m + 1)
            * np.log(boundary.target_reflection)
            / (2.0 * eta * float(boundary.thickness))
        )
        if sample_kind == "E":
            domain_cells = count - 1
            coordinates = np.arange(count, dtype=np.float32)
        else:
            domain_cells = count
            coordinates = np.arange(count, dtype=np.float32) + 0.5
        low_distance = np.clip(pml_cells - coordinates, 0.0, pml_cells)
        high_distance = np.clip(
            coordinates - (domain_cells - pml_cells), 0.0, pml_cells
        )
        distance = np.maximum(low_distance, high_distance)
        active = distance > 0.0
        u = distance / max(float(pml_cells), 1e-30)
        sigma = sigma_max * np.power(u, 3.0)
        kappa = 1.0 + (4.0 - 1.0) * np.power(u, 3.0)
        alpha = np.where(active, 300.0 * (1.0 - u), 0.0)
        return sigma, kappa, alpha

    # Complete Yee storage keeps the wall-aligned Ez plane at x=nx and the
    # half-cell-aligned Hy samples spanning all nx material cells.
    e_count = int(_fields_for_sim(sim).Ez.shape[2])
    h_count = int(_fields_for_sim(sim).Hy.shape[2])
    _, kappa_e_x, _ = expected_profile(e_count, sample_kind="E")
    _, kappa_h_x, _ = expected_profile(h_count, sample_kind="H")

    e_term = program.boundary.cpml.e_terms[4]
    h_term = program.boundary.cpml.h_terms[3]
    assert e_term.a.shape == e_term.b.shape == e_term.inv_kappa.shape
    assert e_term.a.shape[2] == e_term.slab.low + e_term.slab.high
    assert h_term.a.shape[2] == h_term.slab.low + h_term.slab.high
    assert e_term.slab.shape[2] < _fields_for_sim(sim).Ez.shape[2]
    assert h_term.slab.shape[2] < _fields_for_sim(sim).Hy.shape[2]

    h_full_shapes = (
        _fields_for_sim(sim).Hx.shape,
        _fields_for_sim(sim).Hx.shape,
        _fields_for_sim(sim).Hy.shape,
        _fields_for_sim(sim).Hy.shape,
        _fields_for_sim(sim).Hz.shape,
        _fields_for_sim(sim).Hz.shape,
    )
    e_full_shapes = (
        _fields_for_sim(sim).Ex.shape,
        _fields_for_sim(sim).Ex.shape,
        _fields_for_sim(sim).Ey.shape,
        _fields_for_sim(sim).Ey.shape,
        _fields_for_sim(sim).Ez.shape,
        _fields_for_sim(sim).Ez.shape,
    )
    full_psi_cells = 0
    packed_psi_cells = 0
    for term, full_shape in zip(
        (*program.boundary.cpml.h_terms, *program.boundary.cpml.e_terms),
        (*h_full_shapes, *e_full_shapes),
        strict=True,
    ):
        slab_spec = term.slab
        psi_shape = slab_spec.shape
        assert isinstance(slab_spec.axis, int)
        assert isinstance(slab_spec.low, int)
        assert isinstance(slab_spec.high, int)
        assert isinstance(slab_spec.shape, tuple)
        assert slab_spec.shape == psi_shape
        assert slab_spec.low >= 0
        assert slab_spec.high >= 0
        assert slab_spec.low + slab_spec.high == psi_shape[slab_spec.axis]
        assert slab_spec.low + slab_spec.high <= full_shape[slab_spec.axis]
        for dim, (packed_size, full_size) in enumerate(
            zip(psi_shape, full_shape, strict=True)
        ):
            if dim != slab_spec.axis:
                assert packed_size == full_size
        full_psi_cells += int(np.prod(full_shape))
        packed_psi_cells += int(np.prod(psi_shape))
    assert packed_psi_cells < full_psi_cells

    np.testing.assert_allclose(
        np.asarray(e_term.inv_kappa[0, 0, :]),
        np.concatenate(
            (1.0 / kappa_e_x[: e_term.slab.low], 1.0 / kappa_e_x[-e_term.slab.high :])
        ),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(h_term.inv_kappa[0, 0, :]),
        np.concatenate(
            (1.0 / kappa_h_x[: h_term.slab.low], 1.0 / kappa_h_x[-h_term.slab.high :])
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_field_recorder_is_the_only_snapshot_source(small_sim_params):
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[
            FieldRecorder(("Ez",), interval=8, name="frames"),
            FieldRecorder(
                ("Ez", "Hx"),
                interval=8,
                name="line",
                center=(domain / 2, domain / 2, 0.0),
                size=(0.0, domain, 0.0),
            ),
        ],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    result = sim.run(progress=False)

    assert result is not None
    recorder = result.monitor("frames")
    line = result.monitor("line")
    assert result.monitors["line"].fields is line.fields
    assert recorder.snapshot("Ez", 0)["step"] == 8
    with pytest.raises(TypeError):
        recorder.snapshot("Ez", 0)["step"] = 9
    with pytest.raises(ValueError, match="read-only"):
        recorder.snapshot("Ez", 0)["field"].flat[0] = 0.0
    line_points = (
        sim.monitors[1]
        ._line_sample_coords_2d(dx, dx, _fields_for_sim(sim).permittivity.shape)[0]
        .size
    )
    assert line.fields["Ez"].shape == (line.field_steps.size, line_points)
    assert line.fields["Hx"].shape == (line.field_steps.size, line_points)


def test_compiled_static_monitor_physical_dft_uses_centered_tm_xy_sampling():
    point = jnp.asarray([[0]], dtype=jnp.int32)
    zero = jnp.asarray([[0.0]], dtype=jnp.float32)
    indices = (point,) * 6
    sample_weights = (zero,) * 6
    ez_center_indices = jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32)
    ez_center_weights = jnp.full((1, 4), 0.25, dtype=jnp.float32)
    dft_indices = (point, point, ez_center_indices, point, point, point)
    dft_weights = (zero, zero, ez_center_weights, zero, zero, zero)
    program = SimpleNamespace(
        config=RunConfig(
            resolution=1.0,
            dt=1.0,
            num_steps=1,
            plane_2d="xy",
            is_3d=False,
        ),
        monitors=(
            CompiledMonitorSpec(
                name="m_physical",
                monitor_index=0,
                record_interval=1,
                accumulate_power=False,
                power_scale=1.0,
                accumulate_frequency=True,
                freq_record_interval=1,
                freq_count=1,
                freq_hz=jnp.asarray([0.0], dtype=jnp.float32),
                freq_rot_re=jnp.asarray([1.0], dtype=jnp.float32),
                freq_rot_im=jnp.asarray([0.0], dtype=jnp.float32),
                dft_enabled=True,
                dft_record_interval=1,
                dft_t_start=0.0,
                dft_t_end=1.0,
                dft_window_code=0,
                dft_normalization_code=1,
                dft_length_unit=float(LIGHT_SPEED),
                dft_point_count=1,
                dft_component_mask=jnp.asarray([0, 0, 1, 0, 0, 0], dtype=jnp.float32),
                sample_flat_idx=indices,
                sample_weights=sample_weights,
                dft_flat_idx=dft_indices,
                dft_weights=dft_weights,
            ),
        ),
    )

    monitor_state = _monitor_state(
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

    tm_ez = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)

    updated = monitor_runtime.update_monitors(
        program,
        monitor_state,
        abs_step=jnp.asarray(0, dtype=jnp.int32),
        t_phys=jnp.asarray(0.0, dtype=jnp.float32),
        dt_scalar=jnp.asarray(1.0, dtype=jnp.float32),
        ex=jnp.zeros((1, 1), dtype=jnp.float32),
        ey=jnp.zeros((1, 1), dtype=jnp.float32),
        ez=tm_ez,
        hx=jnp.zeros((1, 2), dtype=jnp.float32),
        hy=jnp.zeros((2, 1), dtype=jnp.float32),
        hz=jnp.zeros((1, 1), dtype=jnp.float32),
    )

    expected = 2.5 / np.sqrt(2.0 * np.pi)
    np.testing.assert_allclose(
        updated.dft_vec_re[0, 2, 0, 0], expected, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        updated.dft_vec_im[0, 2, 0, 0], 0.0, rtol=1e-6, atol=1e-6
    )


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
        sources=[],
        monitors=[],
        boundaries=[PML(thickness=1.0 * wl)],
        time=t,
        resolution=dx,
    )

    program = sim.compile(num_steps=20)
    cache = execution_cache(program)
    assert cache.compiled_scan is None

    state0 = initial_program_state(program, t=float(sim.time[0]), current_step=0)
    state1 = run_program(program, state0)
    compiled_scan = cache.compiled_scan
    assert callable(compiled_scan)

    # The default execution variant preserves reusable continuation inputs.
    run_program(program, state1)
    assert cache.compiled_scan is compiled_scan


def test_compiled_jaxpr_has_no_host_callbacks(small_sim_params):
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )

    sim = Simulation(
        design=design,
        sources=[source],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    program = sim.compile(num_steps=8)
    state0 = initial_program_state(program, t=float(sim.time[0]), current_step=0)

    build_program_scan(program)
    jaxpr = jax.make_jaxpr(execution_cache(program).compiled_scan)(
        state0, program.coefficients
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
        center=(2 * wl, height / 2, 0.0),
        size=(0.0, 2.0 * wg_w, wg_w),
        source_time=SampledSignal(signal, dt=dt, freq0=freq),
        direction="+",
        mode_spec=ModeSpec(polarization="tm"),
    )

    sim = Simulation(
        design=design,
        sources=[source],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    program = sim.compile(num_steps=20)
    assert any(spec.timing == "h" for spec in program.sources)
    assert any(spec.timing == "e" for spec in program.sources)

    result = sim.advance(num_steps=20, progress=False)
    assert np.isfinite(np.asarray(result.state.ez)).all()


def test_analytic_signal_quadrature_matches_periodic_sine():
    phase = 2.0 * np.pi * 5.0 * np.arange(64, dtype=float) / 64.0
    quadrature = analytic_signal_quadrature(np.cos(phase))

    np.testing.assert_allclose(quadrature, np.sin(phase), atol=1e-12, rtol=1e-12)


def test_mode_source_uses_explicit_signal_quadrature():
    source = ModeSource(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 1.0, 1.0),
        source_time=SampledSignal(
            np.asarray([1.0, 2.0, 3.0]),
            dt=1.0,
            quadrature=np.asarray([4.0, 5.0, 6.0]),
            freq0=1.0,
        ),
        direction="+",
    )

    np.testing.assert_allclose(source.source_time.quadrature, [4.0, 5.0, 6.0])
    _signal, quadrature = source.source_time.sample(np.asarray([1.0]))
    np.testing.assert_allclose(quadrature, [5.0])


def test_analytic_subband_waveforms_reconstruct_input():
    dt = 1e-15
    t = np.arange(256, dtype=float) * dt
    analytic = 0.8 * np.exp(2j * np.pi * 120e12 * t) + 0.2 * np.exp(
        2j * np.pi * 210e12 * t
    )

    nodes, subbands = _analytic_subband_waveforms(
        analytic,
        dt=dt,
        profile_frequencies=np.asarray([100e12, 180e12, 240e12]),
    )

    assert nodes.shape == (3,)
    assert subbands.shape == (3, analytic.size)
    np.testing.assert_allclose(
        np.sum(subbands, axis=0), analytic, rtol=1e-12, atol=1e-12
    )
    phase = np.exp(-2j * np.pi * nodes[:, None] * t[None, :])
    total_response = phase @ analytic
    subband_response = phase @ subbands.T
    np.testing.assert_allclose(
        subband_response / total_response[:, None],
        np.eye(nodes.size),
        rtol=1e-12,
        atol=1e-12,
    )


def test_frequency_partition_uses_polynomial_interpolation():
    nodes = np.asarray([100e12, 150e12, 200e12], dtype=float)
    frequencies = np.asarray([110e12, 140e12, 175e12], dtype=float)

    weights = partition_weights_by_frequency(frequencies, nodes)
    scaled_nodes = nodes / 100e12
    sampled_quadratic = scaled_nodes**2 @ weights

    np.testing.assert_allclose(np.sum(weights, axis=0), 1.0, atol=1e-14)
    np.testing.assert_allclose(sampled_quadratic, (frequencies / 100e12) ** 2)


def test_cache_reuse_across_equal_chunks(small_sim_params):
    """Equal-sized chunks should reuse the same cached executable."""
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )

    sim = Simulation(
        design=design,
        sources=[source],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    # Run equal-sized continuation chunks through the same compiled program.
    chunk_size = 30
    state = None
    for _ in range(3):
        result = sim.advance(num_steps=chunk_size, state=state, progress=False)
        state = result.state

    # The program should have been compiled only once (all chunks are size 30).
    program = sim.compile(num_steps=chunk_size)
    assert callable(execution_cache(program).compiled_scan)


def test_compile_cache_invalidates_when_specs_change(small_sim_params):
    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    sim = Simulation(
        design=design,
        sources=[source],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    with_source = sim.compile(num_steps=8)
    assert with_source.sources

    no_source = sim.updated_copy(sources=()).compile(num_steps=8)

    assert no_source is not with_source
    assert no_source.sources == ()


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
        sources=[source_a],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )
    sim_chunked = Simulation(
        design=design.copy(),
        sources=[source_b],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    # Single-shot: run all 120 steps at once.
    single_result = sim_single.advance(num_steps=120, progress=False)

    # Chunked: run 4 chunks of 30 steps each.
    state = None
    for _ in range(4):
        chunked_result = sim_chunked.advance(num_steps=30, state=state, progress=False)
        state = chunked_result.state

    ez_single = np.asarray(single_result.state.ez)
    ez_chunked = np.asarray(chunked_result.state.ez)

    assert single_result.state.current_step == chunked_result.state.current_step
    assert np.allclose(ez_single, ez_chunked, rtol=1e-5, atol=1e-6)


def test_compiled_dft_component_monitor_populated(small_sim_params):
    from beamz.simulation.observe import monitor_dft_component, monitor_dft_flux

    wl, dx, _dt, domain, _steps, t, signal = small_sim_params
    design = Design(width=domain, height=domain, material=Material(permittivity=1.0))
    source = GaussianSource(
        position=(domain / 2, domain / 2), width=wl / 6, signal=signal
    )
    freq = LIGHT_SPEED / wl
    monitor = FieldMonitor(
        center=(domain * 0.35, domain * 0.5, 0.0),
        # A 2D simulation may retain a nonzero out-of-plane display span.
        size=(0.0, domain * 0.3, domain * 0.3),
        freqs=[freq],
        fields=("Ez", "Hy"),
        interval=2,
    )
    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[monitor],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )
    result = sim.advance(num_steps=60, progress=False)
    monitor_result = _monitor_result(result)
    ez_dft = np.asarray(monitor_dft_component(monitor_result, "Ez"))
    hy_dft = np.asarray(monitor_dft_component(monitor_result, "Hy"))
    assert ez_dft.shape[0] == 1
    assert hy_dft.shape == ez_dft.shape
    assert ez_dft.shape[1] > 0
    assert np.isfinite(ez_dft).all()
    assert np.isfinite(hy_dft).all()
    assert np.max(np.abs(ez_dft)) > 0.0
    assert np.max(np.abs(hy_dft)) > 0.0
    with pytest.raises(TypeError):
        monitor_result.dft_fields["new"] = np.zeros((1, 1))
    with pytest.raises(ValueError, match="read-only"):
        monitor_result.dft_fields["Ez"].flat[0] = 0.0
    np.testing.assert_allclose(
        monitor_result.power_spectrum, np.zeros((0,), dtype=np.complex64)
    )
    assert np.isfinite(monitor_dft_flux(monitor_result)).all()
    assert not hasattr(monitor, "_dft_accum")


def test_compiled_static_monitor_dft_uses_current_sample_phase():
    point = jnp.asarray([[0]], dtype=jnp.int32)
    zero = jnp.asarray([[0.0]], dtype=jnp.float32)
    one = jnp.asarray([[1.0]], dtype=jnp.float32)
    indices = (point,) * 6
    weights = (zero, zero, one, zero, zero, zero)
    program = SimpleNamespace(
        config=RunConfig(
            resolution=1.0,
            dt=1.0,
            num_steps=1,
            plane_2d="xy",
            is_3d=False,
        ),
        monitors=(
            CompiledMonitorSpec(
                name="m",
                monitor_index=0,
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
                sample_flat_idx=indices,
                sample_weights=weights,
                dft_flat_idx=indices,
                dft_weights=weights,
            ),
        ),
    )

    monitor_state = _monitor_state(
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

    updated = monitor_runtime.update_monitors(
        program,
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
    )

    np.testing.assert_allclose(
        updated.dft_vec_re[0, 2, 0, 0], 2.0, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        updated.dft_vec_im[0, 2, 0, 0], 0.0, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(updated.freq_phase_re[0, 0], 0.0, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(updated.freq_phase_im[0, 0], -1.0, rtol=1e-7, atol=1e-7)
