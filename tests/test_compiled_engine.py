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
from beamz.const import EPS_0
from beamz.devices.monitors.compiler import CompiledMonitorSpec
from beamz.devices.sources import compiler as source_compiler
from beamz.devices.sources.compiler import (
    _analytic_subband_waveforms,
    _as_slab_spec,
    _compile_mode_source_3d,
    _sample_waveform,
)
from beamz.devices.sources.mode import (
    _analytic_signal_quadrature,
    _ModeSource3DResidual,
)
from beamz.shared_kernels import (
    CPML_3D_E_DERIVATIVES,
    CPML_3D_H_DERIVATIVES,
    build_cpml_3d_primitive_terms,
)
from beamz.simulation import ops
from beamz.simulation.boundaries import (
    build_h_boundary_views_for_e_3d,
    create_metallic_boundary_masks,
    initialize_full_pec_3d_state,
)
from beamz.simulation.compiled import (
    CompiledRunConfig,
    CompiledSimulation,
    EngineState,
    MonitorState,
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


def _engine_state_for_sim(sim: Simulation) -> EngineState:
    if (not sim.is_3d) and sim.plane_2d == "xy":
        tm_ez = sim.fields.Ez
        tm_hx = sim.fields.Hx
        tm_hy = sim.fields.Hy
    else:
        tm_ez = jnp.zeros((0, 0), dtype=sim.fields.Ez.dtype)
        tm_hx = jnp.zeros((0, 0), dtype=sim.fields.Hx.dtype)
        tm_hy = jnp.zeros((0, 0), dtype=sim.fields.Hy.dtype)
    if sim.is_3d:
        try:
            fp_state = (
                sim.fields.full_pec_3d_state
                if sim.fields.full_pec_3d_state is not None
                else initialize_full_pec_3d_state(sim.fields)
            )
        except Exception:
            fp_ex = jnp.zeros((0, 0, 0), dtype=sim.fields.Ex.dtype)
            fp_ey = jnp.zeros((0, 0, 0), dtype=sim.fields.Ey.dtype)
            fp_ez = jnp.zeros((0, 0, 0), dtype=sim.fields.Ez.dtype)
            fp_hx = jnp.zeros((0, 0, 0), dtype=sim.fields.Hx.dtype)
            fp_hy = jnp.zeros((0, 0, 0), dtype=sim.fields.Hy.dtype)
            fp_hz = jnp.zeros((0, 0, 0), dtype=sim.fields.Hz.dtype)
        else:
            fp_ex = fp_state.Ex
            fp_ey = fp_state.Ey
            fp_ez = fp_state.Ez
            fp_hx = fp_state.Hx
            fp_hy = fp_state.Hy
            fp_hz = fp_state.Hz
    else:
        fp_ex = jnp.zeros((0, 0, 0), dtype=sim.fields.Ex.dtype)
        fp_ey = jnp.zeros((0, 0, 0), dtype=sim.fields.Ey.dtype)
        fp_ez = jnp.zeros((0, 0, 0), dtype=sim.fields.Ez.dtype)
        fp_hx = jnp.zeros((0, 0, 0), dtype=sim.fields.Hx.dtype)
        fp_hy = jnp.zeros((0, 0, 0), dtype=sim.fields.Hy.dtype)
        fp_hz = jnp.zeros((0, 0, 0), dtype=sim.fields.Hz.dtype)

    return EngineState(
        ex=sim.fields.Ex,
        ey=sim.fields.Ey,
        ez=sim.fields.Ez,
        hx=sim.fields.Hx,
        hy=sim.fields.Hy,
        hz=sim.fields.Hz,
        tm_ez=tm_ez,
        tm_hx=tm_hx,
        tm_hy=tm_hy,
        fp_ex=fp_ex,
        fp_ey=fp_ey,
        fp_ez=fp_ez,
        fp_hx=fp_hx,
        fp_hy=fp_hy,
        fp_hz=fp_hz,
        cpml_psi_h_terms=jnp.zeros((2, 0, 0), dtype=sim.fields.Hx.dtype),
        cpml_psi_e_terms=jnp.zeros((2, 0, 0), dtype=sim.fields.Ez.dtype),
        cpml3d_psi_h_terms=tuple(
            jnp.zeros((0, 0, 0), dtype=sim.fields.Hx.dtype) for _ in range(6)
        ),
        cpml3d_psi_e_terms=tuple(
            jnp.zeros((0, 0, 0), dtype=sim.fields.Ez.dtype) for _ in range(6)
        ),
        t=jnp.asarray(sim.t, dtype=jnp.float32),
        current_step=jnp.asarray(sim.current_step, dtype=jnp.int32),
    )


class _PlaneCurrentSheetSource3D:
    def __init__(self, signal, ix: int, iy0: int, iy1: int, iz0: int, iz1: int):
        self.signal = signal
        self.ix = int(ix)
        self.iy0 = int(iy0)
        self.iy1 = int(iy1)
        self.iz0 = int(iz0)
        self.iz1 = int(iz1)

    def compile_source_specs(
        self,
        *,
        fields,
        dt: float,
        num_steps: int,
        t0: float,
        resolution: float,
        total_steps: int | None = None,
    ):
        del resolution
        idx = (slice(self.iz0, self.iz1), slice(self.iy0, self.iy1), self.ix)
        eps_region = np.asarray(fields.eps_z[idx], dtype=np.float32)
        sig_region = np.asarray(fields.sig_z[idx], dtype=np.float32)
        denom = 1.0 + sig_region * (float(dt) / (2.0 * EPS_0 * eps_region))
        source_coeff = (float(dt) / (EPS_0 * eps_region)) / denom
        coeff = -source_coeff
        waveform = _sample_waveform(
            lambda t_sample, _dt: self.signal(float(t_sample)),
            t0=t0,
            dt=dt,
            num_steps=num_steps,
            offset_fn=lambda t, dt_: t + 0.5 * dt_,
            total_steps=total_steps,
        )
        return (
            _as_slab_spec(
                component="Ez",
                timing="e",
                index=idx,
                coeff=coeff,
                waveform=waveform,
                target_shape=tuple(fields.Ez.shape),
            ),
        )


def test_run_compiled_supports_3d_custom_current_source():
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

        def compile_source_specs(
            self,
            *,
            fields,
            dt: float,
            num_steps: int,
            t0: float,
            resolution: float,
            total_steps: int | None = None,
        ):
            del resolution
            eps_region = np.asarray(fields.eps_z[self._indices], dtype=np.float32)
            sig_region = np.asarray(fields.sig_z[self._indices], dtype=np.float32)
            denom = 1.0 + sig_region * (float(dt) / (2.0 * EPS_0 * eps_region))
            source_coeff = (float(dt) / (EPS_0 * eps_region)) / denom
            coeff = -self._voxel_weights * source_coeff
            waveform = _sample_waveform(
                lambda t_sample, _dt: self.signal(float(t_sample)),
                t0=t0,
                dt=dt,
                num_steps=num_steps,
                offset_fn=lambda t, dt_: t + 0.5 * dt_,
                total_steps=total_steps,
            )
            return (
                _as_slab_spec(
                    component="Ez",
                    timing="e",
                    index=self._indices,
                    coeff=coeff,
                    waveform=waveform,
                    target_shape=tuple(fields.Ez.shape),
                ),
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
        sources=[source_b],
        boundaries=[PEC(edges="all")],
        time=t,
        resolution=dx,
    )

    sim_compiled.run_compiled(progress=False)

    assert sim_compiled._compiled_program is not None
    assert sim_compiled._compiled_program.fp_h_decay_x.size == 0
    assert sim_compiled._compiled_program.fp_e_decay_x.size == 0
    assert sim_compiled._compiled_program.fp_e_source_x.size > 0

    assert sim_compiled.current_step == len(t)
    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        arr = np.asarray(getattr(sim_compiled.fields, component))
        assert arr.size > 0
        assert np.isfinite(arr).all()
    assert float(np.max(np.abs(np.asarray(sim_compiled.fields.Ez)))) > 0.0


def test_run_compiled_supports_2d_cpml_small_case(small_sim_params):
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

    sim.run_compiled(progress=False)

    for component in ("Ez", "Hx", "Hy"):
        arr = np.asarray(getattr(sim.fields, component))
        assert arr.size > 0
        assert np.isfinite(arr).all()


def test_run_compiled_supports_3d_cpml_small_case():
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

    sim.run_compiled(progress=False)

    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        arr = np.asarray(getattr(sim.fields, component))
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

    assert program.use_primitive_cpml_3d_terms

    cz = sim.fields.permittivity.shape[0] // 2
    cy = sim.fields.permittivity.shape[1] // 2
    cx = sim.fields.permittivity.shape[2] // 2
    cx_e = program.cpml3d_kappa_e_terms[4].shape[2] // 2
    cx_h = program.cpml3d_kappa_h_terms[3].shape[2] // 2

    assert np.asarray(sim.pml_data["kappa_x"], dtype=np.float64)[
        cz, cy, cx
    ] == pytest.approx(1.0)
    assert np.asarray(sim.pml_data["kappa_y"], dtype=np.float64)[
        cz, cy, cx
    ] == pytest.approx(1.0)
    assert np.asarray(program.cpml3d_kappa_e_terms[4], dtype=np.float64)[
        0, 0, cx_e
    ] == pytest.approx(1.0)
    assert np.asarray(program.cpml3d_kappa_h_terms[3], dtype=np.float64)[
        0, 0, cx_h
    ] == pytest.approx(1.0)


def test_cpml_3d_primitive_terms_fall_back_for_nonseparable_profiles():
    shape = (3, 4, 5)
    pml_data = {}
    for spec in (*CPML_3D_H_DERIVATIVES, *CPML_3D_E_DERIVATIVES):
        axis = {"z": 0, "y": 1, "x": 2}[spec.derivative_axis]
        profile_shape = [1, 1, 1]
        profile_shape[axis] = shape[axis]
        base = jnp.linspace(0.0, 1.0, shape[axis], dtype=jnp.float32).reshape(
            profile_shape
        )
        separable = jnp.broadcast_to(base, shape)
        pml_data[f"cpml3d_{spec.name}_sigma"] = separable
        pml_data[f"cpml3d_{spec.name}_kappa"] = 1.0 + separable
        pml_data[f"cpml3d_{spec.name}_alpha"] = 0.1 * separable

    key = f"cpml3d_{CPML_3D_E_DERIVATIVES[0].name}_sigma"
    nonseparable = pml_data[key].at[1, 1, 1].add(0.25)
    pml_data[key] = nonseparable

    assert build_cpml_3d_primitive_terms(pml_data) is None


def test_compiled_3d_metallic_edge_zeroing_matches_masks():
    fields = SimpleNamespace(
        Ex=jnp.ones((3, 4, 5), dtype=jnp.float32),
        Ey=jnp.ones((3, 4, 5), dtype=jnp.float32),
        Ez=jnp.ones((3, 4, 5), dtype=jnp.float32),
        Hx=jnp.ones((3, 4, 5), dtype=jnp.float32),
        Hy=jnp.ones((3, 4, 5), dtype=jnp.float32),
        Hz=jnp.ones((3, 4, 5), dtype=jnp.float32),
    )
    edges = frozenset({"front", "bottom", "left"})
    masks = create_metallic_boundary_masks(
        fields,
        [PEC(edges=list(edges))],
        is_3d=True,
    )

    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        field = getattr(fields, component)
        expected = jnp.where(masks[component], 0.0, field)
        actual = CompiledSimulation._apply_metal_edges_3d(field, component, edges)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


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

    assert program.use_cpml_3d
    assert program.e_decay_x.shape == (0, 0, 0)
    assert program.e_source_x.shape == (0, 0, 0)
    assert program.h_decay_x.shape == (0, 0, 0)
    assert program.h_source_x.shape == (0, 0, 0)
    assert program.e_source_lossless_x.shape == (0, 0, 0)
    assert program.e_source_lossless_y.shape == (0, 0, 0)
    assert program.e_source_lossless_z.shape == (0, 0, 0)
    assert program.e_conductivity_x is sim.fields.sig_x
    assert program.e_conductivity_y is sim.fields.sig_y
    assert program.e_conductivity_z is sim.fields.sig_z
    np.testing.assert_allclose(
        np.asarray(program.e_inv_permittivity_x),
        np.asarray(1.0 / sim.fields.eps_x),
    )
    np.testing.assert_allclose(
        np.asarray(program.e_inv_permittivity_y),
        np.asarray(1.0 / sim.fields.eps_y),
    )
    np.testing.assert_allclose(
        np.asarray(program.e_inv_permittivity_z),
        np.asarray(1.0 / sim.fields.eps_z),
    )
    assert program.ex_metal_mask.shape == (0, 0, 0)
    assert program.hx_metal_mask.shape == (0, 0, 0)
    assert program.field_shape_ex == tuple(sim.fields.Ex.shape)
    assert program.field_shape_hx == tuple(sim.fields.Hx.shape)
    assert program.h_sigma_m_x is sim.fields.sigma_m_hx
    assert program.h_sigma_m_y is sim.fields.sigma_m_hy
    assert program.h_sigma_m_z is sim.fields.sigma_m_hz
    assert program.h_source_lossless_x.shape == (0, 0, 0)
    assert program.h_source_lossless_y.shape == (0, 0, 0)
    assert program.h_source_lossless_z.shape == (0, 0, 0)
    sim.run_compiled(num_steps=1, progress=False)


def test_compiled_3d_snapshot_shape_uses_field_reference():
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

    program = sim.compile(num_steps=2, snapshot_field="Ez", snapshot_interval=1)
    snapshot_state = program._empty_snapshot_state()

    assert program.e_source_lossless_z.shape == (0, 0, 0)
    assert program.e_source_z.shape == sim.fields.Ez.shape
    assert program._snapshot_field_shape() == tuple(sim.fields.Ez.shape)
    assert snapshot_state is not None
    assert snapshot_state[0].shape == (2, *sim.fields.Ez.shape)


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
    assert not program.use_cpml_3d
    assert program.e_decay_x.shape == (0, 0, 0)
    assert program.e_source_x.shape == (0, 0, 0)
    assert program.h_decay_x.shape == (0, 0, 0)
    assert program.h_source_x.shape == (0, 0, 0)
    assert program.e_source_lossless_x.shape == (0, 0, 0)
    assert program.h_source_lossless_x.shape == (0, 0, 0)
    assert program.e_conductivity_x is sim.fields.sig_x
    assert program.h_sigma_m_x is sim.fields.sigma_m_hx
    np.testing.assert_allclose(
        np.asarray(program.e_inv_permittivity_x),
        np.asarray(1.0 / sim.fields.eps_x),
    )

    sim.run_compiled(num_steps=1, progress=False)


def test_material_3d_permittivity_e_update_matches_dense_source_grid():
    key = jax.random.PRNGKey(7)
    keys = jax.random.split(key, 9)
    hx = jax.random.normal(keys[0], (3, 4, 6), dtype=jnp.float32)
    hy = jax.random.normal(keys[1], (3, 5, 5), dtype=jnp.float32)
    hz = jax.random.normal(keys[2], (4, 4, 5), dtype=jnp.float32)
    ex = jax.random.normal(keys[3], (4, 5, 5), dtype=jnp.float32)
    ey = jax.random.normal(keys[4], (4, 4, 6), dtype=jnp.float32)
    ez = jax.random.normal(keys[5], (3, 5, 6), dtype=jnp.float32)
    eps_x = 1.0 + jnp.abs(jax.random.normal(keys[6], ex.shape, dtype=jnp.float32))
    eps_y = 1.0 + jnp.abs(jax.random.normal(keys[7], ey.shape, dtype=jnp.float32))
    eps_z = 1.0 + jnp.abs(jax.random.normal(keys[8], ez.shape, dtype=jnp.float32))
    inv_eps_x = 1.0 / eps_x
    inv_eps_y = 1.0 / eps_y
    inv_eps_z = 1.0 / eps_z
    dt = jnp.asarray(1.0e-17, dtype=jnp.float32)
    resolution = jnp.asarray(2.5e-8, dtype=jnp.float32)
    views = build_h_boundary_views_for_e_3d(hx, hy, hz, None)

    dense = ops.fused_update_e_lossless_3d(
        hx,
        hy,
        hz,
        ex,
        ey,
        ez,
        dt / (jnp.asarray(EPS_0, dtype=jnp.float32) * eps_x),
        dt / (jnp.asarray(EPS_0, dtype=jnp.float32) * eps_y),
        dt / (jnp.asarray(EPS_0, dtype=jnp.float32) * eps_z),
        resolution,
        boundary_views=views,
    )
    material = ops.fused_update_e_lossless_3d_inv_permittivity(
        hx,
        hy,
        hz,
        ex,
        ey,
        ez,
        inv_eps_x,
        inv_eps_y,
        inv_eps_z,
        dt,
        resolution,
        boundary_views=views,
    )

    for dense_component, material_component in zip(dense, material, strict=True):
        np.testing.assert_allclose(
            np.asarray(material_component),
            np.asarray(dense_component),
            rtol=2e-6,
            atol=1e-6,
        )


def test_lossless_h_update_incremental_matches_curl_formula():
    key = jax.random.PRNGKey(11)
    keys = jax.random.split(key, 6)
    ex = jax.random.normal(keys[0], (4, 5, 5), dtype=jnp.float32)
    ey = jax.random.normal(keys[1], (4, 4, 6), dtype=jnp.float32)
    ez = jax.random.normal(keys[2], (3, 5, 6), dtype=jnp.float32)
    hx = jax.random.normal(keys[3], (3, 4, 6), dtype=jnp.float32)
    hy = jax.random.normal(keys[4], (3, 5, 5), dtype=jnp.float32)
    hz = jax.random.normal(keys[5], (4, 4, 5), dtype=jnp.float32)
    h_src = jnp.asarray(3.0e-12, dtype=jnp.float32)
    resolution = jnp.asarray(2.5e-8, dtype=jnp.float32)
    inv_res = 1.0 / resolution

    expected_hx = (
        hx
        - h_src
        * ((ez[:, 1:, :] - ez[:, :-1, :]) - (ey[1:, :, :] - ey[:-1, :, :]))
        * inv_res
    )
    expected_hy = (
        hy
        - h_src
        * ((ex[1:, :, :] - ex[:-1, :, :]) - (ez[:, :, 1:] - ez[:, :, :-1]))
        * inv_res
    )
    expected_hz = (
        hz
        - h_src
        * ((ey[:, :, 1:] - ey[:, :, :-1]) - (ex[:, 1:, :] - ex[:, :-1, :]))
        * inv_res
    )

    actual = ops.fused_update_h_lossless_3d(
        ex, ey, ez, hx, hy, hz, h_src, h_src, h_src, resolution
    )
    for actual_component, expected_component in zip(
        actual, (expected_hx, expected_hy, expected_hz), strict=True
    ):
        np.testing.assert_allclose(
            np.asarray(actual_component),
            np.asarray(expected_component),
            rtol=2e-6,
            atol=1e-6,
        )


def test_primitive_material_3d_lossy_updates_match_dense_coefficients():
    key = jax.random.PRNGKey(13)
    keys = jax.random.split(key, 15)
    hx = jax.random.normal(keys[0], (3, 4, 6), dtype=jnp.float32)
    hy = jax.random.normal(keys[1], (3, 5, 5), dtype=jnp.float32)
    hz = jax.random.normal(keys[2], (4, 4, 5), dtype=jnp.float32)
    ex = jax.random.normal(keys[3], (4, 5, 5), dtype=jnp.float32)
    ey = jax.random.normal(keys[4], (4, 4, 6), dtype=jnp.float32)
    ez = jax.random.normal(keys[5], (3, 5, 6), dtype=jnp.float32)
    eps_x = 1.0 + jnp.abs(jax.random.normal(keys[6], ex.shape, dtype=jnp.float32))
    eps_y = 1.0 + jnp.abs(jax.random.normal(keys[7], ey.shape, dtype=jnp.float32))
    eps_z = 1.0 + jnp.abs(jax.random.normal(keys[8], ez.shape, dtype=jnp.float32))
    inv_eps_x = 1.0 / eps_x
    inv_eps_y = 1.0 / eps_y
    inv_eps_z = 1.0 / eps_z
    sig_x = jnp.abs(jax.random.normal(keys[9], ex.shape, dtype=jnp.float32)) * 0.1
    sig_y = jnp.abs(jax.random.normal(keys[10], ey.shape, dtype=jnp.float32)) * 0.1
    sig_z = jnp.abs(jax.random.normal(keys[11], ez.shape, dtype=jnp.float32)) * 0.1
    sigma_m_x = jnp.abs(jax.random.normal(keys[12], hx.shape, dtype=jnp.float32)) * 0.1
    sigma_m_y = jnp.abs(jax.random.normal(keys[13], hy.shape, dtype=jnp.float32)) * 0.1
    sigma_m_z = jnp.abs(jax.random.normal(keys[14], hz.shape, dtype=jnp.float32)) * 0.1
    dt = jnp.asarray(1.0e-17, dtype=jnp.float32)
    resolution = jnp.asarray(2.5e-8, dtype=jnp.float32)
    views = build_h_boundary_views_for_e_3d(hx, hy, hz, None)

    (h_decay_x, h_src_x, _), (h_decay_y, h_src_y, _), (h_decay_z, h_src_z, _) = (
        ops.precompute_h_update_coefficients(sigma_m_x, dt),
        ops.precompute_h_update_coefficients(sigma_m_y, dt),
        ops.precompute_h_update_coefficients(sigma_m_z, dt),
    )
    dense_h = ops.fused_update_h_lossy_3d(
        ex,
        ey,
        ez,
        hx,
        hy,
        hz,
        h_decay_x,
        h_src_x,
        h_decay_y,
        h_src_y,
        h_decay_z,
        h_src_z,
        resolution,
    )
    primitive_h = ops.fused_update_h_lossy_3d_material(
        ex,
        ey,
        ez,
        hx,
        hy,
        hz,
        sigma_m_x,
        sigma_m_y,
        sigma_m_z,
        dt,
        resolution,
    )

    (e_decay_x, e_src_x, _), (e_decay_y, e_src_y, _), (e_decay_z, e_src_z, _) = (
        ops.precompute_e_update_coefficients(
            ex.shape, sig_x, eps_x, dt, (slice(None),) * 3
        ),
        ops.precompute_e_update_coefficients(
            ey.shape, sig_y, eps_y, dt, (slice(None),) * 3
        ),
        ops.precompute_e_update_coefficients(
            ez.shape, sig_z, eps_z, dt, (slice(None),) * 3
        ),
    )
    dense_e = ops.fused_update_e_lossy_3d(
        hx,
        hy,
        hz,
        ex,
        ey,
        ez,
        e_decay_x,
        e_src_x,
        e_decay_y,
        e_src_y,
        e_decay_z,
        e_src_z,
        resolution,
        boundary_views=views,
    )
    primitive_e = ops.fused_update_e_lossy_3d_material(
        hx,
        hy,
        hz,
        ex,
        ey,
        ez,
        sig_x,
        inv_eps_x,
        sig_y,
        inv_eps_y,
        sig_z,
        inv_eps_z,
        dt,
        resolution,
        boundary_views=views,
    )

    for primitive_component, dense_component in zip(
        (*primitive_h, *primitive_e), (*dense_h, *dense_e), strict=True
    ):
        np.testing.assert_allclose(
            np.asarray(primitive_component),
            np.asarray(dense_component),
            rtol=2e-6,
            atol=1e-6,
        )


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
        int(np.asarray(getattr(sim.fields, name)).nbytes)
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    )
    assert report["totals_by_category"]["yee_fields"] == field_bytes
    assert report["compiled"]["totals_by_category"]["compiled_update_coefficients"] > 0
    compiled_names = {entry["name"] for entry in report["compiled"]["entries"]}
    referenced_names = {
        entry["name"] for entry in report["compiled"]["referenced_inputs"]["entries"]
    }
    assert not any(
        key.startswith("use_") and key.endswith("_3d_e_coefficients")
        for key in report["compiled"]["config"]
    )
    assert "e_inv_permittivity_x" in compiled_names
    assert "e_inv_permittivity_x" not in referenced_names
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

    assert program.e_decay_x.shape == (0, 0, 0)
    assert program.e_source_x.shape == (0, 0, 0)
    assert program.h_decay_x.shape == (0, 0, 0)
    assert program.h_source_x.shape == (0, 0, 0)
    assert program.e_conductivity_x is sim.fields.sig_x
    np.testing.assert_allclose(
        np.asarray(program.e_inv_permittivity_x),
        np.asarray(1.0 / sim.fields.eps_x),
    )
    assert program.h_sigma_m_x is sim.fields.sigma_m_hx

    sim.run_compiled(num_steps=1, progress=False)


def test_compiled_3d_cpml_profiles_match_expected_x_boundary_embedding(monkeypatch):
    monkeypatch.setenv("BEAMZ_CPML_PACKED_PSI", "1")

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

    nx = int(sim.fields.permittivity.shape[2])
    pml_cells = int(round(thickness / dx))

    def expected_profile(count: int, *, sample_kind: str, domain_cells: int):
        sigma = np.zeros((count,), dtype=np.float32)
        kappa = np.ones((count,), dtype=np.float32)
        alpha = np.zeros((count,), dtype=np.float32)
        sigma_max = float(sim.boundaries[0].sigma_max)

        if sample_kind == "E":
            low_d = np.arange(pml_cells - 1, -1, -1, dtype=np.float32)[
                : min(count, pml_cells)
            ]
            high_d = np.insert(
                np.arange(0.5, pml_cells - 0.5, 1.0, dtype=np.float32),
                0,
                0.0,
            )[: min(count, pml_cells)]
        else:
            low_d = np.append(
                np.arange(pml_cells - 1.5, -0.5, -1.0, dtype=np.float32),
                0.0,
            )[: min(count, pml_cells)]

        u = np.clip(low_d / max(float(pml_cells), 1e-30), 0.0, 1.0)
        side_sigma = sigma_max * np.power(u, 3.0)
        side_kappa = 1.0 + (4.0 - 1.0) * np.power(u, 3.0)
        side_alpha = 300.0 * np.power(1.0 - u, 1.0)
        sigma[: len(low_d)] = np.maximum(sigma[: len(low_d)], side_sigma)
        kappa[: len(low_d)] = np.maximum(kappa[: len(low_d)], side_kappa)
        alpha[: len(low_d)] = np.maximum(alpha[: len(low_d)], side_alpha)

        if sample_kind == "E":
            offset = 0.0
        else:
            offset = 0.5
        coords = np.arange(count, dtype=np.float32) + np.float32(offset)
        d = np.clip(coords - (float(domain_cells) - float(pml_cells)), 0.0, pml_cells)
        mask = d > 0.0
        u = np.clip(d / max(float(pml_cells), 1e-30), 0.0, 1.0)
        side_sigma = sigma_max * np.power(u, 3.0)
        side_kappa = 1.0 + (4.0 - 1.0) * np.power(u, 3.0)
        side_alpha = 300.0 * np.power(1.0 - u, 1.0)
        sigma = np.where(mask, np.maximum(sigma, side_sigma), sigma)
        kappa = np.where(mask, np.maximum(kappa, side_kappa), kappa)
        alpha = np.where(mask, np.maximum(alpha, side_alpha), alpha)
        return sigma, kappa, alpha

    sigma_e_x, kappa_e_x, alpha_e_x = expected_profile(
        nx, sample_kind="E", domain_cells=nx
    )
    sigma_h_x, kappa_h_x, alpha_h_x = expected_profile(
        max(nx - 1, 0), sample_kind="H", domain_cells=nx
    )

    assert program.use_primitive_cpml_3d_terms
    assert program.use_cpml_3d_packed_psi
    assert program.cpml3d_a_e_terms[4].shape == (1, 1, nx)
    assert program.cpml3d_b_e_terms[4].shape == (1, 1, nx)
    assert program.cpml3d_inv_kappa_e_terms[4].shape == (1, 1, nx)
    assert program.cpml3d_sigma_e_terms[4].shape == (1, 1, nx)
    assert program.cpml3d_sigma_h_terms[3].shape == (1, 1, max(nx - 1, 0))
    assert program.cpml3d_e_psi_shapes[4] == program.cpml3d_e_slab_specs[4].shape
    assert program.cpml3d_h_psi_shapes[3] == program.cpml3d_h_slab_specs[3].shape
    assert program.cpml3d_e_psi_shapes[4][2] < sim.fields.Ez.shape[2]
    assert program.cpml3d_h_psi_shapes[3][2] < sim.fields.Hy.shape[2]

    h_full_shapes = (
        sim.fields.Hx.shape,
        sim.fields.Hx.shape,
        sim.fields.Hy.shape,
        sim.fields.Hy.shape,
        sim.fields.Hz.shape,
        sim.fields.Hz.shape,
    )
    e_full_shapes = (
        sim.fields.Ex.shape,
        sim.fields.Ex.shape,
        sim.fields.Ey.shape,
        sim.fields.Ey.shape,
        sim.fields.Ez.shape,
        sim.fields.Ez.shape,
    )
    full_psi_cells = 0
    packed_psi_cells = 0
    for slab_spec, psi_shape, full_shape in zip(
        (*program.cpml3d_h_slab_specs, *program.cpml3d_e_slab_specs),
        (*program.cpml3d_h_psi_shapes, *program.cpml3d_e_psi_shapes),
        (*h_full_shapes, *e_full_shapes),
        strict=True,
    ):
        assert isinstance(slab_spec.axis, int)
        assert isinstance(slab_spec.low, int)
        assert isinstance(slab_spec.high, int)
        assert isinstance(slab_spec.shape, tuple)
        assert slab_spec.shape == psi_shape
        assert slab_spec.low >= 0
        assert slab_spec.high >= 0
        assert slab_spec.low + slab_spec.high == psi_shape[slab_spec.axis]
        assert slab_spec.low + slab_spec.high <= full_shape[slab_spec.axis]
        for dim, (packed_size, full_size) in enumerate(zip(psi_shape, full_shape)):
            if dim != slab_spec.axis:
                assert packed_size == full_size
        full_psi_cells += int(np.prod(full_shape))
        packed_psi_cells += int(np.prod(psi_shape))
    assert packed_psi_cells < full_psi_cells

    np.testing.assert_allclose(
        np.asarray(program.cpml3d_sigma_e_terms[4][0, 0, :]),
        sigma_e_x,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(program.cpml3d_kappa_e_terms[4][0, 0, :]),
        kappa_e_x,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(program.cpml3d_alpha_e_terms[4][0, 0, :]),
        alpha_e_x,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(program.cpml3d_inv_kappa_e_terms[4][0, 0, :]),
        1.0 / kappa_e_x,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(program.cpml3d_sigma_h_terms[3][0, 0, :]),
        sigma_h_x,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(program.cpml3d_kappa_h_terms[3][0, 0, :]),
        kappa_h_x,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(program.cpml3d_alpha_h_terms[3][0, 0, :]),
        alpha_h_x,
        rtol=1e-6,
        atol=1e-6,
    )


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
        sources=[source],
        monitors=[monitor],
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
        sources=[source_a],
        monitors=[monitor_a],
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
        sources=[source_b],
        monitors=[monitor_b],
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


def test_run_snapshot_path_does_not_fall_back_to_python_step(small_sim_params):
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

    def _unexpected_step():
        raise AssertionError("snapshot runs should stay on the compiled engine path")

    sim.step = _unexpected_step
    result = sim.run(snapshot_field="Ez", snapshot_interval=8, progress=False)

    assert result is not None
    assert len(result["snapshots"]) > 0
    assert result["snapshots"][0]["step"] == 8


def test_compiled_snapshot_state_warns_for_large_preallocation(
    small_sim_params, monkeypatch
):
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
    program = sim.compile(num_steps=2, snapshot_field="Ez", snapshot_interval=1)

    monkeypatch.setenv("BEAMZ_SNAPSHOT_WARN_GIB", "1e-8")
    with pytest.warns(RuntimeWarning, match="Compiled field snapshots will allocate"):
        snapshot_state = program._empty_snapshot_state()

    assert snapshot_state is not None
    assert snapshot_state[0].shape[0] == 2


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
        power_spectrum_frequencies=[freq],
        power_spectrum_record_interval=1,
    )

    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[monitor],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )
    sim.run_compiled(num_steps=60, progress=False)

    assert monitor.power_spectrum.shape == (1,)
    assert np.isfinite(monitor.power_spectrum).all()

    power = np.asarray(monitor.power_history, dtype=np.float64)
    ts = np.asarray(monitor.power_timestamps, dtype=np.float64)
    direct = np.sum(power * np.exp(-1j * 2.0 * np.pi * freq * ts)) * dt
    assert np.allclose(
        monitor.power_spectrum[0],
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
        power_spectrum_frequencies=freqs,
        power_spectrum_record_interval=1,
    )
    sim_full = Simulation(
        design=design.copy(),
        sources=[source_a],
        monitors=[monitor_a],
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
        power_spectrum_frequencies=freqs,
        power_spectrum_record_interval=1,
    )
    sim_chunked = Simulation(
        design=design.copy(),
        sources=[source_b],
        monitors=[monitor_b],
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

    s_full = np.asarray(monitor_a.power_spectrum)
    s_chunked = np.asarray(monitor_b.power_spectrum)
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
        power_spectrum_frequencies=[freq],
        power_spectrum_record_interval=1,
        record_fields=False,
    )
    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[monitor],
        boundaries=[PML(thickness=0.6 * wl, edges="all")],
        time=t,
        resolution=dx,
    )
    sim.run_compiled(num_steps=12, progress=False)

    spec = np.asarray(monitor.power_spectrum)
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
        sources=[source],
        monitors=[monitor],
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
    np.testing.assert_allclose(
        monitor.power_spectrum, np.zeros((0,), dtype=np.complex64)
    )
    assert np.isfinite(monitor.get_dft_flux()).all()


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


def test_compiled_static_monitor_physical_dft_uses_centered_tm_xy_sampling():
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
            name="m_physical",
            monitor_index=0,
            is_3d=False,
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
            dft_centered_tm_xy_sampling=True,
            dft_point_count=1,
            dft_component_mask=jnp.asarray([0, 0, 1, 0, 0, 0], dtype=jnp.float32),
            dft_target_x=jnp.asarray([0.5], dtype=jnp.float32),
            dft_target_y=jnp.asarray([0.5], dtype=jnp.float32),
            x_ex=jnp.asarray([0], dtype=jnp.int32),
            y_ex=jnp.asarray([0], dtype=jnp.int32),
            valid_ex=jnp.asarray([0.0], dtype=jnp.float32),
            x_ey=jnp.asarray([0], dtype=jnp.int32),
            y_ey=jnp.asarray([0], dtype=jnp.int32),
            valid_ey=jnp.asarray([0.0], dtype=jnp.float32),
            x_ez=jnp.asarray([0], dtype=jnp.int32),
            y_ez=jnp.asarray([0], dtype=jnp.int32),
            valid_ez=jnp.asarray([0.0], dtype=jnp.float32),
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

    tm_ez = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)

    updated = program._update_monitors(
        monitor_state,
        abs_step=jnp.asarray(0, dtype=jnp.int32),
        t_phys=jnp.asarray(0.0, dtype=jnp.float32),
        dt_scalar=jnp.asarray(1.0, dtype=jnp.float32),
        ex=jnp.zeros((1, 1), dtype=jnp.float32),
        ey=jnp.zeros((1, 1), dtype=jnp.float32),
        ez=jnp.zeros((1, 1), dtype=jnp.float32),
        hx=jnp.zeros((1, 1), dtype=jnp.float32),
        hy=jnp.zeros((1, 1), dtype=jnp.float32),
        hz=jnp.zeros((1, 1), dtype=jnp.float32),
        tm_ez=tm_ez,
        tm_hx=jnp.zeros((1, 2), dtype=jnp.float32),
        tm_hy=jnp.zeros((2, 1), dtype=jnp.float32),
        monitors_2d=program.monitor_specs,
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
    assert program.compile_count == 0

    eng0 = _engine_state_for_sim(sim)
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

    eng1, _, _, _ = program.run(eng0, mon0)
    assert program.compile_count == 1

    # Recreate states since donation invalidates buffers.
    eng1_input = EngineState(
        ex=eng1.ex,
        ey=eng1.ey,
        ez=eng1.ez,
        hx=eng1.hx,
        hy=eng1.hy,
        hz=eng1.hz,
        tm_ez=eng1.tm_ez,
        tm_hx=eng1.tm_hx,
        tm_hy=eng1.tm_hy,
        fp_ex=eng1.fp_ex,
        fp_ey=eng1.fp_ey,
        fp_ez=eng1.fp_ez,
        fp_hx=eng1.fp_hx,
        fp_hy=eng1.fp_hy,
        fp_hz=eng1.fp_hz,
        cpml_psi_h_terms=eng1.cpml_psi_h_terms,
        cpml_psi_e_terms=eng1.cpml_psi_e_terms,
        cpml3d_psi_h_terms=eng1.cpml3d_psi_h_terms,
        cpml3d_psi_e_terms=eng1.cpml3d_psi_e_terms,
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
        sources=[source],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    program = sim.compile(num_steps=8)
    eng0 = _engine_state_for_sim(sim)
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
        sources=[source],
        boundaries=[PML(thickness=1.2 * wl)],
        time=t,
        resolution=dx,
    )

    program = sim.compile(num_steps=20)
    assert any(spec.timing == "h" for spec in program.source_specs)
    assert any(spec.timing == "e" for spec in program.source_specs)

    sim.run_compiled(num_steps=20, progress=False)
    assert np.isfinite(np.asarray(sim.fields.Ez)).all()


def test_analytic_signal_quadrature_matches_periodic_sine():
    phase = 2.0 * np.pi * 5.0 * np.arange(64, dtype=float) / 64.0
    quadrature = _analytic_signal_quadrature(np.cos(phase))

    np.testing.assert_allclose(quadrature, np.sin(phase), atol=1e-12, rtol=1e-12)


def test_mode_source_uses_explicit_signal_quadrature():
    source = ModeSource.__new__(ModeSource)
    source.signal = np.asarray([1.0, 2.0, 3.0], dtype=float)
    source.signal_quadrature = np.asarray([4.0, 5.0, 6.0], dtype=float)
    source._signal_quadrature = None
    source._signal_quadrature_signature = None

    np.testing.assert_allclose(source._get_signal_quadrature(), [4.0, 5.0, 6.0])
    assert source._get_signal_quadrature_value(1.0, 1.0) == 5.0


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


def test_compile_3d_multifrequency_mode_source_uses_temporary_profile_sources(
    monkeypatch,
):
    dt = 1e-15
    source = ModeSource(
        grid=SimpleNamespace(),
        center=(0.0, 0.0, 0.0),
        width=1.0,
        height=1.0,
        wavelength=LIGHT_SPEED / 200e12,
        pol="te",
        signal=np.ones(64, dtype=float),
        direction="+x",
        profile_frequencies=np.asarray([230e12, 170e12, 210e12]),
    )
    fields = SimpleNamespace(permittivity=jnp.ones((3, 3, 3), dtype=jnp.float32))
    seen: list[tuple[bool, float, object]] = []

    def fake_initialize(self, permittivity, resolution, dt=None):
        del dt
        self._initialized = True
        self._is_3d = True
        self._grid_shape = tuple(np.asarray(permittivity).shape)
        self._resolution = float(resolution)

    def fake_compile(profile_src, *_args, **_kwargs):
        seen.append(
            (
                profile_src is source,
                float(LIGHT_SPEED / profile_src.wavelength),
                profile_src.profile_frequencies,
            )
        )
        return ()

    monkeypatch.setattr(ModeSource, "initialize", fake_initialize)
    monkeypatch.setattr(source_compiler, "_compile_mode_source_3d", fake_compile)

    specs = source_compiler._compile_mode_source(
        source,
        fields,
        dt=dt,
        num_steps=16,
        t0=0.0,
        resolution=1.0,
        total_steps=64,
    )

    assert specs == ()
    assert [item[0] for item in seen] == [False, False, False]
    np.testing.assert_allclose(
        [item[1] for item in seen],
        [170e12, 210e12, 230e12],
        rtol=1e-15,
        atol=0.0,
    )
    assert [item[2] for item in seen] == [None, None, None]


def test_compile_3d_mode_source_derives_chebyshev_profile_frequencies(monkeypatch):
    dt = 1e-15
    freq0 = 200e12
    fwidth = 20e12
    source_time = SimpleNamespace(freq0=freq0, fwidth=fwidth)
    source = ModeSource(
        grid=SimpleNamespace(),
        center=(0.0, 0.0, 0.0),
        width=1.0,
        height=1.0,
        wavelength=LIGHT_SPEED / freq0,
        pol="te",
        signal=np.ones(64, dtype=float),
        direction="+x",
        source_time=source_time,
        num_freqs=3,
    )
    fields = SimpleNamespace(permittivity=jnp.ones((3, 3, 3), dtype=jnp.float32))
    seen_freqs: list[float] = []

    def fake_initialize(self, permittivity, resolution, dt=None):
        del dt
        self._initialized = True
        self._is_3d = True
        self._grid_shape = tuple(np.asarray(permittivity).shape)
        self._resolution = float(resolution)

    def fake_compile(profile_src, *_args, **_kwargs):
        seen_freqs.append(float(LIGHT_SPEED / profile_src.wavelength))
        return ()

    monkeypatch.setattr(ModeSource, "initialize", fake_initialize)
    monkeypatch.setattr(source_compiler, "_compile_mode_source_3d", fake_compile)

    source_compiler._compile_mode_source(
        source,
        fields,
        dt=dt,
        num_steps=16,
        t0=0.0,
        resolution=1.0,
        total_steps=64,
    )

    k = np.arange(3, dtype=float)
    expected = np.sort(freq0 + 1.5 * fwidth * np.cos((2.0 * k + 1.0) * np.pi / 6.0))
    np.testing.assert_allclose(seen_freqs, expected, rtol=1e-15, atol=0.0)


def test_compile_3d_broadband_mode_source_defers_profile_solves(monkeypatch):
    dt = 1e-15
    freq0 = 200e12
    fwidth = 20e12
    source_time = SimpleNamespace(freq0=freq0, fwidth=fwidth)
    eps_full = np.ones((6, 7), dtype=float)
    crop_slices = (slice(1, 4), slice(2, 6))
    source = ModeSource(
        grid=SimpleNamespace(),
        center=(0.0, 0.0, 0.0),
        width=1.0,
        height=1.0,
        wavelength=LIGHT_SPEED / freq0,
        pol="te",
        signal=np.ones(64, dtype=float),
        direction="+x",
        source_time=source_time,
        num_freqs=3,
        mode_eps_profile_full=eps_full,
        mode_crop_slices=crop_slices,
        mode_index=1,
        mode_target_neff=2.3,
        mode_num_modes=2,
    )
    fields = SimpleNamespace(permittivity=jnp.ones((3, 6, 7), dtype=jnp.float32))
    seen_profile_sources = []

    def fail_solve_modes(**_kwargs):
        raise AssertionError("broadband compilation should defer profile solves.")

    def fake_initialize(self, permittivity, resolution, dt=None):
        del dt
        self._initialized = True
        self._is_3d = True
        self._grid_shape = tuple(np.asarray(permittivity).shape)
        self._resolution = float(resolution)

    def fake_compile(profile_src, *_args, **_kwargs):
        seen_profile_sources.append(profile_src)
        return ()

    monkeypatch.setattr(ModeSource, "initialize", fake_initialize)
    monkeypatch.setattr(source_compiler, "_compile_mode_source_3d", fake_compile)
    monkeypatch.setattr(source_compiler, "solve_modes", fail_solve_modes, raising=False)

    source_compiler._compile_mode_source(
        source,
        fields,
        dt=dt,
        num_steps=16,
        t0=0.0,
        resolution=1.0,
        total_steps=64,
    )

    assert len(seen_profile_sources) == 3
    for profile_src in seen_profile_sources:
        assert profile_src.mode_neff is None
        assert profile_src.mode_e_field is None
        assert profile_src.mode_h_field is None
        assert profile_src.mode_eps_profile_full is eps_full
        assert profile_src.mode_crop_slices == crop_slices
        assert profile_src.mode_index == 1
        assert profile_src.mode_target_neff == 2.3
        assert profile_src.mode_num_modes == 2


def test_compile_3d_mode_source_reinitializes_missing_launch_dt(monkeypatch):
    dt = 1e-15
    source = ModeSource(
        grid=SimpleNamespace(),
        center=(0.0, 0.0, 0.0),
        width=1.0,
        height=1.0,
        wavelength=LIGHT_SPEED / 200e12,
        pol="te",
        signal=np.ones(64, dtype=float),
        direction="+x",
    )
    source._initialized = True
    source._is_3d = True
    source._grid_shape = (3, 3, 3)
    source._resolution = 1.0
    source._launch_dt = None
    source._k_num_axis = None
    source._omega_launch = 2.0 * np.pi * LIGHT_SPEED / source.wavelength
    fields = SimpleNamespace(permittivity=jnp.ones((3, 3, 3), dtype=jnp.float32))
    seen_dt: list[float | None] = []

    def fake_initialize(self, permittivity, resolution, dt=None):
        seen_dt.append(dt)
        self._initialized = True
        self._is_3d = True
        self._grid_shape = tuple(np.asarray(permittivity).shape)
        self._resolution = float(resolution)
        self._launch_dt = None if dt is None else float(dt)
        self._k_num_axis = 1.0
        self._omega_launch = 2.0 * np.pi * LIGHT_SPEED / self.wavelength

    def fake_compile(profile_src, *_args, **_kwargs):
        assert profile_src is source
        assert np.isclose(profile_src._launch_dt, dt)
        assert profile_src._k_num_axis is not None
        return ()

    monkeypatch.setattr(ModeSource, "initialize", fake_initialize)
    monkeypatch.setattr(source_compiler, "_compile_mode_source_3d", fake_compile)

    specs = source_compiler._compile_mode_source(
        source,
        fields,
        dt=dt,
        num_steps=16,
        t0=0.0,
        resolution=1.0,
        total_steps=64,
    )

    assert specs == ()
    assert seen_dt == [dt]


def test_compile_3d_mode_source_uses_compact_phasor_residual_slabs():
    fields = SimpleNamespace(
        permittivity=jnp.ones((2, 2, 2)),
        permeability=jnp.ones((2, 2, 2)),
        Ex=jnp.zeros((2, 2, 1)),
        Ey=jnp.zeros((2, 1, 2)),
        Ez=jnp.zeros((1, 2, 2)),
        Hx=jnp.zeros((1, 1, 2)),
        Hy=jnp.zeros((1, 2, 1)),
        Hz=jnp.zeros((2, 1, 1)),
        eps_x=jnp.full((2, 2, 1), 2.0),
        eps_y=jnp.full((2, 1, 2), 3.0),
        eps_z=jnp.full((1, 2, 2), 4.0),
    )

    def compact_residuals(_fields, *, dt):
        del _fields, dt
        return (
            _ModeSource3DResidual(
                component="Hy",
                timing="h",
                index=(slice(0, 1), slice(0, 1), slice(0, 1)),
                residual=np.asarray([[[1.0 + 2.0j]]], dtype=np.complex128),
            ),
            _ModeSource3DResidual(
                component="Ex",
                timing="e",
                index=(slice(0, 1), slice(1, 2), slice(0, 1)),
                residual=np.asarray([[[3.0 - 4.0j]]], dtype=np.complex128),
            ),
        )

    source = SimpleNamespace(
        _axis="z",
        pol="te",
        _direction_sign=1.0,
        _compute_discrete_3d_phasor_residuals=compact_residuals,
    )
    waveform = jnp.asarray([2.0, 3.0], dtype=jnp.float32)
    quadrature = jnp.asarray([5.0, 7.0], dtype=jnp.float32)

    specs = _compile_mode_source_3d(
        source,
        fields,
        dt=0.25,
        resolution=7.0,
        h_waveform=waveform,
        e_waveform=waveform,
        h_quadrature_waveform=quadrature,
        e_quadrature_waveform=quadrature,
        t0=1.0,
    )

    hy_specs = [spec for spec in specs if spec.component == "Hy"]
    ex_specs = [spec for spec in specs if spec.component == "Ex"]
    assert len(hy_specs) == 2
    assert len(ex_specs) == 2
    assert hy_specs[0].timing == "h"
    assert hy_specs[0].slab_starts == (0, 0, 0)
    assert hy_specs[0].slab_sizes == (1, 1, 1)
    np.testing.assert_allclose(np.asarray(hy_specs[0].coeff).reshape(()), 1.0)
    np.testing.assert_allclose(np.asarray(hy_specs[1].coeff).reshape(()), -2.0)
    assert ex_specs[0].timing == "e"
    assert ex_specs[0].slab_starts == (0, 1, 0)
    assert ex_specs[0].slab_sizes == (1, 1, 1)
    np.testing.assert_allclose(np.asarray(ex_specs[0].coeff).reshape(()), 3.0)
    np.testing.assert_allclose(np.asarray(ex_specs[1].coeff).reshape(()), 4.0)
    np.testing.assert_allclose(np.asarray(hy_specs[0].waveform), np.asarray(waveform))
    np.testing.assert_allclose(np.asarray(hy_specs[1].waveform), np.asarray(quadrature))


def test_cache_reuse_across_equal_chunks(small_sim_params):
    """Equal-sized chunks should reuse the same compiled program (compile_count == 1)."""
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

    # Run with record_interval to force multiple equal-sized chunks.
    chunk_size = 30
    sim.run_compiled(num_steps=90, record_interval=chunk_size, progress=False)

    # The program should have been compiled only once (all chunks are size 30).
    assert sim._compiled_program is not None
    assert sim._compiled_program.compile_count == 1


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
    assert with_source.source_specs

    sim.sources = []
    no_source = sim.compile(num_steps=8)

    assert no_source is not with_source
    assert no_source.source_specs == ()


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
    sim_single.run_compiled(num_steps=120, progress=False)

    # Chunked: run 4 chunks of 30 steps each.
    sim_chunked.run_compiled(num_steps=120, record_interval=30, progress=False)

    ez_single = np.asarray(sim_single.fields.Ez)
    ez_chunked = np.asarray(sim_chunked.fields.Ez)

    assert sim_single.current_step == sim_chunked.current_step
    assert np.allclose(ez_single, ez_chunked, rtol=1e-5, atol=1e-6)
