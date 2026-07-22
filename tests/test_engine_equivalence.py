from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    PML,
    Design,
    FieldMonitor,
    FieldRecorder,
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
from beamz.const import EPS_0
from beamz.devices.sources.compiler import _sample_waveform
from beamz.devices.sources.specs import CustomSource

FIELD_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
TEST_WAVELENGTH = 1.55 * um

pytestmark = [pytest.mark.compiled, pytest.mark.component]


def _make_2d_sim(
    *,
    plane_2d: str,
    steps: int,
    sources=None,
    boundaries=None,
    monitors=None,
) -> tuple[Simulation, float]:
    wl = TEST_WAVELENGTH
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=2, safety_factor=0.9, points_per_wavelength=6
    )
    time = np.arange(steps, dtype=float) * dt
    design = Design(
        width=2.4 * wl,
        height=2.0 * wl,
        material=Material(permittivity=1.0),
    )
    sim = Simulation(
        design=design,
        sources=[],
        monitors=list(monitors or ()),
        boundaries=list(boundaries or ()),
        time=time,
        resolution=dx,
        plane_2d=plane_2d,
    )
    sim = sim.updated_copy(
        sources=tuple(_source_for_sim(source, sim) for source in sources or ())
    )
    return sim, wl


def _make_3d_sim(
    *, steps: int, sources=None, boundaries=None, monitors=None
) -> tuple[Simulation, float]:
    wl = TEST_WAVELENGTH
    dx, dt = calc_optimal_fdtd_params(
        wl, 1.0, dims=3, safety_factor=0.9, points_per_wavelength=5
    )
    time = np.arange(steps, dtype=float) * dt
    design = Design(
        width=1.8 * wl,
        height=1.8 * wl,
        depth=1.8 * wl,
        material=Material(permittivity=1.0),
    )
    sim = Simulation(
        design=design,
        sources=[],
        monitors=list(monitors or ()),
        boundaries=list(boundaries or ()),
        time=time,
        resolution=dx,
    )
    sim = sim.updated_copy(
        sources=tuple(_source_for_sim(source, sim) for source in sources or ())
    )
    return sim, wl


def _source_for_sim(source, sim):
    method = getattr(source, "to_custom_spec", None)
    return method(sim) if callable(method) else source


def _material_values(fields, name, index, target_shape):
    value = np.asarray(getattr(fields, name))
    if value.ndim == 0:
        value = np.broadcast_to(value, target_shape)
    return np.asarray(value[index], dtype=np.float32)


def _make_2d_tm_mode_source_sim(*, steps: int) -> tuple[Simulation, float]:
    wl = TEST_WAVELENGTH
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

    time = np.arange(steps, dtype=float) * dt
    freq = LIGHT_SPEED / wl
    signal = ramped_cosine(
        time,
        amplitude=0.1,
        frequency=freq,
        ramp_duration=2 / freq,
        t_max=max(time[-1], dt) * 0.5,
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
        time=time,
        resolution=dx,
    )
    return sim, wl


def _seed_field_payload(sim: Simulation, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    payload = {}
    fields = sim.compile().grid
    for idx, name in enumerate(FIELD_COMPONENTS, start=1):
        field = np.asarray(getattr(fields, name))
        if field.size == 0:
            payload[name] = field.astype(np.float32, copy=True)
            continue
        payload[name] = rng.normal(
            loc=0.0,
            scale=0.015 / idx,
            size=field.shape,
        ).astype(np.float32)
    return payload


def _state_with_payload(sim: Simulation, payload: dict[str, np.ndarray]):
    state = sim.initial_state()
    updates = {name.lower(): jnp.asarray(values) for name, values in payload.items()}
    return state._replace(**updates)


def _seed_interior_field_payload(sim: Simulation, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    payload = {}
    fields = sim.compile().grid
    for idx, name in enumerate(FIELD_COMPONENTS, start=1):
        field = np.zeros_like(np.asarray(getattr(fields, name)), dtype=np.float32)
        if field.ndim == 3 and min(field.shape) > 2:
            interior = (slice(1, -1), slice(1, -1), slice(1, -1))
            field[interior] = rng.normal(
                loc=0.0,
                scale=0.015 / idx,
                size=field[interior].shape,
            ).astype(np.float32)
        payload[name] = field
    return payload


def _run_reference(sim: Simulation, steps: int, state=None):
    state = sim.initial_state() if state is None else state
    for _ in range(steps):
        state = sim.step(state)
    return state


def _assert_fields_close(
    reference,
    compiled,
    *,
    atol: float,
    rtol: float,
) -> None:
    assert reference.current_step == compiled.current_step
    for name in FIELD_COMPONENTS:
        np.testing.assert_allclose(
            np.asarray(getattr(reference, name.lower())),
            np.asarray(getattr(compiled, name.lower())),
            atol=atol,
            rtol=rtol,
            err_msg=f"field mismatch for {name}",
        )


def _assert_monitor_close(
    reference,
    compiled,
    *,
    power_atol: float,
    power_rtol: float,
    dft_components: tuple[str, ...] = (),
    dft_atol: float | None = None,
    dft_rtol: float | None = None,
) -> None:
    ref_power = np.asarray(reference.power_history, dtype=np.float64)
    cmp_power = np.asarray(compiled.power_history, dtype=np.float64)
    ref_ts = np.asarray(reference.power_timestamps, dtype=np.float64)
    cmp_ts = np.asarray(compiled.power_timestamps, dtype=np.float64)
    np.testing.assert_allclose(cmp_power, ref_power, atol=power_atol, rtol=power_rtol)
    np.testing.assert_allclose(cmp_ts, ref_ts, atol=1e-21, rtol=0.0)
    for component in dft_components:
        np.testing.assert_allclose(
            np.asarray(compiled.get_dft_component(component), dtype=np.complex128),
            np.asarray(reference.get_dft_component(component), dtype=np.complex128),
            atol=power_atol if dft_atol is None else dft_atol,
            rtol=power_rtol if dft_rtol is None else dft_rtol,
            err_msg=f"monitor DFT mismatch for {component}",
        )


class _PointElectricCurrentSource:
    def __init__(
        self, component: str, index: tuple[int, ...], *, frequency_scale: float
    ):
        self.component = str(component)
        self.index = tuple(int(v) for v in index)
        self.frequency_scale = float(frequency_scale)
        self._advanced_index = tuple(
            np.asarray([v], dtype=np.int32) for v in self.index
        )

    def _signal(self, t_sample: float) -> float:
        return float(
            0.6 * np.sin(self.frequency_scale * float(t_sample))
            + 0.25 * np.cos(0.5 * self.frequency_scale * float(t_sample))
        )

    def get_source_terms(self, fields, t, dt, current_step, resolution, design):
        del fields, current_step, resolution, design
        signal_value = self._signal(float(t) + 0.5 * float(dt))
        values = np.asarray([-signal_value], dtype=np.float32)
        return {self.component: (values, self._advanced_index)}, {}

    def to_custom_spec(self, sim):
        fields = sim.compile().grid
        dt = sim.dt
        num_steps = sim.num_steps
        t0 = float(sim.time[0])
        total_steps = sim.num_steps
        axis = "z" if not sim.is_3d else self.component[-1].lower()
        target_shape = (
            tuple(fields.Ez.shape)
            if not sim.is_3d
            else tuple(getattr(fields, self.component).shape)
        )
        eps_region = _material_values(fields, f"eps_{axis}", self.index, target_shape)
        sig_region = _material_values(fields, f"sig_{axis}", self.index, target_shape)
        denom = 1.0 + sig_region * (float(dt) / (2.0 * EPS_0 * eps_region))
        source_coeff = (float(dt) / (EPS_0 * eps_region)) / denom
        coeff = np.asarray([-source_coeff], dtype=np.float32)
        waveform = _sample_waveform(
            lambda t_sample, _dt: self._signal(float(t_sample)),
            t0=t0,
            dt=dt,
            num_steps=num_steps,
            offset_fn=lambda t_sample, dt_sample: t_sample + 0.5 * dt_sample,
            total_steps=total_steps,
        )
        return CustomSource(
            component=self.component,
            timing="e",
            index=self._advanced_index,
            coeff=coeff,
            waveform=waveform,
            target_shape=target_shape,
        )


def _center_index(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(axis // 2) for axis in shape)


@pytest.mark.parametrize("plane_2d", ["xy", "yz", "xz"])
def test_step_and_compiled_match_seeded_fields_without_sources_2d(plane_2d):
    steps = 7
    reference, _ = _make_2d_sim(plane_2d=plane_2d, steps=steps)
    compiled, _ = _make_2d_sim(plane_2d=plane_2d, steps=steps)

    seed = {"xy": 1101, "yz": 2202, "xz": 3303}[plane_2d]
    initial_fields = _seed_field_payload(reference, seed=seed)
    reference_initial = _state_with_payload(reference, initial_fields)
    compiled_initial = _state_with_payload(compiled, initial_fields)

    reference_state = _run_reference(reference, steps, reference_initial)
    result = compiled.advance(num_steps=steps, progress=False, state=compiled_initial)

    _assert_fields_close(reference_state, result.state, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("plane_2d", ["xy", "yz", "xz"])
def test_step_and_compiled_match_point_source_without_pml_2d(plane_2d):
    steps = 9
    probe, _ = _make_2d_sim(plane_2d=plane_2d, steps=steps)
    source_index = _center_index(probe.compile().grid.Ez.shape)
    component = {"xy": "Ez", "yz": "Ex", "xz": "Ey"}[plane_2d]
    source_a = _PointElectricCurrentSource(
        component, source_index, frequency_scale=3.5e14
    )
    source_b = _PointElectricCurrentSource(
        component, source_index, frequency_scale=3.5e14
    )
    reference, _ = _make_2d_sim(plane_2d=plane_2d, steps=steps, sources=[source_a])
    compiled, _ = _make_2d_sim(plane_2d=plane_2d, steps=steps, sources=[source_b])

    reference_state = _run_reference(reference, steps)
    result = compiled.advance(num_steps=steps, progress=False)

    _assert_fields_close(reference_state, result.state, atol=2e-6, rtol=2e-6)


def test_step_and_compiled_match_seeded_fields_with_cpml_2d_xy():
    steps = 6
    pml_thickness = 0.45 * TEST_WAVELENGTH
    reference, _ = _make_2d_sim(
        plane_2d="xy",
        steps=steps,
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
    )
    compiled, _ = _make_2d_sim(
        plane_2d="xy",
        steps=steps,
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
    )

    initial_fields = _seed_field_payload(reference, seed=2405)
    reference_initial = _state_with_payload(reference, initial_fields)
    compiled_initial = _state_with_payload(compiled, initial_fields)

    reference_state = _run_reference(reference, steps, reference_initial)
    result = compiled.advance(num_steps=steps, progress=False, state=compiled_initial)

    _assert_fields_close(reference_state, result.state, atol=3e-5, rtol=2e-4)


@pytest.mark.parametrize("plane_2d", ["yz", "xz"])
def test_step_and_compiled_match_seeded_fields_with_sigma_pml_2d(plane_2d):
    steps = 6
    pml_thickness = 0.45 * TEST_WAVELENGTH
    reference, _ = _make_2d_sim(
        plane_2d=plane_2d,
        steps=steps,
        boundaries=[PML(thickness=pml_thickness)],
    )
    compiled, _ = _make_2d_sim(
        plane_2d=plane_2d,
        steps=steps,
        boundaries=[PML(thickness=pml_thickness)],
    )

    initial_fields = _seed_field_payload(
        reference, seed=2607 if plane_2d == "yz" else 2709
    )
    reference_initial = _state_with_payload(reference, initial_fields)
    compiled_initial = _state_with_payload(compiled, initial_fields)

    reference_state = _run_reference(reference, steps, reference_initial)
    result = compiled.advance(num_steps=steps, progress=False, state=compiled_initial)

    _assert_fields_close(reference_state, result.state, atol=3e-5, rtol=2e-4)


def test_step_and_compiled_match_point_source_with_cpml_2d_xy():
    steps = 9
    pml_thickness = 0.45 * TEST_WAVELENGTH
    probe, _ = _make_2d_sim(
        plane_2d="xy",
        steps=steps,
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
    )
    source_index = _center_index(probe.compile().grid.Ez.shape)
    source_a = _PointElectricCurrentSource("Ez", source_index, frequency_scale=3.5e14)
    source_b = _PointElectricCurrentSource("Ez", source_index, frequency_scale=3.5e14)
    reference, _ = _make_2d_sim(
        plane_2d="xy",
        steps=steps,
        sources=[source_a],
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
    )
    compiled, _ = _make_2d_sim(
        plane_2d="xy",
        steps=steps,
        sources=[source_b],
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
    )

    reference_state = _run_reference(reference, steps)
    result = compiled.advance(num_steps=steps, progress=False)

    _assert_fields_close(reference_state, result.state, atol=5e-5, rtol=5e-4)


def test_step_and_compiled_match_seeded_fields_without_sources_3d():
    steps = 5
    reference, _ = _make_3d_sim(steps=steps)
    compiled, _ = _make_3d_sim(steps=steps)

    # Keep the deterministic seed away from the compact 3D high-wall omission so
    # this test validates the shared bulk Yee update, not ghost-boundary choices.
    initial_fields = _seed_interior_field_payload(reference, seed=3301)
    reference_initial = _state_with_payload(reference, initial_fields)
    compiled_initial = _state_with_payload(compiled, initial_fields)

    reference_state = _run_reference(reference, steps, reference_initial)
    result = compiled.advance(num_steps=steps, progress=False, state=compiled_initial)

    _assert_fields_close(reference_state, result.state, atol=2e-6, rtol=2e-6)


def test_step_and_compiled_match_point_source_without_pml_3d():
    steps = 7
    probe, _ = _make_3d_sim(steps=steps)
    source_index = _center_index(probe.compile().grid.Ez.shape)
    source_a = _PointElectricCurrentSource("Ez", source_index, frequency_scale=3.0e14)
    source_b = _PointElectricCurrentSource("Ez", source_index, frequency_scale=3.0e14)
    reference, _ = _make_3d_sim(steps=steps, sources=[source_a])
    compiled, _ = _make_3d_sim(steps=steps, sources=[source_b])

    reference_state = _run_reference(reference, steps)
    result = compiled.advance(num_steps=steps, progress=False)

    _assert_fields_close(reference_state, result.state, atol=2e-6, rtol=2e-6)


def test_step_and_compiled_match_seeded_fields_with_cpml_3d():
    steps = 4
    pml_thickness = 0.45 * TEST_WAVELENGTH
    reference, _ = _make_3d_sim(
        steps=steps,
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
    )
    compiled, _ = _make_3d_sim(
        steps=steps,
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
    )

    initial_fields = _seed_field_payload(reference, seed=4407)
    reference_initial = _state_with_payload(reference, initial_fields)
    compiled_initial = _state_with_payload(compiled, initial_fields)

    reference_state = _run_reference(reference, steps, reference_initial)
    result = compiled.advance(num_steps=steps, progress=False, state=compiled_initial)

    _assert_fields_close(reference_state, result.state, atol=4e-5, rtol=3e-4)


def test_step_and_compiled_match_point_source_with_cpml_3d():
    steps = 6
    pml_thickness = 0.45 * TEST_WAVELENGTH
    probe, _ = _make_3d_sim(
        steps=steps,
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
    )
    source_index = _center_index(probe.compile().grid.Ez.shape)
    source_a = _PointElectricCurrentSource("Ez", source_index, frequency_scale=3.0e14)
    source_b = _PointElectricCurrentSource("Ez", source_index, frequency_scale=3.0e14)
    reference, _ = _make_3d_sim(
        steps=steps,
        sources=[source_a],
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
    )
    compiled, _ = _make_3d_sim(
        steps=steps,
        sources=[source_b],
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
    )

    reference_state = _run_reference(reference, steps)
    result = compiled.advance(num_steps=steps, progress=False)

    _assert_fields_close(reference_state, result.state, atol=6e-5, rtol=6e-4)


def test_step_and_compiled_match_tm_mode_source_snapshots_without_fallback():
    steps = 20
    snapshot_interval = 5
    reference, _ = _make_2d_tm_mode_source_sim(steps=steps)
    compiled, _ = _make_2d_tm_mode_source_sim(steps=steps)
    compiled = compiled.updated_copy(
        monitors=(
            *compiled.monitors,
            FieldRecorder(("Ez",), snapshot_interval, name="frames"),
        )
    )

    reference_snapshots = []
    reference_steps = []
    reference_state = reference.initial_state()
    for _ in range(steps):
        reference_state = reference.step(reference_state)
        if reference_state.current_step % snapshot_interval == 0:
            reference_snapshots.append(np.asarray(reference_state.ez, dtype=np.float32))
            reference_steps.append(reference_state.current_step)
    result = compiled.advance(num_steps=steps, progress=False)
    recording = result.results.monitor("frames")

    compiled_snapshots = np.asarray(recording.fields["Ez"], dtype=np.float32)
    compiled_steps = recording.field_steps.tolist()

    assert compiled_steps == reference_steps
    assert len(compiled_snapshots) == len(reference_snapshots)
    for ref_frame, cmp_frame in zip(
        reference_snapshots, compiled_snapshots, strict=True
    ):
        np.testing.assert_allclose(ref_frame, cmp_frame, atol=3e-5, rtol=3e-4)

    _assert_fields_close(reference_state, result.state, atol=3e-5, rtol=3e-4)


def test_compiled_2d_monitor_power_and_dft_is_result_owned():
    steps = 12
    probe, wl = _make_2d_sim(plane_2d="xy", steps=steps)
    source_index = _center_index(probe.compile().grid.Ez.shape)
    source_a = _PointElectricCurrentSource("Ez", source_index, frequency_scale=3.5e14)
    source_b = _PointElectricCurrentSource("Ez", source_index, frequency_scale=3.5e14)
    freq = LIGHT_SPEED / wl

    def _build_monitor():
        return FieldMonitor(
            center=(1.55 * wl, wl, 0.0),
            size=(0.0, 1.1 * wl, 0.0),
            interval=3,
            freqs=[freq],
            fields=("Ez", "Hy"),
        )

    ref_monitor = _build_monitor()
    cmp_monitor = _build_monitor()
    reference, _ = _make_2d_sim(
        plane_2d="xy", steps=steps, sources=[source_a], monitors=[ref_monitor]
    )
    compiled, _ = _make_2d_sim(
        plane_2d="xy", steps=steps, sources=[source_b], monitors=[cmp_monitor]
    )

    reference_result = reference.advance(num_steps=steps, progress=False)
    result = compiled.advance(num_steps=steps, progress=False)
    reference_monitor = reference_result.results.monitors["monitor_0"]
    compiled_monitor = result.results.monitors["monitor_0"]

    _assert_fields_close(reference_result.state, result.state, atol=2e-6, rtol=2e-6)
    assert not hasattr(ref_monitor, "power_history")
    assert not hasattr(cmp_monitor, "power_history")
    _assert_monitor_close(
        reference_monitor,
        compiled_monitor,
        power_atol=2e-6,
        power_rtol=2e-6,
        dft_components=("Ez", "Hy"),
        dft_atol=2e-6,
        dft_rtol=2e-6,
    )


def test_compiled_3d_monitor_power_and_dft_is_result_owned():
    steps = 8
    probe, wl = _make_3d_sim(steps=steps)
    source_index = _center_index(probe.compile().grid.Ez.shape)
    source_a = _PointElectricCurrentSource("Ez", source_index, frequency_scale=3.0e14)
    source_b = _PointElectricCurrentSource("Ez", source_index, frequency_scale=3.0e14)
    freq = LIGHT_SPEED / wl

    def _build_monitor():
        return FieldMonitor(
            center=(1.25 * wl, 0.9 * wl, 0.9 * wl),
            size=(0.0, 1.1 * wl, 1.1 * wl),
            interval=2,
            freqs=[freq],
            fields=("Ex", "Hz"),
        )

    ref_monitor = _build_monitor()
    cmp_monitor = _build_monitor()
    reference, _ = _make_3d_sim(steps=steps, sources=[source_a], monitors=[ref_monitor])
    compiled, _ = _make_3d_sim(steps=steps, sources=[source_b], monitors=[cmp_monitor])

    reference_result = reference.advance(num_steps=steps, progress=False)
    result = compiled.advance(num_steps=steps, progress=False)
    reference_monitor = reference_result.results.monitors["monitor_0"]
    compiled_monitor = result.results.monitors["monitor_0"]

    _assert_fields_close(reference_result.state, result.state, atol=2e-6, rtol=2e-6)
    assert not hasattr(ref_monitor, "power_history")
    assert not hasattr(cmp_monitor, "power_history")
    _assert_monitor_close(
        reference_monitor,
        compiled_monitor,
        power_atol=3e-6,
        power_rtol=3e-6,
        dft_components=("Ex", "Hz"),
        dft_atol=3e-6,
        dft_rtol=3e-6,
    )
