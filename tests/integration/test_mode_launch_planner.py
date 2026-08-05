from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from beamz import GaussianPulse, ModeSource, ModeSpec, SampledSignal
from beamz.devices.modes.discrete import DiscreteMode
from beamz.devices.sources import compiler as source_compiler
from beamz.devices.sources import mode_launch as mode_launch_module
from beamz.devices.sources.compiler import (
    _broadband_launch_amplitude_scales,
    compile_source_specs,
)
from beamz.devices.sources.mode_launch import (
    Mode3DLaunchPlan,
    _launch_amplitude_scale,
    _launch_power_diagnostics_3d,
    plan_mode_source_launch,
)
from beamz.devices.sources.planar_tfsf import ModeSource3DResidual
from beamz.devices.sources.solve import mode_plane_context
from beamz.devices.sources.specs import FieldProfile3D
from beamz.devices.sources.time import (
    chebyshev_frequency_nodes,
    sample_source_waveforms,
)
from beamz.lattice import component_shapes
from beamz.simulation.observe import source_normalization as _source_normalization
from tests.utils import compiled_grid

ROOT = Path(__file__).resolve().parents[2]


def _uniform_3d_fields(shape=(5, 5, 6), resolution=1.0):
    fields = compiled_grid(
        permittivity=np.ones(shape, dtype=np.float32) * 2.25,
        conductivity=np.zeros(shape, dtype=np.float32),
        permeability=np.ones(shape, dtype=np.float32),
        resolution=resolution,
    )
    fields.boundaries = []
    return fields


def _direct_yee_2d_fields(shape, *, resolution=1.0):
    fields = compiled_grid(
        permittivity=np.ones(shape, dtype=np.float32) * 2.25,
        conductivity=np.zeros(shape, dtype=np.float32),
        permeability=np.ones(shape, dtype=np.float32),
        resolution=resolution,
    )
    yee_materials = {"eps_z": np.ones(tuple(value + 1 for value in shape))}
    fields.material_grid = replace(
        fields.material_grid,
        yee_materials=yee_materials,
        smoothing="farjadpour_diagonal",
    )
    return fields


def _full_tensor_2d_fields(shape, *, resolution=1.0):
    fields = _direct_yee_2d_fields(shape, resolution=resolution)
    support_shape = fields.material_grid.yee_materials["eps_z"].shape
    tensor = np.zeros((6, *support_shape), dtype=np.float32)
    tensor[:3] = 2.25
    tensor[3] = 0.1
    fields.material_grid = replace(
        fields.material_grid,
        yee_tensors={"eps_z": tensor},
        smoothing="farjadpour_full",
    )
    return fields


def _mode_source(**overrides):
    source_time = overrides.pop(
        "source_time", GaussianPulse(freq0=2.0e14, fwidth=2.0e13)
    )
    signal = overrides.pop("signal", None)
    profile_frequencies = overrides.pop("profile_frequencies", None)
    if source_time is None:
        source_time = SampledSignal(
            np.ones(8) if signal is None else signal,
            dt=1e-16,
            freq0=2.0e14,
        )
    mode_spec = ModeSpec(
        num_freqs=1 if profile_frequencies is None else len(profile_frequencies),
        polarization="te",
    )
    kwargs = dict(
        center=(2.5, 2.5, 2.5),
        size=(0.0, 2.0, 2.0),
        source_time=source_time,
        direction="+",
        mode_spec=mode_spec,
    )
    kwargs.update(overrides)
    return ModeSource(**kwargs)


def _fake_discrete_mode():
    profile = np.ones((2, 2), dtype=np.complex128)
    profiles = {"Ey": profile, "Hz": profile}
    return DiscreteMode(
        neff=1.5 + 0.0j,
        profiles=profiles,
        backward_profiles={"Ey": profile.copy(), "Hz": -profile},
        component_indices={
            "Ey": (slice(1, 3), slice(1, 3), 1),
            "Hz": (slice(1, 3), slice(1, 3), 1),
        },
        axis="x",
        direction="+x",
        transverse_axes=("z", "y"),
        phase_reference_component="Hz",
        phase_reference_coord=1.5,
        phase_plane_coord=1.5,
        k_num_axis=2.0,
        power_scale=1.0,
        diagnostics={},
    )


def test_mode_source_frequency_nodes_cover_tidy_style_band():
    freq0 = 2.0e14
    fwidth = 0.1 * freq0

    nodes = chebyshev_frequency_nodes(freq0, fwidth, 7)

    half_span = 1.5 * fwidth * np.cos(np.pi / 14.0)
    assert nodes[0] == pytest.approx(freq0 - half_span)
    assert nodes[-1] == pytest.approx(freq0 + half_span)


def test_mode_launch_planner_consumes_native_discrete_mode(monkeypatch):
    fields = _uniform_3d_fields()
    source = _mode_source()
    before = dict(source.__dict__)
    seen = {}

    def fake_solve_beamz_mode(spec):
        seen.update(vars(spec))
        return _fake_discrete_mode()

    monkeypatch.setattr(mode_launch_module, "solve_beamz_mode", fake_solve_beamz_mode)

    plan = plan_mode_source_launch(source, fields, resolution=1.0, dt=1e-15)

    assert isinstance(plan, Mode3DLaunchPlan)
    assert {res.component for res in plan.residuals} == {"Ey", "Hz"}
    assert seen["axis"] == "x"
    assert seen["direction"] == "+x"
    assert seen["solver_direction"] == "+x"
    assert seen["polarization"] == "te"
    assert source.__dict__ == before
    for removed_attr in (
        "_initialized",
        "_discrete_mode",
        "_Ey_profile",
        "_Hz_profile",
    ):
        assert not hasattr(source, removed_attr)


def test_mode_launch_receives_same_diagonal_tensor_material_view(monkeypatch):
    fields = _uniform_3d_fields()
    shape = fields.permittivity.shape
    epsilon = np.stack(
        [
            np.full(shape, value, dtype=np.float32)
            for value in (2.0, 3.0, 4.0, 0.0, 0.0, 0.0)
        ]
    )
    mu = np.stack(
        [
            np.full(shape, value, dtype=np.float32)
            for value in (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        ]
    )
    fields.material_grid = replace(
        fields.material_grid,
        tensors={"epsilon": epsilon, "mu": mu},
    )
    seen = {}

    def fake_solve(spec):
        seen.update(vars(spec))
        return _fake_discrete_mode()

    monkeypatch.setattr(mode_launch_module, "solve_beamz_mode", fake_solve)
    plan_mode_source_launch(_mode_source(), fields, resolution=1.0, dt=1e-15)

    assert set(seen["diagonal_permittivity"]) == {"xx", "yy", "zz"}
    assert np.all(seen["diagonal_permittivity"]["xx"] == 2.0)
    assert np.all(seen["diagonal_permittivity"]["yy"] == 3.0)
    assert np.all(seen["diagonal_permittivity"]["zz"] == 4.0)


def test_3d_mode_launch_receives_direct_yee_material_view(monkeypatch):
    fields = _uniform_3d_fields()
    shapes = component_shapes(fields.material_grid.shape)
    yee_materials = {
        name: np.full(shapes[component], value)
        for name, component, value in (
            ("eps_x", "Ex", 2.0),
            ("eps_y", "Ey", 3.0),
            ("eps_z", "Ez", 4.0),
            ("mu_hx", "Hx", 1.0),
            ("mu_hy", "Hy", 1.0),
            ("mu_hz", "Hz", 1.0),
        )
    }
    fields.material_grid = replace(
        fields.material_grid,
        yee_materials=yee_materials,
        smoothing="farjadpour_diagonal",
    )
    seen = {}

    def fake_solve(spec):
        seen.update(vars(spec))
        return _fake_discrete_mode()

    monkeypatch.setattr(mode_launch_module, "solve_beamz_mode", fake_solve)
    plan_mode_source_launch(_mode_source(), fields, resolution=1.0, dt=1e-15)

    assert np.all(seen["component_permittivity"]["Ex"] == 2.0)
    assert np.all(seen["component_permittivity"]["Ey"] == 3.0)
    assert np.all(seen["component_permittivity"]["Ez"] == 4.0)


def test_mode_launch_rejects_conductive_material_profile():
    fields = _uniform_3d_fields()
    fields.conductivity = np.full(fields.permittivity.shape, 0.1)

    with pytest.raises(ValueError, match="conductive material"):
        plan_mode_source_launch(_mode_source(), fields, resolution=1.0, dt=1e-15)


def test_2d_mode_launch_rejects_full_off_diagonal_materials():
    fields = _full_tensor_2d_fields((5, 6))
    source = _mode_source(mode_spec=ModeSpec(polarization="tm"))

    with pytest.raises(ValueError, match="full off-diagonal"):
        plan_mode_source_launch(source, fields, resolution=1.0, dt=1e-15)


def test_2d_te_mode_launch_uses_te_solver_and_te_yee_components(monkeypatch):
    fields = compiled_grid(
        permittivity=np.ones((7, 8), dtype=np.float32) * 2.25,
        conductivity=np.zeros((7, 8), dtype=np.float32),
        permeability=np.ones((7, 8), dtype=np.float32),
        resolution=1.0,
        polarization="te",
    )
    source = _mode_source(
        center=(3.0, 3.0, 0.0),
        size=(0.0, 4.0, 1.0),
        mode_spec=ModeSpec(polarization=None),
    )
    seen = {}

    def fake_solve_modes(**kwargs):
        seen.update(kwargs)
        count = np.asarray(kwargs["eps"]).size
        e = np.zeros((1, 3, count), dtype=np.complex128)
        h = np.zeros_like(e)
        e[0, 1] = 2.0
        h[0, 2] = 1.0
        return np.asarray([1.5]), e, h, None

    monkeypatch.setattr(mode_launch_module, "solve_modes", fake_solve_modes)
    plan = plan_mode_source_launch(source, fields, resolution=1.0, dt=1e-15)

    assert {entry.component for entry in plan.entries} == {"Ey", "Hz"}
    assert seen["filter_pol"] == "te"
    assert seen["grid_edges"] is None
    assert plan.normal_spacing is None


def test_2d_mode_source_rejects_polarization_mismatch():
    fields = compiled_grid(
        permittivity=np.ones((7, 8), dtype=np.float32),
        conductivity=np.zeros((7, 8), dtype=np.float32),
        permeability=np.ones((7, 8), dtype=np.float32),
        resolution=1.0,
        polarization="te",
    )
    source = _mode_source(mode_spec=ModeSpec(polarization="tm"))

    with pytest.raises(ValueError, match="does not match"):
        plan_mode_source_launch(source, fields, resolution=1.0, dt=1e-15)


def test_public_mode_solve_rejects_full_off_diagonal_materials():
    fields = _full_tensor_2d_fields((5, 6))
    simulation = SimpleNamespace(fields=fields, resolution=1.0)
    plane = SimpleNamespace(
        center=(2.5, 2.5, 2.5),
        size=(0.0, 2.0, 2.0),
        axis="x",
    )

    with pytest.raises(ValueError, match="full off-diagonal"):
        mode_plane_context(simulation=simulation, plane=plane)


def test_public_mode_solve_rejects_conductive_materials():
    fields = compiled_grid(
        permittivity=np.ones((5, 6)),
        conductivity=np.ones((5, 6)),
        permeability=np.ones((5, 6)),
        resolution=1.0,
    )
    simulation = SimpleNamespace(
        fields=fields,
        resolution=1.0,
        design=SimpleNamespace(width=6.0, height=5.0, depth=0.0),
    )
    plane = SimpleNamespace(center=(2.5, 2.5, 2.5), size=(0.0, 2.0, 2.0))

    with pytest.raises(ValueError, match="conductive material"):
        mode_plane_context(simulation=simulation, plane=plane)


def test_mode_launch_recomputes_nonfinite_discrete_wave_number(monkeypatch):
    mode = replace(_fake_discrete_mode(), k_num_axis=np.nan)
    calls = []
    monkeypatch.setattr(
        mode_launch_module,
        "solve_mode_plane_3d",
        lambda *_args, **_kwargs: mode,
    )
    monkeypatch.setattr(
        mode_launch_module,
        "_numeric_wave_number",
        lambda *args: calls.append(args) or 3.0,
    )

    plan = plan_mode_source_launch(
        _mode_source(),
        _uniform_3d_fields(),
        resolution=1.0,
        dt=1e-15,
    )

    assert isinstance(plan, Mode3DLaunchPlan)
    assert len(calls) == 1


def test_project_packages_the_native_solver_without_external_micromode():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lockfile = (ROOT / "uv.lock").read_text(encoding="utf-8")

    assert '"micromode' not in pyproject
    assert 'name = "micromode"' not in lockfile
    assert "mode-io" not in pyproject
    assert "h5py" not in pyproject


def test_mode_source_compile_is_deterministic_and_does_not_mutate_source(monkeypatch):
    fields = _uniform_3d_fields()
    source = _mode_source(
        source_time=GaussianPulse(freq0=2.0e14, fwidth=2.0e13),
        signal=None,
        profile_frequencies=np.asarray([1.9e14, 2.1e14], dtype=float),
    )
    before = dict(source.__dict__)
    calls = []

    def fake_plan_mode_source_launch(profile_source, fields_arg, *, resolution, dt):
        del fields_arg, resolution, dt
        calls.append(float(profile_source.frequency))
        residual = ModeSource3DResidual(
            component="Hz",
            timing="h",
            index=(slice(1, 3), slice(1, 3), slice(1, 2)),
            residual=np.ones((2, 2, 1), dtype=np.complex128)
            * (1.0 / float(profile_source.frequency)),
        )
        return Mode3DLaunchPlan((residual,))

    monkeypatch.setattr(
        source_compiler,
        "plan_mode_source_launch",
        fake_plan_mode_source_launch,
    )

    first = compile_source_specs(
        (source,),
        fields,
        dt=1e-15,
        resolution=1.0,
        num_steps=6,
        t0=0.0,
        total_steps=6,
    )
    second = compile_source_specs(
        (source,),
        fields,
        dt=1e-15,
        resolution=1.0,
        num_steps=6,
        t0=0.0,
        total_steps=6,
    )

    assert source.__dict__ == before
    assert len(first) == len(second) == 2
    assert len(calls) == 4
    for lhs, rhs in zip(first, second, strict=True):
        assert lhs.component == rhs.component == "Hz"
        assert lhs.timing == rhs.timing == "h"
        np.testing.assert_allclose(np.asarray(lhs.coeff), np.asarray(rhs.coeff))
        np.testing.assert_allclose(np.asarray(lhs.waveform), np.asarray(rhs.waveform))


def test_mode_source_flux_normalization_uses_waveform_only():
    fields = _uniform_3d_fields()
    freq0 = 2.0e14
    freqs = np.asarray([freq0, 1.05 * freq0], dtype=float)
    source = _mode_source(
        source_time=GaussianPulse(freq0=freq0, fwidth=0.1 * freq0),
        signal=None,
    )
    before = dict(source.__dict__)

    normalization = _source_normalization(
        [source],
        freqs,
        fields=fields,
        dt=1e-15,
    )
    assert normalization is not None

    expected = source.source_spectrum(freqs, normalize=True)
    np.testing.assert_allclose(normalization.field_amplitude_norm, expected)
    assert normalization.launch_power_ratio is None
    assert source.__dict__ == before


def test_sampled_mode_source_normalization_does_not_replan_launch(monkeypatch):
    fields = _uniform_3d_fields()
    freqs = np.asarray([2.0e14, 2.1e14], dtype=float)
    source = _mode_source(signal=np.ones(8, dtype=float), source_time=None)
    before = dict(source.__dict__)
    calls = []

    def fake_plan_mode_source_launch(profile_source, fields_arg, *, resolution, dt):
        calls.append((profile_source, fields_arg, resolution, dt))
        return Mode3DLaunchPlan((), launch_power_ratio=1.44)

    monkeypatch.setattr(
        mode_launch_module,
        "plan_mode_source_launch",
        fake_plan_mode_source_launch,
    )

    normalization = _source_normalization(
        [source],
        freqs,
        time=np.arange(8, dtype=float),
        fields=fields,
        dt=1e-15,
    )
    assert normalization is not None

    assert normalization.launch_power_ratio is None
    assert calls == []
    assert source.__dict__ == before


def test_mode_launch_amplitude_scale_normalizes_measured_launch_power():
    assert _launch_amplitude_scale(0.25) == pytest.approx(2.0)
    assert _launch_amplitude_scale(4.0) == pytest.approx(0.5)
    assert _launch_amplitude_scale(None) == pytest.approx(1.0)
    assert _launch_amplitude_scale(0.0) == pytest.approx(1.0)


def test_launch_diagnostics_use_yee_plane_power_contract(monkeypatch):
    fields = _uniform_3d_fields()
    source = _mode_source(power=2.0)
    profile = np.ones((2, 2), dtype=np.complex128)
    field_profile = FieldProfile3D(
        components={"Ey": profile, "Hz": profile},
        indices={
            "Ey": (slice(1, 3), slice(1, 3), 1),
            "Hz": (slice(1, 3), slice(1, 3), 1),
        },
        axis="x",
        direction_sign=1.0,
        omega=2.0,
        k_axis=1.0,
        phase_ref_coord=1.5,
        phase_plane_coord=1.5,
    )
    monkeypatch.setattr(
        mode_launch_module,
        "_reconstructed_3d_launch_phasor_state",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        mode_launch_module.planar_tfsf,
        "deembed_3d_phasor_profiles",
        lambda *args, **kwargs: field_profile.components,
    )
    monkeypatch.setattr(
        mode_launch_module,
        "_yee_plane_power_3d",
        lambda *args, **kwargs: 2.12,
    )

    ratio, power = _launch_power_diagnostics_3d(
        source,
        field_profile,
        (),
        fields,
        resolution=1.0,
        dt=0.1,
        requested_power=2.0,
    )

    assert power == pytest.approx(2.12)
    assert ratio == pytest.approx(1.06)


def test_mode_launch_plan_reports_scaled_net_launched_power():
    plan = Mode3DLaunchPlan(
        (),
        launch_amplitude_scale=0.9,
        unscaled_launched_power=1.2,
    )

    assert plan.launched_power == pytest.approx(1.2 * 0.9**2)


def test_broadband_launch_scales_normalize_every_profile_node():
    nodes = np.asarray([1.0, 2.0, 3.0])
    ratios = np.asarray([1.0, 4.0, 9.0])
    plans = [
        SimpleNamespace(launch_power_ratio=ratio, launch_amplitude_scale=1.0)
        for ratio in ratios
    ]

    scales = _broadband_launch_amplitude_scales(nodes, plans)

    np.testing.assert_allclose(scales, [1.0, 0.5, 1.0 / 3.0])
    np.testing.assert_allclose(ratios * scales**2, np.ones_like(ratios))
    assert scales[1] == pytest.approx(_launch_amplitude_scale(ratios[1]))


def test_multifrequency_source_normalization_uses_requested_waveform(monkeypatch):
    fields = _uniform_3d_fields()
    freq0 = 2.0e14
    fwidth = 0.1 * freq0
    freqs = np.asarray([0.95 * freq0, freq0, 1.05 * freq0], dtype=float)
    nodes = np.asarray([0.9 * freq0, freq0, 1.1 * freq0], dtype=float)
    time = np.arange(4096, dtype=float) * 1e-16
    source = _mode_source(
        source_time=GaussianPulse(freq0=freq0, fwidth=fwidth),
        signal=None,
        profile_frequencies=nodes,
    )

    def fake_plan_mode_source_launch(profile_source, fields_arg, *, resolution, dt):
        raise AssertionError("source-time result normalization must not re-plan launch")

    monkeypatch.setattr(
        mode_launch_module,
        "plan_mode_source_launch",
        fake_plan_mode_source_launch,
    )

    normalization = _source_normalization(
        [source],
        freqs,
        time=time,
        fields=fields,
        dt=1e-16,
    )

    assert normalization is not None
    signal, quadrature = sample_source_waveforms(
        source.source_time,
        t0=float(time[0]),
        dt=1e-16,
        num_steps=time.size,
        total_steps=time.size,
    )
    del quadrature
    expected_signal = np.asarray(signal, dtype=float)
    phase = np.exp(1j * 2.0 * np.pi * time[:, None] * freqs[None, :])
    expected = (2.0 / time.size) * np.sum(expected_signal[:, None] * phase, axis=0)

    np.testing.assert_allclose(normalization.field_amplitude_norm, expected)


def test_mode_source_compile_applies_launch_amplitude_scale(monkeypatch):
    fields = _uniform_3d_fields()
    source = _mode_source(
        source_time=GaussianPulse(freq0=2.0e14, fwidth=2.0e13),
        signal=None,
    )
    before = dict(source.__dict__)

    residual = ModeSource3DResidual(
        component="Hz",
        timing="h",
        index=(slice(1, 3), slice(1, 3), slice(1, 2)),
        residual=np.ones((2, 2, 1), dtype=np.complex128),
    )

    def fake_plan_mode_source_launch(profile_source, fields_arg, *, resolution, dt):
        del profile_source, fields_arg, resolution, dt
        return Mode3DLaunchPlan(
            (residual,),
            launch_power_ratio=1.44,
            launch_amplitude_scale=1.2,
            unscaled_launched_power=0.8,
        )

    monkeypatch.setattr(
        source_compiler,
        "plan_mode_source_launch",
        fake_plan_mode_source_launch,
    )

    specs = compile_source_specs(
        (source,),
        fields,
        dt=1e-15,
        resolution=1.0,
        num_steps=6,
        t0=0.0,
        total_steps=6,
    )

    assert len(specs) == 1
    np.testing.assert_allclose(
        np.asarray(specs[0].coeff),
        np.full((2, 2, 1), 1.2),
    )
    assert specs[0].source_index == 0
    assert specs[0].launched_power == pytest.approx(0.8 * 1.2**2)
    assert source.__dict__ == before


def test_mode_source_compile_accepts_reload_equivalent_3d_launch_plan(monkeypatch):
    fields = _uniform_3d_fields()
    source = _mode_source(
        source_time=GaussianPulse(freq0=2.0e14, fwidth=2.0e13),
        signal=None,
    )
    residual = ModeSource3DResidual(
        component="Hz",
        timing="h",
        index=(slice(1, 3), slice(1, 3), slice(1, 2)),
        residual=np.ones((2, 2, 1), dtype=np.complex128),
    )

    def fake_plan_mode_source_launch(profile_source, fields_arg, *, resolution, dt):
        del profile_source, fields_arg, resolution, dt
        return SimpleNamespace(
            residuals=(residual,),
            launch_amplitude_scale=1.2,
        )

    monkeypatch.setattr(
        source_compiler,
        "plan_mode_source_launch",
        fake_plan_mode_source_launch,
    )

    specs = compile_source_specs(
        (source,),
        fields,
        dt=1e-15,
        resolution=1.0,
        num_steps=6,
        t0=0.0,
        total_steps=6,
    )

    assert len(specs) == 1
    np.testing.assert_allclose(
        np.asarray(specs[0].coeff),
        np.full((2, 2, 1), 1.2),
    )


def test_mode_source_rejects_removed_runtime_launch_attributes():
    with pytest.raises(TypeError, match="unexpected keyword"):
        ModeSource(
            center=(0.0, 0.0, 0.0),
            size=(0.0, 1.0, 1.0),
            source_time=GaussianPulse(freq0=2e14, fwidth=2e13),
            direction="+",
            _initialized=True,
        )
