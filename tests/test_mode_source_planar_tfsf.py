from __future__ import annotations

import numpy as np
import pytest

from beamz.devices.sources import compiler as source_compiler
from beamz.devices.sources._profiles import FieldProfile3D
from beamz.devices.sources.compiler import apply_compiled_source_specs
from tests.test_3d_constitutive_sampling import (
    _build_centered_straight_guide_sim,
    _build_test_source,
    _source_basis_branch_metrics,
    _source_plane_deembedded_phasor_profiles,
    _target_and_residual_reconstructed_phasor_step,
)


def _realize_component_specs(field, specs):
    out = field * 0.0
    return np.asarray(apply_compiled_source_specs(out, 0, tuple(specs)))


def test_mode_source_exports_runtime_field_profile_3d():
    sim = _build_centered_straight_guide_sim(ppw=6, axis="x")
    source, _dx = _build_test_source(sim, direction="+x", pol="te")
    profiles, indices = source._get_3d_profiles_and_indices()

    field_profile = source._field_profile_3d()

    assert isinstance(field_profile, FieldProfile3D)
    assert field_profile.axis == "x"
    assert field_profile.direction_sign == pytest.approx(1.0)
    assert field_profile.omega == pytest.approx(float(source._omega_launch))
    assert field_profile.k_axis == pytest.approx(float(source._k_num_axis))
    assert field_profile.phase_ref_coord == pytest.approx(source._phase_ref_coord)
    assert field_profile.phase_plane_coord == pytest.approx(source._phase_plane_coord)
    assert set(field_profile.components) == {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
    assert set(field_profile.indices) == set(field_profile.components)
    for component, profile in profiles.items():
        np.testing.assert_allclose(
            field_profile.components[component],
            np.asarray(profile, dtype=np.complex128),
            rtol=0.0,
            atol=0.0,
        )
        assert field_profile.indices[component] == indices[component]


def test_mode_source_planar_tfsf_residual_reconstructs_launched_side_update():
    """The 3D ModeSource residual is the discrete planar TF/SF correction."""
    sim = _build_centered_straight_guide_sim(ppw=6, axis="x")
    source, _dx = _build_test_source(sim, direction="+x", pol="te")
    dt = float(sim.dt)

    target_next, reconstructed_next = _target_and_residual_reconstructed_phasor_step(
        source,
        sim.fields,
        dt=dt,
    )

    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        np.testing.assert_allclose(
            reconstructed_next[component],
            target_next[component],
            rtol=1e-6,
            atol=1e-6,
        )


def test_mode_source_planar_tfsf_keeps_source_profile_forward_pure():
    """The modal source profile remains a single launched branch before update."""
    sim = _build_centered_straight_guide_sim(ppw=6, axis="x")
    source, _dx = _build_test_source(sim, direction="+x", pol="te")
    dt = float(sim.dt)
    profiles, _indices = source._get_3d_profiles_and_indices()

    profile_metrics = _source_basis_branch_metrics(
        source,
        {
            name: np.asarray(value, dtype=np.complex128)
            for name, value in profiles.items()
        },
    )
    target_next, reconstructed_next = _target_and_residual_reconstructed_phasor_step(
        source,
        sim.fields,
        dt=dt,
    )
    target_update_metrics = _source_basis_branch_metrics(
        source,
        _source_plane_deembedded_phasor_profiles(
            source,
            target_next,
            t_e=dt,
            t_h=0.5 * dt,
        ),
    )
    residual_update_metrics = _source_basis_branch_metrics(
        source,
        _source_plane_deembedded_phasor_profiles(
            source,
            reconstructed_next,
            t_e=dt,
            t_h=0.5 * dt,
        ),
    )

    assert profile_metrics["forward_abs"] > 0.0
    assert profile_metrics["backward_ratio"] < 1e-12
    assert target_update_metrics["forward_abs"] > 0.9
    assert target_update_metrics["backward_ratio"] < 0.15
    for key in ("forward_abs", "backward_abs", "backward_ratio"):
        assert residual_update_metrics[key] == pytest.approx(
            target_update_metrics[key],
            rel=5e-6,
            abs=5e-8,
        )


def test_compiled_mode_source_applies_same_planar_tfsf_phasor_residual():
    """Compiled ModeSource specs must represent the same planar TF/SF residual."""
    sim = _build_centered_straight_guide_sim(ppw=6, axis="x")
    source, _dx = _build_test_source(sim, direction="+x", pol="te")
    dt = float(sim.dt)

    specs = source_compiler._compile_mode_source(
        source,
        sim.fields,
        dt=dt,
        num_steps=3,
        t0=0.0,
        resolution=float(sim.resolution),
        total_steps=3,
    )
    specs_by_phase = {
        (spec.timing, spec.component): [
            item
            for item in specs
            if item.timing == spec.timing and item.component == spec.component
        ]
        for spec in specs
    }

    h_delta = source._compute_discrete_3d_h_phasor_delta(sim.fields, dt=dt)
    e_delta = source._compute_discrete_3d_e_phasor_delta(sim.fields, dt=dt)
    expected = {
        ("h", "Hx"): np.real(h_delta["Hx"]),
        ("h", "Hy"): np.real(h_delta["Hy"]),
        ("h", "Hz"): np.real(h_delta["Hz"]),
        ("e", "Ex"): np.real(e_delta["Ex"]),
        ("e", "Ey"): np.real(e_delta["Ey"]),
        ("e", "Ez"): np.real(e_delta["Ez"]),
    }

    for (timing, component), expected_delta in expected.items():
        realized = _realize_component_specs(
            getattr(sim.fields, component),
            specs_by_phase.get((timing, component), ()),
        )
        np.testing.assert_allclose(
            realized,
            expected_delta,
            rtol=1e-5,
            atol=1e-6,
        )
