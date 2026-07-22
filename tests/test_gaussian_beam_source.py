from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    PML,
    Box,
    Design,
    FluxMonitor,
    GaussianBeamSource,
    GaussianPulse,
    GridSpec,
    Material,
    Simulation,
    um,
)
from beamz.devices.sources.compiler import (
    compile_source_specs,
    field_profile_phasor_residuals,
    gaussian_beam_field_profile,
)
from beamz.devices.sources.planar_tfsf import (
    advance_incident_e_3d,
    advance_incident_h_3d,
    build_incident_3d_phasor_state,
    expand_3d_residuals,
    mask_incident_3d_state_to_launched_side,
)


def _empty_3d_fields(shape=(10, 10, 10)):
    nz, ny, nx = (int(v) for v in shape)
    ex = np.zeros((nz, ny, nx - 1), dtype=np.float64)
    ey = np.zeros((nz, ny - 1, nx), dtype=np.float64)
    ez = np.zeros((nz - 1, ny, nx), dtype=np.float64)
    hx = np.zeros((nz - 1, ny - 1, nx), dtype=np.float64)
    hy = np.zeros((nz - 1, ny, nx - 1), dtype=np.float64)
    hz = np.zeros((nz, ny - 1, nx - 1), dtype=np.float64)
    return SimpleNamespace(
        boundaries=[],
        permittivity=np.ones((nz, ny, nx), dtype=np.float64),
        permeability=np.ones((nz, ny, nx), dtype=np.float64),
        conductivity=np.zeros((nz, ny, nx), dtype=np.float64),
        total_conductivity=np.zeros((nz, ny, nx), dtype=np.float64),
        Ex=ex,
        Ey=ey,
        Ez=ez,
        Hx=hx,
        Hy=hy,
        Hz=hz,
        eps_x=np.ones_like(ex),
        eps_y=np.ones_like(ey),
        eps_z=np.ones_like(ez),
        sig_x=np.zeros_like(ex),
        sig_y=np.zeros_like(ey),
        sig_z=np.zeros_like(ez),
        region_x=(slice(None), slice(None), slice(None)),
        region_y=(slice(None), slice(None), slice(None)),
        region_z=(slice(None), slice(None), slice(None)),
        sigma_m_hx=np.zeros_like(hx),
        sigma_m_hy=np.zeros_like(hy),
        sigma_m_hz=np.zeros_like(hz),
    )


def _pulse(wavelength):
    freq = LIGHT_SPEED / float(wavelength)
    return GaussianPulse(
        freq0=freq,
        fwidth=freq,
        offset=0.0,
        remove_dc_component=False,
    )


def test_gaussian_beam_source_emits_compiled_planar_tfsf_specs():
    fields = _empty_3d_fields()
    source = GaussianBeamSource(
        center=(1.0 * um, 1.0 * um, 1.0 * um),
        size=(1.2 * um, 1.2 * um),
        source_time=_pulse(1.55 * um),
        direction="+x",
        angle_theta=0.0,
        angle_phi=0.0,
        pol_angle=0.0,
        waist_radius=0.35 * um,
    )
    dt = 0.15 * um / (LIGHT_SPEED * np.sqrt(3.0))

    specs = compile_source_specs(
        (source,),
        fields,
        dt=dt,
        resolution=0.25 * um,
        num_steps=3,
        t0=0.0,
        total_steps=3,
    )

    assert specs
    assert {"h", "e"} <= {spec.timing for spec in specs}
    assert any(spec.component.startswith("H") for spec in specs)
    assert any(spec.component.startswith("E") for spec in specs)
    assert any(spec.is_slab for spec in specs)


def test_gaussian_beam_source_empty_space_propagation_smoke():
    wavelength = 1.55 * um
    dx = 0.35 * um
    dt = 0.45 * dx / (LIGHT_SPEED * np.sqrt(3.0))
    source = GaussianBeamSource(
        center=(-0.45 * um, 0.0, 0.0),
        size=(1.4 * um, 1.4 * um),
        source_time=_pulse(wavelength),
        direction="+x",
        angle_theta=0.0,
        angle_phi=0.0,
        pol_angle=0.0,
        waist_radius=0.45 * um,
        power=1.0,
    )
    sim = Simulation(
        domain=(2.8 * um, 2.8 * um, 2.8 * um),
        background=Material(permittivity=1.0),
        sources=[source],
        monitors=[],
        boundaries=[],
        time=np.arange(0.0, 5 * dt, dt),
        resolution=dx,
    )

    state = sim.initial_state()
    for _ in range(3):
        state = sim.step(state)

    energy = sum(
        float(np.sum(np.asarray(getattr(state, component.lower())) ** 2))
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    )
    assert np.isfinite(energy)
    assert energy > 0.0


def test_gaussian_beam_source_has_low_source_plane_residual_error():
    fields = _empty_3d_fields(shape=(11, 11, 11))
    dt = 0.12 * um / (LIGHT_SPEED * np.sqrt(3.0))
    resolution = 0.25 * um
    source = GaussianBeamSource(
        center=(1.25 * um, 1.25 * um, 1.25 * um),
        size=(1.5 * um, 1.5 * um),
        source_time=_pulse(1.55 * um),
        direction="+x",
        angle_theta=0.08,
        angle_phi=0.2,
        pol_angle=0.1,
        waist_radius=0.45 * um,
    )
    profile = gaussian_beam_field_profile(source, fields, resolution=resolution)

    full_prev = build_incident_3d_phasor_state(
        profile,
        fields,
        resolution=resolution,
        t_e=0.0,
        t_h=-0.5 * dt,
        masked=False,
        max_shift=source.max_shift,
    )
    masked_prev = build_incident_3d_phasor_state(
        profile,
        fields,
        resolution=resolution,
        t_e=0.0,
        t_h=-0.5 * dt,
        masked=True,
        max_shift=source.max_shift,
    )
    h_full_next = advance_incident_h_3d(
        fields,
        full_prev,
        dt,
        resolution=resolution,
    )
    h_target_next = mask_incident_3d_state_to_launched_side(
        profile,
        h_full_next,
        resolution=resolution,
    )
    h_mask_next = advance_incident_h_3d(
        fields,
        masked_prev,
        dt,
        resolution=resolution,
    )
    e_full_next = advance_incident_e_3d(
        fields,
        full_prev,
        h_full_next,
        dt,
        resolution=resolution,
    )
    e_target_next = mask_incident_3d_state_to_launched_side(
        profile,
        e_full_next,
        resolution=resolution,
    )
    e_mask_next = advance_incident_e_3d(
        fields,
        masked_prev,
        h_target_next,
        dt,
        resolution=resolution,
    )

    residuals = field_profile_phasor_residuals(
        profile,
        fields,
        dt=dt,
        resolution=resolution,
        max_shift=source.max_shift,
    )
    h_delta = expand_3d_residuals(residuals, fields, ("Hx", "Hy", "Hz"))
    e_delta = expand_3d_residuals(residuals, fields, ("Ex", "Ey", "Ez"))

    for component in ("Hx", "Hy", "Hz"):
        np.testing.assert_allclose(
            h_mask_next[component] + h_delta[component],
            h_target_next[component],
            rtol=1e-6,
            atol=1e-8,
        )
    for component in ("Ex", "Ey", "Ez"):
        np.testing.assert_allclose(
            e_mask_next[component] + e_delta[component],
            e_target_next[component],
            rtol=1e-6,
            atol=1e-8,
        )


def test_gaussian_beam_source_compiled_engine_support():
    wavelength = 1.55 * um
    dx = 0.4 * um
    dt = 0.45 * dx / (LIGHT_SPEED * np.sqrt(3.0))
    source = GaussianBeamSource(
        center=(0.0, 0.0, 0.35 * um),
        size=(1.2 * um, 1.2 * um),
        source_time=_pulse(wavelength),
        direction="-z",
        angle_theta=0.1,
        angle_phi=0.2,
        pol_angle=0.0,
        waist_radius=0.45 * um,
    )
    sim = Simulation(
        domain=(2.4 * um, 2.4 * um, 2.4 * um),
        background=Material(permittivity=1.0),
        sources=[source],
        monitors=[],
        boundaries=[],
        time=np.arange(0.0, 4 * dt, dt),
        resolution=dx,
    )

    program = cast(Any, sim.compile(num_steps=2))

    assert program.sources
    assert {"h", "e"} <= {spec.timing for spec in program.sources}
    result = sim.advance(num_steps=1, progress=False)
    total = sum(
        float(np.sum(np.asarray(getattr(result.state, component.lower())) ** 2))
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    )
    assert np.isfinite(total)


def test_gaussian_beam_source_flux_normalizes_by_sampled_waveform():
    wavelength = 1.55 * um
    freq0 = LIGHT_SPEED / wavelength
    source = GaussianBeamSource(
        center=(0.0, 0.0, 1.2 * um),
        size=(5.5 * um, 5.5 * um),
        source_time=GaussianPulse(freq0=freq0, fwidth=freq0 / 5.0, offset=4.0),
        direction="-z",
        pol_angle=np.pi / 2.0,
        waist_radius=1.2 * um,
        wavelength=wavelength,
        power=1.0,
    )
    monitors = cast(
        Any,
        [
            FluxMonitor(
                center=(0.0, 0.0, 0.75 * um),
                size=(5.8 * um, 5.8 * um, 0.0),
                freqs=[freq0],
                name="forward",
            ),
            FluxMonitor(
                center=(0.0, 0.0, 1.75 * um),
                size=(5.8 * um, 5.8 * um, 0.0),
                freqs=[freq0],
                name="backward",
            ),
        ],
    )
    design = Design(
        width=7.0 * um,
        height=7.0 * um,
        depth=6.0 * um,
        material=Material(permittivity=1.0),
    )
    sim = Simulation(
        design=design,
        sources=[source],
        monitors=monitors,
        boundaries=[PML(thickness=0.6 * um)],
        grid_spec=GridSpec.auto(
            min_steps_per_wvl=8,
            wavelength=wavelength,
            courant=0.48,
        ),
        run_time=260e-15,
    )

    results = sim.run(progress=False)
    assert results is not None
    forward_result = results.monitors["forward"]
    backward_result = results.monitors["backward"]
    assert forward_result is not None
    assert backward_result is not None
    forward = -float(np.asarray(forward_result.flux)[0])
    backward = float(np.asarray(backward_result.flux)[0])

    assert forward == pytest.approx(0.1, rel=0.25)
    assert backward < 0.04


def test_gaussian_beam_source_grating_coupler_like_compile_smoke():
    wavelength = 1.55 * um
    dx = 0.35 * um
    dt = 0.45 * dx / (LIGHT_SPEED * np.sqrt(3.0))
    silicon = Material(permittivity=12.0)
    design = Design(
        width=3.0 * um,
        height=2.0 * um,
        depth=2.4 * um,
        material=Material(permittivity=1.0),
    )
    design += Box(
        center=(1.5 * um, 1.0 * um, 0.55 * um),
        size=(1.8 * um, 0.5 * um, 0.22 * um),
        material=silicon,
    )
    for idx in range(3):
        design += Box(
            center=((1.0 + 0.28 * idx) * um, 1.0 * um, 0.75 * um),
            size=(0.12 * um, 0.5 * um, 0.12 * um),
            material=silicon,
        )
    source = GaussianBeamSource(
        center=(1.5 * um, 1.0 * um, 1.65 * um),
        size=(1.5 * um, 1.1 * um),
        source_time=_pulse(wavelength),
        direction="-z",
        angle_theta=0.25,
        angle_phi=0.0,
        pol_angle=0.0,
        waist_radius=0.55 * um,
    )
    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[],
        boundaries=[],
        time=np.arange(0.0, 3 * dt, dt),
        resolution=dx,
    )

    program = cast(Any, sim.compile(num_steps=1))

    assert program.sources
    assert any(spec.component.startswith("E") for spec in program.sources)
    assert any(spec.component.startswith("H") for spec in program.sources)
