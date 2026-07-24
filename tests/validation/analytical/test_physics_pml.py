"""Physics validation tests for absorbing-layer and CPML boundaries.

Tests verify:
1. The graded absorber reduces outgoing-wave reflections
2. Energy decays monotonically after source stops
"""

import jax.numpy as jnp
import numpy as np
import pytest

from beamz import (
    EPS_0,
    LIGHT_SPEED,
    MU_0,
    PML,
    Design,
    FieldRecorder,
    GaussianSource,
    Material,
    ModeSource,
    ModeSpec,
    SampledSignal,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    um,
)
from beamz.devices._boundary_compile import _AbsorberCompiler
from beamz.devices.sources.specs import CustomSource
from beamz.simulation.kernels import (
    tm_xy_curl_h_to_e_2d,
)
from tests.utils import compute_field_energy


class _TaperedLineEzSource:
    """Small test-only line current source for CPML reflection measurements."""

    def __init__(self, *, x, y0, y1, signal):
        self.x = float(x)
        self.y0 = float(y0)
        self.y1 = float(y1)
        self.signal = jnp.asarray(signal)
        self._indices = None
        self._profile = None

    def _initialize(self, fields, resolution):
        ny, nx = fields.Ez.shape
        ix = int(np.clip(round(self.x / resolution), 0, nx - 1))
        y0 = int(np.clip(round(self.y0 / resolution), 0, ny - 1))
        y1 = int(np.clip(round(self.y1 / resolution), y0 + 1, ny))
        n = max(y1 - y0, 1)

        weights = np.ones(n, dtype=np.float32)
        taper = max(1, n // 8)
        if taper > 1:
            ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, taper)))
            weights[:taper] = ramp
            weights[-taper:] = ramp[::-1]

        self._indices = (slice(y0, y1), ix)
        self._profile = jnp.asarray(weights)

    def inject(self, fields, t, dt, current_step, resolution, design):
        if self._indices is None:
            self._initialize(fields, resolution)

        idx_float = (t + 0.5 * dt) / dt
        idx_low = jnp.floor(idx_float).astype(jnp.int32)
        idx_high = idx_low + 1
        frac = idx_float - jnp.floor(idx_float)
        signal_len = self.signal.shape[0]
        value = (1.0 - frac) * self.signal[jnp.clip(idx_low, 0, signal_len - 1)]
        value += frac * self.signal[jnp.clip(idx_high, 0, signal_len - 1)]
        value = jnp.where((idx_low >= 0) & (idx_low < signal_len - 1), value, 0.0)

        injection = -self._profile * value * dt / EPS_0
        fields.Ez = fields.Ez.at[self._indices].add(injection)

    def to_custom_spec(self, sim):
        fields = sim.compile().grid
        if self._indices is None:
            self._initialize(fields, sim.resolution)
        dt = float(sim.dt)
        t0 = float(sim.time[0])
        signal = np.asarray(self.signal, dtype=np.float32).reshape(-1)
        values = np.zeros((int(sim.num_steps),), dtype=np.float32)
        for step in range(values.size):
            sample = t0 + step * dt + 0.5 * dt
            idx_float = sample / dt
            idx_low = int(np.floor(idx_float))
            idx_high = idx_low + 1
            frac = idx_float - np.floor(idx_float)
            if 0 <= idx_low < signal.size - 1:
                values[step] = (1.0 - frac) * signal[idx_low] + frac * signal[idx_high]
        return CustomSource(
            component="Ez",
            timing="e",
            index=self._indices,
            coeff=-np.asarray(self._profile, dtype=np.float32) * dt / EPS_0,
            waveform=values,
            target_shape=tuple(fields.Ez.shape),
        )


def _homogeneous_cpml_reflection_db(
    *, points_per_wavelength: int, refractive_index: float = 1.0
) -> float:
    """Measure time-gated normal-incidence CPML amplitude reflection in dB."""

    wavelength = 1.0 * um
    frequency = LIGHT_SPEED / wavelength
    dx, dt = calc_optimal_fdtd_params(
        wavelength,
        refractive_index,
        dims=2,
        safety_factor=0.95,
        points_per_wavelength=points_per_wavelength,
    )

    width = 14.0 * wavelength
    height = 8.0 * wavelength
    pml_thickness = 1.0 * wavelength
    design = Design(
        width=width,
        height=height,
        material=Material(permittivity=refractive_index**2),
    )

    period = 1.0 / frequency
    time = np.arange(0.0, 38.0 * period, dt)
    pulse_center = 4.0 * period
    pulse_sigma = 0.7 * period
    signal = np.exp(-((time - pulse_center) ** 2) / (2.0 * pulse_sigma**2))
    signal *= np.cos(2.0 * np.pi * frequency * time)

    source_x = 3.0 * wavelength
    probe_x = 6.0 * wavelength
    pml_start_x = width - pml_thickness
    source = _TaperedLineEzSource(
        x=source_x,
        y0=2.0 * wavelength,
        y1=6.0 * wavelength,
        signal=signal.astype(np.float32),
    )
    cpml_alpha_normalized = 0.2
    cpml_alpha_max = 2.0 * EPS_0 * cpml_alpha_normalized / max(float(dt), 1e-30)
    sim = Simulation(
        design=design,
        sources=[],
        boundaries=[
            PML(
                thickness=pml_thickness,
                formulation="cpml",
                alpha_max=cpml_alpha_max,
            )
        ],
        time=time,
        resolution=dx,
    )
    sim = sim.updated_copy(sources=(source.to_custom_spec(sim),))

    probe_ix = int(round(probe_x / dx))
    probe_y0 = int(round(3.0 * wavelength / dx))
    probe_y1 = int(round(5.0 * wavelength / dx))
    samples = []
    state = sim.initial_state()
    for _ in range(len(time)):
        state = sim.step(state)
        ez = np.asarray(state.ez)
        samples.append(float(np.mean(ez[probe_y0:probe_y1, probe_ix])))

    samples = np.asarray(samples)
    sample_times = np.arange(samples.size) * dt
    speed = LIGHT_SPEED / refractive_index
    incident_center = pulse_center + (probe_x - source_x) / speed
    reflected_center = (
        pulse_center + ((pml_start_x - source_x) + (pml_start_x - probe_x)) / speed
    )

    incident_window = (sample_times >= incident_center - 2.0 * period) & (
        sample_times <= incident_center + 2.5 * period
    )
    reflected_window = (sample_times >= reflected_center - 2.0 * period) & (
        sample_times <= reflected_center + 4.0 * period
    )
    incident = float(np.max(np.abs(samples[incident_window])))
    reflected = float(np.max(np.abs(samples[reflected_window])))
    return 20.0 * np.log10(max(reflected, 1e-300) / max(incident, 1e-300))


def _expected_cpml_profile_on_yee_coordinates(
    *,
    domain_cells: int,
    pml_cells: int,
    sample_kind: str,
    sigma_max: float,
    kappa_max: float,
    alpha_max: float,
    order: float,
):
    """Evaluate the CPML polynomial independently on physical Yee coordinates."""

    if sample_kind == "E":
        coordinates = np.arange(domain_cells + 1, dtype=np.float64)
    elif sample_kind == "H":
        coordinates = np.arange(domain_cells, dtype=np.float64) + 0.5
    else:
        raise ValueError(sample_kind)
    low_distance = np.clip(pml_cells - coordinates, 0.0, pml_cells)
    high_distance = np.clip(coordinates - (domain_cells - pml_cells), 0.0, pml_cells)
    distance = np.maximum(low_distance, high_distance)
    active = distance > 0.0
    normalized = distance / float(pml_cells)
    sigma = sigma_max * normalized**order
    kappa = 1.0 + (kappa_max - 1.0) * normalized**order
    alpha = np.where(active, alpha_max * (1.0 - normalized), 0.0)
    return sigma, kappa, alpha


@pytest.mark.parametrize("sample_kind", ["E", "H"])
def test_cpml_profiles_follow_physical_complete_yee_coordinates(sample_kind):
    domain_cells = 24
    pml_cells = 6
    spacing = 0.1
    sigma_max = 10.0
    kappa_max = 3.0
    alpha_max = 2.0
    order = 3.0
    total_samples = domain_cells + 1 if sample_kind == "E" else domain_cells
    compiler = _AbsorberCompiler(
        PML(
            thickness=pml_cells * spacing,
            formulation="cpml",
            sigma_max=sigma_max,
            kappa_max=kappa_max,
            alpha_max=alpha_max,
            m=int(order),
        )
    )

    actual = compiler._compute_fdtdx_staggered_profile_1d(
        total_samples,
        spacing,
        True,
        True,
        sample_kind=sample_kind,
        domain_cells=domain_cells,
    )
    expected = _expected_cpml_profile_on_yee_coordinates(
        domain_cells=domain_cells,
        pml_cells=pml_cells,
        sample_kind=sample_kind,
        sigma_max=sigma_max,
        kappa_max=kappa_max,
        alpha_max=alpha_max,
        order=order,
    )

    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(np.asarray(got), want, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("sample_kind", ["E", "H"])
def test_cpml_complete_yee_profiles_mirror_low_and_high_boundaries(sample_kind):
    domain_cells = 20
    pml_cells = 5
    total_samples = domain_cells + 1 if sample_kind == "E" else domain_cells
    compiler = _AbsorberCompiler(
        PML(
            thickness=float(pml_cells),
            formulation="cpml",
            sigma_max=4.0,
            kappa_max=2.5,
            alpha_max=0.5,
        )
    )

    low = compiler._compute_fdtdx_staggered_profile_1d(
        total_samples,
        1.0,
        True,
        False,
        sample_kind=sample_kind,
        domain_cells=domain_cells,
    )
    high = compiler._compute_fdtdx_staggered_profile_1d(
        total_samples,
        1.0,
        False,
        True,
        sample_kind=sample_kind,
        domain_cells=domain_cells,
    )

    for low_values, high_values in zip(low, high, strict=True):
        np.testing.assert_allclose(
            np.asarray(low_values), np.asarray(high_values)[::-1], rtol=0.0, atol=1e-7
        )


@pytest.mark.simulation
class TestPMLAbsorption:
    """Verify PML boundary absorbs waves properly."""

    def test_cpml_profile_generation_exposes_auxiliary_coefficients(
        self, vacuum_domain_small
    ):
        design = vacuum_domain_small["design"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]

        sim = Simulation(
            design=design,
            sources=[],
            boundaries=[
                PML(thickness=vacuum_domain_small["wavelength"], formulation="cpml")
            ],
            time=np.array([0.0, dt], dtype=float),
            resolution=dx,
        )

        assert sim.pml_data["formulation"] == "cpml"
        for key in ("sigma_x", "sigma_y", "kappa_x", "kappa_y", "alpha_x", "alpha_y"):
            assert key in sim.pml_data

    def test_cpml_keeps_auxiliary_loss_out_of_material_updates(
        self, vacuum_domain_small
    ):
        design = vacuum_domain_small["design"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]
        wavelength = vacuum_domain_small["wavelength"]

        sim = Simulation(
            design=design,
            sources=[],
            boundaries=[PML(thickness=wavelength, formulation="cpml")],
            time=np.array([0.0, dt], dtype=float),
            resolution=dx,
        )

        sigma_shell = np.asarray(
            sim.pml_data["sigma_x"], dtype=np.float64
        ) + np.asarray(sim.pml_data["sigma_y"], dtype=np.float64)
        fields = sim.compile().grid
        total_sigma = np.asarray(fields.total_conductivity, dtype=np.float64)

        np.testing.assert_allclose(total_sigma, np.asarray(fields.conductivity))
        assert float(np.max(sigma_shell)) > 0.0
        assert float(np.max(total_sigma)) == pytest.approx(
            float(np.max(fields.conductivity))
        )

    def test_sponge_absorber_still_contributes_loss_in_material_updates(
        self, vacuum_domain_small
    ):
        design = vacuum_domain_small["design"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]
        wavelength = vacuum_domain_small["wavelength"]

        sim = Simulation(
            design=design,
            sources=[],
            boundaries=[PML(thickness=wavelength, formulation="sponge")],
            time=np.array([0.0, dt], dtype=float),
            resolution=dx,
        )

        sigma_shell = np.asarray(
            sim.pml_data["sigma_x"], dtype=np.float64
        ) + np.asarray(sim.pml_data["sigma_y"], dtype=np.float64)
        total_sigma = np.asarray(
            sim.compile().grid.total_conductivity, dtype=np.float64
        )

        assert float(np.max(sigma_shell)) > 0.0
        assert float(np.max(total_sigma)) >= float(np.max(sigma_shell))
        assert sim.pml_data["formulation"] == "sponge"

    def test_cpml_default_alpha_is_nonzero_when_omitted(self, vacuum_domain_small):
        design = vacuum_domain_small["design"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]
        wavelength = vacuum_domain_small["wavelength"]

        sim = Simulation(
            design=design,
            sources=[],
            boundaries=[PML(thickness=wavelength, formulation="cpml")],
            time=np.array([0.0, dt], dtype=float),
            resolution=dx,
        )

        alpha_x = np.asarray(sim.pml_data["alpha_x"], dtype=np.float64)
        alpha_y = np.asarray(sim.pml_data["alpha_y"], dtype=np.float64)

        assert sim.boundaries[0].alpha_max is None
        assert float(np.max(alpha_x)) > 0.0
        assert float(np.max(alpha_y)) > 0.0

    def test_cpml_default_sigma_uses_target_reflection_formula_2d(
        self, vacuum_domain_small
    ):
        design = vacuum_domain_small["design"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]
        wavelength = vacuum_domain_small["wavelength"]
        pml = PML(
            thickness=wavelength,
            formulation="cpml",
            target_reflection=1e-6,
            m=3,
        )

        sim = Simulation(
            design=design,
            sources=[],
            boundaries=[pml],
            time=np.array([0.0, dt], dtype=float),
            resolution=dx,
        )

        eta = np.sqrt(MU_0 / EPS_0)
        unscaled = -4.0 * np.log(1e-6) / (2.0 * eta * wavelength)
        assert pml.sigma_max is None
        assert float(np.max(np.asarray(sim.pml_data["sigma_x"], dtype=np.float64))) == pytest.approx(unscaled)  # fmt: skip

    def test_cpml_default_sigma_uses_target_reflection_formula_3d(self):
        dx = 0.1 * um
        dt = dx / (2.0 * LIGHT_SPEED)
        thickness = 12.0 * dx
        pml = PML(
            thickness=thickness,
            formulation="cpml",
            target_reflection=1e-6,
            m=3,
        )
        design = Design(
            width=3.0 * um,
            height=2.0 * um,
            depth=2.0 * um,
            material=Material(permittivity=1.0),
        )

        sim = Simulation(
            design=design,
            sources=[],
            boundaries=[pml],
            time=np.array([0.0, dt], dtype=float),
            resolution=dx,
        )

        eta = np.sqrt(MU_0 / EPS_0)
        unscaled = -4.0 * np.log(1e-6) / (2.0 * eta * thickness)
        assert pml.sigma_max is None
        assert float(np.max(np.asarray(sim.pml_data["sigma_x"], dtype=np.float64))) == pytest.approx(unscaled)  # fmt: skip

    def test_split_cpml_boundaries_preserve_identity_kappa_2d(
        self, vacuum_domain_small
    ):
        design = vacuum_domain_small["design"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]
        wavelength = vacuum_domain_small["wavelength"]

        sim = Simulation(
            design=design,
            sources=[],
            boundaries=[
                PML(edges=["left", "right"], thickness=wavelength, formulation="cpml"),
                PML(edges=["top", "bottom"], thickness=wavelength, formulation="cpml"),
            ],
            time=np.array([0.0, dt], dtype=float),
            resolution=dx,
        )

        fields = sim.compile().grid
        cy = fields.permittivity.shape[0] // 2
        cx = fields.permittivity.shape[1] // 2
        assert np.asarray(sim.pml_data["kappa_x"], dtype=np.float64)[
            cy, cx
        ] == pytest.approx(1.0)
        assert np.asarray(sim.pml_data["kappa_y"], dtype=np.float64)[
            cy, cx
        ] == pytest.approx(1.0)

        tm_xy = sim.pml_data["tm_xy_cpml"]
        assert np.asarray(tm_xy["Ez_x_kappa"], dtype=np.float64)[
            cy, cx
        ] == pytest.approx(1.0)
        assert np.asarray(tm_xy["Ez_y_kappa"], dtype=np.float64)[
            cy, cx
        ] == pytest.approx(1.0)

        plan = sim.compile().boundary.cpml
        assert len(plan.h_terms) == len(plan.e_terms) == 2
        assert all(
            not hasattr(term, "sigma") for term in (*plan.h_terms, *plan.e_terms)
        )

    def test_tm_xy_curl_h_to_e_updates_open_boundary_nodes(self):
        hx = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )
        hy = np.array(
            [
                [0.5, 1.5],
                [2.5, 3.5],
                [4.5, 5.5],
            ],
            dtype=np.float32,
        )

        curl = tm_xy_curl_h_to_e_2d(hx, hy, 1.0, (3, 3), frozenset())

        assert curl.shape == (3, 3)
        assert not np.allclose(np.asarray(curl[0, :]), 0.0)
        assert not np.allclose(np.asarray(curl[-1, :]), 0.0)
        assert not np.allclose(np.asarray(curl[:, 0]), 0.0)
        assert not np.allclose(np.asarray(curl[:, -1]), 0.0)

    @pytest.mark.slow
    @pytest.mark.parametrize("points_per_wavelength", [10, 11, 12])
    def test_cpml_homogeneous_vacuum_reflection_below_minus_40_db(
        self, points_per_wavelength, validation_metrics
    ):
        reflection_db = _homogeneous_cpml_reflection_db(
            points_per_wavelength=points_per_wavelength,
            refractive_index=1.0,
        )

        validation_metrics.check_upper(
            "time-gated CPML amplitude reflection",
            measured=reflection_db,
            upper_bound=-40.0,
            unit="dB",
            resolution=f"{points_per_wavelength} ppw in vacuum",
            metadata={"incidence": "normal", "refractive_index": 1.0},
        )

    @pytest.mark.slow
    def test_cpml_homogeneous_dielectric_reflection_below_minus_40_db(
        self, validation_metrics
    ):
        reflection_db = _homogeneous_cpml_reflection_db(
            points_per_wavelength=12,
            refractive_index=1.5,
        )

        validation_metrics.check_upper(
            "time-gated CPML amplitude reflection",
            measured=reflection_db,
            upper_bound=-40.0,
            unit="dB",
            resolution="12 ppw in n=1.5",
            metadata={"incidence": "normal", "refractive_index": 1.5},
        )

    @pytest.mark.slow
    def test_cpml_absorbs_slab_waveguide_mode_after_turnoff(
        self, waveguide_domain, validation_metrics
    ):
        design = waveguide_domain["design"]
        wavelength = waveguide_domain["wavelength"]
        dx = waveguide_domain["dx"]
        dt = waveguide_domain["dt"]
        domain_height = waveguide_domain["domain_height"]
        core_width = waveguide_domain["core_width"]

        frequency = LIGHT_SPEED / wavelength
        time = np.arange(0.0, 45.0 / frequency, dt)
        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=3.0 / frequency,
            t_max=15.0 / frequency,
        )

        source = ModeSource(
            center=(3.0 * wavelength, domain_height / 2.0, 0.0),
            size=(0.0, 3.0 * core_width, core_width),
            source_time=SampledSignal(signal, dt=dt, freq0=frequency),
            direction="+",
            mode_spec=ModeSpec(polarization="tm"),
        )
        sim = Simulation(
            design=design,
            sources=[source],
            boundaries=[PML(thickness=wavelength, formulation="cpml")],
            time=time,
            resolution=dx,
        )

        sim = sim.updated_copy(monitors=(*sim.monitors, FieldRecorder(("Ez",), 60)))
        result = sim.run()
        frames = [np.asarray(frame) for frame in result.monitor("fields").fields["Ez"]]
        energies = np.asarray([compute_field_energy(frame, dx) for frame in frames])
        peak_energy = float(np.max(energies))
        late_energy = float(np.mean(energies[-3:]))
        residual_db = 10.0 * np.log10(max(late_energy, 1e-300) / peak_energy)

        last_frame = frames[-1]
        source_ix = int(round(3.0 * wavelength / dx))
        upstream_energy = float(compute_field_energy(last_frame[:, :source_ix], dx))
        downstream_energy = float(compute_field_energy(last_frame[:, source_ix:], dx))
        upstream_fraction = upstream_energy / (upstream_energy + downstream_energy)

        validation_metrics.check_upper(
            "waveguide CPML residual energy",
            measured=residual_db,
            upper_bound=-25.0,
            unit="dB",
            resolution=f"{wavelength / dx:.1f} vacuum-wavelength ppw",
            metadata={"mode": "TM", "turnoff_periods": 15.0},
        )
        validation_metrics.check_upper(
            "late upstream energy fraction",
            measured=upstream_fraction,
            upper_bound=0.25,
            unit="fraction",
            resolution=f"{wavelength / dx:.1f} vacuum-wavelength ppw",
            metadata={"mode": "TM", "turnoff_periods": 15.0},
        )

    def test_late_field_energy_stays_below_coarse_bound(self, vacuum_domain_small):
        """Late electric-field energy should remain below 10% of its peak."""
        design = vacuum_domain_small["design"]
        wavelength = vacuum_domain_small["wavelength"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]

        frequency = LIGHT_SPEED / wavelength
        n_periods = 20
        t_total = n_periods / frequency
        time = np.arange(0, t_total, dt)

        # Short pulse so incident and reflected are temporally separated
        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total * 0.3,  # Source stops at 30%
        )

        # Source at center
        source = GaussianSource(
            position=(design.width / 2, design.height / 2),
            width=wavelength / 4,
            signal=signal,
        )

        # Thicker PML for better absorption
        pml_thickness = 1.5 * wavelength

        sim = Simulation(
            design=design,
            sources=[source],
            boundaries=[PML(thickness=pml_thickness)],
            time=time,
            resolution=dx,
        )

        sim = sim.updated_copy(monitors=(*sim.monitors, FieldRecorder(("Ez",), 20)))
        result = sim.run()

        # Compute energy at each snapshot
        energies = [
            compute_field_energy(Ez, dx) for Ez in result.monitor("fields").fields["Ez"]
        ]

        # Find peak energy (during excitation)
        peak_energy = max(energies)
        peak_idx = energies.index(peak_energy)

        # Check late-time energy (should be very small after absorption)
        late_idx = int(len(energies) * 0.9)
        assert peak_energy > 0.0, "PML reflection premise requires a nonzero pulse."
        assert late_idx > peak_idx, (
            "PML reflection premise failed: the field-energy peak overlaps the "
            "late-time reflection window."
        )
        late_energy = np.mean(energies[late_idx:])
        late_energy_fraction = late_energy / peak_energy

        assert late_energy_fraction < 0.10, (
            f"Late field energy {late_energy_fraction * 100:.1f}% exceeds 10% "
            "of the pulse peak."
        )

    def test_energy_decay_with_pml(self, vacuum_domain_small):
        """Energy should decay monotonically after source stops.

        Physics: With absorbing boundaries, EM energy leaves the domain
        and should decrease steadily.

        Tolerance: Energy ratio < 1.02 between consecutive measurements
        """
        design = vacuum_domain_small["design"]
        wavelength = vacuum_domain_small["wavelength"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]

        frequency = LIGHT_SPEED / wavelength
        n_periods = 15
        t_total = n_periods / frequency
        time = np.arange(0, t_total, dt)

        # Source stops early
        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total * 0.25,
        )

        source = GaussianSource(
            position=(design.width / 2, design.height / 2),
            width=wavelength / 4,
            signal=signal,
        )

        sim = Simulation(
            design=design,
            sources=[source],
            boundaries=[PML(thickness=wavelength)],
            time=time,
            resolution=dx,
        )

        sim = sim.updated_copy(monitors=(*sim.monitors, FieldRecorder(("Ez",), 10)))
        result = sim.run()

        energies = [
            compute_field_energy(Ez, dx) for Ez in result.monitor("fields").fields["Ez"]
        ]

        # After source stops (~35% accounting for ramp), check monotonic decay
        source_stop_idx = int(len(energies) * 0.4)
        post_source = energies[source_stop_idx:]

        # Allow small fluctuations (2%) but no sustained growth
        max_ratio = 1.02
        growth_count = 0
        for i in range(1, len(post_source)):
            if post_source[i - 1] > 1e-30:  # Skip near-zero
                ratio = post_source[i] / post_source[i - 1]
                if ratio > max_ratio:
                    growth_count += 1
                    assert growth_count < 3, (
                        f"Sustained energy growth detected: ratio={ratio:.3f} "
                        f"at step {source_stop_idx + i}"
                    )

    @pytest.mark.parametrize("pml_layers_wl", [0.5, 1.0, 1.5])
    def test_pml_absorption_stays_below_coarse_bound(
        self, vacuum_domain_small, pml_layers_wl
    ):
        """Each supported PML thickness should satisfy a coarse absorption bound."""
        design = vacuum_domain_small["design"]
        wavelength = vacuum_domain_small["wavelength"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]

        frequency = LIGHT_SPEED / wavelength
        t_total = 15 / frequency
        time = np.arange(0, t_total, dt)

        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total * 0.3,
        )

        source = GaussianSource(
            position=(design.width / 2, design.height / 2),
            width=wavelength / 4,
            signal=signal,
        )

        pml_thickness = pml_layers_wl * wavelength

        sim = Simulation(
            design=design,
            sources=[source],
            boundaries=[PML(thickness=pml_thickness)],
            time=time,
            resolution=dx,
        )

        sim = sim.updated_copy(monitors=(*sim.monitors, FieldRecorder(("Ez",), 20)))
        result = sim.run()

        energies = [
            compute_field_energy(Ez, dx) for Ez in result.monitor("fields").fields["Ez"]
        ]

        peak_energy = max(energies)
        assert peak_energy > 0.0, "PML stability premise requires a nonzero pulse."
        late_energy = np.mean(energies[-3:]) if len(energies) >= 3 else energies[-1]

        late_energy_fraction = late_energy / peak_energy
        assert late_energy_fraction < 0.20, (
            f"PML with {pml_layers_wl} wavelength thickness has "
            f"{late_energy_fraction * 100:.1f}% late field energy, exceeds 20%"
        )

    def test_pml_does_not_cause_instability(self, vacuum_domain_small):
        """PML should not cause numerical instability.

        Some PML implementations can be unstable, especially at corners
        or with certain parameter choices.
        """
        design = vacuum_domain_small["design"]
        wavelength = vacuum_domain_small["wavelength"]
        dx = vacuum_domain_small["dx"]
        dt = vacuum_domain_small["dt"]

        frequency = LIGHT_SPEED / wavelength
        t_total = 30 / frequency  # Long simulation to catch late instabilities
        time = np.arange(0, t_total, dt)

        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total * 0.2,
        )

        source = GaussianSource(
            position=(design.width / 2, design.height / 2),
            width=wavelength / 4,
            signal=signal,
        )

        sim = Simulation(
            design=design,
            sources=[source],
            boundaries=[PML(thickness=wavelength)],
            time=time,
            resolution=dx,
        )

        sim = sim.updated_copy(monitors=(*sim.monitors, FieldRecorder(("Ez",), 100)))
        result = sim.run()

        # Check for field explosion
        max_reasonable = 1e10
        for i, Ez in enumerate(result.monitor("fields").fields["Ez"]):
            max_field = np.max(np.abs(Ez))
            assert max_field < max_reasonable, (
                f"PML instability detected at snapshot {i}: max={max_field:.2e}"
            )

        # Check that energy eventually decays (not stuck at high level)
        energies = [
            compute_field_energy(Ez, dx) for Ez in result.monitor("fields").fields["Ez"]
        ]
        peak_energy = max(energies)
        assert peak_energy > 0.0, "PML stability premise requires a nonzero pulse."
        decay_ratio = energies[-1] / peak_energy
        assert decay_ratio < 0.5, (
            f"Energy not decaying with PML: final/peak = {decay_ratio:.2f}"
        )
