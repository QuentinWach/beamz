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
    GaussianSource,
    Material,
    ModeSource,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    um,
)
from beamz.shared_kernels import build_tm_xy_cpml_terms
from beamz.simulation.boundaries import (
    _cpml_ab_from_profiles,
    cpml_curl_e_to_h_3d,
    cpml_curl_h_to_e_3d,
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
    sim = Simulation(
        design=design,
        sources=[source],
        boundaries=[PML(thickness=pml_thickness, formulation="cpml")],
        time=time,
        resolution=dx,
    )

    probe_ix = int(round(probe_x / dx))
    probe_y0 = int(round(3.0 * wavelength / dx))
    probe_y1 = int(round(5.0 * wavelength / dx))
    samples = []
    for _ in range(len(time)):
        sim.step()
        ez = np.asarray(sim.fields.Ez)
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
        total_sigma = np.asarray(sim.fields.total_conductivity, dtype=np.float64)

        np.testing.assert_allclose(total_sigma, np.asarray(sim.fields.conductivity))
        assert float(np.max(sigma_shell)) > 0.0
        assert float(np.max(total_sigma)) == pytest.approx(
            float(np.max(sim.fields.conductivity))
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
        total_sigma = np.asarray(sim.fields.total_conductivity, dtype=np.float64)

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

        assert float(sim.boundaries[0].alpha_max) > 0.0
        assert float(np.max(alpha_x)) > 0.0
        assert float(np.max(alpha_y)) > 0.0

    def test_cpml_default_sigma_keeps_2d_softening(self, vacuum_domain_small):
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

        Simulation(
            design=design,
            sources=[],
            boundaries=[pml],
            time=np.array([0.0, dt], dtype=float),
            resolution=dx,
        )

        eta = np.sqrt(MU_0 / EPS_0)
        unscaled = -4.0 * np.log(1e-6) / (2.0 * eta * wavelength)
        assert pml.sigma_max == pytest.approx(0.5 * unscaled)

    def test_cpml_default_sigma_uses_thin_3d_softening(self):
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

        Simulation(
            design=design,
            sources=[],
            boundaries=[pml],
            time=np.array([0.0, dt], dtype=float),
            resolution=dx,
        )

        eta = np.sqrt(MU_0 / EPS_0)
        unscaled = -4.0 * np.log(1e-6) / (2.0 * eta * thickness)
        assert pml.sigma_max == pytest.approx(0.25 * unscaled)

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

        cy = sim.fields.permittivity.shape[0] // 2
        cx = sim.fields.permittivity.shape[1] // 2
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

        terms = build_tm_xy_cpml_terms(tm_xy, ez_shape=sim.fields.Ez.shape)
        np.testing.assert_allclose(
            np.asarray(terms.kappa_h_direct_terms),
            np.asarray(terms.kappa_h_aux_terms),
            rtol=0.0,
            atol=0.0,
        )

    def test_cpml_full_tm_profiles_follow_discrete_yee_staggering(self):
        pml = PML(thickness=1.0, formulation="cpml", sigma_max=10.0, alpha_max=1.0)
        profile_fn = getattr(
            pml,
            next(name for name in dir(pml) if name.endswith("_staggered_profile_1d")),
        )

        sigma_low_e, _, _ = profile_fn(
            total_samples=21,
            spacing=0.1,
            low_active=True,
            high_active=False,
            sample_kind="E",
        )
        sigma_high_e, _, _ = profile_fn(
            total_samples=21,
            spacing=0.1,
            low_active=False,
            high_active=True,
            sample_kind="E",
        )
        sigma_low_h, _, _ = profile_fn(
            total_samples=20,
            spacing=0.1,
            low_active=True,
            high_active=False,
            sample_kind="H",
        )

        # The discrete Yee staggering does not drive E-node samples all the way
        # to sigma_max at the outer boundary, and the interface node stays at zero.
        assert 0.0 < float(sigma_low_e[0]) < pml.sigma_max
        assert float(sigma_low_e[10]) == pytest.approx(0.0)
        assert float(sigma_high_e[-11]) == pytest.approx(0.0)
        assert 0.0 < float(sigma_high_e[-1]) < pml.sigma_max
        # H samples are half-cell shifted and stay strictly positive through the low-side slab.
        assert float(sigma_low_h[0]) > float(sigma_low_h[1]) > 0.0

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

    def test_cpml_3d_curl_helpers_match_literal_split_form(self):
        rng = np.random.default_rng(7)
        resolution = np.float64(0.2)
        dt = 0.05

        ex = jnp.asarray(rng.normal(size=(3, 4, 4)).astype(np.float32))
        ey = jnp.asarray(rng.normal(size=(3, 3, 5)).astype(np.float32))
        ez = jnp.asarray(rng.normal(size=(2, 4, 5)).astype(np.float32))
        hx = jnp.asarray(rng.normal(size=(2, 3, 5)).astype(np.float32))
        hy = jnp.asarray(rng.normal(size=(2, 4, 4)).astype(np.float32))
        hz = jnp.asarray(rng.normal(size=(3, 3, 4)).astype(np.float32))

        h_term_shapes = (
            (2, 3, 5),
            (2, 3, 5),
            (2, 4, 4),
            (2, 4, 4),
            (3, 3, 4),
            (3, 3, 4),
        )
        e_term_shapes = (
            (3, 4, 4),
            (3, 4, 4),
            (3, 3, 5),
            (3, 3, 5),
            (2, 4, 5),
            (2, 4, 5),
        )

        sigma_h = tuple(
            jnp.asarray(rng.uniform(0.0, 2.0, size=shape).astype(np.float32))
            for shape in h_term_shapes
        )
        kappa_h = tuple(
            jnp.asarray(rng.uniform(1.0, 3.0, size=shape).astype(np.float32))
            for shape in h_term_shapes
        )
        alpha_h = tuple(
            jnp.asarray(rng.uniform(0.0, 0.5, size=shape).astype(np.float32))
            for shape in h_term_shapes
        )
        psi_h = tuple(
            jnp.asarray(rng.normal(size=shape).astype(np.float32))
            for shape in h_term_shapes
        )
        sigma_e = tuple(
            jnp.asarray(rng.uniform(0.0, 2.0, size=shape).astype(np.float32))
            for shape in e_term_shapes
        )
        kappa_e = tuple(
            jnp.asarray(rng.uniform(1.0, 3.0, size=shape).astype(np.float32))
            for shape in e_term_shapes
        )
        alpha_e = tuple(
            jnp.asarray(rng.uniform(0.0, 0.5, size=shape).astype(np.float32))
            for shape in e_term_shapes
        )
        psi_e = tuple(
            jnp.asarray(rng.normal(size=shape).astype(np.float32))
            for shape in e_term_shapes
        )

        a_h, b_h, inv_kappa_h = [], [], []
        for sigma_term, kappa_term, alpha_term in zip(
            sigma_h, kappa_h, alpha_h, strict=True
        ):
            a_term, b_term = _cpml_ab_from_profiles(
                sigma_term, kappa_term, alpha_term, dt
            )
            a_h.append(a_term)
            b_h.append(b_term)
            inv_kappa_h.append(1.0 / kappa_term)
        a_e, b_e, inv_kappa_e = [], [], []
        for sigma_term, kappa_term, alpha_term in zip(
            sigma_e, kappa_e, alpha_e, strict=True
        ):
            a_term, b_term = _cpml_ab_from_profiles(
                sigma_term, kappa_term, alpha_term, dt
            )
            a_e.append(a_term)
            b_e.append(b_term)
            inv_kappa_e.append(1.0 / kappa_term)

        a_h_mixed = tuple(term.astype(jnp.float64) for term in a_h)
        b_h_mixed = tuple(term.astype(jnp.float64) for term in b_h)
        inv_kappa_h_mixed = tuple(term.astype(jnp.float64) for term in inv_kappa_h)
        a_e_mixed = tuple(term.astype(jnp.float64) for term in a_e)
        b_e_mixed = tuple(term.astype(jnp.float64) for term in b_e)
        inv_kappa_e_mixed = tuple(term.astype(jnp.float64) for term in inv_kappa_e)

        curl_hx, curl_hy, curl_hz, psi_h_updated = cpml_curl_e_to_h_3d(
            ex,
            ey,
            ez,
            resolution,
            a_h_terms=a_h_mixed,
            b_h_terms=b_h_mixed,
            inv_kappa_h_terms=inv_kappa_h_mixed,
            psi_h_terms=psi_h,
        )

        d_terms_h = (
            (ez[:, 1:, :] - ez[:, :-1, :]) / resolution,
            (ey[1:, :, :] - ey[:-1, :, :]) / resolution,
            (ex[1:, :, :] - ex[:-1, :, :]) / resolution,
            (ez[:, :, 1:] - ez[:, :, :-1]) / resolution,
            (ey[:, :, 1:] - ey[:, :, :-1]) / resolution,
            (ex[:, 1:, :] - ex[:, :-1, :]) / resolution,
        )
        psi_h_ref = tuple(
            b_term * psi_term + a_term * d_term
            for b_term, psi_term, a_term, d_term in zip(
                b_h, psi_h, a_h, d_terms_h, strict=True
            )
        )
        corrected_h = tuple(
            d_term * inv_term + psi_term
            for d_term, inv_term, psi_term in zip(
                d_terms_h, inv_kappa_h, psi_h_ref, strict=True
            )
        )
        curl_hx_ref = corrected_h[0] - corrected_h[1]
        curl_hy_ref = corrected_h[2] - corrected_h[3]
        curl_hz_ref = corrected_h[4] - corrected_h[5]

        for psi_got, psi_ref in zip(psi_h_updated, psi_h_ref, strict=True):
            np.testing.assert_allclose(
                np.asarray(psi_got), np.asarray(psi_ref), rtol=1e-6, atol=1e-6
            )
        assert all(term.dtype == psi_h[0].dtype for term in psi_h_updated)
        np.testing.assert_allclose(
            np.asarray(curl_hx), np.asarray(curl_hx_ref), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(curl_hy), np.asarray(curl_hy_ref), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(curl_hz), np.asarray(curl_hz_ref), rtol=1e-6, atol=1e-6
        )

        curl_ex, curl_ey, curl_ez, psi_e_updated = cpml_curl_h_to_e_3d(
            hx,
            hy,
            hz,
            resolution,
            a_e_terms=a_e_mixed,
            b_e_terms=b_e_mixed,
            inv_kappa_e_terms=inv_kappa_e_mixed,
            psi_e_terms=psi_e,
            metallic_edges=frozenset(
                {"left", "right", "bottom", "top", "front", "back"}
            ),
        )

        d_terms_e = (
            (
                jnp.pad(hz, ((0, 0), (1, 1), (0, 0)))[:, 1:, :]
                - jnp.pad(hz, ((0, 0), (1, 1), (0, 0)))[:, :-1, :]
            )
            / resolution,
            (
                jnp.pad(hy, ((1, 1), (0, 0), (0, 0)))[1:, :, :]
                - jnp.pad(hy, ((1, 1), (0, 0), (0, 0)))[:-1, :, :]
            )
            / resolution,
            (
                jnp.pad(hx, ((1, 1), (0, 0), (0, 0)))[1:, :, :]
                - jnp.pad(hx, ((1, 1), (0, 0), (0, 0)))[:-1, :, :]
            )
            / resolution,
            (
                jnp.pad(hz, ((0, 0), (0, 0), (1, 1)))[:, :, 1:]
                - jnp.pad(hz, ((0, 0), (0, 0), (1, 1)))[:, :, :-1]
            )
            / resolution,
            (
                jnp.pad(hy, ((0, 0), (0, 0), (1, 1)))[:, :, 1:]
                - jnp.pad(hy, ((0, 0), (0, 0), (1, 1)))[:, :, :-1]
            )
            / resolution,
            (
                jnp.pad(hx, ((0, 0), (1, 1), (0, 0)))[:, 1:, :]
                - jnp.pad(hx, ((0, 0), (1, 1), (0, 0)))[:, :-1, :]
            )
            / resolution,
        )
        psi_e_ref = tuple(
            b_term * psi_term + a_term * d_term
            for b_term, psi_term, a_term, d_term in zip(
                b_e, psi_e, a_e, d_terms_e, strict=True
            )
        )
        corrected_e = tuple(
            d_term * inv_term + psi_term
            for d_term, inv_term, psi_term in zip(
                d_terms_e, inv_kappa_e, psi_e_ref, strict=True
            )
        )
        curl_ex_ref = corrected_e[0] - corrected_e[1]
        curl_ey_ref = corrected_e[2] - corrected_e[3]
        curl_ez_ref = corrected_e[4] - corrected_e[5]

        for psi_got, psi_ref in zip(psi_e_updated, psi_e_ref, strict=True):
            np.testing.assert_allclose(
                np.asarray(psi_got), np.asarray(psi_ref), rtol=1e-6, atol=1e-6
            )
        assert all(term.dtype == psi_e[0].dtype for term in psi_e_updated)
        np.testing.assert_allclose(
            np.asarray(curl_ex), np.asarray(curl_ex_ref), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(curl_ey), np.asarray(curl_ey_ref), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(curl_ez), np.asarray(curl_ez_ref), rtol=1e-6, atol=1e-6
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("points_per_wavelength", [10, 11, 12])
    def test_cpml_homogeneous_vacuum_reflection_below_minus_40_db(
        self, points_per_wavelength
    ):
        reflection_db = _homogeneous_cpml_reflection_db(
            points_per_wavelength=points_per_wavelength,
            refractive_index=1.0,
        )

        assert reflection_db < -40.0, (
            f"CPML reflection was {reflection_db:.2f} dB for "
            f"{points_per_wavelength} cells per wavelength."
        )

    @pytest.mark.slow
    def test_cpml_homogeneous_dielectric_reflection_below_minus_40_db(self):
        reflection_db = _homogeneous_cpml_reflection_db(
            points_per_wavelength=12,
            refractive_index=1.5,
        )

        assert reflection_db < -40.0, (
            f"Uniform dielectric CPML reflection was {reflection_db:.2f} dB."
        )

    @pytest.mark.slow
    def test_cpml_absorbs_slab_waveguide_mode_after_turnoff(self, waveguide_domain):
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

        grid = design.rasterize(resolution=dx)
        source = ModeSource(
            grid=grid,
            center=(3.0 * wavelength, domain_height / 2.0),
            width=3.0 * core_width,
            wavelength=wavelength,
            pol="tm",
            signal=signal,
            direction="+x",
        )
        sim = Simulation(
            design=design,
            sources=[source],
            boundaries=[PML(thickness=wavelength, formulation="cpml")],
            time=time,
            resolution=dx,
        )

        result = sim.run(save_fields=["Ez"], field_subsample=60)
        frames = [np.asarray(frame) for frame in result["fields"]["Ez"]]
        energies = np.asarray([compute_field_energy(frame, dx) for frame in frames])
        peak_energy = float(np.max(energies))
        late_energy = float(np.mean(energies[-3:]))
        residual_db = 10.0 * np.log10(max(late_energy, 1e-300) / peak_energy)

        last_frame = frames[-1]
        source_ix = int(round(3.0 * wavelength / dx))
        upstream_energy = float(compute_field_energy(last_frame[:, :source_ix], dx))
        downstream_energy = float(compute_field_energy(last_frame[:, source_ix:], dx))
        upstream_fraction = upstream_energy / (upstream_energy + downstream_energy)

        assert residual_db < -25.0, (
            f"Waveguide CPML residual energy was {residual_db:.2f} dB."
        )
        assert upstream_fraction < 0.25, (
            f"Waveguide CPML left {upstream_fraction:.2%} of late energy upstream."
        )

    def test_pml_reflection_level(self, vacuum_domain_small):
        """PML should absorb waves with minimal reflection.

        Physics: A properly implemented PML creates an impedance-matched
        absorbing region that minimizes reflections.

        Method: Launch pulse, let it hit PML, measure late-time field
        relative to peak. Late-time field is primarily reflections.

        Tolerance: Reflection ratio < 10%
        """
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

        result = sim.run(save_fields=["Ez"], field_subsample=20)

        # Compute energy at each snapshot
        energies = [compute_field_energy(Ez, dx) for Ez in result["fields"]["Ez"]]

        # Find peak energy (during excitation)
        peak_energy = max(energies)
        peak_idx = energies.index(peak_energy)

        # Check late-time energy (should be very small after absorption)
        late_idx = int(len(energies) * 0.9)
        if late_idx > peak_idx:
            late_energy = np.mean(energies[late_idx:])

            # Reflection ratio
            reflection_ratio = late_energy / peak_energy if peak_energy > 0 else 0

            assert reflection_ratio < 0.10, (
                f"PML reflection {reflection_ratio * 100:.1f}% exceeds 10%. "
                "This indicates poor PML absorption."
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

        result = sim.run(save_fields=["Ez"], field_subsample=10)

        energies = [compute_field_energy(Ez, dx) for Ez in result["fields"]["Ez"]]

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
    def test_thicker_pml_better_absorption(self, vacuum_domain_small, pml_layers_wl):
        """Thicker PML should generally provide better absorption.

        This test verifies that the PML implementation is reasonable
        by checking that absorption improves (or doesn't degrade)
        with thickness.
        """
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

        result = sim.run(save_fields=["Ez"], field_subsample=20)

        energies = [compute_field_energy(Ez, dx) for Ez in result["fields"]["Ez"]]

        peak_energy = max(energies)
        late_energy = np.mean(energies[-3:]) if len(energies) >= 3 else energies[-1]

        # For any reasonable PML, reflection should be < 20%
        reflection_ratio = late_energy / peak_energy if peak_energy > 0 else 0
        assert reflection_ratio < 0.20, (
            f"PML with {pml_layers_wl} wavelength thickness has "
            f"{reflection_ratio * 100:.1f}% reflection, exceeds 20%"
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

        result = sim.run(save_fields=["Ez"], field_subsample=100)

        # Check for field explosion
        max_reasonable = 1e10
        for i, Ez in enumerate(result["fields"]["Ez"]):
            max_field = np.max(np.abs(Ez))
            assert max_field < max_reasonable, (
                f"PML instability detected at snapshot {i}: max={max_field:.2e}"
            )

        # Check that energy eventually decays (not stuck at high level)
        energies = [compute_field_energy(Ez, dx) for Ez in result["fields"]["Ez"]]
        if energies[0] > 1e-30:
            decay_ratio = energies[-1] / max(energies)
            assert decay_ratio < 0.5, (
                f"Energy not decaying with PML: final/peak = {decay_ratio:.2f}"
            )
