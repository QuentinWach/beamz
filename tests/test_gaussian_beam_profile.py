import numpy as np
import pytest

from beamz.devices.sources._gaussian_beam import GaussianBeamProfile
from beamz.devices.sources._profiles import FieldProfile3D


def test_gaussian_beam_profile_has_transverse_gaussian_envelope():
    beam = GaussianBeamProfile(
        center=(10.0, 10.0, 10.0),
        size=(6.0, 6.0),
        direction="+x",
        angle_theta=0.0,
        angle_phi=0.0,
        pol_angle=0.0,
        waist_radius=1.0,
        waist_distance=0.0,
        wavelength=5.0,
    )

    profile = beam.field_profile(resolution=0.25, grid_shape=(80, 80, 80))

    assert isinstance(profile, FieldProfile3D)
    assert profile.axis == "x"
    assert profile.direction_sign == 1.0
    ey = np.abs(profile.components["Ey"])
    assert ey.shape[0] > 8 and ey.shape[1] > 8
    center_value = float(np.max(ey))
    edge_value = float(max(np.max(ey[0, :]), np.max(ey[-1, :])))
    assert center_value > 0.0
    assert edge_value < 0.02 * center_value


def test_gaussian_beam_profile_vectors_are_orthogonal_to_each_other_and_k():
    beam = GaussianBeamProfile(
        center=(8.0, 8.0, 8.0),
        size=(5.0, 5.0),
        direction="+z",
        angle_theta=0.31,
        angle_phi=0.47,
        pol_angle=0.62,
        waist_radius=1.4,
        waist_distance=0.3,
        wavelength=3.0,
        background_index=1.5,
    )

    k_hat = beam.propagation_unit_vector()
    e_hat = beam.electric_unit_vector()
    h_hat = beam.magnetic_unit_vector()

    np.testing.assert_allclose(np.linalg.norm(k_hat), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(e_hat), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(h_hat), 1.0, atol=1e-12)
    assert abs(float(np.dot(k_hat, e_hat))) < 1e-12
    assert abs(float(np.dot(k_hat, h_hat))) < 1e-12
    assert abs(float(np.dot(e_hat, h_hat))) < 1e-12
    assert float(np.dot(np.cross(e_hat, h_hat), k_hat)) == pytest.approx(1.0)

    profile = beam.field_profile(resolution=0.25, grid_shape=(64, 64, 64))
    assert profile.k_axis == pytest.approx(beam.propagation_vector()[2])


def test_gaussian_beam_profile_transverse_phase_matches_tilt_direction():
    beam = GaussianBeamProfile(
        center=(10.0, 10.0, 10.0),
        size=(7.0, 7.0),
        direction="+z",
        angle_theta=0.24,
        angle_phi=0.0,
        pol_angle=0.0,
        waist_radius=3.0,
        waist_distance=0.0,
        wavelength=10.0,
    )
    profile = beam.field_profile(resolution=0.25, grid_shape=(80, 80, 80))

    ex = profile.components["Ex"]
    y_mid = ex.shape[0] // 2
    row = ex[y_mid, :]
    x_slice = profile.indices["Ex"][2]
    x_coords = (np.arange(x_slice.start, x_slice.stop) + 0.5) * 0.25
    phase = np.unwrap(np.angle(row))
    slope, _intercept = np.polyfit(x_coords, phase, deg=1)

    assert slope == pytest.approx(-beam.propagation_vector()[0], rel=2e-2, abs=2e-3)


def test_gaussian_beam_profile_power_normalization_is_sane():
    beam = GaussianBeamProfile(
        center=(10.0, 10.0, 10.0),
        size=(8.0, 8.0),
        direction="+x",
        angle_theta=0.0,
        angle_phi=0.0,
        pol_angle=0.0,
        waist_radius=1.5,
        waist_distance=0.0,
        wavelength=5.0,
        power=2.0,
    )
    resolution = 0.2
    profile = beam.field_profile(resolution=resolution, grid_shape=(100, 100, 100))

    ey = profile.components["Ey"]
    ez = profile.components["Ez"]
    hy = profile.components["Hy"]
    hz = profile.components["Hz"]
    flux = 0.5 * np.real(np.sum(ey * np.conjugate(hz) - ez * np.conjugate(hy)))
    power = float(flux * resolution**2)

    assert np.isfinite(power)
    assert power == pytest.approx(2.0, rel=0.15)
