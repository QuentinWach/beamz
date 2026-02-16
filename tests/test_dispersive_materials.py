import numpy as np
import pytest

from beamz.design.materials import (
    BK7_Sellmeier,
    SiO2_Sellmeier,
    Aluminum_Drude,
    Copper_Drude,
    DebyeMaterial,
    DrudeLorentzMaterial,
    DrudeMaterial,
    Gold_Drude,
    Gold_DrudeLorentz,
    LorentzMaterial,
    PoleResidueMaterial,
    Silver_Drude,
    Water_Debye,
)


def test_sellmeier_indices_and_group_index_dispersion():
    wl = 1.55e-6

    n_sio2 = np.real(SiO2_Sellmeier.n_complex(wavelength=wl)).reshape(())
    n_bk7 = np.real(BK7_Sellmeier.n_complex(wavelength=wl)).reshape(())

    assert n_sio2 == pytest.approx(1.444, rel=0.02)
    assert n_bk7 == pytest.approx(1.50, rel=0.03)

    ng = SiO2_Sellmeier.group_index(wavelength=wl)
    disp = SiO2_Sellmeier.dispersion_ps_nm_km(wavelength=wl)
    assert ng > 1.0
    assert np.isfinite(disp)


@pytest.mark.parametrize(
    "model", [Gold_Drude, Silver_Drude, Aluminum_Drude, Copper_Drude]
)
def test_drude_models_have_expected_sign_and_skin_depth(model):
    wl = 1.55e-6
    eps = model.epsilon(wavelength=wl).reshape(())

    assert np.real(eps) < 0.0
    assert np.imag(eps) >= 0.0

    depth = model.skin_depth(wavelength=wl)
    assert depth > 0.0


def test_lorentz_resonance_shows_absorption_peak():
    c = 299792458.0
    w0 = 2.0 * np.pi * c / 1.0e-6

    model = LorentzMaterial(
        name="TestLorentz",
        eps_inf=2.0,
        resonances=[w0],
        strengths=[1.0],
        dampings=[0.05 * w0],
        plasma_frequencies=[w0],
    )

    eps_on = model.epsilon(wavelength=1.0e-6).reshape(())
    eps_off = model.epsilon(wavelength=1.6e-6).reshape(())

    assert np.imag(eps_on) > np.imag(eps_off)


def test_debye_water_relaxation_trend():
    eps_low = Water_Debye.epsilon(frequency=1e9).reshape(())
    eps_high = Water_Debye.epsilon(frequency=1e11).reshape(())

    assert np.real(eps_low) > np.real(eps_high)


def test_pole_residue_is_vectorized_and_finite():
    model = PoleResidueMaterial(
        name="TestPoleResidue",
        eps_inf=1.5,
        poles=[-1e14 + 2j * np.pi * 2e14, -2e14 + 2j * np.pi * 4e14],
        residues=[1e13 + 1e12j, 5e12 + 2e12j],
    )

    wl = np.array([1.3e-6, 1.55e-6, 2.0e-6])
    eps = model.epsilon(wavelength=wl)

    assert eps.shape == wl.shape
    assert np.all(np.isfinite(np.real(eps)))
    assert np.all(np.isfinite(np.imag(eps)))


def test_drude_lorentz_model_sanity():
    eps = Gold_DrudeLorentz.epsilon(wavelength=1.55e-6).reshape(())
    assert np.isfinite(np.real(eps))
    assert np.isfinite(np.imag(eps))
    assert np.real(eps) < 0.0


def test_to_material_requires_explicit_operating_point():
    with pytest.raises(ValueError):
        Gold_Drude.to_material()


def test_to_material_returns_simple_material():
    mat = Gold_Drude.to_material(wavelength=1.55e-6)
    assert np.isfinite(mat.permittivity)
    assert mat.conductivity >= 0.0


def test_get_sample_on_dispersive_models_fails_fast():
    with pytest.raises(ValueError, match="to_material"):
        SiO2_Sellmeier.get_sample()


def test_generic_debye_and_drude_lorentz_instantiation():
    custom_debye = DebyeMaterial(
        name="CustomDebye",
        eps_inf=2.5,
        debye_strengths=[10.0, 5.0],
        relaxation_times=[1e-10, 1e-12],
        sigma_dc=0.01,
    )
    eps = custom_debye.epsilon(frequency=1e10).reshape(())
    assert np.isfinite(np.real(eps))

    custom_dl = DrudeLorentzMaterial(
        name="CustomDL",
        eps_inf=1.0,
        drude_plasma_frequency=1e16,
        drude_damping=1e14,
        lorentz_resonances=[5e15],
        lorentz_strengths=[0.5],
        lorentz_dampings=[2e14],
        lorentz_plasma_frequencies=[1e16],
    )
    eps_dl = custom_dl.epsilon(wavelength=1.55e-6).reshape(())
    assert np.isfinite(np.real(eps_dl))


def test_drude_material_constructor_roundtrip():
    model = DrudeMaterial(
        name="SimpleDrude", eps_inf=3.0, plasma_frequency=1e16, damping=1e14
    )
    eps = model.epsilon(wavelength=1.55e-6).reshape(())
    assert np.isfinite(np.real(eps))
