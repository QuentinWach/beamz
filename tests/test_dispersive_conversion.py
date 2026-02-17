import numpy as np

from beamz.design.library import gold, silicon, sio2, water
from beamz.design.materials import (
    DebyeMaterial,
    DrudeMaterial,
    LorentzMaterial,
    PoleResidueMaterial,
    SellmeierMaterial,
)
from beamz.simulation.dispersion import evaluate_epsilon_from_poles


def _assert_model_matches_poles(model, freqs, rtol=2e-6, atol=2e-6):
    spec = model.to_canonical_poles()
    eps_ref = np.asarray(model.epsilon(frequency=freqs))
    eps_poles = evaluate_epsilon_from_poles(freqs, spec.eps_inf, spec.poles)
    assert np.all(np.isfinite(eps_poles))
    assert np.allclose(eps_ref, eps_poles, rtol=rtol, atol=atol)


def test_sellmeier_exact_mapping_matches_model():
    model = sio2()
    assert isinstance(model, SellmeierMaterial)
    freqs = np.linspace(1.2e14, 4.0e14, 31)
    _assert_model_matches_poles(model, freqs, rtol=2e-5, atol=2e-5)


def test_drude_mapping_matches_model():
    model = gold()
    assert isinstance(model, DrudeMaterial)
    freqs = np.linspace(5.0e13, 8.0e14, 35)
    _assert_model_matches_poles(model, freqs, rtol=2e-5, atol=2e-5)


def test_lorentz_mapping_matches_model():
    model = silicon()
    assert isinstance(model, LorentzMaterial)
    freqs = np.linspace(1.0e13, 8.0e14, 41)
    _assert_model_matches_poles(model, freqs, rtol=2e-5, atol=2e-5)


def test_debye_mapping_matches_model():
    model = water()
    assert isinstance(model, DebyeMaterial)
    freqs = np.linspace(1.0e8, 4.0e11, 41)
    _assert_model_matches_poles(model, freqs, rtol=2e-5, atol=2e-5)


def test_pole_residue_validation_and_mapping():
    model = PoleResidueMaterial(
        eps_inf=1.5,
        poles=((1.2e14 + 2.5e14j, 3.1e12 - 2.0e12j),),
    )
    spec = model.to_canonical_poles()
    assert spec.eps_inf == 1.5
    assert len(spec.poles) == 2
    freqs = np.linspace(2.0e13, 5.0e14, 31)
    _assert_model_matches_poles(model, freqs, rtol=2e-5, atol=2e-5)
