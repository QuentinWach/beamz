import matplotlib
import numpy as np
import pytest

from beamz import um
from beamz.design.library import gold, sio2, water
from beamz.visual.dispersion_validation import (
    plot_dispersion_validation,
    run_pulse_through_slab,
)

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def showcase_results():
    return {
        "sellmeier": run_pulse_through_slab(
            name="SiO2 Sellmeier Slab (test)",
            material=sio2(),
            wavelength_center_m=1.2 * um,
            slab_thickness_m=3.0 * um,
            domain_size_m=(18.0 * um, 6.0 * um),
            resolution_m=0.5 * um,
            num_steps=56,
            source_width_m=0.5 * um,
        ),
        "drude": run_pulse_through_slab(
            name="Gold Drude Slab (test)",
            material=gold(),
            wavelength_center_m=0.8 * um,
            slab_thickness_m=2.4 * um,
            domain_size_m=(18.0 * um, 6.0 * um),
            resolution_m=0.6 * um,
            num_steps=52,
            source_width_m=0.4 * um,
        ),
        "debye": run_pulse_through_slab(
            name="Water Debye Slab (test)",
            material=water(),
            wavelength_center_m=6.0e-4,
            slab_thickness_m=1.5e-3,
            domain_size_m=(6.0e-3, 2.0e-3),
            resolution_m=6.0e-5,
            num_steps=80,
            source_width_m=1.5e-4,
        ),
    }


def _passband(result):
    mask = result.passband_mask
    assert np.sum(mask) >= 6
    return mask


def test_drude_smoke_extraction_finite_and_loss_trend(showcase_results):
    result = showcase_results["drude"]
    mask = _passband(result)
    assert np.all(np.isfinite(result.n_extracted[mask]))
    assert np.all(np.isfinite(result.k_extracted[mask]))

    freq = result.frequency_hz[mask]
    k_ref = result.k_reference[mask]
    k_ext = result.k_extracted[mask]
    slope_ref = np.polyfit(freq, k_ref, 1)[0]
    slope_ext = np.polyfit(freq, k_ext, 1)[0]
    assert np.sign(slope_ext) == np.sign(slope_ref) or np.isclose(slope_ext, 0.0, atol=1e-12)


def test_sellmeier_match_in_central_passband(showcase_results):
    result = showcase_results["sellmeier"]
    mask = _passband(result)
    err_n = np.abs(result.n_extracted[mask] - result.n_reference[mask])
    assert float(np.mean(err_n)) < 0.35


def test_debye_eps_real_relaxation_trend(showcase_results):
    result = showcase_results["debye"]
    mask = _passband(result)
    freq = result.frequency_hz[mask]
    eps_r = np.real(result.epsilon_extracted[mask])
    order = np.argsort(freq)
    eps_sorted = eps_r[order]
    k = max(2, len(eps_sorted) // 5)
    low_mean = float(np.mean(eps_sorted[:k]))
    high_mean = float(np.mean(eps_sorted[-k:]))
    assert low_mean > high_mean


def test_plot_generation_headless(showcase_results):
    result = showcase_results["sellmeier"]
    fig, _ = plot_dispersion_validation(result, show=False, animate=False)
    assert fig is not None

