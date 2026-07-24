"""Slab-waveguide eigenmode validation against closed dispersion equations."""

from __future__ import annotations

import numpy as np
import pytest

from beamz import LIGHT_SPEED, um
from beamz.devices.sources.solve import solve_modes
from tests.utils import slab_waveguide_neff_te, slab_waveguide_neff_tm


@pytest.mark.simulation
@pytest.mark.parametrize(
    ("beamz_polarization", "analytical_solver", "analytical_label"),
    [
        # BeamZ labels polarization by its canonical 2-D field convention.
        ("tm", slab_waveguide_neff_te, "TE"),
        ("te", slab_waveguide_neff_tm, "TM"),
    ],
)
@pytest.mark.parametrize("grid_spacing_nm", [40, 25, 15])
def test_symmetric_slab_neff_matches_transcendental_solution(
    beamz_polarization,
    analytical_solver,
    analytical_label,
    grid_spacing_nm,
    validation_metrics,
):
    """Both slab polarizations meet a 0.2% neff gate over grid refinement."""
    wavelength = 1.55 * um
    core_width = 0.60 * um
    n_core = 2.04
    n_clad = 1.444
    transverse_span = 5.0 * um
    resolution = grid_spacing_nm * 1e-9
    transverse_cells = int(round(transverse_span / resolution))
    core_cells = int(round(core_width / resolution))
    core_start = (transverse_cells - core_cells) // 2
    permittivity = np.full(transverse_cells, n_clad**2, dtype=float)
    permittivity[core_start : core_start + core_cells] = n_core**2

    neffs, _ = solve_modes(
        permittivity,
        omega=2.0 * np.pi * LIGHT_SPEED / wavelength,
        dL=resolution,
        m=1,
        direction="+x",
        filter_pol=beamz_polarization,
        return_fields=False,
        target_neff=0.95 * n_core,
    )
    measured_neff = float(np.real(neffs[0]))
    analytical_neff = analytical_solver(
        n_core,
        n_clad,
        core_width,
        wavelength,
        mode=0,
    )
    assert analytical_neff is not None
    assert n_clad < measured_neff < n_core

    validation_metrics.check(
        f"symmetric slab {analytical_label}0 effective index",
        measured=measured_neff,
        reference=analytical_neff,
        tolerance="waveguide_neff",
        resolution=f"{grid_spacing_nm} nm transverse grid",
        metadata={
            "beamz_polarization": beamz_polarization,
            "analytical_polarization": analytical_label,
            "n_core": n_core,
            "n_clad": n_clad,
            "core_width_um": core_width / um,
            "wavelength_um": wavelength / um,
            "transverse_cells": transverse_cells,
        },
    )
