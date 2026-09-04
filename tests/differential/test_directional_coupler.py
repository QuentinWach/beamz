"""Differential validation of the passive-SOI directional coupler."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tests.differential.passive_soi.common import (
    expected_layer_fingerprints,
    generate_layout,
    layer_union_sha256,
    load_passive_soi_case,
    max_spectrum_difference,
    write_layout_gds,
)
from tests.differential.passive_soi.directional_coupler import (
    build_directional_coupler_simulation,
    paper_cross_power_range,
    run_directional_coupler_benchmark,
)


def _port_values(port):
    center = np.asarray(getattr(port, "dcenter", port.center), dtype=float)
    width = float(getattr(port, "dwidth", port.width))
    return center, width, float(port.orientation)


def test_generated_directional_coupler_matches_paper_geometry():
    case = load_passive_soi_case("directional_coupler")
    component = generate_layout(case)

    for layer, expected in expected_layer_fingerprints(case).items():
        assert layer_union_sha256(component, layer) == expected


def test_generated_directional_coupler_preserves_paper_ports():
    case = load_passive_soi_case("directional_coupler")
    component = generate_layout(case)
    expected_ports = case.geometry["ports"]

    assert {port.name for port in component.ports} == set(expected_ports)
    for port in component.ports:
        center, width, orientation = _port_values(port)
        expected = expected_ports[port.name]
        np.testing.assert_allclose(center, expected["center_um"], atol=1e-12, rtol=0.0)
        assert width == pytest.approx(expected["width_um"], abs=1e-12)
        assert orientation == pytest.approx(expected["orientation_deg"], abs=1e-12)


def test_directional_coupler_gds_is_generated_on_demand(tmp_path):
    case = load_passive_soi_case("directional_coupler")
    destination = tmp_path / "directional_coupler.gds"

    written = write_layout_gds(case, destination)

    assert written == destination
    assert destination.is_file()
    assert destination.stat().st_size > 0


def test_directional_coupler_simulation_uses_paper_stack_and_domain():
    case = load_passive_soi_case("directional_coupler")
    simulation, ports, frequencies = build_directional_coupler_simulation(
        resolution_ppw=6
    )
    wide_simulation, _, wide_frequencies = build_directional_coupler_simulation(
        resolution_ppw=6, wavelength_span_nm=50.0
    )
    high_resolution_simulation, _, _ = build_directional_coupler_simulation(
        resolution_ppw=20
    )

    np.testing.assert_allclose(
        (simulation.design.width, simulation.design.height, simulation.design.depth),
        np.asarray([42.0, 7.0, 4.0]) * 1e-6,
        rtol=0.0,
        atol=1e-15,
    )
    assert {port.name for port in ports} == {"o1", "o2", "o3", "o4"}
    assert {port.size[1] / 1e-6 for port in ports} == {3.0}
    assert frequencies.size == 5
    assert wide_frequencies.size == 11
    assert simulation.sources[0].mode_spec.num_freqs == 3
    assert wide_simulation.sources[0].mode_spec.num_freqs == 6
    np.testing.assert_allclose(
        np.diff(frequencies), np.diff(frequencies)[0], rtol=1e-12
    )
    np.testing.assert_allclose(
        np.diff(wide_frequencies), np.diff(wide_frequencies)[0], rtol=1e-12
    )
    assert not simulation.grid.is_uniform
    assert simulation.boundaries[0].formulation == "cpml"
    assert high_resolution_simulation.boundaries[0].formulation == "sponge"
    reference_z_min = case.geometry["simulation"]["domain_bounds_um"]["z"][0]
    assert {
        round(structure.z / 1e-6 + reference_z_min, 12)
        for structure in simulation.design.structures
    } == {0.0}


@pytest.mark.hardware
@pytest.mark.slow
@pytest.mark.parametrize(
    "resolution_ppw",
    [6, 10, 15, 20, 25],
    ids=lambda value: f"{value}ppw",
)
def test_directional_coupler_cross_power_agrees_with_published_solver_range(
    resolution_ppw, validation_metrics
):
    case = load_passive_soi_case("directional_coupler")
    artifact_root = os.environ.get("BEAMZ_VALIDATION_ARTIFACT_DIR")
    artifact_dir = (
        Path(artifact_root) / "directional_coupler" / f"{resolution_ppw}ppw"
        if artifact_root
        else None
    )
    result = run_directional_coupler_benchmark(
        resolution_ppw=resolution_ppw,
        progress=True,
        artifact_dir=artifact_dir,
    )
    lower, upper = paper_cross_power_range(case, resolution_ppw)
    metadata = {
        "execution_backend": result.backend,
        "published_lumerical_tidy3d_range": [lower, upper],
        "through_te0_power": result.through_power,
        "total_output_te0_power": result.total_output_power,
        "excess_loss": result.excess_loss,
        "runtime_s": result.runtime_s,
        "gcups": result.gcups,
        "cells": result.cells,
        "steps": result.steps,
        "grid_shape": result.grid_shape,
        "termination_reason": result.termination_reason,
        "wavelength_span_nm": result.wavelength_span_nm,
    }
    validation_metrics.check_lower(
        "directional coupler TE0 cross power at 1550 nm",
        measured=result.cross_power,
        lower_bound=lower,
        tolerance="cross_solver",
        unit="fraction",
        resolution=f"{resolution_ppw} cells per wavelength",
        backend="beamz-vs-published-lumerical-tidy3d-range",
        metadata=metadata,
    )
    validation_metrics.check_upper(
        "directional coupler TE0 cross power at 1550 nm",
        measured=result.cross_power,
        upper_bound=upper,
        tolerance="cross_solver",
        unit="fraction",
        resolution=f"{resolution_ppw} cells per wavelength",
        backend="beamz-vs-published-lumerical-tidy3d-range",
        metadata=metadata,
    )
    validation_metrics.check_upper(
        "directional coupler total output TE0 power at 1550 nm",
        measured=result.total_output_power,
        upper_bound=1.02,
        unit="fraction",
        resolution=f"{resolution_ppw} cells per wavelength",
        backend="beamz",
        metadata={
            "rationale": (
                "A passive device cannot create power; 2% allows modal projection "
                "and discretization error."
            )
        },
    )


@pytest.mark.hardware
@pytest.mark.slow
def test_directional_coupler_spectrum_is_consistent_across_source_bandwidths(
    validation_metrics,
):
    case = load_passive_soi_case("directional_coupler")
    protocol = case.geometry["simulation"]["published_bandwidth_consistency"]
    artifact_root = os.environ.get("BEAMZ_VALIDATION_ARTIFACT_DIR")
    results = []
    for span_nm in protocol["source_spans_nm"]:
        artifact_dir = (
            Path(artifact_root)
            / "directional_coupler"
            / "bandwidth_consistency"
            / f"{span_nm:g}nm"
            if artifact_root
            else None
        )
        results.append(
            run_directional_coupler_benchmark(
                resolution_ppw=int(protocol["resolution_ppw"]),
                wavelength_span_nm=float(span_nm),
                progress=True,
                artifact_dir=artifact_dir,
            )
        )

    difference = max_spectrum_difference(
        results[0].wavelengths_um,
        results[0].cross_power_spectrum,
        results[1].wavelengths_um,
        results[1].cross_power_spectrum,
    )
    validation_metrics.check_upper(
        "directional coupler maximum TE0 cross-power difference between source bandwidths",
        measured=difference,
        upper_bound=float(protocol["max_abs_difference"]),
        unit="fraction",
        resolution=f"{protocol['resolution_ppw']} cells per wavelength",
        backend="beamz-vs-published-bandwidth-consistency",
        metadata={
            "execution_backends": [result.backend for result in results],
            "source_spans_nm": protocol["source_spans_nm"],
            "wavelengths_um": [result.wavelengths_um for result in results],
            "cross_power_spectra": [result.cross_power_spectrum for result in results],
            "through_power_spectra": [
                result.through_power_spectrum for result in results
            ],
            "excess_loss_spectra": [result.excess_loss_spectrum for result in results],
            "published_value_basis": protocol["value_basis"],
        },
    )
