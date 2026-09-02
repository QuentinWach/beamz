"""Geometry, simulation, and observables for the passive-SOI crossing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tests.differential.case_schema import DifferentialCase
from tests.differential.passive_soi.common import (
    domain_bounds_um,
    domain_size_um,
    generate_layout,
    load_passive_soi_case,
    port_center_and_direction,
    reference_absorber_warning_scope,
    reference_frequencies,
)


@dataclass(frozen=True)
class CrossingBenchmarkResult:
    """Paper-comparable crossing observables and execution metadata."""

    resolution_ppw: int
    wavelength_span_nm: float
    backend: str
    wavelengths_um: tuple[float, ...]
    through_power_spectrum: tuple[float, ...]
    total_output_power_spectrum: tuple[float, ...]
    excess_loss_spectrum: tuple[float, ...]
    through_power: float
    total_output_power: float
    excess_loss: float
    runtime_s: float
    gcups: float
    cells: int
    steps: int
    grid_shape: tuple[int, int, int]
    termination_reason: str


def paper_through_power(case: DifferentialCase, resolution_ppw: int) -> float:
    """Return the paper's reported crossing through power at 1550 nm."""
    references = case.geometry["simulation"]["published_through_power_1550nm_span20nm"]
    return float(references[str(int(resolution_ppw))]["consensus"])


def _component_polygons(component: Any, layer: tuple[int, int]):
    polygons = component.get_polygons_points(by="tuple").get(tuple(layer), ())
    if not polygons:
        raise ValueError(f"component {component.name!r} has no layer {tuple(layer)}")
    return polygons


def _crossing_design(case: DifferentialCase):
    """Extrude both paper GDS layers and extend the four guides to the domain."""
    from beamz import Design, Material, Polygon, Rectangle, µm

    component = generate_layout(case)
    bounds = domain_bounds_um(case)
    width_um, height_um, depth_um = domain_size_um(case)
    x_offset_um, y_offset_um = -bounds["x"][0], -bounds["y"][0]
    silicon = Material(case.materials["silicon_n_at_1p55_um"] ** 2)
    silica = Material(case.materials["silica_n_at_1p55_um"] ** 2)
    design = Design(
        width=width_um * µm,
        height=height_um * µm,
        depth=depth_um * µm,
        background=silica,
    )

    for layer in case.geometry["layers"].values():
        thickness = float(layer["thickness_m"])
        beamz_z_um = float(layer["zmin_um"]) - bounds["z"][0]
        for points in _component_polygons(component, tuple(layer["gds"])):
            design += Polygon(
                vertices=tuple(
                    (
                        (float(x) + x_offset_um) * µm,
                        (float(y) + y_offset_um) * µm,
                    )
                    for x, y in np.asarray(points, dtype=float)[:, :2]
                ),
                z=beamz_z_um * µm,
                depth=thickness,
                material=silicon,
            )

    core_depth = float(case.geometry["layers"]["core"]["thickness_m"])
    core_z_um = float(case.geometry["layers"]["core"]["zmin_um"]) - bounds["z"][0]
    extension = float(case.geometry["simulation"]["port_extension_um"]) * µm
    for port in case.geometry["ports"].values():
        x_um, y_um = port["center_um"]
        width = float(port["width_um"]) * µm
        orientation = int(round(float(port["orientation_deg"]))) % 360
        x = (x_um + x_offset_um) * µm
        y = (y_um + y_offset_um) * µm
        if orientation == 180:
            position, size = (x - extension, y - width / 2), (extension, width)
        elif orientation == 0:
            position, size = (x, y - width / 2), (extension, width)
        elif orientation == 270:
            position, size = (x - width / 2, y - extension), (width, extension)
        elif orientation == 90:
            position, size = (x - width / 2, y), (width, extension)
        else:
            raise ValueError(f"unsupported crossing port orientation {orientation}")
        design += Rectangle(
            position=(*position, core_z_um * µm),
            width=size[0],
            height=size[1],
            depth=core_depth,
            material=silicon,
        )
    return design.unified_polygons()


def build_crossing_simulation(
    *,
    resolution_ppw: int,
    wavelength_span_nm: float = 20.0,
    diagnostics: bool = False,
):
    """Build the paper-matched BeamZ crossing simulation without executing it."""
    from beamz import (
        LIGHT_SPEED,
        Absorber,
        FieldMonitor,
        GaussianPulse,
        ModeSpec,
        Port,
        Simulation,
        dxdt,
        µm,
    )
    from beamz.design.raster import RasterOptions

    case = load_passive_soi_case("crossing")
    protocol = case.geometry["simulation"]
    if int(resolution_ppw) not in protocol["resolutions_cells_per_wavelength"]:
        raise ValueError(f"unsupported paper resolution {resolution_ppw}")
    if float(wavelength_span_nm) not in protocol["wavelength_spans_nm"]:
        raise ValueError(f"unsupported paper wavelength span {wavelength_span_nm}")

    design = _crossing_design(case)
    bounds = domain_bounds_um(case)
    wavelength_center = float(protocol["wavelength_center_um"]) * µm
    frequencies = reference_frequencies(case, wavelength_span_nm)
    dx, _ = dxdt(
        wavelength_center,
        n_max=float(case.materials["silicon_n_at_1p55_um"]),
        dims=3,
        safety_factor=0.999,
        points_per_wavelength=int(resolution_ppw),
    )

    mode_spec = ModeSpec(polarization="te")
    core_layer = case.geometry["layers"]["core"]
    z_center_um = (
        float(core_layer["zmin_um"])
        + 0.5 * float(core_layer["thickness_m"]) / µm
        - bounds["z"][0]
    )
    z_center = z_center_um * µm
    z_span = 2.0 * µm
    transverse_span = 4.5 * µm
    ports = []
    for name in case.geometry["ports"]:
        center, direction = port_center_and_direction(
            case,
            name,
            inward_offset_um=0.5 if name == "o1" else 0.0,
            z_center=z_center,
        )
        orientation = (
            int(round(float(case.geometry["ports"][name]["orientation_deg"]))) % 360
        )
        size = (
            (0.0, transverse_span, z_span)
            if orientation in {0, 180}
            else (transverse_span, 0.0, z_span)
        )
        ports.append(
            Port(
                center=center,
                size=size,
                name=name,
                direction=direction,
                mode_spec=mode_spec,
            )
        )
    ports = tuple(ports)
    source_center, source_direction = port_center_and_direction(
        case, "o1", inward_offset_um=0.0, z_center=z_center
    )
    source_port = Port(
        center=source_center,
        size=(0.0, transverse_span, z_span),
        name="source",
        direction=source_direction,
        mode_spec=mode_spec,
    )
    frequency_width = float(np.ptp(frequencies))
    source_time = GaussianPulse(
        freq0=LIGHT_SPEED / wavelength_center,
        fwidth=frequency_width,
        # Match Tidy3D's default offset of five temporal widths. BeamZ's offset
        # is expressed in 1/fwidth rather than 1/(2*pi*fwidth).
        offset=5.0 / (2.0 * np.pi),
    )
    source = source_port.to_source(
        freq0=source_time.freq0,
        fwidth=frequency_width,
        num_freqs=round(float(wavelength_span_nm) / 10.0) + 1,
        source_time=source_time,
    )
    run_time = 15.0 * domain_size_um(case)[0] * µm * 2.0 / LIGHT_SPEED
    monitors = [port.to_monitor(frequencies) for port in ports]
    if diagnostics:
        monitors.append(
            FieldMonitor(
                center=(0.5 * design.width, 0.5 * design.height, z_center),
                size=(design.width, design.height, 0.0),
                freqs=frequencies,
                fields=("Ex", "Ey", "Ez"),
                name="crossing_xy",
            )
        )
    simulation = Simulation(
        design=design,
        sources=[source],
        monitors=monitors,
        boundaries=[Absorber(edges="all", thickness=1.0 * µm)],
        run_time=run_time,
        resolution=dx,
        raster_options=RasterOptions(
            quality="balanced", smoothing="farjadpour_diagonal"
        ),
    )
    return simulation, ports, frequencies


def _save_crossing_artifacts(
    directory: Path, simulation, results, scattering, *, execution_backend: str
) -> None:
    """Persist raw and visual evidence for one crossing run."""
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)

    field_monitor = next(
        monitor for monitor in simulation.monitors if monitor.name == "crossing_xy"
    )
    cross_section = {
        "z": field_monitor.center[2],
        "y": 0.5 * simulation.design.height,
    }
    fig, _ = simulation.plot(
        **cross_section,
        source_markers=False,
        monitor_markers=False,
    )
    fig.savefig(directory / "geometry_cross_sections.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, _ = simulation.plot(
        **cross_section,
    )
    fig.savefig(directory / "simulation_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, _ = results.plot_field(
        monitor_name="crossing_xy",
        field_name="E",
        frequency=float(np.median(scattering.frequencies)),
        val="abs^2",
    )
    fig.savefig(directory / "field_E_abs2_1550nm.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    source_time = simulation.sources[0].source_time
    source_signal, source_quadrature = source_time.sample(simulation.time)
    arrays: dict[str, np.ndarray] = {
        "frequencies_hz": np.asarray(scattering.frequencies),
        "source_time_s": np.asarray(simulation.time),
        "source_signal": np.asarray(source_signal),
        "source_quadrature": np.asarray(source_quadrature),
        "valid_mask": np.asarray(scattering.diagnostics["valid_mask"]),
        "incident_power": np.asarray(scattering.diagnostics["P_in"]),
        "guided_output_power": np.asarray(scattering.diagnostics["P_guided_out"]),
        "power_sum": np.asarray(scattering.diagnostics["power_sum"]),
        "loss_estimate": np.asarray(scattering.diagnostics["loss_est"]),
    }
    for (output, source), values in scattering.s_matrix.items():
        arrays[f"S_{output}_{source}"] = np.asarray(values)
    for monitor_name, monitor_results in results.monitors.items():
        arrays[f"{monitor_name}__frequencies_hz"] = np.asarray(
            monitor_results.get_dft_frequencies()
        )
        for component in monitor_results.dft_fields:
            arrays[f"{monitor_name}__{component}"] = np.asarray(
                monitor_results.get_dft_component(component)
            )
    np.savez_compressed(directory / "monitor_data.npz", **arrays)

    performance = results.performance
    termination = results.termination
    metadata = {
        "execution_backend": execution_backend,
        "resolution_m": float(simulation.resolution),
        "grid_shape": list(simulation.grid.shape),
        "grid_is_uniform": bool(simulation.grid.is_uniform),
        "raster_quality": simulation.raster_options.quality,
        "raster_smoothing": simulation.raster_options.smoothing,
        "boundary_formulation": simulation.boundaries[0].formulation,
        "steps": int(performance.steps if performance else simulation.num_steps),
        "runtime_s": float(performance.runtime_s if performance else float("nan")),
        "gcups": float(performance.gcups if performance else float("nan")),
        "cells": int(performance.cells if performance else 0),
        "termination_reason": termination.reason if termination else "time_limit",
        "termination_field_decay": termination.field_decay if termination else None,
    }
    (directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_crossing_benchmark(
    *,
    resolution_ppw: int,
    wavelength_span_nm: float = 20.0,
    progress: bool = False,
    backend: str | None = None,
    artifact_dir: str | Path | None = None,
) -> CrossingBenchmarkResult:
    """Execute one crossing case and extract paper-comparable TE0 powers."""
    from beamz import LIGHT_SPEED, AutoTermination, µm
    from beamz.analysis import s_parameters

    simulation, ports, frequencies = build_crossing_simulation(
        resolution_ppw=resolution_ppw,
        wavelength_span_nm=wavelength_span_nm,
        diagnostics=artifact_dir is not None,
    )
    with reference_absorber_warning_scope():
        program = simulation.compile(progress=progress, backend=backend)
        execution_backend = program.config.backend
        results = simulation.run(
            progress=progress,
            backend=execution_backend,
            termination=AutoTermination(
                field_decay=1e-5,
                monitor_change=None,
                consecutive_checks=1,
            ),
        )
    scattering = s_parameters(
        results,
        source_port="o1",
        ports=ports,
        output_ports=["o2", "o3", "o4"],
        frequencies=frequencies,
        min_incident_db=-45.0,
    )
    if artifact_dir is not None:
        _save_crossing_artifacts(
            Path(artifact_dir),
            simulation,
            results,
            scattering,
            execution_backend=execution_backend,
        )
    wavelengths_um = LIGHT_SPEED / np.asarray(scattering.frequencies) / µm
    center = int(np.argmin(np.abs(wavelengths_um - 1.55)))
    output_spectra = {
        name: np.abs(np.asarray(scattering.s_matrix[(name, "o1")])) ** 2
        for name in ("o2", "o3", "o4")
    }
    total_output_spectrum = sum(output_spectra.values())
    through_spectrum = output_spectra["o3"]
    excess_loss_spectrum = 1.0 - total_output_spectrum
    performance = results.performance
    termination = results.termination
    return CrossingBenchmarkResult(
        resolution_ppw=int(resolution_ppw),
        wavelength_span_nm=float(wavelength_span_nm),
        backend=execution_backend,
        wavelengths_um=tuple(float(value) for value in wavelengths_um),
        through_power_spectrum=tuple(float(value) for value in through_spectrum),
        total_output_power_spectrum=tuple(
            float(value) for value in total_output_spectrum
        ),
        excess_loss_spectrum=tuple(float(value) for value in excess_loss_spectrum),
        through_power=float(through_spectrum[center]),
        total_output_power=float(total_output_spectrum[center]),
        excess_loss=float(excess_loss_spectrum[center]),
        runtime_s=float(performance.runtime_s if performance else float("nan")),
        gcups=float(performance.gcups if performance else float("nan")),
        cells=int(performance.cells if performance else 0),
        steps=int(performance.steps if performance else simulation.num_steps),
        grid_shape=tuple(int(value) for value in simulation.grid.shape),
        termination_reason=termination.reason if termination else "time_limit",
    )
