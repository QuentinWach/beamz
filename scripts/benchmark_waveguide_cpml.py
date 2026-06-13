#!/usr/bin/env python3
"""Compact straight-waveguide CPML reflection benchmark.

This mirrors the modal diagnostics in 0_waveguide_demo.ipynb without notebook
plots or a full-field DFT monitor. It intentionally reports the source-side
backward-injection estimate separately from the two CPML-return estimates.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not (root / "beamz" / "__init__.py").exists():
        raise RuntimeError(f"Could not locate BeamZ checkout from {__file__!r}.")
    return root


def _prefer_local_packages() -> tuple[Any, Any, Path, Path]:
    root = _repo_root()
    micromode_python = root.parent / "micromode" / "python"
    local_paths = [root]
    if micromode_python.exists():
        local_paths.append(micromode_python)

    sys.path = [
        p
        for p in sys.path
        if all(Path(p or ".").resolve() != local for local in local_paths)
    ]
    sys.path[:0] = [str(path) for path in local_paths]
    for name in tuple(sys.modules):
        if name == "beamz" or name.startswith("beamz."):
            del sys.modules[name]
        if name == "micromode" or name.startswith("micromode."):
            del sys.modules[name]

    import beamz as bz  # noqa: PLC0415
    import micromode  # noqa: PLC0415

    beamz_path = Path(bz.__file__).resolve()
    micromode_path = Path(micromode.__file__).resolve()
    if root not in beamz_path.parents:
        raise RuntimeError(f"BeamZ resolved outside this checkout: {beamz_path}")
    if not hasattr(micromode, "ModePlaneSpec") or not hasattr(
        micromode, "solve_beamz_mode"
    ):
        raise RuntimeError(
            "micromode resolved without the BEAMZ discrete mode contract: "
            f"{micromode_path}"
        )
    return bz, micromode, beamz_path, micromode_path


def _git_branch(root: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), "branch", "--show-current"],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() or "(detached HEAD)"


def _safe_ratio(numerator, denominator, floor: float = 1e-18):
    numerator = np.asarray(numerator, dtype=np.complex128)
    denominator = np.asarray(denominator, dtype=np.complex128)
    return np.where(
        np.abs(denominator) >= floor,
        numerator / denominator,
        0.0 + 0.0j,
    )


def _amp_db(values, floor: float = 1e-12):
    return 20.0 * np.log10(np.maximum(np.abs(values), floor))


def _cpml_sigma_max(bz, *, order: int, target_reflection: float, thickness: float):
    eta = math.sqrt(float(bz.MU_0) / float(bz.EPS_0))
    return (
        -(order + 1)
        * math.log(max(float(target_reflection), 1e-16))
        / (2.0 * eta * max(float(thickness), 1e-30))
    )


def _cpml_boundary(bz, *, edges, thickness: float, dt: float, args, sigma_scale):
    kwargs: dict[str, Any] = {
        "edges": edges,
        "thickness": thickness,
        "formulation": "cpml",
        "m": args.cpml_order,
        "kappa_max": args.kappa_max,
        "target_reflection": args.target_reflection,
    }
    if sigma_scale is not None:
        kwargs["sigma_max"] = _cpml_sigma_max(
            bz,
            order=args.cpml_order,
            target_reflection=args.target_reflection,
            thickness=thickness,
        ) * sigma_scale
    if args.alpha_normalized is not None:
        kwargs["alpha_max"] = (
            2.0 * float(bz.EPS_0) * args.alpha_normalized / max(float(dt), 1e-30)
        )
    return bz.PML(**kwargs)


def build_and_run(args):
    bz, _micromode, beamz_path, micromode_path = _prefer_local_packages()
    root = _repo_root()
    um = bz.um

    waveguide_length = args.waveguide_length_um * um
    core_width = 0.445 * um
    core_height = 0.220 * um
    port_extension = args.port_extension_um * um
    background_ext = args.background_ext_um * um
    substrate_height = port_extension
    cladding_height = core_height + port_extension
    interior_size = (
        waveguide_length,
        core_width + 2.0 * port_extension,
        substrate_height + cladding_height + background_ext,
    )

    mat_air = bz.Material(permittivity=1.0)
    mat_si = bz.Material(permittivity=3.48**2)
    mat_sio2 = bz.Material(permittivity=1.45**2)

    lambda0 = args.wavelength_um * um
    freq0 = bz.LIGHT_SPEED / lambda0
    source_fwidth = freq0 * args.fwidth_fraction
    source_time = bz.GaussianPulse(freq0=freq0, fwidth=source_fwidth)
    mode_spec = bz.ModeSpec(num_modes=1, target_neff=0.98 * 3.48, polarization="te")
    if args.num_freqs <= 1:
        sparam_freqs = np.asarray([freq0], dtype=float)
    else:
        sparam_freqs = np.linspace(
            freq0 - 0.5 * source_fwidth,
            freq0 + 0.5 * source_fwidth,
            args.num_freqs,
        )
    sparam_wavelengths_um = bz.LIGHT_SPEED / sparam_freqs / um

    grid_spec = bz.GridSpec.auto(
        min_steps_per_wvl=args.steps_per_wavelength,
        wavelength=lambda0,
    )
    max_index = math.sqrt(
        max(mat_air.permittivity, mat_si.permittivity, mat_sio2.permittivity)
    )
    grid_resolution = grid_spec.resolve_resolution(max_index=max_index)
    pml_t = args.cpml_cells * grid_resolution
    if args.domain_mode == "interior":
        sim_size = tuple(float(v + 2.0 * pml_t) for v in interior_size)
        non_pml_size = interior_size
    else:
        sim_size = interior_size
        non_pml_size = tuple(
            float(max(v - 2.0 * pml_t, 0.0)) for v in interior_size
        )
    run_time = args.run_time_cycles / freq0

    non_pml_x_min = -0.5 * non_pml_size[0]
    non_pml_x_max = 0.5 * non_pml_size[0]
    src_x = non_pml_x_min + args.source_offset_um * um
    diagnostic_gap = args.diagnostic_gap_um * um
    source_back_x = max(src_x - diagnostic_gap, non_pml_x_min + 0.2 * um)
    source_forward_x = min(src_x + diagnostic_gap, non_pml_x_max - 0.2 * um)

    design = bz.Design(background=mat_air)
    design += bz.Box(
        center=(0, 0, -0.5 * sim_size[2]),
        size=(bz.inf, bz.inf, sim_size[2]),
        material=mat_sio2,
    )
    design += bz.Box(
        center=(0, 0, 0.5 * core_height),
        size=(bz.inf, core_width, core_height),
        material=mat_si,
    )

    requested_source_plane_scale = math.sqrt(args.source_area_fraction)
    non_pml_y = non_pml_size[1]
    non_pml_z = non_pml_size[2]
    source_plane_y = min(interior_size[1] * requested_source_plane_scale, non_pml_y)
    source_plane_z = min(interior_size[2] * requested_source_plane_scale, non_pml_z)
    source_plane_size = (0.0, source_plane_y, source_plane_z)
    source_plane_area_fraction = (source_plane_y * source_plane_z) / (
        interior_size[1] * interior_size[2]
    )
    src_plane = bz.Box(center=(src_x, 0, 0.0), size=source_plane_size)

    modal_monitor_common = dict(
        size=source_plane_size,
        freqs=sparam_freqs,
        mode_spec=mode_spec,
        polarization="te",
        mode_index=0,
    )
    source_back_monitor = bz.ModeMonitor(
        center=(source_back_x, 0, 0.0),
        direction="+x",
        name="source_back",
        reference_monitor="source_forward",
        **modal_monitor_common,
    )
    source_forward_monitor = bz.ModeMonitor(
        center=(source_forward_x, 0, 0.0),
        direction="-x",
        name="source_forward",
        **modal_monitor_common,
    )

    provisional_dt = grid_spec.resolve_time_step(grid_resolution, dims=3)
    x_sigma_scale = (
        args.x_sigma_scale
        if args.x_sigma_scale is not None
        else args.sigma_scale
    )
    transverse_sigma_scale = (
        args.transverse_sigma_scale
        if args.transverse_sigma_scale is not None
        else args.sigma_scale
    )
    boundary_spec = bz.BoundarySpec(
        (
            _cpml_boundary(
                bz,
                edges=["left", "right"],
                thickness=pml_t,
                dt=provisional_dt,
                args=args,
                sigma_scale=x_sigma_scale,
            ),
            _cpml_boundary(
                bz,
                edges=["bottom", "top", "front", "back"],
                thickness=pml_t,
                dt=provisional_dt,
                args=args,
                sigma_scale=transverse_sigma_scale,
            ),
        )
    )

    t0 = time.perf_counter()
    sim0 = bz.Simulation(
        domain=sim_size,
        grid_spec=grid_spec,
        design=design,
        sources=[],
        monitors=[source_back_monitor, source_forward_monitor],
        boundary_spec=boundary_spec,
        run_time=run_time,
    )
    setup_wall_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    mode_solver = bz.ModeSolver(
        simulation=sim0,
        plane=src_plane,
        mode_spec=mode_spec,
        freqs=[freq0],
    )
    modes = mode_solver.solve()
    mode_source = mode_solver.to_source(
        mode_index=0,
        direction="+x",
        source_time=source_time,
        polarization="te",
        power=1.0,
    )
    mode_source.initialize(sim0.fields.permittivity, sim0.resolution, dt=sim0.dt)
    mode_solve_wall_s = time.perf_counter() - t0

    sim = sim0.copy(update={"sources": [mode_source]})
    shape_zyx = tuple(int(v) for v in sim.fields.permittivity.shape)
    cells = int(np.prod(shape_zyx))

    t0 = time.perf_counter()
    sim_data = sim.run_compiled(progress=args.progress)
    run_wall_s = time.perf_counter() - t0
    del sim_data

    source_port = bz.PortSpec(
        name="source_back",
        monitor_name="source_back",
        reference_monitor="source_forward",
        direction="+x",
        polarization="te",
        mode_index=0,
        incident_wave="minus",
        scattered_wave="plus",
    )
    forward_port = bz.PortSpec(
        name="source_forward",
        monitor_name="source_forward",
        direction="-x",
        polarization="te",
        mode_index=0,
        incident_wave="minus",
        scattered_wave="plus",
    )
    sparam_result = sim.get_S_matrix_modal_dft(
        source_port=source_port,
        ports=[source_port, forward_port],
        output_ports=[source_port, forward_port],
        frequencies=sparam_freqs,
        as_sax=False,
        return_diagnostics=True,
        min_incident_db=args.min_incident_db,
    )
    waves = sparam_result["diagnostics"]["waves"]
    source_incident_selector = str(source_port.incident_wave).lower()
    a_incident = np.asarray(
        waves["source_back"][f"a_incident_{source_incident_selector}"],
        dtype=np.complex128,
    )
    valid = np.asarray(sparam_result["diagnostics"]["valid_mask"], dtype=bool)

    modal_traces = {
        "back_monitor_minus_x_db": _amp_db(
            _safe_ratio(waves["source_back"]["a_plus"], a_incident)
        ),
        "back_monitor_plus_x_db": _amp_db(
            _safe_ratio(waves["source_back"]["a_minus"], a_incident)
        ),
        "front_monitor_plus_x_db": _amp_db(
            _safe_ratio(waves["source_forward"]["a_plus"], a_incident)
        ),
        "front_monitor_minus_x_db": _amp_db(
            _safe_ratio(waves["source_forward"]["a_minus"], a_incident)
        ),
    }
    center_idx = int(np.argmin(np.abs(sparam_freqs - freq0)))

    neff = float(np.real(np.asarray(modes.neffs).reshape(-1)[0]))
    result = {
        "metadata": {
            "beamz_path": str(beamz_path),
            "micromode_path": str(micromode_path),
            "branch": _git_branch(root),
            "grid_zyx": shape_zyx,
            "cells": cells,
            "steps": int(sim.num_steps),
            "resolution_um": float(grid_resolution / um),
            "dt_s": float(sim.dt),
            "cpml_cells": int(args.cpml_cells),
            "cpml_thickness_um": float(pml_t / um),
            "domain_mode": args.domain_mode,
            "interior_size_um": [float(v / um) for v in interior_size],
            "simulation_size_um": [float(v / um) for v in sim_size],
            "sigma_scale": args.sigma_scale,
            "x_sigma_scale": x_sigma_scale,
            "transverse_sigma_scale": transverse_sigma_scale,
            "kappa_max": float(args.kappa_max),
            "alpha_normalized": args.alpha_normalized,
            "cpml_order": int(args.cpml_order),
            "target_reflection": float(args.target_reflection),
            "source_plane_area_fraction": float(source_plane_area_fraction),
            "mode_neff": neff,
            "setup_wall_s": setup_wall_s,
            "mode_solve_wall_s": mode_solve_wall_s,
            "run_wall_s": run_wall_s,
        },
        "wavelength_um": sparam_wavelengths_um.tolist(),
        "valid": valid.tolist(),
        **{name: values.tolist() for name, values in modal_traces.items()},
        "center": {
            "wavelength_um": float(sparam_wavelengths_um[center_idx]),
            "back_monitor_minus_x_db": float(
                modal_traces["back_monitor_minus_x_db"][center_idx]
            ),
            "back_monitor_plus_x_db": float(
                modal_traces["back_monitor_plus_x_db"][center_idx]
            ),
            "front_monitor_plus_x_db": float(
                modal_traces["front_monitor_plus_x_db"][center_idx]
            ),
            "front_monitor_minus_x_db": float(
                modal_traces["front_monitor_minus_x_db"][center_idx]
            ),
            "valid": bool(valid[center_idx]),
        },
    }
    return result


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpml-cells", type=int, default=12)
    parser.add_argument("--steps-per-wavelength", type=int, default=15)
    parser.add_argument("--run-time-cycles", type=float, default=120.0)
    parser.add_argument("--num-freqs", type=int, default=11)
    parser.add_argument("--wavelength-um", type=float, default=1.55)
    parser.add_argument("--fwidth-fraction", type=float, default=0.1)
    parser.add_argument("--waveguide-length-um", type=float, default=5.0)
    parser.add_argument("--port-extension-um", type=float, default=1.0)
    parser.add_argument("--background-ext-um", type=float, default=0.1)
    parser.add_argument("--source-offset-um", type=float, default=0.75)
    parser.add_argument("--diagnostic-gap-um", type=float, default=0.50)
    parser.add_argument("--source-area-fraction", type=float, default=0.65)
    parser.add_argument(
        "--domain-mode",
        choices=("total", "interior"),
        default="total",
        help=(
            "'total' keeps the notebook's fixed simulation box; 'interior' "
            "keeps the non-PML design region fixed and adds CPML outside it."
        ),
    )
    parser.add_argument("--target-reflection", type=float, default=1e-6)
    parser.add_argument("--cpml-order", type=int, default=3)
    parser.add_argument("--kappa-max", type=float, default=2.0)
    parser.add_argument(
        "--sigma-scale",
        type=float,
        default=None,
        help="Override BeamZ default CPML sigma scale; omit to use PML default.",
    )
    parser.add_argument(
        "--x-sigma-scale",
        type=float,
        default=None,
        help="Override only left/right CPML sigma scale.",
    )
    parser.add_argument(
        "--transverse-sigma-scale",
        type=float,
        default=None,
        help="Override only bottom/top/front/back CPML sigma scale.",
    )
    parser.add_argument(
        "--alpha-normalized",
        type=float,
        default=None,
        help="Override normalized CFS alpha; omit to use PML default.",
    )
    parser.add_argument("--min-incident-db", type=float, default=-45.0)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--json-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_and_run(args)
    center = result["center"]
    if not args.json_only:
        print("Waveguide CPML modal benchmark")
        print(f"center wavelength = {center['wavelength_um']:.4f} um")
        print(
            "back monitor: -x wave toward left CPML = "
            f"{center['back_monitor_minus_x_db']:.2f} dB"
        )
        print(
            "back monitor: +x wave toward source = "
            f"{center['back_monitor_plus_x_db']:.2f} dB"
        )
        print(
            "front monitor: +x wave toward right CPML = "
            f"{center['front_monitor_plus_x_db']:.2f} dB"
        )
        print(
            "front monitor: -x wave returning from right CPML = "
            f"{center['front_monitor_minus_x_db']:.2f} dB"
        )
        print(
            "valid frequency bins = "
            f"{sum(result['valid'])}/{len(result['valid'])}"
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
