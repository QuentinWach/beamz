from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

C0 = 299_792_458.0


@dataclass(frozen=True)
class PaperBenchmarkConfig:
    cell_x_um: float = 6.0
    cell_y_um: float = 4.0
    cell_z_um: float = 1.5
    total_time_fs: float = 200.0
    courant_safety: float = 0.99
    wavelength_um: float = 1.55
    n_clad: float = 1.44
    n_core: float = 3.48
    pml_um: float = 0.4
    slab_thickness_um: float = 0.22
    slab_center_z_um: float = 0.75
    input_wg_x0_um: float = 0.5
    input_wg_x1_um: float = 1.6
    input_wg_yc_um: float = 1.55
    input_wg_width_um: float = 0.45
    output_wg_x0_um: float = 4.4
    output_wg_x1_um: float = 5.5
    output_wg_yc_um: float = 2.45
    output_wg_width_um: float = 0.75
    source_x_um: float = 0.95
    source_y_um: float = 1.55
    source_span_y_um: float = 1.20
    source_span_z_um: float = 0.90
    source_ramp_fs: float = 40.0

    @property
    def total_time_s(self) -> float:
        return self.total_time_fs * 1e-15

    @property
    def source_ramp_s(self) -> float:
        return self.source_ramp_fs * 1e-15

    @property
    def frequency_hz(self) -> float:
        return C0 / (self.wavelength_um * 1e-6)

    @property
    def meep_frequency(self) -> float:
        return 1.0 / self.wavelength_um

    @property
    def slab_z0_um(self) -> float:
        return self.slab_center_z_um - 0.5 * self.slab_thickness_um

    @property
    def beamz_polygon_vertices_um(self) -> list[tuple[float, float, float]]:
        z0 = self.slab_z0_um
        return [
            (1.60, 1.325, z0),
            (2.05, 1.180, z0),
            (2.65, 1.120, z0),
            (3.25, 1.190, z0),
            (3.85, 1.420, z0),
            (4.40, 2.075, z0),
            (4.40, 2.825, z0),
            (3.90, 3.000, z0),
            (3.10, 2.940, z0),
            (2.30, 2.760, z0),
            (1.72, 2.280, z0),
            (1.60, 1.775, z0),
        ]

    @property
    def meep_polygon_vertices_um(self) -> list[tuple[float, float]]:
        return [(x - 0.5 * self.cell_x_um, y - 0.5 * self.cell_y_um) for x, y, _ in self.beamz_polygon_vertices_um]


def _beamz_xyz(x_um: float, y_um: float, z_um: float, cfg: PaperBenchmarkConfig) -> tuple[float, float, float]:
    return (
        x_um * 1e-6,
        y_um * 1e-6,
        z_um * 1e-6,
    )


def _beamz_xy(x_um: float, y_um: float) -> tuple[float, float]:
    return (x_um * 1e-6, y_um * 1e-6)


def _rect_position_from_center(
    x0_um: float,
    x1_um: float,
    yc_um: float,
    width_um: float,
    z0_um: float,
) -> tuple[float, float, float]:
    return (x0_um * 1e-6, (yc_um - 0.5 * width_um) * 1e-6, z0_um * 1e-6)


def _num_steps_for_resolution(resolution_nm: float, cfg: PaperBenchmarkConfig) -> tuple[int, float]:
    dx_m = resolution_nm * 1e-9
    dt_s = cfg.courant_safety * dx_m / (C0 * math.sqrt(3.0))
    num_steps = int(math.floor(cfg.total_time_s / dt_s))
    return max(num_steps, 2), dt_s


def _paper_cells(resolution_nm: float, cfg: PaperBenchmarkConfig) -> tuple[int, int, int]:
    return (
        int(round((cfg.cell_x_um * 1000.0) / resolution_nm)),
        int(round((cfg.cell_y_um * 1000.0) / resolution_nm)),
        int(round((cfg.cell_z_um * 1000.0) / resolution_nm)),
    )


def _signal(time_axis: np.ndarray, cfg: PaperBenchmarkConfig) -> np.ndarray:
    ramp = np.tanh(np.maximum(time_axis, 0.0) / max(cfg.source_ramp_s, 1e-30)) ** 2
    return ramp * np.cos(2.0 * np.pi * cfg.frequency_hz * time_axis)


def _scalar_runtime(
    *,
    setup_s: float,
    compile_s: float,
    run_s: float,
    total_s: float,
    num_steps: int,
    grid_shape: tuple[int, int, int],
) -> dict[str, Any]:
    cells = int(np.prod(np.asarray(grid_shape, dtype=np.int64)))
    cell_updates = cells * int(num_steps)
    comp_updates = 6 * cell_updates
    run_den = max(run_s, 1e-12)
    return {
        "setup_s": float(setup_s),
        "compile_s": float(compile_s),
        "run_s": float(run_s),
        "total_s": float(total_s),
        "num_steps": int(num_steps),
        "grid_shape": [int(v) for v in grid_shape],
        "cells": cells,
        "cell_updates": int(cell_updates),
        "component_updates": int(comp_updates),
        "mcells_per_s": float(cell_updates / run_den / 1e6),
        "mcomponents_per_s": float(comp_updates / run_den / 1e6),
    }


def _normalize_plane(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak <= 1e-30:
        return np.zeros_like(arr, dtype=np.float64)
    return arr / peak


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(b.ravel()))
    if den <= 1e-30:
        return 0.0
    return float(np.linalg.norm((a - b).ravel()) / den)


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    if aa.size == 0 or bb.size == 0:
        return 1.0
    aa = aa - np.mean(aa)
    bb = bb - np.mean(bb)
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na <= 1e-30 or nb <= 1e-30:
        return 1.0
    return float(np.dot(aa, bb) / (na * nb))


def _shape_or_transpose(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape == target_shape:
        return arr
    if arr.T.shape == target_shape:
        return arr.T
    raise ValueError(f"Plane shape mismatch: got {arr.shape}, expected {target_shape}")


def _center_component(arr: np.ndarray, axis: int, target_shape: tuple[int, int, int]) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    out = np.zeros(target_shape, dtype=np.float64)
    if axis == 2:
        out[..., 0] = arr[..., 0]
        out[..., -1] = arr[..., -1]
        if target_shape[2] > 2:
            out[..., 1:-1] = 0.5 * (arr[..., :-1] + arr[..., 1:])
    elif axis == 1:
        out[:, 0, :] = arr[:, 0, :]
        out[:, -1, :] = arr[:, -1, :]
        if target_shape[1] > 2:
            out[:, 1:-1, :] = 0.5 * (arr[:, :-1, :] + arr[:, 1:, :])
    elif axis == 0:
        out[0, ...] = arr[0, ...]
        out[-1, ...] = arr[-1, ...]
        if target_shape[0] > 2:
            out[1:-1, ...] = 0.5 * (arr[:-1, ...] + arr[1:, ...])
    else:
        raise ValueError(f"Unsupported axis {axis}")
    return out


def _extract_beamz_planes(sim) -> dict[str, np.ndarray]:
    perm_shape = tuple(int(v) for v in np.asarray(sim.fields.permittivity).shape)
    ex = _center_component(np.asarray(sim.fields.Ex), 2, perm_shape)
    ey = _center_component(np.asarray(sim.fields.Ey), 1, perm_shape)
    ez = _center_component(np.asarray(sim.fields.Ez), 0, perm_shape)
    e_mag = np.sqrt(ex**2 + ey**2 + ez**2)
    iz = perm_shape[0] // 2
    iy = perm_shape[1] // 2
    ix = perm_shape[2] // 2
    return {
        "xy": np.asarray(e_mag[iz, :, :], dtype=np.float64),
        "xz": np.asarray(e_mag[:, iy, :], dtype=np.float64),
        "yz": np.asarray(e_mag[:, :, ix], dtype=np.float64),
    }


def run_beamz_single(
    cfg: PaperBenchmarkConfig,
    resolution_nm: float,
    *,
    capture_accuracy: bool,
) -> dict[str, Any]:
    from beamz import Design, Material, ModeSource, PML, Polygon, Rectangle, Simulation

    num_steps, dt_s = _num_steps_for_resolution(resolution_nm, cfg)
    dx_m = resolution_nm * 1e-9
    t = np.arange(num_steps, dtype=np.float64) * dt_s
    signal = _signal(t, cfg)

    start_total = time.perf_counter()
    setup_start = time.perf_counter()

    design = Design(
        width=cfg.cell_x_um * 1e-6,
        height=cfg.cell_y_um * 1e-6,
        depth=cfg.cell_z_um * 1e-6,
        material=Material(cfg.n_clad**2),
    )
    design += Rectangle(
        position=_rect_position_from_center(
            cfg.input_wg_x0_um,
            cfg.input_wg_x1_um,
            cfg.input_wg_yc_um,
            cfg.input_wg_width_um,
            cfg.slab_z0_um,
        ),
        width=(cfg.input_wg_x1_um - cfg.input_wg_x0_um) * 1e-6,
        height=cfg.input_wg_width_um * 1e-6,
        depth=cfg.slab_thickness_um * 1e-6,
        material=Material(cfg.n_core**2),
    )
    design += Rectangle(
        position=_rect_position_from_center(
            cfg.output_wg_x0_um,
            cfg.output_wg_x1_um,
            cfg.output_wg_yc_um,
            cfg.output_wg_width_um,
            cfg.slab_z0_um,
        ),
        width=(cfg.output_wg_x1_um - cfg.output_wg_x0_um) * 1e-6,
        height=cfg.output_wg_width_um * 1e-6,
        depth=cfg.slab_thickness_um * 1e-6,
        material=Material(cfg.n_core**2),
    )
    design += Polygon(
        vertices=[_beamz_xyz(x, y, z, cfg) for x, y, z in cfg.beamz_polygon_vertices_um],
        material=Material(cfg.n_core**2),
        depth=cfg.slab_thickness_um * 1e-6,
        z=cfg.slab_z0_um * 1e-6,
    )

    grid = design.rasterize(resolution=dx_m)
    source = ModeSource(
        grid=grid,
        center=_beamz_xyz(cfg.source_x_um, cfg.source_y_um, cfg.slab_center_z_um, cfg),
        width=cfg.source_span_y_um * 1e-6,
        height=cfg.source_span_z_um * 1e-6,
        wavelength=cfg.wavelength_um * 1e-6,
        pol="tm",
        signal=signal,
        direction="+x",
    )
    source.initialize(grid.permittivity, dx_m)

    sim = Simulation(
        design=design,
        devices=[source],
        boundaries=[PML(edges="all", thickness=cfg.pml_um * 1e-6)],
        time=t,
        resolution=dx_m,
    )
    setup_s = time.perf_counter() - setup_start

    compile_start = time.perf_counter()
    sim.compile(num_steps=num_steps)
    compile_s = time.perf_counter() - compile_start

    run_start = time.perf_counter()
    sim.run_fast(progress=False)
    run_s = time.perf_counter() - run_start
    total_s = time.perf_counter() - start_total

    payload = {
        "backend": "beamz",
        "resolution_nm": float(resolution_nm),
        "runtime": _scalar_runtime(
            setup_s=setup_s,
            compile_s=compile_s,
            run_s=run_s,
            total_s=total_s,
            num_steps=num_steps,
            grid_shape=tuple(int(v) for v in np.asarray(sim.fields.permittivity).shape),
        ),
    }
    if capture_accuracy:
        payload["planes"] = {k: v.tolist() for k, v in _extract_beamz_planes(sim).items()}
    return payload


def _meep_center(x_um: float, y_um: float, z_um: float, cfg: PaperBenchmarkConfig) -> tuple[float, float, float]:
    return (
        x_um - 0.5 * cfg.cell_x_um,
        y_um - 0.5 * cfg.cell_y_um,
        z_um - 0.5 * cfg.cell_z_um,
    )


def _extract_meep_planes(sim, cfg: PaperBenchmarkConfig, resolution_nm: float) -> dict[str, np.ndarray]:
    import meep as mp

    nx, ny, nz = _paper_cells(resolution_nm, cfg)

    def e_mag(center: mp.Vector3, size: mp.Vector3, target_shape: tuple[int, int]) -> np.ndarray:
        ex = _shape_or_transpose(np.asarray(sim.get_array(center=center, size=size, component=mp.Ex)), target_shape)
        ey = _shape_or_transpose(np.asarray(sim.get_array(center=center, size=size, component=mp.Ey)), target_shape)
        ez = _shape_or_transpose(np.asarray(sim.get_array(center=center, size=size, component=mp.Ez)), target_shape)
        return np.sqrt(np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2)

    return {
        "xy": e_mag(mp.Vector3(0, 0, 0), mp.Vector3(cfg.cell_x_um, cfg.cell_y_um, 0), (ny, nx)),
        "xz": e_mag(mp.Vector3(0, 0, 0), mp.Vector3(cfg.cell_x_um, 0, cfg.cell_z_um), (nz, nx)),
        "yz": e_mag(mp.Vector3(0, 0, 0), mp.Vector3(0, cfg.cell_y_um, cfg.cell_z_um), (nz, ny)),
    }


def run_meep_single(
    cfg: PaperBenchmarkConfig,
    resolution_nm: float,
    *,
    capture_accuracy: bool,
) -> dict[str, Any]:
    import meep as mp

    num_steps, dt_s = _num_steps_for_resolution(resolution_nm, cfg)
    dx_m = resolution_nm * 1e-9
    resolution_px_per_um = 1000.0 / resolution_nm
    meep_courant = cfg.courant_safety / math.sqrt(3.0)
    run_time_um_c = (num_steps * dt_s) * C0 / 1e-6
    ramp_um_c = cfg.source_ramp_s * C0 / 1e-6

    mp.verbosity(0)
    start_total = time.perf_counter()
    setup_start = time.perf_counter()

    core = mp.Medium(index=cfg.n_core)
    clad = mp.Medium(index=cfg.n_clad)
    slab_center = cfg.slab_center_z_um - 0.5 * cfg.cell_z_um
    input_cx = 0.5 * (cfg.input_wg_x0_um + cfg.input_wg_x1_um) - 0.5 * cfg.cell_x_um
    input_cy = cfg.input_wg_yc_um - 0.5 * cfg.cell_y_um
    output_cx = 0.5 * (cfg.output_wg_x0_um + cfg.output_wg_x1_um) - 0.5 * cfg.cell_x_um
    output_cy = cfg.output_wg_yc_um - 0.5 * cfg.cell_y_um

    geometry = [
        mp.Block(
            size=mp.Vector3(cfg.input_wg_x1_um - cfg.input_wg_x0_um, cfg.input_wg_width_um, cfg.slab_thickness_um),
            center=mp.Vector3(input_cx, input_cy, slab_center),
            material=core,
        ),
        mp.Block(
            size=mp.Vector3(cfg.output_wg_x1_um - cfg.output_wg_x0_um, cfg.output_wg_width_um, cfg.slab_thickness_um),
            center=mp.Vector3(output_cx, output_cy, slab_center),
            material=core,
        ),
        mp.Prism(
            vertices=[mp.Vector3(x, y) for x, y in cfg.meep_polygon_vertices_um],
            height=cfg.slab_thickness_um,
            center=mp.Vector3(0, 0, slab_center),
            material=core,
        ),
    ]
    sx, sy, sz = _meep_center(cfg.source_x_um, cfg.source_y_um, cfg.slab_center_z_um, cfg)
    sources = [
        mp.EigenModeSource(
            src=mp.ContinuousSource(frequency=cfg.meep_frequency, width=ramp_um_c),
            center=mp.Vector3(sx, sy, sz),
            size=mp.Vector3(0.0, cfg.source_span_y_um, cfg.source_span_z_um),
            direction=mp.X,
            eig_band=1,
            eig_match_freq=True,
        )
    ]
    sim = mp.Simulation(
        cell_size=mp.Vector3(cfg.cell_x_um, cfg.cell_y_um, cfg.cell_z_um),
        geometry=geometry,
        sources=sources,
        boundary_layers=[mp.PML(thickness=cfg.pml_um)],
        default_material=clad,
        resolution=resolution_px_per_um,
        Courant=meep_courant,
        dimensions=3,
    )
    sim.init_sim()
    setup_s = time.perf_counter() - setup_start

    run_start = time.perf_counter()
    sim.run(until=run_time_um_c)
    run_s = time.perf_counter() - run_start
    total_s = time.perf_counter() - start_total

    payload = {
        "backend": "meep",
        "resolution_nm": float(resolution_nm),
        "runtime": _scalar_runtime(
            setup_s=setup_s,
            compile_s=0.0,
            run_s=run_s,
            total_s=total_s,
            num_steps=num_steps,
            grid_shape=_paper_cells(resolution_nm, cfg)[::-1],
        ),
        "meep_courant": float(meep_courant),
    }
    if capture_accuracy:
        payload["planes"] = {k: v.tolist() for k, v in _extract_meep_planes(sim, cfg, resolution_nm).items()}
    return payload


def _compare_accuracy(beamz_planes: dict[str, Any], meep_planes: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for plane in ("xy", "xz", "yz"):
        b = _normalize_plane(np.asarray(beamz_planes[plane], dtype=np.float64))
        m = _normalize_plane(np.asarray(meep_planes[plane], dtype=np.float64))
        out[plane] = {
            "relative_l2": _relative_l2(b, m),
            "correlation": _corrcoef(b, m),
            "max_abs_diff": float(np.max(np.abs(b - m))) if b.size else 0.0,
        }
    out["summary"] = {
        "mean_relative_l2": float(np.mean([out[p]["relative_l2"] for p in ("xy", "xz", "yz")])),
        "mean_correlation": float(np.mean([out[p]["correlation"] for p in ("xy", "xz", "yz")])),
    }
    return out


def _run_meep_subprocess(
    cfg: PaperBenchmarkConfig,
    meep_env: str,
    args: list[str],
) -> dict[str, Any]:
    cmd = [
        "conda",
        "run",
        "-n",
        meep_env,
        "python",
        str(Path(__file__).resolve()),
        "--backend",
        "meep",
        "--emit-json-only",
    ] + args
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def _parse_resolutions(spec: str) -> list[float]:
    vals = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    if not vals:
        raise ValueError("At least one resolution is required.")
    return vals


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("beamz", "meep", "both"), default="both")
    parser.add_argument("--mode", choices=("performance", "accuracy", "both"), default="both")
    parser.add_argument(
        "--resolutions-nm",
        default="25,20,10,5,2.5",
        help="Comma-separated performance sweep in nm.",
    )
    parser.add_argument(
        "--accuracy-resolution-nm",
        type=float,
        default=20.0,
        help="Resolution used for solver-to-solver accuracy capture.",
    )
    parser.add_argument(
        "--meep-env",
        default=os.getenv("BEAMZ_MEEP_ENV", "beamz-meep"),
        help="Conda env used when meep is not importable in the current interpreter.",
    )
    parser.add_argument("--out-json", default="")
    parser.add_argument("--emit-json-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = PaperBenchmarkConfig()
    perf_resolutions = _parse_resolutions(args.resolutions_nm)

    payload: dict[str, Any] = {
        "paper_setup": {
            "domain_um": [cfg.cell_x_um, cfg.cell_y_um, cfg.cell_z_um],
            "time_fs": cfg.total_time_fs,
            "fdtdx_courant_safety_factor": cfg.courant_safety,
            "meep_courant": cfg.courant_safety / math.sqrt(3.0),
            "performance_resolutions_nm": perf_resolutions,
            "accuracy_resolution_nm": float(args.accuracy_resolution_nm),
        },
        "design": {
            "type": "plausible_silicon_compact_coupler",
            "wavelength_um": cfg.wavelength_um,
            "n_clad": cfg.n_clad,
            "n_core": cfg.n_core,
            "notes": "Synthetic silicon slab coupler used to match the paper's box, duration, and resolution sweep.",
        },
    }

    want_beamz = args.backend in {"beamz", "both"}
    want_meep = args.backend in {"meep", "both"}
    want_perf = args.mode in {"performance", "both"}
    want_acc = args.mode in {"accuracy", "both"}

    if want_perf and want_beamz:
        payload.setdefault("performance", {})["beamz"] = [
            run_beamz_single(cfg, res_nm, capture_accuracy=False) for res_nm in perf_resolutions
        ]

    if want_perf and want_meep:
        try:
            if args.backend == "meep":
                payload.setdefault("performance", {})["meep"] = [
                    run_meep_single(cfg, res_nm, capture_accuracy=False) for res_nm in perf_resolutions
                ]
            else:
                import meep  # noqa: F401

                payload.setdefault("performance", {})["meep"] = [
                    run_meep_single(cfg, res_nm, capture_accuracy=False) for res_nm in perf_resolutions
                ]
        except ModuleNotFoundError:
            if args.backend == "meep":
                raise
            meep_payload = _run_meep_subprocess(
                cfg,
                args.meep_env,
                [
                    "--mode",
                    "performance",
                    "--resolutions-nm",
                    args.resolutions_nm,
                ],
            )
            payload.setdefault("performance", {})["meep"] = meep_payload.get("performance", {}).get("meep", [])

    if want_acc and want_beamz:
        beamz_acc = run_beamz_single(cfg, args.accuracy_resolution_nm, capture_accuracy=True)
        payload.setdefault("accuracy", {})["beamz"] = beamz_acc

    if want_acc and want_meep:
        try:
            if args.backend == "meep":
                meep_acc = run_meep_single(cfg, args.accuracy_resolution_nm, capture_accuracy=True)
            else:
                import meep  # noqa: F401

                meep_acc = run_meep_single(cfg, args.accuracy_resolution_nm, capture_accuracy=True)
        except ModuleNotFoundError:
            if args.backend == "meep":
                raise
            meep_payload = _run_meep_subprocess(
                cfg,
                args.meep_env,
                [
                    "--mode",
                    "accuracy",
                    "--accuracy-resolution-nm",
                    str(args.accuracy_resolution_nm),
                ],
            )
            meep_acc = meep_payload.get("accuracy", {}).get("meep", {})
        payload.setdefault("accuracy", {})["meep"] = meep_acc

    if want_acc and {"beamz", "meep"}.issubset(payload.get("accuracy", {})):
        payload["accuracy"]["comparison"] = _compare_accuracy(
            payload["accuracy"]["beamz"]["planes"],
            payload["accuracy"]["meep"]["planes"],
        )

    output = json.dumps(payload, indent=None if args.emit_json_only else 2, sort_keys=True)
    if args.out_json:
        Path(args.out_json).write_text(output)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
