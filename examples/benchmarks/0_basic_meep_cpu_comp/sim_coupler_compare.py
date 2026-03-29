from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = next(
    candidate
    for candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents)
    if (candidate / "beamz").exists() and (candidate / "examples").exists()
)
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
        return [
            (x - 0.5 * self.cell_x_um, y - 0.5 * self.cell_y_um)
            for x, y, _ in self.beamz_polygon_vertices_um
        ]


def _beamz_xyz(
    x_um: float,
    y_um: float,
    z_um: float,
    cfg: PaperBenchmarkConfig,
) -> tuple[float, float, float]:
    del cfg
    return (x_um * 1e-6, y_um * 1e-6, z_um * 1e-6)


def _rect_position_from_center(
    x0_um: float,
    x1_um: float,
    yc_um: float,
    width_um: float,
    z0_um: float,
) -> tuple[float, float, float]:
    return (x0_um * 1e-6, (yc_um - 0.5 * width_um) * 1e-6, z0_um * 1e-6)


def _num_steps_for_resolution(
    resolution_nm: float, cfg: PaperBenchmarkConfig
) -> tuple[int, float]:
    dx_m = resolution_nm * 1e-9
    dt_s = cfg.courant_safety * dx_m / (C0 * math.sqrt(3.0))
    num_steps = int(math.floor(cfg.total_time_s / dt_s))
    return max(num_steps, 2), dt_s


def _paper_cells(
    resolution_nm: float, cfg: PaperBenchmarkConfig
) -> tuple[int, int, int]:
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
    total_den = max(total_s, 1e-12)
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
        "gcups_run": float(cell_updates / run_den / 1e9),
        "gcups_total": float(cell_updates / total_den / 1e9),
        "gcompups_run": float(comp_updates / run_den / 1e9),
        "gcompups_total": float(comp_updates / total_den / 1e9),
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
    raise ValueError(
        f"Plane shape mismatch: got {arr.shape}, expected {target_shape}"
    )


def _center_component(
    arr: np.ndarray, axis: int, target_shape: tuple[int, int, int]
) -> np.ndarray:
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
        vertices=[
            _beamz_xyz(x, y, z, cfg)
            for x, y, z in cfg.beamz_polygon_vertices_um
        ],
        material=Material(cfg.n_core**2),
        depth=cfg.slab_thickness_um * 1e-6,
        z=cfg.slab_z0_um * 1e-6,
    )

    grid = design.rasterize(resolution=dx_m)
    source = ModeSource(
        grid=grid,
        center=_beamz_xyz(
            cfg.source_x_um, cfg.source_y_um, cfg.slab_center_z_um, cfg
        ),
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
            grid_shape=tuple(
                int(v) for v in np.asarray(sim.fields.permittivity).shape
            ),
        ),
    }
    if capture_accuracy:
        payload["planes"] = {
            k: v.tolist() for k, v in _extract_beamz_planes(sim).items()
        }
    return payload


def _meep_center(
    x_um: float, y_um: float, z_um: float, cfg: PaperBenchmarkConfig
) -> tuple[float, float, float]:
    return (
        x_um - 0.5 * cfg.cell_x_um,
        y_um - 0.5 * cfg.cell_y_um,
        z_um - 0.5 * cfg.cell_z_um,
    )


def _extract_meep_planes(
    sim, cfg: PaperBenchmarkConfig, resolution_nm: float
) -> dict[str, np.ndarray]:
    import meep as mp

    nx, ny, nz = _paper_cells(resolution_nm, cfg)

    def e_mag(
        center: mp.Vector3, size: mp.Vector3, target_shape: tuple[int, int]
    ) -> np.ndarray:
        ex = _shape_or_transpose(
            np.asarray(sim.get_array(center=center, size=size, component=mp.Ex)),
            target_shape,
        )
        ey = _shape_or_transpose(
            np.asarray(sim.get_array(center=center, size=size, component=mp.Ey)),
            target_shape,
        )
        ez = _shape_or_transpose(
            np.asarray(sim.get_array(center=center, size=size, component=mp.Ez)),
            target_shape,
        )
        return np.sqrt(np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2)

    return {
        "xy": e_mag(
            mp.Vector3(0, 0, 0),
            mp.Vector3(cfg.cell_x_um, cfg.cell_y_um, 0),
            (ny, nx),
        ),
        "xz": e_mag(
            mp.Vector3(0, 0, 0),
            mp.Vector3(cfg.cell_x_um, 0, cfg.cell_z_um),
            (nz, nx),
        ),
        "yz": e_mag(
            mp.Vector3(0, 0, 0),
            mp.Vector3(0, cfg.cell_y_um, cfg.cell_z_um),
            (nz, ny),
        ),
    }


def run_meep_single(
    cfg: PaperBenchmarkConfig,
    resolution_nm: float,
    *,
    capture_accuracy: bool,
) -> dict[str, Any]:
    import meep as mp

    num_steps, dt_s = _num_steps_for_resolution(resolution_nm, cfg)
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
    output_cx = (
        0.5 * (cfg.output_wg_x0_um + cfg.output_wg_x1_um) - 0.5 * cfg.cell_x_um
    )
    output_cy = cfg.output_wg_yc_um - 0.5 * cfg.cell_y_um

    geometry = [
        mp.Block(
            size=mp.Vector3(
                cfg.input_wg_x1_um - cfg.input_wg_x0_um,
                cfg.input_wg_width_um,
                cfg.slab_thickness_um,
            ),
            center=mp.Vector3(input_cx, input_cy, slab_center),
            material=core,
        ),
        mp.Block(
            size=mp.Vector3(
                cfg.output_wg_x1_um - cfg.output_wg_x0_um,
                cfg.output_wg_width_um,
                cfg.slab_thickness_um,
            ),
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
    sx, sy, sz = _meep_center(
        cfg.source_x_um, cfg.source_y_um, cfg.slab_center_z_um, cfg
    )
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
        payload["planes"] = {
            k: v.tolist()
            for k, v in _extract_meep_planes(sim, cfg, resolution_nm).items()
        }
    return payload


def _compare_accuracy(
    beamz_planes: dict[str, Any], meep_planes: dict[str, Any]
) -> dict[str, Any]:
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
        "mean_relative_l2": float(
            np.mean([out[p]["relative_l2"] for p in ("xy", "xz", "yz")])
        ),
        "mean_correlation": float(
            np.mean([out[p]["correlation"] for p in ("xy", "xz", "yz")])
        ),
    }
    return out


def _run_meep_subprocess(
    meep_env: str, args: list[str], *, emit_json_only: bool = True
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
    ]
    if emit_json_only:
        cmd.append("--emit-json-only")
    cmd.extend(args)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr or stdout or f"exit status {proc.returncode}"
        env_hint = (
            f"Meep subprocess failed for conda env '{meep_env}'. "
            "If the environment does not exist yet, create it with:\n"
            "  conda env create -f examples/benchmarks/meep_environment.yml\n"
            f"Underlying error: {detail}"
        )
        raise RuntimeError(env_hint)
    try:
        return _extract_json_object(proc.stdout)
    except ValueError as exc:
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        detail = stderr or stdout or "no subprocess output captured"
        raise RuntimeError(
            "Meep subprocess completed but did not return a parseable JSON payload. "
            f"Captured output: {detail}"
        ) from exc


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("No JSON object found in subprocess stdout.")


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


def _safe_slug(spec: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in spec)


def _sample_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "ci95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    sem = float(std / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {
        "count": int(arr.size),
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95": float(1.96 * sem),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _paper_row_info(
    resolution_nm: float, cfg: PaperBenchmarkConfig
) -> dict[str, Any]:
    nx, ny, nz = _paper_cells(resolution_nm, cfg)
    num_steps, _ = _num_steps_for_resolution(resolution_nm, cfg)
    return {
        "resolution_nm": float(resolution_nm),
        "cells": int(nx * ny * nz),
        "steps": int(num_steps),
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
    }


def _build_interleaved_schedule(
    backends: list[str],
    resolutions_nm: list[float],
    repeats: int,
) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    order_index = 0
    for round_index in range(repeats):
        backend_order = list(backends)
        resolution_order = list(resolutions_nm)
        if round_index % 2 == 1:
            backend_order.reverse()
            resolution_order.reverse()
        for resolution_nm in resolution_order:
            for backend in backend_order:
                schedule.append(
                    {
                        "order_index": order_index,
                        "round_index": round_index,
                        "repeat_index": round_index + 1,
                        "backend": backend,
                        "resolution_nm": float(resolution_nm),
                    }
                )
                order_index += 1
    return schedule


def _run_single_backend_entry(
    entry: dict[str, Any],
    cfg: PaperBenchmarkConfig,
    *,
    meep_env: str,
    meep_available: bool,
) -> dict[str, Any]:
    backend = str(entry["backend"])
    resolution_nm = float(entry["resolution_nm"])
    if backend == "beamz":
        result = run_beamz_single(cfg, resolution_nm, capture_accuracy=False)
    elif backend == "meep":
        if meep_available:
            result = run_meep_single(cfg, resolution_nm, capture_accuracy=False)
        else:
            payload = _run_meep_subprocess(
                meep_env,
                [
                    "--mode",
                    "performance",
                    "--resolutions-nm",
                    str(resolution_nm),
                    "--performance-repeats",
                    "1",
                ],
            )
            raw_runs = payload.get("performance", {}).get("raw_runs", [])
            if not raw_runs:
                raise RuntimeError(
                    "Nested Meep performance run returned no raw run entries."
                )
            result = {"runtime": raw_runs[0]}
    else:
        raise ValueError(f"Unsupported backend {backend!r}")

    runtime = dict(result["runtime"])
    runtime["resolution_nm"] = resolution_nm
    runtime["backend"] = backend
    runtime["order_index"] = int(entry["order_index"])
    runtime["round_index"] = int(entry["round_index"])
    runtime["repeat_index"] = int(entry["repeat_index"])
    runtime["timestamp_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    return runtime


def _summarize_performance_runs(
    rows: list[dict[str, Any]], cfg: PaperBenchmarkConfig
) -> list[dict[str, Any]]:
    metrics = (
        "setup_s",
        "compile_s",
        "run_s",
        "total_s",
        "gcups_run",
        "gcups_total",
        "gcompups_run",
        "gcompups_total",
    )
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["backend"]), float(row["resolution_nm"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (backend, resolution_nm), group in sorted(
        grouped.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        base = {
            "backend": backend,
            **_paper_row_info(resolution_nm, cfg),
            "repeats": len(group),
        }
        for metric in metrics:
            stats = _sample_stats([float(row[metric]) for row in group])
            for key, value in stats.items():
                suffix = "n" if key == "count" else key
                base[f"{metric}_{suffix}"] = value
        summary_rows.append(base)
    return summary_rows


def _build_paper_table(
    summary_rows: list[dict[str, Any]], cfg: PaperBenchmarkConfig
) -> list[dict[str, Any]]:
    by_key = {
        (str(row["backend"]), float(row["resolution_nm"])): row for row in summary_rows
    }
    backends = sorted({str(row["backend"]) for row in summary_rows})
    rows: list[dict[str, Any]] = []
    for resolution_nm in sorted({float(row["resolution_nm"]) for row in summary_rows}):
        row = _paper_row_info(resolution_nm, cfg)
        for backend in backends:
            src = by_key.get((backend, resolution_nm))
            if src is None:
                continue
            for metric in ("run_s", "total_s", "gcups_run", "gcups_total"):
                row[f"{backend}_{metric}_mean"] = src[f"{metric}_mean"]
                row[f"{backend}_{metric}_std"] = src[f"{metric}_std"]
                row[f"{backend}_{metric}_sem"] = src[f"{metric}_sem"]
                row[f"{backend}_{metric}_ci95"] = src[f"{metric}_ci95"]
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    headers: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                headers.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _default_results_dir(
    backend: str, mode: str, resolutions_nm: list[float], repeats: int
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_slug = _safe_slug("-".join(f"{v:g}" for v in resolutions_nm))
    return (
        REPO_ROOT
        / "benchmarks"
        / "results"
        / f"paper_style_coupler_{backend}_{mode}_{res_slug}_r{repeats}_{stamp}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("beamz", "meep", "both"), default="both"
    )
    parser.add_argument(
        "--mode", choices=("performance", "accuracy", "both"), default="both"
    )
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
        "--performance-repeats",
        type=int,
        default=3,
        help="Number of interleaved repeats per backend/resolution for performance mode.",
    )
    parser.add_argument(
        "--results-dir",
        default="",
        help="Directory for JSON/CSV benchmark artifacts.",
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
    if args.performance_repeats < 1:
        raise ValueError("--performance-repeats must be >= 1.")

    try:
        import meep as _meep  # noqa: F401

        meep_available = True
    except ModuleNotFoundError:
        meep_available = False

    payload: dict[str, Any] = {
        "paper_setup": {
            "domain_um": [cfg.cell_x_um, cfg.cell_y_um, cfg.cell_z_um],
            "time_fs": cfg.total_time_fs,
            "fdtdx_courant_safety_factor": cfg.courant_safety,
            "meep_courant": cfg.courant_safety / math.sqrt(3.0),
            "performance_resolutions_nm": perf_resolutions,
            "accuracy_resolution_nm": float(args.accuracy_resolution_nm),
            "performance_repeats": int(args.performance_repeats),
        },
        "design": {
            "type": "plausible_silicon_compact_coupler",
            "wavelength_um": cfg.wavelength_um,
            "n_clad": cfg.n_clad,
            "n_core": cfg.n_core,
            "notes": (
                "Synthetic silicon slab coupler used to match the paper's box, "
                "duration, and resolution sweep."
            ),
        },
    }

    want_beamz = args.backend in {"beamz", "both"}
    want_meep = args.backend in {"meep", "both"}
    want_perf = args.mode in {"performance", "both"}
    want_acc = args.mode in {"accuracy", "both"}

    result_dir: Path | None = None
    if not args.emit_json_only:
        result_dir = (
            Path(args.results_dir)
            if args.results_dir
            else _default_results_dir(
                args.backend,
                args.mode,
                perf_resolutions,
                int(args.performance_repeats),
            )
        )
        result_dir.mkdir(parents=True, exist_ok=True)

    if want_perf:
        perf_backends: list[str] = []
        if want_beamz:
            perf_backends.append("beamz")
        if want_meep:
            perf_backends.append("meep")
        schedule = _build_interleaved_schedule(
            perf_backends, perf_resolutions, int(args.performance_repeats)
        )
        raw_rows: list[dict[str, Any]] = []
        for entry in schedule:
            raw_rows.append(
                _run_single_backend_entry(
                    entry,
                    cfg,
                    meep_env=args.meep_env,
                    meep_available=meep_available,
                )
            )
        summary_rows = _summarize_performance_runs(raw_rows, cfg)
        table_rows = _build_paper_table(summary_rows, cfg)
        payload["performance"] = {
            "schedule": schedule,
            "raw_runs": raw_rows,
            "summary": summary_rows,
            "paper_table": table_rows,
        }

        if result_dir is not None:
            raw_csv = result_dir / "performance_raw_runs.csv"
            summary_csv = result_dir / "performance_summary.csv"
            table_csv = result_dir / "performance_paper_table.csv"
            _write_csv(raw_csv, raw_rows)
            _write_csv(summary_csv, summary_rows)
            _write_csv(table_csv, table_rows)
            payload.setdefault("artifacts", {})["performance_raw_csv"] = str(raw_csv)
            payload.setdefault("artifacts", {})["performance_summary_csv"] = str(
                summary_csv
            )
            payload.setdefault("artifacts", {})["performance_paper_table_csv"] = str(
                table_csv
            )

    if want_acc and want_beamz:
        beamz_acc = run_beamz_single(
            cfg, args.accuracy_resolution_nm, capture_accuracy=True
        )
        payload.setdefault("accuracy", {})["beamz"] = beamz_acc

    if want_acc and want_meep:
        if meep_available:
            meep_acc = run_meep_single(
                cfg, args.accuracy_resolution_nm, capture_accuracy=True
            )
        else:
            meep_payload = _run_meep_subprocess(
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

    if result_dir is not None:
        payload.setdefault("artifacts", {})["results_dir"] = str(result_dir)

    output = json.dumps(
        payload, indent=None if args.emit_json_only else 2, sort_keys=True
    )
    if args.out_json:
        Path(args.out_json).write_text(output)
    elif result_dir is not None:
        manifest = result_dir / "benchmark_manifest.json"
        manifest.write_text(output)
        payload.setdefault("artifacts", {})["manifest_json"] = str(manifest)
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
