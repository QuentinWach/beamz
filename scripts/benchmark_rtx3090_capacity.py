#!/usr/bin/env python3
"""Sweep 3D CUDA FDTD resolution through the RTX 3090 memory limit.

The realistic workload keeps the physical silicon waveguide from
``examples/3D_basics/waveguide.py`` fixed while shrinking a regular-grid cell.
Each point runs in a fresh child process so allocator state and a deliberate
capacity failure cannot contaminate later timing.  A matched source-free PEC
run at every successful shape provides the custom-CUDA update ceiling.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from rtx3090_capacity import (
        BARE_WORKLOAD,
        MODAL_WORKLOAD,
        CapacityFailure,
        CapacityMeasurement,
        CapacitySweep,
        write_capacity_artifacts,
    )
except ModuleNotFoundError:
    from scripts.rtx3090_capacity import (
        BARE_WORKLOAD,
        MODAL_WORKLOAD,
        CapacityFailure,
        CapacityMeasurement,
        CapacitySweep,
        write_capacity_artifacts,
    )


DEFAULT_RESOLUTIONS_NM = (
    80.0,
    64.0,
    52.0,
    44.0,
    38.0,
    33.0,
    29.0,
    26.0,
    23.0,
    20.5,
    18.5,
    16.5,
    15.0,
    13.5,
    12.25,
    11.25,
    10.5,
    9.75,
    9.0,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _revision(root: Path) -> str:
    commit = _git(root, "rev-parse", "HEAD")
    return commit + ("-dirty" if _git(root, "status", "--porcelain") else "")


def _parse_mib(value: str) -> int:
    return int(float(value.strip())) * 2**20


def _gpu_metadata() -> dict[str, Any]:
    query = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    name, driver, total_mib, used_mib = (
        part.strip() for part in query.splitlines()[0].split(",")
    )
    smi = subprocess.run(
        ("nvidia-smi",), check=True, capture_output=True, text=True
    ).stdout
    cuda_version = next(
        (
            line.split(marker, 1)[1].split()[0]
            for line in smi.splitlines()
            for marker in ("CUDA Version:", "CUDA UMD Version:")
            if marker in line
        ),
        None,
    )
    return {
        "device": name,
        "driver_version": driver,
        "cuda_version": cuda_version,
        "total_memory_bytes": _parse_mib(total_mib),
        "used_memory_bytes": _parse_mib(used_mib),
    }


def _process_gpu_memory_bytes() -> int:
    try:
        output = subprocess.run(
            (
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return 0
    for line in output.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) == 2 and fields[0] == str(os.getpid()):
            try:
                return _parse_mib(fields[1])
            except ValueError:
                return 0
    return 0


def _child_environment(root: Path, allocator_fraction: float) -> dict[str, str]:
    """Build a child environment that always imports the revision being timed.

    A benchmark child executes this script from ``scripts/``.  Python therefore
    does not automatically put the repository root before an installed BeamZ
    package.  Make the source root explicit, just as the comparison benchmark
    does, so an editable installation or another checkout cannot silently
    replace the candidate under measurement.
    """
    environment = os.environ.copy()
    source_paths = [str(root)]
    if previous_path := environment.get("PYTHONPATH"):
        source_paths.append(previous_path)
    environment["PYTHONPATH"] = os.pathsep.join(source_paths)
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(allocator_fraction)
    environment["BEAMZ_DISABLE_JAX_PERSISTENT_CACHE"] = "1"
    environment.pop("JAX_COMPILATION_CACHE_DIR", None)
    return environment


def _looks_like_gpu_oom(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "cuda_error_out_of_memory",
            "resource exhausted",
            "out of memory",
            "failed to allocate",
            "allocation failed",
            "cuda malloc",
            "cublas_status_alloc_failed",
        )
    )


def _run_child(
    *,
    root: Path,
    workload: str,
    resolution_nm: float,
    shape: tuple[int, int, int] | None,
    timesteps: int,
    samples: int,
    warmups: int,
    allocator_fraction: float,
    timeout_s: float,
) -> CapacityMeasurement | CapacityFailure:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--workload",
        workload,
        "--resolutions-nm",
        str(resolution_nm),
        "--timesteps",
        str(timesteps),
        "--samples",
        str(samples),
        "--warmups",
        str(warmups),
        "--allocator-fraction",
        str(allocator_fraction),
    ]
    if shape is not None:
        command.extend(("--shape", *(str(value) for value in shape)))
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=_child_environment(root, allocator_fraction),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        detail = "\n".join(
            part
            for part in (
                error.stdout.decode()
                if isinstance(error.stdout, bytes)
                else error.stdout,
                error.stderr.decode()
                if isinstance(error.stderr, bytes)
                else error.stderr,
            )
            if part
        )
        return CapacityFailure(
            workload=workload,
            resolution_nm=resolution_nm,
            kind="timeout",
            returncode=-1,
            detail=detail[-4_000:],
        )
    combined = "\n".join((completed.stdout, completed.stderr)).strip()
    if completed.returncode != 0:
        return CapacityFailure(
            workload=workload,
            resolution_nm=resolution_nm,
            kind="gpu_oom" if _looks_like_gpu_oom(combined) else "child_error",
            returncode=completed.returncode,
            detail=combined[-4_000:],
        )
    try:
        payload = json.loads(completed.stdout)
        return CapacityMeasurement.from_child_payload(payload)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        return CapacityFailure(
            workload=workload,
            resolution_nm=resolution_nm,
            kind="invalid_output",
            returncode=completed.returncode,
            detail=f"{error}\n{combined[-4_000:]}",
        )


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / ".capacity-checkpoint.json"


def _write_checkpoint(
    args: argparse.Namespace,
    measurements: list[CapacityMeasurement],
    failures: list[CapacityFailure],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "schema_version": 2,
            "resolutions_nm": list(args.resolutions_nm),
            "timesteps": args.timesteps,
            "samples": args.samples,
            "warmups": args.warmups,
            "allocator_fraction": args.allocator_fraction,
            "skip_bare": args.skip_bare,
        },
        "measurements": [asdict(item) for item in measurements],
        "failures": [asdict(item) for item in failures],
    }
    _checkpoint_path(args.output_dir).write_text(json.dumps(payload, indent=2) + "\n")


def _load_checkpoint(
    args: argparse.Namespace,
) -> tuple[list[CapacityMeasurement], list[CapacityFailure]]:
    path = _checkpoint_path(args.output_dir)
    if not args.resume or not path.is_file():
        return [], []
    payload = json.loads(path.read_text())
    expected = {
        "schema_version": 2,
        "resolutions_nm": list(args.resolutions_nm),
        "timesteps": args.timesteps,
        "samples": args.samples,
        "warmups": args.warmups,
        "allocator_fraction": args.allocator_fraction,
        "skip_bare": args.skip_bare,
    }
    if payload.get("protocol") != expected:
        raise RuntimeError(
            f"checkpoint protocol differs from this run: {path}. "
            "Use --no-resume or restore the original arguments."
        )
    measurements = [
        CapacityMeasurement.from_child_payload(item)
        for item in payload.get("measurements", [])
    ]
    failures = [
        CapacityFailure(
            workload=str(item["workload"]),
            resolution_nm=float(item["resolution_nm"]),
            kind=str(item["kind"]),
            returncode=int(item["returncode"]),
            detail=str(item["detail"]),
        )
        for item in payload.get("failures", [])
    ]
    return measurements, failures


def _waveguide_simulation(resolution: float, timesteps: int):
    import numpy as np

    import beamz as bz

    micrometre = 1e-6
    wavelength = 1.55 * micrometre
    width = 6.5 * micrometre
    height = 6.5 * micrometre
    depth = 4.0 * micrometre
    dt = 0.95 * resolution / (bz.LIGHT_SPEED * math.sqrt(3.0))
    times = np.arange(timesteps, dtype=np.float64) * dt
    step = np.arange(timesteps, dtype=np.float64)
    envelope = np.exp(-(((step - 0.25 * timesteps) / (0.10 * timesteps)) ** 2))
    waveform = (
        envelope * np.sin(2.0 * np.pi * bz.LIGHT_SPEED / wavelength * times)
    ).astype(np.float32)

    design = bz.Design(
        width=width,
        height=height,
        depth=depth,
        material=bz.Material(1.0),
    )
    design += bz.Rectangle(
        position=(0.0, 0.0, 0.0),
        width=width,
        height=height,
        depth=2.0 * micrometre,
        material=bz.Material(1.44**2),
    )
    design += bz.Rectangle(
        position=(0.0, 3.0 * micrometre, 2.0 * micrometre),
        width=width,
        height=0.5 * micrometre,
        depth=0.22 * micrometre,
        material=bz.Material(3.48**2),
    )
    source = bz.ModeSource(
        center=(3.25 * micrometre, 3.25 * micrometre, 2.11 * micrometre),
        size=(0.0, 3.5 * micrometre, 0.8 * micrometre),
        source_time=bz.SampledSignal(
            waveform,
            dt=dt,
            freq0=bz.LIGHT_SPEED / wavelength,
        ),
        direction="+",
        mode_spec=bz.ModeSpec(polarization="tm"),
    )
    return bz.Simulation(
        design=design,
        sources=[source],
        monitors=[],
        boundaries=[bz.PML(edges="all", formulation="cpml")],
        time=times,
        resolution=resolution,
    )


def _bare_simulation(resolution: float, timesteps: int, shape: tuple[int, int, int]):
    import numpy as np

    import beamz as bz
    from beamz.design import MaterialGrid

    dt = 0.95 * resolution / (bz.LIGHT_SPEED * math.sqrt(3.0))
    nz, ny, nx = shape
    material_grid = MaterialGrid(
        permittivity=np.ones(shape, dtype=np.float32),
        conductivity=np.float32(0.0),
        permeability=np.float32(1.0),
        resolution=resolution,
        shape=shape,
    )
    return bz.Simulation(
        material_grid=material_grid,
        size=(nx * resolution, ny * resolution, nz * resolution),
        sources=[],
        monitors=[],
        boundaries=[bz.PEC(edges="all")],
        time=np.arange(timesteps, dtype=np.float64) * dt,
    )


def _block(state) -> None:
    state.ez.block_until_ready()


def _run_child_benchmark(args: argparse.Namespace) -> None:
    import jax
    import jaxlib

    import beamz
    from beamz.simulation import sharding as sharding_runtime
    from beamz.simulation.execute import build_scan, initial_program_state

    devices = tuple(jax.devices("gpu"))
    if len(devices) != 1:
        raise RuntimeError(f"expected exactly one visible GPU, found {devices!r}")
    device = devices[0]
    initial_stats = device.memory_stats() or {}
    allocator_limit = int(initial_stats.get("bytes_limit", 0))
    if allocator_limit <= 0:
        raise RuntimeError("JAX did not report a positive GPU allocator limit")

    resolution_nm = float(args.resolutions_nm[0])
    resolution = resolution_nm * 1e-9
    started = time.perf_counter()
    if args.workload == MODAL_WORKLOAD:
        simulation = _waveguide_simulation(resolution, args.timesteps)
    elif args.workload == BARE_WORKLOAD:
        if args.shape is None:
            raise ValueError("the bare workload requires --shape NZ NY NX")
        simulation = _bare_simulation(
            resolution, args.timesteps, tuple(int(value) for value in args.shape)
        )
    else:
        raise ValueError(f"unknown workload {args.workload!r}")
    simulation.clear_compiled_cache()
    program = simulation.compile(
        num_steps=args.timesteps,
        backend="cuda_streamed",
    )
    setup_s = time.perf_counter() - started
    shape = tuple(int(value) for value in program.grid.permittivity.shape)
    state = initial_program_state(
        program,
        t=float(simulation.time[0]),
        current_step=0,
        monitor_steps=args.timesteps,
    )
    coefficients = sharding_runtime.place_tree(program, program.coefficients)
    scan = build_scan(program, donate_state=False)

    started = time.perf_counter()
    lowered = scan.lower(state, coefficients)
    trace_lower_s = time.perf_counter() - started
    started = time.perf_counter()
    executable = lowered.compile()
    executable_compile_s = time.perf_counter() - started

    for _ in range(args.warmups):
        result = executable(state, coefficients)
        _block(result)
        del result
    gc.collect()

    samples: list[float] = []
    for _ in range(args.samples):
        started = time.perf_counter()
        result = executable(state, coefficients)
        _block(result)
        samples.append(time.perf_counter() - started)
        del result
        gc.collect()

    stats = device.memory_stats() or {}
    from beamz.simulation.backend import cuda_backend_status

    status = cuda_backend_status(register=False)
    if not status.available or status.extension_version is None:
        raise RuntimeError(status.reason or "CUDA component unavailable")
    cpml_psi_precision = (
        str(state.cpml_psi_h_terms[0].dtype) if state.cpml_psi_h_terms else "float32"
    )
    payload = {
        "workload": args.workload,
        "resolution_nm": resolution_nm,
        "grid_zyx": shape,
        "timesteps": args.timesteps,
        "warmups": args.warmups,
        "warm_runtime_samples_s": samples,
        "setup_s": setup_s,
        "trace_lower_s": trace_lower_s,
        "executable_compile_s": executable_compile_s,
        "source_spec_count": len(program.sources),
        "peak_bytes_in_use": int(stats.get("peak_bytes_in_use", 0)),
        "peak_pool_bytes": int(stats.get("peak_pool_bytes", 0)),
        "live_bytes_in_use": int(stats.get("bytes_in_use", 0)),
        "process_memory_bytes": _process_gpu_memory_bytes(),
        "allocator_limit_bytes": int(stats.get("bytes_limit", allocator_limit)),
        "backend": program.config.backend,
        "device": str(getattr(device, "device_kind", device)),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "beamz_version": beamz.__version__,
        "field_precision": str(state.ex.dtype),
        "cpml_psi_precision": cpml_psi_precision,
        "cuda_component_version": status.extension_version,
        "cuda_abi_version": status.abi_version,
        "cuda_flags": int(program.config.cuda_flags),
    }
    print(json.dumps(payload, allow_nan=False))


def run_sweep(args: argparse.Namespace) -> tuple[CapacitySweep, dict[str, Path]]:
    root = Path(__file__).resolve().parents[1]
    metadata = _gpu_metadata()
    device = str(metadata["device"])
    if not args.allow_other_gpu and "RTX 3090" not in device.upper():
        raise RuntimeError(
            f"this protocol targets an RTX 3090; detected {device!r}. "
            "Pass --allow-other-gpu for a non-canonical sweep."
        )
    started_at = _utc_now()
    measurements, failures = _load_checkpoint(args)
    if measurements or failures:
        print(
            f"resuming {len(measurements)} successful and {len(failures)} failed "
            f"attempts from {_checkpoint_path(args.output_dir)}",
            flush=True,
        )

    if args.finalize_checkpoint:
        measured_resolutions = {
            item.resolution_nm
            for item in measurements
            if item.workload == MODAL_WORKLOAD
        }
        first_unmeasured = next(
            (
                resolution
                for resolution in args.resolutions_nm
                if resolution not in measured_resolutions
            ),
            None,
        )
        if first_unmeasured is not None and not any(
            failure.kind in {"gpu_oom", "shared_gpu_safety_stop"}
            and failure.workload == MODAL_WORKLOAD
            for failure in failures
        ):
            failures.append(
                CapacityFailure(
                    workload=MODAL_WORKLOAD,
                    resolution_nm=first_unmeasured,
                    kind="shared_gpu_safety_stop",
                    returncode=-1,
                    detail=(
                        "The T3/Chromium GPU process closed twice when the sweep "
                        "crossed the 16 GiB JAX pool transition. The user requested "
                        "continuation without destabilizing the shared desktop, so "
                        "the allocation was not repeated."
                    ),
                )
            )
            _write_checkpoint(args, measurements, failures)

    resolutions_to_run = () if args.finalize_checkpoint else args.resolutions_nm
    for index, resolution_nm in enumerate(resolutions_to_run, start=1):
        existing_modal_oom = next(
            (
                failure
                for failure in failures
                if failure.workload == MODAL_WORKLOAD
                and failure.resolution_nm == resolution_nm
                and failure.kind == "gpu_oom"
            ),
            None,
        )
        if existing_modal_oom is not None:
            print(
                f"[{index}/{len(args.resolutions_nm)}] resumed GPU OOM at "
                f"{resolution_nm:g} nm",
                flush=True,
            )
            break

        modal = next(
            (
                item
                for item in measurements
                if item.workload == MODAL_WORKLOAD
                and item.resolution_nm == resolution_nm
            ),
            None,
        )
        if modal is None:
            print(
                f"[{index}/{len(args.resolutions_nm)}] realistic modal waveguide at "
                f"{resolution_nm:g} nm",
                flush=True,
            )
            attempt = _run_child(
                root=root,
                workload=MODAL_WORKLOAD,
                resolution_nm=resolution_nm,
                shape=None,
                timesteps=args.timesteps,
                samples=args.samples,
                warmups=args.warmups,
                allocator_fraction=args.allocator_fraction,
                timeout_s=args.child_timeout,
            )
            if isinstance(attempt, CapacityFailure):
                failures.append(attempt)
                _write_checkpoint(args, measurements, failures)
                print(
                    f"  {attempt.kind}: child return code {attempt.returncode}",
                    flush=True,
                )
                if attempt.kind != "gpu_oom":
                    raise RuntimeError(
                        "capacity child failed before a valid GPU OOM:\n"
                        f"{attempt.detail}"
                    )
                break
            modal = attempt
            measurements.append(modal)
            _write_checkpoint(args, measurements, failures)
            print(
                f"  {modal.cells:,} cells, {modal.median_gcups:.3f} GCUPS, "
                f"{modal.peak_memory_gib:.2f} GiB active peak",
                flush=True,
            )
        else:
            print(
                f"[{index}/{len(args.resolutions_nm)}] resumed {resolution_nm:g} nm: "
                f"{modal.cells:,} cells, {modal.median_gcups:.3f} GCUPS",
                flush=True,
            )

        if args.skip_bare:
            continue

        bare = next(
            (
                item
                for item in measurements
                if item.workload == BARE_WORKLOAD
                and item.resolution_nm == resolution_nm
            ),
            None,
        )
        existing_bare_oom = any(
            failure.workload == BARE_WORKLOAD
            and failure.resolution_nm == resolution_nm
            and failure.kind == "gpu_oom"
            for failure in failures
        )
        if bare is None and not existing_bare_oom:
            print("  matched bare PEC ceiling", flush=True)
            bare_attempt = _run_child(
                root=root,
                workload=BARE_WORKLOAD,
                resolution_nm=resolution_nm,
                shape=modal.grid_zyx,
                timesteps=args.timesteps,
                samples=args.samples,
                warmups=args.warmups,
                allocator_fraction=args.allocator_fraction,
                timeout_s=args.child_timeout,
            )
            if isinstance(bare_attempt, CapacityFailure):
                failures.append(bare_attempt)
                _write_checkpoint(args, measurements, failures)
                if bare_attempt.kind != "gpu_oom":
                    raise RuntimeError(f"bare child failed:\n{bare_attempt.detail}")
                print("  bare workload reached GPU OOM", flush=True)
            else:
                measurements.append(bare_attempt)
                _write_checkpoint(args, measurements, failures)
                print(
                    f"  {bare_attempt.median_gcups:.3f} GCUPS, "
                    f"{bare_attempt.peak_memory_gib:.2f} GiB active peak",
                    flush=True,
                )
        elif bare is not None:
            print(
                f"  resumed bare ceiling: {bare.median_gcups:.3f} GCUPS",
                flush=True,
            )

    if not measurements:
        detail = failures[-1].detail if failures else "no child attempts completed"
        raise RuntimeError(f"capacity sweep produced no measurements:\n{detail}")
    sweep = CapacitySweep(
        beamz_revision=_revision(root),
        device=device,
        driver_version=(
            None
            if metadata["driver_version"] is None
            else str(metadata["driver_version"])
        ),
        cuda_version=(
            None if metadata["cuda_version"] is None else str(metadata["cuda_version"])
        ),
        total_gpu_memory_bytes=int(metadata["total_memory_bytes"]),
        baseline_gpu_memory_bytes=int(metadata["used_memory_bytes"]),
        allocator_fraction=args.allocator_fraction,
        timesteps=args.timesteps,
        samples=args.samples,
        warmups=args.warmups,
        started_at=started_at,
        completed_at=_utc_now(),
        measurements=tuple(measurements),
        failures=tuple(failures),
    )
    return sweep, write_capacity_artifacts(sweep, args.output_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolutions-nm",
        nargs="+",
        type=float,
        default=DEFAULT_RESOLUTIONS_NM,
        help="regular-grid cell sizes, ordered from coarse to fine",
    )
    parser.add_argument("--timesteps", type=int, default=192)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument(
        "--allocator-fraction",
        type=float,
        default=0.92,
        help="fraction of physical VRAM exposed to the JAX allocator",
    )
    parser.add_argument("--child-timeout", type=float, default=900.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/rtx3090-capacity"),
    )
    parser.add_argument("--skip-bare", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume a matching per-point checkpoint in the output directory",
    )
    parser.add_argument(
        "--finalize-checkpoint",
        action="store_true",
        help="write artifacts from saved points and record a shared-GPU safety stop",
    )
    parser.add_argument("--allow-other-gpu", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--workload",
        choices=(MODAL_WORKLOAD, BARE_WORKLOAD),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--shape", nargs=3, type=int, metavar=("NZ", "NY", "NX"), help=argparse.SUPPRESS
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.samples < 3:
        raise SystemExit("--samples must be at least three")
    if args.timesteps <= 0 or args.warmups <= 0:
        raise SystemExit("--timesteps and --warmups must be positive")
    if not 0.0 < args.allocator_fraction <= 1.0:
        raise SystemExit("--allocator-fraction must lie in (0, 1]")
    if args.child_timeout <= 0.0:
        raise SystemExit("--child-timeout must be positive")
    resolutions = tuple(float(value) for value in args.resolutions_nm)
    if any(not math.isfinite(value) or value <= 0.0 for value in resolutions):
        raise SystemExit("all --resolutions-nm values must be positive and finite")
    if any(
        left <= right for left, right in zip(resolutions, resolutions[1:], strict=False)
    ):
        raise SystemExit("--resolutions-nm must be strictly decreasing")
    if args.child and args.workload is None:
        raise SystemExit("--child requires --workload")
    if args.child and args.finalize_checkpoint:
        raise SystemExit("--finalize-checkpoint is a parent-process option")


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    if args.child:
        _run_child_benchmark(args)
        return
    sweep, paths = run_sweep(args)
    modal_summary = sweep.summary[MODAL_WORKLOAD]
    print(
        f"realistic peak: {modal_summary['best_median_gcups']:.3f} GCUPS; "
        f"largest success: {modal_summary['largest_successful_cells']:,} cells"
    )
    for label, path in paths.items():
        print(f"{label}: {path}")
    if not any(failure.kind == "gpu_oom" for failure in sweep.failures):
        if any(failure.kind == "shared_gpu_safety_stop" for failure in sweep.failures):
            print(
                "note: capacity is a safe shared-desktop lower bound; no further "
                "full-VRAM allocation was attempted",
                file=sys.stderr,
            )
        else:
            print(
                "warning: resolution list ended before a GPU OOM; capacity is only "
                "lower-bounded",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
