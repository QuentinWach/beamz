from __future__ import annotations

import json
import os
from argparse import Namespace
from datetime import datetime, timezone

import pytest

from scripts.benchmark_rtx3090_capacity import (
    _child_environment,
    _load_checkpoint,
    _looks_like_gpu_oom,
    _waveguide_simulation,
    _write_checkpoint,
)
from scripts.rtx3090_capacity import (
    BARE_WORKLOAD,
    MODAL_WORKLOAD,
    CapacityFailure,
    CapacityMeasurement,
    CapacitySweep,
    write_capacity_artifacts,
)


def _measurement(
    workload: str,
    shape: tuple[int, int, int],
    gcups: float,
    resolution_nm: float,
    peak_bytes: int,
) -> CapacityMeasurement:
    timesteps = 100
    cells = shape[0] * shape[1] * shape[2]
    runtime = cells * timesteps / (gcups * 1e9)
    return CapacityMeasurement(
        workload=workload,
        resolution_nm=resolution_nm,
        grid_zyx=shape,
        timesteps=timesteps,
        warmups=2,
        warm_runtime_samples_s=tuple(
            runtime * factor for factor in (1.02, 0.99, 1.00, 1.01, 0.98)
        ),
        setup_s=1.0,
        trace_lower_s=0.01,
        executable_compile_s=0.1,
        source_spec_count=8 if workload == MODAL_WORKLOAD else 0,
        peak_bytes_in_use=peak_bytes,
        peak_pool_bytes=int(peak_bytes * 1.08),
        live_bytes_in_use=int(peak_bytes * 0.85),
        process_memory_bytes=int(peak_bytes * 1.12),
        allocator_limit_bytes=22 * 2**30,
        backend="cuda_streamed",
        field_precision="float32",
        cpml_psi_precision="float32",
        python_version="3.11.15",
        jax_version="0.9.0",
        jaxlib_version="0.9.0",
        beamz_version="0.4.3",
        cuda_component_version="0.12.0",
        cuda_abi_version=12,
        cuda_flags=128,
    )


def _sweep() -> CapacitySweep:
    measurements = []
    for scale in (1, 2, 3, 4, 5):
        shape = (16 * scale, 24 * scale, 32 * scale)
        measurements.extend(
            (
                _measurement(
                    MODAL_WORKLOAD,
                    shape,
                    4.0 + 0.8 * scale,
                    80.0 / scale,
                    300_000_000 + scale**3 * 120_000_000,
                ),
                _measurement(
                    BARE_WORKLOAD,
                    shape,
                    7.0 + scale,
                    80.0 / scale,
                    200_000_000 + scale**3 * 70_000_000,
                ),
            )
        )
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return CapacitySweep(
        beamz_revision="abc123",
        device="NVIDIA GeForce RTX 3090",
        driver_version="610.43.03",
        cuda_version="13.0",
        total_gpu_memory_bytes=24 * 2**30,
        baseline_gpu_memory_bytes=600 * 2**20,
        allocator_fraction=0.92,
        timesteps=100,
        samples=5,
        warmups=2,
        started_at=now,
        completed_at=now,
        measurements=tuple(measurements),
        failures=(
            CapacityFailure(
                workload=MODAL_WORKLOAD,
                resolution_nm=12.0,
                kind="gpu_oom",
                returncode=1,
                detail="RESOURCE_EXHAUSTED",
            ),
        ),
    )


def test_capacity_measurement_derives_gcups_interval_and_memory():
    measurement = _measurement(
        MODAL_WORKLOAD,
        (32, 48, 64),
        8.0,
        40.0,
        2**30,
    )

    assert measurement.cells == 32 * 48 * 64
    assert measurement.median_gcups == pytest.approx(8.0)
    assert measurement.gcups_ci95[0] <= 8.0 <= measurement.gcups_ci95[1]
    assert measurement.peak_memory_gib == pytest.approx(1.0)
    assert measurement.allocator_utilization > 0.0


def test_capacity_summary_reports_peak_plateau_fit_and_oom_bracket():
    summary = _sweep().summary

    assert summary[MODAL_WORKLOAD]["point_count"] == 5
    assert summary[MODAL_WORKLOAD]["best_median_gcups"] == pytest.approx(8.0)
    assert summary[MODAL_WORKLOAD]["saturated_point_count"] >= 2
    assert summary[BARE_WORKLOAD]["best_median_gcups"] == pytest.approx(12.0)
    assert summary["capacity"]["first_gpu_oom_resolution_nm"] == pytest.approx(12.0)
    assert summary["capacity"]["fitted_bytes_per_cell"] > 0.0
    assert summary["bare_to_modal_best_gcups_ratio"] == pytest.approx(1.5)


def test_capacity_artifacts_include_raw_data_graph_and_report_contract(tmp_path):
    paths = write_capacity_artifacts(_sweep(), tmp_path)

    payload = json.loads(paths["json"].read_text())
    artifact = json.loads(paths["artifact"].read_text())
    assert payload["schema_version"] == "beamz.performance/rtx3090-capacity-v2"
    assert payload["execution_provenance"]["cuda_abi_version"] == 12
    assert len(payload["measurements"]) == 10
    assert paths["csv"].read_text().startswith("workload,resolution_nm")
    assert paths["graph"].read_bytes().startswith(b"\x89PNG")
    assert artifact["surface"] == "report"
    assert artifact["snapshot"]["status"] == "ready"
    assert len(artifact["manifest"]["charts"]) == 3
    assert artifact["manifest"]["blocks"][0]["body"].startswith("# RTX 3090 CUDA FDTD")


@pytest.mark.parametrize(
    "message",
    (
        "CUDA_ERROR_OUT_OF_MEMORY",
        "RESOURCE_EXHAUSTED: failed to allocate 1.2GiB",
        "cuBLAS status CUBLAS_STATUS_ALLOC_FAILED",
    ),
)
def test_gpu_oom_classifier_recognizes_allocator_failures(message):
    assert _looks_like_gpu_oom(message)


def test_gpu_oom_classifier_does_not_hide_unrelated_child_errors():
    assert not _looks_like_gpu_oom("ValueError: modal plane is outside the domain")


def test_capacity_child_environment_prefers_the_requested_source_root(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("PYTHONPATH", "/an/unrelated/beamz")

    environment = _child_environment(tmp_path, 0.92)

    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(tmp_path),
        "/an/unrelated/beamz",
    ]


def test_capacity_checkpoint_round_trips_completed_attempts(tmp_path):
    args = Namespace(
        output_dir=tmp_path,
        resolutions_nm=(80.0, 40.0),
        timesteps=100,
        samples=5,
        warmups=2,
        allocator_fraction=0.92,
        skip_bare=False,
        resume=True,
    )
    measurement = _measurement(MODAL_WORKLOAD, (32, 48, 64), 8.0, 80.0, 2**30)
    failure = CapacityFailure(
        workload=MODAL_WORKLOAD,
        resolution_nm=40.0,
        kind="gpu_oom",
        returncode=1,
        detail="RESOURCE_EXHAUSTED",
    )

    _write_checkpoint(args, [measurement], [failure])
    measurements, failures = _load_checkpoint(args)

    assert measurements == [measurement]
    assert failures == [failure]


def test_modal_capacity_workload_explicitly_uses_cpml():
    simulation = _waveguide_simulation(80e-9, 4)

    assert len(simulation.boundaries) == 1
    assert simulation.boundaries[0].formulation == "cpml"
    assert simulation.boundaries[0].thickness is None
    assert simulation.boundaries[0].DEFAULT_CELLS == 12
