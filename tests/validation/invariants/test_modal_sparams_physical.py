from __future__ import annotations

import numpy as np
import pytest

from tests.modal_sparam_physical_case import (
    StraightWaveguideSParamConfig,
    run_straight_waveguide_sparam_case,
    summarize_center_metrics,
)


@pytest.mark.simulation
@pytest.mark.compiled
def test_straight_waveguide_modal_sparams_are_physical_across_resolution_and_distance(
    validation_metrics,
):
    cfg = StraightWaveguideSParamConfig()
    results = [
        run_straight_waveguide_sparam_case(resolution_ppw=ppw, cfg=cfg)
        for ppw in (10, 12)
    ]

    center_metrics = {r.resolution_ppw: summarize_center_metrics(r) for r in results}
    for result in results:
        metrics = center_metrics[result.resolution_ppw]
        resolution = f"{result.resolution_ppw} ppw"
        metadata = {
            "device": "straight slab waveguide",
            "wavelength_um": cfg.wavelength_um,
            "frequency_bins": len(result.frequencies_hz),
            "steps": result.steps_to_decay,
            "dx_nm": result.dx_nm,
        }
        validation_metrics.check_upper(
            "straight-waveguide S11",
            measured=metrics["s11_db"],
            upper_bound=-35.0,
            unit="dB",
            resolution=resolution,
            metadata=metadata,
        )
        for monitor_name in ("mid", "far"):
            monitor_metadata = {**metadata, "monitor": monitor_name}
            validation_metrics.check_upper(
                "absolute straight-waveguide S21 insertion loss",
                measured=abs(metrics[f"s21_{monitor_name}_db"]),
                upper_bound=0.01,
                unit="dB",
                resolution=resolution,
                metadata=monitor_metadata,
            )
            validation_metrics.check(
                "guided S-parameter power sum",
                measured=metrics[f"power_sum_{monitor_name}"],
                reference=1.0,
                tolerance="normalized_power_balance",
                unit="fraction",
                resolution=resolution,
                metadata=monitor_metadata,
            )
            validation_metrics.check_upper(
                "reference-plane phase-linearity residual",
                measured=result.phase_residual_rad_by_monitor[monitor_name],
                upper_bound=0.01,
                unit="rad",
                resolution=resolution,
                metadata=monitor_metadata,
            )
            validation_metrics.check_upper(
                "modal projection condition number",
                measured=float(np.nanmax(result.condition_numbers[monitor_name])),
                upper_bound=1e3,
                resolution=resolution,
                metadata=monitor_metadata,
            )

        mid = np.asarray(result.s21_by_monitor["mid"], dtype=np.complex128)
        far = np.asarray(result.s21_by_monitor["far"], dtype=np.complex128)
        mag_delta_db = 20.0 * np.log10(
            np.maximum(np.abs(far), 1e-12)
        ) - 20.0 * np.log10(np.maximum(np.abs(mid), 1e-12))
        validation_metrics.check_upper(
            "reference-plane S21 magnitude change",
            measured=float(np.max(np.abs(mag_delta_db))),
            upper_bound=0.01,
            unit="dB",
            resolution=resolution,
            metadata=metadata,
        )

    ppw10, ppw12 = results
    for monitor_name in ("mid", "far"):
        s21_10 = np.asarray(ppw10.s21_by_monitor[monitor_name], dtype=np.complex128)
        s21_12 = np.asarray(ppw12.s21_by_monitor[monitor_name], dtype=np.complex128)
        center = int(np.argmin(np.abs(ppw10.wavelengths_um - cfg.wavelength_um)))
        mag10 = 20.0 * np.log10(max(float(abs(s21_10[center])), 1e-12))
        mag12 = 20.0 * np.log10(max(float(abs(s21_12[center])), 1e-12))
        validation_metrics.check_upper(
            f"S21 {monitor_name} resolution change",
            measured=abs(mag12 - mag10),
            upper_bound=0.01,
            unit="dB",
            resolution="10 -> 12 ppw",
            metadata={"device": "straight slab waveguide", "monitor": monitor_name},
        )

    slope_10 = ppw10.phase_slope_s_by_monitor["far"]
    slope_12 = ppw12.phase_slope_s_by_monitor["far"]
    validation_metrics.check_upper(
        "reference-plane group-delay resolution change",
        measured=abs(slope_12 - slope_10),
        upper_bound=0.5e-15,
        unit="s",
        resolution="10 -> 12 ppw",
        metadata={"device": "straight slab waveguide", "monitor": "far"},
    )
