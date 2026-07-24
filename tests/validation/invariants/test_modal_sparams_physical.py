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
def test_straight_waveguide_modal_sparams_are_physical_across_resolution_and_distance():
    cfg = StraightWaveguideSParamConfig()
    results = [
        run_straight_waveguide_sparam_case(resolution_ppw=ppw, cfg=cfg)
        for ppw in (10, 12)
    ]

    center_metrics = {r.resolution_ppw: summarize_center_metrics(r) for r in results}
    for result in results:
        metrics = center_metrics[result.resolution_ppw]
        assert metrics["s11_db"] < -30.0
        for monitor_name in ("mid", "far"):
            assert abs(metrics[f"s21_{monitor_name}_db"]) < 0.10
            assert abs(metrics[f"power_sum_{monitor_name}"] - 1.0) < 0.03
            assert result.phase_residual_rad_by_monitor[monitor_name] < 0.08
            assert np.nanmax(result.condition_numbers[monitor_name]) < 1e3

        mid = np.asarray(result.s21_by_monitor["mid"], dtype=np.complex128)
        far = np.asarray(result.s21_by_monitor["far"], dtype=np.complex128)
        mag_delta_db = 20.0 * np.log10(
            np.maximum(np.abs(far), 1e-12)
        ) - 20.0 * np.log10(np.maximum(np.abs(mid), 1e-12))
        assert float(np.max(np.abs(mag_delta_db))) < 0.05

    ppw10, ppw12 = results
    for monitor_name in ("mid", "far"):
        s21_10 = np.asarray(ppw10.s21_by_monitor[monitor_name], dtype=np.complex128)
        s21_12 = np.asarray(ppw12.s21_by_monitor[monitor_name], dtype=np.complex128)
        center = int(np.argmin(np.abs(ppw10.wavelengths_um - cfg.wavelength_um)))
        mag10 = 20.0 * np.log10(max(float(abs(s21_10[center])), 1e-12))
        mag12 = 20.0 * np.log10(max(float(abs(s21_12[center])), 1e-12))
        assert abs(mag12 - mag10) < 0.05

    slope_10 = ppw10.phase_slope_s_by_monitor["far"]
    slope_12 = ppw12.phase_slope_s_by_monitor["far"]
    assert abs(slope_12 - slope_10) < 0.5e-15
