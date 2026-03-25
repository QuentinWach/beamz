"""Tiny fixed-configuration BeamZ crossing example."""

from __future__ import annotations

from pathlib import Path

from beamz_crossing import run_crossing

OUT_DIR = Path("benchmarks/results/tiny_beamz_crossing")


def main() -> None:
    run_crossing(
        component_name="ebeam_crossing4",
        wl0=1550.0e-9,
        wl_min=1530.0e-9,
        wl_max=1570.0e-9,
        num_freqs=51,
        n_core=3.47,
        n_clad=1.44,
        polarization="te",
        points_per_wavelength=10,
        layer=None,
        use_pdk_stack=True,
        z_crop_auto=True,
        margin_z_above_um=0.5,
        margin_z_below_um=0.5,
        extension_um=1.5,
        port_overlap_um=0.10,
        core_t_um=0.22,
        clad_below_um=0.5,
        clad_above_um=0.5,
        top_clad_shift_um=0.0,
        min_bottom_clad_um=0.8,
        monitor_candidates=1,
        mode_search_max=0,
        pml_um=1.0,
        port_margin_um=0.5,
        source_port_offset_um=0.1,
        distance_source_to_monitors_um=0.2,
        run_after_sources_uoc=90.0,
        animation_frames=0,
        write_plots=True,
        write_mode_plots=False,
        write_animation=False,
        show_progress=True,
        out_dir=OUT_DIR,
        wave_dominance_min_db=6.0,
        strict_normalization_qa=False,
        source_direction_mode="inward",
    )


if __name__ == "__main__":
    main()
