"""Standard BeamZ crossing example."""

from __future__ import annotations

import argparse
from pathlib import Path

from _beamz_crossing_impl import run_crossing


DEFAULTS = {
    "n_core": 3.47,
    "n_clad": 1.44,
    "polarization": "te",
    "layer": None,
    "use_pdk_stack": True,
    "z_crop_auto": True,
    "margin_z_above_um": 0.5,
    "margin_z_below_um": 0.5,
    "extension_um": 1.5,
    "port_overlap_um": 0.10,
    "core_t_um": 0.22,
    "clad_below_um": 0.5,
    "clad_above_um": 0.5,
    "top_clad_shift_um": 0.0,
    "min_bottom_clad_um": 0.8,
    "monitor_candidates": 1,
    "mode_search_max": 0,
    "pml_um": 1.0,
    "port_margin_um": 0.5,
    "source_port_offset_um": 0.1,
    "distance_source_to_monitors_um": 0.2,
    "animation_frames": 0,
    "write_plots": True,
    "write_mode_plots": False,
    "write_animation": False,
    "wave_dominance_min_db": 6.0,
    "strict_normalization_qa": False,
    "source_direction_mode": "inward",
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standard BeamZ crossing example.")
    parser.add_argument("--component", default="ebeam_crossing4", help="Crossing component name from the active PDK.")
    parser.add_argument("--wl0-nm", type=float, default=1550.0, help="Center wavelength in nm.")
    parser.add_argument("--wl-min-nm", type=float, default=1530.0, help="Sweep minimum wavelength in nm.")
    parser.add_argument("--wl-max-nm", type=float, default=1570.0, help="Sweep maximum wavelength in nm.")
    parser.add_argument("--num-freqs", type=int, default=51, help="Number of DFT frequency points.")
    parser.add_argument("--points-per-wavelength", type=int, default=10, help="Grid resolution in points per wavelength.")
    parser.add_argument(
        "--run-after-sources-uoc",
        type=float,
        default=90.0,
        help="Requested minimum settling window after the source tail in um/c units.",
    )
    parser.add_argument("--quiet-run", action="store_true", help="Disable compiled-run progress output.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmarks/results/beamz_crossing"),
        help="Output directory for the S-parameter and overview plots.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_freqs < 2:
        raise ValueError("--num-freqs must be >= 2.")
    if args.wl_min_nm >= args.wl_max_nm:
        raise ValueError("--wl-min-nm must be smaller than --wl-max-nm.")
    if not (args.wl_min_nm <= args.wl0_nm <= args.wl_max_nm):
        raise ValueError("--wl0-nm must be within [wl-min-nm, wl-max-nm].")


def main() -> None:
    args = build_argparser().parse_args()
    validate_args(args)
    run_crossing(
        component_name=args.component,
        wl0=args.wl0_nm * 1e-9,
        wl_min=args.wl_min_nm * 1e-9,
        wl_max=args.wl_max_nm * 1e-9,
        num_freqs=args.num_freqs,
        points_per_wavelength=args.points_per_wavelength,
        run_after_sources_uoc=args.run_after_sources_uoc,
        show_progress=not args.quiet_run,
        out_dir=args.out_dir,
        **DEFAULTS,
    )


if __name__ == "__main__":
    main()
