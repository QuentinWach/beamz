"""Pulse-through-slab dispersion showcase.

Usage:
  uv run python examples/7_dispersion_showcase.py --case all
  uv run python examples/7_dispersion_showcase.py --case drude --save
  uv run python examples/7_dispersion_showcase.py --case all --fast
"""

from __future__ import annotations

import argparse
from pathlib import Path

from beamz import um
from beamz.design.library import gold, sio2, water
from beamz.visual.dispersion_validation import (
    plot_dispersion_validation,
    print_dispersion_metrics,
    run_pulse_through_slab,
)


def _build_cases(fast: bool) -> dict[str, dict]:
    steps_scale = 0.7 if fast else 1.0
    return {
        "sellmeier": {
            "name": "SiO2 Sellmeier Slab",
            "material": sio2(),
            "wavelength_center_m": 1.2 * um,
            "slab_thickness_m": 3.0 * um,
            "domain_size_m": (18.0 * um, 6.0 * um),
            "resolution_m": 0.5 * um,
            "num_steps": int(80 * steps_scale),
            "source_width_m": 0.5 * um,
        },
        "drude": {
            "name": "Gold Drude Slab",
            "material": gold(),
            "wavelength_center_m": 0.8 * um,
            "slab_thickness_m": 2.4 * um,
            "domain_size_m": (18.0 * um, 6.0 * um),
            "resolution_m": 0.6 * um,
            "num_steps": int(64 * steps_scale),
            "source_width_m": 0.4 * um,
        },
        "debye": {
            "name": "Water Debye Slab",
            "material": water(),
            "wavelength_center_m": 6.0e-4,
            "slab_thickness_m": 1.5e-3,
            "domain_size_m": (6.0e-3, 2.0e-3),
            "resolution_m": 6.0e-5,
            "num_steps": int(120 * steps_scale),
            "source_width_m": 1.5e-4,
        },
    }


def _save_animation(anim, path: Path) -> None:
    try:
        anim.save(path, writer="pillow", fps=16)
        print(f"Saved animation: {path}")
    except Exception as exc:
        print(f"Could not save animation to {path}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispersion validation showcase.")
    parser.add_argument(
        "--case",
        choices=["sellmeier", "drude", "debye", "all"],
        default="all",
        help="Which case to run.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save validation figures (and animations when available).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use reduced-step settings for a quicker run.",
    )
    args = parser.parse_args()

    cases = _build_cases(args.fast)
    selected = list(cases.keys()) if args.case == "all" else [args.case]

    out_dir = Path("artifacts/dispersion_showcase")
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for key in selected:
        cfg = cases[key]
        print(f"\nRunning: {cfg['name']}")
        result = run_pulse_through_slab(**cfg)
        results.append(result)

        title = cfg["name"]
        if result.used_fallback:
            title += " (Spectral Fallback)"
        fig_path = out_dir / f"{key}_validation.png" if args.save else None
        fig, anim = plot_dispersion_validation(
            result,
            title=title,
            show=True,
            save_path=fig_path,
            animate=True,
        )
        if fig_path is not None:
            print(f"Saved figure: {fig_path}")
        if args.save and anim is not None:
            _save_animation(anim, out_dir / f"{key}_field.gif")

    print("\nDispersion validation summary:")
    print_dispersion_metrics(results)


if __name__ == "__main__":
    main()

