#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"


@dataclass(frozen=True)
class SummaryRow:
    backend: str
    resolution_nm: float
    repeats: int
    time_mean: float
    time_ci95: float
    gcups_mean: float
    gcups_ci95: float


def _latest_results_dir() -> Path:
    candidates = sorted(
        RESULTS_ROOT.glob("*/performance_summary.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No performance_summary.csv found under benchmarks/results.")
    return candidates[0].parent


def _load_summary_rows(
    results_dirs: list[Path], *, time_basis: str, throughput_basis: str
) -> list[SummaryRow]:
    time_mean_key = f"{time_basis}_s_mean"
    time_ci95_key = f"{time_basis}_s_ci95"
    gcups_mean_key = f"gcups_{throughput_basis}_mean"
    gcups_ci95_key = f"gcups_{throughput_basis}_ci95"

    rows: list[SummaryRow] = []
    seen_keys: set[tuple[str, float]] = set()
    for results_dir in results_dirs:
        summary_csv = results_dir / "performance_summary.csv"
        if not summary_csv.exists():
            raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")
        with summary_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                row = SummaryRow(
                    backend=str(raw["backend"]),
                    resolution_nm=float(raw["resolution_nm"]),
                    repeats=int(raw["repeats"]),
                    time_mean=float(raw[time_mean_key]),
                    time_ci95=float(raw[time_ci95_key]),
                    gcups_mean=float(raw[gcups_mean_key]),
                    gcups_ci95=float(raw[gcups_ci95_key]),
                )
                key = (row.backend, row.resolution_nm)
                if key in seen_keys:
                    raise ValueError(
                        "Duplicate backend/resolution rows found across result sets: "
                        f"{row.backend} at {row.resolution_nm:g} nm"
                    )
                seen_keys.add(key)
                rows.append(row)
    if not rows:
        raise ValueError("No rows found in the provided performance_summary.csv files.")
    return rows


def _plot_comparison(
    rows: list[SummaryRow],
    *,
    results_dir: Path,
    time_basis: str,
    throughput_basis: str,
    machine_label: str,
) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.size": 10,
            "savefig.bbox": "tight",
        }
    )

    backends = ["beamz", "meep"]
    backend_labels = {"beamz": "BeamZ", "meep": "Meep"}
    colors = {"beamz": "#2F5D8A", "meep": "#D97B29"}
    resolutions = sorted({row.resolution_nm for row in rows}, reverse=True)
    repeats = max(row.repeats for row in rows)

    row_map = {(row.backend, row.resolution_nm): row for row in rows}
    x = np.arange(len(resolutions), dtype=np.float64)
    width = 0.36
    annotation_padding = 12.0

    fig, ax = plt.subplots(1, 1, figsize=(8.4, 4.8))
    max_bar_top = 0.0
    for idx, backend in enumerate(backends):
        offset = (idx - 0.5) * width
        values = []
        errors = []
        for resolution_nm in resolutions:
            row = row_map.get((backend, resolution_nm))
            if row is None:
                values.append(np.nan)
                errors.append(0.0)
            else:
                values.append(row.time_mean)
                errors.append(row.time_ci95)
                max_bar_top = max(max_bar_top, row.time_mean + row.time_ci95)
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=colors[backend],
            yerr=errors,
            capsize=4,
            linewidth=0,
            error_kw={"elinewidth": 1.1, "capthick": 1.1},
            zorder=3,
        )
        for bar, resolution_nm, err in zip(bars, resolutions, errors, strict=False):
            row = row_map.get((backend, resolution_nm))
            if row is None:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + err + annotation_padding,
                f"{row.gcups_mean:.3f} ± {row.gcups_ci95:.3f}\nGCUPS",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#243141",
                linespacing=0.95,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{res:g}" for res in resolutions])
    ax.set_xlabel("Resolution (nm)")
    ax.set_ylabel(f"{time_basis.capitalize()} time (s)")
    ax.grid(axis="y", color="#D9DDE3", linewidth=0.8, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0.0, max_bar_top + annotation_padding + max_bar_top * 0.12)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[backend], linewidth=0)
        for backend in backends
    ]
    labels = [backend_labels[backend] for backend in backends]
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
    )

    fig.text(
        0.5,
        0.06,
        (
            f"{machine_label}  |  Synthetic 3D coupler  |  "
            f"mean ± 95% CI  |  n={repeats} interleaved runs"
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#4A5563",
    )
    fig.subplots_adjust(top=0.84, bottom=0.20)

    stem = f"comparison_bar_chart_{time_basis}_{throughput_basis}"
    png_path = results_dir / f"{stem}.png"
    pdf_path = results_dir / f"{stem}.pdf"
    fig.savefig(png_path, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot BeamZ vs Meep bar charts from saved paper-style coupler benchmark results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "One or more benchmark result directories containing performance_summary.csv. "
            "Defaults to the most recent one."
        ),
    )
    parser.add_argument(
        "--time-basis",
        choices=("run", "total"),
        default="run",
        help="Which saved time metric to plot.",
    )
    parser.add_argument(
        "--throughput-basis",
        choices=("run", "total"),
        default="run",
        help="Which saved throughput metric to plot.",
    )
    parser.add_argument(
        "--machine-label",
        default="MacBook Air M4, 24GB, 2025",
        help="Machine description shown in the subtitle.",
    )
    args = parser.parse_args()

    if args.results_dir:
        results_dirs = [path.resolve() for path in args.results_dir]
        results_dir = results_dirs[0]
    else:
        results_dir = _latest_results_dir()
        results_dirs = [results_dir]
    rows = _load_summary_rows(
        results_dirs,
        time_basis=args.time_basis,
        throughput_basis=args.throughput_basis,
    )
    png_path, pdf_path = _plot_comparison(
        rows,
        results_dir=results_dir,
        time_basis=args.time_basis,
        throughput_basis=args.throughput_basis,
        machine_label=args.machine_label,
    )
    print(png_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
