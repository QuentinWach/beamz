"""Export pinned Tidy3D automatic-grid boundaries for differential tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tidy3d as td

from tests.differential.tidy3d_grid_adapter import (
    TIDY3D_REFERENCE_VERSION,
    parity_cases,
    tidy3d_edges,
)

DEFAULT_OUTPUT = (
    Path(__file__).parents[1] / "tests" / "differential" / "tidy3d_grid_references.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if td.__version__ != TIDY3D_REFERENCE_VERSION:
        raise RuntimeError(
            "Tidy3D grid references must be exported with "
            f"tidy3d=={TIDY3D_REFERENCE_VERSION}; found {td.__version__}."
        )
    payload = {
        "schema": "beamz.tidy3d-grid-reference/v1",
        "tidy3d_version": td.__version__,
        "length_unit": "um",
        "cases": {
            case.name: {
                axis: edges.tolist() for axis, edges in tidy3d_edges(case, td).items()
            }
            for case in parity_cases()
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(payload['cases'])} Tidy3D grids to {args.output}")


if __name__ == "__main__":
    main()
