#!/usr/bin/env python3
"""Execute a Jupyter notebook's Python cells in source order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def code_cells(path: Path) -> tuple[str, ...]:
    """Load the non-empty Python code cells from a notebook."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"{path} does not contain a notebook cell list")

    sources = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", ())
        text = source if isinstance(source, str) else "".join(source)
        if text.strip():
            sources.append(text)
    if not sources:
        raise ValueError(f"{path} contains no executable Python cells")
    return tuple(sources)


def execute_notebook(path: Path) -> dict[str, Any]:
    """Execute all Python cells in one shared, kernel-like namespace."""
    namespace: dict[str, Any] = {
        "__file__": str(path),
        "__name__": "__beamz_notebook__",
    }
    for index, source in enumerate(code_cells(path), start=1):
        code = compile(source, f"{path}:code-cell-{index}", "exec")
        exec(code, namespace)
    return namespace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    # Prefer the checkout containing this executor over an editable installation
    # from another worktree.  This mirrors how the notebooks are published and
    # makes their execution test validate the code under review.
    sys.path.insert(0, str(ROOT))
    execute_notebook(args.notebook)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
