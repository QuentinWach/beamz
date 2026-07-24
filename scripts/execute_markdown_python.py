#!/usr/bin/env python3
"""Execute Python code fences from a Markdown document in source order."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

PYTHON_FENCE = re.compile(
    r"^```python[ \t]*\n(?P<source>.*?)^```[ \t]*$",
    flags=re.MULTILINE | re.DOTALL,
)


def python_blocks(markdown: str) -> tuple[str, ...]:
    """Extract executable Python fences while preserving document order."""
    return tuple(match.group("source") for match in PYTHON_FENCE.finditer(markdown))


def execute_markdown(path: Path) -> dict[str, Any]:
    """Execute all Python fences in one shared, script-like namespace."""
    namespace: dict[str, Any] = {
        "__file__": str(path),
        "__name__": "__beamz_documentation__",
    }
    markdown = path.read_text(encoding="utf-8")
    blocks = python_blocks(markdown)
    if not blocks:
        raise ValueError(f"{path} contains no executable Python fences")
    for index, source in enumerate(blocks, start=1):
        code = compile(source, f"{path}:python-fence-{index}", "exec")
        exec(code, namespace)
    return namespace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    args = parser.parse_args()
    execute_markdown(args.document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
