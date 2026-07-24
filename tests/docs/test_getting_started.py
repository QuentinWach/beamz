from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_getting_started_executes_in_reduced_test_mode():
    environment = {
        **os.environ,
        "BEAMZ_DOCS_TEST": "1",
        "JAX_PLATFORMS": "cpu",
        "MPLBACKEND": "Agg",
    }

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "execute_markdown_python.py"),
            str(ROOT / "docs" / "getting-started.md"),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        timeout=120,
    )
