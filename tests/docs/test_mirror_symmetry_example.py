from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "2D_basics" / "mirror_symmetry.py"


def test_mirror_symmetry_example_compiles():
    compile(EXAMPLE.read_text(encoding="utf-8"), str(EXAMPLE), "exec")


def test_mirror_symmetry_example_executes_in_reduced_mode():
    environment = {
        **os.environ,
        "BEAMZ_DOCS_TEST": "1",
        "JAX_PLATFORMS": "cpu",
    }
    subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=ROOT,
        env=environment,
        check=True,
        timeout=120,
    )
