import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd):
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Command failed ({cmd}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def test_static_thermal_example_module_runs():
    _run([sys.executable, "-m", "examples.thermal_static"])


def test_static_thermal_wrapper_runs():
    _run([sys.executable, "examples/7_thermal_static.py"])
