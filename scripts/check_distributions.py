"""Reject Python distributions containing local or generated workspace state."""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

FORBIDDEN_PARTS = {
    ".beamz_cache",
    ".cache",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "htmlcov",
    "site",
}
FORBIDDEN_FILES = {".coverage", "coverage.xml", ".DS_Store"}
REQUIRED = {
    "sdist": {
        "Cargo.toml",
        "beamz/__init__.py",
        "beamz/py.typed",
        "pyproject.toml",
        "rust/fdtd-raster-core/src/lib.rs",
        "rust/fdtd-raster-py/src/lib.rs",
    },
    "wheel": {"beamz/__init__.py", "beamz/py.typed"},
}


def _archive_names(path: Path) -> list[str]:
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            return archive.getnames()
    with zipfile.ZipFile(path) as archive:
        return archive.namelist()


def _logical_names(names: list[str], kind: str) -> set[str]:
    unsafe = [
        name
        for name in names
        if name.startswith("/") or "\\" in name or ".." in PurePosixPath(name).parts
    ]
    if unsafe:
        raise ValueError(f"unsafe archive paths: {', '.join(unsafe)}")
    if kind == "wheel":
        return {name.rstrip("/") for name in names}
    roots = {PurePosixPath(name).parts[0] for name in names if name}
    if len(roots) != 1:
        raise ValueError(f"sdist must have one archive root, found {sorted(roots)}")
    return {"/".join(PurePosixPath(name).parts[1:]).rstrip("/") for name in names}


def _forbidden(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return (
        any(part in FORBIDDEN_PARTS or part.endswith(".egg-info") for part in parts)
        or any(
            part in FORBIDDEN_FILES or part.startswith(".coverage.") for part in parts
        )
        or name.endswith((".pyc", ".pyo"))
    )


def main(dist: Path = Path("dist")) -> int:
    archives = {
        "sdist": sorted(dist.glob("*.tar.gz")),
        "wheel": sorted(dist.glob("*.whl")),
    }
    errors: list[str] = []
    for kind, paths in archives.items():
        if not paths:
            errors.append(f"Missing {kind} in {dist}")
        for path in paths:
            raw_names = _archive_names(path)
            try:
                names = _logical_names(raw_names, kind)
            except ValueError as exc:
                errors.append(f"{path}: {exc}")
                continue
            forbidden = sorted(name for name in names if _forbidden(name))
            missing = sorted(REQUIRED[kind] - names)
            if kind == "wheel" and not any(
                name.endswith(".dist-info/METADATA") for name in names
            ):
                missing.append("*.dist-info/METADATA")
            if kind == "wheel" and not any(
                name.startswith("beamz/design/raster/_native.")
                and name.endswith((".so", ".pyd", ".dylib"))
                for name in names
            ):
                missing.append("beamz/design/raster/_native.<platform-extension>")
            if forbidden:
                errors.append(f"{path}: forbidden entries: {', '.join(forbidden)}")
            if missing:
                errors.append(f"{path}: missing entries: {', '.join(missing)}")
            print(f"Checked {path}: {len(names)} entries, {path.stat().st_size} bytes")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist")
    raise SystemExit(main(target))
