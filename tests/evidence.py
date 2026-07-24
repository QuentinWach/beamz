"""Evidence classification shared by collection checks and test tooling."""

from __future__ import annotations

from pathlib import Path

PRIMARY_EVIDENCE_MARKERS = frozenset(
    {"contract", "invariant", "validation", "characterization"}
)


def evidence_markers_for_path(path: Path, *, tests_root: Path) -> tuple[str, ...]:
    """Return the primary evidence marker followed by orthogonal scope markers."""
    relative = path.resolve().relative_to(tests_root.resolve())
    category = relative.parts[0]
    name = relative.stem

    if category in {"unit", "contracts", "integration", "docs", "pdk"}:
        markers = ["contract"]
    elif category == "kernels":
        markers = ["invariant"]
    elif category == "validation":
        validation_kind = relative.parts[1]
        markers = ["invariant" if validation_kind == "invariants" else "validation"]
    elif category == "characterization":
        markers = ["characterization"]
    elif category == "differential":
        markers = ["validation", "differential"]
    elif category == "hardware":
        markers = ["contract", "hardware"]
    elif category == "performance":
        markers = ["characterization", "performance"]
    else:
        raise ValueError(
            f"{relative} is outside the evidence-oriented test directories"
        )

    if category == "pdk":
        markers.append("pdk")
    if category == "docs":
        markers.append("docs")
    if category == "characterization" and any(
        token in name for token in ("smoke", "unverified", "mixed")
    ):
        markers.append("smoke")
    return tuple(markers)
