"""Raster cache and signature helpers for Design."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

RASTER_CACHE_VERSION = "v5"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _to_jsonable(value):
    """Convert arbitrary values into deterministic JSON-safe primitives."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return float(f"{value:.16g}")
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.size <= 1024:
            return [_to_jsonable(v) for v in arr.reshape(-1).tolist()]
        digest = hashlib.sha256(arr.tobytes()).hexdigest()
        return {
            "__ndarray__": {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "sha256": digest,
            }
        }
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in sorted(value.items())}
    return repr(value)


def _material_signature(material):
    if material is None:
        return None
    keys = ("permittivity", "permeability", "conductivity")
    return {k: _to_jsonable(getattr(material, k, None)) for k in keys}


def _structure_signature(structure):
    sig = {
        "type": type(structure).__name__,
        "bbox": (
            _to_jsonable(structure.get_bounding_box())
            if hasattr(structure, "get_bounding_box")
            else None
        ),
        "material": _material_signature(getattr(structure, "material", None)),
    }
    keys = (
        "position",
        "width",
        "height",
        "depth",
        "z",
        "radius",
        "inner_radius",
        "outer_radius",
        "is_pml",
        "optimize",
    )
    for key in keys:
        if hasattr(structure, key):
            sig[key] = _to_jsonable(getattr(structure, key))
    if hasattr(structure, "vertices"):
        sig["vertices"] = _to_jsonable(getattr(structure, "vertices"))
    if hasattr(structure, "interiors"):
        sig["interiors"] = _to_jsonable(getattr(structure, "interiors"))
    return sig


def _boundary_signature(boundary):
    if boundary is None:
        return None
    raw = {}
    for key, val in getattr(boundary, "__dict__", {}).items():
        if key.startswith("_"):
            continue
        raw[key] = _to_jsonable(val)
    return {"type": type(boundary).__name__, "attrs": raw}


def _grid_kind_for_type(grid_type):
    if not isinstance(grid_type, type):
        return None
    for cls in getattr(grid_type, "__mro__", ()):
        name = getattr(cls, "__name__", "").lower()
        if name == "regulargrid3d":
            return "3d"
        if name == "regulargrid":
            return "2d"
    return None


def _grid_kind_for_request(design_obj, grid_type, kwargs):
    explicit_kind = _grid_kind_for_type(grid_type)
    if explicit_kind is not None:
        return explicit_kind
    if isinstance(grid_type, str):
        gt = grid_type.lower()
        if gt in {"regular3d", "3d"}:
            return "3d"
        if gt in {"regular", "regulargrid", "2d"}:
            return "2d"
        if gt in {"auto", "auto-select", "autoselect"}:
            force_3d = bool(kwargs.get("force_3d", False))
            auto_select = bool(kwargs.get("auto_select", True))
            if force_3d or (auto_select and design_obj.is_3d and design_obj.depth > 0):
                return "3d"
            return "2d"
    return "3d" if (design_obj.is_3d and design_obj.depth > 0) else "2d"


def _normalize_aa_config(kwargs):
    mode = str(kwargs.get("aa_mode", "legacy_grid") or "legacy_grid").strip().lower()
    samples = kwargs.get("aa_samples", 64)
    seed = kwargs.get("aa_seed", 0)
    return {
        "mode": mode,
        "samples": int(64 if samples is None else samples),
        "seed": int(0 if seed is None else seed),
        "scramble": "cp_cell_v1" if mode == "stratified_jitter" else "none",
    }


def _design_cache_key(design_obj, resolution, grid_kind, resolution_z, aa_config):
    raster_env = {
        "fast_3d": os.getenv("BEAMZ_RASTER_FAST_3D"),
        "fast_min_voxels": os.getenv("BEAMZ_RASTER_FAST_MIN_VOXELS", "1000000"),
    }
    payload = {
        "version": RASTER_CACHE_VERSION,
        "grid_kind": grid_kind,
        "resolution_xy": _to_jsonable(float(resolution)),
        "resolution_z": _to_jsonable(float(resolution_z)),
        "antialiasing": _to_jsonable(aa_config),
        "raster_env": raster_env,
        "domain": {
            "width": _to_jsonable(float(design_obj.width)),
            "height": _to_jsonable(float(design_obj.height)),
            "depth": _to_jsonable(float(design_obj.depth)),
            "is_3d": bool(design_obj.is_3d),
        },
        "structures": [_structure_signature(s) for s in design_obj.structures],
        "boundaries": [
            _boundary_signature(b) for b in getattr(design_obj, "boundaries", [])
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _raster_cache_dir():
    default_dir = Path(".beamz_cache") / "raster"
    return Path(os.getenv("BEAMZ_RASTER_CACHE_DIR", str(default_dir)))


def _raster_cache_path(cache_key):
    return _raster_cache_dir() / f"{cache_key}.npz"


def _save_grid_to_cache(grid, cache_path: Path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        permittivity=np.asarray(grid.permittivity),
        permeability=np.asarray(grid.permeability),
        conductivity=np.asarray(grid.conductivity),
    )


def _build_grid_from_cached_arrays(
    *,
    design_obj,
    resolution,
    resolution_z,
    grid_kind,
    arrays,
):
    from beamz.design.meshing import RegularGrid, RegularGrid3D

    if grid_kind == "3d":
        grid = object.__new__(RegularGrid3D)
        grid.design = design_obj
        grid.resolution = resolution
        grid.resolution_xy = resolution
        grid.resolution_z = resolution_z
        grid.is_3d = True
        grid.dx = resolution
        grid.dy = resolution
        grid.dz = resolution_z
        grid.width = design_obj.width
        grid.height = design_obj.height
        grid.depth = design_obj.depth
    else:
        grid = object.__new__(RegularGrid)
        grid.design = design_obj
        grid.resolution = resolution
        grid.is_3d = False
        grid.dx = resolution
        grid.dy = resolution
        grid.width = design_obj.width
        grid.height = design_obj.height

    for name in ("permittivity", "permeability", "conductivity"):
        setattr(grid, name, np.asarray(arrays[name]))
    grid.shape = grid.permittivity.shape
    return grid
