"""Immutable solver-facing raster result."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .schema import _CACHE_SCHEMA_VERSION, _ENGINE_VERSION

_TENSORS = ("epsilon", "mu", "conductivity")
_METADATA = {
    "cache_schema",
    "engine_version",
    "scene_hash",
    "diagnostics_json",
    "grid_x_edges",
    "grid_y_edges",
    "grid_z_edges",
    *(f"tensor_{name}" for name in _TENSORS),
}


def _readonly(value: Any, *, copy: bool = False) -> np.ndarray:
    array = np.array(value, copy=copy) if copy else np.asarray(value)
    array.setflags(write=False)
    return array


def _yee_tensor_shapes(
    cell_shape: tuple[int, int, int],
) -> dict[str, tuple[int, ...]]:
    nz, ny, nx = cell_shape
    ex, ey, ez = (
        (nz + 1, ny + 1, nx),
        (nz + 1, ny, nx + 1),
        (
            nz,
            ny + 1,
            nx + 1,
        ),
    )
    return {
        "epsilon_ex": ex,
        "epsilon_ey": ey,
        "epsilon_ez": ez,
        "epsilon_node": (nz + 1, ny + 1, nx + 1),
        "conductivity_ex": ex,
        "conductivity_ey": ey,
        "conductivity_ez": ez,
        "mu_hx": (nz, ny, nx + 1),
        "mu_hy": (nz, ny + 1, nx),
        "mu_hz": (nz + 1, ny, nx),
    }


@dataclass(frozen=True, slots=True, init=False)
class RasterResult:
    grid_edges: tuple[np.ndarray, np.ndarray, np.ndarray]
    smoothing: str
    yee_tensors: Mapping[str, np.ndarray]
    tensors: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, Any]
    scene_hash: str
    cache_hit: bool

    def __init__(
        self,
        native: Any,
        *,
        grid_edges: tuple[np.ndarray, np.ndarray, np.ndarray],
        smoothing: str,
    ):
        payload = dict(native.take_arrays())
        diagnostics = json.loads(str(payload.pop("diagnostics_json")))
        scene_hash = str(payload.pop("scene_hash"))
        tensors = {name: payload.pop(f"tensor_{name}") for name in _TENSORS}
        self._assign(
            grid_edges,
            smoothing,
            payload,
            tensors,
            diagnostics,
            scene_hash,
            cache_hit=False,
        )

    def _assign(
        self,
        grid_edges: tuple[np.ndarray, np.ndarray, np.ndarray],
        smoothing: str,
        yee_tensors: Mapping[str, Any],
        tensors: Mapping[str, Any],
        diagnostics: Mapping[str, Any],
        scene_hash: str,
        *,
        cache_hit: bool,
    ) -> None:
        assign = object.__setattr__
        edges = tuple(_readonly(values, copy=True) for values in grid_edges)
        cell_shape = (len(edges[2]) - 1, len(edges[1]) - 1, len(edges[0]) - 1)
        shapes = _yee_tensor_shapes(cell_shape)
        unknown = set(yee_tensors) - set(shapes)
        if unknown:
            raise ValueError(
                f"Native raster output has unknown fields: {sorted(unknown)}"
            )
        checked_yee_tensors = {}
        for name, value in yee_tensors.items():
            array = _readonly(value)
            if (
                array.ndim != 4
                or array.shape[0] not in (1, 3, 6)
                or array.shape[1:] != shapes[name]
            ):
                raise ValueError(
                    f"{name} has shape {array.shape}; expected compact "
                    f"(1|3|6, {', '.join(str(value) for value in shapes[name])})."
                )
            checked_yee_tensors[name] = array
        assign(self, "grid_edges", edges)
        assign(self, "smoothing", str(smoothing))
        assign(self, "yee_tensors", MappingProxyType(checked_yee_tensors))
        checked_tensors = {}
        for name in _TENSORS:
            array = _readonly(tensors[name])
            if (
                array.ndim != 4
                or array.shape[0] not in (1, 3, 6)
                or array.shape[1:] != cell_shape
            ):
                raise ValueError(
                    f"{name} tensor has shape {array.shape}; expected compact "
                    f"(1|3|6, {cell_shape[0]}, {cell_shape[1]}, {cell_shape[2]})."
                )
            checked_tensors[name] = array
        assign(self, "tensors", MappingProxyType(checked_tensors))
        assign(self, "diagnostics", MappingProxyType(dict(diagnostics)))
        assign(self, "scene_hash", str(scene_hash))
        assign(self, "cache_hit", bool(cache_hit))

    @property
    def is_uniform(self) -> bool:
        return all(
            np.allclose(np.diff(edges), np.diff(edges)[0], rtol=1e-12, atol=0.0)
            for edges in self.grid_edges
        )

    def _cache_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "cache_schema": np.asarray(_CACHE_SCHEMA_VERSION),
            "engine_version": np.asarray(_ENGINE_VERSION),
            "scene_hash": np.asarray(self.scene_hash),
            "diagnostics_json": np.asarray(json.dumps(dict(self.diagnostics))),
        }
        payload.update(
            {f"tensor_{name}": values for name, values in self.tensors.items()}
        )
        payload.update(
            {
                f"grid_{axis}_edges": edges
                for axis, edges in zip(("x", "y", "z"), self.grid_edges, strict=True)
            }
        )
        payload.update(
            {f"yee__{name}": values for name, values in self.yee_tensors.items()}
        )
        return payload

    @classmethod
    def _from_cache(
        cls,
        path: Path,
        *,
        scene_hash: str,
        grid_edges: tuple[np.ndarray, np.ndarray, np.ndarray],
        smoothing: str,
    ) -> RasterResult:
        with np.load(path, allow_pickle=False) as values:
            if (
                int(values["cache_schema"]) != _CACHE_SCHEMA_VERSION
                or str(values["engine_version"]) != _ENGINE_VERSION
                or str(values["scene_hash"]) != scene_hash
            ):
                raise ValueError("Raster cache identity does not match.")
            cached_edges = tuple(
                values[f"grid_{axis}_edges"].copy() for axis in ("x", "y", "z")
            )
            if any(
                not np.array_equal(actual, expected)
                for actual, expected in zip(cached_edges, grid_edges, strict=True)
            ):
                raise ValueError("Raster cache grid does not match.")
            unknown = {
                name
                for name in values.files
                if name not in _METADATA and not name.startswith("yee__")
            }
            if unknown:
                raise ValueError(f"Raster cache has unknown fields: {sorted(unknown)}")
            result = cls.__new__(cls)
            result._assign(
                cached_edges,  # type: ignore[arg-type]
                smoothing,
                {
                    name.removeprefix("yee__"): values[name].copy()
                    for name in values.files
                    if name.startswith("yee__")
                },
                {name: values[f"tensor_{name}"].copy() for name in _TENSORS},
                json.loads(str(values["diagnostics_json"])),
                scene_hash,
                cache_hit=True,
            )
            return result
