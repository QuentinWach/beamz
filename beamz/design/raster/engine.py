"""Compilation, rasterization, and persistent result caching."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _native  # type: ignore[attr-defined]
from .result import RasterResult
from .schema import _CACHE_SCHEMA_VERSION, _ENGINE_VERSION, Grid, Scene


@dataclass(frozen=True, slots=True)
class RasterOptions:
    quality: str = "balanced"
    smoothing: str = "volume"
    components: str = "all"

    def __post_init__(self) -> None:
        choices = {
            "quality": {"fast", "balanced", "reference"},
            "smoothing": {"volume", "farjadpour_diagonal", "farjadpour_full"},
            "components": {"all", "two_dimensional_tm", "two_dimensional_te"},
        }
        for name, allowed in choices.items():
            value = str(getattr(self, name)).strip().lower()
            if value not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}.")
            object.__setattr__(self, name, value)


class CompiledScene:
    """A validated scene that can be rasterized repeatedly."""

    def __init__(self, scene: Scene):
        if not isinstance(scene, Scene):
            raise TypeError("scene must be a Scene.")
        self._native = _native.compile_scene(scene.to_json())
        self.hash = str(self._native.scene_hash)

    def rasterize(
        self,
        grid: Grid,
        *,
        options: RasterOptions | None = None,
        cache_directory: str | Path | None = None,
    ) -> RasterResult:
        if not isinstance(grid, Grid):
            raise TypeError("grid must be a Grid.")
        options = RasterOptions() if options is None else options
        if not isinstance(options, RasterOptions):
            raise TypeError("options must be RasterOptions.")
        cache_path = self._cache_path(grid, options, cache_directory)
        if cache_path is not None and cache_path.exists():
            try:
                return RasterResult._from_cache(
                    cache_path,
                    scene_hash=self.hash,
                    grid_edges=grid.edges,
                    smoothing=options.smoothing,
                )
            except (EOFError, KeyError, OSError, ValueError, zipfile.BadZipFile):
                cache_path.unlink(missing_ok=True)

        native = self._native.rasterize(
            tuple(edges.tolist() for edges in grid.edges),
            options.quality,
            options.smoothing,
            options.components,
        )
        result = RasterResult(
            native,
            grid_edges=grid.edges,
            smoothing=options.smoothing,
        )
        if cache_path is not None:
            temporary = cache_path.with_name(
                f".{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.npz"
            )
            try:
                np.savez_compressed(temporary, **result._cache_payload())
                temporary.replace(cache_path)
            finally:
                temporary.unlink(missing_ok=True)
        return result

    def _cache_path(
        self,
        grid: Grid,
        options: RasterOptions,
        directory: str | Path | None,
    ) -> Path | None:
        if directory is None:
            return None
        digest = hashlib.blake2b(digest_size=20)
        digest.update(self.hash.encode())
        for edges in grid.edges:
            digest.update(edges.tobytes())
        digest.update(
            json.dumps(
                {
                    "schema": _CACHE_SCHEMA_VERSION,
                    "engine": _ENGINE_VERSION,
                    "quality": options.quality,
                    "smoothing": options.smoothing,
                    "components": options.components,
                },
                sort_keys=True,
            ).encode()
        )
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{digest.hexdigest()}.npz"


def compile_scene(scene: Scene) -> CompiledScene:
    return CompiledScene(scene)


def rasterize(
    scene: Scene,
    grid: Grid,
    *,
    options: RasterOptions | None = None,
    cache_directory: str | Path | None = None,
) -> RasterResult:
    return compile_scene(scene).rasterize(
        grid, options=options, cache_directory=cache_directory
    )
