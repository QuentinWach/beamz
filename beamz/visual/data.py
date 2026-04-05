"""Small plotting-data containers shared across design, monitor, and simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from beamz.visual.helpers import get_si_scale_and_label


@dataclass(frozen=True)
class Slice2D:
    values: np.ndarray
    extent: tuple[float, float, float, float]
    value_label: str
    plane: str = "xy"
    position: float | None = None
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    style: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "values", np.asarray(self.values))
        object.__setattr__(self, "extent", tuple(float(v) for v in self.extent))
        object.__setattr__(self, "style", dict(self.style))

    @property
    def max_extent(self) -> float:
        return max(abs(v) for v in self.extent) if self.extent else 0.0

    def scaled_extent(self) -> tuple[tuple[float, float, float, float], float, str]:
        scale, unit = get_si_scale_and_label(max(self.max_extent, 1e-12))
        return tuple(v * scale for v in self.extent), scale, unit

    def plot(
        self,
        *,
        ax=None,
        cmap="viridis",
        colorbar=True,
        colorbar_label=None,
        abs_value=False,
        origin="lower",
        aspect="equal",
        **imshow_kwargs,
    ):
        import matplotlib.pyplot as plt

        created_fig = None
        if ax is None:
            created_fig, ax = plt.subplots()

        extent, _, unit = self.scaled_extent()
        values = np.abs(self.values) if abs_value else self.values
        im = ax.imshow(
            values,
            origin=origin,
            extent=extent,
            aspect=aspect,
            cmap=cmap,
            **imshow_kwargs,
        )
        ax.set_xlabel(self.x_label or f"{self.plane[0]} ({unit})")
        ax.set_ylabel(self.y_label or f"{self.plane[1]} ({unit})")
        if self.title:
            ax.set_title(self.title)
        if colorbar:
            label = colorbar_label or self.value_label
            owner_fig = created_fig or ax.figure
            owner_fig.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
        return ax


@dataclass(frozen=True)
class Trace1D:
    values: np.ndarray
    coords: np.ndarray
    coord_label: str
    value_label: str
    title: str | None = None
    style: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "values", np.asarray(self.values))
        object.__setattr__(self, "coords", np.asarray(self.coords, dtype=float))
        object.__setattr__(self, "style", dict(self.style))

    @property
    def max_coord(self) -> float:
        return float(np.max(np.abs(self.coords))) if self.coords.size else 0.0

    def scaled_coords(self) -> tuple[np.ndarray, float, str]:
        scale, unit = _coord_scale_and_label(
            self.coord_label, max(self.max_coord, 1e-18)
        )
        return self.coords * scale, scale, unit

    def plot(self, *, ax=None, abs_value=False, **plot_kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        coords, _, unit = self.scaled_coords()
        values = np.abs(self.values) if abs_value else self.values
        ax.plot(coords, values, **plot_kwargs)
        ax.set_xlabel(f"{self.coord_label} ({unit})")
        ax.set_ylabel(self.value_label)
        if self.title:
            ax.set_title(self.title)
        return ax


def _coord_scale_and_label(coord_label: str, value: float) -> tuple[float, str]:
    label = str(coord_label).strip().lower()
    if label in {"time", "t"}:
        if value < 1e-12:
            return 1e15, "fs"
        if value < 1e-9:
            return 1e12, "ps"
        if value < 1e-6:
            return 1e9, "ns"
        if value < 1e-3:
            return 1e6, "µs"
        if value < 1.0:
            return 1e3, "ms"
        return 1.0, "s"
    return get_si_scale_and_label(value)
