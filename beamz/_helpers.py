from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TextIO

import numpy as np

from beamz.const import LIGHT_SPEED

_STATUS_MARKER = "* "
_LOGGER = logging.getLogger(__name__)


def positive_float(value, *, name: str) -> float:
    """Return a positive finite float or raise a field-specific error."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


def positive_integer(value, *, name: str) -> int:
    """Return a positive integer without silently truncating other numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def env_bool(name: str, default: bool = False) -> bool:
    """Read a conventional truthy environment flag with an explicit default."""
    value = os.getenv(name)
    return (
        bool(default)
        if value is None
        else value.strip().lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        }
    )


def get_si_scale_and_label(value):
    """Choose a readable SI length scale for a value in metres.

    Parameters
    ----------
    value : float
        Representative physical length in metres.

    Returns
    -------
    tuple of (float, str)
        Multiplicative scale and unit label among millimetres, micrometres,
        nanometres, and picometres.
    """
    if value >= 1e-3:
        return 1e3, "mm"
    if value >= 1e-6:
        return 1e6, "um"
    if value >= 1e-9:
        return 1e9, "nm"
    return 1e12, "pm"


def calc_optimal_fdtd_params(
    wavelength,
    n_max,
    dims=2,
    safety_factor=0.999,
    points_per_wavelength=10,
    width=None,
    height=None,
    depth=None,
):
    """Estimate stable uniform spatial and temporal FDTD steps.

    Parameters
    ----------
    wavelength : float
        Minimum vacuum wavelength of interest in metres.
    n_max : float
        Maximum refractive index in the simulation.
    dims : int, default=2
        Number of simulated spatial dimensions.
    safety_factor : float, default=0.999
        Fraction of the dimensional Courant limit.
    points_per_wavelength : int, default=10
        Minimum cells per wavelength inside the highest-index material.
    width : float, optional
        Domain x extent in metres, used only for grid-size diagnostics.
    height : float, optional
        Domain y extent in metres, used only for grid-size diagnostics.
    depth : float, optional
        Domain z extent in metres, used only for 3D grid-size diagnostics.

    Returns
    -------
    tuple of (float, float)
        Uniform spatial resolution in metres and time step in seconds.
    """
    resolution = wavelength / (n_max * points_per_wavelength)
    dt = safety_factor * resolution / (LIGHT_SPEED * np.sqrt(float(dims)))
    if width and height:
        nx = int(width / resolution)
        ny = int(height / resolution)
        nz = int(depth / resolution) if (dims == 3 and depth) else 1
        total_cells = nx * ny * nz
        if total_cells > 5e6:
            display_status(
                f"Large simulation grid detected ({total_cells / 1e6:.1f}M cells).",
                "warning",
            )
    return resolution, dt


def dxdt(
    wavelength,
    n_max=1.0,
    dims=2,
    safety_factor=0.999,
    points_per_wavelength=10,
    **kwargs,
):
    """Return the recommended FDTD spatial and temporal steps.

    Parameters
    ----------
    wavelength : float
        Minimum vacuum wavelength of interest in metres.
    n_max : float, default=1.0
        Maximum refractive index in the simulation.
    dims : int, default=2
        Number of simulated spatial dimensions.
    safety_factor : float, default=0.999
        Fraction of the dimensional Courant limit.
    points_per_wavelength : int, default=10
        Minimum cells per wavelength in the highest-index material.
    **kwargs
        Optional domain dimensions forwarded for grid-size diagnostics.

    Returns
    -------
    tuple of (float, float)
        Spatial resolution in metres and time step in seconds.
    """
    return calc_optimal_fdtd_params(
        wavelength=wavelength,
        n_max=n_max,
        dims=dims,
        safety_factor=safety_factor,
        points_per_wavelength=points_per_wavelength,
        **kwargs,
    )


def display_status(status: str, status_type: str = "info") -> None:
    """Log a compact status message.

    Parameters
    ----------
    status : str
        Human-readable message.
    status_type : {"info", "success", "warning", "error"}, default="info"
        Logging level category.
    """
    level = {"success": "info", "warning": "warning", "error": "error"}.get(
        status_type, "info"
    )
    getattr(_LOGGER, level)(status)


@dataclass
class _PlainTask:
    description: str
    total: int | None
    completed: int = 0


class PlainProgress:
    def __init__(
        self, *, file: TextIO | None = None, inline: bool = True, enabled: bool = False
    ):
        self._file = file
        self._inline = inline
        self._enabled = bool(enabled)
        self._tasks: dict[int, _PlainTask] = {}
        self._next_task_id = 0

    @property
    def _output(self) -> TextIO:
        import sys

        return self._file if self._file is not None else sys.stdout

    def __enter__(self) -> PlainProgress:
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        status = "failed" if exc_type is not None else "done"
        for task in self._tasks.values():
            if task.total is None:
                self._write_line(f"{task.description} {status}")
            else:
                done = min(task.completed, task.total)
                self._write_line(f"{task.description} {status} ({done}/{task.total})")

    def add_task(self, description: str, total: int | None = None) -> int:
        task_id = self._next_task_id
        self._next_task_id += 1
        normalized_total = None if total is None else max(int(total), 0)
        self._tasks[task_id] = _PlainTask(description, normalized_total)
        if normalized_total is None:
            self._write_line(description)
        else:
            self._write_progress(description, completed=0, total=normalized_total)
        return task_id

    def update(
        self, task_id: int, *, advance: int = 0, completed: int | None = None
    ) -> None:
        task = self._tasks[task_id]
        task.completed = max(
            int(completed) if completed is not None else task.completed + int(advance),
            0,
        )
        if task.total is not None:
            self._write_progress(
                task.description,
                completed=min(task.completed, task.total),
                total=task.total,
            )

    def _can_inline(self) -> bool:
        isatty = getattr(self._output, "isatty", None)
        return bool(self._inline and callable(isatty) and isatty())

    def _write_line(self, message: str) -> None:
        if not self._enabled:
            return
        if self._can_inline():
            self._output.write(f"\r{_STATUS_MARKER}{message}\n")
            self._output.flush()
        else:
            self._output.write(f"{_STATUS_MARKER}{message}\n")

    def _write_progress(self, description: str, *, completed: int, total: int) -> None:
        if not self._enabled:
            return
        message = _format_progress_message(
            completed,
            total,
            label=description.rstrip("."),
            unit="items",
        )
        if self._can_inline():
            self._output.write(f"\r{_STATUS_MARKER}{message}")
            self._output.flush()
        elif completed == 0:
            self._output.write(f"{_STATUS_MARKER}{description}\n")


def _format_progress_message(
    completed: int,
    total: int,
    *,
    label: str = "Progress",
    unit: str = "steps",
) -> str:
    safe_total = max(int(total), 1)
    safe_completed = min(max(int(completed), 0), safe_total)
    pct = 100.0 * safe_completed / safe_total
    return f"{label}: {pct:.0f}% ({safe_completed}/{safe_total} {unit})"


def _print_inline_progress(
    completed: int,
    total: int,
    *,
    label: str = "Progress",
    unit: str = "steps",
    file: TextIO | None = None,
) -> None:
    import sys

    output = file if file is not None else sys.stdout
    message = _format_progress_message(completed, total, label=label, unit=unit)
    output.write(f"\r{_STATUS_MARKER}{message}")
    output.flush()


def _finish_inline_progress(*, file: TextIO | None = None) -> None:
    import sys

    output = file if file is not None else sys.stdout
    output.write("\n")
    output.flush()


def create_plain_progress(*, enabled: bool = False) -> PlainProgress:
    """Create the dependency-free command-line progress reporter.

    Returns
    -------
    PlainProgress
        Context manager supporting task creation and incremental updates. It is
        silent unless ``enabled=True``.
    """
    return PlainProgress(enabled=enabled)
