from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from beamz.const import LIGHT_SPEED
from beamz.devices.sources.mode import ModeSource
from beamz.devices.sources.solve import solve_modes
from beamz.simulation.specs import GaussianPulse, ModeSpec


def _plane_axis_and_spans(plane):
    center = getattr(plane, "center", None)
    size = getattr(plane, "size", None)
    if center is None or size is None:
        center = getattr(plane, "position", (0.0, 0.0, 0.0))
        size = getattr(plane, "size", (0.0, 1.0, 1.0))
    if len(center) == 2:
        center = (center[0], center[1], 0.0)
    if len(size) == 2:
        size = (size[0], size[1], 0.0)
    normal_index = int(np.argmin(np.abs(np.asarray(size, dtype=float))))
    axis = ("x", "y", "z")[normal_index]
    transverse = [float(size[idx]) for idx in range(3) if idx != normal_index]
    return axis, tuple(float(v) for v in center), tuple(transverse)


def _plane_center_and_size(plane):
    center = getattr(plane, "center", None)
    size = getattr(plane, "size", None)
    if center is None or size is None:
        center = getattr(plane, "position", (0.0, 0.0, 0.0))
        size = getattr(plane, "size", (0.0, 1.0, 1.0))
    if len(center) == 2:
        center = (center[0], center[1], 0.0)
    if len(size) == 2:
        size = (size[0], size[1], 0.0)
    return tuple(float(v) for v in center), tuple(float(v) for v in size)


def _profile_crop_slices(eps_profile, *, profile_axes, center, size, resolution):
    """Return slices that crop an extracted cross-section to a finite plane."""
    grid_axis_to_coord_index = {0: 2, 1: 1, 2: 0}
    slices = []
    for dim, grid_axis in enumerate(profile_axes):
        coord_index = grid_axis_to_coord_index[int(grid_axis)]
        span = float(size[coord_index])
        if span <= 0.0 or not np.isfinite(span):
            slices.append(slice(None))
            continue
        midpoint = float(center[coord_index])
        start = int(np.floor((midpoint - 0.5 * span) / resolution))
        stop = int(np.ceil((midpoint + 0.5 * span) / resolution))
        start = int(np.clip(start, 0, eps_profile.shape[dim] - 1))
        stop = int(np.clip(stop, start + 1, eps_profile.shape[dim]))
        slices.append(slice(start, stop))
    return tuple(slices)


def _crop_profile_to_plane(eps_profile, *, profile_axes, center, size, resolution):
    """Crop an extracted cross-section to the finite transverse size of a plane."""
    return eps_profile[
        _profile_crop_slices(
            eps_profile,
            profile_axes=profile_axes,
            center=center,
            size=size,
            resolution=resolution,
        )
    ]


def _chebyshev_frequency_nodes(freq0: float, fwidth: float, count: int) -> np.ndarray:
    """Return sorted Chebyshev nodes over the Tidy3D broadband source interval."""
    n = int(count)
    if n <= 1:
        return np.asarray([float(freq0)], dtype=float)
    center = float(freq0)
    half_width = 1.5 * float(fwidth)
    k = np.arange(n, dtype=float)
    nodes = center + half_width * np.cos((2.0 * k + 1.0) * np.pi / (2.0 * n))
    return np.sort(nodes.astype(float))


@dataclass(frozen=True)
class ModeData:
    frequencies: np.ndarray
    neffs: np.ndarray
    e_fields: np.ndarray
    h_fields: np.ndarray
    eps_profiles: np.ndarray
    resolution: float

    def to_dataframe(self):
        rows = []
        index = []
        dx_um = float(self.resolution) * 1e6
        for f_idx, freq in enumerate(self.frequencies):
            neff_row = np.atleast_1d(self.neffs[f_idx])
            eps_profile = np.asarray(self.eps_profiles[f_idx])
            finite = np.real(eps_profile[np.isfinite(eps_profile)])
            unique = (
                np.unique(np.round(finite, decimals=8))
                if finite.size
                else np.asarray(())
            )
            if unique.size >= 2:
                core_threshold = 0.5 * (float(unique[-2]) + float(unique[-1]))
            elif unique.size == 1:
                core_threshold = float(unique[-1])
            else:
                core_threshold = np.inf
            core = np.real(eps_profile) >= core_threshold
            wavelength = LIGHT_SPEED / float(freq)
            wavelength_um = wavelength * 1e6
            wavelength_cm = wavelength * 100.0
            for mode_index, neff in enumerate(neff_row):
                e_field = np.asarray(self.e_fields[f_idx, mode_index])
                ey = np.squeeze(e_field[1])
                ez = np.squeeze(e_field[2])
                intensity_te = np.abs(ey) ** 2
                intensity_tm = np.abs(ez) ** 2
                intensity = intensity_te + intensity_tm
                total_te = float(np.sum(intensity_te))
                total_tm = float(np.sum(intensity_tm))
                total = max(total_te + total_tm, 1e-30)
                core_te = (
                    float(np.sum(intensity_te[core])) if core.shape == ey.shape else 0.0
                )
                core_tm = (
                    float(np.sum(intensity_tm[core])) if core.shape == ez.shape else 0.0
                )
                numerator = (float(np.sum(intensity)) * dx_um**2) ** 2
                denominator = max(float(np.sum(intensity**2)) * dx_um**2, 1e-30)
                k_eff = float(max(np.imag(neff), 0.0))
                loss_db_cm = (
                    0.0
                    if k_eff == 0.0
                    else (4.0 * np.pi * k_eff / wavelength_cm) * (10.0 / np.log(10.0))
                )
                rows.append(
                    {
                        "wavelength": float(wavelength_um),
                        "n eff": float(np.real(neff)),
                        "k eff": k_eff,
                        "loss (dB/cm)": loss_db_cm,
                        "TE (Ey) fraction": total_te / total,
                        "wg TE fraction": core_te / max(total_te, 1e-30),
                        "wg TM fraction": core_tm / max(total_tm, 1e-30),
                        "mode area": numerator / denominator,
                    }
                )
                index.append((float(freq), int(mode_index)))
        return pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(index, names=("f", "mode_index")),
        )


class ModeSolver:
    """Convenience wrapper for solving modes on a simulation plane."""

    def __init__(
        self,
        *,
        simulation,
        plane,
        mode_spec: ModeSpec | None = None,
        freqs,
    ):
        self.simulation = simulation
        self.plane = plane
        self.mode_spec = mode_spec if mode_spec is not None else ModeSpec()
        self.freqs = np.asarray(freqs, dtype=float).reshape(-1)
        if self.freqs.size == 0:
            raise ValueError("ModeSolver requires at least one frequency.")
        self._modes = None

    def solve(self):
        eps, axis, center, axis_index, grid_index, eps_profile_full, crop_slices = (
            self._plane_eps_context()
        )
        del eps, grid_index
        eps_profile = eps_profile_full[crop_slices]
        neffs_by_freq = []
        e_by_freq = []
        h_by_freq = []
        eps_by_freq = []
        for freq in self.freqs:
            neffs, e_fields, h_fields, _ = solve_modes(
                eps=eps_profile,
                omega=2.0 * np.pi * float(freq),
                dL=self.simulation.resolution,
                m=int(self.mode_spec.num_modes),
                direction=f"-{axis}",
                filter_pol=self.mode_spec.polarization,
                target_neff=self.mode_spec.target_neff,
                return_fields=True,
            )
            neffs_by_freq.append(np.asarray(neffs))
            e_by_freq.append(np.asarray(e_fields))
            h_by_freq.append(np.asarray(h_fields))
            eps_by_freq.append(np.asarray(eps_profile))
        self._modes = ModeData(
            frequencies=self.freqs,
            neffs=np.asarray(neffs_by_freq),
            e_fields=np.asarray(e_by_freq),
            h_fields=np.asarray(h_by_freq),
            eps_profiles=np.asarray(eps_by_freq),
            resolution=float(self.simulation.resolution),
        )
        return self._modes

    def _plane_eps_context(self):
        grid = self.simulation.design.rasterize(resolution=self.simulation.resolution)
        eps = np.asarray(grid.permittivity)
        axis, center, _spans = _plane_axis_and_spans(self.plane)
        offset = getattr(self.simulation, "coordinate_offset", (0.0, 0.0, 0.0))
        center = tuple(c + o for c, o in zip(center, offset, strict=True))
        axis_index = {"z": 0, "y": 1, "x": 2}[axis]
        grid_index = int(
            np.clip(
                round(
                    center[{"z": 2, "y": 1, "x": 0}[axis]] / self.simulation.resolution
                ),
                0,
                eps.shape[axis_index] - 1,
            )
        )
        eps_profile_full = np.take(eps, grid_index, axis=axis_index)
        _plane_center, plane_size = _plane_center_and_size(self.plane)
        profile_axes = tuple(idx for idx in range(eps.ndim) if idx != axis_index)
        crop_slices = _profile_crop_slices(
            eps_profile_full,
            profile_axes=profile_axes,
            center=center,
            size=plane_size,
            resolution=float(self.simulation.resolution),
        )
        return (
            eps,
            axis,
            center,
            axis_index,
            grid_index,
            eps_profile_full,
            crop_slices,
        )

    def to_source(
        self,
        *,
        mode_index=0,
        direction="+",
        source_time=None,
        polarization=None,
        power=1.0,
    ):
        if source_time is None:
            freq0 = float(self.freqs[len(self.freqs) // 2])
            source_time = GaussianPulse(freq0=freq0, fwidth=freq0 / 10.0)
        signal, signal_quadrature = source_time.sample(self.simulation.time)
        axis, center, spans = _plane_axis_and_spans(self.plane)
        offset = getattr(self.simulation, "coordinate_offset", (0.0, 0.0, 0.0))
        center = tuple(c + o for c, o in zip(center, offset, strict=True))
        sign = str(direction)[0] if str(direction).startswith(("+", "-")) else "+"
        full_direction = f"{sign}{axis}"
        freq0 = float(getattr(source_time, "freq0", self.freqs[len(self.freqs) // 2]))
        (
            _eps,
            _axis,
            _center_abs,
            _axis_index,
            _grid_index,
            eps_profile_full,
            crop_slices,
        ) = self._plane_eps_context()
        mode_count = max(int(mode_index) + 1, int(self.mode_spec.num_modes))
        selected = int(mode_index)
        profile_freqs = None
        num_freqs = getattr(self.mode_spec, "num_freqs", None)
        if getattr(self.mode_spec, "num_freqs", None):
            profile_freqs = _chebyshev_frequency_nodes(
                freq0,
                float(getattr(source_time, "fwidth", freq0 / 10.0)),
                int(self.mode_spec.num_freqs),
            )
        return ModeSource(
            grid=self.simulation.design.rasterize(
                resolution=self.simulation.resolution
            ),
            center=center,
            width=float(spans[0]),
            height=float(spans[1]) if len(spans) > 1 else None,
            wavelength=LIGHT_SPEED / freq0,
            pol=polarization or getattr(self.mode_spec, "polarization", None) or "te",
            signal=signal,
            signal_quadrature=signal_quadrature,
            source_time=source_time,
            profile_frequencies=profile_freqs,
            num_freqs=num_freqs,
            mode_eps_profile_full=eps_profile_full,
            mode_crop_slices=crop_slices,
            mode_index=selected,
            mode_target_neff=self.mode_spec.target_neff,
            mode_num_modes=mode_count,
            direction=full_direction,
            power=power,
        )

    def sim_with_source(
        self,
        *,
        mode_index=0,
        direction="+",
        source_time=None,
        polarization=None,
        power=1.0,
    ):
        source = self.to_source(
            mode_index=mode_index,
            direction=direction,
            source_time=source_time,
            polarization=polarization,
            power=power,
        )
        return self.simulation.copy(update={"sources": [source]})

    def plot_field_components(self, *_, **kwargs):
        from beamz.visual.mpl import plot_mode_fields

        if "field_names" in kwargs:
            kwargs["components"] = tuple(kwargs.pop("field_names"))
        kwargs.pop("mode_indices", None)
        kwargs.setdefault("val", "abs")
        if "f" in kwargs:
            frequency = float(kwargs.pop("f"))
        else:
            frequency = float(self.freqs[0])
        axis, center, _spans = _plane_axis_and_spans(self.plane)
        if axis != "x":
            raise NotImplementedError(
                "ModeSolver plotting currently supports x-normal planes."
            )
        offset = getattr(self.simulation, "coordinate_offset", (0.0, 0.0, 0.0))
        plane_x = center[0] + offset[0]
        if "origin" not in kwargs:
            kwargs["origin"] = offset
        if "window" not in kwargs:
            plane = self.plane
            plane_center = getattr(plane, "center", center)
            plane_size = getattr(plane, "size", (0.0, 0.0, 0.0))
            if len(plane_center) == 2:
                plane_center = (plane_center[0], plane_center[1], 0.0)
            if len(plane_size) == 2:
                plane_size = (plane_size[0], plane_size[1], 0.0)
            y_center = float(plane_center[1]) + float(offset[1])
            z_center = float(plane_center[2]) + float(offset[2])
            kwargs["window"] = (
                y_center - 0.5 * float(plane_size[1]),
                y_center + 0.5 * float(plane_size[1]),
                z_center - 0.5 * float(plane_size[2]),
                z_center + 0.5 * float(plane_size[2]),
            )
        return plot_mode_fields(
            self.simulation.design.rasterize(resolution=self.simulation.resolution),
            plane_x=plane_x,
            wavelength=LIGHT_SPEED / frequency,
            polarization=getattr(self.mode_spec, "polarization", None),
            num_modes=int(self.mode_spec.num_modes),
            target_neff=getattr(self.mode_spec, "target_neff", None),
            show=kwargs.pop("show", False),
            **kwargs,
        )
