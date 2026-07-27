"""Labeled-data and modal-analysis adapters for raw simulation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from beamz.analysis.data import analysis_data, analysis_inputs
from beamz.const import LIGHT_SPEED
from beamz.devices._immutable import readonly_array
from beamz.devices.monitors.monitors import ModeMonitor


def mode_data_to_dataframe(modes):
    """Return effective index, loss, and mode area as a pandas DataFrame."""
    import pandas as pd

    rows = []
    index = []
    dx_um = float(modes.resolution) * 1e6
    for frequency_index, frequency in enumerate(modes.frequencies):
        wavelength_um = LIGHT_SPEED / float(frequency) * 1e6
        wavelength_cm = LIGHT_SPEED / float(frequency) * 100.0
        for mode_index, neff in enumerate(np.atleast_1d(modes.neffs[frequency_index])):
            electric = np.asarray(modes.e_fields[frequency_index, mode_index])
            intensity = np.sum(np.abs(electric) ** 2, axis=0)
            numerator = (float(np.sum(intensity)) * dx_um**2) ** 2
            denominator = max(float(np.sum(intensity**2)) * dx_um**2, 1e-30)
            loss = float(max(np.imag(neff), 0.0))
            rows.append(
                {
                    "wavelength": wavelength_um,
                    "n eff": float(np.real(neff)),
                    "k eff": loss,
                    "loss (dB/cm)": 0.0
                    if loss == 0.0
                    else 4.0 * np.pi * loss / wavelength_cm * (10.0 / np.log(10.0)),
                    "mode area": numerator / denominator,
                }
            )
            index.append((float(frequency), int(mode_index)))
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=("f", "mode_index")),
    )


_FIELD_UNITS = {
    "Ex": "V/m",
    "Ey": "V/m",
    "Ez": "V/m",
    "Hx": "A/m",
    "Hy": "A/m",
    "Hz": "A/m",
    "permittivity": "relative",
}


def _xarray():
    import xarray as xr

    return xr


def _spatial_dims(ndim, plane_2d="xy"):
    if ndim == 3:
        return ("z", "y", "x")
    if ndim == 2:
        return {"yz": ("z", "y"), "xz": ("z", "x")}.get(
            str(plane_2d).lower(), ("y", "x")
        )
    if ndim == 1:
        return ("s",)
    return tuple(f"dim_{idx}" for idx in range(ndim))


def _time_coords(length, *, times=None, steps=None):
    if times is not None and len(times) == length:
        coords = {"t": ("t", np.asarray(times, dtype=float), {"units": "s"})}
        if steps is not None and len(steps) == length:
            coords["step"] = ("t", np.asarray(steps, dtype=int), {})
        return "t", coords
    coords = {"frame": ("frame", np.arange(length, dtype=int), {})}
    if steps is not None and len(steps) == length:
        coords["step"] = ("frame", np.asarray(steps, dtype=int), {})
    return "frame", coords


def _axis_coords(dims, shape, resolution=None):
    coords = {}
    for dim, length in zip(dims, shape, strict=True):
        values = np.arange(int(length), dtype=float)
        attrs = {}
        if resolution is not None and dim in {"x", "y", "z", "s"}:
            values *= float(resolution)
            attrs["units"] = "m"
        coords[dim] = (dim, values, attrs)
    return coords


def _yee_coords(name, dims, shape, simulation):
    if len(shape) not in {2, 3}:
        return None
    resolution = simulation.resolution
    raw_shape = simulation.fields.grid_shape
    try:
        if len(shape) == 3:
            from beamz.lattice import component_coordinates_3d_um

            coords_um = component_coordinates_3d_um(
                name, raw_shape if len(raw_shape) == 3 else shape, resolution * 1e6
            )
        else:
            from beamz.lattice import component_coordinates_2d_um

            coords_um = component_coordinates_2d_um(
                name,
                raw_shape,
                resolution * 1e6,
                simulation.plane_2d,
            )
    except Exception:
        return None
    if any(
        dim not in coords_um or len(coords_um[dim]) != length
        for dim, length in zip(dims, shape, strict=True)
    ):
        return None
    return {
        dim: (dim, np.asarray(coords_um[dim], dtype=float) * 1e-6, {"units": "m"})
        for dim in dims
    }


def _field_data_array(values, *, name, simulation, times=None, steps=None):
    arr = np.asarray(values)
    if arr.ndim < 2:
        raise ValueError(f"Field {name!r} must be at least 2D, got {arr.shape}.")
    plane = simulation.plane_2d
    resolution = simulation.resolution
    if arr.ndim in {3, 4}:
        time_dim, coords = _time_coords(arr.shape[0], times=times, steps=steps)
        spatial_dims = _spatial_dims(arr.ndim - 1, plane)
        dims = (time_dim, *spatial_dims)
        shape = arr.shape[1:]
    else:
        coords = {}
        spatial_dims = _spatial_dims(arr.ndim, plane)
        dims, shape = spatial_dims, arr.shape
    spatial_coords = _yee_coords(str(name), spatial_dims, shape, simulation)
    coords.update(spatial_coords or _axis_coords(spatial_dims, shape, resolution))
    attrs: dict[str, object] = {
        "component": str(name),
        "units": _FIELD_UNITS.get(str(name), ""),
        "design_width": simulation.width,
        "design_height": simulation.height,
        "design_depth": simulation.depth,
    }
    return _xarray().DataArray(
        arr, dims=dims, coords=coords, name=str(name), attrs=attrs
    )


def to_xarray(results):
    """Return stored simulation fields as an xarray Dataset."""
    xr = _xarray()
    if isinstance(results, xr.Dataset):
        return results
    inputs = analysis_inputs(results)
    data = next(
        (
            item
            for item in inputs.values()
            if not item.frequencies.size
            and any(name not in {"power", "flux", "step"} for name in item.fields)
        ),
        next(iter(inputs.values())),
    )
    fields = {
        name: values
        for name, values in data.fields.items()
        if name not in {"power", "flux", "step"}
    }
    data_vars = {
        name: _field_data_array(
            values,
            name=name,
            simulation=data.coordinates,
            times=data.coordinates.time,
            steps=data.fields.get("step"),
        )
        for name, values in fields.items()
    }
    width, height, depth = (
        data.coordinates.width,
        data.coordinates.height,
        data.coordinates.depth,
    )
    return xr.Dataset(
        data_vars=data_vars,
        attrs={
            "beamz_kind": "SimulationResults",
            "resolution": data.resolution,
            "plane_2d": data.plane_2d,
            "design_width": width,
            "design_height": height,
            "design_depth": depth,
        },
    )


def monitor_to_xarray(result):
    """Return one canonical analysis contract as an xarray Dataset."""
    xr = _xarray()
    data = analysis_data(result)
    data_vars = {}
    times = data.coordinates.time
    for name, values in data.fields.items():
        if name in {"t", "step"}:
            continue
        arr = np.asarray(values)
        if arr.size == 0:
            continue
        if (
            name != "power"
            and data.frequencies.size
            and arr.shape[0] == data.frequencies.size
        ):
            dims = ("f", *tuple(f"sample_{idx}" for idx in range(1, arr.ndim)))
            coords = {"f": ("f", data.frequencies, {"units": "Hz"})}
            data_vars[name] = xr.DataArray(arr, dims=dims, coords=coords, name=name)
            continue
        if arr.ndim == 1:
            arr = arr[:, None]
        dim, coords = _time_coords(arr.shape[0], times=times)
        spatial = _spatial_dims(arr.ndim - 1)
        coords.update(_axis_coords(spatial, arr.shape[1:]))
        data_vars[name] = xr.DataArray(
            arr,
            dims=(dim, *spatial),
            coords=coords,
            name=name,
            attrs={"units": _FIELD_UNITS.get(str(name), "")},
        )
    return xr.Dataset(
        data_vars=data_vars,
        attrs={"monitor_name": data.name},
    )


@dataclass(frozen=True)
class ModeMonitorData:
    """Store labeled modal amplitudes and projection diagnostics.

    Parameters
    ----------
    monitor : object
        Detached mode-monitor configuration.
    amps : array-like or xarray.DataArray
        Complex forward/backward modal amplitudes.
    flux : array-like or xarray.DataArray, optional
        Total measured flux in watts.
    modal_flux : array-like or xarray.DataArray, optional
        Flux reconstructed from projected modes.
    projection_residual : array-like or xarray.DataArray, optional
        Relative field reconstruction residual.
    condition_number : array-like or xarray.DataArray, optional
        Conditioning of the modal overlap system.
    """

    monitor: object
    amps: Any
    flux: Any = None
    modal_flux: Any = None
    projection_residual: Any = None
    condition_number: Any = None

    def __post_init__(self):
        for name in (
            "amps",
            "flux",
            "modal_flux",
            "projection_residual",
            "condition_number",
        ):
            value = getattr(self, name)
            if value is not None:
                if type(value).__module__.startswith("xarray"):
                    value = value.copy(deep=True)
                    value.data = readonly_array(value.data)
                else:
                    value = readonly_array(value)
                object.__setattr__(self, name, value)

    def to_xarray(self):
        """Return amplitudes and diagnostics as an xarray Dataset."""
        return _xarray().Dataset(
            data_vars={
                name: value
                for name, value in vars(self).items()
                if name != "monitor" and value is not None
            }
        )


def _data_array_1d(values, *, name, freqs):
    return _xarray().DataArray(
        np.asarray(values),
        dims=("f",),
        coords={"f": ("f", np.asarray(freqs, dtype=float), {"units": "Hz"})},
        name=name,
    )


def mode_data(results, name):
    """Project one raw mode-monitor result into labeled modal data.

    Parameters
    ----------
    results : SimulationResults
        Detached simulation results containing the named mode monitor.
    name : str
        Monitor result name.

    Returns
    -------
    ModeMonitorData
        Forward/backward amplitudes, flux, and projection diagnostics labeled by
        frequency and mode.

    Raises
    ------
    KeyError
        If no monitor result exists under ``name``.
    """
    from beamz.analysis import sparameters as _sp
    from beamz.devices.modes.specs import ModeSpec
    from beamz.devices.ports import Port

    data = analysis_data(results, name)
    monitor = data.monitor_geometry
    if not isinstance(monitor, ModeMonitor):
        raise TypeError(f"Monitor {name!r} is not a mode monitor.")
    freqs = data.frequencies
    num_modes = int(monitor.mode_spec.num_modes or 1)
    monitor_name = str(monitor.name or name)
    ports = [
        Port(
            center=monitor.center,
            size=monitor.size,
            name=f"{monitor_name}_m{idx}",
            monitor_name=monitor_name,
            direction="+",
            mode_spec=ModeSpec(
                mode_index=idx,
                num_modes=num_modes,
                polarization=monitor.mode_spec.polarization or "te",
            ),
        )
        for idx in range(num_modes)
    ]
    waves = _sp._extract_port_waves_dft(results, ports=ports, frequencies=freqs)
    amps = np.zeros((freqs.size, 2, num_modes), dtype=np.complex128)
    for idx, port in enumerate(ports):
        wave = waves[port.name]
        positive_selector, negative_selector = _sp._wave_selectors(
            port, is_3d=data.is_3d
        )
        amps[:, 0, idx] = wave[f"a_{positive_selector}"]
        amps[:, 1, idx] = wave[f"a_{negative_selector}"]
    diagnostics = waves[ports[0].name]
    default = np.full(freqs.size, np.nan)
    modal_flux = np.asarray(
        diagnostics.get("projected_signed_power", default), dtype=float
    )
    xr = _xarray()
    amp_data = xr.DataArray(
        amps,
        dims=("f", "direction", "mode_index"),
        coords={
            "f": ("f", freqs, {"units": "Hz"}),
            "direction": ("direction", np.asarray(["+", "-"], dtype=object)),
            "mode_index": ("mode_index", np.arange(num_modes)),
        },
        name="amps",
        attrs={"monitor_name": monitor_name},
    )
    flux = np.asarray(data.fields.get("flux", np.full(freqs.size, np.nan)))
    return ModeMonitorData(
        monitor=monitor,
        amps=amp_data,
        flux=flux
        if hasattr(flux, "dims")
        else _data_array_1d(flux, name="flux", freqs=freqs),
        modal_flux=_data_array_1d(modal_flux, name="modal_flux", freqs=freqs),
        projection_residual=_data_array_1d(
            diagnostics.get("projection_residual", default),
            name="projection_residual",
            freqs=freqs,
        ),
        condition_number=_data_array_1d(
            diagnostics.get("condition_number", default),
            name="condition_number",
            freqs=freqs,
        ),
    )
