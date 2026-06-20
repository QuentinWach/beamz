from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from beamz.simulation.core import PortSpec, _source_spectrum_normalization


@dataclass(frozen=True)
class ModeMonitorData:
    """Labeled modal-amplitude data for a ModeMonitor."""

    monitor: object
    amps: xr.DataArray
    flux: xr.DataArray | None = None

    def to_xarray(self):
        data_vars = {"amps": self.amps}
        if self.flux is not None:
            data_vars["flux"] = self.flux
        return xr.Dataset(
            data_vars=data_vars,
            attrs={"monitor_name": getattr(self.monitor, "name", None)},
        )


def mode_monitor_data(simulation, monitor):
    freqs = np.asarray(monitor.get_dft_frequencies(), dtype=float)
    mode_spec = getattr(monitor, "mode_spec", None)
    num_modes = int(getattr(mode_spec, "num_modes", 1) or 1)
    name = str(getattr(monitor, "name", "mode"))

    ports = [
        PortSpec(
            name=f"{name}_m{idx}",
            monitor_name=name,
            direction=getattr(monitor, "direction"),
            polarization=getattr(monitor, "polarization"),
            mode_index=idx,
        )
        for idx in range(num_modes)
    ]
    waves = simulation.extract_port_waves_dft(ports=ports, frequencies=freqs)

    amps = np.zeros((freqs.size, 2, num_modes), dtype=np.complex128)
    monitor_direction = str(getattr(monitor, "direction", "+x"))
    for idx, port in enumerate(ports):
        wave = waves[port.name]
        if monitor_direction.startswith("+"):
            plus = wave["a_minus"]
            minus = wave["a_plus"]
        else:
            plus = wave["a_plus"]
            minus = wave["a_minus"]
        amps[:, 0, idx] = np.asarray(plus, dtype=np.complex128)
        amps[:, 1, idx] = np.asarray(minus, dtype=np.complex128)

    source_norm = _source_spectrum_normalization(
        simulation.sources,
        freqs,
        time=getattr(simulation, "time", None),
        monitor=monitor,
    )
    if source_norm is not None:
        norm = np.asarray(source_norm, dtype=np.complex128).reshape(-1)
        if norm.size == freqs.size:
            valid = np.abs(norm) > 1e-12
            amps = np.divide(
                amps,
                norm[:, None, None],
                out=np.zeros_like(amps, dtype=np.complex128),
                where=valid[:, None, None],
            )

    amp_da = xr.DataArray(
        amps,
        dims=("f", "direction", "mode_index"),
        coords={
            "f": ("f", freqs, {"units": "Hz"}),
            "direction": ("direction", np.asarray(["+", "-"], dtype=object)),
            "mode_index": ("mode_index", np.arange(num_modes, dtype=int)),
        },
        name="amps",
        attrs={"monitor_name": name},
    )

    flux_da = None
    if hasattr(monitor, "get_dft_flux"):
        try:
            flux = np.asarray(monitor.get_dft_flux(), dtype=float)
        except Exception:
            flux = np.asarray(())
        if flux.size == freqs.size:
            if source_norm is not None:
                norm = np.asarray(source_norm, dtype=np.complex128).reshape(-1)
                if norm.size == flux.size:
                    scale = np.abs(norm) ** 2
                    flux = np.divide(
                        flux,
                        scale,
                        out=np.zeros_like(flux, dtype=float),
                        where=scale > 1e-24,
                    )
            flux_da = xr.DataArray(
                flux,
                dims=("f",),
                coords={"f": ("f", freqs, {"units": "Hz"})},
                name="flux",
                attrs={"monitor_name": name},
            )

    return ModeMonitorData(monitor=monitor, amps=amp_da, flux=flux_da)
