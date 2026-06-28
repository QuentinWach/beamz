from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from beamz.const import LIGHT_SPEED, µm
from beamz.design.core import Design
from beamz.design.materials import Material
from beamz.design.structures import Box, Structure
from beamz.devices.monitors.monitors import Monitor
from beamz.devices.ports import Port
from beamz.devices.sources.compiler import (
    apply_compiled_source_specs,
    compile_source_specs,
    source_supports_compiled_specs,
)
from beamz.devices.sources.mode import (
    _detect_transverse_symmetry_axes,
    _enforce_componentwise_parity,
    _make_3d_mode_basis_profiles,
    _modal_overlap_3d_profiles,
    _normalize_3d_profiles_by_flux,
    _numeric_phase_delay,
    _select_core_confined_mode_index,  # noqa: F401 - compatibility monkeypatch hook
    _solve_numeric_k_axis,
)
from beamz.devices.sources.solve import solve_beamz_mode_plane, solve_modes
from beamz.simulation.boundaries import (
    PML,
    Boundary,
    create_metallic_boundary_masks,
    has_full_pec_3d,
    initialize_full_pec_3d_state,
    normalize_boundaries,
    sync_full_pec_3d_from_compact,
)
from beamz.simulation.compiled import (
    EngineState,
    MonitorState,
    _add_array_entries,
    _memory_report,
    compile_simulation,
    monitor_dft_accumulator_dtype,
    monitor_dft_point_size,
    monitor_frequency_size,
    monitor_state_size,
    sharding_cache_token,
)
from beamz.simulation.fields import Fields
from beamz.simulation.step_sequence import run_step_sequence
from beamz.simulation.yee import (
    component_coordinates_3d_um,
    sample_voxel_grid_at_component_3d,
    sample_voxel_grid_at_e_component_3d_centered,
)
from beamz.visual.helpers import _finish_inline_progress, _print_inline_progress


def _copy_with_update(obj, update=None):
    import copy

    copied = copy.deepcopy(obj)
    if update:
        for key, value in dict(update).items():
            setattr(copied, key, value)
    return copied


def _compiled_cache_value_token(value, *, _seen=None):
    """Build a conservative cache token for values that affect compiled specs."""
    if _seen is None:
        _seen = set()

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, slice):
        return ("slice", value.start, value.stop, value.step)
    if callable(value):
        return ("callable", id(value))

    try:
        arr = np.asarray(value)
    except Exception:
        arr = None
    if arr is not None and hasattr(value, "shape") and hasattr(value, "dtype"):
        token = ("array", tuple(int(v) for v in arr.shape), str(arr.dtype))
        if arr.size <= 4096:
            try:
                token = (*token, hash(arr.tobytes()))
            except Exception:
                token = (*token, id(value))
        else:
            token = (*token, id(value))
        return token

    obj_id = id(value)
    if obj_id in _seen:
        return ("cycle", obj_id)
    _seen.add(obj_id)
    try:
        if isinstance(value, Mapping):
            return (
                "mapping",
                tuple(
                    sorted(
                        (
                            _compiled_cache_value_token(k, _seen=_seen),
                            _compiled_cache_value_token(v, _seen=_seen),
                        )
                        for k, v in value.items()
                    )
                ),
            )
        if isinstance(value, (tuple, list)):
            return (
                type(value).__name__,
                tuple(_compiled_cache_value_token(v, _seen=_seen) for v in value),
            )
        if isinstance(value, set):
            return (
                "set",
                tuple(
                    sorted(_compiled_cache_value_token(v, _seen=_seen) for v in value)
                ),
            )

        attrs = getattr(value, "__dict__", None)
        if attrs is not None:
            runtime_attrs = set(getattr(value, "_RUNTIME_ATTRS", set()))
            public_attrs = tuple(
                sorted(
                    (
                        key,
                        _compiled_cache_value_token(attr_value, _seen=_seen),
                    )
                    for key, attr_value in attrs.items()
                    if key not in runtime_attrs and key != "_state"
                )
            )
            return (
                value.__class__.__module__,
                value.__class__.__qualname__,
                obj_id,
                public_attrs,
            )
        return (type(value).__module__, type(value).__qualname__, repr(value))
    finally:
        _seen.discard(obj_id)


def _compiled_cache_sequence_token(values):
    return _compiled_cache_value_token(tuple(values or ()))


def _material_index(material) -> float:
    eps = getattr(material, "permittivity", 1.0)
    try:
        eps_value = float(np.max(np.real(np.asarray(eps))))
    except Exception:
        eps_value = 1.0
    return float(np.sqrt(max(eps_value, 1.0)))


def _max_index_for_specs(background, structures) -> float:
    values = [_material_index(background)]
    for structure in structures or ():
        material = getattr(structure, "material", None) or getattr(
            structure, "medium", None
        )
        if material is not None:
            values.append(_material_index(material))
    return max(values) if values else 1.0


def _design_depth(design) -> float:
    return float(getattr(design, "depth", 0.0) or 0.0)


def _design_domain(design):
    return (float(design.width), float(design.height), _design_depth(design))


def _design_is_3d(design) -> bool:
    return bool(getattr(design, "is_3d", False) and _design_depth(design) > 0.0)


def _structure_to_domain(structure, offset, domain_size):
    if isinstance(structure, Structure):
        return structure.to_beamz_structure(offset=offset, domain_size=domain_size)
    if isinstance(structure, Box):
        geometry = structure
        if any(not np.isfinite(v) for v in geometry.size):
            clipped_size = tuple(
                (
                    float(domain)
                    if not np.isfinite(size)
                    else min(float(size), float(domain))
                )
                for size, domain in zip(geometry.size, domain_size, strict=True)
            )
            geometry = Box(
                center=geometry.center,
                size=clipped_size,
                material=geometry.material,
            )
        return geometry.to_rectangle(offset=offset, material=geometry.material)
    if hasattr(structure, "to_beamz_structure"):
        return structure.to_beamz_structure(offset=offset, domain_size=domain_size)

    copied = _copy_with_update(structure)
    if hasattr(copied, "shift"):
        copied = copied.shift(*offset)
    return copied


def _shift_device_to_domain(device, offset):
    if hasattr(device, "shifted"):
        return device.shifted(offset)
    copied = _copy_with_update(device)
    offset = tuple(float(v) for v in offset)
    if hasattr(copied, "center") and copied.center is not None:
        copied.center = tuple(
            a + b for a, b in zip(copied.center, offset, strict=False)
        )
    if hasattr(copied, "position") and copied.position is not None:
        copied.position = tuple(
            a + b for a, b in zip(copied.position, offset, strict=False)
        )
    return copied


def _pml_merge_mode(key: str) -> str:
    key = str(key)
    if key == "mask":
        return "or"
    if key == "formulation":
        return "validate"
    if "kappa" in key or "alpha" in key:
        return "max"
    return "add"


def _merge_pml_payload(lhs, rhs, *, key: str | None = None):
    if isinstance(lhs, Mapping) and isinstance(rhs, Mapping):
        merged = dict(lhs)
        for child_key, child_value in rhs.items():
            if child_key in merged:
                merged[child_key] = _merge_pml_payload(
                    merged[child_key], child_value, key=child_key
                )
            else:
                merged[child_key] = child_value
        return merged

    merge_mode = _pml_merge_mode("" if key is None else key)
    if merge_mode == "validate":
        lhs_formulation = str(lhs).lower()
        rhs_formulation = str(rhs).lower()
        if lhs_formulation != rhs_formulation:
            raise ValueError(
                "Cannot merge PML payloads with different formulations: "
                f"{lhs_formulation!r} != {rhs_formulation!r}."
            )
        return lhs_formulation

    lhs_arr = jnp.asarray(lhs)
    rhs_arr = jnp.asarray(rhs)
    if merge_mode == "or":
        return lhs_arr.astype(bool) | rhs_arr.astype(bool)
    if merge_mode == "max":
        return jnp.maximum(lhs_arr, rhs_arr)
    return lhs_arr + rhs_arr


def _safe_modal_overlap_3d(
    field_profiles, mode_profiles, axis, d_area, direction_sign=1.0
):
    """Call the 3D modal overlap helper with backward-compatible kwargs."""

    try:
        return _modal_overlap_3d_profiles(
            field_profiles,
            mode_profiles,
            axis,
            d_area,
            direction_sign=direction_sign,
        )
    except TypeError as exc:
        if "direction_sign" not in str(exc):
            raise
        return _modal_overlap_3d_profiles(
            field_profiles,
            mode_profiles,
            axis,
            d_area,
        )


def _projection_solver_direction_3d(direction: str, axis: str) -> str:
    """Match the 3D local-mode solver branch conventions used by ModeSource."""

    direction = str(direction).lower()
    axis = str(axis).lower()
    if axis in {"x", "y"}:
        return ("-" if direction.startswith("+") else "+") + axis
    return direction


@dataclass(frozen=True)
class PortSpec:
    """Modal extraction metadata for one named port.

    ``reference_monitor`` selects an alternate incident-wave normalization plane.
    Scattered waves are still read from ``monitor_name``; no reference-plane
    subtraction is applied implicitly.
    """

    name: str
    monitor_name: str
    direction: Literal["+x", "-x", "+y", "-y", "+z", "-z"]
    polarization: Literal["tm", "te"]
    mode_index: int = 0
    reference_monitor: str | None = None
    incident_wave: Literal["plus", "minus", "auto"] = "plus"
    scattered_wave: Literal["plus", "minus", "auto"] = "minus"


def _sampled_source_spectrum_normalization(source, freqs, *, time, monitor=None):
    """Return the source waveform in BeamZ's native DFT normalization."""
    freq_arr = np.asarray(freqs, dtype=float).reshape(-1)
    time_arr = np.asarray(time, dtype=float).reshape(-1)
    if freq_arr.size == 0 or time_arr.size == 0:
        return None

    dft_normalization = str(getattr(monitor, "dft_normalization", "native")).lower()
    if dft_normalization != "native":
        return None

    dt = float(np.median(np.diff(time_arr))) if time_arr.size > 1 else 0.0
    signal = getattr(source, "signal", None)
    if isinstance(signal, (np.ndarray, list, tuple, jnp.ndarray)):
        signal_arr = np.asarray(signal, dtype=float).reshape(-1)
        n = min(signal_arr.size, time_arr.size)
        if n <= 0:
            return None
        signal_arr = signal_arr[:n]
        sample_times = time_arr[:n]
    elif hasattr(source, "_get_signal_value") and dt > 0.0:
        sample_times = time_arr
        signal_arr = np.asarray(
            [float(source._get_signal_value(t, dt)) for t in sample_times],
            dtype=float,
        )
    else:
        return None

    dft_t_start = float(getattr(monitor, "dft_t_start", 0.0))
    dft_t_end = getattr(monitor, "dft_t_end", None)
    dft_t_end = np.inf if dft_t_end is None else float(dft_t_end)
    record_interval = max(1, int(getattr(monitor, "dft_record_interval", 1)))
    steps = np.arange(sample_times.size, dtype=int)
    mask = (
        (sample_times >= dft_t_start)
        & (sample_times <= dft_t_end)
        & ((steps % record_interval) == 0)
    )
    if not np.any(mask):
        return None

    sample_times = sample_times[mask]
    signal_arr = signal_arr[mask]
    if str(getattr(monitor, "dft_window", "rect")).lower() == "hann" and np.isfinite(
        dft_t_end
    ):
        span = max(dft_t_end - dft_t_start, 1e-30)
        tau = np.clip((sample_times - dft_t_start) / span, 0.0, 1.0)
        weights = 0.5 * (1.0 - np.cos(2.0 * np.pi * tau))
    else:
        weights = np.ones_like(sample_times, dtype=float)

    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-30:
        return None

    phase = np.exp(1j * 2.0 * np.pi * sample_times[:, None] * freq_arr[None, :])
    spectrum = (2.0 / weight_sum) * np.sum(
        (weights * signal_arr)[:, None] * phase,
        axis=0,
    )
    if np.any(np.abs(spectrum) > 1e-12):
        return np.asarray(spectrum, dtype=np.complex128)
    return None


def _source_spectrum_normalization(
    sources,
    freqs,
    *,
    time=None,
    monitor=None,
    fields=None,
    dt=None,
) -> np.ndarray | None:
    """Return a source spectrum for source-normalized DFT outputs."""
    freq_arr = np.asarray(freqs, dtype=float).reshape(-1)
    if freq_arr.size == 0:
        return None
    spectra = []
    for source in sources or ():
        if time is not None:
            spectrum = _sampled_source_spectrum_normalization(
                source,
                freq_arr,
                time=time,
                monitor=monitor,
            )
            if spectrum is not None:
                spectra.append(
                    _apply_source_launch_power_normalization(
                        source,
                        spectrum,
                        freq_arr,
                        fields=fields,
                        dt=dt,
                    )
                )
                continue
        spectrum = None
        if hasattr(source, "source_spectrum"):
            spectrum = source.source_spectrum(freq_arr, normalize=True)
        source_time = getattr(source, "source_time", None)
        if (
            spectrum is None
            and source_time is not None
            and hasattr(source_time, "dft_normalization_spectrum")
        ):
            spectrum = source_time.dft_normalization_spectrum(freq_arr)
        if (
            spectrum is None
            and source_time is not None
            and hasattr(source_time, "spectrum")
        ):
            spectrum = source_time.spectrum(freq_arr, normalize=True)
        if spectrum is None:
            continue
        spectrum = np.asarray(spectrum, dtype=np.complex128).reshape(-1)
        if spectrum.shape == freq_arr.shape and np.any(np.abs(spectrum) > 1e-12):
            spectra.append(
                _apply_source_launch_power_normalization(
                    source,
                    spectrum,
                    freq_arr,
                    fields=fields,
                    dt=dt,
                )
            )
    if not spectra:
        return None
    return spectra[0]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _design_grid_shape_estimate(design, resolution) -> tuple[int, ...]:
    size = _design_domain(design) if design is not None else None
    if size is None:
        return ()
    res = float(resolution)
    if (not np.isfinite(res)) or res <= 0.0:
        return ()
    dims = tuple(float(value) for value in size)
    if len(dims) < 3 or dims[2] <= 0.0:
        return tuple(max(1, int(round(float(dim) / res))) for dim in dims[:2])
    # BeamZ stores 3D material arrays in z/y/x order.
    return (
        max(1, int(round(dims[2] / res))),
        max(1, int(round(dims[1] / res))),
        max(1, int(round(dims[0] / res))),
    )


def _setup_device_auto_threshold_bytes() -> int:
    raw = os.getenv("BEAMZ_SETUP_CPU_MIN_GIB", "1.0").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 1.0
    return max(0, int(value * 1024**3))


def _has_accelerator_backend() -> bool:
    try:
        import jax

        return any(device.platform != "cpu" for device in jax.devices())
    except Exception:
        return False


def _should_setup_on_cpu(policy, *, design, resolution) -> bool:
    raw = os.getenv("BEAMZ_SETUP_DEVICE", "auto") if policy is None else policy
    if raw is None:
        raw = "auto"
    if not isinstance(raw, str):
        return False
    normalized = raw.strip().lower()
    if normalized in {"cpu", "host"}:
        return True
    if normalized in {"default", "device", "gpu", "accelerator"}:
        return False
    if normalized != "auto":
        raise ValueError(
            "setup_device must be one of 'auto', 'cpu', or 'default', "
            f"got {raw!r}."
        )
    if _env_bool("BEAMZ_SETUP_CPU", False):
        return True
    if not _has_accelerator_backend():
        return False
    shape = _design_grid_shape_estimate(design, resolution)
    if not shape:
        return False
    scalar_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    return scalar_bytes >= _setup_device_auto_threshold_bytes()


def _setup_device_context(policy, *, design, resolution):
    resolved_policy = (
        os.getenv("BEAMZ_SETUP_DEVICE", "auto") if policy is None else str(policy)
    )
    if not _should_setup_on_cpu(resolved_policy, design=design, resolution=resolution):
        return nullcontext(None), "default"
    try:
        import jax

        cpu_devices = jax.devices("cpu")
        if not cpu_devices:
            return nullcontext(None), "default"
        return jax.default_device(cpu_devices[0]), "cpu"
    except Exception:
        return nullcontext(None), "default"


def _setup_device_policy_label(policy) -> str:
    return os.getenv("BEAMZ_SETUP_DEVICE", "auto") if policy is None else str(policy)


def _apply_source_launch_power_normalization(
    source,
    spectrum: np.ndarray,
    freqs: np.ndarray,
    *,
    fields=None,
    dt=None,
) -> np.ndarray:
    """Fold a source's launch-power calibration into a DFT amplitude spectrum."""
    method = getattr(source, "_launch_power_normalization_spectrum", None)
    if method is None:
        return spectrum
    try:
        power = method(freqs, fields=fields, dt=dt)
    except Exception:
        return spectrum
    if power is None:
        return spectrum

    power_arr = np.asarray(power, dtype=float).reshape(-1)
    if power_arr.shape != np.asarray(freqs, dtype=float).reshape(-1).shape:
        return spectrum
    valid = np.isfinite(power_arr) & (power_arr > 1e-24)
    if not np.any(valid):
        return spectrum

    amplitude = np.ones_like(power_arr, dtype=np.float64)
    amplitude[valid] = np.sqrt(power_arr[valid])
    return np.asarray(spectrum, dtype=np.complex128) * amplitude


@dataclass(frozen=True)
class MonitorResults:
    """Snapshot of one monitor's recorded outputs."""

    monitor: Monitor
    fields: dict[str, tuple[Any, ...]]
    power_history: np.ndarray
    power_timestamps: np.ndarray
    power_spectrum: np.ndarray
    frequency_flux_spectrum: np.ndarray
    source_spectrum_normalization: np.ndarray | None = None
    objective_value: float | None = None
    data: Any = None

    @property
    def flux(self):
        if hasattr(self.monitor, "get_dft_flux"):
            freqs = np.asarray(
                getattr(self.monitor, "get_dft_frequencies")(), dtype=float
            )
            values = np.asarray(self.monitor.get_dft_flux(), dtype=float)
            norm = self.source_spectrum_normalization
            if norm is not None:
                norm_arr = np.asarray(norm, dtype=np.complex128).reshape(-1)
                if norm_arr.size == values.size:
                    scale = np.abs(norm_arr) ** 2
                    valid = scale > 1e-24
                    values = np.divide(
                        values,
                        scale,
                        out=np.zeros_like(values, dtype=float),
                        where=valid,
                    )
            if freqs.size == values.size:
                try:
                    import xarray as xr

                    return xr.DataArray(
                        values,
                        dims=("f",),
                        coords={"f": ("f", freqs, {"units": "Hz"})},
                        name="flux",
                    )
                except Exception:
                    return values
            return values
        return np.asarray(self.frequency_flux_spectrum, dtype=float)

    def _plot_proxy(self):
        if self.data is not None and hasattr(self.data, "data_vars"):
            fields: dict[str, list[Any]] = {}
            for name, da in self.data.data_vars.items():
                if str(name) not in {
                    "power",
                    "power_spectrum",
                    "frequency_flux_spectrum",
                }:
                    if "t" in da.dims:
                        fields.setdefault(
                            "t", list(np.asarray(da.coords["t"], dtype=float))
                        )
                        fields[str(name)] = [
                            np.asarray(da.isel(t=idx)) for idx in range(da.sizes["t"])
                        ]
                    elif "frame" in da.dims:
                        fields[str(name)] = [
                            np.asarray(da.isel(frame=idx))
                            for idx in range(da.sizes["frame"])
                        ]
            power = (
                np.asarray(self.data["power"])
                if "power" in self.data.data_vars
                else np.asarray(self.power_history, dtype=float)
            )
            power_timestamps = (
                np.asarray(self.data["power"].coords.get("t", ()), dtype=float)
                if "power" in self.data.data_vars and "t" in self.data["power"].coords
                else np.asarray(self.power_timestamps, dtype=float)
            )
            power_spectrum = (
                np.asarray(self.data["power_spectrum"])
                if "power_spectrum" in self.data.data_vars
                else np.asarray(self.power_spectrum)
            )
            return SimpleNamespace(
                fields=fields
                or {name: list(values) for name, values in self.fields.items()},
                power_history=list(power),
                power_timestamps=list(power_timestamps),
                power_spectrum=power_spectrum,
                power_spectrum_frequencies=np.asarray(
                    getattr(self.monitor, "power_spectrum_frequencies", ())
                ),
                monitor_type=getattr(self.monitor, "monitor_type", "line"),
                start=getattr(self.monitor, "start", (0.0, 0.0)),
                end=getattr(self.monitor, "end", (0.0, 0.0)),
                size=getattr(self.monitor, "size", (0.0, 0.0)),
                name=getattr(self.monitor, "name", None),
            )
        return SimpleNamespace(
            fields={name: list(values) for name, values in self.fields.items()},
            power_history=list(np.asarray(self.power_history, dtype=float)),
            power_timestamps=list(np.asarray(self.power_timestamps, dtype=float)),
            power_spectrum=np.asarray(self.power_spectrum),
            power_spectrum_frequencies=np.asarray(
                getattr(self.monitor, "power_spectrum_frequencies", ())
            ),
            monitor_type=getattr(self.monitor, "monitor_type", "line"),
            start=getattr(self.monitor, "start", (0.0, 0.0)),
            end=getattr(self.monitor, "end", (0.0, 0.0)),
            size=getattr(self.monitor, "size", (0.0, 0.0)),
            name=getattr(self.monitor, "name", None),
        )

    def field_plot_data(self, **kwargs):
        """Return monitor-field plot data from this result."""
        from beamz.visual.data import monitor_field_plot_data

        return monitor_field_plot_data(self._plot_proxy(), **kwargs)

    def power_plot_data(self, **kwargs):
        """Return monitor-power plot data from this result."""
        from beamz.visual.data import monitor_power_plot_data

        return monitor_power_plot_data(self._plot_proxy(), **kwargs)

    def plot(self, **kwargs):
        """Plot recorded monitor field data from this result."""
        from beamz.visual.mpl import plot_monitor_field

        kwargs.setdefault("show", False)
        return plot_monitor_field(self._plot_proxy(), **kwargs)

    def show(self, **kwargs):
        """Display recorded monitor field data from this result."""
        kwargs.setdefault("show", True)
        return self.plot(**kwargs)

    def plot_fields(self, **kwargs):
        """Alias for :meth:`plot`."""
        return self.plot(**kwargs)

    def plot_power(self, **kwargs):
        """Plot monitor power history from this result."""
        from beamz.visual.mpl import plot_monitor_power

        kwargs.setdefault("show", False)
        return plot_monitor_power(self._plot_proxy(), **kwargs)

    def show_power(self, **kwargs):
        """Display monitor power history from this result."""
        kwargs.setdefault("show", True)
        return self.plot_power(**kwargs)

    def to_xarray(self):
        """Return this monitor result as an xarray Dataset."""
        return self.data

    @classmethod
    def from_monitor(
        cls,
        monitor: Monitor,
        *,
        source_spectrum_normalization: np.ndarray | None = None,
    ) -> "MonitorResults":
        fields = {
            name: tuple(values)
            for name, values in getattr(monitor, "fields", {}).items()
        }
        power_spectrum = np.asarray(
            getattr(monitor, "power_spectrum", ()), dtype=np.complex64
        )
        state = getattr(monitor, "_state", None)
        legacy_flux = (
            getattr(state, "_frequency_flux_spectrum_legacy", None)
            if state is not None
            else None
        )
        if legacy_flux is None:
            legacy_flux = power_spectrum
        result = cls(
            monitor=monitor,
            fields=fields,
            power_history=np.asarray(
                getattr(monitor, "power_history", ()), dtype=float
            ),
            power_timestamps=np.asarray(
                getattr(monitor, "power_timestamps", ()), dtype=float
            ),
            power_spectrum=power_spectrum,
            frequency_flux_spectrum=np.asarray(legacy_flux, dtype=np.complex64),
            source_spectrum_normalization=source_spectrum_normalization,
            objective_value=getattr(monitor, "objective_value", None),
        )
        from dataclasses import replace

        from beamz.data.xarray import monitor_dataset

        return replace(result, data=monitor_dataset(result))


@dataclass(frozen=True)
class SimulationResults(Mapping[str, Any]):
    """Primary run output with backward-compatible mapping access."""

    simulation: "Simulation"
    fields: Any = None
    field_times: np.ndarray | None = None
    field_steps: np.ndarray | None = None
    monitors: tuple[Monitor, ...] = ()
    monitor_results: dict[str, MonitorResults] | None = None
    snapshots: tuple[dict[str, Any], ...] = ()

    def __post_init__(self):
        if self.fields is None or hasattr(self.fields, "data_vars"):
            return
        from beamz.data.xarray import simulation_fields_dataset

        object.__setattr__(
            self,
            "fields",
            simulation_fields_dataset(
                self.simulation,
                self.fields,
                field_times=self.field_times,
                field_steps=self.field_steps,
            ),
        )

    def __getitem__(self, key: str) -> Any:
        if self.monitor_results and key in self.monitor_results:
            result = self.monitor_results[key]
            monitor = result.monitor
            if type(monitor).__name__ == "ModeMonitor":
                from beamz.data.modal import mode_monitor_data

                return mode_monitor_data(self.simulation, monitor)
            return result
        payload = self.to_dict()
        if key not in payload:
            raise KeyError(key)
        return payload[key]

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = {}
        if self.fields is not None:
            payload["fields"] = self.fields
        if self.field_times is not None:
            payload["field_times"] = self.field_times
        if self.field_steps is not None:
            payload["field_steps"] = self.field_steps
        if self.monitors:
            payload["monitors"] = list(self.monitors)
        if self.monitor_results:
            payload["monitor_results"] = self.monitor_results
        if self.snapshots:
            payload["snapshots"] = list(self.snapshots)
        return payload

    def plot_field(self, monitor_name=None, field=None, **kwargs):
        """Plot a stored field frame from this result."""
        if monitor_name is None and "field_monitor_name" in kwargs:
            monitor_name = kwargs.pop("field_monitor_name")
        if field is None and "field_name" in kwargs:
            field = kwargs.pop("field_name")
        if "f" in kwargs and "frequency" not in kwargs:
            kwargs["frequency"] = kwargs.pop("f")
        if monitor_name is not None:
            monitor_results = self.monitor_results or {}
            if monitor_name not in monitor_results:
                raise KeyError(monitor_name)
            monitor_result = monitor_results[monitor_name]
            monitor = monitor_result.monitor
            if field is None:
                field = kwargs.pop("field", "Ez")
            dft_frequencies = (
                np.asarray(monitor.get_dft_frequencies(), dtype=float)
                if hasattr(monitor, "get_dft_frequencies")
                else np.asarray(())
            )
            if dft_frequencies.size == 0:
                if "frequency" in kwargs:
                    raise ValueError(
                        f"Monitor '{monitor_name}' has no frequency-domain field data."
                    )
                return monitor_result.plot(field=field, **kwargs)
            from beamz.visual.mpl import plot_tidy3d_dft_field

            kwargs.setdefault("show", False)
            return plot_tidy3d_dft_field(
                self.simulation,
                monitor,
                field=field,
                **kwargs,
            )
        from beamz.visual.mpl import plot_simulation_field

        kwargs.setdefault("show", False)
        return plot_simulation_field(self, **kwargs)

    def plot(self, **kwargs):
        """Plot stored simulation snapshots with the matplotlib backend."""
        if not self.snapshots:
            return self.plot_field(**kwargs)
        from beamz.visual.mpl import show_snapshots

        kwargs.setdefault("show", False)
        return show_snapshots(self.snapshots, **kwargs)

    def show(self, **kwargs):
        """Display stored simulation snapshots or stored fields."""
        if not self.snapshots:
            kwargs.setdefault("show", True)
            return self.plot_field(**kwargs)

        kwargs.setdefault("show", True)
        return self.plot(**kwargs)

    def animate(self, **kwargs):
        """Animate stored simulation snapshots."""
        if not self.snapshots:
            raise RuntimeError("No snapshots available. Run with snapshot_field first.")
        from beamz.visual.mpl import show_snapshots

        kwargs.setdefault("show", True)
        return show_snapshots(self.snapshots, **kwargs)

    def save_video(self, filename, **kwargs):
        """Save stored snapshots or saved field frames as a video."""
        if self.snapshots:
            from beamz.visual.mpl import save_snapshot_video

            return save_snapshot_video(self.snapshots, filename=filename, **kwargs)
        if self.fields is not None:
            from beamz.visual.mpl import save_field_video

            return save_field_video(self, filename=filename, **kwargs)
        raise RuntimeError(
            "No snapshots or saved fields available. Run with snapshot_field or "
            "save_fields first."
        )

    def to_xarray(self):
        """Return stored simulation fields as an xarray Dataset."""
        if self.fields is not None:
            return self.fields
        from beamz.data.xarray import simulation_fields_dataset

        return simulation_fields_dataset(self.simulation, None)

    @classmethod
    def from_run(
        cls,
        simulation: "Simulation",
        *,
        fields: dict[str, np.ndarray] | None = None,
        field_times: np.ndarray | list[float] | None = None,
        field_steps: np.ndarray | list[int] | None = None,
        monitors: list[Monitor] | tuple[Monitor, ...] = (),
        snapshots: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    ) -> "SimulationResults" | None:
        monitor_tuple = tuple(monitors)
        snapshot_tuple = tuple(snapshots)
        monitor_results = {}
        for idx, monitor in enumerate(monitor_tuple):
            name = getattr(monitor, "name", None) or f"monitor_{idx}"
            source_norm = None
            if hasattr(monitor, "get_dft_frequencies"):
                try:
                    source_norm = _source_spectrum_normalization(
                        simulation.sources,
                        monitor.get_dft_frequencies(),
                        time=getattr(simulation, "time", None),
                        monitor=monitor,
                        fields=getattr(simulation, "fields", None),
                        dt=getattr(simulation, "dt", None),
                    )
                except Exception:
                    source_norm = None
            monitor_results[name] = MonitorResults.from_monitor(
                monitor,
                source_spectrum_normalization=source_norm,
            )
        if fields is None and not monitor_tuple and not snapshot_tuple:
            return None
        fields_dataset = None
        if fields is not None:
            from beamz.data.xarray import simulation_fields_dataset

            fields_dataset = simulation_fields_dataset(
                simulation,
                fields,
                field_times=field_times,
                field_steps=field_steps,
            )
        return cls(
            simulation=simulation,
            fields=fields_dataset,
            field_times=(
                None if field_times is None else np.asarray(field_times, dtype=float)
            ),
            field_steps=(
                None if field_steps is None else np.asarray(field_steps, dtype=int)
            ),
            monitors=monitor_tuple,
            monitor_results=monitor_results or None,
            snapshots=snapshot_tuple,
        )


class Simulation:
    """FDTD simulation class supporting both 2D and 3D electromagnetic simulations."""

    def __init__(
        self,
        design: Design = None,
        sources: list = None,
        monitors: list[Monitor] = None,
        boundaries: list[Boundary] = None,
        thermal=None,
        resolution: float = 0.02 * µm,
        time: np.ndarray = None,
        plane_2d: str = "xy",
        *,
        domain=None,
        size=None,
        structures: list | None = None,
        material=None,
        background=None,
        medium=None,
        boundary_spec=None,
        grid_spec=None,
        run_time: float | None = None,
        setup_device: Literal["auto", "cpu", "default"] | None = None,
    ):
        coordinate_offset = (0.0, 0.0, 0.0)
        if domain is not None and size is not None:
            domain_tuple = tuple(float(v) for v in domain)
            size_tuple = tuple(float(v) for v in size)
            if domain_tuple != size_tuple:
                raise ValueError("Pass only one of domain=... or size=....")
        if domain is None:
            domain = size
        if medium is not None:
            warnings.warn(
                "Simulation(..., medium=...) is deprecated; use material=... or "
                "Design(background=...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if structures:
            warnings.warn(
                "Simulation(..., structures=[...]) is deprecated; add geometry "
                "with material=... to a Design and pass design=... instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if (
            background is not None
            and material is not None
            and background is not material
        ):
            raise ValueError("Pass only one of background=... or material=....")
        background_material = background if background is not None else material
        if background_material is None:
            background_material = medium

        if design is None:
            if domain is None:
                raise ValueError("Simulation requires either design=... or domain=....")
            sim_size = tuple(float(v) for v in domain)
            if len(sim_size) == 2:
                sim_size = (sim_size[0], sim_size[1], 0.0)
            if len(sim_size) != 3:
                raise ValueError("Simulation size must be a 2D or 3D tuple.")
            background_material = (
                background_material
                if background_material is not None
                else Material(1.0)
            )
            design = Design(
                width=sim_size[0],
                height=sim_size[1],
                depth=sim_size[2],
                background=background_material,
            )
            coordinate_offset = (
                0.5 * sim_size[0],
                0.5 * sim_size[1],
                0.5 * sim_size[2],
            )
            for structure in structures or ():
                design += _structure_to_domain(structure, coordinate_offset, sim_size)

            if grid_spec is not None:
                resolution = grid_spec.resolve_resolution(
                    max_index=_max_index_for_specs(
                        background_material, structures or ()
                    )
                )
            if time is None and run_time is not None:
                spec = grid_spec
                if spec is None:
                    from beamz.simulation.specs import GridSpec

                    spec = GridSpec.uniform(resolution)
                dims = 3 if sim_size[2] > 0 else 2
                dt = spec.resolve_time_step(resolution, dims=dims)
                time = np.arange(0.0, float(run_time) + 0.5 * dt, dt)
        elif domain is not None and getattr(design, "_centered_coordinates", False):
            sim_size = tuple(float(v) for v in domain)
            if len(sim_size) == 2:
                sim_size = (sim_size[0], sim_size[1], 0.0)
            if len(sim_size) != 3:
                raise ValueError("Simulation size must be a 2D or 3D tuple.")
            background_material = (
                background_material
                if background_material is not None
                else getattr(design, "background", None)
            )
            if background_material is None and design.structures:
                background_material = getattr(design.structures[0], "material", None)
            if background_material is None:
                background_material = Material(1.0)
            source_design = design
            design = Design(
                width=sim_size[0],
                height=sim_size[1],
                depth=sim_size[2],
                background=background_material,
            )
            coordinate_offset = (
                0.5 * sim_size[0],
                0.5 * sim_size[1],
                0.5 * sim_size[2],
            )
            for structure in source_design.structures[1:]:
                design += _structure_to_domain(structure, coordinate_offset, sim_size)

            if grid_spec is not None:
                resolution = grid_spec.resolve_resolution(
                    max_index=_max_index_for_specs(
                        background_material, source_design.structures[1:]
                    )
                )
            if time is None and run_time is not None:
                spec = grid_spec
                if spec is None:
                    from beamz.simulation.specs import GridSpec

                    spec = GridSpec.uniform(resolution)
                dims = 3 if sim_size[2] > 0 else 2
                dt = spec.resolve_time_step(resolution, dims=dims)
                time = np.arange(0.0, float(run_time) + 0.5 * dt, dt)

        if boundary_spec is not None:
            boundaries = list(getattr(boundary_spec, "boundaries", boundary_spec))
        if coordinate_offset != (0.0, 0.0, 0.0):
            sources = [
                _shift_device_to_domain(source, coordinate_offset)
                for source in (sources or [])
            ]
            monitors = [
                _shift_device_to_domain(monitor, coordinate_offset)
                for monitor in (monitors or [])
            ]
        self.design = design
        self.size = _design_domain(design) if design is not None else None
        self.domain = self.size
        self.coordinate_offset = coordinate_offset
        self.grid_spec = grid_spec
        self.run_time = run_time
        self.setup_device_policy = _setup_device_policy_label(setup_device)
        sources = sources or []
        monitors = monitors or []
        boundaries = normalize_boundaries(boundaries, is_3d=_design_is_3d(design))
        self.resolution = resolution
        self.is_3d = _design_is_3d(design)
        self.plane_2d = plane_2d.lower()
        if self.plane_2d not in ["xy", "yz", "xz"]:
            self.plane_2d = "xy"
        self.sources, self.monitors = self._normalize_specs(
            design=design,
            sources=sources,
            monitors=monitors,
        )

        # Initialize time stepping first
        if time is None or len(time) < 2:
            raise ValueError("FDTD requires a time array with at least two entries")
        self.time, self.dt, self.num_steps = time, float(time[1] - time[0]), len(time)
        self.t, self.current_step = float(time[0]), 0

        setup_context, resolved_setup_device = _setup_device_context(
            setup_device,
            design=design,
            resolution=resolution,
        )
        self.setup_device_resolved = resolved_setup_device
        with setup_context:
            # Get material grids from design (design owns the material grids, we reference them)
            permittivity, conductivity, permeability = design.get_material_grids(
                resolution
            )
            # Check for PML boundaries before creating fields (to avoid double material init)
            pml_boundaries = [b for b in boundaries if isinstance(b, PML)]

            # Create field storage (fields owns the E/H field arrays, references material grids)
            self.fields = Fields(
                permittivity,
                conductivity,
                permeability,
                resolution,
                plane_2d=self.plane_2d,
                _init_materials=not pml_boundaries,
            )

            # Initialize PML regions if present
            if pml_boundaries:
                # Create PML regions (do this once, not every timestep)
                pml_data = {}
                for pml in pml_boundaries:
                    new_data = pml.create_pml_regions(
                        self.fields,
                        design,
                        resolution,
                        self.dt,
                        plane_2d=self.plane_2d,
                    )
                    if not pml_data:
                        pml_data = dict(new_data)
                        continue
                    pml_data = _merge_pml_payload(pml_data, new_data)
                self.pml_data = pml_data

                # Set effective conductivity for PML
                self.fields.set_pml_conductivity(pml_data)
            else:
                self.pml_data = None

            # Store boundary references (no duplication)
            self.boundaries = boundaries
            self.fields.set_metallic_masks(
                create_metallic_boundary_masks(
                    self.fields,
                    self.boundaries,
                    is_3d=self.is_3d,
                    plane_2d=self.plane_2d,
                )
            )
        self.fields.boundaries = self.boundaries

        # Optional thermal coupling
        self.thermal = thermal
        if self.thermal is not None and getattr(self.thermal, "enabled", True):
            self.thermal.initialize(self)

        # Compiled program cache for v0.3 packed-source/monitor execution.
        self._compiled_program = None
        self._compiled_program_signature = None
        self._compiled_program_cache = {}
        self._compiled_monitor_state = None
        self._compiled_step_source_specs = None
        self._imperative_step_sources = None

    def copy(self, *, update=None):
        """Return a configuration copy of the simulation."""
        copied = _copy_with_update(self, update=update)
        copied.current_step = 0
        copied.t = (
            float(copied.time[0]) if getattr(copied, "time", None) is not None else 0.0
        )
        copied._compiled_program = None
        copied._compiled_program_signature = None
        copied._compiled_program_cache = {}
        copied._compiled_monitor_state = None
        return copied

    @staticmethod
    def _dedupe_devices(devices):
        seen = set()
        ordered = []
        for device in devices:
            key = id(device)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(device)
        return ordered

    @classmethod
    def _normalize_specs(cls, *, design, sources, monitors):
        normalized_sources = list(sources)
        normalized_monitors = list(monitors)

        normalized_sources = cls._dedupe_devices(normalized_sources)
        normalized_monitors = cls._dedupe_devices(normalized_monitors)
        duplicate_monitor_names = sorted(
            {
                str(name)
                for name in (
                    getattr(monitor, "name", None) for monitor in normalized_monitors
                )
                if name
                and sum(
                    1
                    for monitor in normalized_monitors
                    if getattr(monitor, "name", None) == name
                )
                > 1
            }
        )
        if duplicate_monitor_names:
            names = ", ".join(duplicate_monitor_names)
            raise ValueError(
                "Simulation._normalize_specs found duplicate Monitor.name values: "
                f"{names}. Monitor names must be unique because PortSpec.monitor_name "
                "resolution depends on them."
            )
        return normalized_sources, normalized_monitors

    def step(self):
        """Perform one FDTD time step with correct Huygens source timing.

        Order: H-update → M-injection → E-update → J-injection → legacy sources
        """
        if self.current_step >= self.num_steps:
            return False

        def _pre_e(sim):
            sim._inject_legacy_sources()
            sim._apply_compiled_source_phase("pre_e")
            return sim

        def _prepare(sim):
            return sim, sim._collect_source_terms()

        def _update_h(sim, payload):
            _source_j, source_m = payload
            sim.fields.update_h(sim.dt, source_m=source_m)
            return sim

        def _post_h(sim):
            sim._apply_compiled_source_phase("h")
            sim._inject_h_sources()
            sim.fields.apply_metallic_boundaries_h()
            return sim

        def _update_e(sim, payload):
            source_j, _source_m = payload
            sim.fields.update_e(sim.dt, source_j=source_j)
            return sim

        def _post_e(sim):
            sim._apply_compiled_source_phase("e")
            sim._inject_e_sources()
            sim.fields.apply_metallic_boundaries_e()
            return sim

        def _finalize(sim):
            sim._record_monitors()
            if sim.thermal is not None and getattr(sim.thermal, "enabled", True):
                sim.thermal.step(sim)
            sim.t += sim.dt
            sim.current_step += 1
            return sim

        run_step_sequence(
            self,
            pre_e=_pre_e,
            prepare=_prepare,
            update_h=_update_h,
            post_h=_post_h,
            update_e=_update_e,
            post_e=_post_e,
            finalize=_finalize,
        )
        return True

    def _record_monitors(self):
        """Record data from Monitor devices during simulation."""
        for device in self.monitors:
            should_record = device.should_record(self.current_step)
            dft_every_step = bool(
                getattr(device, "dft_enabled", False)
                and getattr(device, "dft_record_every_step", True)
            )
            if should_record or dft_every_step:
                sample_time = self.t + self.dt
                device._dft_base_dt = self.dt
                if not self.is_3d:
                    device.record_fields_2d(
                        self.fields.Ez,
                        self.fields.Hx,
                        self.fields.Hy,
                        sample_time,
                        self.resolution,
                        self.resolution,
                        self.current_step,
                        Ex=self.fields.Ex,
                        Ey=self.fields.Ey,
                        Hz=self.fields.Hz,
                    )
                else:
                    device.record_fields(
                        self.fields.Ex,
                        self.fields.Ey,
                        self.fields.Ez,
                        self.fields.Hx,
                        self.fields.Hy,
                        self.fields.Hz,
                        sample_time,
                        self.resolution,
                        self.resolution,
                        self.resolution,
                        self.current_step,
                    )

    def _compiled_step_source_groups(self):
        if self._compiled_step_source_specs is not None:
            return self._compiled_step_source_specs

        compiled_sources = [
            device for device in self.sources if source_supports_compiled_specs(device)
        ]
        self._imperative_step_sources = [
            device
            for device in self.sources
            if not source_supports_compiled_specs(device)
        ]
        grouped = {
            "pre_e": {"Ex": [], "Ey": [], "Ez": []},
            "h": {"Hx": [], "Hy": [], "Hz": []},
            "e": {"Ex": [], "Ey": [], "Ez": []},
        }
        if compiled_sources:
            specs = compile_source_specs(
                compiled_sources,
                self.fields,
                dt=self.dt,
                resolution=self.resolution,
                num_steps=self.num_steps,
                t0=float(self.time[0]),
                total_steps=self.num_steps,
            )
            for spec in specs:
                grouped[spec.timing][spec.component].append(spec)
        self._compiled_step_source_specs = {
            timing: {
                component: tuple(component_specs)
                for component, component_specs in component_map.items()
            }
            for timing, component_map in grouped.items()
        }
        return self._compiled_step_source_specs

    def _imperative_sources(self):
        self._compiled_step_source_groups()
        return self._imperative_step_sources

    def _sync_full_pec_after_source_mutation(self):
        if self.is_3d and has_full_pec_3d(self.boundaries):
            if self.fields.full_pec_3d_state is None:
                self.fields.full_pec_3d_state = initialize_full_pec_3d_state(
                    self.fields
                )
            else:
                sync_full_pec_3d_from_compact(
                    self.fields,
                    self.fields.full_pec_3d_state,
                )

    def _apply_compiled_source_phase(self, timing: str):
        phase_specs = self._compiled_step_source_groups().get(timing, {})
        did_apply = False
        for component, specs in phase_specs.items():
            if specs:
                setattr(
                    self.fields,
                    component,
                    apply_compiled_source_specs(
                        getattr(self.fields, component),
                        self.current_step,
                        specs,
                    ),
                )
                did_apply = True
        if did_apply:
            self._sync_full_pec_after_source_mutation()

    def _inject_h_sources(self):
        """Inject magnetic currents (M) into H-fields after H update."""
        for device in self._imperative_sources():
            if hasattr(device, "inject_h"):
                device.inject_h(
                    self.fields,
                    self.t,
                    self.dt,
                    self.current_step,
                    self.resolution,
                    self.design,
                )
        self._sync_full_pec_after_source_mutation()

    def _inject_e_sources(self):
        """Inject electric currents (J) into E-fields after E update."""
        for device in self._imperative_sources():
            if hasattr(device, "inject_e"):
                device.inject_e(
                    self.fields,
                    self.t,
                    self.dt,
                    self.current_step,
                    self.resolution,
                    self.design,
                )
        self._sync_full_pec_after_source_mutation()

    def _inject_legacy_sources(self):
        """Inject from devices that only have inject() (no inject_h/inject_e)."""
        for device in self._imperative_sources():
            if hasattr(device, "inject") and not hasattr(device, "inject_h"):
                device.inject(
                    self.fields,
                    self.t,
                    self.dt,
                    self.current_step,
                    self.resolution,
                    self.design,
                )
        self._sync_full_pec_after_source_mutation()

    def _collect_source_terms(self):
        """Collect electric and magnetic current sources from all devices."""
        source_j = {}  # Electric currents for E-field update
        source_m = {}  # Magnetic currents for H-field update

        for device in self._imperative_sources():
            if hasattr(device, "get_source_terms"):
                j, m = device.get_source_terms(
                    self.fields,
                    self.t,
                    self.dt,
                    self.current_step,
                    self.resolution,
                    self.design,
                )
                for key, val in j.items():
                    source_j.setdefault(key, []).append(val)
                for key, val in m.items():
                    source_m.setdefault(key, []).append(val)

        return source_j, source_m

    def _create_jit_step(self):
        """Deprecated private helper kept only to fail loudly."""
        raise NotImplementedError(
            "Simulation._create_jit_step() is deprecated because it is not kept "
            "mathematically equivalent to the supported `step()` and "
            "`run_compiled()` engines."
        )

    def _create_jit_step_h(self):
        """Deprecated private helper kept only to fail loudly."""
        raise NotImplementedError(
            "Simulation._create_jit_step_h() is deprecated because it is not kept "
            "mathematically equivalent to the supported `step()` and "
            "`run_compiled()` engines."
        )

    def _create_jit_step_e(self):
        """Deprecated private helper kept only to fail loudly."""
        raise NotImplementedError(
            "Simulation._create_jit_step_e() is deprecated because it is not kept "
            "mathematically equivalent to the supported `step()` and "
            "`run_compiled()` engines."
        )

    def compile(
        self,
        num_steps=None,
        snapshot_field=None,
        snapshot_interval=None,
        sharding=None,
    ):
        """Compile the v0.3 packed-data simulation program."""
        if num_steps is None:
            num_steps = self.num_steps - self.current_step
        num_steps = int(num_steps)
        if num_steps <= 0:
            raise ValueError("num_steps must be > 0")
        snapshot_field = None if snapshot_field is None else str(snapshot_field)
        snapshot_interval = (
            0 if snapshot_field is None else max(1, int(snapshot_interval or 10))
        )

        loop_kind_env = os.getenv("BEAMZ_COMPILED_LOOP_KIND", "scan").strip().lower()
        if loop_kind_env in {"fori", "fori_loop", "fori-loop"}:
            loop_kind = "fori_loop"
        elif loop_kind_env == "scan":
            loop_kind = "scan"
        else:
            raise ValueError("Invalid BEAMZ_COMPILED_LOOP_KIND (use: scan, fori_loop).")
        source_single_slab_dense = os.getenv(
            "BEAMZ_SOURCE_SINGLE_SLAB_DENSE", ""
        ).strip().lower() in {"1", "true", "yes", "on"}

        signature = (
            num_steps,
            self.fields.permittivity.shape,
            self.is_3d,
            self.plane_2d,
            loop_kind,
            source_single_slab_dense,
            snapshot_field,
            snapshot_interval,
            sharding_cache_token(sharding),
            _compiled_cache_sequence_token(self.sources),
            _compiled_cache_sequence_token(self.monitors),
            _compiled_cache_sequence_token(self.boundaries),
        )
        cached = self._compiled_program_cache.get(signature)
        if cached is not None:
            self._compiled_program = cached
            self._compiled_program_signature = signature
            return cached

        run_cfg = SimpleNamespace(
            fields=self.fields,
            resolution=self.resolution,
            dt=self.dt,
            num_steps=num_steps,
            plane_2d=self.plane_2d,
            is_3d=self.is_3d,
            total_steps=self.num_steps,
            t0=float(self.time[0]),
            precision="float32",
            loop_kind=loop_kind,
            source_single_slab_dense=source_single_slab_dense,
            snapshot_field=snapshot_field,
            snapshot_interval=snapshot_interval,
            sharding=sharding,
        )
        program = compile_simulation(
            design=self.design,
            sources=self.sources,
            monitors=self.monitors,
            boundaries=self.boundaries,
            run_cfg=run_cfg,
        )
        self._compiled_program_cache[signature] = program
        self._compiled_program = program
        self._compiled_program_signature = signature
        return program

    def memory_estimate(
        self,
        *,
        include_compiled: bool = True,
        num_steps: int | None = None,
        sharding=None,
    ) -> dict:
        """Return a JSON-friendly estimate of simulation and compiled memory."""
        entries: list[dict] = []
        field_arrays = (
            "Ex",
            "Ey",
            "Ez",
            "Hx",
            "Hy",
            "Hz",
            "permittivity",
            "conductivity",
            "permeability",
            "total_conductivity",
            "eps_x",
            "eps_y",
            "eps_z",
            "sig_x",
            "sig_y",
            "sig_z",
            "sigma_m_hx",
            "sigma_m_hy",
            "sigma_m_hz",
            "tm_ez_mask",
            "tm_hx_mask",
            "tm_hy_mask",
            "ex_metal_mask",
            "ey_metal_mask",
            "ez_metal_mask",
            "hx_metal_mask",
            "hy_metal_mask",
            "hz_metal_mask",
        )
        for name in field_arrays:
            if hasattr(self.fields, name):
                category = (
                    "yee_fields"
                    if name in {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
                    else (
                        "material_center_grids"
                        if name
                        in {
                            "permittivity",
                            "conductivity",
                            "permeability",
                            "total_conductivity",
                        }
                        else (
                            "component_material_grids"
                            if name.startswith(("eps_", "sig_", "sigma_m_"))
                            else "field_masks"
                        )
                    )
                )
                _add_array_entries(
                    entries,
                    f"fields.{name}",
                    getattr(self.fields, name),
                    category=category,
                )

        pml_data = getattr(self, "pml_data", None)
        if isinstance(pml_data, Mapping):
            for key, value in pml_data.items():
                _add_array_entries(
                    entries,
                    f"pml_data.{key}",
                    value,
                    category="pml_data",
                )

        report = _memory_report(entries)
        report["grid_shape_zyx"] = [
            int(v) for v in getattr(self.fields.permittivity, "shape", ())
        ]
        report["is_3d"] = bool(self.is_3d)
        if include_compiled:
            program = self.compile(
                num_steps=num_steps if num_steps is not None else None,
                sharding=sharding,
            )
            compiled_report = program.memory_estimate(include_runtime=True)
            report["compiled"] = compiled_report
            report["total_with_compiled_bytes"] = int(report["total_bytes"]) + int(
                compiled_report["total_bytes"]
            )
            report["total_with_compiled_gib"] = (
                report["total_with_compiled_bytes"] / 1024**3
            )
        return report

    def _compiled_runtime_inputs(self, program):
        if (not self.is_3d) and self.plane_2d == "xy" and program.use_physical_tm_xy:
            tm_ez = self.fields.Ez
            tm_hx = self.fields.Hx
            tm_hy = self.fields.Hy
        else:
            tm_ez = jnp.zeros((0, 0), dtype=self.fields.Ez.dtype)
            tm_hx = jnp.zeros((0, 0), dtype=self.fields.Hx.dtype)
            tm_hy = jnp.zeros((0, 0), dtype=self.fields.Hy.dtype)
        if self.is_3d and program.full_pec_3d:
            if self.fields.full_pec_3d_state is None:
                self.fields.full_pec_3d_state = initialize_full_pec_3d_state(
                    self.fields
                )
            fp_state = self.fields.full_pec_3d_state
            ex, ey, ez = fp_state.Ex, fp_state.Ey, fp_state.Ez
            hx, hy, hz = fp_state.Hx, fp_state.Hy, fp_state.Hz
        else:
            ex, ey, ez = self.fields.Ex, self.fields.Ey, self.fields.Ez
            hx, hy, hz = self.fields.Hx, self.fields.Hy, self.fields.Hz

        fp_ex = jnp.zeros((0, 0, 0), dtype=self.fields.Ex.dtype)
        fp_ey = jnp.zeros((0, 0, 0), dtype=self.fields.Ey.dtype)
        fp_ez = jnp.zeros((0, 0, 0), dtype=self.fields.Ez.dtype)
        fp_hx = jnp.zeros((0, 0, 0), dtype=self.fields.Hx.dtype)
        fp_hy = jnp.zeros((0, 0, 0), dtype=self.fields.Hy.dtype)
        fp_hz = jnp.zeros((0, 0, 0), dtype=self.fields.Hz.dtype)

        engine_state = EngineState(
            ex=ex,
            ey=ey,
            ez=ez,
            hx=hx,
            hy=hy,
            hz=hz,
            tm_ez=tm_ez,
            tm_hx=tm_hx,
            tm_hy=tm_hy,
            fp_ex=fp_ex,
            fp_ey=fp_ey,
            fp_ez=fp_ez,
            fp_hx=fp_hx,
            fp_hy=fp_hy,
            fp_hz=fp_hz,
            cpml_psi_h_terms=(
                jnp.zeros_like(program.cpml_sigma_h_terms)
                if program.use_cpml_tm_xy
                else jnp.zeros((2, 0, 0), dtype=self.fields.Hx.dtype)
            ),
            cpml_psi_e_terms=(
                jnp.zeros_like(program.cpml_sigma_e_terms)
                if program.use_cpml_tm_xy
                else jnp.zeros((2, 0, 0), dtype=self.fields.Ez.dtype)
            ),
            cpml3d_psi_h_terms=(
                tuple(
                    jnp.zeros(shape, dtype=self.fields.Hx.dtype)
                    for shape in program.cpml3d_h_psi_shapes
                )
                if program.use_cpml_3d
                else tuple(
                    jnp.zeros((0, 0, 0), dtype=self.fields.Hx.dtype) for _ in range(6)
                )
            ),
            cpml3d_psi_e_terms=(
                tuple(
                    jnp.zeros(shape, dtype=self.fields.Ex.dtype)
                    for shape in program.cpml3d_e_psi_shapes
                )
                if program.use_cpml_3d
                else tuple(
                    jnp.zeros((0, 0, 0), dtype=self.fields.Ez.dtype) for _ in range(6)
                )
            ),
            t=jnp.asarray(self.t, dtype=jnp.float32),
            current_step=jnp.asarray(self.current_step, dtype=jnp.int32),
        )

        dft_dtype = monitor_dft_accumulator_dtype()
        if program.monitor_specs:
            records_horizon = max(1, int(self.num_steps - self.current_step))
            max_records = max(
                1, monitor_state_size(program.monitor_specs, records_horizon)
            )
            max_freq = monitor_frequency_size(program.monitor_specs)
            max_points = monitor_dft_point_size(program.monitor_specs)
            monitor_state = MonitorState(
                powers=jnp.zeros(
                    (len(program.monitor_specs), max_records), dtype=jnp.float32
                ),
                timestamps=jnp.zeros(
                    (len(program.monitor_specs), max_records), dtype=jnp.float32
                ),
                counts=jnp.zeros((len(program.monitor_specs),), dtype=jnp.int32),
                freq_flux_re=jnp.zeros(
                    (len(program.monitor_specs), max_freq), dtype=jnp.float32
                ),
                freq_flux_im=jnp.zeros(
                    (len(program.monitor_specs), max_freq), dtype=jnp.float32
                ),
                freq_phase_re=jnp.ones(
                    (len(program.monitor_specs), max_freq), dtype=jnp.float32
                ),
                freq_phase_im=jnp.zeros(
                    (len(program.monitor_specs), max_freq), dtype=jnp.float32
                ),
                dft_vec_re=jnp.zeros(
                    (len(program.monitor_specs), 6, max_freq, max_points),
                    dtype=dft_dtype,
                ),
                dft_vec_im=jnp.zeros(
                    (len(program.monitor_specs), 6, max_freq, max_points),
                    dtype=dft_dtype,
                ),
                dft_weight_sum=jnp.zeros(
                    (len(program.monitor_specs), max_freq), dtype=dft_dtype
                ),
            )
        else:
            monitor_state = MonitorState(
                powers=jnp.zeros((0, 0), dtype=jnp.float32),
                timestamps=jnp.zeros((0, 0), dtype=jnp.float32),
                counts=jnp.zeros((0,), dtype=jnp.int32),
                freq_flux_re=jnp.zeros((0, 0), dtype=jnp.float32),
                freq_flux_im=jnp.zeros((0, 0), dtype=jnp.float32),
                freq_phase_re=jnp.zeros((0, 0), dtype=jnp.float32),
                freq_phase_im=jnp.zeros((0, 0), dtype=jnp.float32),
                dft_vec_re=jnp.zeros((0, 0, 0, 0), dtype=dft_dtype),
                dft_vec_im=jnp.zeros((0, 0, 0, 0), dtype=dft_dtype),
                dft_weight_sum=jnp.zeros((0, 0), dtype=dft_dtype),
            )
        return engine_state, monitor_state

    def compiled_xla_memory_analysis(
        self, *, num_steps: int | None = None, sharding=None
    ) -> dict:
        """Compile the packed loop and return JAX/XLA memory analysis if available."""
        program = self.compile(num_steps=num_steps, sharding=sharding)
        engine_state, monitor_state = self._compiled_runtime_inputs(program)
        engine_state = program.prepare_engine_state(engine_state)
        monitor_state = program._place_pytree(monitor_state, shard_arrays=False)
        coeffs = program._place_update_coefficients(program._update_coefficients())
        if program._compiled_scan is None:
            program._build_scan()
        snapshot_state = program._empty_snapshot_state()
        if snapshot_state is not None:
            snapshot_state = program._place_pytree(snapshot_state, shard_arrays=False)
            args = (engine_state, monitor_state, coeffs, snapshot_state)
        else:
            args = (engine_state, monitor_state, coeffs)
        compiled = program._compiled_scan.lower(*args).compile()
        analysis = getattr(compiled, "memory_analysis", lambda: None)()
        if analysis is None:
            return {"available": False}
        out = {"available": True}
        for name in dir(analysis):
            if name.startswith("_"):
                continue
            value = getattr(analysis, name)
            if isinstance(value, (int, float, str, bool)) or value is None:
                out[name] = value
        return out

    def run_compiled(
        self,
        num_steps=None,
        record_interval=None,
        record_fields=None,
        progress=True,
        snapshot_field=None,
        snapshot_interval=10,
        snapshot_callback=None,
        store_snapshots=True,
        sharding=None,
    ):
        """Run simulation using the v0.3 single-program compiled scan engine.

        Notes:
        - Source/monitor callbacks are compiled as packed specs.
        - Monitor results are accumulated in-loop and written back to Monitor objects.
        - Field history recording is optional and chunked via repeated compiled runs.
        - Snapshot extraction stays inside the compiled loop and is materialized
          on the host after each compiled chunk completes.
        """
        if self.thermal is not None and getattr(self.thermal, "enabled", True):
            raise NotImplementedError(
                "run_compiled currently does not support thermal coupling."
            )

        if num_steps is None:
            num_steps = self.num_steps - self.current_step
        num_steps = int(num_steps)
        if num_steps <= 0:
            return None

        if record_fields is None:
            record_fields = ["Ez"]
        snapshot_field = None if snapshot_field is None else str(snapshot_field)
        snapshot_interval = (
            0 if snapshot_field is None else max(1, int(snapshot_interval or 10))
        )
        snapshots = []
        snapshot_layout = None
        if snapshot_field is not None:
            from beamz.simulation.snapshots import validate_snapshot_field

            validate_snapshot_field(self, snapshot_field)

        record_every = int(record_interval) if record_interval else None
        if record_every is not None and record_every <= 0:
            raise ValueError("record_interval must be a positive integer")

        field_history = {name: [] for name in record_fields} if record_every else None
        field_times = [] if record_every else None
        field_steps = [] if record_every else None
        if self.current_step == 0:
            self._compiled_monitor_state = None

        # Run in one chunk for max TCUPS by default. For field snapshots, run in equal chunks.
        chunk_size = record_every if record_every else num_steps
        steps_remaining = num_steps
        steps_done = 0
        monitor_state: MonitorState | None = None

        while steps_remaining > 0:
            this_chunk = min(chunk_size, steps_remaining)
            program = self.compile(
                num_steps=this_chunk,
                snapshot_field=snapshot_field,
                snapshot_interval=snapshot_interval,
                sharding=sharding,
            )

            if progress and steps_done == 0 and program.compile_count == 0:
                print(
                    "● JIT compiling v0.3 packed FDTD program...",
                    end=" ",
                    flush=True,
                )

            compiled_dtype = (
                jnp.float32
                if str(program.config.precision).lower() == "float32"
                else jnp.float64
            )
            ex = jnp.asarray(self.fields.Ex, dtype=compiled_dtype)
            ey = jnp.asarray(self.fields.Ey, dtype=compiled_dtype)
            ez = jnp.asarray(self.fields.Ez, dtype=compiled_dtype)
            hx = jnp.asarray(self.fields.Hx, dtype=compiled_dtype)
            hy = jnp.asarray(self.fields.Hy, dtype=compiled_dtype)
            hz = jnp.asarray(self.fields.Hz, dtype=compiled_dtype)

            if (
                (not self.is_3d)
                and self.plane_2d == "xy"
                and program.use_physical_tm_xy
            ):
                tm_ez = ez
                tm_hx = hx
                tm_hy = hy
            else:
                tm_ez = jnp.zeros((0, 0), dtype=compiled_dtype)
                tm_hx = jnp.zeros((0, 0), dtype=compiled_dtype)
                tm_hy = jnp.zeros((0, 0), dtype=compiled_dtype)
            if self.is_3d and program.full_pec_3d:
                if self.fields.full_pec_3d_state is None:
                    self.fields.full_pec_3d_state = initialize_full_pec_3d_state(
                        self.fields
                    )
                fp_state = self.fields.full_pec_3d_state
                ex = jnp.asarray(fp_state.Ex, dtype=compiled_dtype)
                ey = jnp.asarray(fp_state.Ey, dtype=compiled_dtype)
                ez = jnp.asarray(fp_state.Ez, dtype=compiled_dtype)
                hx = jnp.asarray(fp_state.Hx, dtype=compiled_dtype)
                hy = jnp.asarray(fp_state.Hy, dtype=compiled_dtype)
                hz = jnp.asarray(fp_state.Hz, dtype=compiled_dtype)

            fp_ex = jnp.zeros((0, 0, 0), dtype=compiled_dtype)
            fp_ey = jnp.zeros((0, 0, 0), dtype=compiled_dtype)
            fp_ez = jnp.zeros((0, 0, 0), dtype=compiled_dtype)
            fp_hx = jnp.zeros((0, 0, 0), dtype=compiled_dtype)
            fp_hy = jnp.zeros((0, 0, 0), dtype=compiled_dtype)
            fp_hz = jnp.zeros((0, 0, 0), dtype=compiled_dtype)

            engine_state = EngineState(
                ex=ex,
                ey=ey,
                ez=ez,
                hx=hx,
                hy=hy,
                hz=hz,
                tm_ez=tm_ez,
                tm_hx=tm_hx,
                tm_hy=tm_hy,
                fp_ex=fp_ex,
                fp_ey=fp_ey,
                fp_ez=fp_ez,
                fp_hx=fp_hx,
                fp_hy=fp_hy,
                fp_hz=fp_hz,
                cpml_psi_h_terms=(
                    jnp.zeros_like(program.cpml_sigma_h_terms)
                    if program.use_cpml_tm_xy
                    else jnp.zeros((2, 0, 0), dtype=compiled_dtype)
                ),
                cpml_psi_e_terms=(
                    jnp.zeros_like(program.cpml_sigma_e_terms)
                    if program.use_cpml_tm_xy
                    else jnp.zeros((2, 0, 0), dtype=compiled_dtype)
                ),
                cpml3d_psi_h_terms=(
                    tuple(
                        jnp.zeros(shape, dtype=compiled_dtype)
                        for shape in program.cpml3d_h_psi_shapes
                    )
                    if program.use_cpml_3d
                    else tuple(
                        jnp.zeros((0, 0, 0), dtype=compiled_dtype) for _ in range(6)
                    )
                ),
                cpml3d_psi_e_terms=(
                    tuple(
                        jnp.zeros(shape, dtype=compiled_dtype)
                        for shape in program.cpml3d_e_psi_shapes
                    )
                    if program.use_cpml_3d
                    else tuple(
                        jnp.zeros((0, 0, 0), dtype=compiled_dtype) for _ in range(6)
                    )
                ),
                t=jnp.asarray(self.t, dtype=jnp.float32),
                current_step=jnp.asarray(self.current_step, dtype=jnp.int32),
            )

            if monitor_state is None:
                if (
                    self._compiled_monitor_state is not None
                    and program.monitor_specs
                    and int(np.asarray(self._compiled_monitor_state.counts.shape[0]))
                    == len(program.monitor_specs)
                ):
                    monitor_state = self._compiled_monitor_state
                elif program.monitor_specs:
                    records_horizon = max(1, int(self.num_steps - self.current_step))
                    max_records = max(
                        1, monitor_state_size(program.monitor_specs, records_horizon)
                    )
                    max_freq = monitor_frequency_size(program.monitor_specs)
                    max_points = monitor_dft_point_size(program.monitor_specs)
                    dft_dtype = monitor_dft_accumulator_dtype()
                    monitor_state = MonitorState(
                        powers=jnp.zeros(
                            (len(program.monitor_specs), max_records), dtype=jnp.float32
                        ),
                        timestamps=jnp.zeros(
                            (len(program.monitor_specs), max_records), dtype=jnp.float32
                        ),
                        counts=jnp.zeros(
                            (len(program.monitor_specs),), dtype=jnp.int32
                        ),
                        freq_flux_re=jnp.zeros(
                            (len(program.monitor_specs), max_freq), dtype=jnp.float32
                        ),
                        freq_flux_im=jnp.zeros(
                            (len(program.monitor_specs), max_freq), dtype=jnp.float32
                        ),
                        freq_phase_re=jnp.ones(
                            (len(program.monitor_specs), max_freq), dtype=jnp.float32
                        ),
                        freq_phase_im=jnp.zeros(
                            (len(program.monitor_specs), max_freq), dtype=jnp.float32
                        ),
                        dft_vec_re=jnp.zeros(
                            (len(program.monitor_specs), 6, max_freq, max_points),
                            dtype=dft_dtype,
                        ),
                        dft_vec_im=jnp.zeros(
                            (len(program.monitor_specs), 6, max_freq, max_points),
                            dtype=dft_dtype,
                        ),
                        dft_weight_sum=jnp.zeros(
                            (len(program.monitor_specs), max_freq), dtype=dft_dtype
                        ),
                    )
                else:
                    dft_dtype = monitor_dft_accumulator_dtype()
                    monitor_state = MonitorState(
                        powers=jnp.zeros((0, 0), dtype=jnp.float32),
                        timestamps=jnp.zeros((0, 0), dtype=jnp.float32),
                        counts=jnp.zeros((0,), dtype=jnp.int32),
                        freq_flux_re=jnp.zeros((0, 0), dtype=jnp.float32),
                        freq_flux_im=jnp.zeros((0, 0), dtype=jnp.float32),
                        freq_phase_re=jnp.zeros((0, 0), dtype=jnp.float32),
                        freq_phase_im=jnp.zeros((0, 0), dtype=jnp.float32),
                        dft_vec_re=jnp.zeros((0, 0, 0, 0), dtype=dft_dtype),
                        dft_vec_im=jnp.zeros((0, 0, 0, 0), dtype=dft_dtype),
                        dft_weight_sum=jnp.zeros((0, 0), dtype=dft_dtype),
                    )
            self._compiled_monitor_state = monitor_state

            engine_state, monitor_state, _, snapshot_data = program.run(
                engine_state=engine_state,
                monitor_state=monitor_state,
            )
            engine_state.ez.block_until_ready()
            self._compiled_monitor_state = monitor_state
            storage_engine_state = engine_state
            engine_state = program.crop_engine_state(storage_engine_state)

            if progress and steps_done == 0:
                print("done!")

            self.fields.Ex = engine_state.ex
            self.fields.Ey = engine_state.ey
            self.fields.Ez = engine_state.ez
            self.fields.Hx = engine_state.hx
            self.fields.Hy = engine_state.hy
            self.fields.Hz = engine_state.hz
            if self.is_3d and program.full_pec_3d:
                if self.fields.full_pec_3d_state is None:
                    self.fields.full_pec_3d_state = initialize_full_pec_3d_state(
                        self.fields
                    )
                self.fields.full_pec_3d_state.Ex = program._crop_active_component(
                    "Ex", storage_engine_state.ex
                )
                self.fields.full_pec_3d_state.Ey = program._crop_active_component(
                    "Ey", storage_engine_state.ey
                )
                self.fields.full_pec_3d_state.Ez = program._crop_active_component(
                    "Ez", storage_engine_state.ez
                )
                self.fields.full_pec_3d_state.Hx = program._crop_active_component(
                    "Hx", storage_engine_state.hx
                )
                self.fields.full_pec_3d_state.Hy = program._crop_active_component(
                    "Hy", storage_engine_state.hy
                )
                self.fields.full_pec_3d_state.Hz = program._crop_active_component(
                    "Hz", storage_engine_state.hz
                )
            if (
                (not self.is_3d)
                and self.plane_2d == "xy"
                and program.use_physical_tm_xy
            ):
                self.fields.Ez = engine_state.tm_ez
                self.fields.Hx = engine_state.tm_hx
                self.fields.Hy = engine_state.tm_hy
            self.t = float(np.asarray(engine_state.t))
            self.current_step = int(np.asarray(engine_state.current_step))

            if field_history is not None and (self.current_step % record_every == 0):
                for name in record_fields:
                    if hasattr(self.fields, name):
                        field_history[name].append(np.array(getattr(self.fields, name)))
                field_times.append(float(self.t))
                field_steps.append(int(self.current_step))
            if snapshot_data is not None:
                from beamz.simulation.snapshots import collect_compiled_snapshots

                new_snapshots, snapshot_layout = collect_compiled_snapshots(
                    self,
                    field_name=snapshot_field,
                    snapshot_data=snapshot_data,
                    layout=snapshot_layout,
                )
                if snapshot_callback is not None:
                    for snapshot in new_snapshots:
                        snapshot_callback(snapshot)
                if store_snapshots:
                    snapshots.extend(new_snapshots)

            steps_done += this_chunk
            steps_remaining -= this_chunk

            if progress and num_steps > 0:
                _print_inline_progress(steps_done, num_steps)

        if progress:
            _finish_inline_progress()

        if monitor_state is not None:
            program.apply_monitor_state(monitor_state)

        fields_result = None
        if field_history is not None:
            fields_result = {
                k: np.stack(v) if len(v) > 0 else np.zeros((0,))
                for k, v in field_history.items()
            }
        return SimulationResults.from_run(
            self,
            fields=fields_result,
            field_times=field_times,
            field_steps=field_steps,
            monitors=self.monitors,
            snapshots=snapshots,
        )

    def run_compiled_until_decay(
        self,
        monitors,
        *,
        min_time_s=0.0,
        chunk_steps=None,
        lookback_records=12,
        decay_ratio=1e-3,
        progress=True,
    ):
        """Run compiled chunks until monitor power decays after a minimum time."""
        total_steps = int(self.num_steps - self.current_step)
        if total_steps <= 0:
            return 0
        dt = float(self.dt)
        chunk_steps = (
            max(64, min(512, int(np.ceil(total_steps / 24.0))))
            if chunk_steps is None
            else max(1, int(chunk_steps))
        )
        lookback_records = max(2, int(lookback_records))
        min_steps = int(np.ceil(max(0.0, float(min_time_s)) / max(dt, 1e-30)))
        steps_done = 0
        peak = 0.0

        while steps_done < total_steps:
            this_chunk = min(chunk_steps, total_steps - steps_done)
            self.run_compiled(num_steps=this_chunk, progress=False)
            steps_done += this_chunk

            histories = [
                np.abs(np.asarray(mon.power_history, dtype=np.float64))
                for mon in monitors
                if len(mon.power_history)
            ]
            tail = np.inf
            if histories:
                peak = max(peak, max(float(np.max(hist)) for hist in histories))
                tail = max(
                    float(np.max(hist[-lookback_records:])) for hist in histories
                )

            if progress:
                _print_inline_progress(steps_done, total_steps)

            if (
                steps_done >= min_steps
                and peak > 0.0
                and np.isfinite(tail)
                and tail <= float(decay_ratio) * peak
            ):
                break

        if progress:
            _finish_inline_progress()
        return steps_done

    def run_fast(
        self, num_steps=None, record_interval=None, record_fields=None, progress=True
    ):
        """Backward-compatible alias to `run_compiled` in v0.3."""
        return self.run_compiled(
            num_steps=num_steps,
            record_interval=record_interval,
            record_fields=record_fields,
            progress=progress,
        )

    def run_jit_scan(self, num_steps=None, progress=True):
        """Backward-compatible alias to `run_compiled` in v0.3."""
        return self.run_compiled(
            num_steps=num_steps,
            record_interval=None,
            record_fields=None,
            progress=progress,
        )

    @staticmethod
    def _safe_ratio(num, den, eps=1e-18):
        out = np.zeros_like(num, dtype=np.complex128)
        valid = np.abs(den) > eps
        out[valid] = num[valid] / den[valid]
        return out

    @staticmethod
    def _select_wave_component(
        wave_data,
        selector="minus",
        *,
        use_reference=False,
    ):
        sel = str(selector).lower()
        if sel not in {"plus", "minus", "auto"}:
            raise ValueError(
                f"Unsupported wave selector '{selector}'. "
                "Use one of {'plus', 'minus', 'auto'}."
            )

        if use_reference:
            plus = np.asarray(
                wave_data.get(
                    "a_incident_plus",
                    wave_data.get("a_incident", wave_data.get("a_plus")),
                ),
                dtype=np.complex128,
            )
            minus = np.asarray(
                wave_data.get("a_incident_minus", wave_data.get("a_minus")),
                dtype=np.complex128,
            )
        else:
            plus = np.asarray(wave_data.get("a_plus"), dtype=np.complex128)
            minus = np.asarray(wave_data.get("a_minus"), dtype=np.complex128)

        if sel == "plus":
            return plus
        if sel == "minus":
            return minus
        return np.where(np.abs(plus) >= np.abs(minus), plus, minus)

    @staticmethod
    def _opposite_wave_selector(selector):
        sel = str(selector).lower()
        if sel == "plus":
            return "minus"
        if sel == "minus":
            return "plus"
        raise ValueError(
            f"Unsupported wave selector '{selector}'. Use one of {{'plus', 'minus'}}."
        )

    @classmethod
    def _resolve_port_wave_selectors(
        cls,
        spec,
        wave_data,
        *,
        use_reference=False,
    ):
        incident = str(spec.incident_wave).lower()
        scattered = str(spec.scattered_wave).lower()
        if incident != "auto" and scattered != "auto":
            return incident, scattered
        if incident != "auto" and scattered == "auto":
            return incident, cls._opposite_wave_selector(incident)
        if scattered != "auto" and incident == "auto":
            return cls._opposite_wave_selector(scattered), scattered

        if use_reference:
            plus = np.asarray(
                wave_data.get(
                    "a_incident_plus",
                    wave_data.get("a_incident", wave_data.get("a_plus")),
                ),
                dtype=np.complex128,
            )
            minus = np.asarray(
                wave_data.get("a_incident_minus", wave_data.get("a_minus")),
                dtype=np.complex128,
            )
        else:
            plus = np.asarray(wave_data.get("a_plus"), dtype=np.complex128)
            minus = np.asarray(wave_data.get("a_minus"), dtype=np.complex128)

        plus_level = float(np.max(np.abs(plus))) if plus.size else 0.0
        minus_level = float(np.max(np.abs(minus))) if minus.size else 0.0
        dominant = "plus" if plus_level >= minus_level else "minus"
        return dominant, cls._opposite_wave_selector(dominant)

    @staticmethod
    def _format_s_matrix_output(s_matrix, as_sax):
        """Return S-parameter mapping without requiring optional external packages."""
        if as_sax:
            # Keep tuple-key mapping compatible with existing callers while avoiding
            # a hard dependency on the external `sax` package.
            return dict(s_matrix)
        return s_matrix

    @staticmethod
    def _port_name(port):
        if isinstance(port, str):
            return port
        if isinstance(port, (PortSpec, Port)):
            return str(port.name)
        if isinstance(port, Mapping):
            return str(port["name"])
        if hasattr(port, "to_port"):
            return str(port.to_port().name)
        name = getattr(port, "name", None)
        if name:
            return str(name)
        raise ValueError(f"Cannot infer port name from {port!r}.")

    @classmethod
    def _normalize_output_port_names(cls, output_ports, port_map):
        if output_ports is None:
            names = list(port_map.keys())
        else:
            names = [cls._port_name(item) for item in output_ports]
        missing = [name for name in names if name not in port_map]
        if missing:
            raise ValueError(f"output_ports contains unknown ports: {missing}")
        return names

    @staticmethod
    def _normalize_portspecs(ports):
        if isinstance(ports, dict):
            values = list(ports.values())
        else:
            values = list(ports)
        if not values:
            raise ValueError("ports must contain at least one PortSpec.")

        normalized = {}
        for item in values:
            if isinstance(item, PortSpec):
                spec = item
            elif isinstance(item, Port):
                spec = PortSpec(**item.to_portspec_dict())
            elif hasattr(item, "to_portspec_dict"):
                spec = PortSpec(**item.to_portspec_dict())
            else:
                item = dict(item)
                if "monitor" in item or "projection_direction" in item:
                    spec = PortSpec(**Port.from_mapping(item).to_portspec_dict())
                else:
                    spec = PortSpec(
                        name=item["name"],
                        monitor_name=item["monitor_name"],
                        direction=item["direction"],
                        polarization=item["polarization"],
                        mode_index=int(item.get("mode_index", 0)),
                        reference_monitor=item.get("reference_monitor"),
                        incident_wave=str(item.get("incident_wave", "plus")).lower(),
                        scattered_wave=str(item.get("scattered_wave", "minus")).lower(),
                    )
            if spec.direction not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
                raise ValueError(f"Unsupported port direction '{spec.direction}'.")
            pol = str(spec.polarization).lower()
            if pol not in {"tm", "te"}:
                raise ValueError(f"Unsupported polarization '{spec.polarization}'.")
            inc_wave = str(spec.incident_wave).lower()
            scat_wave = str(spec.scattered_wave).lower()
            if inc_wave not in {"plus", "minus", "auto"}:
                raise ValueError(
                    f"Unsupported incident_wave '{spec.incident_wave}' for port '{spec.name}'."
                )
            if scat_wave not in {"plus", "minus", "auto"}:
                raise ValueError(
                    f"Unsupported scattered_wave '{spec.scattered_wave}' for port '{spec.name}'."
                )
            normalized[spec.name] = PortSpec(
                name=spec.name,
                monitor_name=spec.monitor_name,
                direction=spec.direction,
                polarization=pol,
                mode_index=int(spec.mode_index),
                reference_monitor=spec.reference_monitor,
                incident_wave=inc_wave,
                scattered_wave=scat_wave,
            )
        return normalized

    def _named_monitors(self):
        return {
            device.name: device
            for device in self.monitors
            if getattr(device, "name", None)
        }

    def _sample_monitor_component_spectrum(
        self,
        monitor,
        component,
        frequencies=None,
        window="hann",
    ):
        if component not in monitor.fields:
            raise ValueError(
                f"Monitor '{monitor.name}' has no field '{component}'. "
                f"Available: {sorted(monitor.fields.keys())}"
            )
        raw = monitor.fields[component]
        if raw is None or len(raw) == 0:
            raise ValueError(
                f"Monitor '{monitor.name}' has no recorded '{component}' data."
            )
        values = np.asarray(raw)
        if values.ndim == 1:
            values = values[:, None]
        elif values.ndim > 2:
            values = values.reshape(values.shape[0], -1)

        t = np.asarray(monitor.fields.get("t", []), dtype=float)
        n = min(values.shape[0], t.size)
        if n < 2:
            raise ValueError(
                f"Monitor '{monitor.name}' has insufficient samples for FFT extraction."
            )
        values = values[:n]
        t = t[:n]
        values = values - np.mean(values, axis=0, keepdims=True)

        win_key = str(window).lower() if window is not None else "none"
        if win_key in {"hann", "hanning"}:
            w = np.hanning(n)
        elif win_key in {"none", "rect", "rectangular"}:
            w = np.ones(n, dtype=float)
        else:
            raise ValueError(f"Unsupported window '{window}'.")
        values = values * w[:, None]

        dt = float(np.mean(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"Invalid dt inferred from monitor '{monitor.name}'.")
        if np.iscomplexobj(values):
            freq_bins = np.fft.fftfreq(n, d=dt)
            spec_bins = np.fft.ifft(values, axis=0) * n
            keep = freq_bins >= 0
            freq_bins = freq_bins[keep]
            spec_bins = spec_bins[keep]
        else:
            freq_bins = np.fft.rfftfreq(n, d=dt)
            spec_bins = np.conjugate(np.fft.rfft(values, axis=0))

        if frequencies is None:
            phase = self._monitor_projection_phase(component, freq_bins, dt)
            out = spec_bins * phase[:, None]
            return freq_bins, out

        requested = np.atleast_1d(np.asarray(frequencies, dtype=float))
        sampled = np.empty((len(requested), spec_bins.shape[1]), dtype=np.complex128)
        for col in range(spec_bins.shape[1]):
            real_part = np.interp(
                requested, freq_bins, np.real(spec_bins[:, col]), left=0.0, right=0.0
            )
            imag_part = np.interp(
                requested, freq_bins, np.imag(spec_bins[:, col]), left=0.0, right=0.0
            )
            sampled[:, col] = real_part + 1j * imag_part
        phase = self._monitor_projection_phase(component, requested, dt)
        sampled = sampled * phase[:, None]
        return requested, sampled

    @staticmethod
    def _resample_complex_matrix(freq_src, values_src, freq_dst):
        """Resample a DFT component matrix to requested frequencies.

        Canonical input/output shape is `(nfreq, npoints)`. Any trailing spatial
        dimensions are flattened into `npoints` so monitor consumers never need
        to reason about monitor geometry rank.
        """
        freq_src = np.atleast_1d(np.asarray(freq_src, dtype=float))
        src = np.asarray(values_src, dtype=np.complex128)
        if src.ndim == 0:
            src = src.reshape(1, 1)
        elif src.ndim == 1:
            if src.shape[0] == freq_src.size:
                src = src[:, None]
            elif freq_src.size == 1:
                src = src.reshape(1, -1)
            else:
                raise ValueError(
                    "Cannot infer DFT frequency axis for 1D component array: "
                    f"len(values)={src.shape[0]}, nfreq={freq_src.size}"
                )
        else:
            if src.shape[0] != freq_src.size:
                raise ValueError(
                    "DFT component matrix must use frequency on axis 0: "
                    f"got shape={src.shape}, nfreq={freq_src.size}"
                )
            src = src.reshape(src.shape[0], -1)

        if np.allclose(freq_src, freq_dst, rtol=1e-9, atol=0.0) and src.shape[0] == len(
            freq_dst
        ):
            return src
        out = np.empty((len(freq_dst), src.shape[1]), dtype=np.complex128)
        for col in range(src.shape[1]):
            re = np.interp(
                freq_dst, freq_src, np.real(src[:, col]), left=0.0, right=0.0
            )
            im = np.interp(
                freq_dst, freq_src, np.imag(src[:, col]), left=0.0, right=0.0
            )
            out[:, col] = re + 1j * im
        return out

    @staticmethod
    def _monitor_projection_phase(component, frequencies, dt):
        """Phase-align raw monitor phasors to the E-field sample time.

        BeamZ's DFT uses the phasor convention

            f(t) = Re{F exp(-i omega t)}
            F ~= 2 sum_t f(t) exp(+i omega t) / sum_t 1

        Monitors are sampled after the E update at timestamp T = t + dt. At
        that instant E is stored at T, while the leapfrog H fields are stored at
        T - dt/2. If a component is actually sampled at T + tau but accumulated
        with exp(+i omega T), the accumulator returns F exp(-i omega tau). To
        recover the common-time modal phasor F, multiply by exp(+i omega tau).
        Therefore E has tau = 0 and H has tau = -dt/2.
        """
        freq_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
        comp = str(component)
        if comp.startswith("H"):
            return np.exp(-1j * np.pi * freq_arr * float(dt))
        return np.ones_like(freq_arr, dtype=np.complex128)

    @staticmethod
    def _modal_projection_spatial_phase(component, frequencies, plane_delay_s):
        """Phase-align E components from their Yee plane to the H-referenced mode.

        Mode profiles are gauged to the dominant H component, matching the
        ModeSource launch convention. After the temporal Yee correction, E
        samples still need the spatial propagation phase from the E Yee plane
        to that H reference plane.
        """
        freq_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
        comp = str(component)
        delay = float(plane_delay_s)
        if comp.startswith("E") and delay != 0.0:
            return np.exp(1j * 2.0 * np.pi * freq_arr * delay)
        return np.ones_like(freq_arr, dtype=np.complex128)

    def _modal_projection_plane_delay_s(self, spec, frequency, mode_neff):
        """Return the E-to-H modal-plane delay used by S-parameter projection."""
        if getattr(self, "is_3d", False):
            # 3D monitors interpolate every recorded component onto the same
            # physical analysis plane. There is no remaining normal-direction
            # Yee half-cell offset to compensate during modal extraction.
            return 0.0
        freq = float(frequency)
        neff = float(np.real(np.asarray(mode_neff)))
        if (not np.isfinite(freq)) or freq <= 0.0:
            return 0.0
        if (not np.isfinite(neff)) or neff <= 0.0:
            return 0.0
        d_axis = float(getattr(self, "resolution", 0.0) or 0.0)
        if (not np.isfinite(d_axis)) or d_axis <= 0.0:
            return 0.0

        direction_sign = +1.0 if str(spec.direction).startswith("+") else -1.0
        delta_s = direction_sign * 0.5 * d_axis
        if (
            getattr(self, "is_3d", False)
            and hasattr(self, "dt")
            and self.dt is not None
        ):
            omega = 2.0 * np.pi * freq
            k_num = _solve_numeric_k_axis(omega, float(self.dt), d_axis, neff)
            return _numeric_phase_delay(omega, k_num, delta_s)
        return float(delta_s * neff / LIGHT_SPEED)

    def _apply_modal_projection_spatial_phase(
        self, component, values, frequency, projection
    ):
        phase = self._modal_projection_spatial_phase(
            component,
            np.asarray([float(frequency)], dtype=float),
            float(projection.get("modal_plane_delay_s", 0.0)),
        )[0]
        return np.asarray(values, dtype=np.complex128) * phase

    def _sample_monitor_component_dft(self, monitor, component, frequencies):
        if not hasattr(monitor, "get_dft_component"):
            raise ValueError(
                f"Monitor '{monitor.name}' does not support DFT accumulation."
            )
        freq_src = np.asarray(monitor.get_dft_frequencies(), dtype=float)
        if freq_src.size == 0:
            raise ValueError(
                f"Monitor '{monitor.name}' has no configured DFT frequencies."
            )
        values_src = np.asarray(
            monitor.get_dft_component(component), dtype=np.complex128
        )
        values_src = self._resample_complex_matrix(freq_src, values_src, freq_src)
        freq_dst = np.atleast_1d(np.asarray(frequencies, dtype=float))
        sampled = self._resample_complex_matrix(freq_src, values_src, freq_dst)
        phase = self._monitor_projection_phase(component, freq_dst, self.dt)
        sampled = sampled * phase[:, None]
        return freq_dst, sampled

    def _demodulate_monitor_component(
        self,
        monitor,
        component,
        frequency,
        t_start=None,
        avg_cycles=12,
        window="hann",
    ):
        """Demodulate one monitor component at a single CW frequency.

        Returns the complex amplitude vector over monitor samples.
        """
        if component not in monitor.fields:
            raise ValueError(
                f"Monitor '{monitor.name}' has no field '{component}'. "
                f"Available: {sorted(monitor.fields.keys())}"
            )
        raw = monitor.fields[component]
        if raw is None or len(raw) == 0:
            raise ValueError(
                f"Monitor '{monitor.name}' has no recorded '{component}' data."
            )
        values = np.asarray(raw)
        if values.ndim == 1:
            values = values[:, None]
        elif values.ndim > 2:
            values = values.reshape(values.shape[0], -1)

        t = np.asarray(monitor.fields.get("t", []), dtype=float)
        n = min(values.shape[0], t.size)
        if n < 2:
            raise ValueError(
                f"Monitor '{monitor.name}' has insufficient samples for demodulation."
            )
        values = values[:n]
        t = t[:n]
        f0 = float(frequency)
        if not np.isfinite(f0) or f0 <= 0:
            raise ValueError(f"frequency must be positive, got {frequency!r}")

        if t_start is None:
            mask = np.ones(n, dtype=bool)
        else:
            mask = t >= float(t_start)
        if np.count_nonzero(mask) < 2:
            raise ValueError(
                f"Monitor '{monitor.name}' has insufficient post-transient samples."
            )
        t_sel = t[mask]
        v_sel = values[mask]

        if avg_cycles is not None:
            cycles = float(avg_cycles)
            if cycles > 0:
                span = cycles / f0
                t_end = t_sel[0] + span
                keep = t_sel <= t_end
                if np.count_nonzero(keep) >= 2:
                    t_sel = t_sel[keep]
                    v_sel = v_sel[keep]

        n_sel = t_sel.size
        if n_sel < 2:
            raise ValueError(
                f"Monitor '{monitor.name}' has insufficient samples in demod window."
            )
        win_key = str(window).lower() if window is not None else "none"
        if win_key in {"hann", "hanning"}:
            w = np.hanning(n_sel)
        elif win_key in {"none", "rect", "rectangular"}:
            w = np.ones(n_sel, dtype=float)
        else:
            raise ValueError(f"Unsupported window '{window}'.")

        carrier = np.exp(1j * 2.0 * np.pi * f0 * t_sel)[:, None]
        denom = max(float(np.sum(w)), 1e-18)
        demod = (2.0 / denom) * np.sum((w[:, None] * v_sel) * carrier, axis=0)
        if hasattr(self, "dt") and self.dt is not None:
            dt = float(self.dt)
        else:
            dt = 0.0
        phase = self._monitor_projection_phase(component, np.asarray([f0]), dt)[0]
        demod = demod * phase
        return np.asarray(demod, dtype=np.complex128)

    @staticmethod
    def _mode_components_for_port(spec):
        axis = spec.direction[1]
        tm_map = {
            "x": ("Ez", "Hy", 2, 1, -1.0),
            "y": ("Ez", "Hx", 2, 0, 1.0),
            "z": ("Ey", "Hx", 1, 0, -1.0),
        }
        te_map = {
            "x": ("Ey", "Hz", 1, 2, 1.0),
            "y": ("Ex", "Hz", 0, 2, -1.0),
            "z": ("Ex", "Hy", 0, 1, 1.0),
        }
        if axis not in {"x", "y", "z"}:
            raise ValueError(f"Unsupported port axis '{axis}'.")
        e_comp, h_comp, e_idx, h_idx, sign = (
            tm_map[axis] if spec.polarization == "tm" else te_map[axis]
        )
        proj_components_3d = {
            "x": ("Ey", "Ez", "Hy", "Hz"),
            "y": ("Ex", "Ez", "Hx", "Hz"),
            "z": ("Ex", "Ey", "Hx", "Hy"),
        }[axis]
        return {
            "axis": axis,
            "e_component": e_comp,
            "h_component": h_comp,
            "e_mode_index": e_idx,
            "h_mode_index": h_idx,
            "signed_flux_sign": sign,
            "projection_components": (e_comp, h_comp),
            "projection_components_3d": proj_components_3d,
        }

    @staticmethod
    def _remap_3d_solver_components(ex, ey, ez, hx, hy, hz, axis):
        """Return solve_modes 3D components in the global Cartesian basis."""
        if axis not in {"x", "y", "z"}:
            raise ValueError(f"Unsupported axis {axis!r} for 3D mode remap.")
        return ex, ey, ez, hx, hy, hz

    @staticmethod
    def _stagger_3d_solver_components_to_yee(ex, ey, ez, hx, hy, hz, axis):
        """Sample collocated 3D solver fields on Beamz's transverse Yee lattices."""

        def _half(field, ax):
            arr = np.asarray(field, dtype=np.complex128)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.shape[ax] <= 1:
                return arr
            if ax == 0:
                return 0.5 * (arr[:-1, :] + arr[1:, :])
            return 0.5 * (arr[:, :-1] + arr[:, 1:])

        def _both(field):
            arr = np.asarray(field, dtype=np.complex128)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.shape[1] > 1:
                arr = 0.5 * (arr[:, :-1] + arr[:, 1:])
            if arr.shape[0] > 1:
                arr = 0.5 * (arr[:-1, :] + arr[1:, :])
            return arr

        ex = np.asarray(ex, dtype=np.complex128)
        ey = np.asarray(ey, dtype=np.complex128)
        ez = np.asarray(ez, dtype=np.complex128)
        hx = np.asarray(hx, dtype=np.complex128)
        hy = np.asarray(hy, dtype=np.complex128)
        hz = np.asarray(hz, dtype=np.complex128)
        for name, arr in (
            ("Ex", ex),
            ("Ey", ey),
            ("Ez", ez),
            ("Hx", hx),
            ("Hy", hy),
            ("Hz", hz),
        ):
            if arr.ndim not in {1, 2}:
                raise ValueError(
                    f"Expected 3D mode solver component '{name}' to be 1D or 2D, "
                    f"got {arr.shape}."
                )
        if axis == "x":
            return (
                ex,
                _half(ey, 1),
                _half(ez, 0),
                _both(hx),
                _half(hy, 0),
                _half(hz, 1),
            )
        if axis == "y":
            return (
                _half(ex, 1),
                ey,
                _half(ez, 0),
                _half(hx, 0),
                _both(hy),
                _half(hz, 1),
            )
        if axis == "z":
            return (
                _half(ex, 1),
                _half(ey, 0),
                ez,
                _half(hx, 0),
                _half(hy, 1),
                _both(hz),
            )
        raise ValueError(f"Unsupported axis {axis!r} for 3D Yee staggering.")

    @staticmethod
    def _opposite_port_direction(direction):
        direction = str(direction)
        if len(direction) != 2 or direction[0] not in "+-" or direction[1] not in "xyz":
            raise ValueError(f"Unsupported port direction {direction!r}.")
        return ("-" if direction[0] == "+" else "+") + direction[1]

    @staticmethod
    def _mode_parity_signature(component_grids, preferred_component):
        arr = None
        preferred = (preferred_component, "Ex", "Ey", "Ez")
        for name in preferred:
            value = component_grids.get(name)
            if value is None:
                continue
            candidate = np.asarray(value, dtype=np.complex128)
            if candidate.size and np.max(np.abs(candidate)) > 1e-12:
                arr = candidate
                break
        if arr is None:
            return (0.0, 0.0)
        if arr.ndim == 1:
            arr = arr[:, None]
        denom = float(np.sum(np.abs(arr) ** 2))
        if denom <= 1e-18:
            return (0.0, 0.0)
        scores = []
        for axis in range(2):
            flipped = np.flip(arr, axis=axis)
            corr = float(np.real(np.sum(arr * np.conjugate(flipped))) / denom)
            scores.append(float(np.clip(corr, -1.0, 1.0)))
        return tuple(scores)

    @staticmethod
    def _mode_parity_similarity(signature_a, signature_b):
        if signature_a is None or signature_b is None:
            return 0.0
        a = np.asarray(signature_a, dtype=float).reshape(-1)
        b = np.asarray(signature_b, dtype=float).reshape(-1)
        n = int(min(a.size, b.size))
        if n <= 0:
            return 0.0
        a = a[:n]
        b = b[:n]
        return float(np.mean(np.clip(1.0 - 0.5 * np.abs(a - b), 0.0, 1.0)))

    @staticmethod
    def _mode_shape_similarity(
        component_grids_a, component_grids_b, preferred_component
    ):
        def _select(arr_map):
            for name in (preferred_component, "Ex", "Ey", "Ez"):
                value = arr_map.get(name)
                if value is None:
                    continue
                arr = np.asarray(value, dtype=np.complex128)
                if arr.size and np.max(np.abs(arr)) > 1e-12:
                    return np.abs(arr)
            return None

        arr_a = _select(component_grids_a)
        arr_b = _select(component_grids_b)
        if arr_a is None or arr_b is None:
            return 0.0
        if arr_a.ndim == 1:
            arr_a = arr_a[:, None]
        if arr_b.ndim == 1:
            arr_b = arr_b[:, None]
        dim0 = int(min(arr_a.shape[0], arr_b.shape[0]))
        dim1 = int(min(arr_a.shape[1], arr_b.shape[1]))
        if dim0 <= 0 or dim1 <= 0:
            return 0.0
        a = arr_a[:dim0, :dim1].reshape(-1)
        b = arr_b[:dim0, :dim1].reshape(-1)
        a_norm = float(np.linalg.norm(a))
        b_norm = float(np.linalg.norm(b))
        if a_norm <= 1e-18 or b_norm <= 1e-18:
            return 0.0
        return float(np.clip(np.dot(a, b) / (a_norm * b_norm), 0.0, 1.0))

    @staticmethod
    def _plane_axes_for_port_axis(axis: str) -> tuple[str, str]:
        axis = str(axis).lower()
        mapping = {
            "x": ("z", "y"),
            "y": ("z", "x"),
            "z": ("y", "x"),
        }
        try:
            return mapping[axis]
        except KeyError as exc:
            raise ValueError(f"Unsupported port axis {axis!r}.") from exc

    @staticmethod
    def _analysis_plane_sample_area(coord0, coord1, fallback_step: float) -> float:
        def _axis_step(coord):
            arr = np.asarray(coord, dtype=np.float64).reshape(-1)
            if arr.size > 1:
                diffs = np.diff(arr)
                step = float(np.median(np.abs(diffs)))
                if np.isfinite(step) and step > 0.0:
                    return step
            return float(fallback_step)

        return float(_axis_step(coord0) * _axis_step(coord1))

    @staticmethod
    def _clamp_monitor_grid_index(idx, limit):
        if isinstance(idx, slice):
            start = 0 if idx.start is None else int(idx.start)
            stop = limit if idx.stop is None else int(idx.stop)
            start = max(0, min(start, max(limit - 1, 0)))
            stop = max(start + 1, min(stop, limit))
            return slice(start, stop)
        ii = int(idx)
        return max(0, min(ii, limit - 1))

    def _monitor_common_plane_shape_3d(self, monitor) -> tuple[int, int]:
        dims = []
        for comp_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            arr_shape = tuple(np.asarray(getattr(self.fields, comp_name)).shape)
            z_idx, y_idx, x_idx = monitor.get_grid_slice_3d(
                self.resolution,
                self.resolution,
                self.resolution,
                arr_shape,
            )
            z_idx = self._clamp_monitor_grid_index(z_idx, arr_shape[0])
            y_idx = self._clamp_monitor_grid_index(y_idx, arr_shape[1])
            x_idx = self._clamp_monitor_grid_index(x_idx, arr_shape[2])
            sample = np.asarray(
                np.zeros(arr_shape, dtype=np.float32)[z_idx, y_idx, x_idx]
            )
            if sample.ndim != 2:
                sample = np.atleast_2d(sample)
            dims.append(tuple(int(v) for v in sample.shape[:2]))
        if not dims:
            return 0, 0
        return (
            int(min(dim[0] for dim in dims)),
            int(min(dim[1] for dim in dims)),
        )

    def _monitor_analysis_plane_3d(
        self,
        monitor,
        axis: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(monitor, "get_analysis_plane_coords_3d"):
            try:
                coord0, coord1 = monitor.get_analysis_plane_coords_3d(
                    dx=self.resolution,
                    dy=self.resolution,
                    dz=self.resolution,
                    field_shape=tuple(np.asarray(self.fields.permittivity).shape),
                )
                return (
                    np.asarray(coord0, dtype=np.float64),
                    np.asarray(coord1, dtype=np.float64),
                )
            except Exception:
                pass
        perm_shape = tuple(np.asarray(self.fields.permittivity).shape)
        snapped = monitor.get_snapped_region(
            dx=self.resolution,
            dy=self.resolution,
            dz=self.resolution,
            field_shape=perm_shape,
        )
        if snapped is None:
            raise ValueError(f"Monitor '{monitor.name}' has no snapped 3D region.")
        axis0, axis1 = self._plane_axes_for_port_axis(axis)
        interval0 = snapped.axis_interval(axis0)
        interval1 = snapped.axis_interval(axis1)
        if interval0 is None or interval1 is None:
            raise ValueError(
                f"Monitor '{monitor.name}' is missing tangential intervals for axis '{axis}'."
            )
        coord0 = (
            np.arange(int(interval0.start), int(interval0.stop), dtype=np.float64) + 0.5
        ) * float(self.resolution)
        coord1 = (
            np.arange(int(interval1.start), int(interval1.stop), dtype=np.float64) + 0.5
        ) * float(self.resolution)
        common0, common1 = self._monitor_common_plane_shape_3d(monitor)
        return coord0[:common0], coord1[:common1]

    def _monitor_component_plane_coords_3d(
        self,
        monitor,
        component: str,
        axis: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(monitor, "get_analysis_plane_coords_3d"):
            try:
                coord0, coord1 = monitor.get_analysis_plane_coords_3d(
                    dx=self.resolution,
                    dy=self.resolution,
                    dz=self.resolution,
                    field_shape=tuple(np.asarray(self.fields.permittivity).shape),
                )
                return (
                    np.asarray(coord0, dtype=np.float64),
                    np.asarray(coord1, dtype=np.float64),
                )
            except Exception:
                pass
        field = np.asarray(getattr(self.fields, component))
        field_shape = tuple(int(v) for v in field.shape)
        z_idx, y_idx, x_idx = monitor.get_grid_slice_3d(
            self.resolution,
            self.resolution,
            self.resolution,
            field_shape,
        )
        z_idx = self._clamp_monitor_grid_index(z_idx, field_shape[0])
        y_idx = self._clamp_monitor_grid_index(y_idx, field_shape[1])
        x_idx = self._clamp_monitor_grid_index(x_idx, field_shape[2])
        coords_um = component_coordinates_3d_um(
            component,
            tuple(int(v) for v in np.asarray(self.fields.permittivity).shape),
            float(self.resolution / µm),
        )
        axis0, axis1 = self._plane_axes_for_port_axis(axis)
        axis_slices = {"z": z_idx, "y": y_idx, "x": x_idx}
        common0, common1 = self._monitor_common_plane_shape_3d(monitor)
        coord0 = np.asarray(coords_um[axis0][axis_slices[axis0]], dtype=np.float64)[
            :common0
        ] * float(µm)
        coord1 = np.asarray(coords_um[axis1][axis_slices[axis1]], dtype=np.float64)[
            :common1
        ] * float(µm)
        return coord0, coord1

    @staticmethod
    def _interpolate_plane_matrix_2d(
        values: np.ndarray,
        src0: np.ndarray,
        src1: np.ndarray,
        dst0: np.ndarray,
        dst1: np.ndarray,
    ) -> np.ndarray:
        src0 = np.asarray(src0, dtype=np.float64).reshape(-1)
        src1 = np.asarray(src1, dtype=np.float64).reshape(-1)
        dst0 = np.asarray(dst0, dtype=np.float64).reshape(-1)
        dst1 = np.asarray(dst1, dtype=np.float64).reshape(-1)
        arr = np.asarray(values, dtype=np.complex128)
        if arr.shape != (src0.size, src1.size):
            raise ValueError(
                "Plane interpolation shape mismatch: "
                f"values={arr.shape}, src0={src0.size}, src1={src1.size}"
            )
        if np.array_equal(src0, dst0) and np.array_equal(src1, dst1):
            return arr.copy()
        mid = np.empty((src0.size, dst1.size), dtype=np.complex128)
        for row in range(src0.size):
            mid[row, :] = np.interp(dst1, src1, np.real(arr[row, :])) + 1j * np.interp(
                dst1,
                src1,
                np.imag(arr[row, :]),
            )
        out = np.empty((dst0.size, dst1.size), dtype=np.complex128)
        for col in range(dst1.size):
            out[:, col] = np.interp(dst0, src0, np.real(mid[:, col])) + 1j * np.interp(
                dst0,
                src0,
                np.imag(mid[:, col]),
            )
        return out

    def _colocate_monitor_component_matrix_3d(
        self,
        monitor,
        component: str,
        values: np.ndarray,
        *,
        axis: str,
        target0: np.ndarray,
        target1: np.ndarray,
    ) -> np.ndarray:
        src0, src1 = self._monitor_component_plane_coords_3d(
            monitor,
            component,
            axis,
        )
        data = np.asarray(values, dtype=np.complex128)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2:
            raise ValueError(
                f"Expected DFT matrix with shape (nfreq, npoints) for component '{component}', got {data.shape}."
            )
        n_src = int(src0.size * src1.size)
        if data.shape[1] != n_src:
            raise ValueError(
                f"Component '{component}' has {data.shape[1]} samples but expected {n_src} from monitor geometry."
            )
        out = np.empty(
            (data.shape[0], int(len(target0) * len(target1))), dtype=np.complex128
        )
        for idx in range(data.shape[0]):
            plane = data[idx].reshape(src0.size, src1.size)
            interp = self._interpolate_plane_matrix_2d(
                plane,
                src0,
                src1,
                target0,
                target1,
            )
            out[idx, :] = interp.reshape(-1)
        return out

    def _colocate_field_components_to_projection_3d(
        self,
        monitor,
        field_components: Mapping[str, np.ndarray],
        projection: Mapping[str, Any],
    ) -> dict[str, np.ndarray]:
        target0 = projection.get("analysis_coords0")
        target1 = projection.get("analysis_coords1")
        axis = projection.get("axis")
        if target0 is None or target1 is None or axis is None:
            return {
                name: np.asarray(value, dtype=np.complex128)
                for name, value in field_components.items()
            }
        colocated = {}
        for name, value in field_components.items():
            arr = np.asarray(value, dtype=np.complex128)
            was_vector = arr.ndim == 1
            if was_vector:
                arr = arr[None, :]
            interp = self._colocate_monitor_component_matrix_3d(
                monitor,
                name,
                arr,
                axis=str(axis),
                target0=np.asarray(target0, dtype=np.float64),
                target1=np.asarray(target1, dtype=np.float64),
            )
            colocated[name] = interp[0] if was_vector else interp
        return colocated

    @staticmethod
    def _projection_seed_key(spec, monitor, parts, *, is_3d):
        if not is_3d:
            return None
        try:
            proj_components = tuple(
                parts.get("projection_components_3d", parts["projection_components"])
            )
        except Exception:
            return None
        shape_key = []
        try:
            for comp in proj_components:
                arr = np.asarray(monitor.get_dft_component(comp), dtype=np.complex128)
                if arr.ndim != 2:
                    return None
                shape_key.append(int(arr.shape[1]))
        except Exception:
            return None
        return (
            str(spec.name),
            str(getattr(monitor, "name", "")),
            str(spec.direction),
            str(spec.polarization).lower(),
            int(spec.mode_index),
            tuple(proj_components),
            tuple(shape_key),
        )

    def _monitor_profile_slice(self, monitor, axis, pad_cells):
        perm = np.asarray(self.fields.permittivity)
        if perm.ndim == 3:
            z_idx, y_idx, x_idx = monitor.get_grid_slice_3d(
                self.resolution,
                self.resolution,
                self.resolution,
                perm.shape,
            )

            def _clamp(idx, limit):
                if isinstance(idx, slice):
                    start = 0 if idx.start is None else int(idx.start)
                    stop = limit if idx.stop is None else int(idx.stop)
                    start = max(0, min(start, max(limit - 1, 0)))
                    stop = max(start + 1, min(stop, limit))
                    return slice(start, stop)
                ii = int(idx)
                return max(0, min(ii, limit - 1))

            z_idx = _clamp(z_idx, perm.shape[0])
            y_idx = _clamp(y_idx, perm.shape[1])
            x_idx = _clamp(x_idx, perm.shape[2])
            eps_slice = np.asarray(perm[z_idx, y_idx, x_idx], dtype=np.complex128)
            if eps_slice.ndim != 2:
                eps_slice = np.atleast_2d(eps_slice)
            npts = int(eps_slice.size)
            local_idx = np.arange(npts, dtype=int)
            d_area = float(self.resolution) * float(self.resolution)
            return eps_slice, local_idx, d_area
        if perm.ndim != 2:
            raise NotImplementedError("Modal extraction supports 2D or 3D only.")
        points = monitor.get_grid_points_2d(self.resolution, self.resolution)
        if not points:
            raise ValueError(f"Monitor '{monitor.name}' contains no sample points.")
        p = np.asarray(points, dtype=float)
        if axis == "x":
            x_idx = int(np.clip(round(float(np.mean(p[:, 0]))), 0, perm.shape[1] - 1))
            eps_profile_full = perm[:, x_idx]
            sample_idx = np.asarray(
                [int(np.clip(pi[1], 0, perm.shape[0] - 1)) for pi in points], dtype=int
            )
        else:
            y_idx = int(np.clip(round(float(np.mean(p[:, 1]))), 0, perm.shape[0] - 1))
            eps_profile_full = perm[y_idx, :]
            sample_idx = np.asarray(
                [int(np.clip(pi[0], 0, perm.shape[1] - 1)) for pi in points], dtype=int
            )
        lo = max(0, int(np.min(sample_idx)) - int(pad_cells))
        hi = min(len(eps_profile_full), int(np.max(sample_idx)) + int(pad_cells) + 1)
        local_idx = np.clip(sample_idx - lo, 0, max(hi - lo - 1, 0))
        if len(points) > 1:
            step_idx = np.diff(np.asarray(points, dtype=float), axis=0)
            dl = float(np.mean(np.linalg.norm(step_idx, axis=1))) * float(
                self.resolution
            )
        else:
            dl = float(self.resolution)
        dl = max(dl, float(self.resolution) * 1e-9)
        return np.asarray(eps_profile_full[lo:hi], dtype=np.complex128), local_idx, dl

    def _component_index_plane_coords_3d(self, component, index, axis):
        coords_um = component_coordinates_3d_um(
            component,
            tuple(int(v) for v in np.asarray(self.fields.permittivity).shape),
            float(self.resolution / µm),
        )
        axis0, axis1 = self._plane_axes_for_port_axis(axis)
        axis_indices = {"z": index[0], "y": index[1], "x": index[2]}
        coord0 = np.asarray(coords_um[axis0][axis_indices[axis0]], dtype=np.float64)
        coord1 = np.asarray(coords_um[axis1][axis_indices[axis1]], dtype=np.float64)
        return coord0.reshape(-1) * float(µm), coord1.reshape(-1) * float(µm)

    def _discrete_mode_projection_grids_3d(
        self,
        discrete_mode,
        profiles,
        *,
        monitor,
        axis,
        components,
        analysis_coords0,
        analysis_coords1,
    ):
        del monitor
        grids = {}
        samples = {}
        for name in components:
            if name not in profiles:
                continue
            arr = np.asarray(profiles[name], dtype=np.complex128)
            if arr.ndim == 1:
                arr = arr[:, None]
            index = discrete_mode.component_indices.get(name)
            if index is None:
                continue
            src0, src1 = self._component_index_plane_coords_3d(name, index, axis)
            rows = min(int(arr.shape[0]), int(src0.size))
            cols = min(int(arr.shape[1]), int(src1.size))
            if rows <= 0 or cols <= 0:
                continue
            grid = self._interpolate_plane_matrix_2d(
                arr[:rows, :cols],
                src0[:rows],
                src1[:cols],
                np.asarray(analysis_coords0, dtype=np.float64),
                np.asarray(analysis_coords1, dtype=np.float64),
            )
            grids[name] = grid
            samples[name] = grid.reshape(-1)
        return grids, samples

    def _build_discrete_port_projection_3d(
        self,
        *,
        spec,
        monitor,
        frequency,
        parts,
        direction_sign,
        target_neff,
        mode_candidates,
        analysis_coords0,
        analysis_coords1,
    ):
        if type(monitor).__name__ != "ModeMonitor":
            return None
        if not hasattr(monitor, "center") or not hasattr(monitor, "size_spec"):
            return None
        perm = np.asarray(self.fields.permittivity)
        if perm.ndim != 3:
            return None

        axis = parts["axis"]
        axis_index = {"z": 0, "y": 1, "x": 2}[axis]
        try:
            z_idx, y_idx, x_idx = monitor.get_grid_slice_3d(
                self.resolution,
                self.resolution,
                self.resolution,
                perm.shape,
            )
        except Exception:
            return None

        normal_index = {"z": z_idx, "y": y_idx, "x": x_idx}[axis]
        if isinstance(normal_index, slice):
            start = 0 if normal_index.start is None else int(normal_index.start)
            stop = (
                perm.shape[axis_index]
                if normal_index.stop is None
                else int(normal_index.stop)
            )
            plane_index = int(
                np.clip(
                    (start + max(start + 1, stop) - 1) // 2,
                    0,
                    perm.shape[axis_index] - 1,
                )
            )
        else:
            plane_index = int(np.clip(int(normal_index), 0, perm.shape[axis_index] - 1))
        if direction_sign > 0.0:
            offset_index = max(0, plane_index - 1)
        else:
            offset_index = min(max(perm.shape[axis_index] - 2, 0), plane_index + 1)

        mode_spec = getattr(monitor, "mode_spec", None)
        num_modes = int(
            max(
                int(mode_candidates),
                int(getattr(mode_spec, "num_modes", 0) or 0),
                int(spec.mode_index) + 1,
            )
        )
        target = getattr(mode_spec, "target_neff", None)
        if target is None:
            target = target_neff

        center = tuple(float(value) for value in monitor.center)
        size = tuple(float(value) for value in monitor.size_spec)
        if axis == "x":
            width, height = size[1], size[2]
        elif axis == "y":
            width, height = size[0], size[2]
        else:
            width, height = size[0], size[1]

        eps_profile_full = np.take(perm, plane_index, axis=axis_index)
        transverse_axes = self._plane_axes_for_port_axis(axis)
        component_shapes = {
            name: tuple(int(v) for v in np.asarray(getattr(self.fields, name)).shape)
            for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        }
        permittivity_arr = np.asarray(perm)
        permeability_arr = np.ones_like(permittivity_arr, dtype=np.float64)
        component_permittivity = {
            component: np.asarray(
                sample_voxel_grid_at_e_component_3d_centered(
                    permittivity_arr,
                    component,
                    stored_shape=component_shapes[component],
                )
            )
            for component in ("Ex", "Ey", "Ez")
        }
        component_permeability = {
            component: np.asarray(
                sample_voxel_grid_at_component_3d(
                    permeability_arr,
                    component,
                    stored_shape=component_shapes[component],
                )
            )
            for component in ("Hx", "Hy", "Hz")
        }
        try:
            discrete_mode = solve_beamz_mode_plane(
                scalar_permittivity=np.asarray(eps_profile_full, dtype=np.complex128),
                frequency=float(frequency),
                resolution=float(self.resolution),
                dt=None if getattr(self, "dt", None) is None else float(self.dt),
                axis=axis,
                direction=str(spec.direction),
                solver_direction=str(spec.direction),
                transverse_axes=transverse_axes,
                grid_shape=tuple(int(v) for v in perm.shape),
                component_shapes=component_shapes,
                component_permittivity=component_permittivity,
                component_permeability=component_permeability,
                center=center,
                width=float(width),
                height=float(height),
                plane_index=int(plane_index),
                offset_index=int(offset_index),
                mode_index=int(spec.mode_index),
                polarization=str(spec.polarization).lower(),
                target_neff=target,
                num_modes=num_modes,
                aperture_pad_cells=0,
                aperture_window_alpha=0.0,
            )
        except Exception:
            return None
        if discrete_mode is None:
            return None

        proj_components = tuple(parts.get("projection_components_3d", ()))
        if not proj_components:
            return None
        d_area = self._analysis_plane_sample_area(
            analysis_coords0,
            analysis_coords1,
            float(self.resolution),
        )
        plus_grids, plus_components = self._discrete_mode_projection_grids_3d(
            discrete_mode,
            discrete_mode.backward_profiles,
            monitor=monitor,
            axis=axis,
            components=proj_components,
            analysis_coords0=analysis_coords0,
            analysis_coords1=analysis_coords1,
        )
        minus_grids, minus_components = self._discrete_mode_projection_grids_3d(
            discrete_mode,
            discrete_mode.profiles,
            monitor=monitor,
            axis=axis,
            components=proj_components,
            analysis_coords0=analysis_coords0,
            analysis_coords1=analysis_coords1,
        )
        if any(
            name not in plus_components or name not in minus_components
            for name in proj_components
        ):
            return None

        plus_components = _normalize_3d_profiles_by_flux(
            {
                name: np.asarray(plus_components[name], dtype=np.complex128)
                for name in proj_components
            },
            axis=axis,
            d_area=float(d_area),
            direction_sign=float(direction_sign),
        )
        minus_components = _normalize_3d_profiles_by_flux(
            {
                name: np.asarray(minus_components[name], dtype=np.complex128)
                for name in proj_components
            },
            axis=axis,
            d_area=float(d_area),
            direction_sign=float(direction_sign),
        )
        mode_matrix = np.column_stack(
            [
                np.concatenate([plus_components[name] for name in proj_components]),
                np.concatenate([minus_components[name] for name in proj_components]),
            ]
        )
        overlap_matrix = np.asarray(
            [
                [
                    _safe_modal_overlap_3d(
                        plus_components,
                        plus_components,
                        axis,
                        float(d_area),
                        direction_sign=direction_sign,
                    ),
                    _safe_modal_overlap_3d(
                        plus_components,
                        minus_components,
                        axis,
                        float(d_area),
                        direction_sign=direction_sign,
                    ),
                ],
                [
                    _safe_modal_overlap_3d(
                        minus_components,
                        plus_components,
                        axis,
                        float(d_area),
                        direction_sign=direction_sign,
                    ),
                    _safe_modal_overlap_3d(
                        minus_components,
                        minus_components,
                        axis,
                        float(d_area),
                        direction_sign=direction_sign,
                    ),
                ],
            ],
            dtype=np.complex128,
        )
        projection = {
            "e_component": parts["e_component"],
            "h_component": parts["h_component"],
            "components": tuple(proj_components),
            "mode_matrix": mode_matrix,
            "condition_number": float(np.linalg.cond(overlap_matrix)),
            "pinv": np.linalg.pinv(mode_matrix),
            "mode_neff": float(np.real(np.asarray(discrete_mode.neff))),
            "mode_neff_bwd": float(np.real(np.asarray(discrete_mode.neff))),
            "mode_components": {
                name: np.asarray(plus_components[name], dtype=np.complex128)
                for name in proj_components
            },
            "mode_components_bwd": {
                name: np.asarray(minus_components[name], dtype=np.complex128)
                for name in proj_components
            },
            "overlap_matrix": overlap_matrix,
            "axis": axis,
            "direction_sign": float(direction_sign),
            "d_area": float(d_area),
            "power_norm": 1.0,
            "mode_parity": self._mode_parity_signature(
                plus_grids,
                parts["e_component"],
            ),
            "mode_parity_bwd": self._mode_parity_signature(
                minus_grids,
                parts["e_component"],
            ),
            "mode_component_grids": plus_grids,
            "mode_component_grids_bwd": minus_grids,
            "pair_score": np.nan,
            "discrete_contract": "micromode.beamz.DiscreteMode/v1",
            "analysis_coords0": np.asarray(analysis_coords0, dtype=np.float64),
            "analysis_coords1": np.asarray(analysis_coords1, dtype=np.float64),
        }
        projection["modal_plane_delay_s"] = self._modal_projection_plane_delay_s(
            spec,
            frequency,
            projection["mode_neff"],
        )
        return projection

    def _build_port_projection(
        self,
        spec,
        monitor,
        frequency,
        cache,
        mode_pad_cells=6,
        previous_projection=None,
    ):
        key = (spec.name, monitor.name, float(frequency))
        cached = cache.get(key)
        if cached is not None:
            return cached

        parts = self._mode_components_for_port(spec)
        analysis_coords0 = None
        analysis_coords1 = None
        if self.is_3d and hasattr(monitor, "get_snapped_region"):
            analysis_coords0, analysis_coords1 = self._monitor_analysis_plane_3d(
                monitor, parts["axis"]
            )
        eps_profile, local_idx, dl = self._monitor_profile_slice(
            monitor, parts["axis"], mode_pad_cells
        )
        solver_direction = spec.direction
        basis_direction = solver_direction
        backward_direction = self._opposite_port_direction(spec.direction)
        if self.is_3d:
            basis_direction = _projection_solver_direction_3d(
                spec.direction, parts["axis"]
            )
            backward_direction = _projection_solver_direction_3d(
                self._opposite_port_direction(spec.direction),
                parts["axis"],
            )
        direction_sign = +1.0 if str(spec.direction).startswith("+") else -1.0
        omega = 2.0 * np.pi * float(frequency)
        eps_profile_arr = np.asarray(eps_profile)
        n_local_max = float(
            np.sqrt(max(float(np.max(np.real(eps_profile_arr))), 1e-12))
        )
        target_neff = 0.98 * n_local_max
        mode_candidates = max(int(spec.mode_index) + 1, 3 if self.is_3d else 1)

        if self.is_3d and analysis_coords0 is not None and analysis_coords1 is not None:
            projection = self._build_discrete_port_projection_3d(
                spec=spec,
                monitor=monitor,
                frequency=frequency,
                parts=parts,
                direction_sign=direction_sign,
                target_neff=target_neff,
                mode_candidates=mode_candidates,
                analysis_coords0=analysis_coords0,
                analysis_coords1=analysis_coords1,
            )
            if projection is not None:
                cache[key] = projection
                return projection

        def _solve_candidate_set(direction):
            try:
                return solve_modes(
                    eps=eps_profile,
                    omega=omega,
                    dL=float(self.resolution),
                    m=mode_candidates,
                    direction=direction,
                    filter_pol=spec.polarization,
                    target_neff=target_neff,
                    return_fields=True,
                )
            except ValueError:
                return solve_modes(
                    eps=eps_profile,
                    omega=omega,
                    dL=float(self.resolution),
                    m=spec.mode_index + 1,
                    direction=direction,
                    filter_pol=spec.polarization,
                    target_neff=target_neff,
                    return_fields=True,
                )

        def _candidate_record(
            mode_index,
            neff_values,
            e_field_set,
            h_field_set,
            direction,
            *,
            physical_direction=None,
        ):
            sign_direction = (
                physical_direction if physical_direction is not None else direction
            )
            local_sign = +1.0 if str(sign_direction).startswith("+") else -1.0
            if self.is_3d:
                ex_full = np.asarray(
                    np.squeeze(e_field_set[mode_index][0]), dtype=np.complex128
                )
                ey_full = np.asarray(
                    np.squeeze(e_field_set[mode_index][1]), dtype=np.complex128
                )
                ez_full = np.asarray(
                    np.squeeze(e_field_set[mode_index][2]), dtype=np.complex128
                )
                hx_full = np.asarray(
                    np.squeeze(h_field_set[mode_index][0]), dtype=np.complex128
                )
                hy_full = np.asarray(
                    np.squeeze(h_field_set[mode_index][1]), dtype=np.complex128
                )
                hz_full = np.asarray(
                    np.squeeze(h_field_set[mode_index][2]), dtype=np.complex128
                )
                ex_full, ey_full, ez_full, hx_full, hy_full, hz_full = (
                    self._remap_3d_solver_components(
                        ex_full,
                        ey_full,
                        ez_full,
                        hx_full,
                        hy_full,
                        hz_full,
                        parts["axis"],
                    )
                )
                ex_full, ey_full, ez_full, hx_full, hy_full, hz_full = (
                    self._stagger_3d_solver_components_to_yee(
                        ex_full,
                        ey_full,
                        ez_full,
                        hx_full,
                        hy_full,
                        hz_full,
                        parts["axis"],
                    )
                )
                comp_full = {
                    "Ex": ex_full,
                    "Ey": ey_full,
                    "Ez": ez_full,
                    "Hx": hx_full,
                    "Hy": hy_full,
                    "Hz": hz_full,
                }
                for name in tuple(comp_full.keys()):
                    arr = np.asarray(comp_full[name], dtype=np.complex128)
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    comp_full[name] = arr
                symmetric_axes = _detect_transverse_symmetry_axes(eps_profile)
                if symmetric_axes:
                    comp_full = _enforce_componentwise_parity(
                        comp_full,
                        symmetric_axes,
                    )

                proj_components_local = tuple(parts.get("projection_components_3d", ()))
                mon_dim0 = 0
                mon_dim1 = 0
                try:
                    shape_map = {
                        "Ex": tuple(np.asarray(self.fields.Ex).shape),
                        "Ey": tuple(np.asarray(self.fields.Ey).shape),
                        "Ez": tuple(np.asarray(self.fields.Ez).shape),
                        "Hx": tuple(np.asarray(self.fields.Hx).shape),
                        "Hy": tuple(np.asarray(self.fields.Hy).shape),
                        "Hz": tuple(np.asarray(self.fields.Hz).shape),
                    }

                    def _clamp_idx(idx, limit):
                        if isinstance(idx, slice):
                            start = 0 if idx.start is None else int(idx.start)
                            stop = limit if idx.stop is None else int(idx.stop)
                            start = max(0, min(start, max(limit - 1, 0)))
                            stop = max(start + 1, min(stop, limit))
                            return slice(start, stop)
                        ii = int(idx)
                        return max(0, min(ii, limit - 1))

                    def _slice_len(idx, limit):
                        if isinstance(idx, slice):
                            start = 0 if idx.start is None else int(idx.start)
                            stop = limit if idx.stop is None else int(idx.stop)
                            step = 1 if idx.step is None else int(idx.step)
                            if step <= 0:
                                raise ValueError(
                                    "Only positive slice steps are supported."
                                )
                            span = max(0, stop - start)
                            return 0 if span <= 0 else 1 + (span - 1) // step
                        return 1

                    dims0 = []
                    dims1 = []
                    for cname in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                        shp = shape_map[cname]
                        z_idx, y_idx, x_idx = monitor.get_grid_slice_3d(
                            self.resolution,
                            self.resolution,
                            self.resolution,
                            shp,
                        )
                        z_idx = _clamp_idx(z_idx, shp[0])
                        y_idx = _clamp_idx(y_idx, shp[1])
                        x_idx = _clamp_idx(x_idx, shp[2])
                        slice_lens = [
                            _slice_len(idx, lim)
                            for idx, lim in (
                                (z_idx, shp[0]),
                                (y_idx, shp[1]),
                                (x_idx, shp[2]),
                            )
                            if isinstance(idx, slice)
                        ]
                        if len(slice_lens) >= 2:
                            d0, d1 = int(slice_lens[0]), int(slice_lens[1])
                        elif len(slice_lens) == 1:
                            d0, d1 = int(slice_lens[0]), 1
                        else:
                            d0, d1 = 1, 1
                        dims0.append(max(d0, 1))
                        dims1.append(max(d1, 1))
                    mon_dim0 = min(dims0)
                    mon_dim1 = min(dims1)
                    if analysis_coords0 is not None and analysis_coords1 is not None:
                        mon_dim0 = int(np.asarray(analysis_coords0).size)
                        mon_dim1 = int(np.asarray(analysis_coords1).size)
                except Exception:
                    if analysis_coords0 is not None and analysis_coords1 is not None:
                        mon_dim0 = int(np.asarray(analysis_coords0).size)
                        mon_dim1 = int(np.asarray(analysis_coords1).size)
                    else:
                        mon_dim0 = min(
                            int(comp_full[c].shape[0]) for c in proj_components_local
                        )
                        mon_dim1 = min(
                            int(comp_full[c].shape[1]) for c in proj_components_local
                        )

                try:
                    n_monitor = min(
                        int(
                            np.asarray(
                                monitor.get_dft_component(comp_name),
                                dtype=np.complex128,
                            ).shape[1]
                        )
                        for comp_name in proj_components_local
                    )
                    if mon_dim0 > 0 and mon_dim1 > 0:
                        n_monitor = min(n_monitor, int(mon_dim0 * mon_dim1))
                    if mon_dim0 > 0:
                        mon_dim1 = max(
                            1, min(mon_dim1, int(n_monitor // max(mon_dim0, 1)))
                        )
                    if mon_dim1 > 0:
                        mon_dim0 = max(
                            1, min(mon_dim0, int(n_monitor // max(mon_dim1, 1)))
                        )
                except Exception:
                    n_monitor = int(mon_dim0 * mon_dim1)

                crop_dim0 = int(
                    min(
                        mon_dim0,
                        *(comp_full[c].shape[0] for c in proj_components_local),
                    )
                )
                crop_dim1 = int(
                    min(
                        mon_dim1,
                        *(comp_full[c].shape[1] for c in proj_components_local),
                    )
                )
                if crop_dim0 <= 0 or crop_dim1 <= 0:
                    crop_dim0 = int(
                        min(comp_full[c].shape[0] for c in proj_components_local)
                    )
                    crop_dim1 = int(
                        min(comp_full[c].shape[1] for c in proj_components_local)
                    )
                n_target = int(crop_dim0 * crop_dim1)
                if n_monitor > 0:
                    n_target = min(n_target, n_monitor)
                if n_target <= 0:
                    raise ValueError(
                        f"Monitor '{monitor.name}' has zero 3D projection points."
                    )
                crop_dim1 = max(1, min(crop_dim1, n_target))
                crop_dim0 = max(1, min(crop_dim0, n_target // crop_dim1))
                n_target = int(crop_dim0 * crop_dim1)

                component_grids = {}
                comp_samples_local = {}
                for name, arr in comp_full.items():
                    a = np.asarray(arr, dtype=np.complex128)
                    if a.ndim == 1:
                        a = a[:, None]
                    a = a[:crop_dim0, :crop_dim1]
                    if analysis_coords0 is not None and analysis_coords1 is not None:
                        src0, src1 = self._monitor_component_plane_coords_3d(
                            monitor,
                            name,
                            parts["axis"],
                        )
                        src0 = np.asarray(src0, dtype=np.float64)[: a.shape[0]]
                        src1 = np.asarray(src1, dtype=np.float64)[: a.shape[1]]
                        a = self._interpolate_plane_matrix_2d(
                            a,
                            src0,
                            src1,
                            analysis_coords0,
                            analysis_coords1,
                        )
                    component_grids[name] = a
                    comp_samples_local[name] = a.reshape(-1)

                h_ref = comp_samples_local.get(
                    parts["h_component"], np.zeros((0,), dtype=np.complex128)
                )
                phase_rot = 1.0 + 0.0j
                if h_ref.size:
                    i_max = int(np.argmax(np.abs(h_ref)))
                    phase_rot = np.exp(-1j * np.angle(h_ref[i_max]))
                for name in tuple(comp_samples_local.keys()):
                    comp_samples_local[name] = comp_samples_local[name] * phase_rot
                    component_grids[name] = component_grids[name] * phase_rot

                raw_mode_components = {
                    name: np.asarray(
                        comp_samples_local[name], dtype=np.complex128
                    ).reshape(-1)
                    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
                    if name in comp_samples_local
                }
                mode_components, _ = _make_3d_mode_basis_profiles(
                    raw_mode_components,
                    axis=parts["axis"],
                    d_area=float(dl),
                    direction_sign=local_sign,
                )
                return {
                    "mode_index": int(mode_index),
                    "proj_components": proj_components_local,
                    "mode_components": mode_components,
                    "component_grids": component_grids,
                    "parity_signature": self._mode_parity_signature(
                        component_grids,
                        parts["e_component"],
                    ),
                    "direction_sign": float(local_sign),
                    "mode_neff": float(np.real(np.asarray(neff_values[mode_index]))),
                }

            e_fwd_full = np.asarray(
                np.squeeze(e_field_set[mode_index][parts["e_mode_index"]]),
                dtype=np.complex128,
            )
            h_fwd_full = np.asarray(
                np.squeeze(h_field_set[mode_index][parts["h_mode_index"]]),
                dtype=np.complex128,
            )
            if e_fwd_full.ndim > 1:
                e_fwd_full = e_fwd_full[:, 0]
            if h_fwd_full.ndim > 1:
                h_fwd_full = h_fwd_full[:, 0]
            e_fwd = e_fwd_full[local_idx]
            h_fwd = h_fwd_full[local_idx]
            return {
                "mode_index": int(mode_index),
                "proj_components": (parts["e_component"], parts["h_component"]),
                "e_profile": e_fwd,
                "h_profile": h_fwd,
                "mode_neff": float(np.real(np.asarray(neff_values[mode_index]))),
            }

        neff_vals, e_fields, h_fields, _ = _solve_candidate_set(basis_direction)

        if self.is_3d:
            fwd_candidates = [
                _candidate_record(
                    idx,
                    neff_vals,
                    e_fields,
                    h_fields,
                    basis_direction,
                    physical_direction=spec.direction,
                )
                for idx in range(len(neff_vals))
            ]
            if not fwd_candidates:
                raise ValueError(f"No local modes found for port '{spec.name}'.")

            mode = int(min(spec.mode_index, len(fwd_candidates) - 1))
            selected_fwd = fwd_candidates[mode]
            if previous_projection is not None and len(fwd_candidates) > 1:
                prev_components = previous_projection.get("mode_components", {})
                prev_grids = previous_projection.get("mode_component_grids", {})
                prev_parity = previous_projection.get("mode_parity", ())
                prev_neff = float(previous_projection.get("mode_neff", np.nan))
                best_fwd_score = -np.inf
                best_fwd = selected_fwd
                for cand in fwd_candidates:
                    shape_score = self._mode_shape_similarity(
                        prev_grids,
                        cand["component_grids"],
                        parts["e_component"],
                    )
                    parity_score = self._mode_parity_similarity(
                        prev_parity,
                        cand["parity_signature"],
                    )
                    try:
                        overlap_score = abs(
                            _safe_modal_overlap_3d(
                                prev_components,
                                cand["mode_components"],
                                parts["axis"],
                                float(dl),
                                direction_sign=direction_sign,
                            )
                        )
                    except Exception:
                        overlap_score = 0.0
                    neff_delta = (
                        abs(prev_neff - cand["mode_neff"])
                        if np.isfinite(prev_neff)
                        else 0.0
                    )
                    score = 0.80 * shape_score + 0.65 * parity_score
                    score += 0.55 * overlap_score - 0.05 * neff_delta
                    if score > best_fwd_score + 1e-9:
                        best_fwd_score = score
                        best_fwd = cand
                selected_fwd = best_fwd

            neff_vals_bwd, e_fields_bwd, h_fields_bwd, _ = _solve_candidate_set(
                backward_direction
            )
            bwd_candidates = [
                _candidate_record(
                    idx,
                    neff_vals_bwd,
                    e_fields_bwd,
                    h_fields_bwd,
                    backward_direction,
                    physical_direction=self._opposite_port_direction(spec.direction),
                )
                for idx in range(len(neff_vals_bwd))
            ]
            if not bwd_candidates:
                raise ValueError(
                    f"No backward local modes found for port '{spec.name}'."
                )
            selected_bwd = bwd_candidates[0]
            best_pair_score = -np.inf
            for cand in bwd_candidates:
                parity_score = self._mode_parity_similarity(
                    selected_fwd["parity_signature"],
                    cand["parity_signature"],
                )
                shape_score = self._mode_shape_similarity(
                    selected_fwd["component_grids"],
                    cand["component_grids"],
                    parts["e_component"],
                )
                overlap_score = abs(
                    _safe_modal_overlap_3d(
                        selected_fwd["mode_components"],
                        cand["mode_components"],
                        parts["axis"],
                        float(dl),
                        direction_sign=direction_sign,
                    )
                )
                neff_delta = abs(selected_fwd["mode_neff"] - cand["mode_neff"])
                pair_score = 0.65 * shape_score + 0.55 * parity_score
                pair_score += 0.25 * overlap_score - 0.05 * neff_delta
                if pair_score > best_pair_score + 1e-9:
                    best_pair_score = pair_score
                    selected_bwd = cand

            mode_components = {
                name: np.asarray(value, dtype=np.complex128)
                for name, value in selected_fwd["mode_components"].items()
            }
            mode_components_bwd = {
                name: np.asarray(value, dtype=np.complex128)
                for name, value in selected_bwd["mode_components"].items()
            }
            if previous_projection is not None:
                prev_components = previous_projection.get("mode_components", {})
                try:
                    prev_overlap = _safe_modal_overlap_3d(
                        prev_components,
                        mode_components,
                        parts["axis"],
                        float(dl),
                        direction_sign=direction_sign,
                    )
                except Exception:
                    prev_overlap = 0.0 + 0.0j
                if np.abs(prev_overlap) > 1e-18:
                    phase_rot = np.exp(-1j * np.angle(prev_overlap))
                    for name in tuple(mode_components.keys()):
                        mode_components[name] = mode_components[name] * phase_rot
                prev_components_bwd = previous_projection.get("mode_components_bwd", {})
                try:
                    prev_overlap_bwd = _safe_modal_overlap_3d(
                        prev_components_bwd,
                        mode_components_bwd,
                        parts["axis"],
                        float(dl),
                        direction_sign=direction_sign,
                    )
                except Exception:
                    prev_overlap_bwd = 0.0 + 0.0j
                if np.abs(prev_overlap_bwd) > 1e-18:
                    phase_rot_bwd = np.exp(-1j * np.angle(prev_overlap_bwd))
                    for name in tuple(mode_components_bwd.keys()):
                        mode_components_bwd[name] = (
                            mode_components_bwd[name] * phase_rot_bwd
                        )

            # In 3D, keep the full tangential field set in the overlap system.
            # Collapsing back to a single dominant E/H pair makes the source-side
            # decomposition much more sensitive to local non-modal content.
            proj_components = (
                tuple(
                    parts.get(
                        "projection_components_3d", parts["projection_components"]
                    )
                )
                if self.is_3d
                else tuple(parts["projection_components"])
            )
            mode_components_proj = {
                name: np.asarray(mode_components[name], dtype=np.complex128)
                for name in proj_components
                if name in mode_components
            }
            mode_components_bwd_proj = {
                name: np.asarray(mode_components_bwd[name], dtype=np.complex128)
                for name in proj_components
                if name in mode_components_bwd
            }
            mode_components_proj = _normalize_3d_profiles_by_flux(
                mode_components_proj,
                axis=parts["axis"],
                d_area=float(dl),
                direction_sign=direction_sign,
            )
            mode_components_bwd_proj = _normalize_3d_profiles_by_flux(
                mode_components_bwd_proj,
                axis=parts["axis"],
                d_area=float(dl),
                direction_sign=direction_sign,
            )
            fwd_vec = np.concatenate([mode_components_proj[c] for c in proj_components])
            bwd_vec = np.concatenate(
                [mode_components_bwd_proj[c] for c in proj_components]
            )
            mode_matrix = np.column_stack([fwd_vec, bwd_vec])
            overlap_matrix = np.asarray(
                [
                    [
                        _safe_modal_overlap_3d(
                            mode_components_proj,
                            mode_components_proj,
                            parts["axis"],
                            float(dl),
                            direction_sign=direction_sign,
                        ),
                        _safe_modal_overlap_3d(
                            mode_components_proj,
                            mode_components_bwd_proj,
                            parts["axis"],
                            float(dl),
                            direction_sign=direction_sign,
                        ),
                    ],
                    [
                        _safe_modal_overlap_3d(
                            mode_components_bwd_proj,
                            mode_components_proj,
                            parts["axis"],
                            float(dl),
                            direction_sign=direction_sign,
                        ),
                        _safe_modal_overlap_3d(
                            mode_components_bwd_proj,
                            mode_components_bwd_proj,
                            parts["axis"],
                            float(dl),
                            direction_sign=direction_sign,
                        ),
                    ],
                ],
                dtype=np.complex128,
            )
            projection = {
                "e_component": parts["e_component"],
                "h_component": parts["h_component"],
                "components": tuple(proj_components),
                "mode_matrix": mode_matrix,
                "condition_number": float(np.linalg.cond(overlap_matrix)),
                "pinv": np.linalg.pinv(mode_matrix),
                "mode_neff": float(selected_fwd["mode_neff"]),
                "mode_neff_bwd": float(selected_bwd["mode_neff"]),
                "mode_components": {
                    name: np.asarray(mode_components_proj[name], dtype=np.complex128)
                    for name in proj_components
                    if name in mode_components_proj
                },
                "mode_components_bwd": {
                    name: np.asarray(
                        mode_components_bwd_proj[name], dtype=np.complex128
                    )
                    for name in proj_components
                    if name in mode_components_bwd_proj
                },
                "overlap_matrix": np.asarray(overlap_matrix, dtype=np.complex128),
                "axis": parts["axis"],
                "direction_sign": float(direction_sign),
                "d_area": float(dl),
                "power_norm": 1.0,
                "mode_parity": tuple(selected_fwd["parity_signature"]),
                "mode_parity_bwd": tuple(selected_bwd["parity_signature"]),
                "mode_component_grids": {
                    name: np.asarray(value, dtype=np.complex128)
                    for name, value in selected_fwd["component_grids"].items()
                },
                "mode_component_grids_bwd": {
                    name: np.asarray(value, dtype=np.complex128)
                    for name, value in selected_bwd["component_grids"].items()
                },
                "pair_score": float(best_pair_score),
            }
            projection["modal_plane_delay_s"] = self._modal_projection_plane_delay_s(
                spec,
                frequency,
                projection["mode_neff"],
            )
            if analysis_coords0 is not None and analysis_coords1 is not None:
                projection["analysis_coords0"] = np.asarray(
                    analysis_coords0, dtype=np.float64
                )
                projection["analysis_coords1"] = np.asarray(
                    analysis_coords1, dtype=np.float64
                )
            cache[key] = projection
            return projection

        mode = int(min(spec.mode_index, max(len(neff_vals) - 1, 0)))
        candidate = _candidate_record(
            mode, neff_vals, e_fields, h_fields, solver_direction
        )
        e_fwd = np.asarray(candidate["e_profile"], dtype=np.complex128)
        h_fwd = np.asarray(candidate["h_profile"], dtype=np.complex128)
        pm = 0.5 * np.real(
            np.sum(parts["signed_flux_sign"] * e_fwd * np.conjugate(h_fwd)) * dl
        )
        norm = np.sqrt(max(abs(pm), 1e-30))
        e_fwd = e_fwd / norm
        h_fwd = h_fwd / norm
        e_bwd = e_fwd.copy()
        h_bwd = -h_fwd.copy()
        mode_matrix = np.column_stack(
            [
                np.concatenate([e_fwd, h_fwd]),
                np.concatenate([e_bwd, h_bwd]),
            ]
        )
        projection = {
            "e_component": parts["e_component"],
            "h_component": parts["h_component"],
            "components": tuple(candidate["proj_components"]),
            "mode_matrix": mode_matrix,
            "condition_number": float(np.linalg.cond(mode_matrix)),
            "pinv": np.linalg.pinv(mode_matrix),
            "mode_neff": float(candidate["mode_neff"]),
        }
        projection["modal_plane_delay_s"] = self._modal_projection_plane_delay_s(
            spec,
            frequency,
            projection["mode_neff"],
        )
        cache[key] = projection
        return projection

    @staticmethod
    def _modal_power_3d(mode_components, axis, d_area):
        ex = np.asarray(
            mode_components.get("Ex", np.zeros((0,), dtype=np.complex128)),
            dtype=np.complex128,
        )
        ey = np.asarray(
            mode_components.get("Ey", np.zeros((0,), dtype=np.complex128)),
            dtype=np.complex128,
        )
        ez = np.asarray(
            mode_components.get("Ez", np.zeros((0,), dtype=np.complex128)),
            dtype=np.complex128,
        )
        hx = np.asarray(
            mode_components.get("Hx", np.zeros((0,), dtype=np.complex128)),
            dtype=np.complex128,
        )
        hy = np.asarray(
            mode_components.get("Hy", np.zeros((0,), dtype=np.complex128)),
            dtype=np.complex128,
        )
        hz = np.asarray(
            mode_components.get("Hz", np.zeros((0,), dtype=np.complex128)),
            dtype=np.complex128,
        )
        n = int(min(ex.size, ey.size, ez.size, hx.size, hy.size, hz.size))
        if n <= 0:
            return 0.0
        ex = ex[:n]
        ey = ey[:n]
        ez = ez[:n]
        hx = hx[:n]
        hy = hy[:n]
        hz = hz[:n]
        if axis == "x":
            s_axis = ey * np.conjugate(hz) - ez * np.conjugate(hy)
        elif axis == "y":
            s_axis = ez * np.conjugate(hx) - ex * np.conjugate(hz)
        else:
            s_axis = ex * np.conjugate(hy) - ey * np.conjugate(hx)
        return float(0.5 * np.real(np.sum(s_axis) * float(d_area)))

    @staticmethod
    def _modal_projection_reconstruction_residual(field_vec, projection, coeff):
        mode_matrix = np.asarray(
            projection.get("mode_matrix", np.zeros((0, 0), dtype=np.complex128)),
            dtype=np.complex128,
        )
        if (
            mode_matrix.ndim != 2
            or mode_matrix.shape[0] <= 0
            or mode_matrix.shape[1] < 2
        ):
            return np.nan
        field = np.asarray(field_vec, dtype=np.complex128).reshape(-1)
        if field.size <= 0:
            return np.nan
        n = int(min(field.size, mode_matrix.shape[0]))
        if n <= 0:
            return np.nan
        coeff_arr = np.asarray(coeff, dtype=np.complex128).reshape(-1)
        if coeff_arr.size < 2:
            return np.nan
        recon = mode_matrix[:n, :2] @ coeff_arr[:2]
        target = field[:n]
        denom = float(np.linalg.norm(target))
        if denom <= 1e-30 or not np.isfinite(denom):
            return np.nan
        return float(np.linalg.norm(target - recon) / denom)

    @staticmethod
    def _modal_projection_reconstruction_diagnostics_from_matrix(
        field_vec,
        mode_matrix,
        coeff,
        component_slices=(),
    ):
        matrix = np.asarray(mode_matrix, dtype=np.complex128)
        empty = {
            "residual": np.nan,
            "residual_e": np.nan,
            "residual_h": np.nan,
            "residual_balanced": np.nan,
            "e_scale": np.nan + 0.0j,
            "h_scale": np.nan + 0.0j,
        }
        if matrix.ndim != 2 or matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
            return empty
        field = np.asarray(field_vec, dtype=np.complex128).reshape(-1)
        coeff_arr = np.asarray(coeff, dtype=np.complex128).reshape(-1)
        n = int(min(field.size, matrix.shape[0]))
        m = int(min(coeff_arr.size, matrix.shape[1]))
        if n <= 0 or m <= 0:
            return empty
        target = field[:n]
        recon = matrix[:n, :m] @ coeff_arr[:m]

        def _residual(mask):
            if mask.size <= 0:
                return np.nan
            denom = float(np.linalg.norm(target[mask]))
            if denom <= 1e-30 or not np.isfinite(denom):
                return np.nan
            return float(np.linalg.norm(target[mask] - recon[mask]) / denom)

        def _scale_and_residual(mask):
            if mask.size <= 0:
                return np.nan + 0.0j, np.nan
            target_part = target[mask]
            recon_part = recon[mask]
            denom = np.vdot(recon_part, recon_part)
            if abs(denom) <= 1e-30 or not np.isfinite(abs(denom)):
                return np.nan + 0.0j, np.nan
            scale = np.vdot(recon_part, target_part) / denom
            target_norm = float(np.linalg.norm(target_part))
            if target_norm <= 1e-30 or not np.isfinite(target_norm):
                return scale, np.nan
            residual = float(
                np.linalg.norm(target_part - scale * recon_part) / target_norm
            )
            return np.complex128(scale), residual

        all_mask = np.arange(n, dtype=int)
        e_parts = []
        h_parts = []
        for name, start, stop in component_slices:
            lo = max(0, min(int(start), n))
            hi = max(lo, min(int(stop), n))
            if hi <= lo:
                continue
            part = np.arange(lo, hi, dtype=int)
            if str(name).startswith("E"):
                e_parts.append(part)
            elif str(name).startswith("H"):
                h_parts.append(part)
        e_mask = np.concatenate(e_parts) if e_parts else np.asarray([], dtype=int)
        h_mask = np.concatenate(h_parts) if h_parts else np.asarray([], dtype=int)

        e_scale, e_resid_scaled = _scale_and_residual(e_mask)
        h_scale, h_resid_scaled = _scale_and_residual(h_mask)
        balanced_recon = recon.copy()
        if e_mask.size and np.isfinite(abs(e_scale)):
            balanced_recon[e_mask] *= e_scale
        if h_mask.size and np.isfinite(abs(h_scale)):
            balanced_recon[h_mask] *= h_scale
        balanced_denom = float(np.linalg.norm(target))
        balanced = (
            float(np.linalg.norm(target - balanced_recon) / balanced_denom)
            if balanced_denom > 1e-30 and np.isfinite(balanced_denom)
            else np.nan
        )
        return {
            "residual": _residual(all_mask),
            "residual_e": _residual(e_mask),
            "residual_h": _residual(h_mask),
            "residual_balanced": balanced,
            "residual_e_scaled": e_resid_scaled,
            "residual_h_scaled": h_resid_scaled,
            "e_scale": e_scale,
            "h_scale": h_scale,
        }

    @staticmethod
    def _project_modal_coefficients_3d(
        field_components, projection, apply_calibration=True
    ):
        del apply_calibration
        mode_components = projection.get("mode_components", None)
        mode_components_bwd = projection.get("mode_components_bwd", None)
        overlap_matrix = projection.get("overlap_matrix", None)
        axis = str(projection.get("axis", "")).lower()
        d_area = float(projection.get("d_area", 1.0))
        direction_sign = float(projection.get("direction_sign", 1.0))
        if (
            isinstance(mode_components, dict)
            and isinstance(mode_components_bwd, dict)
            and overlap_matrix is not None
            and axis in {"x", "y", "z"}
        ):
            rhs = np.asarray(
                [
                    _safe_modal_overlap_3d(
                        field_components,
                        mode_components,
                        axis,
                        d_area,
                        direction_sign=direction_sign,
                    ),
                    _safe_modal_overlap_3d(
                        field_components,
                        mode_components_bwd,
                        axis,
                        d_area,
                        direction_sign=direction_sign,
                    ),
                ],
                dtype=np.complex128,
            )
            overlap = np.asarray(overlap_matrix, dtype=np.complex128)
            cond = float(np.linalg.cond(overlap))
            if (
                not np.all(np.isfinite(overlap))
                or not np.all(np.isfinite(rhs))
                or not np.isfinite(cond)
            ):
                raise ValueError("Invalid 3D modal overlap system.")
            system = overlap.T
            if cond < 1e8:
                coeff = np.linalg.solve(system, rhs)
            else:
                coeff = np.linalg.pinv(system) @ rhs
            return np.complex128(coeff[0]), np.complex128(coeff[1])

        components = tuple(projection.get("components", ()))
        if len(components) == 0:
            raise ValueError("3D projection missing component list.")

        vec_parts = []
        for comp in components:
            if comp not in field_components:
                raise ValueError(
                    f"Missing field component '{comp}' for 3D modal projection."
                )
            vec_parts.append(
                np.asarray(field_components[comp], dtype=np.complex128).reshape(-1)
            )
        field_vec = np.concatenate(vec_parts).astype(np.complex128, copy=False)

        pinv = np.asarray(
            projection.get("pinv", np.zeros((2, 0), dtype=np.complex128)),
            dtype=np.complex128,
        )
        if pinv.ndim != 2 or pinv.shape[0] < 2:
            raise ValueError("Invalid 3D projection pseudo-inverse shape.")
        n_expected = int(pinv.shape[1])
        if field_vec.size != n_expected:
            if field_vec.size > n_expected:
                field_vec = field_vec[:n_expected]
            else:
                field_vec = np.pad(field_vec, (0, n_expected - field_vec.size))
        coeff = pinv @ field_vec
        a_plus = coeff[0]
        a_minus = coeff[1]
        return np.complex128(a_plus), np.complex128(a_minus)

    @staticmethod
    def _project_modal_coefficients_3d_group(field_components, projections):
        """Project one 3D monitor field onto a coupled forward/backward mode set."""
        projections = tuple(projections)
        if not projections:
            return [], np.nan, np.nan, {}

        first = projections[0]
        components = tuple(first.get("components", ()))
        axis = str(first.get("axis", "")).lower()
        d_area = float(first.get("d_area", 1.0))
        direction_sign = float(first.get("direction_sign", 1.0))
        if len(components) == 0 or axis not in {"x", "y", "z"}:
            raise ValueError("3D modal group projection is missing components or axis.")

        basis = []
        for proj in projections:
            if tuple(proj.get("components", ())) != components:
                raise ValueError("Grouped 3D modal projections must share components.")
            if str(proj.get("axis", "")).lower() != axis:
                raise ValueError("Grouped 3D modal projections must share an axis.")
            basis.append(
                {
                    name: np.asarray(
                        proj.get("mode_components", {}).get(name, []),
                        dtype=np.complex128,
                    ).reshape(-1)
                    for name in components
                }
            )
            basis.append(
                {
                    name: np.asarray(
                        proj.get("mode_components_bwd", {}).get(name, []),
                        dtype=np.complex128,
                    ).reshape(-1)
                    for name in components
                }
            )

        rhs = np.asarray(
            [
                _safe_modal_overlap_3d(
                    field_components,
                    mode,
                    axis,
                    d_area,
                    direction_sign=direction_sign,
                )
                for mode in basis
            ],
            dtype=np.complex128,
        )
        overlap = np.asarray(
            [
                [
                    _safe_modal_overlap_3d(
                        basis_i,
                        basis_j,
                        axis,
                        d_area,
                        direction_sign=direction_sign,
                    )
                    for basis_j in basis
                ]
                for basis_i in basis
            ],
            dtype=np.complex128,
        )
        system = overlap.T
        cond = float(np.linalg.cond(system))
        if (
            not np.all(np.isfinite(system))
            or not np.all(np.isfinite(rhs))
            or not np.isfinite(cond)
        ):
            raise ValueError("Invalid grouped 3D modal overlap system.")
        if cond < 1e8:
            coeff = np.linalg.solve(system, rhs)
        else:
            coeff = np.linalg.pinv(system) @ rhs

        field_parts = [
            np.asarray(field_components[name], dtype=np.complex128).reshape(-1)
            for name in components
        ]
        component_slices = []
        offset = 0
        for name, part in zip(components, field_parts):
            next_offset = offset + int(part.size)
            component_slices.append((name, offset, next_offset))
            offset = next_offset
        field_vec = np.concatenate(field_parts)
        mode_matrix = np.column_stack(
            [
                np.concatenate(
                    [
                        np.asarray(mode[name], dtype=np.complex128).reshape(-1)
                        for name in components
                    ]
                )
                for mode in basis
            ]
        )
        diagnostics = (
            Simulation._modal_projection_reconstruction_diagnostics_from_matrix(
                field_vec,
                mode_matrix,
                coeff,
                component_slices=component_slices,
            )
        )
        residual = diagnostics["residual"]
        return (
            [
                (np.complex128(coeff[2 * idx]), np.complex128(coeff[2 * idx + 1]))
                for idx in range(len(projections))
            ],
            residual,
            cond,
            diagnostics,
        )

    def extract_port_waves(
        self,
        ports,
        frequencies,
        mode_strategy="per_frequency",
        window="hann",
        return_power=True,
    ):
        """Broadband modal extraction using FFT bins.

        Fast and convenient for sweeps, but less robust than CW demodulation
        for strict passivity/loss assessment.
        """
        if (not self.is_3d) and self.plane_2d != "xy":
            raise NotImplementedError(
                "extract_port_waves currently supports 2D simulations in the xy plane."
            )

        port_map = self._normalize_portspecs(ports)
        freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
        if freqs.size == 0:
            raise ValueError("frequencies must contain at least one value.")
        if np.any(freqs <= 0):
            raise ValueError("frequencies must be strictly positive.")

        strategy = str(mode_strategy).lower()
        if strategy not in {"per_frequency", "single", "single_frequency", "center"}:
            raise ValueError(
                f"Unsupported mode_strategy '{mode_strategy}'. "
                "Use 'per_frequency' or 'single'."
            )
        single_freq = float(np.median(freqs))

        monitor_by_name = self._named_monitors()
        for spec in port_map.values():
            if spec.monitor_name not in monitor_by_name:
                raise ValueError(
                    f"Missing monitor '{spec.monitor_name}' for port '{spec.name}'."
                )
            if spec.reference_monitor and spec.reference_monitor not in monitor_by_name:
                raise ValueError(
                    f"Missing reference monitor '{spec.reference_monitor}' for port '{spec.name}'."
                )

        spectrum_cache = {}
        projection_cache = {}
        sibling_projection_cache = {}
        sibling_reference_projection_cache = {}
        waves = {}

        for spec in port_map.values():
            main_monitor = monitor_by_name[spec.monitor_name]
            parts = self._mode_components_for_port(spec)
            sibling_seed_key = self._projection_seed_key(
                spec,
                main_monitor,
                parts,
                is_3d=self.is_3d,
            )
            wanted_components = (
                parts.get(
                    "projection_components_3d",
                    (parts["e_component"], parts["h_component"]),
                )
                if self.is_3d
                else (parts["e_component"], parts["h_component"])
            )
            for comp in wanted_components:
                key = (main_monitor.name, comp)
                if key not in spectrum_cache:
                    _, spectrum_cache[key] = self._sample_monitor_component_spectrum(
                        main_monitor, comp, frequencies=freqs, window=window
                    )

            a_plus = np.zeros(freqs.size, dtype=np.complex128)
            a_minus = np.zeros(freqs.size, dtype=np.complex128)
            last_valid_proj = None
            last_tracked_proj = None
            for idx, f in enumerate(freqs):
                f_mode = float(f if strategy == "per_frequency" else single_freq)
                seed_proj = last_tracked_proj
                if seed_proj is None and sibling_seed_key is not None:
                    seed_proj = sibling_projection_cache.get((idx, sibling_seed_key))
                proj = self._build_port_projection(
                    spec,
                    main_monitor,
                    f_mode,
                    projection_cache,
                    previous_projection=seed_proj,
                )
                proj_neff = float(proj.get("mode_neff", np.nan))
                if (not np.isfinite(proj_neff)) or (proj_neff <= 1e-6):
                    if last_valid_proj is not None:
                        proj = last_valid_proj
                else:
                    last_valid_proj = proj
                    last_tracked_proj = proj
                    if sibling_seed_key is not None:
                        sibling_projection_cache[(idx, sibling_seed_key)] = proj
                proj_components = tuple(
                    proj.get("components", (proj["e_component"], proj["h_component"]))
                )
                if self.is_3d:
                    raw_field_components = {
                        comp: self._apply_modal_projection_spatial_phase(
                            comp,
                            spectrum_cache[(main_monitor.name, comp)][idx],
                            f,
                            proj,
                        )
                        for comp in proj_components
                    }
                    field_components = self._colocate_field_components_to_projection_3d(
                        main_monitor,
                        raw_field_components,
                        proj,
                    )
                    coeff = self._project_modal_coefficients_3d(field_components, proj)
                    a_plus[idx], a_minus[idx] = coeff[0], coeff[1]
                else:
                    field_vec = np.concatenate(
                        [
                            self._apply_modal_projection_spatial_phase(
                                comp,
                                spectrum_cache[(main_monitor.name, comp)][idx],
                                f,
                                proj,
                            )
                            for comp in proj_components
                        ]
                    )
                    coeff = proj["pinv"] @ field_vec
                    a_plus[idx], a_minus[idx] = coeff[0], coeff[1]

            port_waves = {"a_plus": a_plus, "a_minus": a_minus}
            if return_power:
                port_waves["P_plus"] = np.abs(a_plus) ** 2
                port_waves["P_minus"] = np.abs(a_minus) ** 2

            if spec.reference_monitor:
                ref_monitor = monitor_by_name[spec.reference_monitor]
                sibling_ref_seed_key = self._projection_seed_key(
                    spec,
                    ref_monitor,
                    parts,
                    is_3d=self.is_3d,
                )
                for comp in wanted_components:
                    key = (ref_monitor.name, comp)
                    if key not in spectrum_cache:
                        _, spectrum_cache[key] = (
                            self._sample_monitor_component_spectrum(
                                ref_monitor, comp, frequencies=freqs, window=window
                            )
                        )

                a_incident_plus = np.zeros(freqs.size, dtype=np.complex128)
                a_incident_minus = np.zeros(freqs.size, dtype=np.complex128)
                last_valid_ref_proj = None
                last_tracked_ref_proj = None
                for idx, f in enumerate(freqs):
                    f_mode = float(f if strategy == "per_frequency" else single_freq)
                    ref_seed_proj = last_tracked_ref_proj
                    if ref_seed_proj is None and sibling_ref_seed_key is not None:
                        ref_seed_proj = sibling_reference_projection_cache.get(
                            (idx, sibling_ref_seed_key)
                        )
                    proj = self._build_port_projection(
                        spec,
                        ref_monitor,
                        f_mode,
                        projection_cache,
                        previous_projection=ref_seed_proj,
                    )
                    proj_neff = float(proj.get("mode_neff", np.nan))
                    if (not np.isfinite(proj_neff)) or (proj_neff <= 1e-6):
                        if last_valid_ref_proj is not None:
                            proj = last_valid_ref_proj
                    else:
                        last_valid_ref_proj = proj
                        last_tracked_ref_proj = proj
                        if sibling_ref_seed_key is not None:
                            sibling_reference_projection_cache[
                                (idx, sibling_ref_seed_key)
                            ] = proj
                    proj_components = tuple(
                        proj.get(
                            "components", (proj["e_component"], proj["h_component"])
                        )
                    )
                    if self.is_3d:
                        raw_field_components = {
                            comp: self._apply_modal_projection_spatial_phase(
                                comp,
                                spectrum_cache[(ref_monitor.name, comp)][idx],
                                f,
                                proj,
                            )
                            for comp in proj_components
                        }
                        field_components = (
                            self._colocate_field_components_to_projection_3d(
                                ref_monitor,
                                raw_field_components,
                                proj,
                            )
                        )
                        coeff = self._project_modal_coefficients_3d(
                            field_components, proj
                        )
                        a_incident_plus[idx], a_incident_minus[idx] = coeff[0], coeff[1]
                    else:
                        field_vec = np.concatenate(
                            [
                                self._apply_modal_projection_spatial_phase(
                                    comp,
                                    spectrum_cache[(ref_monitor.name, comp)][idx],
                                    f,
                                    proj,
                                )
                                for comp in proj_components
                            ]
                        )
                        coeff = proj["pinv"] @ field_vec
                        a_incident_plus[idx], a_incident_minus[idx] = coeff[0], coeff[1]
                port_waves["a_incident"] = a_incident_plus
                port_waves["a_incident_plus"] = a_incident_plus
                port_waves["a_incident_minus"] = a_incident_minus
                if return_power:
                    port_waves["P_incident"] = np.abs(a_incident_plus) ** 2
                    port_waves["P_incident_plus"] = np.abs(a_incident_plus) ** 2
                    port_waves["P_incident_minus"] = np.abs(a_incident_minus) ** 2

            waves[spec.name] = port_waves
        return waves

    def extract_port_waves_dft(
        self,
        ports,
        frequencies,
        min_incident_db=-40.0,
        return_power=True,
        mode_strategy="per_frequency",
    ):
        """Extract modal port waves from in-simulation DFT monitor accumulators."""
        del min_incident_db  # Used in get_S_matrix_modal_dft validity masking.
        if (not self.is_3d) and self.plane_2d != "xy":
            raise NotImplementedError(
                "extract_port_waves_dft currently supports 2D simulations in the xy plane."
            )

        port_map = self._normalize_portspecs(ports)
        freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
        if freqs.size == 0:
            raise ValueError("frequencies must contain at least one value.")
        if np.any(freqs <= 0):
            raise ValueError("frequencies must be strictly positive.")
        strategy = str(mode_strategy).lower()
        if strategy not in {"per_frequency", "single", "single_frequency", "center"}:
            raise ValueError(
                f"Unsupported mode_strategy '{mode_strategy}'. "
                "Use 'per_frequency' or 'single'."
            )
        single_freq = float(np.median(freqs))

        monitor_by_name = self._named_monitors()
        for spec in port_map.values():
            main = monitor_by_name.get(spec.monitor_name)
            if main is None:
                raise ValueError(
                    f"Missing monitor '{spec.monitor_name}' for port '{spec.name}'."
                )
            if not getattr(main, "dft_enabled", False):
                raise ValueError(
                    f"Monitor '{spec.monitor_name}' must be created with dft_enabled=True."
                )
            if spec.reference_monitor:
                ref = monitor_by_name.get(spec.reference_monitor)
                if ref is None:
                    raise ValueError(
                        f"Missing reference monitor '{spec.reference_monitor}' for port '{spec.name}'."
                    )
                if not getattr(ref, "dft_enabled", False):
                    raise ValueError(
                        f"Reference monitor '{spec.reference_monitor}' must have dft_enabled=True."
                    )

        dft_cache = {}
        projection_cache = {}
        sibling_projection_cache = {}
        sibling_reference_projection_cache = {}
        waves = {}
        group_projection_history = {}

        def _matching_3d_group_specs(spec, monitor_name, *, reference):
            if not self.is_3d:
                return (spec,)
            if reference:
                candidates = [
                    candidate
                    for candidate in port_map.values()
                    if candidate.reference_monitor == monitor_name
                ]
            else:
                candidates = [
                    candidate
                    for candidate in port_map.values()
                    if candidate.monitor_name == monitor_name
                ]
            group = [
                candidate
                for candidate in candidates
                if candidate.direction == spec.direction
                and candidate.polarization == spec.polarization
            ]
            if not group:
                return (spec,)
            return tuple(
                sorted(group, key=lambda item: (int(item.mode_index), item.name))
            )

        def _build_3d_group_projection(spec, monitor, f_mode):
            hist_key = (spec.name, monitor.name)
            previous = group_projection_history.get(hist_key)
            if previous is None:
                proj = self._build_port_projection(
                    spec,
                    monitor,
                    f_mode,
                    projection_cache,
                )
            else:
                proj = self._build_port_projection(
                    spec,
                    monitor,
                    f_mode,
                    projection_cache,
                    previous_projection=previous,
                )
            proj_neff = float(proj.get("mode_neff", np.nan))
            if (not np.isfinite(proj_neff)) or (proj_neff <= 1e-6):
                if previous is not None:
                    return previous
            else:
                group_projection_history[hist_key] = proj
            return proj

        def _project_3d_group_at_monitor(
            spec,
            monitor,
            idx,
            frequency,
            f_mode,
            *,
            reference,
        ):
            group_specs = _matching_3d_group_specs(
                spec,
                monitor.name,
                reference=reference,
            )
            projections = [
                _build_3d_group_projection(group_spec, monitor, f_mode)
                for group_spec in group_specs
            ]
            try:
                group_index = next(
                    index
                    for index, group_spec in enumerate(group_specs)
                    if group_spec.name == spec.name
                )
            except StopIteration:
                group_index = 0
            proj = projections[group_index]
            proj_components = tuple(
                proj.get("components", (proj["e_component"], proj["h_component"]))
            )
            raw_field_components = {
                comp: self._apply_modal_projection_spatial_phase(
                    comp,
                    dft_cache[(monitor.name, comp)][idx],
                    frequency,
                    proj,
                )
                for comp in proj_components
            }
            field_components = self._colocate_field_components_to_projection_3d(
                monitor,
                raw_field_components,
                proj,
            )
            coeffs, residual, group_cond, projection_diag = (
                self._project_modal_coefficients_3d_group(
                    field_components,
                    projections,
                )
            )
            cond_values = [
                float(projection.get("condition_number", np.nan))
                for projection in projections
            ]
            finite_conds = [
                value
                for value in [float(group_cond), *cond_values]
                if np.isfinite(value)
            ]
            cond = max(finite_conds) if finite_conds else np.nan
            return (
                coeffs[group_index],
                residual,
                cond,
                float(proj.get("mode_neff", np.nan)),
                projection_diag,
            )

        for spec in port_map.values():
            parts = self._mode_components_for_port(spec)
            main_monitor = monitor_by_name[spec.monitor_name]
            sibling_seed_key = self._projection_seed_key(
                spec,
                main_monitor,
                parts,
                is_3d=self.is_3d,
            )
            wanted_components = (
                parts.get(
                    "projection_components_3d",
                    (parts["e_component"], parts["h_component"]),
                )
                if self.is_3d
                else (parts["e_component"], parts["h_component"])
            )
            for comp in wanted_components:
                key = (main_monitor.name, comp)
                if key not in dft_cache:
                    _, dft_cache[key] = self._sample_monitor_component_dft(
                        main_monitor, comp, frequencies=freqs
                    )

            a_plus = np.zeros(freqs.size, dtype=np.complex128)
            a_minus = np.zeros(freqs.size, dtype=np.complex128)
            cond_main = np.zeros(freqs.size, dtype=float)
            neff_main = np.full(freqs.size, np.nan, dtype=float)
            residual_main = np.full(freqs.size, np.nan, dtype=float)
            residual_e_main = np.full(freqs.size, np.nan, dtype=float)
            residual_h_main = np.full(freqs.size, np.nan, dtype=float)
            residual_balanced_main = np.full(freqs.size, np.nan, dtype=float)
            e_scale_main = np.full(freqs.size, np.nan + 0.0j, dtype=np.complex128)
            h_scale_main = np.full(freqs.size, np.nan + 0.0j, dtype=np.complex128)
            last_valid_proj = None
            last_tracked_proj = None
            for idx, f in enumerate(freqs):
                f_mode = float(f if strategy == "per_frequency" else single_freq)
                seed_proj = last_tracked_proj
                if seed_proj is None and sibling_seed_key is not None:
                    seed_proj = sibling_projection_cache.get((idx, sibling_seed_key))
                if seed_proj is None:
                    proj = self._build_port_projection(
                        spec,
                        main_monitor,
                        f_mode,
                        projection_cache,
                    )
                else:
                    proj = self._build_port_projection(
                        spec,
                        main_monitor,
                        f_mode,
                        projection_cache,
                        previous_projection=seed_proj,
                    )
                proj_neff = float(proj.get("mode_neff", np.nan))
                if (not np.isfinite(proj_neff)) or (proj_neff <= 1e-6):
                    if last_valid_proj is not None:
                        proj = last_valid_proj
                else:
                    last_valid_proj = proj
                    last_tracked_proj = proj
                    if sibling_seed_key is not None:
                        sibling_projection_cache[(idx, sibling_seed_key)] = proj
                proj_components = tuple(
                    proj.get("components", (proj["e_component"], proj["h_component"]))
                )
                if self.is_3d:
                    coeff, residual, cond, neff, projection_diag = (
                        _project_3d_group_at_monitor(
                            spec,
                            main_monitor,
                            idx,
                            f,
                            f_mode,
                            reference=False,
                        )
                    )
                    a_plus[idx], a_minus[idx] = coeff[0], coeff[1]
                    residual_main[idx] = residual
                    cond_main[idx] = cond
                    neff_main[idx] = neff
                    residual_e_main[idx] = float(
                        projection_diag.get("residual_e", np.nan)
                    )
                    residual_h_main[idx] = float(
                        projection_diag.get("residual_h", np.nan)
                    )
                    residual_balanced_main[idx] = float(
                        projection_diag.get("residual_balanced", np.nan)
                    )
                    e_scale_main[idx] = np.complex128(
                        projection_diag.get("e_scale", np.nan + 0.0j)
                    )
                    h_scale_main[idx] = np.complex128(
                        projection_diag.get("h_scale", np.nan + 0.0j)
                    )
                else:
                    field_vec = np.concatenate(
                        [
                            self._apply_modal_projection_spatial_phase(
                                comp,
                                dft_cache[(main_monitor.name, comp)][idx],
                                f,
                                proj,
                            )
                            for comp in proj_components
                        ]
                    )
                    coeff = proj["pinv"] @ field_vec
                    a_plus[idx], a_minus[idx] = coeff[0], coeff[1]
                    residual_main[idx] = self._modal_projection_reconstruction_residual(
                        field_vec,
                        proj,
                        coeff,
                    )
                    cond_main[idx] = float(proj.get("condition_number", np.nan))
                    neff_main[idx] = float(proj.get("mode_neff", np.nan))

            port_waves = {
                "a_plus": a_plus,
                "a_minus": a_minus,
                "condition_number": cond_main,
                "mode_neff": neff_main,
                "projection_residual": residual_main,
                "projection_residual_e": residual_e_main,
                "projection_residual_h": residual_h_main,
                "projection_residual_balanced": residual_balanced_main,
                "projection_e_scale": e_scale_main,
                "projection_h_scale": h_scale_main,
            }
            if return_power:
                port_waves["P_plus"] = np.abs(a_plus) ** 2
                port_waves["P_minus"] = np.abs(a_minus) ** 2

            if spec.reference_monitor:
                ref_monitor = monitor_by_name[spec.reference_monitor]
                sibling_ref_seed_key = self._projection_seed_key(
                    spec,
                    ref_monitor,
                    parts,
                    is_3d=self.is_3d,
                )
                for comp in wanted_components:
                    key = (ref_monitor.name, comp)
                    if key not in dft_cache:
                        _, dft_cache[key] = self._sample_monitor_component_dft(
                            ref_monitor, comp, frequencies=freqs
                        )
                a_incident_plus = np.zeros(freqs.size, dtype=np.complex128)
                a_incident_minus = np.zeros(freqs.size, dtype=np.complex128)
                cond_ref = np.zeros(freqs.size, dtype=float)
                neff_ref = np.full(freqs.size, np.nan, dtype=float)
                residual_ref = np.full(freqs.size, np.nan, dtype=float)
                residual_e_ref = np.full(freqs.size, np.nan, dtype=float)
                residual_h_ref = np.full(freqs.size, np.nan, dtype=float)
                residual_balanced_ref = np.full(freqs.size, np.nan, dtype=float)
                e_scale_ref = np.full(freqs.size, np.nan + 0.0j, dtype=np.complex128)
                h_scale_ref = np.full(freqs.size, np.nan + 0.0j, dtype=np.complex128)
                last_valid_ref_proj = None
                last_tracked_ref_proj = None
                for idx, f in enumerate(freqs):
                    f_mode = float(f if strategy == "per_frequency" else single_freq)
                    ref_seed_proj = last_tracked_ref_proj
                    if ref_seed_proj is None and sibling_ref_seed_key is not None:
                        ref_seed_proj = sibling_reference_projection_cache.get(
                            (idx, sibling_ref_seed_key)
                        )
                    if ref_seed_proj is None:
                        proj = self._build_port_projection(
                            spec,
                            ref_monitor,
                            f_mode,
                            projection_cache,
                        )
                    else:
                        proj = self._build_port_projection(
                            spec,
                            ref_monitor,
                            f_mode,
                            projection_cache,
                            previous_projection=ref_seed_proj,
                        )
                    proj_neff = float(proj.get("mode_neff", np.nan))
                    if (not np.isfinite(proj_neff)) or (proj_neff <= 1e-6):
                        if last_valid_ref_proj is not None:
                            proj = last_valid_ref_proj
                    else:
                        last_valid_ref_proj = proj
                        last_tracked_ref_proj = proj
                        if sibling_ref_seed_key is not None:
                            sibling_reference_projection_cache[
                                (idx, sibling_ref_seed_key)
                            ] = proj
                    proj_components = tuple(
                        proj.get(
                            "components", (proj["e_component"], proj["h_component"])
                        )
                    )
                    if self.is_3d:
                        coeff, residual, cond, neff, projection_diag = (
                            _project_3d_group_at_monitor(
                                spec,
                                ref_monitor,
                                idx,
                                f,
                                f_mode,
                                reference=True,
                            )
                        )
                        a_incident_plus[idx], a_incident_minus[idx] = coeff[0], coeff[1]
                        residual_ref[idx] = residual
                        cond_ref[idx] = cond
                        neff_ref[idx] = neff
                        residual_e_ref[idx] = float(
                            projection_diag.get("residual_e", np.nan)
                        )
                        residual_h_ref[idx] = float(
                            projection_diag.get("residual_h", np.nan)
                        )
                        residual_balanced_ref[idx] = float(
                            projection_diag.get("residual_balanced", np.nan)
                        )
                        e_scale_ref[idx] = np.complex128(
                            projection_diag.get("e_scale", np.nan + 0.0j)
                        )
                        h_scale_ref[idx] = np.complex128(
                            projection_diag.get("h_scale", np.nan + 0.0j)
                        )
                    else:
                        field_vec = np.concatenate(
                            [
                                self._apply_modal_projection_spatial_phase(
                                    comp,
                                    dft_cache[(ref_monitor.name, comp)][idx],
                                    f,
                                    proj,
                                )
                                for comp in proj_components
                            ]
                        )
                        coeff = proj["pinv"] @ field_vec
                        a_incident_plus[idx], a_incident_minus[idx] = coeff[0], coeff[1]
                        residual_ref[idx] = (
                            self._modal_projection_reconstruction_residual(
                                field_vec,
                                proj,
                                coeff,
                            )
                        )
                        cond_ref[idx] = float(proj.get("condition_number", np.nan))
                        neff_ref[idx] = float(proj.get("mode_neff", np.nan))
                port_waves["a_incident"] = a_incident_plus
                port_waves["a_incident_plus"] = a_incident_plus
                port_waves["a_incident_minus"] = a_incident_minus
                port_waves["reference_condition_number"] = cond_ref
                port_waves["reference_mode_neff"] = neff_ref
                port_waves["reference_projection_residual"] = residual_ref
                port_waves["reference_projection_residual_e"] = residual_e_ref
                port_waves["reference_projection_residual_h"] = residual_h_ref
                port_waves["reference_projection_residual_balanced"] = (
                    residual_balanced_ref
                )
                port_waves["reference_projection_e_scale"] = e_scale_ref
                port_waves["reference_projection_h_scale"] = h_scale_ref
                if return_power:
                    port_waves["P_incident"] = np.abs(a_incident_plus) ** 2
                    port_waves["P_incident_plus"] = np.abs(a_incident_plus) ** 2
                    port_waves["P_incident_minus"] = np.abs(a_incident_minus) ** 2

            waves[spec.name] = port_waves
        return waves

    def get_S_matrix_modal_dft(
        self,
        source_port,
        ports,
        output_ports=None,
        frequencies=None,
        as_sax=True,
        return_diagnostics=True,
        min_incident_db=-40.0,
        mode_strategy="per_frequency",
    ):
        """Broadband modal S extraction from in-simulation DFT monitor accumulators."""
        port_map = self._normalize_portspecs(ports)
        source_port = self._port_name(source_port)
        if source_port not in port_map:
            raise ValueError(f"source_port '{source_port}' not found in ports.")

        monitor_by_name = self._named_monitors()
        if frequencies is None:
            src_spec = port_map[source_port]
            ref_name = src_spec.reference_monitor or src_spec.monitor_name
            src_monitor = monitor_by_name.get(ref_name)
            if src_monitor is None:
                raise ValueError(f"Missing source/reference monitor '{ref_name}'.")
            frequencies = src_monitor.get_dft_frequencies()
        frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))

        waves = self.extract_port_waves_dft(
            ports=port_map.values(),
            frequencies=frequencies,
            min_incident_db=min_incident_db,
            return_power=True,
            mode_strategy=mode_strategy,
        )

        output_ports = self._normalize_output_port_names(output_ports, port_map)

        source_spec = port_map[source_port]
        source_incident_selector, source_scattered_selector = (
            self._resolve_port_wave_selectors(
                source_spec,
                waves[source_port],
                use_reference=bool(source_spec.reference_monitor),
            )
        )
        a_incident = self._select_wave_component(
            waves[source_port],
            selector=source_incident_selector,
            use_reference=bool(source_spec.reference_monitor),
        )
        a_incident = np.asarray(a_incident, dtype=np.complex128)
        max_incident = float(np.max(np.abs(a_incident))) if a_incident.size else 0.0
        rel_floor = max_incident * (10.0 ** (float(min_incident_db) / 20.0))
        abs_floor = max(1e-18, rel_floor)
        valid_mask = np.abs(a_incident) >= abs_floor

        scattered_waves = {}
        s_matrix = {}
        for out_port in output_ports:
            out_spec = port_map[out_port]
            _, out_scattered_selector = self._resolve_port_wave_selectors(
                out_spec,
                waves[out_port],
                use_reference=False,
            )
            b_out = self._select_wave_component(
                waves[out_port],
                selector=out_scattered_selector,
                use_reference=False,
            )
            b_out = np.asarray(b_out, dtype=np.complex128)
            scattered_waves[out_port] = b_out
            ratio = self._safe_ratio(b_out, a_incident)
            ratio = np.where(valid_mask, ratio, 0.0 + 0.0j)
            s_matrix[(out_port, source_port)] = ratio

        self.s_matrix_frequencies = np.asarray(frequencies, dtype=float)
        s_output = self._format_s_matrix_output(s_matrix, as_sax=as_sax)

        if not return_diagnostics:
            return s_output

        p_in = np.abs(a_incident) ** 2
        p_guided_out = np.zeros_like(p_in, dtype=float)
        for out_port in output_ports:
            p_guided_out += np.abs(scattered_waves[out_port]) ** 2
        power_sum = p_guided_out / np.maximum(p_in, 1e-18)
        loss_est = 1.0 - power_sum
        power_sum = np.where(valid_mask, power_sum, np.nan)
        loss_est = np.where(valid_mask, loss_est, np.nan)

        diagnostics = {
            "frequencies": np.asarray(frequencies, dtype=float),
            "source_port": source_port,
            "output_ports": output_ports,
            "mode_strategy": str(mode_strategy).lower(),
            "waves": waves,
            "P_in": p_in,
            "P_guided_out": p_guided_out,
            "power_sum": power_sum,
            "loss_est": loss_est,
            "valid_mask": valid_mask,
            "condition_numbers": {
                name: {
                    "monitor": np.asarray(
                        data.get("condition_number", []), dtype=float
                    ),
                    "reference": np.asarray(
                        data.get("reference_condition_number", []), dtype=float
                    ),
                }
                for name, data in waves.items()
            },
            "source_reference_normalization": {
                "enabled": bool(source_spec.reference_monitor),
                "monitor": source_spec.reference_monitor,
                "incident_wave": source_incident_selector,
                "scattered_wave": source_scattered_selector,
            },
            "monitor_flux_checks": self._modal_dft_flux_diagnostics(
                port_map, monitor_by_name, waves, frequencies
            ),
            "scattered_waves": scattered_waves,
        }
        return {"s_matrix": s_output, "diagnostics": diagnostics}

    @staticmethod
    def _resample_real_vector(freq_src, values_src, freq_dst):
        freq_src = np.atleast_1d(np.asarray(freq_src, dtype=float))
        values = np.atleast_1d(np.asarray(values_src, dtype=float))
        freq_dst = np.atleast_1d(np.asarray(freq_dst, dtype=float))
        if freq_src.size == 0 or values.size == 0:
            return np.full(freq_dst.shape, np.nan, dtype=float)
        n = min(freq_src.size, values.size)
        freq_src = freq_src[:n]
        values = values[:n]
        if n == freq_dst.size and np.allclose(freq_src, freq_dst, rtol=1e-9, atol=0.0):
            return values.astype(float, copy=True)
        return np.interp(freq_dst, freq_src, values, left=np.nan, right=np.nan)

    def _modal_dft_flux_diagnostics(
        self, port_map, monitor_by_name, waves, frequencies
    ):
        """Compare raw DFT flux monitors with selector-aware modal overlap power."""

        freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
        diagnostics = {}
        for name, spec in port_map.items():
            wave = waves.get(name, {})

            def _wave_power(power_key, amplitude_key):
                if power_key in wave:
                    power = np.asarray(wave[power_key], dtype=float)
                elif amplitude_key in wave:
                    amplitude = np.asarray(wave[amplitude_key], dtype=np.complex128)
                    power = np.abs(amplitude) ** 2
                else:
                    power = np.asarray([], dtype=float)
                if power.size != freqs.size:
                    return np.full(freqs.shape, np.nan, dtype=float)
                return power

            p_plus = _wave_power("P_plus", "a_plus")
            p_minus = _wave_power("P_minus", "a_minus")
            modal_sum = p_plus + p_minus
            modal_net = p_plus - p_minus

            incident_selector, scattered_selector = self._resolve_port_wave_selectors(
                spec,
                wave,
                use_reference=bool(spec.reference_monitor),
            )
            if scattered_selector == "plus":
                selected_power = p_plus
                rejected_power = p_minus
            else:
                selected_power = p_minus
                rejected_power = p_plus
            selected_modal_net = selected_power - rejected_power

            def _flux_for_monitor(monitor_name):
                monitor = monitor_by_name.get(monitor_name)
                if monitor is None or not hasattr(monitor, "get_dft_flux"):
                    return np.full(freqs.shape, np.nan, dtype=float)
                try:
                    flux = monitor.get_dft_flux()
                    mon_freqs = monitor.get_dft_frequencies()
                except ValueError:
                    return np.full(freqs.shape, np.nan, dtype=float)
                return self._resample_real_vector(mon_freqs, flux, freqs)

            monitor_flux = _flux_for_monitor(spec.monitor_name)
            entry = {
                "monitor": spec.monitor_name,
                "monitor_flux": monitor_flux,
                "incident_wave": incident_selector,
                "scattered_wave": scattered_selector,
                "P_plus": p_plus,
                "P_minus": p_minus,
                "P_modal_sum": modal_sum,
                "P_modal_net": modal_net,
                "P_selected": selected_power,
                "P_rejected": rejected_power,
                "P_selected_modal_net": selected_modal_net,
                "flux_minus_modal_net": monitor_flux - modal_net,
                "flux_minus_selected_modal_net": monitor_flux - selected_modal_net,
                "abs_flux_minus_modal_sum": np.abs(monitor_flux) - modal_sum,
            }
            if spec.reference_monitor:
                reference_flux = _flux_for_monitor(spec.reference_monitor)
                entry["reference_monitor"] = spec.reference_monitor
                entry["reference_monitor_flux"] = reference_flux
            diagnostics[name] = entry
        return diagnostics

    def extract_port_waves_cw(
        self,
        ports,
        frequency,
        steady_start_time=None,
        avg_cycles=12,
        window="hann",
        mode_strategy="per_frequency",
        return_power=True,
    ):
        """CW modal extraction at one frequency using complex demodulation."""
        if self.is_3d or self.plane_2d != "xy":
            raise NotImplementedError(
                "extract_port_waves_cw currently supports 2D simulations in the xy plane."
            )

        port_map = self._normalize_portspecs(ports)
        f = float(frequency)
        if not np.isfinite(f) or f <= 0:
            raise ValueError(f"frequency must be positive, got {frequency!r}")

        strategy = str(mode_strategy).lower()
        if strategy not in {"per_frequency", "single", "single_frequency", "center"}:
            raise ValueError(
                f"Unsupported mode_strategy '{mode_strategy}'. "
                "Use 'per_frequency' or 'single'."
            )
        f_mode = f

        monitor_by_name = self._named_monitors()
        for spec in port_map.values():
            if spec.monitor_name not in monitor_by_name:
                raise ValueError(
                    f"Missing monitor '{spec.monitor_name}' for port '{spec.name}'."
                )
            if spec.reference_monitor and spec.reference_monitor not in monitor_by_name:
                raise ValueError(
                    f"Missing reference monitor '{spec.reference_monitor}' for port '{spec.name}'."
                )

        projection_cache = {}
        waves = {}
        for spec in port_map.values():
            parts = self._mode_components_for_port(spec)
            main_monitor = monitor_by_name[spec.monitor_name]
            proj = self._build_port_projection(
                spec,
                main_monitor,
                f_mode if strategy == "per_frequency" else f,
                projection_cache,
            )
            e_main = self._demodulate_monitor_component(
                main_monitor,
                parts["e_component"],
                frequency=f,
                t_start=steady_start_time,
                avg_cycles=avg_cycles,
                window=window,
            )
            h_main = self._demodulate_monitor_component(
                main_monitor,
                parts["h_component"],
                frequency=f,
                t_start=steady_start_time,
                avg_cycles=avg_cycles,
                window=window,
            )
            e_main = self._apply_modal_projection_spatial_phase(
                parts["e_component"], e_main, f, proj
            )
            h_main = self._apply_modal_projection_spatial_phase(
                parts["h_component"], h_main, f, proj
            )
            coeff = proj["pinv"] @ np.concatenate([e_main, h_main])
            a_plus = np.complex128(coeff[0])
            a_minus = np.complex128(coeff[1])
            port_waves = {"a_plus": a_plus, "a_minus": a_minus}
            if return_power:
                port_waves["P_plus"] = float(np.abs(a_plus) ** 2)
                port_waves["P_minus"] = float(np.abs(a_minus) ** 2)

            if spec.reference_monitor:
                ref_monitor = monitor_by_name[spec.reference_monitor]
                ref_proj = self._build_port_projection(
                    spec,
                    ref_monitor,
                    f_mode if strategy == "per_frequency" else f,
                    projection_cache,
                )
                e_ref = self._demodulate_monitor_component(
                    ref_monitor,
                    parts["e_component"],
                    frequency=f,
                    t_start=steady_start_time,
                    avg_cycles=avg_cycles,
                    window=window,
                )
                h_ref = self._demodulate_monitor_component(
                    ref_monitor,
                    parts["h_component"],
                    frequency=f,
                    t_start=steady_start_time,
                    avg_cycles=avg_cycles,
                    window=window,
                )
                e_ref = self._apply_modal_projection_spatial_phase(
                    parts["e_component"], e_ref, f, ref_proj
                )
                h_ref = self._apply_modal_projection_spatial_phase(
                    parts["h_component"], h_ref, f, ref_proj
                )
                ref_coeff = ref_proj["pinv"] @ np.concatenate([e_ref, h_ref])
                a_incident_plus = np.complex128(ref_coeff[0])
                a_incident_minus = np.complex128(ref_coeff[1])
                port_waves["a_incident"] = a_incident_plus
                port_waves["a_incident_plus"] = a_incident_plus
                port_waves["a_incident_minus"] = a_incident_minus
                if return_power:
                    port_waves["P_incident"] = float(np.abs(a_incident_plus) ** 2)
                    port_waves["P_incident_plus"] = float(np.abs(a_incident_plus) ** 2)
                    port_waves["P_incident_minus"] = float(
                        np.abs(a_incident_minus) ** 2
                    )

            waves[spec.name] = port_waves
        return waves

    def get_S_matrix_modal(
        self,
        source_port,
        ports,
        output_ports=None,
        frequencies=None,
        mode_strategy="per_frequency",
        as_sax=True,
        return_diagnostics=True,
    ):
        """Broadband modal S-matrix extraction from FFT-sampled monitor spectra.

        This method is fast and useful for exploratory sweeps. For strict
        passivity/loss checks, prefer get_S_matrix_modal_cw(...).
        """
        port_map = self._normalize_portspecs(ports)
        source_port = self._port_name(source_port)
        if source_port not in port_map:
            raise ValueError(f"source_port '{source_port}' not found in ports.")

        monitor_by_name = self._named_monitors()
        if frequencies is None:
            src_spec = port_map[source_port]
            ref_name = src_spec.reference_monitor or src_spec.monitor_name
            src_monitor = monitor_by_name.get(ref_name)
            if src_monitor is None:
                raise ValueError(f"Missing source/reference monitor '{ref_name}'.")
            src_parts = self._mode_components_for_port(src_spec)
            frequencies, _ = self._sample_monitor_component_spectrum(
                src_monitor, src_parts["e_component"], frequencies=None, window="hann"
            )
        else:
            frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))

        waves = self.extract_port_waves(
            ports=port_map.values(),
            frequencies=frequencies,
            mode_strategy=mode_strategy,
            window="hann",
            return_power=True,
        )

        output_ports = self._normalize_output_port_names(output_ports, port_map)

        source_spec = port_map[source_port]
        source_incident_selector, _source_scattered_selector = (
            self._resolve_port_wave_selectors(
                source_spec,
                waves[source_port],
                use_reference=bool(source_spec.reference_monitor),
            )
        )
        a_incident = self._select_wave_component(
            waves[source_port],
            selector=source_incident_selector,
            use_reference=bool(source_spec.reference_monitor),
        )
        s_matrix = {}
        for out_port in output_ports:
            out_spec = port_map[out_port]
            _, out_scattered_selector = self._resolve_port_wave_selectors(
                out_spec,
                waves[out_port],
                use_reference=False,
            )
            b_out = self._select_wave_component(
                waves[out_port],
                selector=out_scattered_selector,
                use_reference=False,
            )
            s_matrix[(out_port, source_port)] = self._safe_ratio(b_out, a_incident)

        self.s_matrix_frequencies = np.asarray(frequencies, dtype=float)
        s_output = self._format_s_matrix_output(s_matrix, as_sax=as_sax)

        if not return_diagnostics:
            return s_output

        p_in = np.abs(a_incident) ** 2
        p_guided_out = np.zeros_like(p_in, dtype=float)
        for out_port in output_ports:
            out_spec = port_map[out_port]
            _, out_scattered_selector = self._resolve_port_wave_selectors(
                out_spec,
                waves[out_port],
                use_reference=False,
            )
            p_guided_out += (
                np.abs(
                    self._select_wave_component(
                        waves[out_port],
                        selector=out_scattered_selector,
                        use_reference=False,
                    )
                )
                ** 2
            )
        power_sum = p_guided_out / np.maximum(p_in, 1e-18)
        diagnostics = {
            "frequencies": np.asarray(frequencies, dtype=float),
            "source_port": source_port,
            "output_ports": output_ports,
            "waves": waves,
            "P_in": p_in,
            "P_guided_out": p_guided_out,
            "power_sum": power_sum,
            "loss_est": 1.0 - power_sum,
            "source_reference_normalization": {
                "enabled": bool(source_spec.reference_monitor),
                "monitor": source_spec.reference_monitor,
                "incident_wave": source_incident_selector,
                "scattered_wave": _source_scattered_selector,
            },
        }
        return {"s_matrix": s_output, "diagnostics": diagnostics}

    def get_S_matrix_modal_cw(
        self,
        source_port,
        ports,
        output_ports=None,
        frequency=None,
        steady_start_time=None,
        avg_cycles=12,
        window="hann",
        mode_strategy="per_frequency",
        as_sax=True,
        return_diagnostics=True,
    ):
        """CW modal S extraction for one source/one frequency.

        Recommended when physically reliable passivity/loss diagnostics matter.
        """
        if frequency is None:
            raise ValueError("frequency is required for get_S_matrix_modal_cw.")

        port_map = self._normalize_portspecs(ports)
        source_port = self._port_name(source_port)
        if source_port not in port_map:
            raise ValueError(f"source_port '{source_port}' not found in ports.")

        waves = self.extract_port_waves_cw(
            ports=port_map.values(),
            frequency=frequency,
            steady_start_time=steady_start_time,
            avg_cycles=avg_cycles,
            window=window,
            mode_strategy=mode_strategy,
            return_power=True,
        )

        output_ports = self._normalize_output_port_names(output_ports, port_map)

        source_spec = port_map[source_port]
        source_incident_selector, _source_scattered_selector = (
            self._resolve_port_wave_selectors(
                source_spec,
                waves[source_port],
                use_reference=bool(source_spec.reference_monitor),
            )
        )
        a_incident = self._select_wave_component(
            waves[source_port],
            selector=source_incident_selector,
            use_reference=bool(source_spec.reference_monitor),
        )
        s_matrix = {}
        for out_port in output_ports:
            out_spec = port_map[out_port]
            _, out_scattered_selector = self._resolve_port_wave_selectors(
                out_spec,
                waves[out_port],
                use_reference=False,
            )
            b_out = self._select_wave_component(
                waves[out_port],
                selector=out_scattered_selector,
                use_reference=False,
            )
            b_vec = np.atleast_1d(np.asarray(b_out, dtype=np.complex128))
            a_vec = np.atleast_1d(np.asarray(a_incident, dtype=np.complex128))
            ratio = self._safe_ratio(b_vec, a_vec)[0]
            s_matrix[(out_port, source_port)] = np.complex128(ratio)

        self.s_matrix_frequencies = np.asarray([float(frequency)], dtype=float)
        s_output = self._format_s_matrix_output(s_matrix, as_sax=as_sax)

        if not return_diagnostics:
            return s_output

        p_in = float(
            np.abs(np.atleast_1d(np.asarray(a_incident, dtype=np.complex128))[0]) ** 2
        )
        p_guided_out = float(
            np.sum(
                [
                    np.abs(
                        self._select_wave_component(
                            waves[out],
                            selector=self._resolve_port_wave_selectors(
                                port_map[out],
                                waves[out],
                                use_reference=False,
                            )[1],
                            use_reference=False,
                        )
                    )
                    ** 2
                    for out in output_ports
                ]
            )
        )
        power_sum = p_guided_out / max(p_in, 1e-18)
        diagnostics = {
            "frequency": float(frequency),
            "source_port": source_port,
            "output_ports": output_ports,
            "waves": waves,
            "P_in": p_in,
            "P_guided_out": p_guided_out,
            "power_sum": power_sum,
            "loss_est": 1.0 - power_sum,
            "source_reference_normalization": {
                "enabled": bool(source_spec.reference_monitor),
                "monitor": source_spec.reference_monitor,
                "incident_wave": source_incident_selector,
                "scattered_wave": _source_scattered_selector,
            },
        }
        return {"s_matrix": s_output, "diagnostics": diagnostics}

    def get_s_matrix_modal(self, *args, **kwargs):
        return self.get_S_matrix_modal(*args, **kwargs)

    def get_s_matrix_modal_dft(self, *args, **kwargs):
        return self.get_S_matrix_modal_dft(*args, **kwargs)

    def get_s_matrix_modal_cw(self, *args, **kwargs):
        return self.get_S_matrix_modal_cw(*args, **kwargs)

    def get_S_matrix(self, *args, **kwargs):
        raise RuntimeError(
            "Simulation.get_S_matrix(...) is deprecated and removed. "
            "Use Simulation.get_S_matrix_modal(...)."
        )

    def get_s_matrix(self, *args, **kwargs):
        raise RuntimeError(
            "Simulation.get_s_matrix(...) is deprecated and removed. "
            "Use Simulation.get_s_matrix_modal(...)."
        )

    def run(self, **kwargs):
        """Run complete FDTD simulation with optional snapshot streaming.

        Supported streaming kwargs:
            - snapshot_field: field component to stream/store (e.g. ``"Ez"``)
            - snapshot_interval: emit every N steps
            - snapshot_callback: callable receiving each snapshot payload
            - store_snapshots: include emitted snapshots in the returned results
            - animate_live / save_video: matplotlib rendering conveniences
            - cmap_limits: "dynamic" or (vmin, vmax) for live snapshot colors
            - save_fields / field_subsample / progress
        """
        removed_visual_keys = (
            "jupyter_live",
            "axis_scale",
            "wavelength",
            "line_color",
            "line_opacity",
            "store_animation",
        )
        removed = [key for key in removed_visual_keys if key in kwargs]
        if removed:
            raise TypeError(
                "These legacy visualization kwargs are no longer supported. "
                f"Unsupported kwargs: {removed}. "
                "Use animate_live, save_video, or snapshot_field/snapshot_callback."
            )

        save_fields = kwargs.get("save_fields")
        field_subsample = int(kwargs.get("field_subsample", 1))
        progress = bool(kwargs.get("progress", False))
        record_interval = field_subsample if save_fields else None
        animate_live = kwargs.get("animate_live")
        save_video = kwargs.get("save_video")
        video_field = kwargs.get("video_field")
        snapshot_field = kwargs.get("snapshot_field")
        if snapshot_field is None:
            snapshot_field = video_field or animate_live
        if snapshot_field is not None:
            snapshot_field = str(snapshot_field)

        snapshot_interval = kwargs.get(
            "snapshot_interval", kwargs.get("animation_interval", 10)
        )
        user_callback = kwargs.get("snapshot_callback")
        callback = user_callback
        if animate_live and save_video is None:
            from beamz.visual import mpl as mpl_backend

            context = {"fig": None, "ax": None}
            cmap = kwargs.get("cmap", "twilight_zero")
            clean_visualization = bool(kwargs.get("clean_visualization", False))
            interpolation = kwargs.get("interpolation", "bicubic")
            pause = float(kwargs.get("pause", 0.001))
            vmin, vmax = mpl_backend.resolve_cmap_limits(
                kwargs.get("cmap_limits", "dynamic"),
                vmin=kwargs.get("vmin"),
                vmax=kwargs.get("vmax"),
            )

            def callback(snapshot):
                if user_callback is not None:
                    user_callback(snapshot)
                fig, ax = mpl_backend.snapshot_figure(
                    snapshot,
                    cmap=cmap,
                    clean_visualization=clean_visualization,
                    interpolation=interpolation,
                    figure=context["fig"],
                    axes=context["ax"],
                    vmin=vmin,
                    vmax=vmax,
                )
                context["fig"], context["ax"] = fig, ax
                mpl_backend._pyplot().show(block=False)
                mpl_backend._pyplot().pause(pause)

        store_snapshots_default = save_video is not None or bool(
            kwargs.get("store_snapshots", True)
        )
        results = self.run_compiled(
            num_steps=None,
            record_interval=record_interval,
            record_fields=save_fields,
            progress=progress,
            snapshot_field=snapshot_field,
            snapshot_interval=snapshot_interval,
            snapshot_callback=callback,
            store_snapshots=store_snapshots_default,
        )
        if save_video is not None and results is not None and results.snapshots:
            from beamz.visual.mpl import save_snapshot_video

            save_snapshot_video(
                results.snapshots,
                filename=save_video,
                fps=int(kwargs.get("video_fps", 30)),
                dpi=int(kwargs.get("video_dpi", 150)),
                cmap=kwargs.get("cmap", "twilight_zero"),
                cmap_limits=kwargs.get("cmap_limits"),
                vmin=kwargs.get("vmin"),
                vmax=kwargs.get("vmax"),
                clean_visualization=bool(kwargs.get("clean_visualization", False)),
                interpolation=kwargs.get("interpolation", "bicubic"),
            )
        return results

    def to_scene(self):
        """Build a 3D scene representation of the simulation setup."""
        from beamz.visual.scene import simulation_to_scene

        return simulation_to_scene(self)

    def to_plot_data(self):
        """Return renderer-agnostic simulation layout data."""
        from beamz.visual.data import simulation_plot_data

        return simulation_plot_data(self)

    def plot(self, **kwargs):
        """Plot the simulation layout using the matplotlib backend."""
        from beamz.visual.mpl import plot_simulation

        kwargs.setdefault("show", False)
        return plot_simulation(self, **kwargs)

    def plot_eps(self, **kwargs):
        """Plot a simulation permittivity slice."""
        from beamz.visual.mpl import plot_simulation_permittivity

        kwargs.setdefault("show", False)
        return plot_simulation_permittivity(self, **kwargs)

    def show(self, *, mode="auto", open_browser=True, **kwargs):
        """Display the simulation layout using the matplotlib backend."""
        del mode, open_browser
        kwargs.setdefault("show", True)
        return self.plot(**kwargs)

    def show_eps(self, **kwargs):
        """Display a simulation permittivity slice."""
        kwargs.setdefault("show", True)
        return self.plot_eps(**kwargs)

    def animate(self, field="Ez", **kwargs):
        """Run the simulation with live matplotlib animation enabled."""
        kwargs.setdefault("animate_live", field)
        return self.run(**kwargs)

    def save_video(self, filename, *, field="Ez", **kwargs):
        """Run the simulation and save a snapshot video."""
        if self.current_step >= self.num_steps:
            raise RuntimeError(
                "Simulation has already completed, so no video frames can be "
                "streamed from Simulation.save_video(...). Use results.save_video(...) "
                "from a run that stored save_fields or snapshot_field, or call "
                "Simulation.save_video(...) before running the simulation."
            )
        kwargs.setdefault("save_video", filename)
        kwargs.setdefault("video_field", field)
        return self.run(**kwargs)

    def show3d(self, *, mode="auto", open_browser=True, **kwargs):
        """Display the simulation setup in the interactive 3D scene viewer."""
        from beamz.visual.scene import view3d

        return view3d(self.to_scene(), mode=mode, open_browser=open_browser, **kwargs)

    def view3d(self, **kwargs):
        """Alias for :meth:`show3d`."""
        return self.show3d(**kwargs)
