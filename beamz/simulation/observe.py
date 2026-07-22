"""Generic monitor and field-recorder sampling and accumulation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from beamz.const import LIGHT_SPEED
from beamz.devices._immutable import readonly_array
from beamz.devices.monitors.compiler import (
    CompiledMonitorSpec,
)
from beamz.devices.monitors.monitors import _line_integral_scale_2d, _line_normal_2d
from beamz.devices.sources.specs import GaussianBeamSource, GaussianSource, ModeSource
from beamz.lattice import yee_flux
from beamz.simulation.model import SimulationState


@dataclass(frozen=True)
class SourceNormalization:
    """Store source-spectrum and launch-power normalization data."""

    waveform_spectrum: np.ndarray
    launch_power_ratio: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "waveform_spectrum",
            readonly_array(self.waveform_spectrum, dtype=np.complex128),
        )
        if self.launch_power_ratio is not None:
            object.__setattr__(
                self,
                "launch_power_ratio",
                readonly_array(self.launch_power_ratio, dtype=float),
            )

    @property
    def field_amplitude_norm(self) -> np.ndarray:
        """Return complex field-amplitude normalization by frequency."""
        spectrum = np.asarray(self.waveform_spectrum, dtype=np.complex128).reshape(-1)
        power = self.launch_power_ratio
        if power is None:
            return spectrum
        power = np.asarray(power, dtype=float).reshape(-1)
        if power.shape != spectrum.shape:
            return spectrum
        scale = np.ones_like(power)
        valid = np.isfinite(power) & (power > 1e-24)
        scale[valid] = np.sqrt(power[valid])
        return spectrum * scale

    @property
    def power_norm(self) -> np.ndarray:
        """Return real power normalization by frequency."""
        spectrum = np.asarray(self.waveform_spectrum, dtype=np.complex128).reshape(-1)
        scale = np.abs(spectrum) ** 2
        power = self.launch_power_ratio
        if power is None:
            return scale
        power = np.asarray(power, dtype=float).reshape(-1)
        if power.shape == spectrum.shape:
            valid = np.isfinite(power) & (power > 1e-24)
            scale[valid] *= power[valid]
        return scale


def _source_launch_power_ratio(source, freqs, *, fields=None, dt=None):
    if not isinstance(source, ModeSource):
        return None
    power = source.launch_power_normalization_spectrum(freqs, fields=fields, dt=dt)
    if power is None:
        return None
    power = np.asarray(power, dtype=float).reshape(-1)
    if power.shape != np.asarray(freqs).reshape(-1).shape:
        return None
    valid = np.isfinite(power) & (power > 1e-24)
    if not np.any(valid):
        return None
    return np.where(valid, power, 1.0)


def _signal_dft_spectrum(signal, sample_times, freqs):
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    times = np.asarray(sample_times, dtype=float).reshape(-1)
    signal = np.asarray(signal, dtype=float).reshape(-1)
    n = min(times.size, signal.size)
    if freqs.size == 0 or n == 0:
        return None
    times, signal = times[:n], signal[:n]
    phase = np.exp(1j * 2.0 * np.pi * times[:, None] * freqs[None, :])
    spectrum = (2.0 / float(n)) * np.sum(signal[:, None] * phase, axis=0)
    return spectrum if np.any(np.abs(spectrum) > 1e-12) else None


def _sampled_source_spectrum(source, freqs, *, time):
    time = np.asarray(time, dtype=float).reshape(-1)
    if time.size == 0:
        return None
    dt = float(np.median(np.diff(time))) if time.size > 1 else 0.0
    if isinstance(source, GaussianSource):
        signal = np.asarray(source.signal, dtype=float).reshape(-1)
        sample_times = time[: signal.size]
    elif isinstance(source, (ModeSource, GaussianBeamSource)) and dt > 0.0:
        from beamz.devices.sources.time import sample_source_waveforms

        signal, _ = sample_source_waveforms(
            source.source_time,
            t0=float(time[0]),
            dt=dt,
            num_steps=time.size,
            total_steps=time.size,
        )
        signal = np.asarray(signal, dtype=float).reshape(-1)
        sample_times = time[: signal.size]
    else:
        return None
    return _signal_dft_spectrum(signal, sample_times, freqs)


def source_normalization(sources, freqs, *, time=None, fields=None, dt=None):
    """Interpret source configuration as a frequency-domain normalization."""
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    if freqs.size == 0:
        return None
    for source in sources or ():
        spectrum = (
            _sampled_source_spectrum(source, freqs, time=time)
            if time is not None
            else None
        )
        if spectrum is None and isinstance(source, (ModeSource, GaussianBeamSource)):
            spectrum = source.source_spectrum(freqs, normalize=True)
        if spectrum is None:
            continue
        spectrum = np.asarray(spectrum, dtype=np.complex128).reshape(-1)
        if spectrum.shape == freqs.shape and np.any(np.abs(spectrum) > 1e-12):
            return SourceNormalization(
                spectrum,
                _source_launch_power_ratio(source, freqs, fields=fields, dt=dt),
            )
    return None


def source_field_amplitude_normalization(normalization):
    if normalization is None:
        return None
    value = (
        normalization.field_amplitude_norm
        if isinstance(normalization, SourceNormalization)
        else normalization
    )
    return np.asarray(value, dtype=np.complex128).reshape(-1)


def monitor_frequencies(monitor):
    recorded = np.asarray(monitor.dft_frequencies, dtype=float).reshape(-1)
    return recorded if recorded.size else np.asarray(monitor.monitor.freqs, dtype=float)


def monitor_dft_component(monitor, component):
    component = str(component)
    fields = monitor.dft_fields
    if component not in fields:
        raise ValueError(f"No DFT data recorded for component '{component}'.")
    values = np.asarray(fields[component], dtype=np.complex128)
    nfreq = monitor_frequencies(monitor).size
    if nfreq == 0:
        raise ValueError(f"Monitor '{monitor.name}' has no configured DFT frequencies.")
    values = values.reshape(nfreq, -1)
    weights = np.maximum(
        np.asarray(monitor.dft_weight_sum, dtype=float), 1e-18
    ).reshape(nfreq, 1)
    return (2.0 / weights) * values


def _flux_component(monitor, component):
    values = monitor_dft_component(monitor, component)
    if component.startswith("H") and monitor.dft_base_dt:
        phase = np.exp(-1j * np.pi * monitor_frequencies(monitor) * monitor.dft_base_dt)
        values = values * phase[:, None]
    return values


def _optional_flux_component(monitor, component):
    if component in monitor.dft_fields:
        return _flux_component(monitor, component)
    for available in monitor.dft_fields:
        return np.zeros_like(_flux_component(monitor, available))
    raise ValueError(f"Monitor '{monitor.name}' has no recorded DFT fields.")


def monitor_dft_flux(monitor):
    dx = float(monitor.resolution or 1.0)
    config = monitor.monitor
    if 0 <= monitor.normal_axis <= 2:
        axis = ("x", "y", "z")[monitor.normal_axis]
        sign = monitor.normal_sign
    else:
        normal = _line_normal_2d(config.start, config.end)
        axis, sign = normal or (config.plane_normal, 1.0)

    def field(name):
        return _optional_flux_component(monitor, name)

    if axis == "x":
        component = field("Ey") * np.conjugate(field("Hz")) - field(
            "Ez"
        ) * np.conjugate(field("Hy"))
    elif axis == "y":
        component = field("Ez") * np.conjugate(field("Hx")) - field(
            "Ex"
        ) * np.conjugate(field("Hz"))
    else:
        component = field("Ex") * np.conjugate(field("Hy")) - field(
            "Ey"
        ) * np.conjugate(field("Hx"))

    measure = float(monitor.power_scale)
    if not measure:
        measure = dx * dx if config.is_3d else _line_integral_scale_2d(axis, dx, dx)
    return 0.5 * np.real(np.sum(sign * component, axis=1)) * measure


def normalization_from_result(results, result, *, source=None):
    """Derive source normalization from a raw run and monitor result."""
    materials = results.metadata.fields.materials or result.material_region
    sources = results.sources if source is None else (results.sources[int(source)],)
    return source_normalization(
        sources,
        monitor_frequencies(result),
        time=results.metadata.time,
        fields=materials,
        dt=results.metadata.dt,
    )


def monitor_flux(result, results=None):
    del results
    if result.dft_fields:
        return monitor_dft_flux(result)
    return np.asarray(result.power_spectrum)


def _scale_frequency_axis(values, scale):
    array = np.asarray(values)
    scale = np.asarray(scale).reshape(-1)
    if not array.size or array.shape[0] != scale.size:
        return array
    divisor = scale.reshape((scale.size,) + (1,) * (array.ndim - 1))
    return np.divide(
        array,
        divisor,
        out=np.zeros_like(array, dtype=np.result_type(array, divisor)),
        where=np.abs(divisor) > 1e-24,
    )


def renormalize_result(result, source: int | None):
    """Return an immutable raw or source-normalized view of a run result."""
    if source is not None:
        source = int(source)
        if source < 0 or source >= len(result.sources):
            raise ValueError(
                f"normalization source {source} is invalid for "
                f"{len(result.sources)} sources."
            )

    raw_monitors = {
        name: replace(
            monitor,
            dft_fields=monitor._raw_dft_fields or {},
            power_spectrum=monitor._raw_power_spectrum,
        )
        for name, monitor in result.monitors.items()
    }
    raw_result = replace(result, monitors=raw_monitors, normalization_source=None)
    if source is None:
        return raw_result

    normalized = {}
    for name, monitor in raw_monitors.items():
        normalization = normalization_from_result(raw_result, monitor, source=source)
        if normalization is None:
            normalized[name] = monitor
            continue
        field_scale = np.asarray(
            normalization.field_amplitude_norm, dtype=np.complex128
        )
        normalized[name] = replace(
            monitor,
            dft_fields={
                component: _scale_frequency_axis(values, field_scale)
                for component, values in monitor.dft_fields.items()
            },
            power_spectrum=_scale_frequency_axis(
                monitor.power_spectrum, normalization.power_norm
            ),
        )
    return replace(raw_result, monitors=normalized, normalization_source=source)


def step_hits_interval(step, interval):
    """Return whether a zero-based step lands on a positive modulo interval."""
    step = jnp.asarray(step, dtype=jnp.int32)
    interval = jnp.maximum(jnp.asarray(interval, dtype=jnp.int32), 1)
    return (step % interval) == 0


def monitor_records_on_step(step, interval):
    """Return whether a monitor emits after this zero-based simulation step."""
    return step_hits_interval(jnp.asarray(step, dtype=jnp.int32) + 1, interval)


def monitor_dft_should_accumulate(enabled, step, t, t_start, t_end, interval):
    """Apply the compiled time window and sampling interval for one DFT sample."""
    t = jnp.asarray(t, dtype=jnp.float32)
    return (
        jnp.asarray(enabled)
        & (t >= jnp.asarray(t_start, dtype=jnp.float32))
        & (t <= jnp.asarray(t_end, dtype=jnp.float32))
        & step_hits_interval(step, interval)
    )


def monitor_dft_window_weight(t, t_start, t_end, use_hann):
    """Return one or the Hann weight selected by a compiled monitor plan."""
    t, t0, t1 = (jnp.asarray(value, dtype=jnp.float32) for value in (t, t_start, t_end))
    span = jnp.maximum(t1 - t0, jnp.asarray(1e-30, dtype=jnp.float32))
    tau = jnp.clip((t - t0) / span, 0.0, 1.0)
    hann = 0.5 * (1.0 - jnp.cos(jnp.asarray(2.0 * np.pi, dtype=jnp.float32) * tau))
    return jnp.where(jnp.asarray(use_hann) & jnp.isfinite(t1) & (t1 > t0), hann, 1.0)


def monitor_dft_sample_scale(
    weight, *, normalization_code, base_dt, record_interval, length_unit
):
    """Scale a nonnegative DFT weight in native or physical units."""
    dtype = jnp.result_type(weight, base_dt, record_interval)
    native = jnp.maximum(jnp.asarray(weight, dtype=dtype), 0.0)
    physical = native * (
        jnp.asarray(base_dt * record_interval * LIGHT_SPEED / length_unit, dtype=dtype)
        / jnp.sqrt(jnp.asarray(2.0 * np.pi, dtype=dtype))
    )
    return jnp.where(jnp.asarray(normalization_code) == 1, physical, native)


def _sample_components(fields, indices, weights):
    """Apply one canonical weighted-gather plan to Ex, Ey, Ez, Hx, Hy, and Hz."""
    return jnp.stack(
        tuple(
            jnp.sum(field.reshape(-1)[flat_idx] * component_weights, axis=-1)
            for field, flat_idx, component_weights in zip(
                fields, indices, weights, strict=True
            )
        ),
        axis=0,
    )


def monitor_dft_accumulator_dtype():
    """Return the real dtype used for compiled monitor DFT accumulators."""

    # Match global precision so long accumulations gain x64 accuracy when enabled.
    return jnp.float64 if bool(jax.config.read("jax_enable_x64")) else jnp.float32


def monitor_state_size(specs: tuple[CompiledMonitorSpec, ...], num_steps: int) -> int:
    """Return the time-record capacity required by compiled monitors."""
    # FieldRecorder frames have dedicated shape-aware buffers and must not inflate the
    # rectangular scalar-power history shared by ordinary monitors.
    power_specs = tuple(spec for spec in specs if spec.accumulate_power)
    if not power_specs:
        return 0
    return int(
        max(
            int(np.ceil(num_steps / max(1, int(spec.record_interval))))
            for spec in power_specs
        )
    )


def monitor_frequency_size(specs: tuple[CompiledMonitorSpec, ...]) -> int:
    """Return the largest frequency count required by any monitor."""
    # Frequency rows are rectangular to keep the scan-state shape static.
    if not specs:
        return 0
    return int(max(int(spec.freq_count) for spec in specs))


def monitor_dft_point_size(specs: tuple[CompiledMonitorSpec, ...]) -> int:
    """Return the largest DFT point count required by any monitor."""
    # Spatial DFT vectors follow the same padded-row policy as frequencies.
    if not specs:
        return 0
    return int(max(int(getattr(spec, "dft_point_count", 0)) for spec in specs))


MONITOR_FIELDS = (
    "powers",
    "timestamps",
    "counts",
    "freq_flux_re",
    "freq_flux_im",
    "freq_phase_re",
    "freq_phase_im",
    "dft_vec_re",
    "dft_vec_im",
    "dft_weight_sum",
    "recorded_fields",
    "recorded_steps",
    "recorded_times",
    "recorded_counts",
)


def empty_monitor_values(program, num_steps: int | None = None) -> dict[str, Any]:
    # 1. Resolve monitor specs and accumulator precision; a no-monitor program receives
    # concrete empty arrays of the same ranks as an enabled state.
    specs = program.monitors
    dft_dtype = monitor_dft_accumulator_dtype()
    if not specs:
        values = tuple(
            jnp.zeros(shape, dtype=dtype)
            for shape, dtype in (
                ((0, 0), jnp.float32),
                ((0, 0), jnp.float32),
                ((0,), jnp.int32),
                ((0, 0), jnp.float32),
                ((0, 0), jnp.float32),
                ((0, 0), jnp.float32),
                ((0, 0), jnp.float32),
                ((0, 0, 0, 0), dft_dtype),
                ((0, 0, 0, 0), dft_dtype),
                ((0, 0), dft_dtype),
            )
        ) + ((), (), (), ())
        return dict(zip(MONITOR_FIELDS, values, strict=True))
    # 2. Derive record, frequency, and point capacities from plan maxima so all monitors
    # share rectangular buffers and continuation keeps one stable pytree.
    n = len(specs)
    max_records = max(
        1,
        monitor_state_size(
            specs,
            program.config.num_steps if num_steps is None else int(num_steps),
        ),
    )
    max_freq = monitor_frequency_size(specs)
    max_points = monitor_dft_point_size(specs)
    recorder_specs = tuple(spec for spec in specs if spec.recorder_index >= 0)
    recorder_steps = max(
        1, int(program.config.num_steps if num_steps is None else num_steps)
    )
    # 3. Initialize scalar phase recurrence at 1+0i and every accumulated quantity at zero.
    values = (
        jnp.zeros((n, max_records), dtype=jnp.float32),
        jnp.zeros((n, max_records), dtype=jnp.float32),
        jnp.zeros((n,), dtype=jnp.int32),
        jnp.zeros((n, max_freq), dtype=jnp.float32),
        jnp.zeros((n, max_freq), dtype=jnp.float32),
        jnp.ones((n, max_freq), dtype=jnp.float32),
        jnp.zeros((n, max_freq), dtype=jnp.float32),
        jnp.zeros((n, 6, max_freq, max_points), dtype=dft_dtype),
        jnp.zeros((n, 6, max_freq, max_points), dtype=dft_dtype),
        jnp.zeros((n, max_freq), dtype=dft_dtype),
        tuple(
            jnp.zeros(
                (
                    int(np.ceil(recorder_steps / spec.record_interval)),
                    *shape,
                ),
                dtype=jnp.float32,
            )
            for spec in recorder_specs
            for shape in spec.field_shapes
        ),
        tuple(
            jnp.zeros(
                (int(np.ceil(recorder_steps / spec.record_interval)),),
                dtype=jnp.int32,
            )
            for spec in recorder_specs
        ),
        tuple(
            jnp.zeros(
                (int(np.ceil(recorder_steps / spec.record_interval)),),
                dtype=jnp.float32,
            )
            for spec in recorder_specs
        ),
        tuple(jnp.zeros((), dtype=jnp.int32) for _ in recorder_specs),
    )
    return dict(zip(MONITOR_FIELDS, values, strict=True))


def _reduce_power(samples: jnp.ndarray, spec: CompiledMonitorSpec):
    """Reduce canonical colocated fields to signed normal flux or flux magnitude."""
    return jnp.asarray(
        yee_flux(
            samples,
            spec.normal_axis,
            normal_sign=spec.normal_sign,
            measure=spec.power_scale,
        ),
        dtype=jnp.float32,
    )


def _accumulate_dft(mon, carry, field_arrays, t_phys, dt_scalar):
    """Gather and accumulate one monitor's compiled field-vector DFT."""
    d_re, d_im, d_w = carry
    dtype = d_re.dtype
    theta = (
        jnp.asarray(2.0 * np.pi, dtype=dtype)
        * jnp.asarray(mon.freq_hz, dtype=dtype)
        * jnp.asarray(t_phys, dtype=dtype)
    )
    phase_re, phase_im = jnp.cos(theta), jnp.sin(theta)
    window = jnp.asarray(
        monitor_dft_window_weight(
            t_phys,
            mon.dft_t_start,
            mon.dft_t_end,
            mon.dft_window_code == 1,
        ),
        dtype=jnp.float32,
    )
    scale = jnp.asarray(
        monitor_dft_sample_scale(
            window,
            normalization_code=mon.dft_normalization_code,
            base_dt=dt_scalar,
            record_interval=mon.dft_record_interval,
            length_unit=mon.dft_length_unit,
        ),
        dtype=dtype,
    )

    # Compilation has already encoded dimensional colocation in these weighted gathers.
    vectors = _sample_components(field_arrays, mon.dft_flat_idx, mon.dft_weights)
    component_mask = mon.dft_component_mask.astype(dtype)[:, None, None]
    delta_re = (
        scale
        * component_mask
        * jnp.einsum("f,cp->cfp", phase_re, vectors.astype(dtype))
    )
    delta_im = (
        scale
        * component_mask
        * jnp.einsum("f,cp->cfp", phase_im, vectors.astype(dtype))
    )
    mi, nf, npnt = mon.monitor_index, mon.freq_count, mon.dft_point_count
    d_re = d_re.at[mi, :, :nf, :npnt].add(delta_re[:, :nf, :npnt])
    d_im = d_im.at[mi, :, :nf, :npnt].add(delta_im[:, :nf, :npnt])
    d_w = d_w.at[mi, :nf].add(jnp.asarray(window, dtype=dtype))
    return d_re, d_im, d_w


def _record_fields(program, state, abs_step, t_phys, fields):
    """Update every compiled FieldRecorder and return replacement state members."""
    recorded_fields = list(state.recorded_fields)
    recorded_steps = list(state.recorded_steps)
    recorded_times = list(state.recorded_times)
    recorded_counts = list(state.recorded_counts)
    for spec in program.monitors:
        if spec.recorder_index < 0:
            continue
        recorder_index = spec.recorder_index
        count = recorded_counts[recorder_index]
        capacity = recorded_steps[recorder_index].shape[0]
        should_record = monitor_records_on_step(abs_step, spec.record_interval) & (
            count < capacity
        )
        slot = jnp.minimum(count, capacity - 1)
        for component, sign, buffer_index, shape, flat_idx, weights in zip(
            spec.canonical_components,
            spec.component_signs,
            spec.field_buffer_indices,
            spec.field_shapes,
            spec.field_interp_flat_idx,
            spec.field_interp_weights,
            strict=True,
        ):
            value = (
                jnp.sum(
                    fields[component].reshape(-1)[flat_idx] * weights, axis=-1
                ).reshape(shape)
                if flat_idx.size
                else fields[component][tuple(slice(0, size) for size in shape)]
            )
            recorded_fields[buffer_index] = jax.lax.cond(
                should_record,
                lambda buffer, value=value * sign, slot=slot: buffer.at[slot].set(
                    value
                ),
                lambda buffer: buffer,
                recorded_fields[buffer_index],
            )
        recorded_steps[recorder_index] = jax.lax.cond(
            should_record,
            lambda buffer, slot=slot: buffer.at[slot].set(abs_step + 1),
            lambda buffer: buffer,
            recorded_steps[recorder_index],
        )
        recorded_times[recorder_index] = jax.lax.cond(
            should_record,
            lambda buffer, slot=slot: buffer.at[slot].set(t_phys),
            lambda buffer: buffer,
            recorded_times[recorder_index],
        )
        recorded_counts[recorder_index] = count + should_record.astype(jnp.int32)
    return {
        "recorded_fields": tuple(recorded_fields),
        "recorded_steps": tuple(recorded_steps),
        "recorded_times": tuple(recorded_times),
        "recorded_counts": tuple(recorded_counts),
    }


def update_monitors(
    program,
    state: SimulationState,
    abs_step: jnp.ndarray,
    t_phys: jnp.ndarray,
    dt_scalar: jnp.ndarray,
    ex: jnp.ndarray,
    ey: jnp.ndarray,
    ez: jnp.ndarray,
    hx: jnp.ndarray,
    hy: jnp.ndarray,
    hz: jnp.ndarray,
) -> SimulationState:
    # 1. Return immediately when no monitor plan exists; touching empty buffers would add
    # needless operations to every compiled timestep.
    if not program.monitors:
        return state

    # 2. Unpack the immutable state into local arrays. Updates below remain functional and
    # are repackaged only once at the end of the timestep.
    powers = state.powers
    timestamps = state.timestamps
    counts = state.counts
    freq_flux_re = state.freq_flux_re
    freq_flux_im = state.freq_flux_im
    freq_phase_re = state.freq_phase_re
    freq_phase_im = state.freq_phase_im
    dft_vec_re = state.dft_vec_re
    dft_vec_im = state.dft_vec_im
    dft_weight_sum = state.dft_weight_sum
    max_records = powers.shape[1]
    field_arrays = (ex, ey, ez, hx, hy, hz)

    # 3. Every scalar and spectral monitor follows the same sampling and accumulator path;
    # geometry and dimensional choices are already encoded in its weighted gathers.
    for mon in program.monitors:
        if mon.recorder_index >= 0:
            continue
        should_record = monitor_records_on_step(abs_step, mon.record_interval)
        can_record = counts[mon.monitor_index] < max_records
        do_record = should_record & can_record & mon.accumulate_power
        do_freq = (
            step_hits_interval(abs_step, mon.freq_record_interval)
            if mon.accumulate_frequency and mon.freq_count > 0
            else jnp.array(False)
        )
        need_sample = do_record | do_freq

        power_sample = jax.lax.cond(
            need_sample,
            lambda _, mon=mon: _reduce_power(
                _sample_components(
                    field_arrays, mon.sample_flat_idx, mon.sample_weights
                ),
                mon,
            ),
            lambda _unused: jnp.array(0.0, dtype=jnp.float32),
            operand=None,
        )
        power_val = jnp.where(
            do_record, power_sample, jnp.array(0.0, dtype=jnp.float32)
        )

        # 4b. Record scalar power and timestamp conditionally while keeping the destination
        # index in bounds for both branches of the compiled selection.
        slot = jnp.minimum(counts[mon.monitor_index], max_records - 1)
        old_power = powers[mon.monitor_index, slot]
        old_ts = timestamps[mon.monitor_index, slot]

        powers = powers.at[mon.monitor_index, slot].set(
            jnp.where(do_record, jnp.asarray(power_val), jnp.asarray(old_power))
        )
        timestamps = timestamps.at[mon.monitor_index, slot].set(
            jnp.where(do_record, t_phys, old_ts)
        )
        counts = counts.at[mon.monitor_index].set(
            counts[mon.monitor_index] + jnp.where(do_record, 1, 0)
        )
        # 4c. Integrate scalar spectral flux with an explicit real/imaginary recurrence,
        # avoiding complex carry types that vary in backend support.
        if mon.accumulate_frequency and mon.freq_count > 0:
            mi = mon.monitor_index
            row_f_re = freq_flux_re[mi, : mon.freq_count]
            row_f_im = freq_flux_im[mi, : mon.freq_count]
            row_ph_re = freq_phase_re[mi, : mon.freq_count]
            row_ph_im = freq_phase_im[mi, : mon.freq_count]
            theta_now = (
                jnp.asarray(2.0 * np.pi, dtype=jnp.float32)
                * jnp.asarray(mon.freq_hz, dtype=jnp.float32)
                * t_phys
            )
            cur_ph_re = jnp.cos(theta_now)
            cur_ph_im = jnp.sin(theta_now)
            delta_re = power_sample * dt_scalar * cur_ph_re
            delta_im = power_sample * dt_scalar * cur_ph_im
            zero_freq = jnp.asarray(0.0, dtype=row_f_re.dtype)
            row_f_re = row_f_re + jnp.where(do_freq, delta_re, zero_freq)
            row_f_im = row_f_im + jnp.where(do_freq, delta_im, zero_freq)
            next_ph_re = row_ph_re * mon.freq_rot_re - row_ph_im * mon.freq_rot_im
            next_ph_im = row_ph_re * mon.freq_rot_im + row_ph_im * mon.freq_rot_re
            row_ph_re = jnp.where(do_freq, next_ph_re, row_ph_re)
            row_ph_im = jnp.where(do_freq, next_ph_im, row_ph_im)
            freq_flux_re = freq_flux_re.at[mi, : mon.freq_count].set(row_f_re)
            freq_flux_im = freq_flux_im.at[mi, : mon.freq_count].set(row_f_im)
            freq_phase_re = freq_phase_re.at[mi, : mon.freq_count].set(row_ph_re)
            freq_phase_im = freq_phase_im.at[mi, : mon.freq_count].set(row_ph_im)
        # 4d. Gate full vector DFT sampling by its time window and cadence because field
        # interpolation is substantially more expensive than updating scalar flux.
        if mon.dft_enabled and mon.freq_count > 0 and mon.dft_point_count > 0:
            do_dft = monitor_dft_should_accumulate(
                mon.dft_enabled and mon.freq_count > 0 and mon.dft_point_count > 0,
                abs_step,
                t_phys,
                mon.dft_t_start,
                mon.dft_t_end,
                mon.dft_record_interval,
            )

            dft_vec_re, dft_vec_im, dft_weight_sum = jax.lax.cond(
                do_dft,
                lambda carry, mon=mon: _accumulate_dft(
                    mon, carry, field_arrays, t_phys, dt_scalar
                ),
                lambda carry: carry,
                (dft_vec_re, dft_vec_im, dft_weight_sum),
            )

    recorder_updates = _record_fields(
        program,
        state,
        abs_step,
        t_phys,
        {"Ex": ex, "Ey": ey, "Ez": ez, "Hx": hx, "Hy": hy, "Hz": hz},
    )
    return state._replace(
        powers=powers,
        timestamps=timestamps,
        counts=counts,
        freq_flux_re=freq_flux_re,
        freq_flux_im=freq_flux_im,
        freq_phase_re=freq_phase_re,
        freq_phase_im=freq_phase_im,
        dft_vec_re=dft_vec_re,
        dft_vec_im=dft_vec_im,
        dft_weight_sum=dft_weight_sum,
        **recorder_updates,
    )
