from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from beamz.analysis import mode_projection as _mp
from beamz.analysis.data import analysis_inputs
from beamz.analysis.modal_projection.colocation import (
    _colocate_field_components_to_projection_3d,
)
from beamz.analysis.modal_projection.diagnostics import (
    _modal_projection_reconstruction_residual,
)
from beamz.analysis.modal_projection.geometry import (
    _apply_modal_projection_spatial_phase,
    _monitor_projection_phase,
)
from beamz.devices._immutable import immutable_snapshot, readonly_array
from beamz.devices.ports import Port


@dataclass(frozen=True, slots=True)
class SParameterResult:
    """Store immutable broadband modal scattering parameters.

    Parameters
    ----------
    s_matrix : mapping
        ``(output_port, source_port)`` keys mapped to complex frequency vectors.
    frequencies : array-like
        Frequencies in hertz shared by every S-parameter vector.
    diagnostics : mapping
        Projection residuals, conditioning, incident thresholds, and related
        extraction metadata.

    Notes
    -----
    Arrays and mappings are copied into read-only result-owned storage.
    """

    s_matrix: Mapping[tuple[str, str], np.ndarray]
    frequencies: np.ndarray
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "s_matrix",
            MappingProxyType(
                {key: readonly_array(value) for key, value in self.s_matrix.items()}
            ),
        )
        object.__setattr__(self, "frequencies", readonly_array(self.frequencies))
        object.__setattr__(self, "diagnostics", immutable_snapshot(self.diagnostics))


def _port_name(port):
    if isinstance(port, str):
        return port
    if isinstance(port, Port):
        return port.name
    raise TypeError(f"Expected a port name or Port, got {type(port).__name__}.")


def _normalize_ports(ports):
    values = tuple(ports.values()) if isinstance(ports, Mapping) else tuple(ports)
    if not values:
        raise ValueError("ports must contain at least one Port.")
    invalid = next((value for value in values if not isinstance(value, Port)), None)
    if invalid is not None:
        raise TypeError(
            f"ports must contain canonical Port objects; got {type(invalid).__name__}."
        )
    if len({port.name for port in values}) != len(values):
        raise ValueError("Port names must be unique.")
    return {port.name: port for port in values}


def _normalize_output_port_names(output_ports, port_map):
    names = (
        list(port_map)
        if output_ports is None
        else [_port_name(p) for p in output_ports]
    )
    missing = [name for name in names if name not in port_map]
    if missing:
        raise ValueError(f"output_ports contains unknown ports: {missing}")
    return names


def _safe_ratio(num, den, eps=1e-18):
    out = np.zeros_like(num, dtype=np.complex128)
    valid = np.abs(den) > eps
    out[valid] = num[valid] / den[valid]
    return out


def _wave_selectors(port: Port, *, is_3d: bool) -> tuple[str, str]:
    """Return incident and scattered modal branches for one canonical port."""
    positive_axis_wave = "minus" if is_3d and port.axis in {"x", "y"} else "plus"
    incident = (
        positive_axis_wave
        if port.direction == "+"
        else ("minus" if positive_axis_wave == "plus" else "plus")
    )
    return incident, "minus" if incident == "plus" else "plus"


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
        re = np.interp(freq_dst, freq_src, np.real(src[:, col]), left=0.0, right=0.0)
        im = np.interp(freq_dst, freq_src, np.imag(src[:, col]), left=0.0, right=0.0)
        out[:, col] = re + 1j * im
    return out


def _sample_monitor_component_dft(data, monitor, component, frequencies):
    freq_src = np.asarray(data.frequencies, dtype=float)
    values_src = np.asarray(data.field(component), dtype=np.complex128)
    if freq_src.size == 0:
        raise ValueError(f"Monitor '{monitor.name}' has no configured DFT frequencies.")
    values_src = _resample_complex_matrix(freq_src, values_src, freq_src)
    freq_dst = np.atleast_1d(np.asarray(frequencies, dtype=float))
    sampled = _resample_complex_matrix(freq_src, values_src, freq_dst)
    phase = _monitor_projection_phase(component, freq_dst, data.dt)
    sampled = sampled * phase[:, None]
    return freq_dst, sampled


def _extract_port_waves_dft(
    sim,
    ports,
    frequencies,
    min_incident_db=-40.0,
    return_power=True,
    mode_strategy="per_frequency",
):
    """Extract modal waves after lowering acquisitions to :class:`AnalysisData`."""
    del min_incident_db  # Used by s_parameters validity masking.
    inputs = analysis_inputs(sim)
    context = next(iter(inputs.values()))
    if (not context.is_3d) and context.plane_2d != "xy":
        raise NotImplementedError(
            "Modal DFT extraction currently supports 2D simulations in the xy plane."
        )

    port_map = _normalize_ports(ports)
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

    monitor_by_name = {
        name: data.monitor_geometry
        for name, data in inputs.items()
        if data.monitor_geometry is not None
    }
    for spec in port_map.values():
        main = monitor_by_name.get(spec.monitor_name)
        if main is None:
            raise ValueError(
                f"Missing monitor '{spec.monitor_name}' for port '{spec.name}'."
            )
        if not inputs[spec.monitor_name].frequencies.size:
            raise ValueError(f"Monitor '{spec.monitor_name}' must define frequencies.")

    dft_cache = {}
    projection_cache = {}
    waves = {}

    def _matching_3d_group_specs(spec, monitor_name):
        if not context.is_3d:
            return (spec,)
        candidates = [
            candidate
            for candidate in port_map.values()
            if candidate.monitor_name == monitor_name
        ]
        group = [
            candidate
            for candidate in candidates
            if candidate.axis == spec.axis
            and candidate.polarization == spec.polarization
        ]
        if not group:
            return (spec,)
        return tuple(sorted(group, key=lambda item: (int(item.mode_index), item.name)))

    def _build_3d_group_projection(spec, monitor, f_mode):
        data = inputs[monitor.name]
        return _mp._build_port_projection(
            data,
            spec,
            monitor,
            f_mode,
            projection_cache,
        )

    def _project_3d_group_at_monitor(
        spec,
        monitor,
        idx,
        frequency,
        f_mode,
    ):
        group_specs = _matching_3d_group_specs(spec, monitor.name)
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
            comp: _apply_modal_projection_spatial_phase(
                comp,
                dft_cache[(monitor.name, comp)][idx],
                frequency,
                proj,
            )
            for comp in proj_components
        }
        field_components = _colocate_field_components_to_projection_3d(
            inputs[monitor.name],
            monitor,
            raw_field_components,
            proj,
        )
        coeffs, residual, group_cond, projection_diag = (
            _mp._project_modal_coefficients_3d_group(
                field_components,
                projections,
            )
        )
        cond_values = [
            float(projection.get("condition_number", np.nan))
            for projection in projections
        ]
        finite_conds = [
            value for value in [float(group_cond), *cond_values] if np.isfinite(value)
        ]
        cond = max(finite_conds) if finite_conds else np.nan
        return (
            coeffs[group_index],
            residual,
            cond,
            float(proj.get("mode_neff", np.nan)),
            projection_diag,
        )

    def _project_monitor(spec, monitor, parts):
        components = (
            parts.get(
                "projection_components_3d",
                (parts["e_component"], parts["h_component"]),
            )
            if context.is_3d
            else (parts["e_component"], parts["h_component"])
        )
        for component in components:
            key = (monitor.name, component)
            if key not in dft_cache:
                _, dft_cache[key] = _sample_monitor_component_dft(
                    inputs[monitor.name], monitor, component, frequencies=freqs
                )

        values = {
            "plus": np.zeros(freqs.size, dtype=np.complex128),
            "minus": np.zeros(freqs.size, dtype=np.complex128),
            "condition_number": np.zeros(freqs.size, dtype=float),
            "mode_neff": np.full(freqs.size, np.nan),
            "projection_residual": np.full(freqs.size, np.nan),
            "projection_residual_e": np.full(freqs.size, np.nan),
            "projection_residual_h": np.full(freqs.size, np.nan),
            "projection_residual_balanced": np.full(freqs.size, np.nan),
            "projected_signed_power": np.full(freqs.size, np.nan),
            "projection_e_scale": np.full(
                freqs.size, np.nan + 0.0j, dtype=np.complex128
            ),
            "projection_h_scale": np.full(
                freqs.size, np.nan + 0.0j, dtype=np.complex128
            ),
        }
        for idx, frequency in enumerate(freqs):
            f_mode = float(frequency if strategy == "per_frequency" else single_freq)
            proj = _mp._build_port_projection(
                inputs[monitor.name],
                spec,
                monitor,
                f_mode,
                projection_cache,
            )

            if context.is_3d:
                coeff, residual, cond, neff, diag = _project_3d_group_at_monitor(
                    spec,
                    monitor,
                    idx,
                    frequency,
                    f_mode,
                )
                values["plus"][idx], values["minus"][idx] = coeff[:2]
                values["projection_residual"][idx] = residual
                values["condition_number"][idx] = cond
                values["mode_neff"][idx] = neff
                for key in (
                    "projection_residual_e",
                    "projection_residual_h",
                    "projection_residual_balanced",
                    "projected_signed_power",
                ):
                    values[key][idx] = float(
                        diag.get(key.removeprefix("projection_"), np.nan)
                    )
                values["projection_e_scale"][idx] = np.complex128(
                    diag.get("e_scale", np.nan + 0.0j)
                )
                values["projection_h_scale"][idx] = np.complex128(
                    diag.get("h_scale", np.nan + 0.0j)
                )
                continue

            proj_components = tuple(
                proj.get("components", (proj["e_component"], proj["h_component"]))
            )
            field_vec = np.concatenate(
                [
                    _apply_modal_projection_spatial_phase(
                        component,
                        dft_cache[(monitor.name, component)][idx],
                        frequency,
                        proj,
                    )
                    for component in proj_components
                ]
            )
            projection_weights = np.asarray(
                proj.get("projection_weights", np.ones(field_vec.size)),
                dtype=float,
            )
            coeff = proj["pinv"] @ (projection_weights * field_vec)
            values["plus"][idx], values["minus"][idx] = coeff[:2]
            values["projection_residual"][idx] = (
                _modal_projection_reconstruction_residual(field_vec, proj, coeff)
            )
            values["condition_number"][idx] = float(
                proj.get("condition_number", np.nan)
            )
            values["mode_neff"][idx] = float(proj.get("mode_neff", np.nan))
        return values

    for spec in port_map.values():
        parts = _mp._mode_components_for_port(spec)
        main_monitor = monitor_by_name[spec.monitor_name]
        main = _project_monitor(
            spec,
            main_monitor,
            parts,
        )
        port_waves = {
            "a_plus": main.pop("plus"),
            "a_minus": main.pop("minus"),
            **main,
        }
        if return_power:
            port_waves["P_plus"] = np.abs(port_waves["a_plus"]) ** 2
            port_waves["P_minus"] = np.abs(port_waves["a_minus"]) ** 2

        waves[spec.name] = port_waves
    return waves


def s_parameters(
    results_or_sim,
    source_port,
    ports,
    output_ports=None,
    frequencies=None,
    min_incident_db=-40.0,
    mode_strategy="per_frequency",
):
    """Extract broadband modal S-parameters from DFT monitor accumulators.

    Parameters
    ----------
    results_or_sim : SimulationResults or mapping of AnalysisData
        Detached canonical analysis inputs containing named mode monitors.
    source_port : str or Port
        Excited input port.
    ports : mapping or sequence
        Port definitions describing monitor names and modal wave conventions.
    output_ports : sequence, optional
        Output ports to extract. All ports are used when omitted.
    frequencies : array-like, optional
        Requested frequencies in hertz. The source-port monitor grid is used
        by default.
    min_incident_db : float, default=-40
        Incident-amplitude floor relative to its peak, in decibels. Samples below
        the threshold are excluded from stable ratios.
    mode_strategy : {"per_frequency", "tracked"}, default="per_frequency"
        Solve modes independently at each frequency or track branch identity
        across the frequency grid.

    Returns
    -------
    SParameterResult
        Read-only complex scattering vectors and extraction diagnostics.

    Raises
    ------
    ValueError
        If a port, monitor, frequency grid, or mode strategy is invalid.

    Notes
    -----
    ``S[(out, src)]`` is the scattered modal amplitude at ``out`` divided by
    the incident modal amplitude at ``src`` under each port's wave convention.

    Examples
    --------
    >>> result = beamz.analysis.s_parameters(
    ...     simulation_results,
    ...     source_port="in",
    ...     ports=[input_port, output_port],
    ... )
    """
    inputs = analysis_inputs(results_or_sim)
    context = next(iter(inputs.values()))
    port_map = _normalize_ports(ports)
    source_port = _port_name(source_port)
    if source_port not in port_map:
        raise ValueError(f"source_port '{source_port}' not found in ports.")

    if frequencies is None:
        src_spec = port_map[source_port]
        src_data = inputs.get(src_spec.monitor_name)
        if src_data is None:
            raise ValueError(f"Missing source monitor '{src_spec.monitor_name}'.")
        frequencies = src_data.frequencies
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))

    waves = _extract_port_waves_dft(
        results_or_sim,
        ports=port_map.values(),
        frequencies=frequencies,
        min_incident_db=min_incident_db,
        return_power=True,
        mode_strategy=mode_strategy,
    )

    output_ports = _normalize_output_port_names(output_ports, port_map)

    source_spec = port_map[source_port]
    source_incident, _ = _wave_selectors(source_spec, is_3d=context.is_3d)
    a_incident = np.asarray(
        waves[source_port][f"a_{source_incident}"],
        dtype=np.complex128,
    )
    max_incident = float(np.max(np.abs(a_incident))) if a_incident.size else 0.0
    rel_floor = max_incident * (10.0 ** (float(min_incident_db) / 20.0))
    abs_floor = max(1e-18, rel_floor)
    valid_mask = np.abs(a_incident) >= abs_floor

    scattered_waves = {}
    s_matrix = {}
    for out_port in output_ports:
        out_spec = port_map[out_port]
        _, scattered = _wave_selectors(out_spec, is_3d=context.is_3d)
        b_out = np.asarray(
            waves[out_port][f"a_{scattered}"],
            dtype=np.complex128,
        )
        scattered_waves[out_port] = b_out
        ratio = _safe_ratio(b_out, a_incident)
        ratio = np.where(valid_mask, ratio, 0.0 + 0.0j)
        s_matrix[(out_port, source_port)] = ratio

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
                "monitor": np.asarray(data.get("condition_number", []), dtype=float),
            }
            for name, data in waves.items()
        },
        "monitor_flux_checks": _modal_dft_flux_diagnostics(
            inputs, port_map, waves, frequencies
        ),
        "scattered_waves": scattered_waves,
    }
    return SParameterResult(
        s_matrix=s_matrix,
        frequencies=np.asarray(frequencies, dtype=float),
        diagnostics=diagnostics,
    )


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


def _modal_dft_flux_diagnostics(inputs, port_map, waves, frequencies):
    """Compare raw DFT flux monitors with selector-aware modal overlap power."""

    freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
    diagnostics = {}
    for name, spec in port_map.items():
        wave = waves.get(name, {})

        def _wave_power(power_key, amplitude_key, wave=wave):
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

        data = inputs.get(spec.monitor_name)
        incident_selector, scattered_selector = _wave_selectors(
            spec, is_3d=bool(data and data.is_3d)
        )
        if scattered_selector == "plus":
            selected_power = p_plus
            rejected_power = p_minus
        else:
            selected_power = p_minus
            rejected_power = p_plus
        selected_modal_net = selected_power - rejected_power

        def _flux_for_monitor(monitor_name):
            monitor_data = inputs.get(monitor_name)
            if monitor_data is None or "flux" not in monitor_data.fields:
                return np.full(freqs.shape, np.nan, dtype=float)
            return _resample_real_vector(
                monitor_data.frequencies, monitor_data.fields["flux"], freqs
            )

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
        diagnostics[name] = entry
    return diagnostics
