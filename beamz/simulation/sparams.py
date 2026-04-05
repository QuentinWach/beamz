"""Port-wave and S-parameter extraction helpers."""

from __future__ import annotations

import numpy as np


def _require_supported_plane(sim, *, function_name, allow_3d):
    if allow_3d:
        if (not sim.is_3d) and sim.plane_2d != "xy":
            raise NotImplementedError(
                f"{function_name} currently supports 2D simulations in the xy plane."
            )
        return
    if sim.is_3d or sim.plane_2d != "xy":
        raise NotImplementedError(
            f"{function_name} currently supports 2D simulations in the xy plane."
        )


def _normalize_frequencies(frequencies):
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if freqs.size == 0:
        raise ValueError("frequencies must contain at least one value.")
    if np.any(freqs <= 0):
        raise ValueError("frequencies must be strictly positive.")
    return freqs


def _resolve_mode_strategy(mode_strategy):
    strategy = str(mode_strategy).lower()
    if strategy not in {"per_frequency", "single", "single_frequency", "center"}:
        raise ValueError(
            f"Unsupported mode_strategy '{mode_strategy}'. "
            "Use 'per_frequency' or 'single'."
        )
    return strategy


def _resolve_output_ports(port_map, output_ports):
    if output_ports is None:
        return list(port_map.keys())
    missing = [name for name in output_ports if name not in port_map]
    if missing:
        raise ValueError(f"output_ports contains unknown ports: {missing}")
    return list(output_ports)


def _validate_port_monitors(port_map, monitor_by_name, *, require_dft=False):
    for spec in port_map.values():
        main = monitor_by_name.get(spec.monitor_name)
        if main is None:
            raise ValueError(
                f"Missing monitor '{spec.monitor_name}' for port '{spec.name}'."
            )
        if require_dft and not getattr(main, "dft_enabled", False):
            raise ValueError(
                f"Monitor '{spec.monitor_name}' must be created with dft_enabled=True."
            )

        if spec.reference_monitor:
            ref = monitor_by_name.get(spec.reference_monitor)
            if ref is None:
                raise ValueError(
                    f"Missing reference monitor '{spec.reference_monitor}' for port '{spec.name}'."
                )
            if require_dft and not getattr(ref, "dft_enabled", False):
                raise ValueError(
                    f"Reference monitor '{spec.reference_monitor}' must have dft_enabled=True."
                )


def _wanted_components(sim, parts):
    if sim.is_3d:
        return tuple(
            parts.get(
                "projection_components_3d",
                (parts["e_component"], parts["h_component"]),
            )
        )
    return (parts["e_component"], parts["h_component"])


def _load_component_cache(cache, monitor, component, loader):
    key = (monitor.name, component)
    if key not in cache:
        _, cache[key] = loader(monitor, component)
    return cache[key]


def _resolve_projection(sim, spec, monitor, frequency, projection_cache, last_valid):
    proj = sim._build_port_projection(spec, monitor, float(frequency), projection_cache)
    proj_neff = float(proj.get("mode_neff", np.nan))
    if (not np.isfinite(proj_neff)) or (proj_neff <= 1e-6):
        if last_valid is not None:
            return last_valid, last_valid
        return proj, last_valid
    return proj, proj


def _modal_coefficients_from_samples(sim, proj, samples):
    proj_components = tuple(
        proj.get("components", (proj["e_component"], proj["h_component"]))
    )
    if sim.is_3d:
        field_components = {
            comp: np.asarray(samples[comp], dtype=np.complex128)
            for comp in proj_components
        }
        coeff = sim._project_modal_coefficients_3d(field_components, proj)
    else:
        field_vec = np.concatenate(
            [np.asarray(samples[comp], dtype=np.complex128) for comp in proj_components]
        )
        coeff = proj["pinv"] @ field_vec
    return np.asarray(coeff, dtype=np.complex128)


def _extract_monitor_coefficients(
    sim,
    spec,
    monitor,
    frequencies,
    *,
    projection_frequencies,
    projection_cache,
    sample_cache,
    loader,
    include_metadata=False,
):
    parts = sim._mode_components_for_port(spec)
    for component in _wanted_components(sim, parts):
        _load_component_cache(sample_cache, monitor, component, loader)

    a_plus = np.zeros(frequencies.size, dtype=np.complex128)
    a_minus = np.zeros(frequencies.size, dtype=np.complex128)
    condition_number = None
    mode_neff = None
    if include_metadata:
        condition_number = np.zeros(frequencies.size, dtype=float)
        mode_neff = np.full(frequencies.size, np.nan, dtype=float)

    last_valid_proj = None
    for idx, f_mode in enumerate(projection_frequencies):
        proj, last_valid_proj = _resolve_projection(
            sim, spec, monitor, f_mode, projection_cache, last_valid_proj
        )
        proj_components = tuple(
            proj.get("components", (proj["e_component"], proj["h_component"]))
        )
        coeff = _modal_coefficients_from_samples(
            sim,
            proj,
            {
                comp: sample_cache[(monitor.name, comp)][idx]
                for comp in proj_components
            },
        )
        a_plus[idx], a_minus[idx] = coeff[0], coeff[1]
        if include_metadata:
            condition_number[idx] = float(proj.get("condition_number", np.nan))
            mode_neff[idx] = float(proj.get("mode_neff", np.nan))

    return a_plus, a_minus, condition_number, mode_neff


def _extract_reference_waves(
    sim,
    spec,
    monitor,
    frequencies,
    *,
    projection_frequencies,
    projection_cache,
    sample_cache,
    loader,
    include_metadata=False,
):
    a_plus, a_minus, condition_number, mode_neff = _extract_monitor_coefficients(
        sim,
        spec,
        monitor,
        frequencies,
        projection_frequencies=projection_frequencies,
        projection_cache=projection_cache,
        sample_cache=sample_cache,
        loader=loader,
        include_metadata=include_metadata,
    )
    data = {
        "a_incident": a_plus,
        "a_incident_plus": a_plus,
        "a_incident_minus": a_minus,
    }
    if include_metadata:
        data["reference_condition_number"] = condition_number
        data["reference_mode_neff"] = mode_neff
    return data


def _extract_cw_monitor_coefficients(
    sim,
    spec,
    monitor,
    frequency,
    *,
    projection_cache,
    steady_start_time,
    avg_cycles,
    window,
):
    parts = sim._mode_components_for_port(spec)
    proj = sim._build_port_projection(spec, monitor, frequency, projection_cache)
    coeff = proj["pinv"] @ np.concatenate(
        [
            sim._demodulate_monitor_component(
                monitor,
                component,
                frequency=frequency,
                t_start=steady_start_time,
                avg_cycles=avg_cycles,
                window=window,
            )
            for component in (parts["e_component"], parts["h_component"])
        ]
    )
    return np.complex128(coeff[0]), np.complex128(coeff[1])


def _extract_port_waves_with_loader(
    sim,
    port_map,
    frequencies,
    *,
    loader,
    require_dft,
    projection_frequencies_for,
    include_metadata,
    return_power,
):
    monitor_by_name = sim._named_monitors()
    _validate_port_monitors(port_map, monitor_by_name, require_dft=require_dft)

    sample_cache = {}
    projection_cache = {}
    waves = {}

    for spec in port_map.values():
        main_monitor = monitor_by_name[spec.monitor_name]
        projection_frequencies = projection_frequencies_for(spec)
        a_plus, a_minus, cond_main, neff_main = _extract_monitor_coefficients(
            sim,
            spec,
            main_monitor,
            frequencies,
            projection_frequencies=projection_frequencies,
            projection_cache=projection_cache,
            sample_cache=sample_cache,
            loader=loader,
            include_metadata=include_metadata,
        )
        port_waves = _build_port_wave_data(
            a_plus,
            a_minus,
            return_power=return_power,
            condition_number=cond_main,
            mode_neff=neff_main,
        )

        if spec.reference_monitor:
            ref_monitor = monitor_by_name[spec.reference_monitor]
            port_waves.update(
                _extract_reference_waves(
                    sim,
                    spec,
                    ref_monitor,
                    frequencies,
                    projection_frequencies=projection_frequencies,
                    projection_cache=projection_cache,
                    sample_cache=sample_cache,
                    loader=loader,
                    include_metadata=include_metadata,
                )
            )
            if return_power:
                _add_reference_power(port_waves)

        waves[spec.name] = port_waves
    return waves


def _build_port_wave_data(
    a_plus,
    a_minus,
    *,
    return_power,
    condition_number=None,
    mode_neff=None,
):
    data = {"a_plus": a_plus, "a_minus": a_minus}
    if condition_number is not None:
        data["condition_number"] = condition_number
    if mode_neff is not None:
        data["mode_neff"] = mode_neff
    if return_power:
        data["P_plus"] = np.abs(a_plus) ** 2
        data["P_minus"] = np.abs(a_minus) ** 2
    return data


def _add_reference_power(data):
    data["P_incident"] = np.abs(data["a_incident_plus"]) ** 2
    data["P_incident_plus"] = np.abs(data["a_incident_plus"]) ** 2
    data["P_incident_minus"] = np.abs(data["a_incident_minus"]) ** 2


def _guided_output_power(sim, waves, port_map, output_ports):
    p_guided_out = None
    for out_port in output_ports:
        out_spec = port_map[out_port]
        component = np.abs(
            sim._select_wave_component(
                waves[out_port],
                selector=out_spec.scattered_wave,
                use_reference=False,
            )
        ) ** 2
        component = np.atleast_1d(np.asarray(component, dtype=float))
        if p_guided_out is None:
            p_guided_out = np.zeros_like(component, dtype=float)
        p_guided_out += component
    return p_guided_out


def _condition_number_summary(waves):
    return {
        name: {
            "monitor": np.asarray(data.get("condition_number", []), dtype=float),
            "reference": np.asarray(
                data.get("reference_condition_number", []), dtype=float
            ),
        }
        for name, data in waves.items()
    }


def _incident_valid_mask(a_incident, min_incident_db):
    max_incident = float(np.max(np.abs(a_incident))) if a_incident.size else 0.0
    rel_floor = max_incident * (10.0 ** (float(min_incident_db) / 20.0))
    abs_floor = max(1e-18, rel_floor)
    return np.abs(a_incident) >= abs_floor


def _resolve_source_frequencies(sim, port_map, source_port, frequencies, *, resolver):
    monitor_by_name = sim._named_monitors()
    if frequencies is not None:
        return _normalize_frequencies(frequencies)

    if source_port not in port_map:
        raise ValueError(f"source_port '{source_port}' not found in ports.")
    src_spec = port_map[source_port]
    ref_name = src_spec.reference_monitor or src_spec.monitor_name
    src_monitor = monitor_by_name.get(ref_name)
    if src_monitor is None:
        raise ValueError(f"Missing source/reference monitor '{ref_name}'.")
    return _normalize_frequencies(resolver(src_spec, src_monitor))


def _assemble_s_matrix(
    sim,
    *,
    source_port,
    port_map,
    waves,
    output_ports,
    frequencies,
    as_sax,
    return_diagnostics,
    min_incident_db=None,
    scalar_output=False,
    include_condition_numbers=False,
):
    output_ports = _resolve_output_ports(port_map, output_ports)
    source_spec = port_map[source_port]
    a_incident = np.atleast_1d(
        np.asarray(
            sim._select_wave_component(
                waves[source_port],
                selector=source_spec.incident_wave,
                use_reference=bool(source_spec.reference_monitor),
            ),
            dtype=np.complex128,
        )
    )

    valid_mask = None
    if min_incident_db is not None:
        valid_mask = _incident_valid_mask(a_incident, min_incident_db)

    s_matrix = {}
    for out_port in output_ports:
        out_spec = port_map[out_port]
        b_out = np.atleast_1d(
            np.asarray(
                sim._select_wave_component(
                    waves[out_port],
                    selector=out_spec.scattered_wave,
                    use_reference=False,
                ),
                dtype=np.complex128,
            )
        )
        ratio = sim._safe_ratio(b_out, a_incident)
        if valid_mask is not None:
            ratio = np.where(valid_mask, ratio, 0.0 + 0.0j)
        s_matrix[(out_port, source_port)] = (
            np.complex128(ratio[0]) if scalar_output else ratio
        )

    freq_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
    sim.s_matrix_frequencies = freq_arr
    s_output = sim._format_s_matrix_output(s_matrix, as_sax=as_sax)
    if not return_diagnostics:
        return s_output

    p_in = np.abs(a_incident) ** 2
    p_guided_out = _guided_output_power(sim, waves, port_map, output_ports)
    power_sum = p_guided_out / np.maximum(p_in, 1e-18)
    loss_est = 1.0 - power_sum

    if valid_mask is not None:
        power_sum = np.where(valid_mask, power_sum, np.nan)
        loss_est = np.where(valid_mask, loss_est, np.nan)

    diagnostics = {
        "source_port": source_port,
        "output_ports": output_ports,
        "waves": waves,
    }
    if scalar_output:
        diagnostics.update(
            {
                "frequency": float(freq_arr[0]),
                "P_in": float(p_in[0]),
                "P_guided_out": float(p_guided_out[0]),
                "power_sum": float(power_sum[0]),
                "loss_est": float(loss_est[0]),
            }
        )
    else:
        diagnostics.update(
            {
                "frequencies": freq_arr,
                "P_in": p_in,
                "P_guided_out": p_guided_out,
                "power_sum": power_sum,
                "loss_est": loss_est,
            }
        )
        if valid_mask is not None:
            diagnostics["valid_mask"] = valid_mask
        if include_condition_numbers:
            diagnostics["condition_numbers"] = _condition_number_summary(waves)

    return {"s_matrix": s_output, "diagnostics": diagnostics}


def extract_port_waves(
    sim,
    ports,
    frequencies,
    mode_strategy="per_frequency",
    window="hann",
    return_power=True,
):
    """Broadband modal extraction using FFT bins."""
    _require_supported_plane(
        sim, function_name="extract_port_waves", allow_3d=True
    )

    port_map = sim._normalize_portspecs(ports)
    freqs = _normalize_frequencies(frequencies)
    strategy = _resolve_mode_strategy(mode_strategy)
    single_freq = float(np.median(freqs))

    def loader(monitor, component):
        return sim._sample_monitor_component_spectrum(
            monitor, component, frequencies=freqs, window=window
        )

    return _extract_port_waves_with_loader(
        sim,
        port_map,
        freqs,
        loader=loader,
        require_dft=False,
        projection_frequencies_for=lambda _spec: (
            freqs if strategy == "per_frequency" else np.full(freqs.shape, single_freq)
        ),
        include_metadata=False,
        return_power=return_power,
    )


def extract_port_waves_dft(
    sim,
    ports,
    frequencies,
    min_incident_db=-40.0,
    return_power=True,
):
    """Extract modal port waves from in-simulation DFT monitor accumulators."""
    del min_incident_db
    _require_supported_plane(
        sim, function_name="extract_port_waves_dft", allow_3d=True
    )

    port_map = sim._normalize_portspecs(ports)
    freqs = _normalize_frequencies(frequencies)
    def loader(monitor, component):
        return sim._sample_monitor_component_dft(
            monitor, component, frequencies=freqs
        )

    return _extract_port_waves_with_loader(
        sim,
        port_map,
        freqs,
        loader=loader,
        require_dft=True,
        projection_frequencies_for=lambda _spec: freqs,
        include_metadata=True,
        return_power=return_power,
    )


def get_s_matrix_modal_dft(
    sim,
    source_port,
    ports,
    output_ports=None,
    frequencies=None,
    as_sax=True,
    return_diagnostics=True,
    min_incident_db=-40.0,
):
    """Broadband modal S extraction from in-simulation DFT monitor accumulators."""
    port_map = sim._normalize_portspecs(ports)
    freqs = _resolve_source_frequencies(
        sim,
        port_map,
        source_port,
        frequencies,
        resolver=lambda _spec, src_monitor: src_monitor.get_dft_frequencies(),
    )

    waves = sim.extract_port_waves_dft(
        ports=ports,
        frequencies=freqs,
        min_incident_db=min_incident_db,
        return_power=True,
    )
    return _assemble_s_matrix(
        sim,
        source_port=source_port,
        port_map=port_map,
        waves=waves,
        output_ports=output_ports,
        frequencies=freqs,
        as_sax=as_sax,
        return_diagnostics=return_diagnostics,
        min_incident_db=min_incident_db,
        scalar_output=False,
        include_condition_numbers=True,
    )


def extract_port_waves_cw(
    sim,
    ports,
    frequency,
    steady_start_time=None,
    avg_cycles=12,
    window="hann",
    mode_strategy="per_frequency",
    return_power=True,
):
    """CW modal extraction at one frequency using complex demodulation."""
    _require_supported_plane(
        sim, function_name="extract_port_waves_cw", allow_3d=False
    )

    port_map = sim._normalize_portspecs(ports)
    f = _normalize_frequencies([frequency])[0]
    _resolve_mode_strategy(mode_strategy)

    monitor_by_name = sim._named_monitors()
    _validate_port_monitors(port_map, monitor_by_name, require_dft=False)

    projection_cache = {}
    waves = {}
    for spec in port_map.values():
        main_monitor = monitor_by_name[spec.monitor_name]
        a_plus, a_minus = _extract_cw_monitor_coefficients(
            sim,
            spec,
            main_monitor,
            f,
            projection_cache=projection_cache,
            steady_start_time=steady_start_time,
            avg_cycles=avg_cycles,
            window=window,
        )
        port_waves = _build_port_wave_data(
            a_plus, a_minus, return_power=return_power
        )

        if spec.reference_monitor:
            ref_monitor = monitor_by_name[spec.reference_monitor]
            a_ref_plus, a_ref_minus = _extract_cw_monitor_coefficients(
                sim,
                spec,
                ref_monitor,
                f,
                projection_cache=projection_cache,
                steady_start_time=steady_start_time,
                avg_cycles=avg_cycles,
                window=window,
            )
            port_waves.update(
                {
                    "a_incident": a_ref_plus,
                    "a_incident_plus": a_ref_plus,
                    "a_incident_minus": a_ref_minus,
                }
            )
            if return_power:
                _add_reference_power(port_waves)

        waves[spec.name] = port_waves
    return waves


def get_s_matrix_modal(
    sim,
    source_port,
    ports,
    output_ports=None,
    frequencies=None,
    mode_strategy="per_frequency",
    as_sax=True,
    return_diagnostics=True,
):
    """Broadband modal S-matrix extraction from FFT-sampled monitor spectra."""
    port_map = sim._normalize_portspecs(ports)
    freqs = _resolve_source_frequencies(
        sim,
        port_map,
        source_port,
        frequencies,
        resolver=lambda src_spec, src_monitor: sim._sample_monitor_component_spectrum(
            src_monitor,
            sim._mode_components_for_port(src_spec)["e_component"],
            frequencies=None,
            window="hann",
        )[0],
    )

    waves = sim.extract_port_waves(
        ports=ports,
        frequencies=freqs,
        mode_strategy=mode_strategy,
        window="hann",
        return_power=True,
    )
    return _assemble_s_matrix(
        sim,
        source_port=source_port,
        port_map=port_map,
        waves=waves,
        output_ports=output_ports,
        frequencies=freqs,
        as_sax=as_sax,
        return_diagnostics=return_diagnostics,
        min_incident_db=None,
        scalar_output=False,
    )


def get_s_matrix_modal_cw(
    sim,
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
    """CW modal S extraction for one source/one frequency."""
    if frequency is None:
        raise ValueError("frequency is required for get_S_matrix_modal_cw.")

    port_map = sim._normalize_portspecs(ports)
    if source_port not in port_map:
        raise ValueError(f"source_port '{source_port}' not found in ports.")

    f = _normalize_frequencies([frequency])[0]
    waves = sim.extract_port_waves_cw(
        ports=ports,
        frequency=f,
        steady_start_time=steady_start_time,
        avg_cycles=avg_cycles,
        window=window,
        mode_strategy=mode_strategy,
        return_power=True,
    )
    return _assemble_s_matrix(
        sim,
        source_port=source_port,
        port_map=port_map,
        waves=waves,
        output_ports=output_ports,
        frequencies=[f],
        as_sax=as_sax,
        return_diagnostics=return_diagnostics,
        min_incident_db=None,
        scalar_output=True,
    )
