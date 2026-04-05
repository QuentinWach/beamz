"""Port-wave and S-parameter extraction helpers."""

from __future__ import annotations

import numpy as np


def extract_port_waves(
    sim,
    ports,
    frequencies,
    mode_strategy="per_frequency",
    window="hann",
    return_power=True,
):
    """Broadband modal extraction using FFT bins."""
    if (not sim.is_3d) and sim.plane_2d != "xy":
        raise NotImplementedError(
            "extract_port_waves currently supports 2D simulations in the xy plane."
        )

    port_map = sim._normalize_portspecs(ports)
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

    monitor_by_name = sim._named_monitors()
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
    waves = {}
    for spec in port_map.values():
        main_monitor = monitor_by_name[spec.monitor_name]
        parts = sim._mode_components_for_port(spec)
        wanted_components = (
            parts.get(
                "projection_components_3d",
                (parts["e_component"], parts["h_component"]),
            )
            if sim.is_3d
            else (parts["e_component"], parts["h_component"])
        )
        for comp in wanted_components:
            key = (main_monitor.name, comp)
            if key not in spectrum_cache:
                _, spectrum_cache[key] = sim._sample_monitor_component_spectrum(
                    main_monitor, comp, frequencies=freqs, window=window
                )

        a_plus = np.zeros(freqs.size, dtype=np.complex128)
        a_minus = np.zeros(freqs.size, dtype=np.complex128)
        last_valid_proj = None
        for idx, f in enumerate(freqs):
            f_mode = float(f if strategy == "per_frequency" else single_freq)
            proj = sim._build_port_projection(
                spec, main_monitor, f_mode, projection_cache
            )
            proj_neff = float(proj.get("mode_neff", np.nan))
            if (not np.isfinite(proj_neff)) or (proj_neff <= 1e-6):
                if last_valid_proj is not None:
                    proj = last_valid_proj
            else:
                last_valid_proj = proj
            proj_components = tuple(
                proj.get("components", (proj["e_component"], proj["h_component"]))
            )
            field_vec = np.concatenate(
                [
                    spectrum_cache[(main_monitor.name, comp)][idx]
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
            for comp in wanted_components:
                key = (ref_monitor.name, comp)
                if key not in spectrum_cache:
                    _, spectrum_cache[key] = sim._sample_monitor_component_spectrum(
                        ref_monitor, comp, frequencies=freqs, window=window
                    )

            a_incident_plus = np.zeros(freqs.size, dtype=np.complex128)
            a_incident_minus = np.zeros(freqs.size, dtype=np.complex128)
            last_valid_ref_proj = None
            for idx, f in enumerate(freqs):
                f_mode = float(f if strategy == "per_frequency" else single_freq)
                proj = sim._build_port_projection(
                    spec, ref_monitor, f_mode, projection_cache
                )
                proj_neff = float(proj.get("mode_neff", np.nan))
                if (not np.isfinite(proj_neff)) or (proj_neff <= 1e-6):
                    if last_valid_ref_proj is not None:
                        proj = last_valid_ref_proj
                else:
                    last_valid_ref_proj = proj
                proj_components = tuple(
                    proj.get("components", (proj["e_component"], proj["h_component"]))
                )
                field_vec = np.concatenate(
                    [
                        spectrum_cache[(ref_monitor.name, comp)][idx]
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
    sim,
    ports,
    frequencies,
    min_incident_db=-40.0,
    return_power=True,
):
    """Extract modal port waves from in-simulation DFT monitor accumulators."""
    del min_incident_db
    if (not sim.is_3d) and sim.plane_2d != "xy":
        raise NotImplementedError(
            "extract_port_waves_dft currently supports 2D simulations in the xy plane."
        )

    port_map = sim._normalize_portspecs(ports)
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if freqs.size == 0:
        raise ValueError("frequencies must contain at least one value.")
    if np.any(freqs <= 0):
        raise ValueError("frequencies must be strictly positive.")

    monitor_by_name = sim._named_monitors()
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
    waves = {}
    for spec in port_map.values():
        parts = sim._mode_components_for_port(spec)
        main_monitor = monitor_by_name[spec.monitor_name]
        wanted_components = (
            parts.get(
                "projection_components_3d",
                (parts["e_component"], parts["h_component"]),
            )
            if sim.is_3d
            else (parts["e_component"], parts["h_component"])
        )
        for comp in wanted_components:
            key = (main_monitor.name, comp)
            if key not in dft_cache:
                _, dft_cache[key] = sim._sample_monitor_component_dft(
                    main_monitor, comp, frequencies=freqs
                )

        a_plus = np.zeros(freqs.size, dtype=np.complex128)
        a_minus = np.zeros(freqs.size, dtype=np.complex128)
        cond_main = np.zeros(freqs.size, dtype=float)
        neff_main = np.full(freqs.size, np.nan, dtype=float)
        last_valid_proj = None
        for idx, f in enumerate(freqs):
            proj = sim._build_port_projection(spec, main_monitor, float(f), projection_cache)
            proj_neff = float(proj.get("mode_neff", np.nan))
            if (not np.isfinite(proj_neff)) or (proj_neff <= 1e-6):
                if last_valid_proj is not None:
                    proj = last_valid_proj
            else:
                last_valid_proj = proj
            proj_components = tuple(
                proj.get("components", (proj["e_component"], proj["h_component"]))
            )
            if sim.is_3d:
                field_components = {
                    comp: np.asarray(
                        dft_cache[(main_monitor.name, comp)][idx],
                        dtype=np.complex128,
                    )
                    for comp in proj_components
                }
                coeff = sim._project_modal_coefficients_3d(field_components, proj)
                a_plus[idx], a_minus[idx] = coeff[0], coeff[1]
            else:
                field_vec = np.concatenate(
                    [dft_cache[(main_monitor.name, comp)][idx] for comp in proj_components]
                )
                coeff = proj["pinv"] @ field_vec
                a_plus[idx], a_minus[idx] = coeff[0], coeff[1]
            cond_main[idx] = float(proj.get("condition_number", np.nan))
            neff_main[idx] = float(proj.get("mode_neff", np.nan))

        port_waves = {
            "a_plus": a_plus,
            "a_minus": a_minus,
            "condition_number": cond_main,
            "mode_neff": neff_main,
        }
        if return_power:
            port_waves["P_plus"] = np.abs(a_plus) ** 2
            port_waves["P_minus"] = np.abs(a_minus) ** 2

        if spec.reference_monitor:
            ref_monitor = monitor_by_name[spec.reference_monitor]
            for comp in wanted_components:
                key = (ref_monitor.name, comp)
                if key not in dft_cache:
                    _, dft_cache[key] = sim._sample_monitor_component_dft(
                        ref_monitor, comp, frequencies=freqs
                    )
            a_incident_plus = np.zeros(freqs.size, dtype=np.complex128)
            a_incident_minus = np.zeros(freqs.size, dtype=np.complex128)
            cond_ref = np.zeros(freqs.size, dtype=float)
            neff_ref = np.full(freqs.size, np.nan, dtype=float)
            last_valid_ref_proj = None
            for idx, f in enumerate(freqs):
                proj = sim._build_port_projection(spec, ref_monitor, float(f), projection_cache)
                proj_neff = float(proj.get("mode_neff", np.nan))
                if (not np.isfinite(proj_neff)) or (proj_neff <= 1e-6):
                    if last_valid_ref_proj is not None:
                        proj = last_valid_ref_proj
                else:
                    last_valid_ref_proj = proj
                proj_components = tuple(
                    proj.get("components", (proj["e_component"], proj["h_component"]))
                )
                if sim.is_3d:
                    field_components = {
                        comp: np.asarray(
                            dft_cache[(ref_monitor.name, comp)][idx],
                            dtype=np.complex128,
                        )
                        for comp in proj_components
                    }
                    coeff = sim._project_modal_coefficients_3d(field_components, proj)
                    a_incident_plus[idx], a_incident_minus[idx] = coeff[0], coeff[1]
                else:
                    field_vec = np.concatenate(
                        [dft_cache[(ref_monitor.name, comp)][idx] for comp in proj_components]
                    )
                    coeff = proj["pinv"] @ field_vec
                    a_incident_plus[idx], a_incident_minus[idx] = coeff[0], coeff[1]
                cond_ref[idx] = float(proj.get("condition_number", np.nan))
                neff_ref[idx] = float(proj.get("mode_neff", np.nan))
            port_waves["a_incident"] = a_incident_plus
            port_waves["a_incident_plus"] = a_incident_plus
            port_waves["a_incident_minus"] = a_incident_minus
            port_waves["reference_condition_number"] = cond_ref
            port_waves["reference_mode_neff"] = neff_ref
            if return_power:
                port_waves["P_incident"] = np.abs(a_incident_plus) ** 2
                port_waves["P_incident_plus"] = np.abs(a_incident_plus) ** 2
                port_waves["P_incident_minus"] = np.abs(a_incident_minus) ** 2

        waves[spec.name] = port_waves
    return waves


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
    if source_port not in port_map:
        raise ValueError(f"source_port '{source_port}' not found in ports.")

    monitor_by_name = sim._named_monitors()
    if frequencies is None:
        src_spec = port_map[source_port]
        ref_name = src_spec.reference_monitor or src_spec.monitor_name
        src_monitor = monitor_by_name.get(ref_name)
        if src_monitor is None:
            raise ValueError(f"Missing source/reference monitor '{ref_name}'.")
        frequencies = src_monitor.get_dft_frequencies()
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))

    waves = sim.extract_port_waves_dft(
        ports=port_map.values(),
        frequencies=frequencies,
        min_incident_db=min_incident_db,
        return_power=True,
    )

    if output_ports is None:
        output_ports = list(port_map.keys())
    else:
        output_ports = list(output_ports)
    missing = [name for name in output_ports if name not in port_map]
    if missing:
        raise ValueError(f"output_ports contains unknown ports: {missing}")

    source_spec = port_map[source_port]
    a_incident = sim._select_wave_component(
        waves[source_port],
        selector=source_spec.incident_wave,
        use_reference=bool(source_spec.reference_monitor),
    )
    a_incident = np.asarray(a_incident, dtype=np.complex128)
    max_incident = float(np.max(np.abs(a_incident))) if a_incident.size else 0.0
    rel_floor = max_incident * (10.0 ** (float(min_incident_db) / 20.0))
    abs_floor = max(1e-18, rel_floor)
    valid_mask = np.abs(a_incident) >= abs_floor

    s_matrix = {}
    for out_port in output_ports:
        out_spec = port_map[out_port]
        b_out = sim._select_wave_component(
            waves[out_port],
            selector=out_spec.scattered_wave,
            use_reference=False,
        )
        b_out = np.asarray(b_out, dtype=np.complex128)
        ratio = sim._safe_ratio(b_out, a_incident)
        ratio = np.where(valid_mask, ratio, 0.0 + 0.0j)
        s_matrix[(out_port, source_port)] = ratio

    sim.s_matrix_frequencies = np.asarray(frequencies, dtype=float)
    s_output = sim._format_s_matrix_output(s_matrix, as_sax=as_sax)

    if not return_diagnostics:
        return s_output

    p_in = np.abs(a_incident) ** 2
    p_guided_out = np.zeros_like(p_in, dtype=float)
    for out_port in output_ports:
        out_spec = port_map[out_port]
        p_guided_out += (
            np.abs(
                sim._select_wave_component(
                    waves[out_port],
                    selector=out_spec.scattered_wave,
                    use_reference=False,
                )
            )
            ** 2
        )
    power_sum = p_guided_out / np.maximum(p_in, 1e-18)
    loss_est = 1.0 - power_sum
    power_sum = np.where(valid_mask, power_sum, np.nan)
    loss_est = np.where(valid_mask, loss_est, np.nan)

    diagnostics = {
        "frequencies": np.asarray(frequencies, dtype=float),
        "source_port": source_port,
        "output_ports": output_ports,
        "waves": waves,
        "P_in": p_in,
        "P_guided_out": p_guided_out,
        "power_sum": power_sum,
        "loss_est": loss_est,
        "valid_mask": valid_mask,
        "condition_numbers": {
            name: {
                "monitor": np.asarray(data.get("condition_number", []), dtype=float),
                "reference": np.asarray(
                    data.get("reference_condition_number", []), dtype=float
                ),
            }
            for name, data in waves.items()
        },
    }
    return {"s_matrix": s_output, "diagnostics": diagnostics}


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
    if sim.is_3d or sim.plane_2d != "xy":
        raise NotImplementedError(
            "extract_port_waves_cw currently supports 2D simulations in the xy plane."
        )

    port_map = sim._normalize_portspecs(ports)
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

    monitor_by_name = sim._named_monitors()
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
        parts = sim._mode_components_for_port(spec)
        main_monitor = monitor_by_name[spec.monitor_name]
        proj = sim._build_port_projection(
            spec,
            main_monitor,
            f_mode if strategy == "per_frequency" else f,
            projection_cache,
        )
        e_main = sim._demodulate_monitor_component(
            main_monitor,
            parts["e_component"],
            frequency=f,
            t_start=steady_start_time,
            avg_cycles=avg_cycles,
            window=window,
        )
        h_main = sim._demodulate_monitor_component(
            main_monitor,
            parts["h_component"],
            frequency=f,
            t_start=steady_start_time,
            avg_cycles=avg_cycles,
            window=window,
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
            ref_proj = sim._build_port_projection(
                spec,
                ref_monitor,
                f_mode if strategy == "per_frequency" else f,
                projection_cache,
            )
            e_ref = sim._demodulate_monitor_component(
                ref_monitor,
                parts["e_component"],
                frequency=f,
                t_start=steady_start_time,
                avg_cycles=avg_cycles,
                window=window,
            )
            h_ref = sim._demodulate_monitor_component(
                ref_monitor,
                parts["h_component"],
                frequency=f,
                t_start=steady_start_time,
                avg_cycles=avg_cycles,
                window=window,
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
                port_waves["P_incident_minus"] = float(np.abs(a_incident_minus) ** 2)

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
    if source_port not in port_map:
        raise ValueError(f"source_port '{source_port}' not found in ports.")

    monitor_by_name = sim._named_monitors()
    if frequencies is None:
        src_spec = port_map[source_port]
        ref_name = src_spec.reference_monitor or src_spec.monitor_name
        src_monitor = monitor_by_name.get(ref_name)
        if src_monitor is None:
            raise ValueError(f"Missing source/reference monitor '{ref_name}'.")
        src_parts = sim._mode_components_for_port(src_spec)
        frequencies, _ = sim._sample_monitor_component_spectrum(
            src_monitor, src_parts["e_component"], frequencies=None, window="hann"
        )
    else:
        frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))

    waves = sim.extract_port_waves(
        ports=port_map.values(),
        frequencies=frequencies,
        mode_strategy=mode_strategy,
        window="hann",
        return_power=True,
    )

    if output_ports is None:
        output_ports = list(port_map.keys())
    else:
        output_ports = list(output_ports)
    missing = [name for name in output_ports if name not in port_map]
    if missing:
        raise ValueError(f"output_ports contains unknown ports: {missing}")

    source_spec = port_map[source_port]
    a_incident = sim._select_wave_component(
        waves[source_port],
        selector=source_spec.incident_wave,
        use_reference=bool(source_spec.reference_monitor),
    )
    s_matrix = {}
    for out_port in output_ports:
        out_spec = port_map[out_port]
        b_out = sim._select_wave_component(
            waves[out_port],
            selector=out_spec.scattered_wave,
            use_reference=False,
        )
        s_matrix[(out_port, source_port)] = sim._safe_ratio(b_out, a_incident)

    sim.s_matrix_frequencies = np.asarray(frequencies, dtype=float)
    s_output = sim._format_s_matrix_output(s_matrix, as_sax=as_sax)

    if not return_diagnostics:
        return s_output

    p_in = np.abs(a_incident) ** 2
    p_guided_out = np.zeros_like(p_in, dtype=float)
    for out_port in output_ports:
        out_spec = port_map[out_port]
        p_guided_out += (
            np.abs(
                sim._select_wave_component(
                    waves[out_port],
                    selector=out_spec.scattered_wave,
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
    }
    return {"s_matrix": s_output, "diagnostics": diagnostics}


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

    waves = sim.extract_port_waves_cw(
        ports=port_map.values(),
        frequency=frequency,
        steady_start_time=steady_start_time,
        avg_cycles=avg_cycles,
        window=window,
        mode_strategy=mode_strategy,
        return_power=True,
    )

    if output_ports is None:
        output_ports = list(port_map.keys())
    else:
        output_ports = list(output_ports)
    missing = [name for name in output_ports if name not in port_map]
    if missing:
        raise ValueError(f"output_ports contains unknown ports: {missing}")

    source_spec = port_map[source_port]
    a_incident = sim._select_wave_component(
        waves[source_port],
        selector=source_spec.incident_wave,
        use_reference=bool(source_spec.reference_monitor),
    )
    s_matrix = {}
    for out_port in output_ports:
        out_spec = port_map[out_port]
        b_out = sim._select_wave_component(
            waves[out_port],
            selector=out_spec.scattered_wave,
            use_reference=False,
        )
        b_vec = np.atleast_1d(np.asarray(b_out, dtype=np.complex128))
        a_vec = np.atleast_1d(np.asarray(a_incident, dtype=np.complex128))
        ratio = sim._safe_ratio(b_vec, a_vec)[0]
        s_matrix[(out_port, source_port)] = np.complex128(ratio)

    sim.s_matrix_frequencies = np.asarray([float(frequency)], dtype=float)
    s_output = sim._format_s_matrix_output(s_matrix, as_sax=as_sax)

    if not return_diagnostics:
        return s_output

    p_in = float(np.abs(np.atleast_1d(np.asarray(a_incident, dtype=np.complex128))[0]) ** 2)
    p_guided_out = float(
        np.sum(
            [
                np.abs(
                    sim._select_wave_component(
                        waves[out],
                        selector=port_map[out].scattered_wave,
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
    }
    return {"s_matrix": s_output, "diagnostics": diagnostics}
