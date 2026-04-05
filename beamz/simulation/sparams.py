"""Port-wave and S-parameter extraction helpers."""

from __future__ import annotations

import numpy as np
from beamz.simulation.modal import (
    extract_cw_monitor_coefficients,
    extract_monitor_coefficients,
    extract_reference_waves,
)
from beamz.simulation.ports import (
    add_reference_power,
    assemble_s_matrix,
    build_port_wave_data,
    normalize_frequencies,
    require_supported_plane,
    resolve_source_frequencies,
    validate_port_monitors,
)


def _resolve_mode_strategy(mode_strategy):
    strategy = str(mode_strategy).lower()
    if strategy not in {"per_frequency", "single", "single_frequency", "center"}:
        raise ValueError(
            f"Unsupported mode_strategy '{mode_strategy}'. "
            "Use 'per_frequency' or 'single'."
        )
    return strategy
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
    validate_port_monitors(port_map, monitor_by_name, require_dft=require_dft)

    sample_cache = {}
    projection_cache = {}
    waves = {}

    for spec in port_map.values():
        main_monitor = monitor_by_name[spec.monitor_name]
        projection_frequencies = projection_frequencies_for(spec)
        a_plus, a_minus, cond_main, neff_main = extract_monitor_coefficients(
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
        port_waves = build_port_wave_data(
            a_plus,
            a_minus,
            return_power=return_power,
            condition_number=cond_main,
            mode_neff=neff_main,
        )

        if spec.reference_monitor:
            ref_monitor = monitor_by_name[spec.reference_monitor]
            port_waves.update(
                extract_reference_waves(
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
                add_reference_power(port_waves)

        waves[spec.name] = port_waves
    return waves
def extract_port_waves(
    sim,
    ports,
    frequencies,
    mode_strategy="per_frequency",
    window="hann",
    return_power=True,
):
    """Broadband modal extraction using FFT bins."""
    require_supported_plane(sim, function_name="extract_port_waves", allow_3d=True)

    port_map = sim._normalize_portspecs(ports)
    freqs = normalize_frequencies(frequencies)
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
    require_supported_plane(sim, function_name="extract_port_waves_dft", allow_3d=True)

    port_map = sim._normalize_portspecs(ports)
    freqs = normalize_frequencies(frequencies)
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
    freqs = resolve_source_frequencies(
        port_map,
        source_port,
        frequencies,
        resolver=lambda src_spec: sim._named_monitors()[
            src_spec.reference_monitor or src_spec.monitor_name
        ].get_dft_frequencies(),
    )

    waves = sim.extract_port_waves_dft(
        ports=ports,
        frequencies=freqs,
        min_incident_db=min_incident_db,
        return_power=True,
    )
    return assemble_s_matrix(
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
    require_supported_plane(
        sim, function_name="extract_port_waves_cw", allow_3d=False
    )

    port_map = sim._normalize_portspecs(ports)
    f = normalize_frequencies([frequency])[0]
    _resolve_mode_strategy(mode_strategy)

    monitor_by_name = sim._named_monitors()
    validate_port_monitors(port_map, monitor_by_name, require_dft=False)

    projection_cache = {}
    waves = {}
    for spec in port_map.values():
        main_monitor = monitor_by_name[spec.monitor_name]
        a_plus, a_minus = extract_cw_monitor_coefficients(
            sim,
            spec,
            main_monitor,
            f,
            projection_cache=projection_cache,
            steady_start_time=steady_start_time,
            avg_cycles=avg_cycles,
            window=window,
        )
        port_waves = build_port_wave_data(
            a_plus, a_minus, return_power=return_power
        )

        if spec.reference_monitor:
            ref_monitor = monitor_by_name[spec.reference_monitor]
            a_ref_plus, a_ref_minus = extract_cw_monitor_coefficients(
                sim,
                spec,
                ref_monitor,
                f,
                projection_cache=projection_cache,
                steady_start_time=steady_start_time,
                avg_cycles=avg_cycles,
                window=window,
            )
            port_waves.update({"a_incident": a_ref_plus, "a_incident_plus": a_ref_plus, "a_incident_minus": a_ref_minus})
            if return_power:
                add_reference_power(port_waves)

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
    freqs = resolve_source_frequencies(
        port_map,
        source_port,
        frequencies,
        resolver=lambda src_spec: sim._sample_monitor_component_spectrum(
            sim._named_monitors()[src_spec.reference_monitor or src_spec.monitor_name],
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
    return assemble_s_matrix(
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

    f = normalize_frequencies([frequency])[0]
    waves = sim.extract_port_waves_cw(
        ports=ports,
        frequency=f,
        steady_start_time=steady_start_time,
        avg_cycles=avg_cycles,
        window=window,
        mode_strategy=mode_strategy,
        return_power=True,
    )
    return assemble_s_matrix(
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
