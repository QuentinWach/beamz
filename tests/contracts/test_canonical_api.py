from dataclasses import fields

import matplotlib.pyplot as plt
import numpy as np
import pytest

import beamz as bz
from beamz.devices.sources.time import sample_source_waveforms
from beamz.simulation.observe import SourceNormalization
from beamz.simulation.observe import source_normalization as _source_normalization
from beamz.simulation.results import FieldMetadata, SimulationMetadata


def test_gaussian_pulse_spectrum_uses_fwidth_standard_deviation():
    pulse = bz.GaussianPulse(freq0=2.0e14, fwidth=2.0e13)

    rel = np.abs(
        pulse.spectrum([pulse.freq0, pulse.freq0 + pulse.fwidth], normalize=True)
    )

    np.testing.assert_allclose(rel, [1.0, np.exp(-0.5)], rtol=1e-12)


def test_gaussian_pulse_dft_normalization_includes_native_monitor_scale():
    pulse = bz.GaussianPulse(freq0=2.0e14, fwidth=2.0e13)

    norm = pulse.dft_normalization_spectrum([pulse.freq0])

    np.testing.assert_allclose(abs(norm[0]), 1.0 / (2.0 * np.pi), rtol=1e-12)


def test_source_normalization_uses_sampled_native_monitor_dft_when_time_is_available():
    pulse = bz.GaussianPulse(freq0=2.0e14, fwidth=2.0e13)
    dt = 1.0e-16
    time = np.arange(0.0, 20.0 / pulse.fwidth, dt)
    signal, _quadrature = pulse.sample(time)
    source = bz.ModeSource(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 1.0, 1.0),
        source_time=pulse,
        direction="+",
        mode_spec=bz.ModeSpec(polarization="te"),
    )

    normalization = _source_normalization([source], [pulse.freq0], time=time)
    assert normalization is not None
    norm = normalization.field_amplitude_norm

    expected = (
        2.0 / time.size * np.sum(signal * np.exp(1j * 2.0 * np.pi * pulse.freq0 * time))
    )
    np.testing.assert_allclose(norm, [expected], rtol=1e-12, atol=1e-15)
    assert abs(norm[0]) < abs(pulse.dft_normalization_spectrum([pulse.freq0])[0])


def test_mode_source_spectrum_includes_modal_power_response():
    pulse = bz.GaussianPulse(freq0=2.0e14, fwidth=2.0e13)
    freqs = np.asarray([0.9 * pulse.freq0, pulse.freq0, 1.1 * pulse.freq0])
    source = bz.ModeSource(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 1.0, 1.0),
        source_time=pulse,
        direction="+",
        mode_spec=bz.ModeSpec(polarization="te"),
    )

    norm = source.source_spectrum(freqs, normalize=True)

    expected = pulse.dft_normalization_spectrum(freqs) * np.sqrt(freqs / pulse.freq0)
    np.testing.assert_allclose(norm, expected, rtol=1e-12)


def test_mode_source_defaults_to_one_watt_launch_power():
    source = bz.ModeSource(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 1.0, 1.0),
        source_time=bz.SampledSignal(np.ones(8, dtype=float), dt=1e-16, freq0=2e14),
        direction="+",
    )

    assert source.power == 1.0
    assert tuple(field.name for field in fields(bz.ModeSource)) == (
        "center",
        "size",
        "source_time",
        "direction",
        "mode_spec",
        "power",
    )


def test_port_creates_matching_canonical_monitor_and_source():
    port = bz.Port(
        center=(0.0, 1.0, 2.0),
        size=(0.0, 2.0, 3.0),
        name="input",
        direction="+",
        mode_spec=bz.ModeSpec(num_modes=2, polarization="te"),
    )

    monitor = port.to_monitor([1.9e14, 2.0e14])
    source = port.to_source(2.0e14, 2.0e13, mode_index=1, num_freqs=3)

    assert isinstance(monitor, bz.ModeMonitor)
    assert monitor.center == port.center and monitor.size == port.size
    assert monitor.mode_spec == port.mode_spec
    assert isinstance(source, bz.ModeSource)
    assert source.signed_direction == "+x"
    assert source.mode_spec.mode_index == 1
    assert source.mode_spec.num_freqs == 3
    assert tuple(field.name for field in fields(bz.Port)) == (
        "center",
        "size",
        "name",
        "direction",
        "mode_spec",
        "monitor_name",
    )


@pytest.mark.parametrize(
    "monitor_type", [bz.FieldMonitor, bz.FluxMonitor, bz.ModeMonitor]
)
def test_frequency_domain_monitors_require_positive_frequencies(monitor_type):
    kwargs = {
        "center": (0.0, 0.0, 0.0),
        "size": (0.0, 1.0, 1.0),
        "freqs": [],
    }

    with pytest.raises(ValueError, match="at least one frequency"):
        monitor_type(**kwargs)
    with pytest.raises(ValueError, match="positive"):
        monitor_type(**(kwargs | {"freqs": [0.0]}))


def test_legacy_generic_monitor_and_portspec_are_not_exported():
    assert not hasattr(bz, "Monitor")
    assert not hasattr(bz, "PortSpec")


def test_source_normalization_separates_waveform_and_launch_power():
    normalization = SourceNormalization(
        waveform_spectrum=np.array([2.0 + 0.0j, 0.0 + 3.0j]),
        launch_power_ratio=np.array([4.0, 0.25]),
    )

    np.testing.assert_allclose(normalization.field_amplitude_norm, [4.0, 1.5j])
    np.testing.assert_allclose(normalization.power_norm, [16.0, 2.25])


def test_monitor_results_apply_distinct_flux_and_mode_normalization(monkeypatch):
    freq = 2.0e14
    normalization = SourceNormalization(
        waveform_spectrum=np.array([2.0 + 0.0j]),
        launch_power_ratio=np.array([9.0]),
    )
    monkeypatch.setattr(
        "beamz.simulation.observe.normalization_from_result",
        lambda results, monitor: normalization,
    )
    flux_monitor = bz.FluxMonitor(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 1.0, 1.0),
        freqs=[freq],
        name="flux",
    )
    flux_result = bz.MonitorResults(
        monitor=flux_monitor,
        fields={},
        power_history=np.asarray([], dtype=float),
        power_timestamps=np.asarray([], dtype=float),
        power_spectrum=np.asarray([36.0]),
        dft_frequencies=np.asarray([freq]),
    )
    np.testing.assert_allclose(flux_result.flux, [36.0])

    mode_monitor = bz.ModeMonitor(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 1.0, 1.0),
        freqs=[freq],
        mode_spec=bz.ModeSpec(num_modes=1),
        name="mode",
    )
    mode_result = bz.MonitorResults(
        monitor=mode_monitor,
        fields={},
        power_history=np.asarray([], dtype=float),
        power_timestamps=np.asarray([], dtype=float),
        power_spectrum=np.asarray([], dtype=np.complex64),
    )

    def fake_extract(*args, **kwargs):
        return {
            "mode_m0": {
                "a_minus": np.array([0.0 + 0.0j]),
                "a_plus": np.array([6.0 + 0.0j]),
                "projected_signed_power": np.array([36.0]),
                "projection_residual": np.array([0.02]),
                "condition_number": np.array([1.5]),
            }
        }

    from beamz.analysis import sparameters as sp

    monkeypatch.setattr(sp, "_extract_port_waves_dft", fake_extract)
    results = bz.SimulationResults(
        metadata=SimulationMetadata(
            dt=1.0,
            resolution=1.0,
            is_3d=False,
            plane_2d="xy",
            coordinate_offset=(0.0, 0.0, 0.0),
            time=np.array([0.0]),
            width=1.0,
            height=1.0,
            depth=0.0,
            fields=FieldMetadata(grid_shape=(1, 1), component_shapes={}),
        ),
        monitors={"mode": mode_result},
    )
    assert results["mode"] is mode_result
    data = results.mode("mode")

    np.testing.assert_allclose(data.amps.sel(direction="+").values[:, 0], [6.0])
    np.testing.assert_allclose(data.modal_flux, [36.0])
    np.testing.assert_allclose(data.projection_residual, [0.02])
    np.testing.assert_allclose(data.condition_number, [1.5])


def test_3d_mode_monitor_labels_positive_axis_power_as_forward(monkeypatch):
    freq = 2.0e14
    monitor = bz.ModeMonitor(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 1.0, 1.0),
        freqs=[freq],
        mode_spec=bz.ModeSpec(num_modes=1),
        name="mode",
    )
    result = bz.MonitorResults(
        monitor=monitor,
        fields={},
        power_history=np.asarray([], dtype=float),
        power_timestamps=np.asarray([], dtype=float),
        power_spectrum=np.asarray([], dtype=np.complex64),
    )

    def fake_extract(*args, **kwargs):
        return {
            "mode_m0": {
                # The 3D x-normal discrete contract uses the raw minus branch
                # for physical +x propagation.
                "a_minus": np.array([2.0 + 0.0j]),
                "a_plus": np.array([0.25 + 0.0j]),
                "projected_signed_power": np.array([3.9375]),
                "projection_residual": np.array([0.0]),
                "condition_number": np.array([1.0]),
            }
        }

    from beamz.analysis import sparameters as sp

    monkeypatch.setattr(sp, "_extract_port_waves_dft", fake_extract)
    results = bz.SimulationResults(
        metadata=SimulationMetadata(
            dt=1.0,
            resolution=1.0,
            is_3d=True,
            plane_2d="xy",
            coordinate_offset=(0.0, 0.0, 0.0),
            time=np.array([0.0]),
            width=1.0,
            height=1.0,
            depth=1.0,
            fields=FieldMetadata(grid_shape=(1, 1, 1), component_shapes={}),
        ),
        monitors={"mode": result},
    )

    data = results.mode("mode")

    np.testing.assert_allclose(data.amps.sel(direction="+").values[:, 0], [2.0])
    np.testing.assert_allclose(data.amps.sel(direction="-").values[:, 0], [0.25])


def test_mode_monitor_analysis_exposes_flux_computed_from_dft_fields():
    freq = 2.0e14
    monitor = bz.ModeMonitor(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 1.0, 1.0),
        freqs=[freq],
        mode_spec=bz.ModeSpec(num_modes=1),
        name="mode",
    )
    zeros = np.zeros((1, 2), dtype=np.complex128)
    result = bz.MonitorResults(
        monitor=monitor,
        fields={},
        power_history=np.asarray([], dtype=float),
        power_timestamps=np.asarray([], dtype=float),
        power_spectrum=np.asarray([], dtype=np.complex64),
        dft_fields={
            "Ex": zeros,
            "Ey": np.ones((1, 2), dtype=np.complex128),
            "Ez": zeros,
            "Hx": zeros,
            "Hy": zeros,
            "Hz": 0.25 * np.ones((1, 2), dtype=np.complex128),
        },
        dft_frequencies=np.asarray([freq]),
        dft_weight_sum=np.asarray([2.0]),
        resolution=1.0,
        power_scale=0.5,
    )
    results = bz.SimulationResults(
        metadata=SimulationMetadata(
            dt=1.0,
            resolution=1.0,
            is_3d=True,
            plane_2d="xy",
            coordinate_offset=(0.0, 0.0, 0.0),
            time=np.array([0.0]),
            width=1.0,
            height=1.0,
            depth=1.0,
            fields=FieldMetadata(grid_shape=(1, 1, 1), component_shapes={}),
        ),
        monitors={"mode": result},
    )

    from beamz.analysis.data import analysis_data

    lowered = analysis_data(results, "mode")

    np.testing.assert_allclose(lowered.fields["flux"], result.get_dft_flux())
    assert np.all(np.isfinite(lowered.fields["flux"]))


def test_legacy_mode_solver_public_api_is_removed():
    assert not hasattr(bz, "ModeSolver")
    assert not hasattr(bz, "solve_modes")
    with pytest.raises(ModuleNotFoundError):
        __import__("beamz.devices.sources.modesolver", fromlist=["ModeSolver"])


def test_design_background_and_material_geometry_builds_design_and_time():
    si = bz.Material(permittivity=12.0)
    sio2 = bz.Material(permittivity=2.0)
    grid_spec = bz.GridSpec.auto(min_steps_per_wvl=10, wavelength=1.55 * bz.um)
    design = bz.Design(background=sio2)
    design += bz.Box(
        center=(0, 0, -1 * bz.um),
        size=(bz.inf, bz.inf, 2 * bz.um),
        material=sio2,
    )
    design += bz.Box(
        center=(0, 0, 0.11 * bz.um),
        size=(4 * bz.um, 0.45 * bz.um, 0.22 * bz.um),
        material=si,
    )

    sim = bz.Simulation(
        domain=(4 * bz.um, 3 * bz.um, 2 * bz.um),
        grid_spec=grid_spec,
        design=design,
        sources=[],
        monitors=[],
        run_time=2e-15,
    )

    assert sim.design.width == 4 * bz.um
    assert sim.domain == (4 * bz.um, 3 * bz.um, 2 * bz.um)
    assert sim.design.depth == 2 * bz.um
    assert isinstance(sim.grid, bz.RectilinearGrid)
    assert sim.grid.minimum_spacing < 1.55 * bz.um / 10
    assert sim.resolution == pytest.approx(sim.grid.minimum_spacing)
    assert sim.time.size >= 2


def test_simulation_rejects_conflicting_domain_and_size():
    with pytest.raises(ValueError, match="domain"):
        bz.Simulation(
            domain=(2.0, 2.0, 1.0),
            size=(2.0, 3.0, 1.0),
            sources=[],
            monitors=[],
            resolution=0.5,
            time=np.array([0.0, 1e-15]),
        )


def test_nonempty_simulation_structures_api_is_removed():
    sio2 = bz.Material(permittivity=2.0)
    core = bz.Box(center=(0.0, 0.0, 0.0), size=(1.0, 0.5, 0.2))

    with pytest.raises(TypeError, match="structures"):
        bz.Simulation(
            size=(2.0, 2.0, 1.0),
            background=sio2,
            structures=[core],
            sources=[],
            monitors=[],
            resolution=0.5,
            time=np.array([0.0, 1e-15]),
        )

    assert not hasattr(bz, "Structure")
    assert not hasattr(bz, "Medium")


def test_semantic_monitor_wrappers_create_dft_planes_and_shift_with_simulation():
    freqs = np.array([1.0, 2.0])
    monitor = bz.FluxMonitor(
        center=(1.0, 0.0, 0.0),
        size=(0.0, 2.0, 3.0),
        freqs=freqs,
        name="flux",
    )

    sim = bz.Simulation(
        size=(10.0, 8.0, 6.0),
        sources=[],
        monitors=[monitor],
        resolution=1.0,
        time=np.array([0.0, 1e-15]),
    )

    shifted = sim.monitors[0]
    assert shifted.name == "flux"
    assert shifted.freqs.size > 0
    np.testing.assert_allclose(shifted.get_dft_frequencies(), freqs)
    assert shifted.start[0] == 6.0
    assert shifted.end[0] == 6.0


@pytest.mark.parametrize(
    ("size", "expected"),
    (
        ((0.0, 2.0, 3.0), {"Ey", "Ez", "Hy", "Hz"}),
        ((2.0, 0.0, 3.0), {"Ex", "Ez", "Hx", "Hz"}),
        ((2.0, 3.0, 0.0), {"Ex", "Ey", "Hx", "Hy"}),
    ),
)
def test_flux_monitor_records_only_tangential_poynting_components(size, expected):
    monitor = bz.FluxMonitor(
        center=(0.0, 0.0, 0.0),
        size=size,
        freqs=(2.0e14,),
    )

    assert set(monitor.dft_components) == expected


def test_flux_monitor_result_is_source_spectrum_normalized(monkeypatch):
    freq0 = 2.0e14
    fwidth = 2.0e13
    freqs = np.array([freq0, freq0 + fwidth])
    source_time = bz.GaussianPulse(freq0=freq0, fwidth=fwidth)
    source_norm = source_time.dft_normalization_spectrum(freqs)
    flux_monitor = bz.FluxMonitor(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 2.0, 2.0),
        freqs=freqs,
        name="flux",
    )
    monkeypatch.setattr(
        type(flux_monitor),
        "get_dft_flux",
        lambda self: 4.0 * np.abs(source_norm) ** 2,
        raising=False,
    )
    sim = bz.Simulation(
        size=(4.0, 4.0, 4.0),
        sources=[],
        monitors=[flux_monitor],
        resolution=1.0,
        time=np.array([0.0, 1e-15]),
    )
    source = bz.GaussianBeamSource(
        center=(0.0, 0.0, 0.0),
        size=(1.0, 1.0),
        source_time=source_time,
        wavelength=bz.LIGHT_SPEED / freq0,
        waist_radius=0.5,
    )

    monitor_result = bz.MonitorResults(
        monitor=flux_monitor,
        fields={},
        power_history=np.empty(0),
        power_timestamps=np.empty(0),
        power_spectrum=4.0 * np.abs(source_norm) ** 2,
        dft_frequencies=freqs,
    )
    grid = sim.compile().grid
    sim = sim.updated_copy(sources=[source])
    results = bz.SimulationResults.from_run(
        sim,
        runtime_fields=grid,
        monitor_results={"flux": monitor_result},
    )

    np.testing.assert_allclose(
        results.monitors["flux"].flux,
        np.full(freqs.shape, 4.0),
    )
    raw = results.renormalize(None)
    np.testing.assert_allclose(
        raw.monitors["flux"].flux,
        4.0 * np.abs(source_norm) ** 2,
    )
    np.testing.assert_allclose(
        raw.renormalize(0).monitors["flux"].flux,
        np.full(freqs.shape, 4.0),
    )


def test_flux_monitor_result_includes_mode_source_launch_calibration(monkeypatch):
    freq0 = 2.0e14
    fwidth = 2.0e13
    freqs = np.array([freq0, freq0 + fwidth])
    source_time = bz.GaussianPulse(freq0=freq0, fwidth=fwidth)
    launch_power = np.array([1.25, 0.75])
    source = bz.ModeSource(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 1.0, 1.0),
        source_time=source_time,
        direction="+",
        mode_spec=bz.ModeSpec(polarization="te"),
    )
    source_norm = source.source_spectrum(freqs, normalize=True)

    def launch_power_spectrum(source_arg, values, **kwargs):
        del source_arg, kwargs
        values = np.asarray(values, dtype=float)
        if values.shape != freqs.shape or not np.allclose(values, freqs):
            return None
        return launch_power

    monkeypatch.setattr(
        bz.ModeSource,
        "launch_power_normalization_spectrum",
        launch_power_spectrum,
    )

    flux_monitor = bz.FluxMonitor(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 2.0, 2.0),
        freqs=freqs,
        name="flux",
    )
    monkeypatch.setattr(
        type(flux_monitor),
        "get_dft_flux",
        lambda self: 4.0 * np.abs(source_norm) ** 2 * launch_power,
        raising=False,
    )
    sim = bz.Simulation(
        size=(4.0, 4.0, 4.0),
        sources=[],
        monitors=[flux_monitor],
        resolution=1.0,
        time=np.array([0.0, 1e-15]),
    )

    monitor_result = bz.MonitorResults(
        monitor=flux_monitor,
        fields={},
        power_history=np.empty(0),
        power_timestamps=np.empty(0),
        power_spectrum=4.0 * np.abs(source_norm) ** 2 * launch_power,
        dft_frequencies=freqs,
    )
    grid = sim.compile().grid
    sim = sim.updated_copy(sources=[source])
    results = bz.SimulationResults.from_run(
        sim,
        runtime_fields=grid,
        monitor_results={"flux": monitor_result},
    )

    np.testing.assert_allclose(
        results.monitors["flux"].flux,
        np.full(freqs.shape, 4.0),
    )


def test_mode_source_can_create_config_from_source_time_plane():
    sim = bz.Simulation(
        size=(2.0, 2.0, 2.0),
        sources=[],
        monitors=[],
        resolution=1.0,
        time=np.linspace(0.0, 3e-15, 4),
    )
    plane = bz.Box(center=(-0.5, 0.0, 0.0), size=(0.0, 1.0, 1.0))
    source_time = bz.GaussianPulse(freq0=2.0e14, fwidth=2.0e13)

    source = bz.ModeSource(
        center=plane.center,
        size=plane.size,
        direction="+",
        source_time=source_time,
        mode_spec=bz.ModeSpec(num_modes=1, polarization="te"),
    )

    assert source.direction == "+"
    assert source.signed_direction == "+x"
    assert source.source_time == source_time
    expected, _quadrature = source_time.sample(np.asarray([sim.time[0]]))
    sampled, _quadrature = source.source_time.sample(np.asarray([sim.time[0]]))
    np.testing.assert_allclose(sampled, expected)


def test_mode_source_solves_modes_and_analysis_plots(monkeypatch):
    sim = bz.Simulation(
        size=(6.0, 6.0, 6.0),
        sources=[],
        monitors=[],
        resolution=1.0,
        time=np.linspace(0.0, 3e-15, 4),
    )
    plane = bz.Box(center=(0.0, 0.0, 0.0), size=(0.0, 2.0, 4.0))
    source_time = bz.GaussianPulse(freq0=2.0e14, fwidth=2.0e13)
    source = bz.ModeSource(
        center=plane.center,
        size=plane.size,
        direction="+",
        source_time=source_time,
        mode_spec=bz.ModeSpec(num_modes=2, polarization="te", target_neff=2.3),
    )
    seen = {}

    def fake_solve_modes(**kwargs):
        eps = np.asarray(kwargs["eps"])
        seen["eps_shape"] = eps.shape
        seen["direction"] = kwargs["direction"]
        fields = np.ones((2, 3, *eps.shape), dtype=np.complex128)
        return np.array([2.4 + 0.0j, 1.8 + 0.0j]), fields, fields, 0

    monkeypatch.setattr("beamz.devices.sources.solve.solve_modes", fake_solve_modes)

    modes = source.solve_modes(sim, freqs=[source_time.freq0])
    from beamz.analysis.plotting import plot_mode_field_components

    fig, axes, neffs = plot_mode_field_components(
        modes,
        field_names=("Ey", "Ez"),
        mode_indices=(0, 1),
        show=False,
    )

    assert source.direction == "+"
    assert source.signed_direction == "+x"
    assert not hasattr(source, "with_modes")
    assert not hasattr(source, "mode_e_field")
    assert not hasattr(modes, "plot_field_components")
    assert seen["eps_shape"] == (4, 2)
    assert seen["direction"] == "+x"
    assert axes.shape == (2, 2)
    np.testing.assert_allclose(neffs, [2.4 + 0.0j, 1.8 + 0.0j])
    np.testing.assert_allclose(modes.neffs[0], [2.4 + 0.0j, 1.8 + 0.0j])
    plt.close(fig)


def test_mode_source_samples_source_time_without_precomputed_signal():
    pulse = bz.GaussianPulse(freq0=2.0e14, fwidth=2.0e13)
    source = bz.ModeSource(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 1.0, 1.0),
        direction="+",
        source_time=pulse,
        mode_spec=bz.ModeSpec(num_modes=1, polarization="te"),
    )

    signal, quadrature = pulse.sample(np.asarray([1.25e-14]))
    sampled_signal, sampled_quadrature = source.source_time.sample(
        np.asarray([1.25e-14])
    )

    assert not hasattr(source, "signal")
    np.testing.assert_allclose(sampled_signal, signal)
    np.testing.assert_allclose(sampled_quadrature, quadrature)


def test_mode_source_compiler_samples_source_time_as_full_waveform():
    pulse = bz.GaussianPulse(freq0=2.0e14, fwidth=2.0e13)
    source = bz.ModeSource(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 1.0, 1.0),
        direction="+",
        source_time=pulse,
        mode_spec=bz.ModeSpec(num_modes=1, polarization="te"),
    )

    signal, quadrature = sample_source_waveforms(
        source.source_time,
        t0=0.0,
        dt=5.0e-16,
        num_steps=1024,
        offset_fn=lambda t, step: t + 0.5 * step,
    )
    analytic_real, analytic_quadrature = sample_source_waveforms(
        source.source_time,
        t0=0.0,
        dt=5.0e-16,
        num_steps=1024,
    )
    analytic = np.asarray(analytic_real) + 1j * np.asarray(analytic_quadrature)

    assert np.max(np.abs(np.asarray(signal))) > 0.1
    assert np.max(np.abs(np.asarray(quadrature))) > 0.1
    assert np.max(np.abs(np.real(analytic))) > 0.1
    assert np.max(np.abs(np.imag(analytic))) > 0.1


def test_simulation_copy_update_normalizes_replaced_sources_once():
    source = bz.ModeSource(
        center=(-0.5, 0.0, 0.0),
        size=(0.0, 1.0, 1.0),
        direction="+",
        source_time=bz.GaussianPulse(freq0=2.0e14, fwidth=2.0e13),
        mode_spec=bz.ModeSpec(num_modes=1),
    )
    sim0 = bz.Simulation(
        size=(2.0, 2.0, 2.0),
        sources=[],
        monitors=[],
        resolution=1.0,
        time=np.linspace(0.0, 3e-15, 4),
    )

    sim = sim0.updated_copy(sources=[source])

    assert sim is not sim0
    assert sim.sources != [source]
    assert sim.sources[0].center == pytest.approx((0.5, 1.0, 1.0))
    assert source.center == pytest.approx((-0.5, 0.0, 0.0))
    assert sim0.sources == ()
    assert sim.coordinate_offset == sim0.coordinate_offset
    assert sim.initial_state().current_step == 0


def test_mode_source_rejects_precomputed_runtime_launch_fields():
    with pytest.raises(TypeError, match="unexpected keyword"):
        bz.ModeSource(
            center=(1.0, 1.0, 1.0),
            size=(0.0, 2.0, 2.0),
            source_time=bz.SampledSignal(np.ones(4), dt=1e-16, freq0=2e14),
            direction="+",
            mode_neff=2.25 + 0.0j,
            mode_e_field=np.ones((3, 3, 3), dtype=np.complex128),
        )


def test_mode_data_dataframe_exposes_lightweight_inspection_columns():
    data = bz.ModeData(
        frequencies=np.array([2.0e14]),
        neffs=np.array([[2.4 + 0.0j, 1.5 + 0.0j]]),
        e_fields=np.ones((1, 2, 3, 2, 2), dtype=np.complex128),
        h_fields=np.ones((1, 2, 3, 2, 2), dtype=np.complex128),
        eps_profiles=np.array([[[1.0, 12.0], [1.0, 12.0]]]),
        resolution=0.1 * bz.um,
    )

    from beamz.analysis import mode_data_to_dataframe

    df = mode_data_to_dataframe(data)

    assert not hasattr(data, "to_dataframe")
    assert list(df.columns) == [
        "wavelength",
        "n eff",
        "k eff",
        "loss (dB/cm)",
        "mode area",
    ]
    assert df.index.names == ["f", "mode_index"]
