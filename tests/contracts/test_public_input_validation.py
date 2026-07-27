"""Broad public-API contracts for configuration validation and edge behavior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import beamz as bz


@dataclass(frozen=True)
class _WrongModeSpec:
    num_modes: int = 1


@dataclass(frozen=True)
class _SampleOnly:
    freq0: float = 2e14

    def sample(self, time):
        values = np.zeros_like(np.asarray(time, dtype=float))
        return values, values


@dataclass(frozen=True)
class _SpectrumOnly:
    freq0: float = 2e14

    def sample(self, time):
        values = np.zeros_like(np.asarray(time, dtype=float))
        return values, values

    def spectrum(self, freqs, *, normalize=False):
        del normalize
        return np.full(np.asarray(freqs).shape, 3.0 + 2.0j)


@dataclass(frozen=True)
class _LegacySpectrumOnly:
    def spectrum(self, freqs):
        return np.full(np.asarray(freqs).shape, 3.0 + 2.0j)


def _port(**changes):
    values = {
        "center": (0.0, 0.0, 0.0),
        "size": (0.0, 2.0, 1.0),
        "name": "in",
        "direction": "+",
    }
    values.update(changes)
    return bz.Port(**values)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"center": (0.0, 0.0)}, "three values"),
        ({"center": (np.inf, 0.0, 0.0)}, "finite"),
        ({"size": (0.0, -1.0, 1.0)}, "non-negative"),
        ({"size": (0.0, 0.0, 1.0)}, "exactly one zero"),
        ({"name": ""}, "name cannot be empty"),
        ({"direction": "forward"}, "direction must"),
        ({"monitor_name": ""}, "monitor_name cannot be empty"),
        ({"mode_spec": _WrongModeSpec()}, "must be a ModeSpec"),
    ],
)
def test_port_rejects_invalid_configuration(changes, match):
    with pytest.raises((TypeError, ValueError), match=match):
        _port(**changes)


def test_port_derivations_translation_and_device_factories_are_consistent():
    port = _port(mode_spec=bz.ModeSpec(num_modes=3, mode_index=1))

    assert port.axis == "x"
    assert port.signed_direction == "+x"
    assert port.projection_direction == "+x"
    assert port.num_modes == 3
    assert port.mode_index == 1
    assert port.polarization == "te"
    assert port.mode_spec.polarization == "te"

    shifted = port.shifted((1.0, 2.0, 3.0))
    assert shifted.center == (1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="three values"):
        port.shifted((1.0, 2.0))

    monitor = port.to_monitor([1e14, 2e14])
    assert monitor.center == port.center
    assert monitor.name == port.monitor_name
    assert monitor.mode_spec is port.mode_spec

    default_source = port.to_source(2e14, 2e13)
    assert default_source.mode_spec.mode_index == port.mode_index
    assert default_source.mode_spec.polarization == port.polarization

    generated = port.to_source(2e14, 2e13, mode_index=2, num_freqs=0)
    assert isinstance(generated.source_time, bz.GaussianPulse)
    assert generated.mode_spec.mode_index == 2
    assert generated.mode_spec.num_freqs == 1

    supplied_time = bz.GaussianPulse(1.5e14, 1e13)
    supplied = port.to_source(2e14, 2e13, source_time=supplied_time)
    assert supplied.source_time is supplied_time


def _field_monitor(**changes):
    values = {
        "center": (0.0, 0.0, 0.0),
        "size": (0.0, 2.0, 1.0),
        "freqs": [2e14],
    }
    values.update(changes)
    return bz.FieldMonitor(**values)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"center": (0.0,)}, "2D or 3D"),
        ({"center": (np.nan, 0.0, 0.0)}, "finite"),
        ({"size": (0.0, -1.0, 1.0)}, "non-negative"),
        ({"freqs": [0.0]}, "positive"),
        ({"freqs": [np.inf]}, "finite"),
        ({"freqs": []}, "at least one"),
        ({"fields": ()}, "Unsupported"),
        ({"fields": ("Ez", "bad")}, "bad"),
    ],
)
def test_field_monitor_rejects_invalid_configuration(changes, match):
    with pytest.raises(ValueError, match=match):
        _field_monitor(**changes)


@pytest.mark.parametrize("monitor_type", [bz.FluxMonitor, bz.ModeMonitor])
def test_planar_monitors_require_one_zero_extent_and_a_frequency(monitor_type):
    kwargs = {"center": (0.0, 0.0, 0.0), "size": (0.0, 1.0, 1.0)}
    with pytest.raises(ValueError, match="at least one"):
        monitor_type(freqs=[], **kwargs)
    with pytest.raises(ValueError, match="exactly one zero"):
        monitor_type(freqs=[2e14], center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))

    if monitor_type is bz.ModeMonitor:
        with pytest.raises(TypeError, match="ModeSpec"):
            monitor_type(freqs=[2e14], mode_spec=_WrongModeSpec(), **kwargs)


@pytest.mark.parametrize(
    ("size", "axis", "components"),
    [
        ((0.0, 2.0, 2.0), "x", ("Ey", "Ez", "Hy", "Hz")),
        ((2.0, 0.0, 2.0), "y", ("Ex", "Ez", "Hx", "Hz")),
        ((2.0, 2.0, 0.0), "z", ("Ex", "Ey", "Hx", "Hy")),
    ],
)
def test_flux_monitor_orientation_controls_geometry_and_components(
    size, axis, components
):
    monitor = bz.FluxMonitor(
        center=(1.0, 2.0, 3.0), size=size, freqs=[2e14], name="flux"
    )

    assert monitor.plane_normal == axis
    assert monitor.dft_components == components
    assert monitor.position == monitor.center
    assert monitor.size_spec == monitor.size
    assert monitor.is_3d
    np.testing.assert_array_equal(monitor.get_dft_frequencies(), [2e14])
    assert monitor.shifted((1.0, -1.0, 2.0)).center == (2.0, 1.0, 5.0)
    with pytest.raises(ValueError, match="three values"):
        monitor.shifted((1.0,))


def test_monitor_unsnapped_geometry_fallbacks_cover_every_orientation():
    horizontal = bz.FluxMonitor(
        center=(2.0, 1.0, 0.0), size=(4.0, 0.0, 1.0), freqs=[2e14]
    )
    vertical = bz.FluxMonitor(
        center=(2.0, 1.0, 0.0), size=(0.0, 2.0, 1.0), freqs=[2e14]
    )
    assert horizontal.get_grid_points_2d(1.0, 1.0) == [
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
    ]
    assert vertical.get_grid_points_2d(1.0, 1.0) == [(2, 0), (2, 1), (2, 2)]

    expected_kinds = {
        "x": (slice, slice, int),
        "y": (slice, int, slice),
        "z": (int, slice, slice),
    }
    for size, axis in (
        ((0.0, 2.0, 2.0), "x"),
        ((2.0, 0.0, 2.0), "y"),
        ((2.0, 2.0, 0.0), "z"),
    ):
        monitor = bz.FluxMonitor(center=(1.0, 1.0, 1.0), size=size, freqs=[2e14])
        region = monitor.get_grid_slice_3d(1.0, 1.0, 1.0, None)
        assert tuple(type(item) for item in region) == expected_kinds[axis]


def test_monitor_2d_normalization_and_snapped_line_coordinates():
    compact = bz.FieldMonitor(
        center=(1.0, 2.0),
        size=(2.0, 0.0),
        freqs=[2e14],
    )
    assert compact.center == (1.0, 2.0, 0.0)
    assert compact.size == (2.0, 0.0, 0.0)

    horizontal = bz.FluxMonitor(
        center=(2.0, 1.0, 0.0),
        size=(4.0, 0.0, 1.0),
        freqs=[2e14],
    )
    vertical = bz.FluxMonitor(
        center=(2.0, 1.0, 0.0),
        size=(0.0, 2.0, 1.0),
        freqs=[2e14],
    )
    np.testing.assert_allclose(
        horizontal._line_sample_coords_2d(1.0, 1.0, (5, 5)),
        ([0.5, 1.5, 2.5, 3.5], [1.0] * 4),
    )
    np.testing.assert_allclose(
        vertical._line_sample_coords_2d(1.0, 1.0, (5, 5)),
        ([2.0] * 2, [0.5, 1.5]),
    )


def test_field_recorder_domain_and_slice_copy_translation_contracts():
    with pytest.raises(ValueError, match="Unsupported"):
        bz.FieldRecorder(components=())
    with pytest.raises(ValueError, match="Unsupported"):
        bz.FieldRecorder(components=("Ez", "bad"))
    with pytest.raises(ValueError, match="both center and size"):
        bz.FieldRecorder(center=(0.0, 0.0, 0.0))

    domain = bz.FieldRecorder()
    assert domain.dft_components is None
    assert domain.shifted((10.0, 20.0, 30.0)) is domain

    region = bz.FieldRecorder(
        components=("Ex", "Hy"),
        center=(1.0, 2.0, 3.0),
        size=(0.0, 2.0, 2.0),
    )
    assert region.region == "slice"
    assert region.shifted((1.0, 1.0, 1.0)).center == (2.0, 3.0, 4.0)
    assert region.updated_copy(name="copy").name == "copy"


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"center": (0.0, 0.0)}, "three values"),
        ({"center": (np.inf, 0.0, 0.0)}, "finite"),
        ({"size": (0.0, -1.0, 1.0)}, "non-negative"),
        ({"size": (0.0, 0.0, 1.0)}, "exactly one zero"),
        ({"direction": "x+"}, "direction must"),
        ({"power": -1.0}, "non-negative"),
        ({"source_time": object()}, "snapshot object-valued"),
        ({"source_time": _SampleOnly(freq0=0.0)}, "positive freq0"),
        ({"mode_spec": _WrongModeSpec()}, "must be a ModeSpec"),
    ],
)
def test_mode_source_rejects_invalid_configuration(changes, match):
    values = {
        "center": (0.0, 0.0, 0.0),
        "size": (0.0, 1.0, 1.0),
        "source_time": bz.GaussianPulse(2e14, 2e13),
        "direction": "+",
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError), match=match):
        bz.ModeSource(**values)


def test_mode_and_legacy_source_edge_contracts():
    source = bz.ModeSource(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 2.0, 1.0),
        source_time=bz.GaussianPulse(2e14, 2e13),
        direction="-",
        mode_spec=bz.ModeSpec(num_freqs=3),
    )
    assert source.axis == "x"
    assert source.signed_direction == "-x"
    assert source.frequency == 2e14
    assert source.transverse_size == (2.0, 1.0)
    assert source.profile_frequencies().shape == (3,)
    assert source.launch_power_normalization_spectrum([2e14]) is None
    with pytest.raises(ValueError, match="three values"):
        source.shifted((1.0,))

    sample_only = source.updated_copy(source_time=_SampleOnly())
    assert sample_only.source_spectrum([2e14]) is None
    spectrum_only = source.updated_copy(source_time=_SpectrumOnly())
    np.testing.assert_array_equal(
        spectrum_only.source_spectrum([1e14, 2e14], normalize=False),
        [3.0 + 2.0j, 3.0 + 2.0j],
    )

    with pytest.raises(ValueError, match="2D or 3D"):
        bz.GaussianSource(position=(0.0,), width=1.0, signal=[1.0])
    with pytest.raises(ValueError, match="positive finite"):
        bz.GaussianSource(position=(0.0, 0.0), width=0.0, signal=[1.0])
    legacy = bz.GaussianSource(position=(1.0, 2.0), width=0.5, signal=[1.0])
    assert legacy.shifted((3.0, 4.0)).position == (4.0, 6.0)


def test_mode_data_selects_nearest_frequency_and_owns_full_profile():
    frequencies = np.array([1e14, 2e14])
    neffs = np.array([[1.5], [1.6]], dtype=complex)
    fields = np.arange(2 * 1 * 2 * 3, dtype=float).reshape(2, 1, 2, 3)
    profiles = np.ones((2, 2, 3))
    full_profiles = np.full((2, 4, 5), 2.0)
    data = bz.ModeData(
        frequencies=frequencies,
        neffs=neffs,
        e_fields=fields,
        h_fields=-fields,
        eps_profiles=profiles,
        eps_profile_fulls=full_profiles,
        resolution=0.1,
        center=(1.0, 2.0, 3.0, 4.0),
    )

    f_idx, m_idx, neff, e_field, h_field, eps_full = data.selected_mode(
        f=1.9e14, mode_index=0
    )
    assert (f_idx, m_idx) == (1, 0)
    assert neff == 1.6
    np.testing.assert_array_equal(e_field, fields[1, 0])
    np.testing.assert_array_equal(h_field, -fields[1, 0])
    np.testing.assert_array_equal(eps_full, full_profiles[1])
    assert data.center == (1.0, 2.0, 3.0)
    assert not data.eps_profile_fulls.flags.writeable

    empty = bz.ModeData(
        frequencies=np.array([]),
        neffs=np.empty((0, 1)),
        e_fields=np.empty((0, 1, 1)),
        h_fields=np.empty((0, 1, 1)),
        eps_profiles=np.empty((0, 1)),
        resolution=0.1,
    )
    with pytest.raises(ValueError, match="no frequencies"):
        empty.selected_mode()


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"center": (0.0, 0.0)}, "3D coordinate"),
        ({"direction": "forward"}, "Unsupported"),
        ({"power": -1.0}, "non-negative"),
        ({"power": np.inf}, "finite"),
    ],
)
def test_gaussian_beam_rejects_invalid_configuration(changes, match):
    values = {
        "center": (0.0, 0.0, 0.0),
        "size": (1.0, 1.0),
        "source_time": bz.GaussianPulse(2e14, 2e13),
        "wavelength": 1.55,
    }
    values.update(changes)
    with pytest.raises(ValueError, match=match):
        bz.GaussianBeamSource(**values)


def test_gaussian_beam_spectrum_fallbacks_and_custom_source_translation():
    beam = bz.GaussianBeamSource(
        center=(0.0, 0.0, 0.0),
        size=(1.0, 1.0),
        source_time=_LegacySpectrumOnly(),
        wavelength=1.55,
    )
    assert beam.shifted((1.0, 2.0, 3.0)).center == (1.0, 2.0, 3.0)
    np.testing.assert_array_equal(
        beam.source_spectrum([2e14], normalize=False), [3.0 + 2.0j]
    )

    no_spectrum = beam.updated_copy(source_time=np.ones(2))
    assert no_spectrum.source_spectrum([2e14]) is None
    assert beam.updated_copy(source_time=None).source_spectrum([2e14]) is None

    custom = bz.CustomSource(
        component="Ez",
        timing="e",
        index=(slice(None), slice(None)),
        coeff=np.ones((2, 2)),
        waveform=np.ones(2),
        target_shape=(2, 2),
    )
    assert custom.shifted((100.0, 100.0)) is custom


def test_design_constructor_and_functional_copy_validate_every_public_field():
    with pytest.raises(ValueError, match="only one"):
        bz.Design(material=bz.Material(), background=bz.Material())
    with pytest.raises(TypeError, match="MaterialProtocol"):
        bz.Design(material=object())
    for field, value, match in (
        ("width", 0.0, "width"),
        ("height", np.inf, "height"),
        ("depth", -1.0, "depth"),
    ):
        with pytest.raises(ValueError, match=match):
            bz.Design(**{field: value})

    design = bz.Design(width=2.0, height=3.0)
    assert not design.is_3d
    assert "2D" in str(design)
    assert design.copy() is design
    assert design.__copy__() is design
    assert design.__deepcopy__({}) is design
    assert design.__eq__(object()) is NotImplemented
    with pytest.raises(TypeError, match="Unknown Design field"):
        design.updated_copy(unknown=True)
    with pytest.raises(TypeError, match="immutable geometry"):
        design.with_structure(object())

    box = bz.Box(size=(1.0, 1.0, 1.0))
    expanded = design.with_structure(box)
    assert expanded.structures == (box,)
    assert design.structures == ()


def test_custom_material_grid_callable_identity_and_validation_contracts():
    for bounds, match in (
        (((0.0, 1.0),), "bounds must"),
        (((1.0, 0.0), (0.0, 1.0)), "Invalid x"),
        (((0.0, 1.0), (1.0, 0.0)), "Invalid y"),
    ):
        with pytest.raises(ValueError, match=match):
            bz.CustomMaterial(bounds=bounds)

    with pytest.raises(TypeError, match="callable"):
        bz.CustomMaterial(permittivity_func=3.0, cache_key="bad")
    with pytest.raises(TypeError, match="cache_key"):
        bz.CustomMaterial(permittivity_func=lambda x, y: x + y)
    with pytest.raises(ValueError, match="requires bounds"):
        bz.CustomMaterial(permittivity_grid=np.ones((2, 2)))
    with pytest.raises(ValueError, match="positive and finite"):
        bz.CustomMaterial(max_permittivity=0.0)
    with pytest.raises(ValueError, match="must include"):
        bz.CustomMaterial(
            permittivity_grid=np.full((2, 2), 4.0),
            bounds=((0.0, 1.0), (0.0, 1.0)),
            max_permittivity=3.0,
        )

    grids = bz.CustomMaterial(
        permittivity_grid=np.array([[2.0, 4.0], [6.0, 8.0]]),
        permeability_grid=np.full((2, 2), 3.0),
        conductivity_grid=np.full((2, 2), 0.5),
        bounds=((0.0, 1.0), (0.0, 1.0)),
    )
    assert grids.permittivity == "grid(2.000-8.000)"
    assert grids.permeability == "grid(3.000-3.000)"
    assert grids.conductivity == "grid(0.500-0.500)"
    eps, mu, sigma = grids.get_sample([0.5], [0.5])
    np.testing.assert_allclose(eps, [5.0])
    np.testing.assert_allclose(mu, [3.0])
    np.testing.assert_allclose(sigma, [0.5])
    assert grids.max_permittivity == 8.0
    with pytest.raises(ValueError, match="Unknown property"):
        grids.update_grid("refractive_index", np.ones((2, 2)))
    assert (
        grids.update_grid("conductivity", np.zeros((2, 2))).conductivity_grid.max() == 0
    )

    functional = bz.CustomMaterial(
        permittivity_func=lambda x, y, z=None: x + y + (0.0 if z is None else z),
        permeability_func=lambda x, y, z=None: 2.0,
        conductivity_func=lambda x, y, z=None: 0.25,
        cache_key=("functional", 1),
        max_permittivity=10.0,
    )
    assert functional.permittivity == "function"
    assert functional.permeability == "function"
    assert functional.conductivity == "function"
    assert functional.get_sample(1.0, 2.0) == (3.0, 2.0, 0.25)
    assert functional.get_sample(1.0, 2.0, 3.0) == (6.0, 2.0, 0.25)
    assert functional.__eq__(object()) is NotImplemented
    with pytest.raises(TypeError, match="explicit.*cache_key"):
        functional.updated_copy(permittivity_func=lambda x, y: 1.0)
    changed = functional.updated_copy(
        permittivity_func=lambda x, y: 1.0,
        cache_key=("functional", 2),
    )
    assert changed != functional
