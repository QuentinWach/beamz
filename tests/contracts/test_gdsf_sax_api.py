import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    Design,
    FieldMonitor,
    Material,
    ModeMonitor,
    ModeSpec,
    Port,
    Rectangle,
    calc_optimal_fdtd_params,
    dxdt,
)
from beamz.analysis import mode_projection as mp
from beamz.analysis import sparameters as sp
from beamz.analysis.data import AnalysisData
from beamz.analysis.modal_projection.colocation import (
    _colocate_field_components_to_projection_3d,
    _colocate_monitor_component_matrix_3d,
    _interpolate_plane_matrix_2d,
)
from beamz.analysis.modal_projection.geometry import (
    _modal_projection_spatial_phase,
    _monitor_component_plane_coords_3d,
    _monitor_projection_phase,
)
from beamz.design.gds import (
    ImportedComponent,
    export_gds,
    import_component,
    import_gds,
)
from beamz.devices.sources.mode_profiles import (
    _MODE_PLANE_APERTURE_PAD_CELLS,
    _MODE_PLANE_APERTURE_WINDOW_ALPHA,
)
from beamz.lattice import component_shape_3d
from beamz.simulation.results import FieldMetadata, MaterialRegion, SimulationMetadata

GDS_WARNING_FILTERS = (
    "ignore:Support for class-based `config` is deprecated.*:DeprecationWarning",
    "ignore:Implicitly cleaning up <TemporaryDirectory.*:ResourceWarning",
    "ignore:unclosed file .*gdsfactory.*:ResourceWarning",
)


def _analysis_contract(*, monitor=None, frequencies=(), shape=(1, 1), dt=0.1):
    is_3d = len(shape) == 3
    component_shapes = (
        {
            component: component_shape_3d(component, shape)
            for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        }
        if is_3d
        else {}
    )
    materials = (
        MaterialRegion(np.ones(shape), np.ones(shape), (0,) * len(shape), shape)
        if is_3d
        else None
    )
    metadata = SimulationMetadata(
        dt=dt,
        resolution=1.0,
        is_3d=is_3d,
        plane_2d="xy",
        coordinate_offset=(0.0, 0.0, 0.0),
        time=np.array([0.0, dt]),
        width=float(shape[-1]),
        height=float(shape[-2]),
        depth=float(shape[0]) if is_3d else 0.0,
        fields=FieldMetadata(shape, component_shapes, materials),
    )
    return AnalysisData(metadata, {}, materials, frequencies, monitor)


def _port(
    *,
    name,
    monitor_name=None,
    direction="+x",
    polarization="tm",
    mode_index=0,
):
    axis = str(direction)[-1]
    size = tuple(0.0 if value == axis else 1.0 for value in "xyz")
    return Port(
        center=(0.0, 0.0, 0.0),
        size=size,
        name=name,
        monitor_name=monitor_name,
        direction=str(direction)[0],
        mode_spec=ModeSpec(mode_index=mode_index, polarization=polarization),
    )


def test_dxdt_alias_matches_calc_optimal_fdtd_params():
    wavelength = 1.55e-6
    kwargs = dict(n_max=2.0, dims=2, safety_factor=0.95, points_per_wavelength=8)
    dx_alias, dt_alias = dxdt(wavelength, **kwargs)
    dx_ref, dt_ref = calc_optimal_fdtd_params(wavelength, **kwargs)
    assert dx_alias == dx_ref
    assert dt_alias == dt_ref


@pytest.mark.skipif(
    importlib.util.find_spec("gdsfactory") is None,
    reason="gdsfactory not installed",
)
@pytest.mark.filterwarnings(*GDS_WARNING_FILTERS)
def test_component_import_returns_materialized_design_and_canonical_ports():
    imported = import_component(
        "mmi1x2", layer=(1, 0), n_core=2.0, n_clad=1.44, xy_padding=2e-6
    )
    loaded_design = imported.design
    ports = {port.name: port for port in imported.ports}

    assert isinstance(imported, ImportedComponent)
    assert imported.port("o1").name == "o1"
    assert loaded_design.width > 0
    assert loaded_design.height > 0
    assert loaded_design.background.permittivity == pytest.approx(1.44**2)
    assert loaded_design.structures

    core_structures = loaded_design.structures
    assert core_structures
    assert all(
        getattr(struct, "material", None) is not None for struct in core_structures
    )

    assert {"o1", "o2", "o3"}.issubset(ports.keys())
    for port_name in ("o1", "o2", "o3"):
        port = ports[port_name]
        assert len(port.center) == 3
        assert sum(np.isclose(port.size, 0.0)) == 1
        assert port.signed_direction in {"+x", "-x", "+y", "-y"}
        cx, cy, _ = port.center
        assert 0 <= cx <= loaded_design.width
        assert 0 <= cy <= loaded_design.height


@pytest.mark.skipif(
    importlib.util.find_spec("gdsfactory") is None,
    reason="gdsfactory not installed",
)
@pytest.mark.filterwarnings(*GDS_WARNING_FILTERS)
def test_gds_export_round_trip_uses_the_same_component_converter(tmp_path):
    source = Design(width=3e-6, height=2e-6)
    source += Rectangle(
        position=(0.5e-6, 0.25e-6),
        width=2e-6,
        height=1.5e-6,
        material=Material(4.0),
    )

    path = export_gds(source, tmp_path / "design.gds")
    imported = import_gds(path, layer=(0, 0), n_core=2.0)

    assert path.is_file()
    assert imported.design.width == pytest.approx(2e-6)
    assert imported.design.height == pytest.approx(1.5e-6)
    assert imported.design.structures
    assert imported.ports == ()


def test_resample_complex_matrix_flattens_trailing_spatial_dims():
    freq_src = np.array([1.0, 2.0], dtype=float)
    freq_dst = np.array([1.0, 2.0], dtype=float)
    src = (np.arange(12, dtype=float) + 1j * np.arange(12, dtype=float)).reshape(
        2, 2, 3
    )

    out = sp._resample_complex_matrix(freq_src, src, freq_dst)
    assert out.shape == (2, 6)
    np.testing.assert_allclose(out, src.reshape(2, 6), rtol=1e-12, atol=1e-12)


def test_monitor_projection_phase_uses_yee_half_step_h_lag():
    dt = 2.0e-15
    freqs = np.array([1.0e12, 4.0e12], dtype=float)

    np.testing.assert_allclose(
        _monitor_projection_phase("Ez", freqs, dt),
        np.ones_like(freqs, dtype=np.complex128),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _monitor_projection_phase("Hy", freqs, dt),
        np.exp(-1j * np.pi * freqs * dt),
        rtol=1e-12,
        atol=1e-12,
    )


def test_modal_projection_spatial_phase_advances_e_to_h_reference_plane():
    freqs = np.array([1.0e12, 4.0e12], dtype=float)
    plane_delay_s = 0.25e-15

    np.testing.assert_allclose(
        _modal_projection_spatial_phase("Ez", freqs, plane_delay_s),
        np.exp(1j * 2.0 * np.pi * freqs * plane_delay_s),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _modal_projection_spatial_phase("Hy", freqs, plane_delay_s),
        np.ones_like(freqs, dtype=np.complex128),
        rtol=1e-12,
        atol=1e-12,
    )


def test_modal_projection_plane_delay_matches_2d_yee_half_cell_delay():
    sim = SimpleNamespace()
    sim.is_3d = False
    sim.resolution = 80e-9
    spec = _port(name="o1", monitor_name="m", direction="+x", polarization="tm")
    delay = mp._modal_projection_plane_delay_s(sim, spec, 193.4e12, 1.8)
    assert delay == pytest.approx(0.5 * sim.resolution * 1.8 / LIGHT_SPEED)

    spec_back = _port(name="o2", monitor_name="m", direction="-x", polarization="tm")
    delay_back = mp._modal_projection_plane_delay_s(sim, spec_back, 193.4e12, 1.8)
    assert delay_back == pytest.approx(delay, rel=1e-12, abs=0.0)


def test_modal_projection_plane_delay_is_zero_for_3d_colocated_monitors():
    sim = SimpleNamespace()
    sim.is_3d = True
    sim.resolution = 80e-9
    sim.dt = 0.1e-15
    spec = _port(name="o1", monitor_name="m", direction="+y", polarization="te")

    delay = mp._modal_projection_plane_delay_s(sim, spec, 193.4e12, 2.4)

    assert delay == pytest.approx(0.0)


def test_interpolate_plane_matrix_2d_preserves_complex_affine_plane():
    src0 = np.array([0.0, 1.0, 2.0], dtype=float)
    src1 = np.array([0.0, 2.0, 4.0], dtype=float)
    dst0 = np.array([0.25, 1.5], dtype=float)
    dst1 = np.array([0.5, 3.0], dtype=float)

    def plane(c0, c1):
        return 2.0 + 3.0 * c0 - 0.5 * c1 + 1j * (-1.0 + 0.25 * c0 + 2.0 * c1)

    values = plane(src0[:, None], src1[None, :])
    expected = plane(dst0[:, None], dst1[None, :])

    actual = _interpolate_plane_matrix_2d(values, src0, src1, dst0, dst1)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_colocate_field_components_to_projection_3d_respects_yee_offsets():
    sim = SimpleNamespace()
    sim.is_3d = True
    sim.resolution = 1.0
    grid_shape = (5, 5, 5)
    field_arrays = {
        comp: np.zeros(component_shape_3d(comp, grid_shape), dtype=np.float32)
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    field_arrays["permittivity"] = np.ones(grid_shape, dtype=np.float32)
    sim.fields = type("F", (), field_arrays)()

    class DummyXMonitor:
        name = "yee_x"

        @staticmethod
        def get_grid_slice_3d(dx, dy, dz, field_shape):
            del dx, dy, dz
            normal_index = min(1, int(field_shape[2]) - 1)
            return slice(1, 4), slice(1, 4), normal_index

    def plane(c0, c1):
        return 0.25 + 0.75 * c0 - 0.4 * c1 + 1j * (1.5 - 0.2 * c0 + 0.6 * c1)

    monitor = DummyXMonitor()
    target0 = np.array([1.5, 2.0, 2.5], dtype=float)
    target1 = np.array([1.5, 2.0, 2.5], dtype=float)
    projection = {
        "axis": "x",
        "analysis_coords0": target0,
        "analysis_coords1": target1,
    }
    field_components = {}
    for comp in ("Ey", "Ez", "Hy", "Hz"):
        src0, src1 = _monitor_component_plane_coords_3d(sim, monitor, comp, "x")
        field_components[comp] = plane(src0[:, None], src1[None, :]).reshape(-1)

    colocated = _colocate_field_components_to_projection_3d(
        sim,
        monitor,
        field_components,
        projection,
    )

    expected = plane(target0[:, None], target1[None, :]).reshape(-1)
    for comp in ("Ey", "Ez", "Hy", "Hz"):
        np.testing.assert_allclose(colocated[comp], expected, rtol=1e-13, atol=1e-13)


def test_build_port_projection_3d_modemonitor_uses_discrete_contract(monkeypatch):
    grid_shape = (4, 4, 4)

    def fail_solve_modes(**kwargs):
        del kwargs
        raise AssertionError("legacy solve_modes should not be used")

    captured = {}

    def fake_solve_discrete_mode_plane(**kwargs):
        captured.update(kwargs)
        profile_shape = (2, 2)
        zeros = np.zeros(profile_shape, dtype=np.complex128)
        forward = {
            "Ex": zeros.copy(),
            "Ey": np.ones(profile_shape, dtype=np.complex128),
            "Ez": zeros.copy(),
            "Hx": zeros.copy(),
            "Hy": zeros.copy(),
            "Hz": np.ones(profile_shape, dtype=np.complex128),
        }
        backward = {
            name: (-value if name.startswith("H") else value.copy())
            for name, value in forward.items()
        }
        indices = {
            "Ex": (slice(1, 3), slice(1, 3), 1),
            "Ey": (slice(1, 3), slice(1, 3), 1),
            "Ez": (slice(1, 3), slice(1, 3), 1),
            "Hx": (slice(1, 3), slice(1, 3), 1),
            "Hy": (slice(1, 3), slice(1, 3), 1),
            "Hz": (slice(1, 3), slice(1, 3), 1),
        }
        return SimpleNamespace(
            neff=2.25,
            profiles=forward,
            backward_profiles=backward,
            component_indices=indices,
            phase_reference_coord=0.25,
            phase_plane_coord=0.5,
        )

    monkeypatch.setattr(mp, "solve_modes", fail_solve_modes)
    monkeypatch.setattr(
        mp,
        "solve_discrete_mode_plane",
        fake_solve_discrete_mode_plane,
    )

    monitor = ModeMonitor(
        center=(1.5, 1.5, 1.5),
        size=(0.0, 2.0, 2.0),
        freqs=(1.0,),
        name="m_discrete_contract",
        mode_spec=ModeSpec(num_modes=4, target_neff=2.25),
    )
    sim = _analysis_contract(monitor=monitor, frequencies=(1.0,), shape=grid_shape)

    projection = mp._build_port_projection(
        sim,
        spec=_port(
            name="p",
            monitor_name="m_discrete_contract",
            direction="+x",
            polarization="te",
        ),
        monitor=monitor,
        frequency=1.0,
        cache={},
    )

    assert captured["direction"] == "+x"
    assert captured["solver_direction"] == "+x"
    assert captured["transverse_axes"] == ("z", "y")
    assert captured["plane_index"] == 1
    assert captured["offset_index"] == 0
    assert captured["num_modes"] == 4
    assert captured["target_neff"] == 2.25
    assert captured["aperture_pad_cells"] == _MODE_PLANE_APERTURE_PAD_CELLS
    assert captured["aperture_window_alpha"] == _MODE_PLANE_APERTURE_WINDOW_ALPHA
    assert captured["scalar_permittivity"].shape == (4, 4)
    assert captured["grid_shape"] == (4, 4, 3)
    assert captured["component_shapes"]["Ey"] == (5, 4, 4)
    assert "component_permittivity" in captured
    assert "component_permeability" in captured
    assert projection["discrete_contract"] == "micromode.beamz.DiscreteMode/v1"
    assert projection["components"] == ("Ey", "Ez", "Hy", "Hz")
    assert np.real(projection["mode_components"]["Hz"][0]) < 0.0
    assert np.real(projection["mode_components_bwd"]["Hz"][0]) > 0.0


def test_project_modal_coefficients_3d_group_recovers_multiple_modes():
    components = ("Ey", "Ez", "Hy", "Hz")

    def _mode(ey, hz):
        return {
            "Ey": np.asarray(ey, dtype=np.complex128),
            "Ez": np.zeros((2,), dtype=np.complex128),
            "Hy": np.zeros((2,), dtype=np.complex128),
            "Hz": np.asarray(hz, dtype=np.complex128),
        }

    def _backward(mode):
        return {
            name: (-value if name.startswith("H") else value.copy())
            for name, value in mode.items()
        }

    mode0 = _mode([1.0, 0.0], [1.0, 0.0])
    mode1 = _mode([0.0, 1.0], [0.0, 1.0])
    mode0_bwd = _backward(mode0)
    mode1_bwd = _backward(mode1)
    projections = [
        {
            "components": components,
            "axis": "x",
            "d_area": 1.0,
            "direction_sign": 1.0,
            "mode_components": mode0,
            "mode_components_bwd": mode0_bwd,
        },
        {
            "components": components,
            "axis": "x",
            "d_area": 1.0,
            "direction_sign": 1.0,
            "mode_components": mode1,
            "mode_components_bwd": mode1_bwd,
        },
    ]
    coeff_true = [
        (0.7 - 0.2j, -0.1 + 0.05j),
        (0.25 + 0.4j, 0.15 - 0.3j),
    ]
    field = {}
    for name in components:
        field[name] = (
            coeff_true[0][0] * mode0[name]
            + coeff_true[0][1] * mode0_bwd[name]
            + coeff_true[1][0] * mode1[name]
            + coeff_true[1][1] * mode1_bwd[name]
        )

    coeff, residual, cond, diagnostics = mp._project_modal_coefficients_3d_group(
        field,
        projections,
    )

    assert cond < 10.0
    assert residual < 1e-12
    assert diagnostics["residual_e"] < 1e-12
    assert diagnostics["residual_h"] < 1e-12
    for actual, expected in zip(coeff, coeff_true, strict=True):
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-10, atol=1e-10)


def test_build_port_projection_2d_preserves_requested_mode_index_without_seed(
    monkeypatch,
):
    sim = SimpleNamespace()
    sim.is_3d = False
    sim.plane_2d = "xy"
    sim.resolution = 1.0
    sim.fields = type(
        "F",
        (),
        {"permittivity": np.ones((5, 5), dtype=float)},
    )()

    def fake_profile(self, monitor, axis, pad_cells):
        del self, monitor, axis, pad_cells
        return np.ones((5,), dtype=np.complex128), np.arange(5, dtype=int), 1.0

    monkeypatch.setattr(mp, "_monitor_profile_slice", fake_profile)

    e0 = np.zeros((3, 5), dtype=np.complex128)
    e1 = np.zeros((3, 5), dtype=np.complex128)
    h0 = np.zeros((3, 5), dtype=np.complex128)
    h1 = np.zeros((3, 5), dtype=np.complex128)
    e0[0, 0] = 1.0
    e1[0, 1] = 3.0
    h0[2, 0] = 1.0
    h1[2, 1] = 3.0

    def fake_solve_modes(
        eps,
        omega,
        dL,
        m,
        direction,
        filter_pol,
        target_neff,
        return_fields,
    ):
        del eps, omega, dL, m, direction, filter_pol, target_neff, return_fields
        return (
            np.array([2.3, 1.8], dtype=float),
            [e0, e1],
            [h0, h1],
            None,
        )

    monkeypatch.setattr(mp, "solve_modes", fake_solve_modes)

    monitor = FieldMonitor(
        center=(0.0, 0.5, 0.0),
        size=(0.0, 1.0, 0.0),
        freqs=[1.0],
        name="m2d_mode1",
    )
    projection = mp._build_port_projection(
        sim,
        spec=_port(
            name="p2d_mode1",
            monitor_name="m2d_mode1",
            direction="+y",
            polarization="te",
            mode_index=1,
        ),
        monitor=monitor,
        frequency=1.0,
        cache={},
    )

    assert projection["mode_neff"] == pytest.approx(1.8)
    assert int(np.argmax(np.abs(projection["mode_matrix"][:5, 0]))) == 1
    assert np.count_nonzero(np.abs(projection["mode_matrix"][:5, 0]) > 1e-12) == 1


def test_extract_port_waves_dft_modal_coefficients_synthetic(monkeypatch):
    freqs = np.array([1.0, 2.0], dtype=float)
    mode_matrix = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    pinv = np.linalg.pinv(mode_matrix)

    a_src = np.array([1.2 + 0.2j, 1.1 - 0.1j])
    b_src = np.array([0.08 - 0.02j, 0.03 + 0.01j])
    a_out = np.array([0.05 + 0.01j, -0.02 + 0.03j])
    b_out = np.array([0.72 - 0.1j, 0.69 + 0.08j])

    dft_map = {
        ("m_src", "Ez"): (a_src + b_src)[:, None],
        ("m_src", "Hy"): (a_src - b_src)[:, None],
        ("m_out", "Ez"): (a_out + b_out)[:, None],
        ("m_out", "Hy"): (a_out - b_out)[:, None],
    }

    def fake_sample(self, monitor, component, frequencies):
        assert np.allclose(frequencies, freqs)
        return np.asarray(frequencies, dtype=float), dft_map[(monitor.name, component)]

    projection_frequencies = []

    def fake_projection(
        self,
        spec,
        monitor,
        frequency,
        cache,
        mode_pad_cells=6,
    ):
        projection_frequencies.append(float(frequency))
        return {
            "e_component": "Ez",
            "h_component": "Hy",
            "mode_matrix": mode_matrix,
            "pinv": pinv,
            "condition_number": 1.0,
            "mode_neff": 2.0,
        }

    monkeypatch.setattr(sp, "_sample_monitor_component_dft", fake_sample)
    monkeypatch.setattr(mp, "_build_port_projection", fake_projection)

    monitors = [
        FieldMonitor(
            center=(0.0, 0.5, 0.0),
            size=(0.0, 1.0, 0.0),
            name="m_src",
            freqs=freqs,
        ),
        FieldMonitor(
            center=(1.0, 0.5, 0.0),
            size=(0.0, 1.0, 0.0),
            name="m_out",
            freqs=freqs,
        ),
    ]
    inputs = {
        monitor.name: _analysis_contract(
            monitor=monitor, frequencies=freqs, shape=(2, 2)
        )
        for monitor in monitors
    }

    ports = [
        _port(
            name="o1",
            monitor_name="m_src",
            direction="+x",
            polarization="tm",
        ),
        _port(name="o2", monitor_name="m_out", direction="+x", polarization="tm"),
    ]
    waves = sp._extract_port_waves_dft(inputs, ports=ports, frequencies=freqs)
    np.testing.assert_allclose(waves["o1"]["a_plus"], a_src, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(waves["o1"]["a_minus"], b_src, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(waves["o2"]["a_plus"], a_out, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(waves["o2"]["a_minus"], b_out, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        waves["o1"]["projection_residual"],
        np.zeros_like(freqs),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        waves["o2"]["projection_residual"],
        np.zeros_like(freqs),
        rtol=1e-12,
        atol=1e-12,
    )

    projection_frequencies.clear()
    sp._extract_port_waves_dft(
        inputs,
        ports=ports,
        frequencies=freqs,
        mode_strategy="single",
    )
    assert projection_frequencies
    np.testing.assert_allclose(
        projection_frequencies,
        np.full(len(projection_frequencies), np.median(freqs)),
    )


def test_s_parameters_keys_shapes_and_valid_mask(monkeypatch):
    inputs = {"analysis": _analysis_contract()}
    freqs = np.array([1.0, 2.0, 3.0], dtype=float)
    waves = {
        "o1": {
            "a_plus": np.array([1.0, 1e-6, 1.0], dtype=np.complex128),
            "a_minus": 0.1 * np.ones_like(freqs, dtype=np.complex128),
            "condition_number": np.array([2.0, 2.0, 2.0], dtype=float),
        },
        "o2": {
            "a_plus": np.zeros_like(freqs, dtype=np.complex128),
            "a_minus": 0.7j * np.ones_like(freqs, dtype=np.complex128),
            "condition_number": np.array([3.0, 3.0, 3.0], dtype=float),
        },
    }

    def fake_extract(
        sim_arg,
        ports,
        frequencies,
        min_incident_db=-40.0,
        return_power=True,
        mode_strategy="per_frequency",
    ):
        del sim_arg, ports, min_incident_db, return_power, mode_strategy
        assert np.allclose(frequencies, freqs)
        return waves

    monkeypatch.setattr(sp, "_extract_port_waves_dft", fake_extract)
    result = sp.s_parameters(
        inputs,
        source_port="o1",
        ports=[
            _port(
                name="o1",
                monitor_name="o1",
                direction="+x",
                polarization="tm",
            ),
            _port(name="o2", monitor_name="o2", direction="+x", polarization="tm"),
        ],
        output_ports=["o1", "o2"],
        frequencies=freqs,
        min_incident_db=-40.0,
    )

    s_matrix = result.s_matrix
    diag = result.diagnostics
    assert set(s_matrix.keys()) == {("o1", "o1"), ("o2", "o1")}
    assert s_matrix[("o1", "o1")].shape == freqs.shape
    assert s_matrix[("o2", "o1")].shape == freqs.shape
    assert diag["valid_mask"].shape == freqs.shape
    assert np.array_equal(diag["valid_mask"], np.array([True, False, True]))
    assert s_matrix[("o2", "o1")][1] == 0.0j
    assert np.isnan(diag["power_sum"][1])


def test_colocate_monitor_component_matrix_3d_interpolates_to_canonical_plane():
    sim = SimpleNamespace()
    sim.is_3d = True
    sim.resolution = 1.0
    sim.fields = type(
        "F",
        (),
        {
            "permittivity": np.ones((5, 6, 7), dtype=float),
            "Ex": np.zeros((5, 6, 6), dtype=float),
            "Ey": np.zeros((5, 5, 7), dtype=float),
            "Ez": np.zeros((4, 6, 7), dtype=float),
            "Hx": np.zeros((4, 5, 7), dtype=float),
            "Hy": np.zeros((4, 6, 6), dtype=float),
            "Hz": np.zeros((5, 5, 6), dtype=float),
        },
    )()

    monitor = FieldMonitor(
        center=(3.5, 2.8, 2.75),
        size=(4.6, 0.0, 3.3),
        name="mcanon",
        freqs=np.array([1.0], dtype=float),
        fields=("Ex",),
    )

    src0, src1 = _monitor_component_plane_coords_3d(sim, monitor, "Ex", axis="y")
    dst0, dst1 = mp._monitor_analysis_plane_3d(sim, monitor, axis="y")
    expected0 = 1.1 + (np.arange(dst0.size, dtype=float) + 0.5) * (
        (4.4 - 1.1) / float(dst0.size)
    )
    expected1 = 1.2 + (np.arange(dst1.size, dtype=float) + 0.5) * (
        (5.8 - 1.2) / float(dst1.size)
    )
    np.testing.assert_allclose(dst0, expected0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(dst1, expected1, atol=1e-12, rtol=0.0)
    src_plane = src0[:, None] + src1[None, :]
    colocated = _colocate_monitor_component_matrix_3d(
        sim,
        monitor,
        "Ex",
        src_plane.reshape(1, -1),
        axis="y",
        target0=dst0,
        target1=dst1,
    )
    dst0_eff = np.clip(dst0, np.min(src0), np.max(src0))
    dst1_eff = np.clip(dst1, np.min(src1), np.max(src1))
    expected = (dst0_eff[:, None] + dst1_eff[None, :]).reshape(1, -1)
    np.testing.assert_allclose(colocated, expected, rtol=1e-9, atol=1e-9)
