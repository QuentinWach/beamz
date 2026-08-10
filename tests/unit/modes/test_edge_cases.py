"""Focused edge contracts for the native mode-solver internals."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import linalg as scipy_linalg
from scipy import sparse
from scipy.sparse import linalg as spla

from beamz.design.grid import RectilinearGrid
from beamz.devices.modes import _scipy, _yee
from beamz.devices.modes import discrete as discrete_module
from beamz.devices.modes import plane as plane_module
from beamz.devices.modes import solver as solver_module
from beamz.devices.modes.discrete import DiscreteMode, ModePlaneSpec
from beamz.devices.modes.fields import (
    _axis_coordinate,
    _axis_index,
    _modal_overlap,
    _numeric_wave_number,
)
from beamz.devices.modes.models import BoundarySpec, Grid, Materials, PmlSpec
from beamz.devices.sources.mode_launch import (
    _power_scale,
    _scale_pair_for_power,
    _scale_profiles_for_power,
    _to_real_profile,
)
from beamz.devices.sources.planar_tfsf import _shift_component_indices_along_axis


def _mode_plane_spec(**changes) -> ModePlaneSpec:
    values = {
        "scalar_permittivity": np.ones((3, 3)),
        "frequency": 2.0e14,
        "resolution": 1.0e-7,
        "dt": None,
        "axis": "x",
        "direction": "+x",
        "transverse_axes": ("z", "y"),
        "grid_shape": (5, 5, 5),
        "center": (2.5e-7, 2.5e-7, 2.5e-7),
        "width": 3.0e-7,
        "height": 3.0e-7,
        "plane_index": 2,
        "offset_index": 1,
        "polarization": "te",
    }
    values.update(changes)
    return ModePlaneSpec(**values)


def _component_profiles(value=1.0) -> dict[str, np.ndarray]:
    shapes = {
        "Ex": (3, 4),
        "Ey": (3, 3),
        "Ez": (2, 4),
        "Hx": (2, 3),
        "Hy": (2, 4),
        "Hz": (3, 3),
    }
    return {
        name: np.full(shape, value, dtype=np.complex128)
        for name, shape in shapes.items()
    }


def _component_materials(profiles):
    electric = {
        name: np.ones_like(profiles[name], dtype=np.complex128)
        for name in ("Ex", "Ey", "Ez")
    }
    magnetic = {
        name: np.ones_like(profiles[name], dtype=np.complex128)
        for name in ("Hx", "Hy", "Hz")
    }
    indices = {name: (slice(None), slice(None)) for name in profiles}
    return electric, magnetic, indices


def test_discrete_mode_solver_receives_exact_rectilinear_transverse_edges(monkeypatch):
    grid = RectilinearGrid(
        [0.0, 0.1e-6, 0.3e-6, 0.6e-6, 1.0e-6, 1.5e-6],
        [0.0, 0.2e-6, 0.5e-6, 0.9e-6],
        [0.0, 0.15e-6, 0.4e-6, 0.8e-6],
    )
    captured = {}
    fields = {
        name: np.ones((3, 3), dtype=np.complex128)
        for name in discrete_module._COMPONENTS
    }
    profiles = _component_profiles()
    indices = {
        name: (slice(0, values.shape[0]), slice(0, values.shape[1]), 1)
        for name, values in profiles.items()
    }

    def fake_solve_grid(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(solver_info={})

    monkeypatch.setattr(discrete_module, "solve_grid", fake_solve_grid)
    monkeypatch.setattr(
        discrete_module,
        "_candidate_modes",
        lambda *_args: [{"neff": 2.0 + 0.0j, "fields": fields}],
    )
    monkeypatch.setattr(
        discrete_module,
        "_build_profiles",
        lambda *_args: (profiles, indices, {"initial_power": 1.0}),
    )

    discrete_module.solve_beamz_mode(
        _mode_plane_spec(
            scalar_permittivity=np.ones((3, 3)),
            grid_shape=grid.shape_zyx,
            grid=grid,
            yee_refinement=False,
        )
    )

    np.testing.assert_allclose(captured["x_edges"], grid.y_edges / 1e-6)
    np.testing.assert_allclose(captured["y_edges"], grid.z_edges / 1e-6)


def test_rectilinear_mode_uses_yee_refinement_when_normal_axis_is_uniform(
    monkeypatch,
):
    grid = RectilinearGrid(
        np.linspace(0.0, 0.5e-6, 6),
        [0.0, 0.1e-6, 0.3e-6, 0.6e-6],
        [0.0, 0.15e-6, 0.4e-6],
    )
    fields = {
        name: np.ones((3, 3), dtype=np.complex128)
        for name in discrete_module._COMPONENTS
    }
    profiles = _component_profiles()
    indices = {
        name: (slice(0, values.shape[0]), slice(0, values.shape[1]), 1)
        for name, values in profiles.items()
    }
    captured = {}

    monkeypatch.setattr(
        discrete_module,
        "solve_grid",
        lambda **_kwargs: SimpleNamespace(solver_info={}),
    )
    monkeypatch.setattr(
        discrete_module,
        "_candidate_modes",
        lambda *_args: [{"neff": 2.0 + 0.0j, "fields": fields}],
    )
    monkeypatch.setattr(
        discrete_module,
        "_build_profiles",
        lambda *_args: (profiles, indices, {"initial_power": 1.0}),
    )

    def fake_refinement(*_args, **kwargs):
        captured.update(kwargs)
        return profiles, 0.0, 1.0, 2.0, 1.0

    monkeypatch.setattr(discrete_module, "refine_x_mode_at_fixed_beta", fake_refinement)
    monkeypatch.setattr(
        discrete_module,
        "validate_x_mode_refinement",
        lambda *_args, **_kwargs: (True, {"rejection_reason": ""}),
    )

    result = discrete_module.solve_beamz_mode(
        _mode_plane_spec(
            scalar_permittivity=np.ones((2, 3)),
            grid_shape=grid.shape_zyx,
            grid=grid,
            component_permittivity={"Ex": np.ones(1)},
            component_permeability={"Hx": np.ones(1)},
        )
    )

    assert result.diagnostics["yee_refinement_eligible"]
    assert result.diagnostics["yee_refinement_accepted"]
    assert captured["normal_spacing"] == pytest.approx(0.1e-6)
    z_coordinates, y_coordinates = captured["transverse_coordinates"]
    np.testing.assert_allclose(z_coordinates, grid.z_edges)
    np.testing.assert_allclose(y_coordinates, grid.y_edges)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"num_cells": (1,)}, "two"),
        ({"sigma_max": 0.0}, "positive"),
        ({"kappa_min": np.nan}, "positive"),
        ({"kappa_min": 2.0, "kappa_max": 1.0}, "greater"),
        ({"order": True}, "integers"),
        ({"order": 1.5}, "integers"),
    ],
)
def test_pml_spec_rejects_invalid_values(changes, match):
    with pytest.raises(ValueError, match=match):
        PmlSpec(**changes)


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: BoundarySpec(low=("pec",)), "two"),
        (lambda: BoundarySpec(low=("pec", "open")), "pec"),
        (lambda: Grid((0.0,), (0.0, 1.0)), "at least two"),
        (lambda: Grid((0.0, np.inf), (0.0, 1.0)), "strictly increasing"),
        (lambda: Grid((0.0, 1.0), (0.0, 1.0), normal_axis=3), "normal_axis"),
    ],
)
def test_compact_models_reject_invalid_configuration(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()


def test_material_models_reject_invalid_tensor_shapes_and_values():
    grid = Grid((0.0, 1.0), (0.0, 1.0))
    tensor = np.zeros((3, 3, 1, 1))
    with pytest.raises(ValueError, match="shape"):
        Materials(grid, tensor[..., 0], tensor)
    invalid = tensor.copy()
    invalid[0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        Materials(grid, invalid, tensor)
    with pytest.raises(ValueError, match="eps_xx"):
        Materials.from_components(
            x_edges=(0.0, 1.0),
            y_edges=(0.0, 1.0),
            eps_xx=np.ones((2, 1)),
        )
    with pytest.raises(ValueError, match="eps_xy"):
        Materials.from_components(
            x_edges=(0.0, 1.0),
            y_edges=(0.0, 1.0),
            eps_xx=np.ones((1, 1)),
            eps_xy=np.ones((2, 1)),
        )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"scalar_permittivity": np.ones(3)}, "2D"),
        ({"axis": "q"}, "axis"),
        ({"direction": "forward"}, "direction"),
        ({"solver_direction": "forward"}, "solver_direction"),
        ({"solver_direction": "+y"}, "solver_direction axis"),
        ({"transverse_axes": ("x", "y")}, "transverse_axes"),
        ({"grid_shape": (1, 2, 3)}, "grid_shape"),
        ({"frequency": 0.0}, "frequency"),
        ({"resolution": np.inf}, "resolution"),
        ({"dt": 0.0}, "dt"),
        ({"polarization": "hybrid"}, "polarization"),
    ],
)
def test_mode_plane_spec_rejects_invalid_values(changes, match):
    with pytest.raises(ValueError, match=match):
        _mode_plane_spec(**changes)


def test_discrete_mode_and_solver_reject_invalid_requests(monkeypatch):
    mode = DiscreteMode(
        neff=1.5,
        profiles={"Ey": np.ones(1)},
        backward_profiles={"Ey": np.ones(1)},
        component_indices={"Ey": (slice(None), slice(None), 1)},
        axis="x",
        direction="+x",
        transverse_axes=("z", "y"),
        phase_reference_component="Ey",
        phase_reference_coord=0.0,
        phase_plane_coord=0.0,
        k_num_axis=1.0,
        power_scale=1.0,
        diagnostics={},
    )
    with pytest.raises(ValueError, match="Unknown component"):
        mode.component("Hz")
    with pytest.raises(TypeError, match="ModePlaneSpec"):
        discrete_module.solve_beamz_mode(object())  # type: ignore[arg-type]

    monkeypatch.setattr(discrete_module, "solve_grid", lambda **_kwargs: object())
    monkeypatch.setattr(discrete_module, "_candidate_modes", lambda *_args: [])
    with pytest.raises(ValueError, match="only 0 modes"):
        discrete_module.solve_beamz_mode(_mode_plane_spec())


def test_discrete_helpers_cover_degenerate_and_reverse_paths():
    assert discrete_module._boundary_refractive_index(np.empty((0, 0))) == 0.0
    assert discrete_module._polarization_fraction({}, "x", None) == 1.0
    assert discrete_module._dominant_phase(np.empty(0)) == 0.0
    singleton = np.ones((1, 2))
    assert discrete_module._stagger_half(singleton, axis=0) is singleton
    assert discrete_module._tukey(0, 0.2).size == 0

    profiles = {"Ey": np.zeros(1), "Hz": np.zeros(1)}
    indices = {
        "Ey": (slice(None), slice(None), 0),
        "Hz": (slice(None), slice(None), 0),
    }
    unchanged, scale, flux = (
        discrete_module._normalize_profiles_by_phase_referenced_flux(
            profiles,
            indices,
            axis="x",
            d_area=1.0,
            direction_sign=1.0,
            omega=1.0,
            k_num=1.0,
            ref_coord=0.0,
            resolution=1.0,
        )
    )
    assert unchanged is profiles
    assert scale == 1.0
    assert flux == 0.0

    parity = discrete_module._enforce_componentwise_parity(
        {"Ey": np.ones(2)}, symmetric_axes=(1,)
    )
    np.testing.assert_array_equal(parity["Ey"], 1.0)
    oriented = discrete_module._runtime_oriented_profiles(
        {"Ez": np.ones(1), "Hx": np.ones(1)}, "y", -1.0
    )
    np.testing.assert_array_equal(oriented["Ez"], -1.0)
    np.testing.assert_array_equal(oriented["Hx"], -1.0)


def test_negative_y_profile_build_flips_electric_components():
    spec = _mode_plane_spec(
        axis="y",
        direction="-y",
        solver_direction="-y",
        transverse_axes=("z", "x"),
    )
    fields = {
        name: np.ones((3, 3), dtype=np.complex128)
        for name in discrete_module._COMPONENTS
    }
    profiles, _indices, _extra = discrete_module._build_y_profiles(fields, spec)
    assert all(
        np.all(values >= 0.0)
        for name, values in profiles.items()
        if name.startswith("E")
    )


@pytest.mark.parametrize("raises", [False, True])
def test_refinement_failure_preserves_seed_mode(monkeypatch, raises):
    fields = {
        name: np.ones((1, 1), dtype=np.complex128)
        for name in discrete_module._COMPONENTS
    }
    profiles = {
        "Ex": np.zeros(1),
        "Ey": np.ones(1),
        "Ez": np.zeros(1),
        "Hx": np.zeros(1),
        "Hy": np.zeros(1),
        "Hz": np.ones(1),
    }
    indices = {name: (slice(None), slice(None), 1) for name in profiles}
    monkeypatch.setattr(
        discrete_module,
        "solve_grid",
        lambda **_kwargs: SimpleNamespace(solver_info={}),
    )
    monkeypatch.setattr(
        discrete_module,
        "_candidate_modes",
        lambda *_args: [{"neff": 2.0 + 0.0j, "fields": fields}],
    )
    monkeypatch.setattr(
        discrete_module,
        "_build_profiles",
        lambda *_args: (profiles, indices, {"initial_power": 1.0}),
    )
    if raises:

        def fail_refinement(*_args, **_kwargs):
            raise RuntimeError("refinement failed")

        monkeypatch.setattr(
            discrete_module, "refine_x_mode_at_fixed_beta", fail_refinement
        )
    else:
        monkeypatch.setattr(
            discrete_module,
            "refine_x_mode_at_fixed_beta",
            lambda *_args, **_kwargs: (profiles, 0.0, 1.0, 2.0, 1.0),
        )
        monkeypatch.setattr(
            discrete_module,
            "validate_x_mode_refinement",
            lambda *_args, **_kwargs: (False, {"rejection_reason": "rejected"}),
        )
    spec = _mode_plane_spec(
        component_permittivity={"Ex": np.ones(1)},
        component_permeability={"Hx": np.ones(1)},
    )
    result = discrete_module.solve_beamz_mode(spec)
    assert not result.diagnostics["yee_refinement_accepted"]
    assert result.diagnostics["yee_refinement_rejection_reason"]


def test_shared_field_helpers_cover_missing_and_physical_fallbacks():
    assert _axis_index(None, "x") is None
    assert _axis_coordinate("Ex", None, "x", 1.0) == 0.0
    grid = RectilinearGrid(
        [0.0, 0.2, 0.5],
        [0.0, 0.3, 0.7],
        [0.0, 0.4, 0.9],
    )
    assert _axis_coordinate("Ey", 0, "x", 1.0, grid) == pytest.approx(0.1)
    assert _axis_coordinate("Hz", 0, "x", 1.0, grid) == pytest.approx(0.2)
    assert _axis_coordinate("Hz", 1, "x", 1.0, grid) == pytest.approx(0.5)
    assert _numeric_wave_number(2.0, None, 1.0, 1.5) > 0.0
    physical = _numeric_wave_number(2.0, np.nan, 1.0, 1.5)
    assert np.isfinite(physical)
    assert _modal_overlap({}, {}, "x", 1.0) == 0.0
    assert _modal_overlap({"Ey": np.ones(1)}, {"Hz": np.ones(1)}, "x", 1.0) != 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"eps": np.ones((2, 2, 2))},
        {"eps": np.ones(3), "npml": -1},
        {"eps": np.ones(3), "m": 0},
    ],
)
def test_solve_modes_rejects_invalid_shape_and_counts(kwargs):
    with pytest.raises(ValueError):
        plane_module.solve_modes(
            omega=1.0,
            dL=1.0,
            **kwargs,
        )


def test_plane_helpers_cover_axes_interval_growth_and_snapping_fallback():
    assert plane_module._plane_extent_by_axis("y", 2.0, 3.0) == {"x": 2.0, "z": 3.0}
    assert plane_module._plane_extent_by_axis("z", 2.0, 3.0) == {"x": 2.0, "y": 3.0}
    with pytest.raises(ValueError, match="Unsupported"):
        plane_module._plane_extent_by_axis("q", 2.0, 3.0)
    assert plane_module._ensure_min_interval(2, 2, 5, min_cells=3) == (0, 3)

    class BrokenRegion:
        def axis_coord(self, _axis):
            return 0.0

        def axis_interval(self, _axis):
            raise RuntimeError("no snapped interval")

    local = plane_module._local_mode_plane_spec(
        np.ones((5, 5)),
        axis="x",
        grid_shape=(5, 5, 5),
        center=(2.5, 2.5, 2.5),
        width=2.0,
        height=2.0,
        plane_index=2,
        offset_index=1,
        resolution=1.0,
        snapped_region=BrokenRegion(),
    )
    assert local["scalar_permittivity"].ndim == 2


def _flat_tensor(nx=2, ny=2):
    tensor = np.zeros((3, 3, nx * ny), dtype=np.complex128)
    for axis in range(3):
        tensor[axis, axis] = 1.0
    return tensor


@pytest.mark.parametrize(
    "solve",
    [_scipy.solve_diagonal_scipy_reference, _scipy.solve_tensorial_scipy_reference],
)
def test_sparse_solvers_validate_tensor_shape_and_mode_count(solve):
    steps = (np.ones(2), np.ones(2))
    common = {
        "dlf": steps,
        "dlb": steps,
        "neff_guess": 1.0,
        "direction": "+",
        "derivative_scale": 1.0,
    }
    invalid = np.zeros((3, 3, 3))
    with pytest.raises(ValueError, match="shape"):
        solve(eps_tensor=invalid, mu_tensor=invalid, num_modes=1, **common)
    tensor = _flat_tensor()
    with pytest.raises(ValueError, match="positive"):
        solve(eps_tensor=tensor, mu_tensor=tensor, num_modes=0, **common)


def test_sparse_helpers_cover_degenerate_pml_and_eigen_paths():
    tensor = _flat_tensor()
    steps = (np.ones(2), np.ones(2))
    with pytest.raises(ValueError, match="omega"):
        _scipy._create_derivative_matrices(
            sparse,
            eps_tensor=tensor,
            mu_tensor=tensor,
            shape=(2, 2),
            dlf=steps,
            dlb=steps,
            omega=None,
            num_pml=(1, 0),
            pml_profile=None,
            dmin_pml=(True, True),
            dmin_pmc=(False, False),
            scale=1.0,
        )
    assert _scipy._make_dxf(sparse, np.ones(1), (1, 2), False).shape == (2, 2)
    assert _scipy._make_dxb(sparse, np.ones(1), (1, 2), False).shape == (2, 2)

    empty = sparse.csc_matrix((2, 2), dtype=np.complex128)
    matrix, vector, guess = _scipy._real_arpack_problem_if_close(empty, None, 1.0)
    assert matrix is empty and vector is None and guess == 1.0
    complex_matrix = sparse.csc_matrix(np.diag([1.0 + 1.0j, 2.0]))
    matrix, _, guess = _scipy._real_arpack_problem_if_close(
        complex_matrix, None, 1.0 + 1.0j
    )
    assert np.iscomplexobj(matrix.data) and guess == 1.0 + 1.0j

    values, vectors = _scipy._selected_eigenpairs(
        sparse.csc_matrix(np.diag([1.0, 2.0])),
        num_modes=1,
        sigma=1.1,
        krylov_dim=2,
        initial_vector=None,
        spla=spla,
        scipy_linalg=scipy_linalg,
    )
    assert values.shape == (1,) and vectors.shape == (2, 1)
    np.testing.assert_array_equal(
        _scipy._create_sfactor("f", 1.0, np.ones(2), 2, 0, True, np.ones(2), None),
        1.0,
    )
    with pytest.raises(ValueError, match="not recognized"):
        _scipy._create_sfactor("q", 1.0, np.ones(2), 2, 1, True, np.ones(2), None)
    backward = _scipy._create_sfactor_b(1.0, np.ones(4), 4, 2, False, np.ones(2), None)
    assert backward[-1] != 1.0


def test_tensorial_solver_supports_backward_direction(monkeypatch):
    tensor = _flat_tensor()
    monkeypatch.setattr(
        _scipy,
        "_selected_eigenpairs",
        lambda *_args, **_kwargs: (
            np.asarray([1.0]),
            np.ones((16, 1), dtype=np.complex128),
        ),
    )
    n_complex, fields, _info = _scipy.solve_tensorial_scipy_reference(
        eps_tensor=tensor,
        mu_tensor=tensor,
        dlf=(np.ones(2), np.ones(2)),
        dlb=(np.ones(2), np.ones(2)),
        num_modes=1,
        neff_guess=1.0,
        direction="-",
        derivative_scale=1.0,
        krylov_dim=4,
    )
    assert n_complex.shape == (1,)
    assert len(fields) == 6


def test_lorentz_helpers_handle_empty_and_zero_modes():
    empty = np.empty(0, dtype=np.complex128)
    empty_mode = _scipy._ModeFields(*(empty.copy() for _ in range(6)))
    _scipy._apply_dominant_e_phase_convention(empty_mode)

    zeros = np.zeros(1, dtype=np.complex128)
    zero_modes = [
        _scipy._ModeFields(*(zeros.copy() for _ in range(6))),
        _scipy._ModeFields(*(zeros.copy() for _ in range(6))),
    ]
    assert _scipy._normalize_to_unit_power(zero_modes[0], np.ones(1)) == 0.0
    _scipy._apply_dominant_e_phase_convention(zero_modes[0])
    info = _scipy._lorentz_orthogonalize_and_normalize(zero_modes, np.ones(1))
    assert info["lorentz_orthogonality_error"] == 0.0


def test_yee_helpers_reject_degenerate_candidates_and_materials():
    zero = _component_profiles(0.0)
    eps, mu, indices = _component_materials(zero)
    accepted, diagnostics = _yee.validate_x_mode_refinement(
        zero,
        zero,
        indices,
        component_permittivity=eps,
        component_permeability=mu,
        omega=1.0,
        dt=None,
        resolution=1.0,
        k_num=1.0,
        direction_sign=1.0,
    )
    assert not accepted
    assert "magnetic overlap" in diagnostics["rejection_reason"]
    assert "non-forward signed power" in diagnostics["rejection_reason"]
    assert _yee._normalized_component_overlap(zero, zero, ("Ex",)) == 0.0
    assert np.isinf(_yee._rms_impedance(zero))
    electric_size = sum(zero[name].size for name in ("Ex", "Ey", "Ez"))
    magnetic_size = sum(zero[name].size for name in ("Hx", "Hy", "Hz"))
    assert np.isinf(
        _yee._energy_ratio(
            zero,
            np.ones(electric_size),
            np.ones(magnetic_size),
        )
    )
    assert np.isinf(_yee._ratio_change(np.inf, 1.0))
    with pytest.raises(ValueError, match="does not match"):
        _yee._material(np.ones((2, 2)), (slice(None),), (3,))


def test_yee_eigenpair_validation_rejects_non_eigenvectors():
    identity = sparse.eye(3, format="csc")

    class FakeSpla:
        @staticmethod
        def eigs(*_args, **_kwargs):
            return np.asarray([2.0]), np.ones((3, 1))

    with pytest.raises(RuntimeError, match="converged eigenpair"):
        _yee._converged_shift_invert_eigenpairs(
            FakeSpla,
            identity,
            identity,
            target=1.0,
            seed=np.ones(3),
            count=1,
        )


def test_yee_beta_correction_handles_flat_eigenvalue_slope(monkeypatch):
    profiles = _component_profiles()
    eps, mu, indices = _component_materials(profiles)
    e_size = sum(profiles[name].size for name in ("Ex", "Ey", "Ez"))
    h_size = sum(profiles[name].size for name in ("Hx", "Hy", "Hz"))
    curl_e = sparse.csc_matrix((h_size, e_size), dtype=np.complex128)
    curl_h = sparse.csc_matrix((e_size, h_size), dtype=np.complex128)
    monkeypatch.setattr(
        _yee,
        "_yee_curl_operators",
        lambda *_args, **_kwargs: (
            curl_e,
            curl_h,
            curl_e.copy(),
            curl_h.copy(),
        ),
    )
    monkeypatch.setattr(
        _yee,
        "_converged_shift_invert_eigenpairs",
        lambda *_args, **_kwargs: (
            np.asarray([0.0]),
            np.ones((e_size, 1), dtype=np.complex128),
        ),
    )
    refined, *_ = _yee.refine_x_mode_at_fixed_beta(
        profiles,
        indices,
        component_permittivity=eps,
        component_permeability=mu,
        omega=1.0,
        dt=None,
        resolution=1.0,
        k_num=0.5,
        direction_sign=1.0,
    )
    assert set(refined) == set(profiles)


def test_solver_helpers_validate_inputs_and_preserve_model_instances():
    with pytest.raises(TypeError, match="Materials"):
        solver_module._solve_materials(material_grid=object(), freqs=[1.0])  # type: ignore[arg-type]
    pml = PmlSpec((1, 1))
    boundary = BoundarySpec(("pmc", "pec"))
    assert solver_module._resolve_pml_spec(pml) is pml
    assert solver_module._resolve_boundary_spec(boundary) is boundary
    with pytest.raises(ValueError, match="length"):
        solver_module._validate_edges("x_edges", (0.0, 1.0), 2)
    with pytest.raises(ValueError, match="strictly increasing"):
        solver_module._validate_edges("x_edges", (0.0, 0.0), 1)
    assert solver_module._resolve_freqs(freqs=None, wavelength=1.55)[0] > 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        solver_module._resolve_freqs(freqs=[np.nan], wavelength=None)
    with pytest.raises(ValueError, match="normal_axis"):
        solver_module._local_fields_to_global({}, normal_axis=3)
    with pytest.raises(ValueError, match="normal_axis"):
        solver_module._field_data_arrays(
            {},
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 1.0]),
            [1.0],
            normal_axis=3,
            normal_coordinate=0.0,
        )
    assert solver_module._default_initial_vector(3).shape == (3,)


def test_solver_infers_target_index_from_material():
    eps = np.full((2, 2), 2.25)
    result = solver_module.solve_grid(
        eps_xx=eps,
        x_edges=(0.0, 1.0, 2.0),
        y_edges=(0.0, 1.0, 2.0),
        freqs=[2.0e14],
        num_modes=1,
        krylov_dim=4,
    )
    assert result.n_complex.shape == (1, 1)


def test_source_scaling_and_index_helpers_cover_edge_paths(caplog):
    caplog.set_level("DEBUG")
    np.testing.assert_array_equal(_to_real_profile([1.0 + 1.0j]), [1.0])
    assert "imag/real peak ratio" in caplog.text
    with pytest.raises(ValueError, match="non-negative finite"):
        _power_scale(np.nan)

    profiles = {"Ey": np.ones(1), "Ez": None}
    scaled = _scale_profiles_for_power(profiles, 4.0)
    np.testing.assert_array_equal(scaled["Ey"], 2.0)
    assert scaled["Ez"] is None
    first, second = _scale_pair_for_power(np.ones(1), np.ones(1), 4.0)
    np.testing.assert_array_equal(first, 2.0)
    np.testing.assert_array_equal(second, 2.0)

    assert _shift_component_indices_along_axis(None, "x", 1, (2, 2, 2)) is None
    assert (
        _shift_component_indices_along_axis(
            (slice(None), slice(None), slice(None)), "x", 1, (2, 2, 2)
        )
        is None
    )
