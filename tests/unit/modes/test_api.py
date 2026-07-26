"""Integration-style tests for the public native mode-solver API."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import xarray as xr

import beamz.devices.modes as mm
from beamz.devices.modes.discrete import _boundary_refractive_index


def test_beamz_mode_specs_validate_selection_and_empty_results():
    with pytest.raises(ValueError, match="polarization"):
        mm.ModeSpec(polarization="longitudinal")

    empty = mm.ModeData(
        frequencies=np.asarray([]),
        neffs=np.empty((0, 1)),
        e_fields=np.empty((0, 1, 3, 0)),
        h_fields=np.empty((0, 1, 3, 0)),
        eps_profiles=np.empty((0, 0)),
        resolution=1.0,
    )
    with pytest.raises(ValueError, match="no frequencies"):
        empty.selected_mode()


def _linspace_edges(start: float, stop: float, count: int) -> tuple[float, ...]:
    """Return linspace edges used by tests."""
    return tuple(float(value) for value in np.linspace(start, stop, count))


def _edge_centers(edges: Sequence[float]) -> np.ndarray:
    """Return edge centers used by tests."""
    edge_array = np.asarray(edges, dtype=float)
    return (edge_array[:-1] + edge_array[1:]) / 2


def _solver_info(data: mm.Result) -> dict[str, Any]:
    """Return solver info used by tests."""
    assert data.solver_info is not None
    return data.solver_info


def _strip_grid(
    nx: int = 5, ny: int = 4
) -> tuple[np.ndarray, tuple[float, ...], tuple[float, ...]]:
    """Return strip grid used by tests."""
    x_edges = _linspace_edges(-1.0, 1.0, nx + 1)
    y_edges = _linspace_edges(-0.8, 0.8, ny + 1)
    x_centers = _edge_centers(x_edges)
    y_centers = _edge_centers(y_edges)
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
    eps = np.where((np.abs(xx) <= 0.35) & (np.abs(yy) <= 0.25), 3.4**2, 1.44**2)
    return eps, x_edges, y_edges


def _beamz_component_shapes(
    grid_shape: tuple[int, int, int],
) -> dict[str, tuple[int, int, int]]:
    """Return BEAMZ 3D Yee component shapes used by contract tests."""
    nz, ny, nx = grid_shape
    return {
        "Ex": (nz, ny, nx - 1),
        "Ey": (nz, ny - 1, nx),
        "Ez": (nz - 1, ny, nx),
        "Hx": (nz - 1, ny - 1, nx),
        "Hy": (nz - 1, ny, nx - 1),
        "Hz": (nz, ny - 1, nx - 1),
    }


def _indexed_shape(
    shape: tuple[int, int, int], index: tuple[object, object, object]
) -> tuple[int, ...]:
    """Return NumPy shape after applying one BEAMZ component index."""
    out = []
    for size, item in zip(shape, index, strict=True):
        if isinstance(item, slice):
            start, stop, step = item.indices(size)
            out.append(max(0, 1 + (stop - start - 1) // step) if stop > start else 0)
    return tuple(out)


def _beamz_mode_plane_spec(axis: str = "x") -> mm.ModePlaneSpec:
    """Return a small guided BEAMZ mode-plane spec."""
    grid_shape = (7, 8, 9)
    resolution = 0.12e-6
    if axis == "x":
        transverse_axes = ("z", "y")
        shape = (grid_shape[0], grid_shape[1])
        plane_index = 3
        offset_index = 2
        direction = "+x"
    elif axis == "y":
        transverse_axes = ("z", "x")
        shape = (grid_shape[0], grid_shape[2])
        plane_index = 3
        offset_index = 2
        direction = "+y"
    else:
        transverse_axes = ("y", "x")
        shape = (grid_shape[1], grid_shape[2])
        plane_index = 3
        offset_index = 2
        direction = "+z"

    rows = np.arange(shape[0]) - 0.5 * (shape[0] - 1)
    cols = np.arange(shape[1]) - 0.5 * (shape[1] - 1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    eps = np.where((np.abs(rr) <= 1.5) & (np.abs(cc) <= 1.5), 2.2**2, 1.44**2)
    return mm.ModePlaneSpec(
        scalar_permittivity=eps,
        frequency=mm.C_0 / 1.55,
        resolution=resolution,
        dt=0.35 * resolution / 299_792_458.0,
        axis=axis,
        direction=direction,
        solver_direction="+y" if axis == "y" else direction,
        transverse_axes=transverse_axes,
        grid_shape=grid_shape,
        component_shapes=_beamz_component_shapes(grid_shape),
        center=(
            0.5 * grid_shape[2] * resolution,
            0.5 * grid_shape[1] * resolution,
            0.5 * grid_shape[0] * resolution,
        ),
        width=0.48e-6,
        height=0.48e-6,
        plane_index=plane_index,
        offset_index=offset_index,
        mode_index=0,
        polarization="te",
        target_neff=2.0,
        num_modes=1,
    )


def _with_component_materials(
    spec: mm.ModePlaneSpec, *, yee_refinement: bool | None = None
) -> mm.ModePlaneSpec:
    component_permittivity = {
        name: np.full(spec.component_shapes[name], 2.25, dtype=np.float64)
        for name in ("Ex", "Ey", "Ez")
    }
    component_permeability = {
        name: np.ones(spec.component_shapes[name], dtype=np.float64)
        for name in ("Hx", "Hy", "Hz")
    }
    updates: dict[str, object] = {
        "component_permittivity": component_permittivity,
        "component_permeability": component_permeability,
    }
    if yee_refinement is not None:
        updates["yee_refinement"] = yee_refinement
    return replace(spec, **updates)


def test_grid_api_solves_with_scipy_solver():
    """Verify grid api solves with scipy solver."""
    eps, x_edges, y_edges = _strip_grid(6, 5)
    freq = mm.C_0 / 1.55

    data = mm.solve_grid(
        eps_xx=eps,
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[freq],
        num_modes=2,
        target_neff=2.5,
        krylov_dim=16,
    )

    assert data.n_complex.shape == (1, 2)
    assert data.n_complex.values[0, 0].real >= data.n_complex.values[0, 1].real
    assert data.field_components["Ex"].shape == (6, 5, 1, 1, 2)
    assert set(data.field_components) == {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
    run_info = _solver_info(data)["runs"][0]
    assert _solver_info(data)["backend"] == "scipy_arpack_reference"
    assert run_info["backend_kind"] == "diagonal_scipy_reference"
    assert run_info["phase_convention"] == "dominant_e_real_positive"
    assert run_info["normalization"] == "lorentz_orthogonal_unit_transverse_power"
    np.testing.assert_allclose(
        run_info["power_norms"], np.ones(2), rtol=1e-10, atol=1e-10
    )
    assert run_info["lorentz_orthogonality_error"] < 1e-8
    assert (
        abs(abs(data.overlap(mode_index=0, kind="power", normalize=False)) - 1.0)
        < 1e-10
    )
    lorentz_matrix = data.overlap_matrix(kind="lorentz").values
    np.testing.assert_allclose(
        lorentz_matrix - np.diag(np.diag(lorentz_matrix)), 0.0, atol=1e-8
    )
    electric = np.stack(
        [
            data.field_components[component].isel(f=0, mode_index=0).values
            for component in ("Ex", "Ey", "Ez")
        ]
    )
    anchor = electric.reshape(-1)[np.argmax(np.abs(electric.reshape(-1)))]
    assert anchor.real >= 0.0
    assert abs(anchor.imag) <= 1e-10 * max(abs(anchor), 1.0)


@pytest.mark.parametrize(
    ("axis", "direction"),
    [(axis, sign + axis) for axis in "xyz" for sign in "+-"],
)
def test_beamz_mode_plane_contract_returns_component_local_profiles(
    axis: str, direction: str
):
    """Verify BEAMZ contract returns component-local profile arrays."""
    spec = replace(
        _beamz_mode_plane_spec(axis),
        direction=direction,
        solver_direction=None,
    )

    discrete = mm.solve_beamz_mode(spec)

    assert discrete.diagnostics["contract"] == "micromode.beamz.DiscreteMode/v1"
    assert discrete.axis == axis
    assert discrete.transverse_axes == spec.transverse_axes
    assert np.isfinite(discrete.neff.real)
    assert set(discrete.profiles) == {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
    assert set(discrete.backward_profiles) == set(discrete.profiles)
    assert set(discrete.component_indices) == set(discrete.profiles)

    for component, profile in discrete.profiles.items():
        index = discrete.component_indices[component]
        assert profile.shape == _indexed_shape(spec.component_shapes[component], index)
        assert discrete.backward_profiles[component].shape == profile.shape
        assert np.isfinite(profile).all()

    assert float(discrete.diagnostics["power_after_phase_reference"]) == pytest.approx(
        1.0, rel=2e-6, abs=2e-6
    )


def test_beamz_mode_plane_spec_snapshots_mutable_inputs():
    base = _beamz_mode_plane_spec()
    eps = np.asarray(base.scalar_permittivity).copy()
    component_eps = np.ones(base.component_shapes["Ex"])
    spec = replace(
        base,
        scalar_permittivity=eps,
        component_permittivity={"Ex": component_eps},
    )

    eps[...] = 0.0
    component_eps[...] = 0.0
    assert np.all(spec.scalar_permittivity != 0.0)
    assert np.all(spec.component_permittivity["Ex"] != 0.0)
    for array in (spec.scalar_permittivity, spec.component_permittivity["Ex"]):
        with pytest.raises(ValueError, match="read-only"):
            array[...] = 0.0
    with pytest.raises(TypeError):
        spec.component_shapes["Ex"] = (1, 1, 1)  # type: ignore[index]


def test_beamz_yee_refinement_can_be_disabled(monkeypatch):
    """Verify callers can explicitly retain the unrefined mode profile."""
    import beamz.devices.modes.discrete as beamz_module

    spec = _with_component_materials(_beamz_mode_plane_spec("x"), yee_refinement=False)

    def unexpected_refinement(*_args, **_kwargs):
        raise AssertionError("disabled mode solve must not call Yee refinement")

    monkeypatch.setattr(
        beamz_module, "refine_x_mode_at_fixed_beta", unexpected_refinement
    )
    discrete = mm.solve_beamz_mode(spec)

    assert discrete.diagnostics["yee_refinement_requested"] is False
    assert discrete.diagnostics["yee_refinement_attempted"] is False
    assert discrete.diagnostics["yee_refinement_accepted"] is False
    assert abs(
        float(discrete.diagnostics["power_after_phase_reference"])
    ) == pytest.approx(1.0, rel=2e-6)


def test_beamz_rejects_unphysical_refinement_and_falls_back_atomically(monkeypatch):
    """Verify a broken E/H candidate cannot replace the original profiles or k."""
    import beamz.devices.modes.discrete as beamz_module

    base_spec = _with_component_materials(
        _beamz_mode_plane_spec("x"), yee_refinement=False
    )
    baseline = mm.solve_beamz_mode(base_spec)

    def broken_refinement(profiles, _indices, **kwargs):
        candidate = {
            name: np.asarray(value, dtype=np.complex128).copy()
            for name, value in profiles.items()
        }
        for component in ("Ex", "Ey", "Ez"):
            candidate[component] *= 1e11
        return candidate, 1e-15, 1.0, 1.2 * float(kwargs["k_num"]), 1.0

    monkeypatch.setattr(beamz_module, "refine_x_mode_at_fixed_beta", broken_refinement)
    refined_spec = _with_component_materials(_beamz_mode_plane_spec("x"))
    assert refined_spec.yee_refinement is True
    guarded = mm.solve_beamz_mode(refined_spec)

    assert guarded.diagnostics["yee_refinement_requested"] is True
    assert guarded.diagnostics["yee_refinement_attempted"] is True
    assert guarded.diagnostics["yee_refinement_accepted"] is False
    assert (
        "E/H impedance change" in guarded.diagnostics["yee_refinement_rejection_reason"]
    )
    assert guarded.k_num_axis == pytest.approx(baseline.k_num_axis)
    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        np.testing.assert_allclose(
            guarded.profiles[component],
            baseline.profiles[component],
            rtol=1e-9,
            atol=1e-9,
        )
    assert abs(
        float(guarded.diagnostics["power_after_phase_reference"])
    ) == pytest.approx(1.0, rel=2e-6)


def test_beamz_boundary_index_uses_all_mode_plane_edges():
    eps = np.ones((5, 6), dtype=float)
    eps[0, 2] = 1.4**2
    eps[-1, 3] = 1.5**2
    eps[2, 0] = 1.6**2
    eps[3, -1] = 1.7**2
    eps[2, 3] = 3.4**2

    assert _boundary_refractive_index(eps) == pytest.approx(1.7)


def test_beamz_mode_plane_contract_validates_direction_axis():
    """Verify BEAMZ contract rejects mismatched direction metadata."""
    spec = _beamz_mode_plane_spec("x")
    with pytest.raises(ValueError, match="direction axis"):
        mm.ModePlaneSpec(
            **{
                **spec.__dict__,
                "direction": "+y",
            }
        )


def test_scipy_solver_reports_operator_diagnostics():
    """Verify scipy solver reports operator diagnostics."""
    eps, x_edges, y_edges = _strip_grid(5, 4)

    data = mm.solve_grid(
        eps_xx=eps,
        x_edges=x_edges,
        y_edges=y_edges,
        wavelength=1.55,
        num_modes=1,
        target_neff=2.5,
        krylov_dim=16,
    )

    run_info = _solver_info(data)["runs"][0]
    assert run_info["backend_kind"] == "diagonal_scipy_reference"
    assert run_info["operator_size"] == 2 * eps.size
    assert run_info["operator_nnz"] > run_info["operator_size"]


def test_materials_api_matches_component_api_for_diagonal_grid():
    """Verify materials api matches component api for diagonal grid."""
    eps, x_edges, y_edges = _strip_grid()
    freq = mm.C_0 / 1.55
    materials = mm.Materials.from_diagonal(eps_xx=eps, x_edges=x_edges, y_edges=y_edges)

    from_materials = mm.solve_modes(
        material_grid=materials,
        freqs=[freq],
        num_modes=2,
        target_neff=2.5,
        krylov_dim=16,
    )
    from_components = mm.solve_grid(
        eps_xx=eps,
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[freq],
        num_modes=2,
        target_neff=2.5,
        krylov_dim=16,
    )

    np.testing.assert_allclose(
        from_materials.n_complex.values, from_components.n_complex.values
    )
    assert from_materials.field_components["Ex"].shape == (5, 4, 1, 1, 2)


def test_scipy_solver_handles_diagonal_grid():
    """Verify scipy solver handles diagonal grid."""
    eps, x_edges, y_edges = _strip_grid(5, 4)
    freq = mm.C_0 / 1.55

    data = mm.solve_grid(
        eps_xx=eps,
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[freq],
        num_modes=2,
        target_neff=2.5,
        krylov_dim=18,
    )

    run_info = _solver_info(data)["runs"][0]
    assert _solver_info(data)["backend"] == "scipy_arpack_reference"
    assert run_info["backend_kind"] == "diagonal_scipy_reference"
    assert run_info["operator_size"] == 2 * eps.size
    assert run_info["operator_nnz"] > run_info["operator_size"]
    np.testing.assert_allclose(
        run_info["power_norms"], np.ones(2), rtol=1e-10, atol=1e-10
    )
    assert run_info["lorentz_orthogonality_error"] < 1e-8


def test_scipy_solver_handles_pml_and_tensorial_paths():
    """Verify scipy solver handles pml and tensorial paths."""
    eps, x_edges, y_edges = _strip_grid(4, 3)

    pml_common = {
        "eps_xx": eps,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "freqs": [mm.C_0 / 1.55],
        "num_modes": 1,
        "target_neff": 2.5,
        "pml": (1, 0),
        "krylov_dim": 18,
    }
    pml_data = mm.solve_grid(**pml_common)

    pml_run = _solver_info(pml_data)["runs"][0]
    assert pml_run["backend_kind"] == "diagonal_scipy_reference"
    assert pml_run["operator_size"] == 2 * eps.size
    assert pml_run["operator_nnz"] > pml_run["operator_size"]

    tensor_common = {
        "eps_xx": eps,
        "eps_yy": np.full_like(eps, 2.2**2),
        "eps_zz": np.full_like(eps, 2.0**2),
        "eps_xz": np.full_like(eps, 0.01),
        "eps_zx": np.full_like(eps, 0.01),
        "x_edges": x_edges,
        "y_edges": y_edges,
        "freqs": [mm.C_0 / 1.55],
        "num_modes": 1,
        "target_neff": 2.2,
        "krylov_dim": 20,
    }
    tensor_data = mm.solve_grid(**tensor_common)

    tensor_run = _solver_info(tensor_data)["runs"][0]
    assert tensor_run["backend_kind"] == "tensorial_scipy_reference"
    assert tensor_run["operator_size"] == 4 * eps.size
    assert tensor_run["operator_nnz"] > tensor_run["operator_size"]


def test_scipy_solver_handles_transformed_grid():
    """Verify scipy solver handles transformed grid."""
    eps, x_edges, y_edges = _strip_grid(4, 3)

    data = mm.solve_grid(
        eps_xx=eps,
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[mm.C_0 / 1.55],
        num_modes=1,
        target_neff=2.5,
        angle_theta=0.08,
        angle_phi=0.25,
        krylov_dim=20,
    )

    assert _solver_info(data)["runs"][0]["backend_kind"] == "tensorial_scipy_reference"


def test_materials_api_accepts_full_tensor_grid():
    """Verify materials api accepts full tensor grid."""
    x_edges = _linspace_edges(-1.0, 1.0, 5)
    y_edges = _linspace_edges(-0.8, 0.8, 4)
    shape = (len(x_edges) - 1, len(y_edges) - 1)
    eps_xx = np.full(shape, 2.2**2)
    eps_yy = np.full(shape, 2.0**2)
    eps_zz = np.full(shape, 1.9**2)
    eps_xz = np.full(shape, 0.03)
    eps_zx = np.full(shape, 0.03)
    materials = mm.Materials.from_components(
        eps_xx=eps_xx,
        eps_yy=eps_yy,
        eps_zz=eps_zz,
        eps_xz=eps_xz,
        eps_zx=eps_zx,
        x_edges=x_edges,
        y_edges=y_edges,
    )

    data = mm.solve_modes(
        material_grid=materials,
        freqs=[mm.C_0 / 1.55],
        num_modes=1,
        target_neff=2.0,
        krylov_dim=20,
    )

    assert not materials.is_diagonal
    assert data.n_complex.shape == (1, 1)
    assert np.isfinite(data.n_complex.values).all()
    assert data.field_components["Ez"].shape == (4, 3, 1, 1, 1)


def test_solver_specs_report_diagnostics_and_persist_to_hdf5(tmp_path):
    """Verify solver specs report diagnostics and persist to hdf5."""
    eps, x_edges, y_edges = _strip_grid(6, 5)
    pml = mm.PmlSpec(
        num_cells=(1, 1), sigma_max=1.6, kappa_min=1.0, kappa_max=2.2, order=2
    )
    boundary = mm.BoundarySpec(low=("pec", "pmc"))

    data = mm.solve_grid(
        eps_xx=eps,
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[mm.C_0 / 1.55],
        num_modes=1,
        target_neff=2.5,
        pml=pml,
        boundary=boundary,
        krylov_dim=18,
    )

    solver_info = _solver_info(data)
    run_info = solver_info["runs"][0]
    assert solver_info["pml"]["num_cells"] == (1, 1)
    assert solver_info["pml"]["sigma_max"] == pytest.approx(1.6)
    assert solver_info["boundary"]["low"] == ("pec", "pmc")
    assert solver_info["backend"]
    assert run_info["operator_size"] > 0
    assert run_info["operator_nnz"] > 0
    assert len(run_info["residuals"]) == 1
    assert run_info["power_norms"][0] == pytest.approx(1.0)
    assert len(run_info["lorentz_norms"]) == 1

    result_path = data.to_hdf5(tmp_path / "diagnostic_result.h5")
    import h5py

    with h5py.File(result_path) as handle:
        assert handle.attrs["format"] == "beamz.devices.modes.Result"
    loaded = mm.Result.from_hdf5(result_path)
    loaded_solver_info = _solver_info(loaded)
    assert loaded_solver_info["pml"]["num_cells"] == [1, 1]
    assert loaded_solver_info["boundary"]["low"] == ["pec", "pmc"]
    assert loaded_solver_info["runs"][0]["operator_size"] == run_info["operator_size"]


def test_slice_api_solves_1d_x_slice_and_matches_single_cell_grid(tmp_path):
    """Verify slice api solves 1d x slice and matches single cell grid."""
    x_edges = _linspace_edges(-1.0, 1.0, 9)
    x_centers = _edge_centers(x_edges)
    eps = np.where(np.abs(x_centers) <= 0.3, 3.4**2, 1.44**2)
    y_edges = (-0.125, 0.125)

    sliced = mm.solve_slice(
        eps_xx=eps,
        coord_edges=x_edges,
        axis="x",
        invariant_width=0.25,
        freqs=[mm.C_0 / 1.55],
        num_modes=2,
        target_neff=2.5,
        krylov_dim=16,
    )
    padded = mm.solve_grid(
        eps_xx=eps[:, None],
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[mm.C_0 / 1.55],
        num_modes=2,
        target_neff=2.5,
        krylov_dim=16,
    )

    np.testing.assert_allclose(sliced.n_complex.values, padded.n_complex.values)
    assert sliced.field_components["Ex"].shape == (8, 1, 1, 1, 2)
    assert sliced.field_components["Ex"].attrs["normal_dim"] == "z"
    np.testing.assert_allclose(
        sliced.field_components["Ex"].coords["y_width"].values, [0.25]
    )
    assert sliced.overlap(mode_index=0, kind="power") == pytest.approx(1.0 + 0.0j)

    path = sliced.to_hdf5(tmp_path / "slice_result.h5")
    loaded = mm.Result.from_hdf5(path)
    assert loaded.field_components["Ex"].attrs["normal_dim"] == "z"
    np.testing.assert_allclose(
        loaded.field_components["Ex"].coords["y_width"].values, [0.25]
    )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ax = sliced.plot_field("Ex", mode_index=0)
    assert ax.lines
    plt.close("all")


def test_materials_slice_api_supports_y_axis_and_tensor_components():
    """Verify materials slice api supports y axis and tensor components."""
    y_edges = _linspace_edges(-0.8, 0.8, 7)
    shape = (len(y_edges) - 1,)
    eps_xx = np.full(shape, 2.2**2)
    eps_yy = np.full(shape, 2.0**2)
    eps_zz = np.full(shape, 1.9**2)
    eps_xz = np.full(shape, 0.02)
    materials = mm.Materials.from_slice(
        coord_edges=y_edges,
        axis="y",
        invariant_width=0.4,
        eps_xx=eps_xx,
        eps_yy=eps_yy,
        eps_zz=eps_zz,
        eps_xz=eps_xz,
    )

    assert materials.shape == (1, 6)
    assert not materials.is_diagonal
    np.testing.assert_allclose(materials.eps_tensor[0, 2, 0, :], eps_xz)

    data = mm.solve_modes(
        material_grid=materials,
        freqs=[mm.C_0 / 1.55],
        num_modes=1,
        target_neff=2.0,
        krylov_dim=16,
    )

    assert data.field_components["Ey"].shape == (1, 6, 1, 1, 1)
    assert data.field_components["Ey"].attrs["normal_dim"] == "z"


def test_y_normal_solve_returns_global_component_names():
    """Verify y normal solve returns global component names."""
    z_edges = _linspace_edges(-1.0, 1.0, 9)
    z_centers = _edge_centers(z_edges)
    eps = np.where(np.abs(z_centers) <= 0.3, 3.4**2, 1.44**2)

    data = mm.solve_slice(
        eps_xx=eps,
        coord_edges=z_edges,
        axis="y",
        invariant_width=0.5,
        freqs=[mm.C_0 / 1.55],
        num_modes=1,
        target_neff=2.5,
        normal_axis=1,
        components=("Ex", "Ez", "Hy", "Hz"),
        krylov_dim=16,
    )

    assert set(data.field_components) == {"Ex", "Ez", "Hy", "Hz"}
    assert data.field_components["Ex"].dims[:3] == ("x", "z", "y")
    assert data.field_components["Ex"].attrs["normal_dim"] == "y"
    assert np.linalg.norm(data.field_components["Ex"].values) > 0
    assert np.linalg.norm(data.field_components["Ez"].values) > 0


def test_x_normal_solve_returns_global_component_names_and_positive_power():
    """Verify x normal solve returns global component names and positive power."""
    y_edges = _linspace_edges(-1.0, 1.0, 7)
    z_edges = _linspace_edges(-0.8, 0.8, 6)
    y_centers = _edge_centers(y_edges)
    z_centers = _edge_centers(z_edges)
    yy, zz = np.meshgrid(y_centers, z_centers, indexing="ij")
    eps = np.where((np.abs(yy) <= 0.35) & (np.abs(zz) <= 0.25), 3.4**2, 1.44**2)

    data = mm.solve_grid(
        eps_xx=eps,
        x_edges=y_edges,
        y_edges=z_edges,
        freqs=[mm.C_0 / 1.55],
        num_modes=1,
        target_neff=2.5,
        normal_axis=0,
        krylov_dim=16,
    )

    assert data.field_components["Ey"].dims[:3] == ("y", "z", "x")
    assert data.field_components["Ey"].attrs["normal_dim"] == "x"
    assert np.linalg.norm(data.field_components["Ey"].values) > 0
    assert np.linalg.norm(data.field_components["Ez"].values) > 0

    power = data.overlap(mode_index=0, kind="power", normalize=False)
    assert power.real > 0
    assert abs(abs(power) - 1.0) < 1e-10


def test_y_normal_power_overlap_is_positive():
    """Verify y normal power overlap is positive."""
    z_edges = _linspace_edges(-1.0, 1.0, 9)
    z_centers = _edge_centers(z_edges)
    eps = np.where(np.abs(z_centers) <= 0.3, 3.4**2, 1.44**2)

    data = mm.solve_slice(
        eps_xx=eps,
        coord_edges=z_edges,
        axis="y",
        invariant_width=0.5,
        freqs=[mm.C_0 / 1.55],
        num_modes=1,
        target_neff=2.5,
        normal_axis=1,
        krylov_dim=16,
    )

    power = data.overlap(mode_index=0, kind="power", normalize=False)
    assert power.real > 0
    assert abs(abs(power) - 1.0) < 1e-10


def test_materials_subpixel_averaging_helpers():
    """Verify materials subpixel averaging helpers."""
    x_edges = _linspace_edges(-1.0, 1.0, 3)
    y_edges = _linspace_edges(-0.5, 0.5, 3)
    samples = np.asarray(
        [
            [1.0, 3.0, 5.0, 7.0],
            [2.0, 4.0, 6.0, 8.0],
            [9.0, 11.0, 13.0, 15.0],
            [10.0, 12.0, 14.0, 16.0],
        ]
    )

    averaged = mm.Materials.average_subpixels(
        samples, shape=(2, 2), subpixel_shape=(2, 2)
    )
    np.testing.assert_allclose(averaged, [[2.5, 6.5], [10.5, 14.5]])

    harmonic = mm.Materials.average_subpixels(
        samples,
        shape=(2, 2),
        subpixel_shape=(2, 2),
        method="harmonic",
    )
    assert np.all(harmonic.real < averaged.real)

    materials = mm.Materials.from_subpixel_diagonal(
        eps_xx=samples,
        x_edges=x_edges,
        y_edges=y_edges,
        subpixel_shape=(2, 2),
    )
    np.testing.assert_allclose(materials.diagonal_eps()[0], averaged)
    assert materials.shape == (2, 2)


def test_angle_and_bend_use_tensorial_path():
    """Verify angle and bend use tensorial path."""
    eps, x_edges, y_edges = _strip_grid()

    data = mm.solve_grid(
        eps_xx=eps,
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[mm.C_0 / 1.55],
        num_modes=1,
        target_neff=2.5,
        angle_theta=0.08,
        angle_phi=0.25,
        bend_radius=8.0,
        bend_axis=0,
        krylov_dim=24,
    )

    assert data.n_complex.shape == (1, 1)
    assert np.isfinite(data.n_complex.values).all()
    assert np.isfinite(data.field_components["Hz"].values).all()


def test_full_tensor_grid_supports_angle_and_bend_transform():
    """Verify full tensor grid supports angle and bend transform."""
    x_edges = _linspace_edges(-1.0, 1.0, 5)
    y_edges = _linspace_edges(-0.8, 0.8, 4)
    shape = (len(x_edges) - 1, len(y_edges) - 1)
    materials = mm.Materials.from_components(
        eps_xx=np.full(shape, 2.2**2),
        eps_yy=np.full(shape, 2.0**2),
        eps_zz=np.full(shape, 1.9**2),
        eps_xz=np.full(shape, 0.02),
        eps_zx=np.full(shape, 0.02),
        x_edges=x_edges,
        y_edges=y_edges,
    )

    data = mm.solve_modes(
        material_grid=materials,
        freqs=[mm.C_0 / 1.55],
        num_modes=1,
        target_neff=2.0,
        angle_theta=0.04,
        angle_phi=0.2,
        bend_radius=10.0,
        bend_axis=0,
        krylov_dim=18,
    )

    assert _solver_info(data)["runs"][0]["backend_kind"] == "tensorial_scipy_reference"
    assert np.isfinite(data.n_complex.values).all()
    assert np.isfinite(data.field_components["Ex"].values).all()


def test_spec_can_drive_grid_solve_options():
    """Verify spec can drive grid solve options."""
    eps, x_edges, y_edges = _strip_grid()
    spec = mm.Spec(num_modes=1, target_neff=2.5)

    data = mm.solve_grid(
        eps_xx=eps,
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[mm.C_0 / 1.55],
        spec=spec,
        krylov_dim=16,
    )

    assert data.n_complex.dtype == np.dtype("complex128")
    assert {field.dtype for field in data.field_components.values()} == {
        np.dtype("complex128")
    }


def test_result_metrics_io_plotting_and_overlap(tmp_path):
    """Verify result metrics io plotting and overlap."""
    eps, x_edges, y_edges = _strip_grid(6, 5)
    data = mm.solve_grid(
        eps_xx=eps,
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[mm.C_0 / 1.55],
        num_modes=2,
        target_neff=2.5,
        krylov_dim=16,
    )

    assert np.isfinite(data.k_eff.values).all()
    assert np.isfinite(data.pol_fraction["te"].values).all()
    assert np.isfinite(data.pol_fraction_waveguide["te"].values).all()
    assert np.isfinite(data.mode_area.values).all()
    frame = data.to_dataframe()
    assert {"n eff", "k eff", "mode area", "TE fraction", "wg TE fraction"} <= set(
        frame.columns
    )

    path = data.to_hdf5(tmp_path / "mode_result.h5")
    loaded = mm.Result.from_hdf5(path)
    np.testing.assert_allclose(loaded.n_complex.values, data.n_complex.values)
    assert set(loaded.field_components) == set(data.field_components)

    assert abs(data.overlap(mode_index=0, kind="power")) > 0
    matrix = data.overlap_matrix(kind="electric")
    assert matrix.shape == (2, 2)
    assert mm.overlap(data, mode_index=0, kind="electric") == pytest.approx(1.0 + 0.0j)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ax = data.plot_field("Ex", mode_index=0)
    assert ax.get_title()
    fig, axes = data.plot_field_components(components=("Ex", "Ey", "Ez"), mode_index=0)
    assert fig is not None
    assert axes.size >= 3
    plt.close("all")


def test_result_from_hdf5_accepts_bytes_solver_info(tmp_path):
    """Verify result from hdf5 accepts bytes solver info."""
    eps, x_edges, y_edges = _strip_grid(6, 5)
    data = mm.solve_grid(
        eps_xx=eps,
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[mm.C_0 / 1.55],
        num_modes=1,
        target_neff=2.5,
        krylov_dim=16,
    )
    path = data.to_hdf5(tmp_path / "bytes_solver_info.h5")

    import h5py

    with h5py.File(path, "a") as handle:
        handle.attrs["solver_info"] = b'{"ok": true}'

    assert mm.Result.from_hdf5(path).solver_info == {"ok": True}


def test_result_overlap_for_synthetic_orthogonal_modes():
    """Verify result overlap for synthetic orthogonal modes."""
    dims = ("x", "y", "z", "f", "mode_index")
    coords = {
        "x": np.asarray([-0.5, 0.5]),
        "y": np.asarray([-0.25, 0.25]),
        "z": np.asarray([0.0]),
        "f": np.asarray([mm.C_0 / 1.55]),
        "mode_index": np.arange(2),
    }
    shape = (2, 2, 1, 1, 2)
    values = {
        component: np.zeros(shape, dtype=np.complex128)
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    values["Ex"][..., 0] = 1.0
    values["Hy"][..., 0] = 1.0
    values["Ey"][..., 1] = 1.0
    values["Hx"][..., 1] = -1.0
    fields = {
        component: xr.DataArray(array, dims=dims, coords=coords)
        for component, array in values.items()
    }
    data = mm.Result(
        n_complex=xr.DataArray(
            [[2.0 + 0j, 1.8 + 0j]],
            dims=("f", "mode_index"),
            coords={"f": coords["f"], "mode_index": coords["mode_index"]},
        ),
        field_components=fields,
    )

    assert data.overlap(mode_index=0, kind="power") == pytest.approx(1.0 + 0.0j)
    assert data.overlap(
        mode_index=0, other_mode_index=1, kind="power"
    ) == pytest.approx(0.0 + 0.0j)
    assert data.overlap(
        mode_index=0, other_mode_index=1, kind="lorentz"
    ) == pytest.approx(0.0 + 0.0j)
    assert mm.overlap(data, mode_index=1, kind="electric") == pytest.approx(1.0 + 0.0j)
    np.testing.assert_allclose(data.overlap_matrix(kind="power").values, np.eye(2))
    np.testing.assert_allclose(data.overlap_matrix(kind="lorentz").values, np.eye(2))


def test_sweep_tracks_modes_by_overlap():
    """Verify sweep tracks modes by overlap."""
    dims = ("x", "y", "z", "f", "mode_index")
    coords = {
        "x": np.asarray([-0.5, 0.5]),
        "y": np.asarray([-0.25, 0.25]),
        "z": np.asarray([0.0]),
        "f": np.asarray([mm.C_0 / 1.55]),
        "mode_index": np.arange(2),
    }
    shape = (2, 2, 1, 1, 2)

    def result(n_values: list[float], order: tuple[str, str]) -> mm.Result:
        """Return result used by tests."""
        fields = {
            component: np.zeros(shape, dtype=np.complex128)
            for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        }
        for mode_index, component in enumerate(order):
            fields[component][..., mode_index] = 1.0
        arrays = {
            component: xr.DataArray(values, dims=dims, coords=coords)
            for component, values in fields.items()
        }
        solver_info = {"backend": "synthetic", "n_values": n_values}
        return mm.Result(
            n_complex=xr.DataArray(
                [np.asarray(n_values, dtype=np.complex128)],
                dims=("f", "mode_index"),
                coords={"f": coords["f"], "mode_index": coords["mode_index"]},
            ),
            field_components=arrays,
            solver_info=solver_info,
        )

    first = result([2.2, 1.9], ("Ex", "Ey"))
    second = result([1.91, 2.18], ("Ey", "Ex"))

    tracked = mm.track_modes_by_overlap([first, second], kind="electric")
    sweep = mm.Sweep(
        values=np.asarray([0.4, 0.5]), results=tracked, parameter_name="width"
    )

    np.testing.assert_allclose(tracked[1].n_complex.values, [[2.18, 1.91]])
    assert tracked[1].solver_info == {"backend": "synthetic", "n_values": [1.91, 2.18]}
    np.testing.assert_allclose(sweep.n_eff[:, 0], [2.2, 2.18])
    assert {"width", "mode_index", "n_eff", "te_fraction"} <= set(
        sweep.to_dataframe().columns
    )


def test_spec_validation_for_core_options():
    """Verify spec validation for core options."""
    spec = mm.Spec(
        num_modes=2,
        target_neff=2.0,
        angle_theta=0.1,
        angle_phi=0.2,
        bend_radius=5.0,
        bend_axis=1,
        pml=mm.PmlSpec(num_cells=(1, 2), sigma_max=1.5),
        boundary=mm.BoundarySpec(low=("pmc", "pec")),
    )

    assert isinstance(spec.pml, mm.PmlSpec)
    assert isinstance(spec.boundary, mm.BoundarySpec)
    assert spec.pml.num_cells == (1, 2)
    assert spec.pml.sigma_max == pytest.approx(1.5)
    assert spec.boundary.dmin_pmc == (True, False)
    assert spec.has_angle
    assert spec.has_bend
    assert spec.has_transform

    with pytest.raises(ValueError, match="num_modes"):
        mm.Spec(num_modes=0)
    with pytest.raises(ValueError, match="bend_axis"):
        mm.Spec(bend_radius=5.0)
    with pytest.raises(ValueError, match="num_cells"):
        mm.PmlSpec(num_cells=(1.5, 0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="order"):
        mm.PmlSpec(order=2.5)  # type: ignore[arg-type]
