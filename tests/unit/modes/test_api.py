"""Core tests for the BeamZ-native mode solver."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from beamz.devices.modes._scipy import _pml_average_all_sides
from beamz.devices.modes.constants import C_0
from beamz.devices.modes.discrete import ModePlaneSpec, solve_beamz_mode
from beamz.devices.modes.solver import solve_grid
from beamz.devices.modes.specs import ModeData, ModeSpec
from beamz.lattice import component_shape_3d


def _edges(start, stop, count):
    return tuple(float(value) for value in np.linspace(start, stop, count))


def _strip(nx=6, ny=5):
    x_edges = _edges(-1.0, 1.0, nx + 1)
    y_edges = _edges(-0.8, 0.8, ny + 1)
    x = (np.asarray(x_edges[:-1]) + x_edges[1:]) / 2
    y = (np.asarray(y_edges[:-1]) + y_edges[1:]) / 2
    xx, yy = np.meshgrid(x, y, indexing="ij")
    eps = np.where((np.abs(xx) <= 0.35) & (np.abs(yy) <= 0.25), 3.4**2, 1.44**2)
    return eps, x_edges, y_edges


def _plane_spec(axis):
    grid_shape = (7, 8, 9)
    resolution = 60e-9
    transverse_axes = {
        "x": ("z", "y"),
        "y": ("z", "x"),
        "z": ("y", "x"),
    }[axis]
    counts = {"x": (7, 8), "y": (7, 9), "z": (8, 9)}[axis]
    rows = np.arange(counts[0]) - 0.5 * (counts[0] - 1)
    cols = np.arange(counts[1]) - 0.5 * (counts[1] - 1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    eps = np.where((np.abs(rr) <= 1.5) & (np.abs(cc) <= 1.5), 2.2**2, 1.44**2)
    direction = f"+{axis}"
    return ModePlaneSpec(
        scalar_permittivity=eps,
        frequency=C_0 / 1.55,
        resolution=resolution,
        dt=0.35 * resolution / 299_792_458.0,
        axis=axis,
        direction=direction,
        solver_direction="+y" if axis == "y" else direction,
        transverse_axes=transverse_axes,
        grid_shape=grid_shape,
        center=tuple(0.5 * count * resolution for count in grid_shape[::-1]),
        width=0.48e-6,
        height=0.48e-6,
        plane_index=3,
        offset_index=2,
        polarization="te",
        target_neff=2.0,
        num_modes=1,
    )


def test_mode_values_validate_and_snapshot_arrays():
    with pytest.raises(ValueError, match="polarization"):
        ModeSpec(polarization="longitudinal")

    data = ModeData(
        frequencies=np.asarray([1.0]),
        neffs=np.asarray([[2.0]]),
        e_fields=np.ones((1, 1, 3, 2)),
        h_fields=np.ones((1, 1, 3, 2)),
        eps_profiles=np.ones((1, 2)),
        resolution=1.0,
    )
    assert data.selected_mode()[2] == 2.0
    assert not data.e_fields.flags.writeable


def test_grid_solve_returns_sorted_modes_fields_and_diagnostics():
    eps, x_edges, y_edges = _strip()
    result = solve_grid(
        eps_xx=eps,
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[C_0 / 1.55],
        num_modes=2,
        target_neff=2.5,
        krylov_dim=16,
    )

    assert result.n_complex.shape == (1, 2)
    assert result.n_eff.values[0, 0] >= result.n_eff.values[0, 1]
    assert set(result.field_components) == {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
    run = result.solver_info["runs"][0]
    assert run["backend_kind"] == "diagonal_scipy_reference"
    np.testing.assert_allclose(run["power_norms"], 1.0, rtol=1e-10, atol=1e-10)
    assert run["lorentz_orthogonality_error"] < 1e-8


def test_grid_solve_supports_pml_boundaries_and_tensor_materials():
    eps, x_edges, y_edges = _strip(5, 4)
    coupling = np.full_like(eps, 0.01)
    result = solve_grid(
        eps_xx=eps,
        eps_xy=coupling,
        eps_yx=coupling,
        x_edges=x_edges,
        y_edges=y_edges,
        freqs=[C_0 / 1.55],
        num_modes=1,
        target_neff=2.4,
        pml=(1, 1),
        boundary=("pmc", "pec"),
        krylov_dim=20,
    )

    assert result.solver_info["runs"][0]["backend_kind"] == "tensorial_scipy_reference"
    assert result.solver_info["pml"]["num_cells"] == (1, 1)
    assert result.solver_info["boundary"]["low"] == ("pmc", "pec")


def test_pml_material_averages_include_each_high_side_cell():
    nx, ny = 4, 3
    values = np.fromfunction(lambda ix, iy: 10 * ix + iy, (nx, ny))
    tensor = np.zeros((3, 3, nx * ny), dtype=np.complex128)
    for component in range(3):
        tensor[component, component] = values.ravel()

    averages = _pml_average_all_sides((nx, ny), (1, 1), tensor)

    np.testing.assert_allclose(
        averages,
        [values[0].mean(), values[-1].mean(), values[:, 0].mean(), values[:, -1].mean()],
    )


@pytest.mark.parametrize(
    ("axis", "dims"),
    [(0, ("y", "z", "x")), (1, ("x", "z", "y")), (2, ("x", "y", "z"))],
)
def test_grid_solve_maps_components_to_global_axes(axis, dims):
    eps, first_edges, second_edges = _strip(5, 4)
    result = solve_grid(
        eps_xx=eps,
        x_edges=first_edges,
        y_edges=second_edges,
        freqs=[C_0 / 1.55],
        num_modes=1,
        target_neff=2.5,
        normal_axis=axis,
        krylov_dim=16,
    )

    assert result.field_components["Ex"].dims[:3] == dims
    assert result.field_components["Ex"].attrs["normal_dim"] == "xyz"[axis]


@pytest.mark.parametrize("axis", "xyz")
def test_discrete_mode_shapes_fields_for_each_beamz_axis(axis):
    spec = _plane_spec(axis)
    mode = solve_beamz_mode(spec)

    assert mode.axis == axis
    assert np.isfinite(mode.neff)
    assert set(mode.profiles) == {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}
    for component, profile in mode.profiles.items():
        expected = component_shape_3d(component, spec.grid_shape)
        index = mode.component_indices[component]
        actual = tuple(
            len(range(*part.indices(size))) if isinstance(part, slice) else 1
            for size, part in zip(expected, index, strict=True)
            if isinstance(part, slice)
        )
        assert profile.shape == actual


def test_discrete_request_snapshots_materials_and_validates_direction():
    spec = _plane_spec("x")
    eps = np.asarray(spec.scalar_permittivity).copy()
    updated = replace(spec, scalar_permittivity=eps)
    eps[...] = 0.0
    assert np.all(updated.scalar_permittivity != 0.0)

    with pytest.raises(ValueError, match="direction axis"):
        replace(spec, direction="+y")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"freqs": None, "wavelength": None},
        {"freqs": [1.0], "num_modes": 0},
        {"freqs": [1.0], "direction": "forward"},
        {"freqs": [1.0], "components": ("Ex", "bad")},
        {"freqs": [1.0], "pml": (-1, 0)},
    ],
)
def test_grid_solve_rejects_invalid_options(kwargs):
    eps, x_edges, y_edges = _strip(3, 3)
    with pytest.raises((TypeError, ValueError)):
        solve_grid(eps_xx=eps, x_edges=x_edges, y_edges=y_edges, **kwargs)
