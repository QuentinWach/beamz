"""Edge contracts for the public native mode-solver models and results."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import beamz.devices.modes as modes


def _result(
    num_modes: int = 1,
    *,
    field_components: tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
    normal_dim: str = "z",
    offset: float = 0.0,
    include_optional: bool = False,
) -> modes.Result:
    frequencies = np.asarray([1.9e14, 2.0e14])
    coords = {
        "x": np.asarray([-0.5, 0.5]) + offset,
        "y": np.asarray([-0.25, 0.25]),
        "z": np.asarray([0.0]),
        "f": frequencies,
        "mode_index": np.arange(num_modes),
    }
    dims = ("x", "y", "z", "f", "mode_index")
    shape = (2, 2, 1, 2, num_modes)
    fields = {}
    for component in field_components:
        values = np.ones(shape, dtype=np.complex128)
        if component in {"Hx", "Ey"}:
            values *= -1j
        fields[component] = xr.DataArray(
            values,
            dims=dims,
            coords=coords,
            attrs={"normal_dim": normal_dim, "label": component, "ignored": [1, 2]},
        )
    n_complex = xr.DataArray(
        np.full((2, num_modes), 2.0 + 0.01j),
        dims=("f", "mode_index"),
        coords={"f": frequencies, "mode_index": np.arange(num_modes)},
    )
    optional = xr.zeros_like(n_complex) if include_optional else None
    return modes.Result(
        n_complex=n_complex,
        field_components=fields,
        n_group=optional,
        dispersion=optional,
        solver_info={
            "array": np.asarray([1, 2]),
            "complex": 1.0 + 2.0j,
            "nested": (np.float64(3.0), np.bool_(True)),
        },
    )


def test_mode_model_validation_and_conversion_edges():
    with pytest.raises(ValueError, match="two non-negative"):
        modes.PmlSpec(num_cells=(1,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        modes.PmlSpec(num_cells=(-1, 0))
    with pytest.raises(ValueError, match="integers"):
        modes.PmlSpec(num_cells=(True, 0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite and positive"):
        modes.PmlSpec(sigma_max=np.inf)
    with pytest.raises(ValueError, match="greater than"):
        modes.PmlSpec(kappa_min=2.0, kappa_max=1.0)
    with pytest.raises(ValueError, match="positive"):
        modes.PmlSpec(order=0)

    pml = modes.PmlSpec.from_num_cells((1, 2))
    assert pml.as_dict()["num_cells"] == (1, 2)
    assert pml.profile_dict()["order"] == 3

    with pytest.raises(ValueError, match="two boundary"):
        modes.BoundarySpec(low=("pec",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="'pec' or 'pmc'"):
        modes.BoundarySpec(low=("pec", "open"))  # type: ignore[arg-type]
    boundary = modes.BoundarySpec(low=("PMC", "pec"))  # type: ignore[arg-type]
    assert boundary.dmin_pmc == (True, False)
    assert boundary.dmin_pml == (False, True)
    assert boundary.as_dict() == {"low": ("pmc", "pec")}

    with pytest.raises(ValueError, match="normal_axis"):
        modes.Grid((0.0, 1.0), (0.0, 1.0), normal_axis=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least two"):
        modes.Grid((0.0,), (0.0, 1.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        modes.Grid((0.0, 0.0), (0.0, 1.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        modes.Grid((0.0, np.inf), (0.0, 1.0))

    grid = modes.Grid((0.0, 1.0), (0.0, 1.0))
    eps_tensor = np.zeros((3, 3, 1, 1), dtype=np.complex128)
    for axis in range(3):
        eps_tensor[axis, axis] = 2.25
    material = modes.Materials(grid=grid, eps_tensor=eps_tensor)
    assert material.is_diagonal
    assert material.flat_eps_tensor().shape == (3, 3, 1)
    assert material.flat_mu_tensor().shape == (3, 3, 1)
    eps_tensor[...] = 0.0
    assert np.all(np.diagonal(material.eps_tensor[:, :, 0, 0]) == 2.25)
    with pytest.raises(ValueError, match="read-only"):
        material.eps_tensor[...] = 0.0

    with pytest.raises(ValueError, match="eps_tensor"):
        modes.Materials(grid=grid, eps_tensor=np.ones((3, 3, 2, 1)))
    with pytest.raises(ValueError, match="same shape"):
        modes.Materials(
            grid=grid,
            eps_tensor=eps_tensor,
            mu_tensor=np.ones((3, 3, 2, 1)),
        )
    invalid = eps_tensor.copy()
    invalid[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        modes.Materials(grid=grid, eps_tensor=invalid)

    with pytest.raises(ValueError, match="eps_xx"):
        modes.Materials.from_diagonal(
            eps_xx=np.ones((2, 1)),
            x_edges=(0.0, 1.0),
            y_edges=(0.0, 1.0),
        )
    with pytest.raises(ValueError, match="eps_yy"):
        modes.Materials.from_diagonal(
            eps_xx=np.ones((1, 1)),
            eps_yy=np.ones((2, 1)),
            x_edges=(0.0, 1.0),
            y_edges=(0.0, 1.0),
        )
    with pytest.raises(ValueError, match="eps_xy"):
        modes.Materials.from_components(
            eps_xx=np.ones((1, 1)),
            eps_xy=np.ones((2, 1)),
            x_edges=(0.0, 1.0),
            y_edges=(0.0, 1.0),
        )
    with pytest.raises(ValueError, match="axis"):
        modes.Materials.from_slice(
            coord_edges=(0.0, 1.0),
            eps_xx=np.ones(1),
            axis="z",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="at least two"):
        modes.Materials.from_slice(coord_edges=(0.0,), eps_xx=np.ones(0))
    with pytest.raises(ValueError, match="finite and positive"):
        modes.Materials.from_slice(
            coord_edges=(0.0, 1.0), eps_xx=np.ones(1), invariant_width=0.0
        )
    with pytest.raises(ValueError, match="shape"):
        modes.Materials.from_slice(coord_edges=(0.0, 1.0, 2.0), eps_xx=np.ones(1))
    with pytest.raises(ValueError, match="eps_xx is required"):
        modes.Materials.from_slice(
            coord_edges=(0.0, 1.0),
            eps_xx=None,  # type: ignore[arg-type]
        )


def test_subpixel_averaging_edges_and_spec_tuple_normalization():
    grouped = np.asarray([[[[1.0, 2.0], [4.0, 8.0]]]])
    geometric = modes.Materials.average_subpixels(
        grouped, shape=(1, 1), subpixel_shape=(2, 2), method="geometric"
    )
    assert geometric[0, 0] == pytest.approx(np.sqrt(8.0))
    assert (
        modes.Materials.average_subpixels(
            grouped, shape=(1, 1), subpixel_shape=(2, 2), method="min"
        )[0, 0]
        == 1.0
    )
    assert (
        modes.Materials.average_subpixels(
            grouped, shape=(1, 1), subpixel_shape=(2, 2), method="max"
        )[0, 0]
        == 8.0
    )

    invalid_cases = [
        ({"shape": (0, 1)}, "positive cell"),
        ({"subpixel_shape": (0, 1)}, "positive sample"),
        ({"values": np.ones((3, 3))}, "must have shape"),
        ({"values": np.zeros((2, 2)), "method": "harmonic"}, "nonzero"),
        (
            {"values": np.asarray([[1.0, -1.0], [1.0, 1.0]]), "method": "geometric"},
            "positive real",
        ),
        ({"method": "median"}, "method must"),  # type: ignore[dict-item]
    ]
    for changes, message in invalid_cases:
        kwargs = {
            "values": np.ones((2, 2)),
            "shape": (1, 1),
            "subpixel_shape": (2, 2),
            "method": "arithmetic",
            **changes,
        }
        with pytest.raises(ValueError, match=message):
            modes.Materials.average_subpixels(**kwargs)  # type: ignore[arg-type]

    spec = modes.Spec(pml=(1, 2), boundary=("pmc", "pec"))
    assert isinstance(spec.pml, modes.PmlSpec)
    assert isinstance(spec.boundary, modes.BoundarySpec)
    assert spec.pml.num_cells == (1, 2)
    assert spec.boundary.low == ("pmc", "pec")
    assert not spec.has_transform
    with pytest.raises(ValueError, match="target_neff"):
        modes.Spec(target_neff=0.0)
    with pytest.raises(ValueError, match="bend_radius"):
        modes.Spec(bend_radius=0.0, bend_axis=0)
    with pytest.raises(ValueError, match="bend_axis"):
        modes.Spec(bend_axis=2)  # type: ignore[arg-type]


def test_result_optional_io_plotting_and_error_contracts(tmp_path):
    data = _result(include_optional=True)
    path = data.to_hdf5(tmp_path / "complete.h5")
    loaded = modes.Result.from_hdf5(path)
    assert loaded.n_group is not None
    assert loaded.dispersion is not None
    assert loaded.solver_info == {
        "array": [1, 2],
        "complex": {"real": 1.0, "imag": 2.0},
        "nested": [3.0, True],
    }
    assert {"group index", "dispersion"} <= set(data.modes_info)

    without_info = modes.Result(data.n_complex, data.field_components)
    assert (
        modes.Result.from_hdf5(
            without_info.to_hdf5(tmp_path / "without_info.h5")
        ).solver_info
        is None
    )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    assert data.plot_field("Ex", f=1.95e14, val="imag", colorbar=False).images
    assert data.plot_field("Ex", val="magnitude", colorbar=False).images
    figure, _axes = data.plot(components=("Ex",), val="abs")
    assert figure is not None
    figure, axes = data.plot_field_components(
        components=("Ex", "Ey", "Ez", "Hx"), colorbar=False
    )
    assert sum(not axis.get_visible() for axis in axes.ravel()) == 2
    plt.close("all")

    with pytest.raises(ValueError, match="not available"):
        data.plot_field("Ebad")
    with pytest.raises(ValueError, match="val must"):
        data.plot_field("Ex", val="phase")
    with pytest.raises(ValueError, match="none of"):
        data.plot_field_components(components=("Ebad",))

    no_spatial = xr.DataArray(
        np.ones((2, 1)),
        dims=("f", "mode_index"),
        coords={"f": data.n_complex.coords["f"], "mode_index": [0]},
    )
    malformed = modes.Result(data.n_complex, {"Ex": no_spatial})
    with pytest.raises(ValueError, match="one or two"):
        malformed.plot_field("Ex")
    with pytest.raises(ValueError, match="at least one spatial"):
        _ = malformed.pol_fraction
    with pytest.raises(ValueError, match="at least one field"):
        _ = modes.Result(data.n_complex, {}).pol_fraction

    line = xr.DataArray(
        np.ones((2, 2, 1)),
        dims=("x", "f", "mode_index"),
        coords={"x": [-0.5, 0.5], "f": data.n_complex.coords["f"], "mode_index": [0]},
        attrs={"normal_dim": "x"},
    )
    incomplete = modes.Result(data.n_complex, {"Ex": line})
    with pytest.raises(ValueError, match="two tangential"):
        _ = incomplete.pol_fraction
    assert {"wavelength", "n eff", "k eff"} <= set(incomplete.modes_info)

    ambiguous = xr.DataArray(
        np.ones((2, 3, 2, 1)),
        dims=("x", "y", "f", "mode_index"),
        coords={
            "x": [-0.5, 0.5],
            "y": [-1.0, 0.0, 1.0],
            "f": data.n_complex.coords["f"],
            "mode_index": [0],
        },
    )
    with pytest.raises(ValueError, match="two tangential"):
        _ = modes.Result(data.n_complex, {"Ex": ambiguous}).pol_fraction


def test_result_metrics_follow_field_mutations():
    result = _result()
    initial_area = result.mode_area

    for component in ("Ex", "Ey", "Ez"):
        values = result.field_components[component].values
        values[...] = 0.0
        values[0, 0, 0, :, :] = 1.0

    assert not np.array_equal(result.mode_area, initial_area)


def test_result_overlap_validation_and_zero_norm():
    complete = _result()
    electric_only = _result(field_components=("Ex", "Ey", "Ez"))
    ex_only = _result(field_components=("Ex",))

    with pytest.raises(ValueError, match="kind must"):
        complete.overlap(kind="invalid")
    with pytest.raises(ValueError, match="electric overlap requires"):
        ex_only.overlap(kind="electric")
    with pytest.raises(ValueError, match="power overlap requires"):
        electric_only.overlap(kind="power")
    with pytest.raises(ValueError, match="matching spatial grids"):
        complete.overlap(_result(offset=0.1), kind="electric")
    with pytest.raises(ValueError, match="normal dimensions"):
        complete.overlap(_result(normal_dim="y"), kind="electric")

    transposed_fields = {
        name: field.transpose("y", "x", "z", "f", "mode_index")
        for name, field in complete.field_components.items()
    }
    transposed = modes.Result(complete.n_complex, transposed_fields)
    with pytest.raises(ValueError, match="field dimensions"):
        complete.overlap(transposed, kind="electric")

    zero_fields = {
        name: xr.zeros_like(field) for name, field in complete.field_components.items()
    }
    zero = modes.Result(complete.n_complex, zero_fields)
    assert zero.overlap(kind="electric") == 0.0j


def test_sweep_and_tracking_validation_edges():
    one = _result(1)
    two = _result(2)
    with pytest.raises(ValueError, match="one-dimensional"):
        modes.Sweep(np.asarray([[1.0]]), (one,))
    with pytest.raises(ValueError, match="same length"):
        modes.Sweep(np.asarray([1.0, 2.0]), (one,))
    with pytest.raises(ValueError, match="at least one"):
        modes.Sweep(np.asarray([]), ())
    with pytest.raises(ValueError, match="same number"):
        modes.Sweep(np.asarray([1.0, 2.0]), (one, two))

    assert modes.track_modes_by_overlap([]) == ()
    with pytest.raises(ValueError, match="at most 8"):
        modes.track_modes_by_overlap([_result(9)])
    with pytest.raises(ValueError, match="same number"):
        modes.track_modes_by_overlap([one, two])

    optional = _result(2, include_optional=True)
    tracked = modes.track_modes_by_overlap([optional, optional])
    assert tracked[1].n_group is not None
    assert tracked[1].dispersion is not None
