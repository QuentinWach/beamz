from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from beamz import LIGHT_SPEED
from beamz.devices.sources.solve import solve_modes


def test_solve_modes_uses_one_native_solve_and_forwards_pml(monkeypatch):
    calls = []
    mode_count = 11
    dims = ("y", "z", "x", "f", "mode_index")
    values = np.ones((3, 4, 1, 1, mode_count), dtype=np.complex128)
    result = SimpleNamespace(
        n_complex=xr.DataArray(
            np.linspace(3.0, 1.0, mode_count)[None, :],
            dims=("f", "mode_index"),
        ),
        field_components={
            name: xr.DataArray(values, dims=dims)
            for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        },
    )

    def fake_solve(**kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr("beamz.devices.sources.solve.solve_fdfd_grid", fake_solve)

    neffs, electric, magnetic, axis = solve_modes(
        np.ones((4, 3)),
        omega=2 * np.pi,
        dL=1e-7,
        npml=2,
        m=3,
        return_fields=True,
    )

    assert len(calls) == 1
    assert calls[0]["num_modes"] == 11
    assert calls[0]["pml"] == (2, 2)
    np.testing.assert_array_equal(calls[0]["eps_xx"], np.ones((4, 3)).T)
    assert neffs.shape == (3,)
    assert electric.shape == magnetic.shape == (3, 3, 4, 3)
    assert axis == 0


@pytest.mark.parametrize("axis", "xyz")
def test_solve_modes_preserves_beamz_plane_axes_for_polarization(axis):
    permittivity = np.full((40, 6), 1.44**2)
    permittivity[16:24, :] = 2.2**2

    neffs, _, _, solved_axis = solve_modes(
        permittivity,
        omega=2 * np.pi * LIGHT_SPEED / 1.55e-6,
        dL=80e-9,
        m=1,
        direction=f"+{axis}",
        filter_pol="te",
        return_fields=True,
        target_neff=2.1,
    )

    assert solved_axis == "xyz".index(axis)
    assert neffs[0].real > 1.8
