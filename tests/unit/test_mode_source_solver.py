from types import SimpleNamespace

import numpy as np
import xarray as xr

from beamz.devices.sources.solve import solve_modes


def test_solve_modes_uses_one_native_solve_and_forwards_pml(monkeypatch):
    calls = []
    mode_count = 11
    dims = ("f", "x", "y", "z", "mode_index")
    values = np.ones((1, 1, 4, 3, mode_count), dtype=np.complex128)
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
    assert neffs.shape == (3,)
    assert electric.shape == magnetic.shape == (3, 3, 4, 3)
    assert axis == 0
