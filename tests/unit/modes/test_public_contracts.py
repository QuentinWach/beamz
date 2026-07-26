import numpy as np
import xarray as xr

from beamz.devices.modes import ModeData, ModeSpec, Result, solve_grid


def test_native_mode_package_exposes_only_supported_api():
    from beamz.devices import modes

    assert modes.__all__ == ["ModeData", "ModeSpec", "Result", "solve_grid"]
    assert ModeSpec and ModeData and solve_grid


def test_result_is_a_minimal_labeled_solver_value():
    indices = xr.DataArray(
        np.asarray([[2.0 + 0.1j]]),
        dims=("f", "mode_index"),
        coords={"f": [1.0], "mode_index": [0]},
    )
    result = Result(indices, {}, solver_info={"backend": "test"})

    np.testing.assert_allclose(result.n_eff, [[2.0]])
    np.testing.assert_allclose(result.k_eff, [[0.1]])
    assert result.solver_info == {"backend": "test"}
