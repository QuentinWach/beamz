"""Solution-level convergence of tensor-smoothed imported geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from beamz.design import raster
from beamz.devices.modes.solver import solve_grid


def _rotated_waveguide_scene() -> raster.Scene:
    corners = np.asarray(((-0.5, -0.15), (0.5, -0.15), (0.5, 0.15), (-0.5, 0.15)))
    angle = np.deg2rad(27.0)
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    polygon = raster.Polygon(tuple(map(tuple, corners @ rotation.T + (2.0, 2.0))))
    return raster.Scene(
        (raster.Material(1.44**2), raster.Material(3.4**2)),
        (raster.Object(raster.ExtrudedPolygon(polygon, 0.0, 1.0), 1),),
    )


def _effective_index(scene: raster.Scene, cells: int) -> float:
    result = raster.rasterize(
        scene,
        raster.Grid.uniform((0, 0, 0), (4, 4, 1), (cells, cells, 1)),
        options=raster.RasterOptions(quality="reference", smoothing="farjadpour_full"),
    )
    epsilon = result.tensors["epsilon"][:, 0].transpose(0, 2, 1)
    components: dict[str, np.ndarray] = {
        name: values
        for name, values in zip(
            ("eps_xx", "eps_yy", "eps_zz", "eps_xy", "eps_xz", "eps_yz"),
            epsilon,
            strict=True,
        )
    }
    components.update(
        eps_yx=components["eps_xy"],
        eps_zx=components["eps_xz"],
        eps_zy=components["eps_yz"],
    )
    modes = solve_grid(
        **components,
        x_edges=np.linspace(0.0, 4.0, cells + 1).tolist(),
        y_edges=np.linspace(0.0, 4.0, cells + 1).tolist(),
        wavelength=[1.55],
        num_modes=1,
        target_neff=3.0,
        krylov_dim=32,
    )
    return float(modes.n_eff.values[0, 0])


@pytest.mark.simulation
def test_farjadpour_waveguide_mode_converges_under_grid_refinement():
    scene = _rotated_waveguide_scene()
    coarse = _effective_index(scene, 12)
    medium = _effective_index(scene, 24)
    fine = _effective_index(scene, 48)

    observed_order = math.log2(abs(coarse - medium) / abs(medium - fine))
    assert 1.44 < coarse < 3.4
    assert 1.44 < fine < 3.4
    assert 1.5 < observed_order < 2.3
