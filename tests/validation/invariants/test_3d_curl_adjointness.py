"""Adjointness check for the canonical complete 3D Yee representation."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import beamz.lattice as lattice
from beamz.devices._boundary_compile import compile_metallic_masks
from beamz.devices.boundaries import PEC
from beamz.lattice import build_h_boundary_views_for_e_3d
from beamz.simulation.kernels import apply_zero_mask
from tests.utils import compiled_grid


def test_complete_3d_pec_curl_pair_is_skew_adjoint(validation_metrics):
    fields = compiled_grid(
        np.ones((6, 8, 18), dtype=np.float32),
        np.zeros((6, 8, 18), dtype=np.float32),
        np.ones((6, 8, 18), dtype=np.float32),
        resolution=1.0,
    )
    fields.metallic_masks = compile_metallic_masks(
        fields.component_shapes, fields.material_grid.shape, [PEC()]
    )
    rng = np.random.default_rng(0)
    values = {
        component: apply_zero_mask(
            jnp.asarray(
                rng.standard_normal(getattr(fields, component).shape, dtype=np.float32)
            ),
            fields.metallic_masks[component],
        )
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }

    curl_h = lattice.curl_e_to_h_3d(values["Ex"], values["Ey"], values["Ez"], 1.0)
    views = build_h_boundary_views_for_e_3d(
        values["Hx"],
        values["Hy"],
        values["Hz"],
        frozenset({"front", "back", "bottom", "top", "left", "right"}),
    )
    curl_e = lattice.curl_h_to_e_3d(
        values["Hx"],
        values["Hy"],
        values["Hz"],
        1.0,
        ex_shape=values["Ex"].shape,
        ey_shape=values["Ey"].shape,
        ez_shape=values["Ez"].shape,
        boundary_views=views,
    )

    lhs = sum(
        jnp.vdot(values[name], curl)
        for name, curl in zip(("Ex", "Ey", "Ez"), curl_e, strict=True)
    )
    rhs = sum(
        jnp.vdot(values[name], curl)
        for name, curl in zip(("Hx", "Hy", "Hz"), curl_h, strict=True)
    )
    scale = abs(float(lhs)) + abs(float(rhs)) + 1e-30
    residual = abs(float(lhs - rhs)) / scale
    validation_metrics.check(
        "relative skew-adjoint residual",
        measured=residual,
        reference=0.0,
        tolerance="kernel_float32",
        resolution="6x8x18 material cells",
        metadata={"boundary": "PEC", "seed": 0},
    )
