"""BeamZ-native equivalent of low-level PML material coupling tests.

The sponge PML merges a graded lossy shell into the compiled conductivity grid.
CPML keeps its sigma/kappa/alpha profiles
auxiliary to the curl correction and leaves ordinary material conductivity
unchanged.
"""

import numpy as np
import pytest

from beamz import PML
from beamz.devices._boundary_compile import compile_absorber_regions
from beamz.lattice import attach_material_coefficients, build_material_coefficients
from tests.utils import compiled_grid

pytestmark = pytest.mark.unit


def _make_fields(shape=(6, 8), *, resolution=0.1, base_sigma=0.0):
    permittivity = np.ones(shape, dtype=np.float32)
    conductivity = np.full(shape, base_sigma, dtype=np.float32)
    permeability = np.ones(shape, dtype=np.float32)
    return compiled_grid(
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=permeability,
        resolution=resolution,
        plane_2d="xy",
    )


def _make_design(shape=(6, 8), *, resolution=0.1):
    return (shape[1] * resolution, shape[0] * resolution, 0.0)


def _attach_pml(fields, pml, design, *, resolution=0.1, dt=1e-15):
    payload = compile_absorber_regions(
        pml, fields, design, resolution=resolution, dt=dt
    )
    fields.has_pml = True
    fields.has_cpml = pml.formulation == "cpml"
    fields.pml_data = payload
    attach_material_coefficients(fields, build_material_coefficients(fields))
    return payload


def test_sponge_pml_merges_left_shell_into_total_conductivity():
    fields = _make_fields()
    design = _make_design()
    pml = PML(edges=["left"], thickness=0.2, sigma_max=6.0, formulation="sponge")

    payload = _attach_pml(fields, pml, design)
    total_sigma = np.asarray(fields.total_conductivity, dtype=np.float64)
    mask = np.asarray(payload["mask"], dtype=bool)
    sigma_shell = np.asarray(payload["sigma_x"] + payload["sigma_y"], dtype=np.float64)

    np.testing.assert_allclose(total_sigma, sigma_shell)
    assert np.any(mask[:, 0])
    assert not np.any(mask[:, -1])
    assert float(np.max(total_sigma[:, 0])) > 0.0
    assert float(np.max(total_sigma[:, -1])) == pytest.approx(0.0)


def test_sponge_pml_merges_right_shell_into_total_conductivity():
    fields = _make_fields()
    design = _make_design()
    pml = PML(edges=["right"], thickness=0.2, sigma_max=6.0, formulation="sponge")

    payload = _attach_pml(fields, pml, design)
    total_sigma = np.asarray(fields.total_conductivity, dtype=np.float64)
    mask = np.asarray(payload["mask"], dtype=bool)

    assert np.any(mask[:, -1])
    assert not np.any(mask[:, 0])
    assert float(np.max(total_sigma[:, -1])) > 0.0
    assert float(np.max(total_sigma[:, 0])) == pytest.approx(0.0)


def test_sponge_pml_preserves_larger_base_conductivity():
    fields = _make_fields(base_sigma=0.0)
    design = _make_design()
    fields.conductivity = fields.conductivity.at[3, 4].set(9.0)
    pml = PML(edges=["left"], thickness=0.2, sigma_max=4.0, formulation="sponge")

    _attach_pml(fields, pml, design)
    total_sigma = np.asarray(fields.total_conductivity, dtype=np.float64)

    assert total_sigma[3, 4] == pytest.approx(9.0)


def test_cpml_auxiliary_profiles_do_not_merge_shell_into_total_conductivity():
    fields = _make_fields(base_sigma=0.25)
    design = _make_design()
    pml = PML(
        edges=["left", "right"],
        thickness=0.2,
        sigma_max=6.0,
        alpha_max=0.5,
        formulation="cpml",
    )

    payload = _attach_pml(fields, pml, design)

    total_sigma = np.asarray(fields.total_conductivity, dtype=np.float64)
    shell = np.asarray(payload["sigma_x"] + payload["sigma_y"], dtype=np.float64)
    np.testing.assert_allclose(total_sigma, np.full_like(total_sigma, 0.25))
    assert float(np.max(shell)) > 0.0
    assert float(np.max(total_sigma)) == pytest.approx(0.25)


def test_removed_sigma_formulation_alias_is_rejected():
    with pytest.raises(ValueError, match="sponge"):
        PML(edges=["left"], thickness=0.2, sigma_max=6.0, formulation="sigma")
