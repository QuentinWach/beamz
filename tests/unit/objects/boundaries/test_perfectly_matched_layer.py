import warnings

import numpy as np
import pytest

from beamz import EPS_0, MU_0, PML, Absorber, um
from beamz.devices._boundary_compile import (
    compile_absorber_regions,
)
from tests.utils import compiled_grid

pytestmark = pytest.mark.unit


def _make_fields_2d(shape=(6, 8), *, resolution=0.1):
    permittivity = np.ones(shape, dtype=np.float32)
    conductivity = np.zeros(shape, dtype=np.float32)
    permeability = np.ones(shape, dtype=np.float32)
    return compiled_grid(
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=permeability,
        resolution=resolution,
        plane_2d="xy",
    )


def _make_fields_3d(shape=(5, 6, 7), *, resolution=0.1):
    permittivity = np.ones(shape, dtype=np.float32)
    conductivity = np.zeros(shape, dtype=np.float32)
    permeability = np.ones(shape, dtype=np.float32)
    return compiled_grid(
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=permeability,
        resolution=resolution,
    )


def _make_design_2d(shape=(6, 8), *, resolution=0.1):
    return (shape[1] * resolution, shape[0] * resolution, 0.0)


def _make_design_3d(shape=(5, 6, 7), *, resolution=0.1):
    return (
        shape[2] * resolution,
        shape[1] * resolution,
        shape[0] * resolution,
    )


def test_pml_parameter_defaults():
    pml = PML()

    assert pml.edges == "all"
    assert pml.thickness == pytest.approx(1.0 * um)
    assert pml.sigma_max is None
    assert pml.m == 3
    assert pml.formulation == "sponge"
    assert pml.kappa_max == pytest.approx(2.0)
    assert pml.alpha_max is None
    assert pml.target_reflection == pytest.approx(1e-6)


def test_sponge_absorber_is_an_explicit_boundary_specification():
    absorber = Absorber(edges=["left"], thickness=0.2, sigma_max=5.0)

    payload = compile_absorber_regions(
        absorber,
        _make_fields_2d(),
        _make_design_2d(),
        resolution=0.1,
        dt=1e-15,
    )

    assert absorber.formulation == "sponge"
    assert absorber.edges == ("left",)
    assert payload["formulation"] == "sponge"
    assert np.any(np.asarray(payload["mask"]))


def test_sigma_max_is_computed_when_omitted():
    fields = _make_fields_2d()
    design = _make_design_2d()
    pml = PML(thickness=0.2, sigma_max=None, formulation="sponge")

    payload = compile_absorber_regions(pml, fields, design, resolution=0.1, dt=1e-15)

    assert pml.sigma_max is None
    assert float(np.max(np.asarray(payload["sigma_x"], dtype=np.float64))) > 0.0
    assert payload["sigma_x"].shape == fields.permittivity.shape
    assert payload["sigma_y"].shape == fields.permittivity.shape
    assert payload["mask"].shape == fields.permittivity.shape
    assert payload["formulation"] == "sponge"


def test_pml_rejects_removed_sigma_formulation_alias():
    with pytest.raises(ValueError, match="sponge"):
        PML(formulation="sigma")


def test_cpml_alpha_is_computed_when_omitted():
    fields = _make_fields_2d()
    design = _make_design_2d()
    pml = PML(
        thickness=0.2,
        sigma_max=5.0,
        alpha_max=None,
        formulation="cpml",
    )

    payload = compile_absorber_regions(pml, fields, design, resolution=0.1, dt=2e-15)

    assert pml.alpha_max is None
    assert float(np.max(np.asarray(payload["alpha_x"], dtype=np.float64))) > 0.0
    assert float(np.max(np.asarray(payload["alpha_y"], dtype=np.float64))) > 0.0


def test_cpml_auto_sigma_uses_target_reflection_formula_directly():
    fields = _make_fields_2d()
    design = _make_design_2d()
    pml = PML(thickness=0.2, sigma_max=None, alpha_max=None, formulation="cpml")
    dt = 2e-15

    payload = compile_absorber_regions(pml, fields, design, resolution=0.1, dt=dt)

    eta = np.sqrt(MU_0 / EPS_0)
    unscaled_sigma = -(
        (pml.m + 1) * np.log(pml.target_reflection) / (2.0 * eta * float(pml.thickness))
    )
    assert pml.sigma_max is None
    assert pml.alpha_max is None
    assert float(np.max(np.asarray(payload["sigma_x"], dtype=np.float64))) == pytest.approx(unscaled_sigma)  # fmt: skip
    assert pytest.approx(0.1) == pml._DEFAULT_CPML_ALPHA_NORMALIZED


def test_cpml_3d_auto_alpha_uses_tuned_default():
    fields = _make_fields_3d()
    design = _make_design_3d()
    pml = PML(thickness=0.2, sigma_max=5.0, alpha_max=None, formulation="cpml")
    dt = 2e-15

    payload = compile_absorber_regions(pml, fields, design, resolution=0.1, dt=dt)

    assert pml.alpha_max is None
    assert float(np.max(np.asarray(payload["alpha_x"], dtype=np.float64))) > 0.0
    assert pytest.approx(0.05) == pml._DEFAULT_3D_CPML_ALPHA_NORMALIZED


def test_cpml_auto_parameters_do_not_leak_between_shared_pml_uses():
    shared = PML(thickness=0.2, sigma_max=None, alpha_max=None, formulation="cpml")
    compile_absorber_regions(
        shared,
        _make_fields_2d(),
        _make_design_2d(),
        resolution=0.1,
        dt=2e-15,
    )

    shared_payload = compile_absorber_regions(
        shared,
        _make_fields_3d(),
        _make_design_3d(),
        resolution=0.1,
        dt=2e-15,
    )
    fresh = PML(
        thickness=0.2,
        sigma_max=None,
        alpha_max=None,
        formulation="cpml",
    )
    fresh_payload = compile_absorber_regions(
        fresh,
        _make_fields_3d(),
        _make_design_3d(),
        resolution=0.1,
        dt=2e-15,
    )

    assert shared.sigma_max is None
    assert shared.alpha_max is None
    np.testing.assert_allclose(shared_payload["alpha_x"], fresh_payload["alpha_x"])


def test_profile_shapes_match_2d_field_grid():
    fields = _make_fields_2d(shape=(7, 9))
    design = _make_design_2d(shape=(7, 9))
    pml = PML(
        edges=["left", "right"],
        thickness=0.2,
        sigma_max=5.0,
        alpha_max=0.5,
        formulation="cpml",
    )

    payload = compile_absorber_regions(pml, fields, design, resolution=0.1, dt=1e-15)

    for key in (
        "mask",
        "sigma_x",
        "sigma_y",
        "sigma_z",
        "kappa_x",
        "kappa_y",
        "kappa_z",
        "alpha_x",
        "alpha_y",
        "alpha_z",
    ):
        assert payload[key].shape == fields.permittivity.shape

    tm_xy = payload["tm_xy_cpml"]
    assert tm_xy["Ez_x_sigma"].shape == fields.Ez.shape
    assert tm_xy["Ez_y_sigma"].shape == fields.Ez.shape
    assert tm_xy["Hx_y_sigma"].shape == fields.Hx.shape
    assert tm_xy["Hy_x_sigma"].shape == fields.Hy.shape


def test_pml_warns_when_material_changes_along_absorber_normal():
    eps = np.ones((6, 8), dtype=np.float32)
    eps[:, 0] = 3.0
    eps[:, 1] = 2.0
    fields = compiled_grid(
        permittivity=eps,
        conductivity=np.zeros_like(eps),
        permeability=np.ones_like(eps),
        resolution=0.1,
        plane_2d="xy",
    )
    design = _make_design_2d(shape=eps.shape)
    pml = PML(edges=["left"], thickness=0.2, sigma_max=5.0)

    with pytest.warns(RuntimeWarning, match="PML material varies"):
        compile_absorber_regions(
            pml,
            fields,
            design,
            resolution=0.1,
            dt=1e-15,
        )


def test_cpml_extends_material_changes_along_absorber_normal():
    eps = np.ones((6, 8), dtype=np.float32)
    eps[:, 0] = 3.0
    eps[:, 1] = 2.0
    sigma = np.zeros_like(eps)
    sigma[:, 0] = 0.7
    sigma[:, 1] = 0.4
    mu = np.ones_like(eps)
    mu[:, 0] = 1.5
    mu[:, 1] = 1.25
    fields = compiled_grid(
        permittivity=eps,
        conductivity=sigma,
        permeability=mu,
        resolution=0.1,
        plane_2d="xy",
    )
    design = _make_design_2d(shape=eps.shape)
    pml = PML(
        edges=["left"],
        thickness=0.2,
        sigma_max=5.0,
        alpha_max=0.5,
        formulation="cpml",
    )

    compile_absorber_regions(
        pml,
        fields,
        design,
        resolution=0.1,
        dt=1e-15,
    )

    np.testing.assert_allclose(np.asarray(fields.permittivity)[:, :2], 1.0)
    np.testing.assert_allclose(np.asarray(fields.conductivity)[:, :2], 0.0)
    np.testing.assert_allclose(np.asarray(fields.permeability)[:, :2], 1.0)


def test_cpml_3d_compact_profiles_extend_material_changes():
    eps = np.ones((5, 6, 7), dtype=np.float32)
    eps[:, :, :2] = 3.0
    sigma = np.zeros_like(eps)
    sigma[:, :, :2] = 0.7
    mu = np.ones_like(eps)
    mu[:, :, :2] = 1.5
    fields = compiled_grid(
        permittivity=eps,
        conductivity=sigma,
        permeability=mu,
        resolution=0.1,
    )
    design = _make_design_3d(shape=eps.shape)
    pml = PML(
        edges=["left"],
        thickness=0.2,
        sigma_max=5.0,
        alpha_max=0.5,
        formulation="cpml",
    )

    payload = compile_absorber_regions(pml, fields, design, resolution=0.1, dt=1e-15)

    assert payload["sigma_x"].shape == (1, 1, eps.shape[2])
    np.testing.assert_allclose(np.asarray(fields.permittivity)[:, :, :2], 1.0)
    np.testing.assert_allclose(np.asarray(fields.conductivity)[:, :, :2], 0.0)
    np.testing.assert_allclose(np.asarray(fields.permeability)[:, :, :2], 1.0)


def test_pml_allows_material_extruded_through_absorber():
    eps = np.ones((6, 8), dtype=np.float32)
    eps[:, :3] = 2.0
    fields = compiled_grid(
        permittivity=eps,
        conductivity=np.zeros_like(eps),
        permeability=np.ones_like(eps),
        resolution=0.1,
        plane_2d="xy",
    )
    design = _make_design_2d(shape=eps.shape)
    pml = PML(edges=["left"], thickness=0.2, sigma_max=5.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile_absorber_regions(
            pml,
            fields,
            design,
            resolution=0.1,
            dt=1e-15,
        )

    assert not [
        warning for warning in caught if "PML material varies" in str(warning.message)
    ]


def test_cpml_3d_base_profiles_stay_axis_compact():
    fields = _make_fields_3d(shape=(5, 6, 7))
    design = _make_design_3d(shape=(5, 6, 7))
    pml = PML(
        edges=["left", "front"],
        thickness=0.2,
        sigma_max=5.0,
        alpha_max=0.5,
        formulation="cpml",
    )

    payload = compile_absorber_regions(pml, fields, design, resolution=0.1, dt=1e-15)

    assert "mask" not in payload
    assert payload["sigma_x"].shape == (1, 1, fields.permittivity.shape[2])
    assert payload["sigma_y"].shape == (1, fields.permittivity.shape[1], 1)
    assert payload["sigma_z"].shape == (fields.permittivity.shape[0], 1, 1)
    assert payload["kappa_x"].shape == payload["sigma_x"].shape
    assert payload["kappa_y"].shape == payload["sigma_y"].shape
    assert payload["kappa_z"].shape == payload["sigma_z"].shape
    assert payload["alpha_x"].shape == payload["sigma_x"].shape
    assert payload["alpha_y"].shape == payload["sigma_y"].shape
    assert payload["alpha_z"].shape == payload["sigma_z"].shape

    assert payload["cpml3d_Hxy_sigma"].shape == (1, fields.Hx.shape[1], 1)
    assert payload["cpml3d_Hyz_sigma"].shape == (fields.Hy.shape[0], 1, 1)
    assert payload["cpml3d_Hzx_sigma"].shape == (1, 1, fields.Hz.shape[2])
    assert payload["cpml3d_Exy_sigma"].shape == (1, fields.Ex.shape[1], 1)
    assert payload["cpml3d_Eyz_sigma"].shape == (fields.Ey.shape[0], 1, 1)
    assert payload["cpml3d_Ezx_sigma"].shape == (1, 1, fields.Ez.shape[2])


def test_face_resolution_matches_dimensionality():
    pml = PML(edges="all", thickness=0.2)

    assert pml._get_edges_for_dimensionality(False) == [
        "left",
        "right",
        "top",
        "bottom",
    ]
    assert pml._get_edges_for_dimensionality(True) == [
        "left",
        "right",
        "top",
        "bottom",
        "front",
        "back",
    ]


def test_explicit_edges_are_preserved():
    pml = PML(edges=["left", "top"], thickness=0.2)

    assert pml._get_edges_for_dimensionality(False) == ["left", "top"]
    assert pml._get_edges_for_dimensionality(True) == ["left", "top"]


def test_thickness_is_preserved():
    pml = PML(thickness=0.35, sigma_max=4.0)
    assert pml.thickness == pytest.approx(0.35)


def test_cpml_grid_profiles_are_directional_and_identity_interior():
    fields = _make_fields_2d(shape=(8, 12))
    design = _make_design_2d(shape=(8, 12))
    pml = PML(
        thickness=0.3,
        sigma_max=10.0,
        alpha_max=1.0,
        formulation="cpml",
    )

    payload = compile_absorber_regions(pml, fields, design, resolution=0.1, dt=1e-15)

    center_y = fields.permittivity.shape[0] // 2
    center_x = fields.permittivity.shape[1] // 2
    x_sigma = np.asarray(payload["sigma_x"][center_y], dtype=np.float64)
    x_kappa = np.asarray(payload["kappa_x"][center_y], dtype=np.float64)
    x_alpha = np.asarray(payload["alpha_x"][center_y], dtype=np.float64)
    y_sigma = np.asarray(payload["sigma_y"][:, center_x], dtype=np.float64)
    y_kappa = np.asarray(payload["kappa_y"][:, center_x], dtype=np.float64)
    y_alpha = np.asarray(payload["alpha_y"][:, center_x], dtype=np.float64)

    assert x_sigma[0] > x_sigma[1] > x_sigma[2] > 0.0
    assert x_sigma[-1] > x_sigma[-2] > x_sigma[-3] > 0.0
    assert x_kappa[0] > x_kappa[1] > x_kappa[2] > 1.0
    assert x_kappa[-1] > x_kappa[-2] > x_kappa[-3] > 1.0
    assert x_alpha[0] == pytest.approx(0.0)
    assert x_alpha[1] < x_alpha[2]
    assert x_alpha[-3] > x_alpha[-2]
    assert x_alpha[-1] == pytest.approx(0.0)
    np.testing.assert_allclose(x_sigma[3:-3], 0.0)
    np.testing.assert_allclose(x_kappa[3:-3], 1.0)
    np.testing.assert_allclose(x_alpha[3:-3], 0.0)

    assert y_sigma[0] > y_sigma[1] > 0.0
    assert y_sigma[-1] > y_sigma[-2] > 0.0
    assert y_kappa[0] > y_kappa[1] > 1.0
    assert y_kappa[-1] > y_kappa[-2] > 1.0
    assert y_alpha[0] == pytest.approx(0.0)
    assert y_alpha[1] > 0.0
    assert y_alpha[-2] > 0.0
    assert y_alpha[-1] == pytest.approx(0.0)
    np.testing.assert_allclose(y_sigma[3:-3], 0.0)
    np.testing.assert_allclose(y_kappa[3:-3], 1.0)
    np.testing.assert_allclose(y_alpha[3:-3], 0.0)
