import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from beamz import EPS_0, MU_0, PML, AbsorbingLayer, Design, Material, um
from beamz.simulation.fields import Fields

pytestmark = pytest.mark.unit


def _make_fields_2d(shape=(6, 8), *, resolution=0.1):
    permittivity = np.ones(shape, dtype=np.float32)
    conductivity = np.zeros(shape, dtype=np.float32)
    permeability = np.ones(shape, dtype=np.float32)
    return Fields(
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
    return Fields(
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=permeability,
        resolution=resolution,
    )


def _make_design_2d(shape=(6, 8), *, resolution=0.1):
    return Design(
        width=shape[1] * resolution,
        height=shape[0] * resolution,
        material=Material(permittivity=1.0, permeability=1.0, conductivity=0.0),
    )


def _make_design_3d(shape=(5, 6, 7), *, resolution=0.1):
    return Design(
        width=shape[2] * resolution,
        height=shape[1] * resolution,
        depth=shape[0] * resolution,
        material=Material(permittivity=1.0, permeability=1.0, conductivity=0.0),
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


def test_sigma_max_is_computed_when_omitted():
    fields = _make_fields_2d()
    design = _make_design_2d()
    pml = PML(thickness=0.2, sigma_max=None, formulation="sponge")

    payload = pml.create_pml_regions(
        fields, design, resolution=0.1, dt=1e-15, plane_2d="xy"
    )

    assert pml.sigma_max is not None
    assert float(pml.sigma_max) > 0.0
    assert payload["sigma_x"].shape == fields.permittivity.shape
    assert payload["sigma_y"].shape == fields.permittivity.shape
    assert payload["mask"].shape == fields.permittivity.shape
    assert payload["formulation"] == "sponge"


def test_absorbing_layer_uses_sponge_formulation():
    absorber = AbsorbingLayer(thickness=0.2, sigma_max=5.0)

    assert absorber.formulation == "sponge"


def test_sigma_alias_maps_to_sponge():
    pml = PML(formulation="sigma")

    assert pml.formulation == "sponge"


def test_cpml_alpha_is_computed_when_omitted():
    fields = _make_fields_2d()
    design = _make_design_2d()
    pml = PML(
        thickness=0.2,
        sigma_max=5.0,
        alpha_max=None,
        formulation="cpml",
    )

    payload = pml.create_pml_regions(
        fields, design, resolution=0.1, dt=2e-15, plane_2d="xy"
    )

    assert pml.alpha_max is not None
    assert float(pml.alpha_max) > 0.0
    assert float(np.max(np.asarray(payload["alpha_x"], dtype=np.float64))) > 0.0
    assert float(np.max(np.asarray(payload["alpha_y"], dtype=np.float64))) > 0.0


def test_cpml_auto_sigma_and_alpha_use_tuned_defaults():
    fields = _make_fields_2d()
    design = _make_design_2d()
    pml = PML(thickness=0.2, sigma_max=None, alpha_max=None, formulation="cpml")
    dt = 2e-15

    pml.create_pml_regions(fields, design, resolution=0.1, dt=dt, plane_2d="xy")

    eta = np.sqrt(MU_0 / EPS_0)
    unscaled_sigma = -(
        (pml.m + 1) * np.log(pml.target_reflection) / (2.0 * eta * float(pml.thickness))
    )
    assert pml.sigma_max == pytest.approx(
        unscaled_sigma * pml._DEFAULT_CPML_SIGMA_SCALE
    )
    assert pml._DEFAULT_CPML_SIGMA_SCALE == pytest.approx(0.5)
    assert pml.alpha_max == pytest.approx(
        2.0 * EPS_0 * pml._DEFAULT_CPML_ALPHA_NORMALIZED / dt
    )
    assert pml._DEFAULT_CPML_ALPHA_NORMALIZED == pytest.approx(0.1)


def test_cpml_3d_auto_alpha_uses_tuned_default():
    fields = _make_fields_3d()
    design = _make_design_3d()
    pml = PML(thickness=0.2, sigma_max=5.0, alpha_max=None, formulation="cpml")
    dt = 2e-15

    pml.create_pml_regions(fields, design, resolution=0.1, dt=dt)

    assert pml.alpha_max == pytest.approx(
        2.0 * EPS_0 * pml._DEFAULT_3D_CPML_ALPHA_NORMALIZED / dt
    )
    assert pml._DEFAULT_3D_CPML_ALPHA_NORMALIZED == pytest.approx(0.05)


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

    payload = pml.create_pml_regions(
        fields, design, resolution=0.1, dt=1e-15, plane_2d="xy"
    )

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
    fields = Fields(
        permittivity=eps,
        conductivity=np.zeros_like(eps),
        permeability=np.ones_like(eps),
        resolution=0.1,
        plane_2d="xy",
    )
    design = _make_design_2d(shape=eps.shape)
    pml = PML(edges=["left"], thickness=0.2, sigma_max=5.0)

    with pytest.warns(RuntimeWarning, match="PML material varies"):
        pml.create_pml_regions(
            fields,
            design,
            resolution=0.1,
            dt=1e-15,
            plane_2d="xy",
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
    fields = Fields(
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

    pml.create_pml_regions(
        fields,
        design,
        resolution=0.1,
        dt=1e-15,
        plane_2d="xy",
    )

    np.testing.assert_allclose(np.asarray(fields.permittivity)[:, :2], 1.0)
    np.testing.assert_allclose(np.asarray(fields.conductivity)[:, :2], 0.0)
    np.testing.assert_allclose(np.asarray(fields.permeability)[:, :2], 1.0)


def test_pml_allows_material_extruded_through_absorber():
    eps = np.ones((6, 8), dtype=np.float32)
    eps[:, :3] = 2.0
    fields = Fields(
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
        pml.create_pml_regions(
            fields,
            design,
            resolution=0.1,
            dt=1e-15,
            plane_2d="xy",
        )

    assert not [
        warning for warning in caught if "PML material varies" in str(warning.message)
    ]


def test_profile_shapes_match_3d_field_grid():
    fields = _make_fields_3d(shape=(5, 6, 7))
    design = _make_design_3d(shape=(5, 6, 7))
    pml = PML(
        edges=["left", "front"],
        thickness=0.2,
        sigma_max=5.0,
        alpha_max=0.5,
        formulation="cpml",
    )

    payload = pml.create_pml_regions(fields, design, resolution=0.1, dt=1e-15)

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

    assert payload["cpml3d_Hxy_sigma"].shape == fields.Hx.shape
    assert payload["cpml3d_Hyz_sigma"].shape == fields.Hy.shape
    assert payload["cpml3d_Hzx_sigma"].shape == fields.Hz.shape
    assert payload["cpml3d_Exy_sigma"].shape == fields.Ex.shape
    assert payload["cpml3d_Eyz_sigma"].shape == fields.Ey.shape
    assert payload["cpml3d_Ezx_sigma"].shape == fields.Ez.shape


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


def test_staggered_profile_shape_dtype_and_monotonicity():
    pml = PML(
        thickness=0.4,
        sigma_max=10.0,
        alpha_max=1.0,
        formulation="cpml",
    )

    sigma_low_e, kappa_low_e, alpha_low_e = pml._compute_fdtdx_staggered_profile_1d(
        total_samples=10,
        spacing=0.1,
        low_active=True,
        high_active=False,
        sample_kind="E",
    )
    sigma_high_h, kappa_high_h, alpha_high_h = pml._compute_fdtdx_staggered_profile_1d(
        total_samples=10,
        spacing=0.1,
        low_active=False,
        high_active=True,
        sample_kind="H",
    )

    assert sigma_low_e.shape == (10,)
    assert kappa_low_e.shape == (10,)
    assert alpha_low_e.shape == (10,)
    assert sigma_low_e.dtype == jnp.float32
    assert kappa_low_e.dtype == jnp.float32
    assert alpha_low_e.dtype == jnp.float32

    assert float(sigma_low_e[0]) > float(sigma_low_e[1]) > float(sigma_low_e[2]) >= 0.0
    assert float(kappa_low_e[0]) > float(kappa_low_e[1]) >= 1.0
    assert float(alpha_low_e[0]) < float(alpha_low_e[1]) < float(alpha_low_e[2])

    assert sigma_high_h.dtype == jnp.float32
    assert kappa_high_h.dtype == jnp.float32
    assert alpha_high_h.dtype == jnp.float32
    assert (
        float(sigma_high_h[-1])
        > float(sigma_high_h[-2])
        > float(sigma_high_h[-3])
        >= 0.0
    )
    assert float(kappa_high_h[-1]) > float(kappa_high_h[-2]) >= 1.0
    assert float(alpha_high_h[-1]) < float(alpha_high_h[-2]) < float(alpha_high_h[-3])


@pytest.mark.parametrize("sample_kind", ["E", "H"])
def test_staggered_profile_directionality_on_both_faces(sample_kind):
    pml = PML(
        thickness=0.4,
        sigma_max=10.0,
        alpha_max=1.0,
        formulation="cpml",
    )
    profile_fn = getattr(
        pml,
        next(name for name in dir(pml) if name.endswith("_staggered_profile_1d")),
    )
    pml_cells = 4

    sigma_low, kappa_low, alpha_low = profile_fn(
        total_samples=12,
        spacing=0.1,
        low_active=True,
        high_active=False,
        sample_kind=sample_kind,
    )
    sigma_high, kappa_high, alpha_high = profile_fn(
        total_samples=12,
        spacing=0.1,
        low_active=False,
        high_active=True,
        sample_kind=sample_kind,
    )

    sigma_low = np.asarray(sigma_low, dtype=np.float64)
    kappa_low = np.asarray(kappa_low, dtype=np.float64)
    alpha_low = np.asarray(alpha_low, dtype=np.float64)
    sigma_high = np.asarray(sigma_high, dtype=np.float64)
    kappa_high = np.asarray(kappa_high, dtype=np.float64)
    alpha_high = np.asarray(alpha_high, dtype=np.float64)

    assert np.all(np.diff(sigma_low[:pml_cells]) <= 0.0)
    assert np.all(np.diff(kappa_low[:pml_cells]) <= 0.0)
    assert np.all(np.diff(alpha_low[:pml_cells]) >= 0.0)
    np.testing.assert_allclose(sigma_low[pml_cells:], 0.0)
    np.testing.assert_allclose(kappa_low[pml_cells:], 1.0)
    np.testing.assert_allclose(alpha_low[pml_cells:], 0.0)

    assert np.all(np.diff(sigma_high[-pml_cells:]) >= 0.0)
    assert np.all(np.diff(kappa_high[-pml_cells:]) >= 0.0)
    assert np.all(np.diff(alpha_high[-pml_cells:]) <= 0.0)
    np.testing.assert_allclose(sigma_high[:-pml_cells], 0.0)
    np.testing.assert_allclose(kappa_high[:-pml_cells], 1.0)
    np.testing.assert_allclose(alpha_high[:-pml_cells], 0.0)


def test_cpml_grid_profiles_are_directional_and_identity_interior():
    fields = _make_fields_2d(shape=(8, 12))
    design = _make_design_2d(shape=(8, 12))
    pml = PML(
        thickness=0.3,
        sigma_max=10.0,
        alpha_max=1.0,
        formulation="cpml",
    )

    payload = pml.create_pml_regions(
        fields, design, resolution=0.1, dt=1e-15, plane_2d="xy"
    )

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
