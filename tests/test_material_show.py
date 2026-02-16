import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from beamz.design.materials import (
    AnisotropicMaterial,
    CustomMaterial,
    DebyeMaterial,
    DrudeMaterial,
    LorentzMaterial,
    Material,
    Material2D,
    PoleResidueMaterial,
    SellmeierMaterial,
)


@pytest.fixture
def show_counter(monkeypatch):
    calls = {"count": 0}

    def _fake_show():
        calls["count"] += 1
        plt.close("all")

    monkeypatch.setattr(plt, "show", _fake_show)
    return calls


def _assert_show_called(show_counter, fn):
    before = show_counter["count"]
    out = fn()
    assert out is None
    assert show_counter["count"] == before + 1


def test_material_show_calls_display(show_counter):
    mat = Material(name="SilicaConst", permittivity=2.1, conductivity=1e-4, k=1.3)
    _assert_show_called(show_counter, lambda: mat.show(num_points=64))


def test_dispersive_materials_show_call_display(show_counter):
    models = [
        SellmeierMaterial(coeffs=((0.6961663, 0.0684043**2),), name="Sellmeier"),
        DrudeMaterial(coeffs=((2.0e15, 1.0e14),), eps_inf=1.5, name="Drude"),
        LorentzMaterial(coeffs=((2.0, 4.0e14, 5.0e13),), eps_inf=1.0, name="Lorentz"),
        DebyeMaterial(coeffs=((70.0, 8.0e-12),), eps_inf=4.9, name="Debye"),
        PoleResidueMaterial(
            eps_inf=1.0,
            poles=((1.0e14 + 2.0e14j, 2.0e12 + 1.0e12j),),
            name="PoleResidue",
        ),
    ]
    for model in models:
        _assert_show_called(show_counter, lambda m=model: m.show(num_points=80))


def test_custom_material_show_grid_mode(show_counter):
    eps_grid = np.linspace(1.9, 2.4, 20 * 24).reshape(20, 24)
    cond_grid = np.linspace(0.0, 2e-3, 20 * 24).reshape(20, 24)
    mat = CustomMaterial(
        permittivity_grid=eps_grid,
        conductivity_grid=cond_grid,
        bounds=((-2.0, 2.0), (-1.0, 1.0)),
    )
    _assert_show_called(show_counter, mat.show)


def test_custom_material_show_function_mode(show_counter):
    mat = CustomMaterial(
        permittivity_func=lambda x, y: 2.1 + 0.15 * x + 0.05 * y,
        conductivity_func=lambda x, y: 1e-3 + 2e-4 * (x**2 + y**2),
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
    )
    _assert_show_called(show_counter, lambda: mat.show(grid_shape=(36, 30)))


def test_custom_material_show_fallback_mode(show_counter):
    mat = CustomMaterial()
    _assert_show_called(show_counter, mat.show)


def test_material2d_show_calls_display(show_counter):
    mat2d = Material2D(
        ss=SellmeierMaterial(coeffs=((0.6961663, 0.0684043**2),), name="ss"),
        tt=Material(permittivity=2.25, name="tt"),
        name="Sheet",
    )
    _assert_show_called(show_counter, lambda: mat2d.show(num_points=72))


def test_anisotropic_show_calls_display(show_counter):
    aniso = AnisotropicMaterial(
        xx=SellmeierMaterial(coeffs=((0.6961663, 0.0684043**2),), name="xx"),
        yy=Material(permittivity=2.2, name="yy"),
        zz=DrudeMaterial(coeffs=((2.0e15, 1.0e14),), eps_inf=1.0, name="zz"),
    )
    _assert_show_called(show_counter, lambda: aniso.show(num_points=72))
