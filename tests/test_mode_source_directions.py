import numpy as np
from types import SimpleNamespace

import beamz.devices.sources.mode as mode
from beamz.devices.sources.mode import ModeSource


def fake_solve_modes(eps, omega, dL, m, direction, filter_pol, return_fields):
    n = eps.size
    e_fields = np.zeros((1, 3, n), dtype=np.complex128)
    h_fields = np.zeros((1, 3, n), dtype=np.complex128)
    if direction in ("+y", "-y"):
        h_fields[0, 2, :] = np.linspace(1.0, 2.0, n)  # Hy dominant
        h_fields[0, 1, :] = 0.1
    else:
        h_fields[0, 1, :] = np.linspace(1.0, 2.0, n)  # Hx dominant
        h_fields[0, 2, :] = 0.1
    e_fields[0, 0, :] = np.linspace(0.5, 1.5, n)  # Ez
    return np.array([1.5]), e_fields, h_fields, 0


def build_source(direction):
    permittivity = np.ones((6, 8))
    grid = SimpleNamespace(height=permittivity.shape[0], width=permittivity.shape[1])
    src = ModeSource(grid=grid, center=(1.0, 1.0), width=1.0, wavelength=1.55e-6, pol="te",
                     signal=np.ones(10), direction=direction)
    return src, permittivity


def test_initialize_plus_x(monkeypatch):
    monkeypatch.setattr(mode, "solve_modes", fake_solve_modes)
    src, eps = build_source("+x")
    src.initialize(eps, resolution=1.0)
    hy_expected = -np.linspace(1.0, 2.0, eps.shape[0])
    assert src._h_component == "Hx"
    assert src._jz_profile.shape[0] == eps.shape[0]
    assert np.allclose(src._jz_profile, hy_expected)
    assert src._my_profile.shape[0] == eps.shape[0]


def test_initialize_plus_y(monkeypatch):
    monkeypatch.setattr(mode, "solve_modes", fake_solve_modes)
    src, eps = build_source("+y")
    src.initialize(eps, resolution=1.0)
    hx_expected = -np.linspace(1.0, 2.0, eps.shape[1])[: eps.shape[1] - 1]
    assert src._h_component == "Hx"
    assert src._jz_profile.shape[0] == eps.shape[1]
    assert np.allclose(src._jz_profile[:eps.shape[1]-1], hx_expected)
    assert src._my_profile.shape[0] == eps.shape[1] - 1
