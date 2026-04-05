"""Modal normalization and overlap helpers for mode sources."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _modal_power_2d(e_profile, h_profile, signed_flux_sign, dl):
    """Return 2D modal power using the same convention as port extraction."""
    e_flat = np.asarray(e_profile, dtype=np.complex128).reshape(-1)
    h_flat = np.asarray(h_profile, dtype=np.complex128).reshape(-1)
    n = int(min(e_flat.size, h_flat.size))
    if n <= 0:
        return 0.0
    return float(
        0.5
        * np.real(
            np.sum(float(signed_flux_sign) * e_flat[:n] * np.conjugate(h_flat[:n])) * float(dl)
        )
    )


def _normalize_2d_pair_by_power(h_profile, e_profile, signed_flux_sign, dl, eps=1e-30):
    """Normalize a 2D Huygens pair so |modal power| equals 1."""
    h = np.asarray(h_profile)
    e = np.asarray(e_profile)
    p = _modal_power_2d(e, h, signed_flux_sign=signed_flux_sign, dl=dl)
    if np.isfinite(p) and abs(p) > eps:
        scale = np.sqrt(1.0 / abs(p))
        h = h * scale
        e = e * scale
    return h, e


def _to_real_profile(profile, imag_ratio_warn=1e-2, eps=1e-30):
    """Project profile to real-valued injection coefficients."""
    arr = np.asarray(profile, dtype=np.complex128)
    re = np.real(arr)
    im = np.imag(arr)
    re_peak = float(np.max(np.abs(re))) if re.size else 0.0
    im_peak = float(np.max(np.abs(im))) if im.size else 0.0
    if re_peak > eps and im_peak / re_peak > imag_ratio_warn:
        logger.debug(
            "Mode profile has non-negligible imaginary content before real projection: "
            "imag/real peak ratio=%.3e",
            im_peak / re_peak,
        )
    return re


def _modal_power_3d_from_profiles(profiles, axis, d_area):
    """Return time-averaged modal power from full 3D profile dictionaries."""
    ex = np.asarray(profiles["Ex"], dtype=np.complex128)
    ey = np.asarray(profiles["Ey"], dtype=np.complex128)
    ez = np.asarray(profiles["Ez"], dtype=np.complex128)
    hx = np.asarray(profiles["Hx"], dtype=np.complex128)
    hy = np.asarray(profiles["Hy"], dtype=np.complex128)
    hz = np.asarray(profiles["Hz"], dtype=np.complex128)

    if ex.ndim == 1:
        ex = ex[:, None]
    if ey.ndim == 1:
        ey = ey[:, None]
    if ez.ndim == 1:
        ez = ez[:, None]
    if hx.ndim == 1:
        hx = hx[:, None]
    if hy.ndim == 1:
        hy = hy[:, None]
    if hz.ndim == 1:
        hz = hz[:, None]
    ny = min(
        ex.shape[0], ey.shape[0], ez.shape[0], hx.shape[0], hy.shape[0], hz.shape[0]
    )
    nx = min(
        ex.shape[1], ey.shape[1], ez.shape[1], hx.shape[1], hy.shape[1], hz.shape[1]
    )
    if ny <= 0 or nx <= 0:
        return 0.0

    ex = ex[:ny, :nx]
    ey = ey[:ny, :nx]
    ez = ez[:ny, :nx]
    hx = hx[:ny, :nx]
    hy = hy[:ny, :nx]
    hz = hz[:ny, :nx]

    if axis == "x":
        s_axis = ey * np.conjugate(hz) - ez * np.conjugate(hy)
    elif axis == "y":
        s_axis = ez * np.conjugate(hx) - ex * np.conjugate(hz)
    else:
        s_axis = ex * np.conjugate(hy) - ey * np.conjugate(hx)
    return float(0.5 * np.real(np.sum(s_axis) * float(d_area)))


def _normalize_3d_profiles_by_flux(profiles, axis, d_area=1.0, eps=1e-18):
    """Normalize 3D source profiles so |modal power| equals 1."""
    flux = _modal_power_3d_from_profiles(profiles, axis=axis, d_area=d_area)
    if (not np.isfinite(flux)) or abs(flux) <= eps:
        return profiles

    scale = float(np.sqrt(1.0 / max(abs(flux), eps)))
    scale = float(np.clip(scale, 1e-6, 1e6))
    for key, value in profiles.items():
        profiles[key] = np.asarray(value) * scale
    return profiles


def _backward_3d_mode_from_forward(profiles):
    """Return the backward-going counterpart of a forward 3D modal field set."""
    out = {}
    for key, value in profiles.items():
        arr = np.asarray(value, dtype=np.complex128)
        out[key] = -arr if key.startswith("H") else arr.copy()
    return out


def _make_3d_mode_basis_profiles(profiles, axis, d_area=1.0):
    """Build unit-flux forward/backward 3D basis fields from one solved mode."""
    forward = {
        key: np.asarray(value, dtype=np.complex128) for key, value in profiles.items()
    }
    forward = _normalize_3d_profiles_by_flux(forward, axis=axis, d_area=d_area)
    backward = _backward_3d_mode_from_forward(forward)
    return forward, backward


def _modal_overlap_3d_profiles(field_profiles, mode_profiles, axis, d_area):
    """Symmetric power overlap between a field sample and a 3D modal basis field."""
    comp_map = {
        "x": ("Ey", "Ez", "Hz", "Hy"),
        "y": ("Ez", "Ex", "Hx", "Hz"),
        "z": ("Ex", "Ey", "Hy", "Hx"),
    }
    try:
        e1, e2, h1, h2 = comp_map[str(axis)]
    except KeyError as exc:
        raise ValueError(f"Unsupported axis {axis!r}") from exc

    arrays = {}
    n_common = None
    for name in (e1, e2, h1, h2):
        f_arr = np.asarray(field_profiles[name], dtype=np.complex128).reshape(-1)
        m_arr = np.asarray(mode_profiles[name], dtype=np.complex128).reshape(-1)
        n_local = int(min(f_arr.size, m_arr.size))
        if n_local <= 0:
            return np.complex128(0.0 + 0.0j)
        n_common = n_local if n_common is None else min(n_common, n_local)
        arrays[name] = (f_arr, m_arr)

    n_common = int(max(0, n_common or 0))
    if n_common <= 0:
        return np.complex128(0.0 + 0.0j)

    ef1 = arrays[e1][0][:n_common]
    ef2 = arrays[e2][0][:n_common]
    hf1 = arrays[h1][0][:n_common]
    hf2 = arrays[h2][0][:n_common]
    em1 = arrays[e1][1][:n_common]
    em2 = arrays[e2][1][:n_common]
    hm1 = arrays[h1][1][:n_common]
    hm2 = arrays[h2][1][:n_common]

    overlap = (
        0.25
        * np.sum(
            ef1 * np.conjugate(hm1)
            - ef2 * np.conjugate(hm2)
            + np.conjugate(em1) * hf1
            - np.conjugate(em2) * hf2
        )
        * float(d_area)
    )
    return np.complex128(overlap)


def _project_3d_profiles_to_real(profiles):
    """Project 3D source profiles to real-valued runtime injection arrays."""
    out = {}
    for key, value in profiles.items():
        out[key] = _to_real_profile(value)
    return out
