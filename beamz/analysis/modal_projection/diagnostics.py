"""Diagnostics for modal projection coefficient solves."""

from __future__ import annotations

import numpy as np


def _modal_projection_reconstruction_residual(field_vec, projection, coeff):
    mode_matrix = np.asarray(
        projection.get("mode_matrix", np.zeros((0, 0), dtype=np.complex128)),
        dtype=np.complex128,
    )
    if mode_matrix.ndim != 2 or mode_matrix.shape[0] <= 0 or mode_matrix.shape[1] < 2:
        return np.nan
    field = np.asarray(field_vec, dtype=np.complex128).reshape(-1)
    if field.size <= 0:
        return np.nan
    n = int(min(field.size, mode_matrix.shape[0]))
    if n <= 0:
        return np.nan
    coeff_arr = np.asarray(coeff, dtype=np.complex128).reshape(-1)
    if coeff_arr.size < 2:
        return np.nan
    recon = mode_matrix[:n, :2] @ coeff_arr[:2]
    target = field[:n]
    denom = float(np.linalg.norm(target))
    if denom <= 1e-30 or not np.isfinite(denom):
        return np.nan
    return float(np.linalg.norm(target - recon) / denom)


def _modal_projection_reconstruction_diagnostics_from_matrix(
    field_vec,
    mode_matrix,
    coeff,
    component_slices=(),
):
    matrix = np.asarray(mode_matrix, dtype=np.complex128)
    empty = {
        "residual": np.nan,
        "residual_e": np.nan,
        "residual_h": np.nan,
        "residual_balanced": np.nan,
        "e_scale": np.nan + 0.0j,
        "h_scale": np.nan + 0.0j,
    }
    if matrix.ndim != 2 or matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        return empty
    field = np.asarray(field_vec, dtype=np.complex128).reshape(-1)
    coeff_arr = np.asarray(coeff, dtype=np.complex128).reshape(-1)
    n = int(min(field.size, matrix.shape[0]))
    m = int(min(coeff_arr.size, matrix.shape[1]))
    if n <= 0 or m <= 0:
        return empty
    target = field[:n]
    recon = matrix[:n, :m] @ coeff_arr[:m]

    def _residual(mask):
        if mask.size <= 0:
            return np.nan
        denom = float(np.linalg.norm(target[mask]))
        if denom <= 1e-30 or not np.isfinite(denom):
            return np.nan
        return float(np.linalg.norm(target[mask] - recon[mask]) / denom)

    def _scale_and_residual(mask):
        if mask.size <= 0:
            return np.nan + 0.0j, np.nan
        target_part = target[mask]
        recon_part = recon[mask]
        denom = np.vdot(recon_part, recon_part)
        if abs(denom) <= 1e-30 or not np.isfinite(abs(denom)):
            return np.nan + 0.0j, np.nan
        scale = np.vdot(recon_part, target_part) / denom
        target_norm = float(np.linalg.norm(target_part))
        if target_norm <= 1e-30 or not np.isfinite(target_norm):
            return scale, np.nan
        residual = float(np.linalg.norm(target_part - scale * recon_part) / target_norm)
        return np.complex128(scale), residual

    all_mask = np.arange(n, dtype=int)
    e_parts = []
    h_parts = []
    for name, start, stop in component_slices:
        lo = max(0, min(int(start), n))
        hi = max(lo, min(int(stop), n))
        if hi <= lo:
            continue
        part = np.arange(lo, hi, dtype=int)
        if str(name).startswith("E"):
            e_parts.append(part)
        elif str(name).startswith("H"):
            h_parts.append(part)
    e_mask = np.concatenate(e_parts) if e_parts else np.asarray([], dtype=int)
    h_mask = np.concatenate(h_parts) if h_parts else np.asarray([], dtype=int)

    e_scale, e_resid_scaled = _scale_and_residual(e_mask)
    h_scale, h_resid_scaled = _scale_and_residual(h_mask)
    balanced_recon = recon.copy()
    if e_mask.size and np.isfinite(abs(e_scale)):
        balanced_recon[e_mask] *= e_scale
    if h_mask.size and np.isfinite(abs(h_scale)):
        balanced_recon[h_mask] *= h_scale
    balanced_denom = float(np.linalg.norm(target))
    balanced = (
        float(np.linalg.norm(target - balanced_recon) / balanced_denom)
        if balanced_denom > 1e-30 and np.isfinite(balanced_denom)
        else np.nan
    )
    return {
        "residual": _residual(all_mask),
        "residual_e": _residual(e_mask),
        "residual_h": _residual(h_mask),
        "residual_balanced": balanced,
        "residual_e_scaled": e_resid_scaled,
        "residual_h_scaled": h_resid_scaled,
        "e_scale": e_scale,
        "h_scale": h_scale,
    }
