"""3D Huygens injection helpers for mode sources."""

import logging

import numpy as np

from beamz.const import EPS_0, MU_0

logger = logging.getLogger(__name__)

_HUYGENS_SIGNS = {
    "x": {
        "e": [("Ey", "Hz", -1), ("Ez", "Hy", +1)],
        "h": [("Hy", "Ez", -1), ("Hz", "Ey", +1)],
    },
    "y": {
        "e": [("Ex", "Hz", -1), ("Ez", "Hx", +1)],
        "h": [("Hx", "Ez", +1), ("Hz", "Ex", -1)],
    },
    "z": {
        "e": [("Ex", "Hy", -1), ("Ey", "Hx", +1)],
        "h": [("Hx", "Ey", +1), ("Hy", "Ex", -1)],
    },
}


def _get_3d_huygens_terms(axis, pol):
    """Return 3D sign terms with TE gauge parity matched to 2D conventions."""
    del pol
    return list(_HUYGENS_SIGNS[axis]["e"]), list(_HUYGENS_SIGNS[axis]["h"])


def _inject_3d_e_fields(fields, profiles, indices, signal_e, dt, resolution, axis, pol):
    """Inject E-field components for 3D Huygens source."""
    e_terms, _ = _get_3d_huygens_terms(axis, pol)
    for e_comp, h_source, sign in e_terms:
        _inject_e_component(
            fields,
            e_comp,
            profiles,
            indices,
            h_source,
            signal_e,
            dt,
            resolution,
            sign=sign,
        )


def _inject_3d_h_fields(fields, profiles, indices, signal_h, dt, resolution, axis, pol):
    """Inject H-field components for 3D Huygens source."""
    _, h_terms = _get_3d_huygens_terms(axis, pol)
    for h_comp, e_source, sign in h_terms:
        _inject_h_component(
            fields,
            h_comp,
            profiles,
            indices,
            e_source,
            signal_h,
            dt,
            resolution,
            sign=sign,
        )


def _inject_3d_fields(
    fields, profiles, indices, signal_e, signal_h, dt, resolution, axis="x", pol="tm"
):
    """Inject all 3D Huygens field components."""
    _inject_3d_h_fields(fields, profiles, indices, signal_h, dt, resolution, axis, pol)
    _inject_3d_e_fields(fields, profiles, indices, signal_e, dt, resolution, axis, pol)


def _inject_e_component(
    fields, comp, profiles, indices, j_source, sig, dt, res, sign=-1
):
    """Inject one E-field component via `J = n x H`."""
    profile = profiles.get(comp)
    idx = indices.get(comp)
    j_term = profiles.get(j_source)
    if profile is None or idx is None or j_term is None:
        return
    target = getattr(fields, comp)[idx]
    j_term = _match_shape(j_term, target.shape)
    if j_term is None:
        logger.debug("Shape mismatch injecting %s, skipping", comp)
        return
    eps = fields.permittivity[idx]
    setattr(
        fields,
        comp,
        getattr(fields, comp)
        .at[idx]
        .add(sign * j_term * sig * dt / (EPS_0 * eps * res)),
    )


def _inject_h_component(
    fields, comp, profiles, indices, m_source, sig, dt, res, sign=-1
):
    """Inject one H-field component via `M = -n x E`."""
    profile = profiles.get(comp)
    idx = indices.get(comp)
    m_term = profiles.get(m_source)
    if profile is None or idx is None or m_term is None:
        return
    target = getattr(fields, comp)[idx]
    m_term = _match_shape(m_term, target.shape)
    if m_term is None:
        logger.debug("Shape mismatch injecting %s, skipping", comp)
        return
    mu = getattr(fields, "permeability", None)
    mu_val = mu[idx] if mu is not None else 1.0
    setattr(
        fields,
        comp,
        getattr(fields, comp)
        .at[idx]
        .add(sign * m_term * sig * dt / (MU_0 * mu_val * res)),
    )


def _match_shape(profile, target_shape):
    """Match profile shape to target field shape, trimming or padding as needed."""
    if profile is None:
        return None
    profile = np.squeeze(profile)
    if profile.shape == target_shape:
        return profile
    if profile.ndim == len(target_shape):
        slices = tuple(
            slice(0, min(profile.shape[i], target_shape[i]))
            for i in range(profile.ndim)
        )
        trimmed = profile[slices]
        if trimmed.shape == target_shape:
            return trimmed
        result = np.zeros(target_shape, dtype=profile.dtype)
        result[tuple(slice(0, trimmed.shape[i]) for i in range(trimmed.ndim))] = trimmed
        return result
    return None
