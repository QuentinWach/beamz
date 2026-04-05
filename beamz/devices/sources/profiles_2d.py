"""2D profile construction helpers for mode-source injection."""

import numpy as np

from beamz.devices.sources.profiles_basis import (
    _normalize_2d_pair_by_power,
    _to_real_profile,
)
from beamz.devices.sources.profiles_common import _impedance_match_e_profile
from beamz.devices.sources.windows import _scipy_tukey


def _align_2d_impedance_pair(h_field, e_field, z_target):
    h_profile = np.squeeze(h_field)
    e_profile = np.squeeze(e_field)
    idx_max = np.argmax(np.abs(h_profile))
    phase_ref = np.angle(h_profile.flatten()[idx_max])
    h_profile = h_profile * np.exp(-1j * phase_ref)
    e_profile = e_profile * np.exp(-1j * phase_ref)
    return h_profile, _impedance_match_e_profile(e_profile, h_profile, z_target)


def _stagger_2d_pair(h_field, e_field):
    return 0.5 * (h_field[:-1] + h_field[1:]), 0.5 * (e_field[:-1] + e_field[1:])


def _crop_window_2d_pair(h_profile, e_profile, start: int, end: int):
    stop = min(end, len(h_profile), len(e_profile))
    width = max(0, stop - start)
    window = _scipy_tukey(width, alpha=0.3) if width > 2 else np.ones(max(1, width))
    h_cropped = h_profile[start:stop]
    e_cropped = e_profile[start:stop]
    if len(h_cropped) == len(window):
        h_cropped = h_cropped * window
        e_cropped = e_cropped * window
    return h_cropped, e_cropped


def _finalize_2d_launch_pair(
    h_profile,
    e_profile,
    *,
    sign_h,
    sign_e,
    signed_flux_sign,
    resolution,
):
    h_profile = sign_h * h_profile
    e_profile = sign_e * e_profile
    h_profile, e_profile = _normalize_2d_pair_by_power(
        h_profile,
        e_profile,
        signed_flux_sign=signed_flux_sign,
        dl=resolution,
    )
    h_profile = _to_real_profile(h_profile)
    e_profile = _to_real_profile(e_profile)
    return _normalize_2d_pair_by_power(
        h_profile,
        e_profile,
        signed_flux_sign=signed_flux_sign,
        dl=resolution,
    )
