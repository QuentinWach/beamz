"""Compatibility re-exports for mode-source profile helpers."""

from beamz.devices.sources.profiles_2d import (
    _align_2d_impedance_pair,
    _crop_window_2d_pair,
    _finalize_2d_launch_pair,
    _stagger_2d_pair,
)
from beamz.devices.sources.profiles_3d import _build_3d_profiles
from beamz.devices.sources.profiles_common import (
    _axis_index_from_component_indices,
    _component_axis_coord,
    _dominant_3d_pair,
    _impedance_match_3d_tangential_pairs,
    _impedance_match_e_profile,
    _numeric_impedance_axis,
    _numeric_phase_delay,
    _parse_direction,
    _remap_3d_solver_components,
    _select_3d_impedance_index,
    _select_3d_phase_ref,
    _select_core_confined_mode_index,
    _solve_numeric_k_axis,
)
from beamz.devices.sources.profiles_basis import (
    _backward_3d_mode_from_forward,
    _make_3d_mode_basis_profiles,
    _modal_overlap_3d_profiles,
    _modal_power_2d,
    _modal_power_3d_from_profiles,
    _normalize_2d_pair_by_power,
    _normalize_3d_profiles_by_flux,
    _project_3d_profiles_to_real,
    _to_real_profile,
)
