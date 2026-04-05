from __future__ import annotations

from dataclasses import dataclass

from beamz.devices.sources.spec import GaussianSourceSpec, ModeSourceSpec


@dataclass(slots=True)
class GaussianSourceState:
    spatial_profile_ez: object = None
    grid_indices: object = None


@dataclass(slots=True)
class ModeSourceState:
    Ex_profile: object = None
    Ey_profile: object = None
    Ez_profile: object = None
    Hx_profile: object = None
    Hy_profile: object = None
    Hz_profile: object = None
    Ex_indices: object = None
    Ey_indices: object = None
    Ez_indices: object = None
    Hx_indices: object = None
    Hy_indices: object = None
    Hz_indices: object = None
    jz_profile: object = None
    my_profile: object = None
    mz_profile: object = None
    jy_profile: object = None
    jx_profile: object = None
    ez_indices: object = None
    h_indices: object = None
    hz_indices: object = None
    e_indices: object = None
    h_component: object = None
    e_component: object = None
    neff: object = None
    impedance_neff: object = None
    dt_physical: float = 0.0
    launch_dt: object = None
    initialized: bool = False
    resolution: object = None
    is_3d: bool = False
    grid_shape: object = None
    eps_profile_2d: object = None
    axis: object = None
    transverse_start: int | None = None
    transverse_end: int | None = None
    x_start: int | None = None
    x_end: int | None = None
    y_start: int | None = None
    y_end: int | None = None
    z_start: int | None = None
    z_end: int | None = None


def create_source_state(spec):
    if isinstance(spec, GaussianSourceSpec):
        return GaussianSourceState()
    if isinstance(spec, ModeSourceSpec):
        return ModeSourceState()
    raise TypeError(f"unsupported source spec type: {type(spec).__name__}")


def source_state_for(spec, *, source=None, state=None):
    expected_type = type(create_source_state(spec))
    if isinstance(state, expected_type):
        return state
    source_state = getattr(source, "state", None)
    if isinstance(source_state, expected_type):
        return source_state
    return create_source_state(spec)
