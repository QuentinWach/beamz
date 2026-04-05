import numpy as np

from beamz.const import EPS_0, MU_0
from beamz.devices.sources.inject import _inject_3d_e_fields, _inject_3d_h_fields


def get_signal_value(source, time, dt):
    """Interpolate the source waveform at an arbitrary time."""
    signal = source.spec.signal
    idx_float = float(time / dt)
    idx_low = int(np.floor(idx_float))
    idx_high = idx_low + 1
    frac = idx_float - idx_low

    if 0 <= idx_low < len(signal) - 1:
        return (1.0 - frac) * signal[idx_low] + frac * signal[idx_high]
    if idx_low == len(signal) - 1:
        return signal[idx_low]
    return 0.0


def ensure_initialized(source, fields, resolution, dt):
    state = source.state
    needs_reinit = (
        (not state.initialized)
        or (state.grid_shape != fields.permittivity.shape)
        or (state.resolution is None)
        or (not np.isclose(state.resolution, resolution))
    )
    if (
        (not needs_reinit)
        and state.is_3d
        and ((state.launch_dt is None) or (not np.isclose(state.launch_dt, dt)))
    ):
        needs_reinit = True
    if needs_reinit:
        source.initialize(fields.permittivity, resolution, dt=dt)


def inject_h(source, fields, t, dt, resolution):
    """Inject magnetic current into H-fields after the H update."""
    ensure_initialized(source, fields, resolution, dt)
    signal_value_h = get_signal_value(source, t + 0.5 * dt, dt)

    if source.state.Ex_profile is not None and source.state.is_3d:
        inject_3d_h(source, fields, signal_value_h, dt, resolution)
    else:
        inject_2d_h(source, fields, signal_value_h, dt, resolution)


def inject_e(source, fields, t, dt, resolution):
    """Inject electric current into E-fields after the E update."""
    ensure_initialized(source, fields, resolution, dt)
    signal_time_e = t + 0.5 * dt + source.state.dt_physical
    signal_value_e = get_signal_value(source, signal_time_e, dt)

    if source.state.Ex_profile is not None and source.state.is_3d:
        inject_3d_e(source, fields, signal_value_e, dt, resolution)
    else:
        inject_2d_e(source, fields, signal_value_e, dt, resolution)


def inject(source, fields, t, dt, current_step, resolution, design):
    """Backward-compatible combined injection entry point."""
    del current_step, design
    inject_h(source, fields, t, dt, resolution)
    inject_e(source, fields, t, dt, resolution)


def get_3d_profiles_and_indices(source):
    state = source.state
    profiles = {
        "Ex": state.Ex_profile,
        "Ey": state.Ey_profile,
        "Ez": state.Ez_profile,
        "Hx": state.Hx_profile,
        "Hy": state.Hy_profile,
        "Hz": state.Hz_profile,
    }
    indices = {
        "Ex": state.Ex_indices,
        "Ey": state.Ey_indices,
        "Ez": state.Ez_indices,
        "Hx": state.Hx_indices,
        "Hy": state.Hy_indices,
        "Hz": state.Hz_indices,
    }
    return profiles, indices


def inject_3d_h(source, fields, signal_h, dt, resolution):
    """Inject H-field components for 3D Huygens source."""
    profiles, indices = get_3d_profiles_and_indices(source)
    _inject_3d_h_fields(
        fields, profiles, indices, signal_h, dt, resolution, source.state.axis, source.spec.pol
    )


def inject_3d_e(source, fields, signal_e, dt, resolution):
    """Inject E-field components for 3D Huygens source."""
    profiles, indices = get_3d_profiles_and_indices(source)
    _inject_3d_e_fields(
        fields, profiles, indices, signal_e, dt, resolution, source.state.axis, source.spec.pol
    )


def inject_2d_h(source, fields, signal_h, dt, resolution):
    """Inject magnetic current into H-fields for 2D."""
    state = source.state
    if source.spec.pol == "tm":
        if state.h_indices is not None and state.my_profile is not None:
            mu_val = getattr(fields, "permeability", None)
            mu_at_source = mu_val[state.h_indices] if mu_val is not None else 1.0
            my_term = state.my_profile * signal_h / resolution
            h_injection = -my_term * dt / (MU_0 * mu_at_source)

            if state.h_component == "Hx":
                fields.Hx = fields.Hx.at[state.h_indices].add(h_injection)
            else:
                fields.Hy = fields.Hy.at[state.h_indices].add(h_injection)
    else:
        if state.hz_indices is not None and state.mz_profile is not None:
            mu_val = getattr(fields, "permeability", None)
            mu_at_source = mu_val[state.hz_indices] if mu_val is not None else 1.0
            mz_term = state.mz_profile * signal_h / resolution
            hz_injection = +mz_term * dt / (MU_0 * mu_at_source)
            fields.Hz = fields.Hz.at[state.hz_indices].add(hz_injection)


def inject_2d_e(source, fields, signal_e, dt, resolution):
    """Inject electric current into E-fields for 2D."""
    state = source.state
    if source.spec.pol == "tm":
        if state.ez_indices is not None and state.jz_profile is not None:
            eps_at_source = fields.permittivity[state.ez_indices]
            jz_term = state.jz_profile * signal_e / resolution
            ez_injection = +jz_term * dt / (EPS_0 * eps_at_source)
            fields.Ez = fields.Ez.at[state.ez_indices].add(ez_injection)
    else:
        if state.e_indices is not None:
            j_profile = state.jx_profile if state.e_component == "Ex" else state.jy_profile
            if j_profile is not None:
                eps_at_source = fields.permittivity[state.e_indices]
                j_term = j_profile * signal_e / resolution
                e_injection = -j_term * dt / (EPS_0 * eps_at_source)

                if state.e_component == "Ex":
                    fields.Ex = fields.Ex.at[state.e_indices].add(e_injection)
                else:
                    fields.Ey = fields.Ey.at[state.e_indices].add(e_injection)
