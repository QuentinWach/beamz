import numpy as np

from beamz.const import EPS_0, MU_0
from beamz.devices.sources.inject import _inject_3d_e_fields, _inject_3d_h_fields


def get_signal_value(source, time, dt):
    """Interpolate the source waveform at an arbitrary time."""
    idx_float = float(time / dt)
    idx_low = int(np.floor(idx_float))
    idx_high = idx_low + 1
    frac = idx_float - idx_low

    if 0 <= idx_low < len(source.signal) - 1:
        return (1.0 - frac) * source.signal[idx_low] + frac * source.signal[idx_high]
    if idx_low == len(source.signal) - 1:
        return source.signal[idx_low]
    return 0.0


def ensure_initialized(source, fields, resolution, dt):
    needs_reinit = (
        (not source._initialized)
        or (source._grid_shape != fields.permittivity.shape)
        or (source._resolution is None)
        or (not np.isclose(source._resolution, resolution))
    )
    if (
        (not needs_reinit)
        and getattr(source, "_is_3d", False)
        and ((source._launch_dt is None) or (not np.isclose(source._launch_dt, dt)))
    ):
        needs_reinit = True
    if needs_reinit:
        source.initialize(fields.permittivity, resolution, dt=dt)


def inject_h(source, fields, t, dt, resolution):
    """Inject magnetic current into H-fields after the H update."""
    ensure_initialized(source, fields, resolution, dt)
    signal_value_h = get_signal_value(source, t + 0.5 * dt, dt)

    if source._Ex_profile is not None and source._is_3d:
        inject_3d_h(source, fields, signal_value_h, dt, resolution)
    else:
        inject_2d_h(source, fields, signal_value_h, dt, resolution)


def inject_e(source, fields, t, dt, resolution):
    """Inject electric current into E-fields after the E update."""
    ensure_initialized(source, fields, resolution, dt)
    signal_time_e = t + 0.5 * dt + source._dt_physical
    signal_value_e = get_signal_value(source, signal_time_e, dt)

    if source._Ex_profile is not None and source._is_3d:
        inject_3d_e(source, fields, signal_value_e, dt, resolution)
    else:
        inject_2d_e(source, fields, signal_value_e, dt, resolution)


def inject(source, fields, t, dt, current_step, resolution, design):
    """Backward-compatible combined injection entry point."""
    del current_step, design
    inject_h(source, fields, t, dt, resolution)
    inject_e(source, fields, t, dt, resolution)


def get_3d_profiles_and_indices(source):
    profiles = {
        "Ex": source._Ex_profile,
        "Ey": source._Ey_profile,
        "Ez": source._Ez_profile,
        "Hx": source._Hx_profile,
        "Hy": source._Hy_profile,
        "Hz": source._Hz_profile,
    }
    indices = {
        "Ex": source._Ex_indices,
        "Ey": source._Ey_indices,
        "Ez": source._Ez_indices,
        "Hx": source._Hx_indices,
        "Hy": source._Hy_indices,
        "Hz": source._Hz_indices,
    }
    return profiles, indices


def inject_3d_h(source, fields, signal_h, dt, resolution):
    """Inject H-field components for 3D Huygens source."""
    profiles, indices = get_3d_profiles_and_indices(source)
    _inject_3d_h_fields(
        fields, profiles, indices, signal_h, dt, resolution, source._axis, source.pol
    )


def inject_3d_e(source, fields, signal_e, dt, resolution):
    """Inject E-field components for 3D Huygens source."""
    profiles, indices = get_3d_profiles_and_indices(source)
    _inject_3d_e_fields(
        fields, profiles, indices, signal_e, dt, resolution, source._axis, source.pol
    )


def inject_2d_h(source, fields, signal_h, dt, resolution):
    """Inject magnetic current into H-fields for 2D."""
    if source.pol == "tm":
        if source._h_indices is not None and source._my_profile is not None:
            mu_val = getattr(fields, "permeability", None)
            mu_at_source = mu_val[source._h_indices] if mu_val is not None else 1.0
            my_term = source._my_profile * signal_h / resolution
            h_injection = -my_term * dt / (MU_0 * mu_at_source)

            if source._h_component == "Hx":
                fields.Hx = fields.Hx.at[source._h_indices].add(h_injection)
            else:
                fields.Hy = fields.Hy.at[source._h_indices].add(h_injection)
    else:
        if source._hz_indices is not None and source._mz_profile is not None:
            mu_val = getattr(fields, "permeability", None)
            mu_at_source = mu_val[source._hz_indices] if mu_val is not None else 1.0
            mz_term = source._mz_profile * signal_h / resolution
            hz_injection = +mz_term * dt / (MU_0 * mu_at_source)
            fields.Hz = fields.Hz.at[source._hz_indices].add(hz_injection)


def inject_2d_e(source, fields, signal_e, dt, resolution):
    """Inject electric current into E-fields for 2D."""
    if source.pol == "tm":
        if source._ez_indices is not None and source._jz_profile is not None:
            eps_at_source = fields.permittivity[source._ez_indices]
            jz_term = source._jz_profile * signal_e / resolution
            ez_injection = +jz_term * dt / (EPS_0 * eps_at_source)
            fields.Ez = fields.Ez.at[source._ez_indices].add(ez_injection)
    else:
        if source._e_indices is not None:
            j_profile = (
                source._jx_profile
                if source._e_component == "Ex"
                else source._jy_profile
            )
            if j_profile is not None:
                eps_at_source = fields.permittivity[source._e_indices]
                j_term = j_profile * signal_e / resolution
                e_injection = -j_term * dt / (EPS_0 * eps_at_source)

                if source._e_component == "Ex":
                    fields.Ex = fields.Ex.at[source._e_indices].add(e_injection)
                else:
                    fields.Ey = fields.Ey.at[source._e_indices].add(e_injection)
