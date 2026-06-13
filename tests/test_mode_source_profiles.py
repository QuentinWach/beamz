import numpy as np

from beamz.devices.sources.mode import _build_3d_profiles


def _synthetic_x_mode_fields(shape=(8, 10)):
    z, y = np.indices(shape, dtype=float)
    envelope = np.exp(-((z - 3.5) ** 2 + (y - 4.5) ** 2) / 18.0)
    phase = np.exp(0.08j * z - 0.05j * y)
    ey = envelope * phase
    hz = 0.7 * envelope * np.conjugate(phase)
    ez = 0.25 * envelope * np.exp(0.04j * z)
    hy = -0.18 * envelope * np.exp(-0.03j * y)
    ex = 0.05 * envelope * np.exp(0.02j * z)
    hx = 0.03 * envelope * np.exp(-0.02j * y)
    return ex, ey, ez, hx, hy, hz


def test_3d_mode_source_profiles_preserve_solver_field_balance():
    fields = _synthetic_x_mode_fields()

    base_profiles, _, _ = _build_3d_profiles(
        *fields,
        axis="x",
        direction="+x",
        center=(0.55, 0.50, 0.40),
        width=0.80,
        height=0.60,
        center_idx=5,
        offset_idx=5,
        grid_shape=(8, 10, 12),
        resolution=0.1,
        impedance_neff=1.1,
        omega=2.0 * np.pi * 2.0e14,
        dt=None,
    )
    shifted_profiles, _, _ = _build_3d_profiles(
        *fields,
        axis="x",
        direction="+x",
        center=(0.55, 0.50, 0.40),
        width=0.80,
        height=0.60,
        center_idx=5,
        offset_idx=5,
        grid_shape=(8, 10, 12),
        resolution=0.1,
        impedance_neff=3.4,
        omega=2.0 * np.pi * 2.0e14,
        dt=2.0e-17,
    )

    for name, profile in base_profiles.items():
        np.testing.assert_allclose(
            shifted_profiles[name],
            profile,
            rtol=0.0,
            atol=0.0,
            err_msg=f"{name} profile changed after altering impedance-only inputs.",
        )
