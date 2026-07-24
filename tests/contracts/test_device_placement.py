import numpy as np

from beamz import FluxMonitor, um


def test_monitor_plane_3d_snap_uses_same_centered_plane_convention():
    monitor = FluxMonitor(
        center=(1.0 * um, 0.7 * um, 0.5 * um),
        size=(0.0, 0.8 * um, 0.6 * um),
        freqs=(2e14,),
        name="m3d",
    )

    snapped = monitor.get_snapped_region(
        dx=0.2 * um, dy=0.2 * um, dz=0.2 * um, field_shape=(5, 8, 10)
    )
    z_idx, y_idx, x_idx = monitor.get_grid_slice_3d(
        0.2 * um, 0.2 * um, 0.2 * um, (5, 8, 10)
    )

    assert snapped is not None
    assert snapped.normal_axis == "x"
    assert snapped.plane_index == 4
    assert snapped.plane_coord == 0.9 * um
    assert (z_idx, y_idx, x_idx) == (slice(1, 4), slice(1, 6), 4)
    np.testing.assert_allclose(snapped.center, (0.9 * um, 0.7 * um, 0.5 * um))
    np.testing.assert_allclose(monitor.position, (1.0 * um, 0.7 * um, 0.5 * um))


def test_shifted_monitor_does_not_reuse_snapped_runtime_cache():
    monitor = FluxMonitor(
        center=(1.0 * um, 0.7 * um, 0.5 * um),
        size=(0.0, 0.8 * um, 0.6 * um),
        freqs=(2e14,),
        name="m3d",
    )
    original = monitor.get_snapped_region(
        dx=0.2 * um, dy=0.2 * um, dz=0.2 * um, field_shape=(5, 8, 10)
    )

    shifted = monitor.shifted((0.2 * um, 0.0, 0.0))
    snapped = shifted.get_snapped_region(
        dx=0.2 * um, dy=0.2 * um, dz=0.2 * um, field_shape=(5, 8, 10)
    )

    assert original is not None
    assert snapped is not None
    assert snapped.plane_index == 6
    np.testing.assert_allclose(snapped.center, (1.3 * um, 0.7 * um, 0.5 * um))
