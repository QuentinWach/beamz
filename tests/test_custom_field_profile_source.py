from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from beamz.devices.sources._custom_profile import _CustomFieldProfileSource
from beamz.devices.sources._profiles import FieldProfile3D
from beamz.devices.sources._time import _real_phasor_sample
from beamz.devices.sources.compiler import (
    apply_compiled_source_specs,
    compile_source_specs,
)


def _empty_3d_fields(shape=(7, 7, 7)):
    nz, ny, nx = (int(v) for v in shape)
    ex = np.zeros((nz, ny, nx - 1), dtype=np.float64)
    ey = np.zeros((nz, ny - 1, nx), dtype=np.float64)
    ez = np.zeros((nz - 1, ny, nx), dtype=np.float64)
    hx = np.zeros((nz - 1, ny - 1, nx), dtype=np.float64)
    hy = np.zeros((nz - 1, ny, nx - 1), dtype=np.float64)
    hz = np.zeros((nz, ny - 1, nx - 1), dtype=np.float64)
    return SimpleNamespace(
        boundaries=None,
        permittivity=np.ones((nz, ny, nx), dtype=np.float64),
        permeability=np.ones((nz, ny, nx), dtype=np.float64),
        conductivity=np.zeros((nz, ny, nx), dtype=np.float64),
        total_conductivity=np.zeros((nz, ny, nx), dtype=np.float64),
        Ex=ex,
        Ey=ey,
        Ez=ez,
        Hx=hx,
        Hy=hy,
        Hz=hz,
        eps_x=np.ones_like(ex),
        eps_y=np.ones_like(ey),
        eps_z=np.ones_like(ez),
        sig_x=np.zeros_like(ex),
        sig_y=np.zeros_like(ey),
        sig_z=np.zeros_like(ez),
        region_x=(slice(None), slice(None), slice(None)),
        region_y=(slice(None), slice(None), slice(None)),
        region_z=(slice(None), slice(None), slice(None)),
        sigma_m_hx=np.zeros_like(hx),
        sigma_m_hy=np.zeros_like(hy),
        sigma_m_hz=np.zeros_like(hz),
    )


def _planeish_x_profile():
    yy, zz = np.meshgrid(
        np.linspace(-1.0, 1.0, 3),
        np.linspace(-1.0, 1.0, 3),
        indexing="xy",
    )
    envelope = np.exp(-(yy**2 + zz**2))
    return FieldProfile3D(
        components={
            "Ey": envelope * (1.0 + 0.25j),
            "Hz": envelope * (0.18 - 0.12j),
        },
        indices={
            "Ey": (slice(2, 5), slice(2, 5), 3),
            "Hz": (slice(2, 5), slice(2, 5), 3),
        },
        axis="x",
        direction_sign=1.0,
        omega=0.8,
        k_axis=0.2,
        phase_ref_coord=4.0,
        phase_plane_coord=3.5,
    )


def _applied_component(specs, fields, component, *, step=0):
    component_specs = tuple(spec for spec in specs if spec.component == component)
    return np.asarray(
        apply_compiled_source_specs(
            jnp.zeros_like(jnp.asarray(getattr(fields, component))),
            step,
            component_specs,
        )
    )


def test_custom_field_profile_source_compiles_planar_tfsf_specs():
    fields = _empty_3d_fields()
    source = _CustomFieldProfileSource(
        profile=_planeish_x_profile(),
        signal=np.asarray([1.0, 0.25, -0.5], dtype=np.float64),
        signal_quadrature=np.asarray([0.5, -0.25, 0.0], dtype=np.float64),
    )
    dt = 0.1
    resolution = 1.0

    specs = compile_source_specs(
        [source],
        fields,
        dt=dt,
        resolution=resolution,
        num_steps=3,
        t0=0.0,
        total_steps=3,
    )

    residuals = source._compute_discrete_3d_phasor_residuals(
        fields,
        dt=dt,
        resolution=resolution,
    )
    expected = {
        component: np.zeros_like(getattr(fields, component), dtype=np.float64)
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    for residual in residuals:
        expected[residual.component][residual.index] += _real_phasor_sample(
            residual.residual,
            1.0,
            0.5,
        )

    assert specs
    for component, values in expected.items():
        np.testing.assert_allclose(
            _applied_component(specs, fields, component),
            values,
            rtol=1e-6,
            atol=1e-8,
        )


def test_custom_field_profile_source_power_scales_field_amplitude():
    fields = _empty_3d_fields()
    unit = _CustomFieldProfileSource(
        profile=_planeish_x_profile(),
        signal=np.ones(3, dtype=np.float64),
        power=1.0,
    )
    stronger = _CustomFieldProfileSource(
        profile=_planeish_x_profile(),
        signal=np.ones(3, dtype=np.float64),
        power=4.0,
    )

    unit_residuals = unit._compute_discrete_3d_phasor_residuals(
        fields,
        dt=0.1,
        resolution=1.0,
    )
    strong_residuals = stronger._compute_discrete_3d_phasor_residuals(
        fields,
        dt=0.1,
        resolution=1.0,
    )

    assert len(unit_residuals) == len(strong_residuals)
    for unit_residual, strong_residual in zip(
        unit_residuals,
        strong_residuals,
        strict=True,
    ):
        assert unit_residual.component == strong_residual.component
        assert unit_residual.timing == strong_residual.timing
        assert unit_residual.index == strong_residual.index
        np.testing.assert_allclose(
            strong_residual.residual,
            2.0 * unit_residual.residual,
            rtol=1e-12,
            atol=1e-12,
        )
