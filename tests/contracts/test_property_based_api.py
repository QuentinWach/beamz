"""Generative contracts over BeamZ's immutable public configuration surface."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import beamz as bz
from beamz.lattice import linear_interpolation_plan

FINITE_COORDINATE = st.floats(
    min_value=-1e3,
    max_value=1e3,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
POSITIVE_EXTENT = st.floats(
    min_value=1e-6,
    max_value=1e3,
    allow_nan=False,
    allow_infinity=False,
)


@settings(max_examples=60, deadline=None)
@given(
    axis=st.integers(min_value=0, max_value=2),
    center=st.tuples(FINITE_COORDINATE, FINITE_COORDINATE, FINITE_COORDINATE),
    extents=st.tuples(POSITIVE_EXTENT, POSITIVE_EXTENT),
    offset=st.tuples(FINITE_COORDINATE, FINITE_COORDINATE, FINITE_COORDINATE),
    direction=st.sampled_from(("+", "-")),
)
def test_port_translation_is_invertible_and_preserves_device_contract(
    axis, center, extents, offset, direction
):
    size_values = iter(extents)
    size = tuple(0.0 if index == axis else next(size_values) for index in range(3))
    port = bz.Port(center=center, size=size, name="p", direction=direction)

    shifted = port.shifted(offset)
    restored = shifted.shifted(tuple(-value for value in offset))
    np.testing.assert_allclose(restored.center, port.center, rtol=1e-6, atol=1e-4)
    assert shifted.size == port.size
    assert shifted.direction == port.direction

    monitor = shifted.to_monitor([1e14, 2e14])
    assert monitor.plane_normal == ("x", "y", "z")[axis]
    assert monitor.mode_spec == shifted.mode_spec
    assert monitor.name == shifted.monitor_name


@settings(max_examples=60, deadline=None)
@given(
    values=st.lists(
        st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        min_size=1,
        max_size=24,
    ),
    dt=st.floats(
        min_value=1e-12,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_sampled_signal_detaches_every_generated_input(values, dt):
    mutable = np.asarray(values, dtype=float)
    signal = bz.SampledSignal(mutable, dt=dt)
    owned = signal.values.copy()

    mutable[:] = 12345.0

    np.testing.assert_array_equal(signal.values, owned)
    assert not signal.values.flags.writeable
    assert not signal.quadrature.flags.writeable
    sampled, quadrature = signal.sample(
        [signal.start_time, signal.start_time + (len(values) - 1) * dt]
    )
    assert sampled.shape == quadrature.shape == (2,)
    assert np.isfinite(sampled).all()
    assert np.isfinite(quadrature).all()


@st.composite
def _interpolation_cases(draw):
    source = sorted(
        draw(
            st.lists(
                st.integers(min_value=-100, max_value=100),
                min_size=2,
                max_size=12,
                unique=True,
            )
        )
    )
    targets = draw(
        st.lists(
            st.floats(
                min_value=float(source[0]),
                max_value=float(source[-1]),
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
            min_size=1,
            max_size=20,
        )
    )
    return np.asarray(source, dtype=float), np.asarray(targets, dtype=float)


@settings(max_examples=80, deadline=None)
@given(case=_interpolation_cases())
def test_yee_interpolation_reproduces_every_affine_field(case):
    source, target = case
    low, high, weight_low, weight_high = linear_interpolation_plan(source, target)
    np.testing.assert_allclose(weight_low + weight_high, 1.0, atol=1e-7)

    slope, intercept = 2.75, -1.25
    source_values = slope * source + intercept
    interpolated = weight_low * source_values[low] + weight_high * source_values[high]
    np.testing.assert_allclose(
        interpolated,
        slope * target + intercept,
        rtol=2e-6,
        atol=2e-5,
    )
