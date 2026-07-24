from __future__ import annotations

import numpy as np
import pytest

from beamz import FieldRecorder
from tests.validation.invariants.test_engine_equivalence import (
    FIELD_COMPONENTS,
    _assert_fields_close,
    _make_2d_sim,
    _run_reference,
    _seed_field_payload,
    _state_with_payload,
)

pytestmark = [pytest.mark.compiled, pytest.mark.component]


def _linear_combination(
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
    *,
    first_scale: float,
    second_scale: float,
) -> dict[str, np.ndarray]:
    return {
        name: (
            first_scale * np.asarray(first[name])
            + second_scale * np.asarray(second[name])
        ).astype(np.float32)
        for name in FIELD_COMPONENTS
    }


@pytest.mark.parametrize("plane_2d", ["xy", "xz", "yz"])
def test_source_free_execution_is_linear_across_all_2d_planes(plane_2d):
    steps = 4
    sim, _ = _make_2d_sim(plane_2d=plane_2d, steps=steps)
    first = _seed_field_payload(sim, seed=101)
    second = _seed_field_payload(sim, seed=202)
    combined = _linear_combination(
        first,
        second,
        first_scale=1.75,
        second_scale=-0.4,
    )

    first_result = sim.advance(
        num_steps=steps,
        state=_state_with_payload(sim, first),
        progress=False,
    ).state
    second_result = sim.advance(
        num_steps=steps,
        state=_state_with_payload(sim, second),
        progress=False,
    ).state
    combined_result = sim.advance(
        num_steps=steps,
        state=_state_with_payload(sim, combined),
        progress=False,
    ).state

    for component in FIELD_COMPONENTS:
        field_name = component.lower()
        expected = 1.75 * np.asarray(
            getattr(first_result, field_name)
        ) - 0.4 * np.asarray(getattr(second_result, field_name))
        np.testing.assert_allclose(
            np.asarray(getattr(combined_result, field_name)),
            expected,
            rtol=3e-5,
            atol=3e-6,
            err_msg=f"linearity failed for {plane_2d} {component}",
        )


@pytest.mark.parametrize("plane_2d", ["xy", "xz", "yz"])
def test_full_chunked_and_repeated_step_execution_are_equivalent(plane_2d):
    steps = 6
    sim, _ = _make_2d_sim(plane_2d=plane_2d, steps=steps)
    payload = _seed_field_payload(sim, seed=303)

    full = sim.advance(
        num_steps=steps,
        state=_state_with_payload(sim, payload),
        progress=False,
    ).state

    chunked = _state_with_payload(sim, payload)
    for _ in range(3):
        chunked = sim.advance(
            num_steps=2,
            state=chunked,
            progress=False,
        ).state

    stepped = _run_reference(sim, steps, _state_with_payload(sim, payload))

    _assert_fields_close(full, chunked, atol=2e-6, rtol=2e-6)
    _assert_fields_close(full, stepped, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize(
    ("plane_2d", "recorded_component"),
    [("xy", "Ez"), ("xz", "Ey"), ("yz", "Ex")],
)
def test_adding_passive_field_recorder_does_not_change_fields(
    plane_2d, recorded_component
):
    steps = 4
    plain, _ = _make_2d_sim(plane_2d=plane_2d, steps=steps)
    monitored, _ = _make_2d_sim(
        plane_2d=plane_2d,
        steps=steps,
        monitors=[
            FieldRecorder(
                components=(recorded_component,),
                interval=1,
                name="passive",
            )
        ],
    )
    payload = _seed_field_payload(plain, seed=404)

    plain_state = plain.advance(
        num_steps=steps,
        state=_state_with_payload(plain, payload),
        progress=False,
    ).state
    monitored_state = monitored.advance(
        num_steps=steps,
        state=_state_with_payload(monitored, payload),
        progress=False,
    ).state

    _assert_fields_close(plain_state, monitored_state, atol=1e-7, rtol=1e-7)


def test_integer_cell_translation_commutes_with_interior_update():
    steps = 2
    sim, _ = _make_2d_sim(plane_2d="xy", steps=steps)
    rng = np.random.default_rng(505)
    payload = {}
    for component in FIELD_COMPONENTS:
        shape = np.asarray(getattr(sim.compile().grid, component)).shape
        values = np.zeros(shape, dtype=np.float32)
        if len(shape) == 2 and min(shape) > 8:
            values[3:-3, 4:-4] = rng.normal(
                scale=0.01,
                size=values[3:-3, 4:-4].shape,
            )
        payload[component] = values
    shifted = {
        component: np.pad(values[..., :-1], ((0, 0), (1, 0)))
        if values.ndim == 2 and values.shape[1] > 1
        else values.copy()
        for component, values in payload.items()
    }

    original_result = sim.advance(
        num_steps=steps,
        state=_state_with_payload(sim, payload),
        progress=False,
    ).state
    shifted_result = sim.advance(
        num_steps=steps,
        state=_state_with_payload(sim, shifted),
        progress=False,
    ).state

    for component in FIELD_COMPONENTS:
        field_name = component.lower()
        original = np.asarray(getattr(original_result, field_name))
        actual = np.asarray(getattr(shifted_result, field_name))
        if original.ndim != 2 or original.shape[1] <= 8:
            np.testing.assert_array_equal(actual, original)
            continue
        expected = np.pad(original[..., :-1], ((0, 0), (1, 0)))
        np.testing.assert_allclose(
            actual[3:-3, 3:-3],
            expected[3:-3, 3:-3],
            rtol=2e-6,
            atol=2e-6,
            err_msg=f"translation invariance failed for {component}",
        )
