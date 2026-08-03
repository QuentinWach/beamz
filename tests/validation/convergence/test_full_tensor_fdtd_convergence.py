"""Convergence of the full-tensor TE FDTD constitutive update."""

from __future__ import annotations

import math
from functools import cache

import jax.numpy as jnp
import numpy as np
import pytest

import beamz as bz
from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.design import raster
from beamz.lattice import advance_h_field
from beamz.simulation.kernels import (
    advance_e_centered_tensor,
    te_xy_curl_e_to_h_2d,
    te_xy_curl_h_to_e_2d,
)


def _centroid(field, spacing):
    energy = np.sum(np.asarray(field) ** 2, axis=0)
    coordinates = (np.arange(energy.size) + 0.5) * spacing
    return float(np.sum(coordinates * energy) / np.sum(energy))


def _anisotropic_packet_displacement(cells: int) -> float:
    length = 192e-6
    spacing = length / cells
    epsilon = np.asarray(((2.0, 0.8), (0.8, 3.0)))
    inverse = np.linalg.inv(epsilon)
    effective_epsilon = np.linalg.det(epsilon) / epsilon[0, 0]
    impedance = np.sqrt(MU_0 / (EPS_0 * effective_epsilon))
    x_e = np.arange(cells + 1) * spacing
    x_h = (np.arange(cells) + 0.5) * spacing
    center, width, wavelength = 60e-6, 12e-6, 24e-6

    def packet(x):
        return np.exp(-(((x - center) / width) ** 2)) * np.cos(
            2.0 * np.pi * (x - center) / wavelength
        )

    ey = jnp.asarray(np.tile(packet(x_e), (4, 1)), dtype=jnp.float32)
    ex = jnp.asarray(
        np.tile((-epsilon[0, 1] / epsilon[0, 0]) * packet(x_h), (5, 1)),
        dtype=jnp.float32,
    )
    hz = jnp.asarray(np.tile(packet(x_h) / impedance, (4, 1)), dtype=jnp.float32)
    diagonals = (
        jnp.full_like(ex, inverse[0, 0]),
        jnp.full_like(ey, inverse[1, 1]),
    )
    offdiagonal = jnp.zeros((5, cells + 1, 3), dtype=jnp.float32)
    offdiagonal = offdiagonal.at[..., 0].set(inverse[0, 1])
    start = _centroid(hz, spacing)
    duration = 30e-15
    steps = round(duration / (0.2 * spacing / LIGHT_SPEED))
    dt = duration / steps
    for _ in range(steps):
        curl_hz = te_xy_curl_e_to_h_2d(ex, ey, spacing, hz.shape)
        hz = advance_h_field(hz, curl_hz, 0.0, dt)
        curls = te_xy_curl_h_to_e_2d(hz, spacing, ex.shape, ey.shape, frozenset())
        ex, ey = advance_e_centered_tensor(
            (ex, ey), curls, diagonals, offdiagonal, ("Ex", "Ey"), dt
        )
    return _centroid(hz, spacing) - start


def test_full_tensor_te_update_is_second_order_under_refinement():
    coarse, medium, fine = (
        _anisotropic_packet_displacement(cells) for cells in (96, 192, 384)
    )
    observed_order = math.log2(abs(coarse - medium) / abs(medium - fine))

    assert 1.5 < observed_order < 2.2


def _sloped_interface_simulation(cells: int, time: np.ndarray):
    length = 4.0 * bz.um
    spacing = length / cells
    scene = raster.Scene(
        (raster.Material(), raster.Material(4.0)),
        (
            raster.Object(
                raster.ExtrudedPolygon(
                    raster.Polygon(
                        (
                            (-length, 2.0 * length),
                            (2.0 * length, -length),
                            (2.0 * length, 2.0 * length),
                        )
                    ),
                    0.0,
                    1.0,
                ),
                1,
            ),
        ),
    )
    simulation = bz.Simulation(
        scene=scene,
        raster_grid=raster.Grid.uniform(
            (0.0, 0.0, 0.0), (length, length, 1.0), (cells, cells, 1)
        ),
        polarization="te",
        time=time,
        sources=(),
        normalize_source=None,
        raster_options=raster.RasterOptions(smoothing="farjadpour_full"),
    )
    state = simulation.initial_state()
    coordinates = (np.arange(cells) + 0.5) * spacing
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    initial_hz = np.exp(
        -((x - 1.7 * bz.um) ** 2 + (y - 1.7 * bz.um) ** 2) / (2.0 * (0.18 * bz.um) ** 2)
    )
    state = state._replace(hz=jnp.asarray(initial_hz, dtype=state.hz.dtype))
    return simulation, state


def _yee_coordinates(component: str, shape, spacing: float):
    y_offset, x_offset = {
        "Ex": (0.0, 0.5),
        "Ey": (0.5, 0.0),
        "Hz": (0.5, 0.5),
    }[component]
    y = (np.arange(shape[0]) + y_offset) * spacing
    x = (np.arange(shape[1]) + x_offset) * spacing
    return np.meshgrid(y, x, indexing="ij")


def _exact_normal_pulse(
    component: str,
    y,
    x,
    time: float,
    *,
    normal,
    interface_coordinate: float,
):
    """Return the exact normally incident TE pulse on one Yee component."""

    epsilon_left, epsilon_right = 1.0, 4.0
    index_left, index_right = np.sqrt((epsilon_left, epsilon_right))
    speed_left = LIGHT_SPEED / index_left
    impedance_left = np.sqrt(MU_0 / (EPS_0 * epsilon_left))
    impedance_right = np.sqrt(MU_0 / (EPS_0 * epsilon_right))
    reflection = (index_left - index_right) / (index_left + index_right)
    transmission = 2.0 * index_left / (index_left + index_right)
    pulse_center = interface_coordinate - 0.9 * bz.um
    pulse_width = 0.3 * bz.um
    wavelength = 1.2 * bz.um

    def profile(coordinate):
        relative = coordinate - pulse_center
        return np.exp(-0.5 * (relative / pulse_width) ** 2) * np.cos(
            2.0 * np.pi * relative / wavelength
        )

    coordinate = normal[0] * x + normal[1] * y
    incident = profile(coordinate - speed_left * time)
    reflected = profile(2.0 * interface_coordinate - coordinate - speed_left * time)
    transmitted = profile(
        interface_coordinate
        + (index_right / index_left) * (coordinate - interface_coordinate)
        - speed_left * time
    )
    left = coordinate <= interface_coordinate
    electric = np.where(
        left,
        incident + reflection * reflected,
        transmission * transmitted,
    )
    magnetic = np.where(
        left,
        (incident - reflection * reflected) / impedance_left,
        transmission * transmitted / impedance_right,
    )
    tangent = (-normal[1], normal[0])
    return {"Ex": tangent[0] * electric, "Ey": tangent[1] * electric, "Hz": magnetic}[
        component
    ]


@cache
def _analytical_interface_error(
    cells: int,
    *,
    smoothing="farjadpour_full",
    interface_offset=0.07 * bz.um,
    angle_degrees=27.0,
    anisotropic=True,
) -> float:
    """Run the public raster-to-FDTD path and return its global exact-solution error."""

    length = 8.0 * bz.um
    spacing = length / cells
    angle = np.deg2rad(angle_degrees)
    normal = np.asarray((np.cos(angle), np.sin(angle)))
    tangent = np.asarray((-normal[1], normal[0]))
    # The right material's principal axes follow the interface. Normal incidence with
    # tangential E therefore sees epsilon=4 exactly, while epsilon=6 still exercises
    # intrinsic Cartesian off-diagonal coupling after rotation.
    epsilon_right_xy = (
        6.0 * np.outer(normal, normal) + 4.0 * np.outer(tangent, tangent)
        if anisotropic
        else 4.0 * np.eye(2)
    )
    epsilon_right = (
        (epsilon_right_xy[0, 0], epsilon_right_xy[0, 1], 0.0),
        (epsilon_right_xy[1, 0], epsilon_right_xy[1, 1], 0.0),
        (0.0, 0.0, 3.0),
    )
    interface_point = (
        np.asarray((0.5 * length, 0.5 * length)) + interface_offset * normal
    )
    interface_coordinate = float(normal @ interface_point)
    extent = 3.0 * length
    polygon = tuple(
        map(
            tuple,
            (
                interface_point - extent * tangent,
                interface_point + extent * tangent,
                interface_point + extent * tangent + extent * normal,
                interface_point - extent * tangent + extent * normal,
            ),
        )
    )
    steps = cells
    dt = 0.2 * spacing / LIGHT_SPEED
    simulation = bz.Simulation(
        scene=raster.Scene(
            (
                raster.Material(),
                raster.Material(epsilon_right if anisotropic else 4.0),
            ),
            (
                raster.Object(
                    raster.ExtrudedPolygon(raster.Polygon(polygon), 0.0, 1.0),
                    1,
                ),
            ),
        ),
        raster_grid=raster.Grid.uniform(
            (0.0, 0.0, 0.0), (length, length, 1.0), (cells, cells, 1)
        ),
        polarization="te",
        time=np.arange(steps) * dt,
        sources=(),
        normalize_source=None,
        raster_options=raster.RasterOptions(smoothing=smoothing),
    )
    state = simulation.initial_state()
    initial = {}
    for component, field, time in (
        ("Ex", state.ex, 0.0),
        ("Ey", state.ey, 0.0),
        ("Hz", state.hz, -0.5 * dt),
    ):
        y, x = _yee_coordinates(component, field.shape, spacing)
        initial[component] = jnp.asarray(
            _exact_normal_pulse(
                component,
                y,
                x,
                time,
                normal=normal,
                interface_coordinate=interface_coordinate,
            ),
            dtype=field.dtype,
        )
    state = state._replace(ex=initial["Ex"], ey=initial["Ey"], hz=initial["Hz"])
    final = simulation.advance(state=state, progress=False).state

    ex = 0.5 * (np.asarray(final.ex[:-1]) + np.asarray(final.ex[1:]))
    ey = 0.5 * (np.asarray(final.ey[:, :-1]) + np.asarray(final.ey[:, 1:]))
    hz = np.asarray(final.hz)
    y, x = _yee_coordinates("Hz", hz.shape, spacing)
    exact_ex = _exact_normal_pulse(
        "Ex",
        y,
        x,
        steps * dt,
        normal=normal,
        interface_coordinate=interface_coordinate,
    )
    exact_ey = _exact_normal_pulse(
        "Ey",
        y,
        x,
        steps * dt,
        normal=normal,
        interface_coordinate=interface_coordinate,
    )
    exact_hz = _exact_normal_pulse(
        "Hz",
        y,
        x,
        (steps - 0.5) * dt,
        normal=normal,
        interface_coordinate=interface_coordinate,
    )
    coordinate = normal[0] * x + normal[1] * y
    epsilon = np.where(
        (coordinate <= interface_coordinate)[..., None, None],
        np.eye(2),
        epsilon_right_xy,
    )
    electric_error = np.stack((ex - exact_ex, ey - exact_ey), axis=-1)
    exact_electric = np.stack((exact_ex, exact_ey), axis=-1)
    error_density = (
        EPS_0
        * np.einsum("...i,...ij,...j->...", electric_error, epsilon, electric_error)
        + MU_0 * (hz - exact_hz) ** 2
    )
    exact_density = (
        EPS_0
        * np.einsum("...i,...ij,...j->...", exact_electric, epsilon, exact_electric)
        + MU_0 * exact_hz**2
    )
    # The run lasts 0.2 L/c, so boundary disturbances travel at most 1.6 um.
    margin = 2.0 * bz.um
    mask = (
        (x >= margin) & (x <= length - margin) & (y >= margin) & (y <= length - margin)
    )
    return math.sqrt(float(np.sum(error_density[mask]) / np.sum(exact_density[mask])))


def _convergence_statistics(grids, errors):
    """Return pairwise orders and a log-log fit over the three finest grids."""

    pairwise_orders = tuple(
        math.log2(coarse / fine)
        for coarse, fine in zip(errors[:-1], errors[1:], strict=True)
    )
    log_spacing = np.log(8.0 * bz.um / np.asarray(grids[-3:], dtype=float))
    log_error = np.log(np.asarray(errors[-3:], dtype=float))
    fitted_order, intercept = np.polyfit(log_spacing, log_error, 1)
    fitted = fitted_order * log_spacing + intercept
    residual = np.sum((log_error - fitted) ** 2)
    total = np.sum((log_error - np.mean(log_error)) ** 2)
    return pairwise_orders, float(fitted_order), float(1.0 - residual / total)


@pytest.mark.simulation
@pytest.mark.parametrize(
    ("angle_degrees", "interface_offset"),
    (
        (17.0, 0.07 * bz.um),
        (27.0, 0.07 * bz.um),
        (27.0, -0.11 * bz.um),
        (41.0, -0.11 * bz.um),
    ),
)
def test_sloped_anisotropic_interface_matches_exact_pulse_at_second_order(
    angle_degrees,
    interface_offset,
    validation_metrics,
):
    """Global Ex/Ey/Hz error converges quadratically across angle and grid phase."""

    grids = (48, 96, 192, 384)
    errors = tuple(
        _analytical_interface_error(
            cells,
            angle_degrees=angle_degrees,
            interface_offset=interface_offset,
        )
        for cells in grids
    )
    orders, fitted_order, fit_r_squared = _convergence_statistics(grids, errors)
    metadata = {
        "angle_degrees": angle_degrees,
        "interface_offset_um": interface_offset / bz.um,
        "grid_sizes": list(grids),
        "normalized_energy_errors": list(errors),
        "pairwise_orders": list(orders),
        "oracle": "exact source-free normal-incidence pulse",
        "epsilon_right_normal_tangent": [6.0, 4.0],
    }

    assert all(
        coarse > fine for coarse, fine in zip(errors[:-1], errors[1:], strict=True)
    )
    assert all(1.8 < order < 2.2 for order in orders[-2:])
    assert fitted_order < 2.2
    validation_metrics.check_lower(
        "asymptotic global-field convergence order",
        measured=fitted_order,
        lower_bound=1.8,
        resolution="96 -> 192 -> 384 cells",
        metadata=metadata,
    )
    validation_metrics.check_lower(
        "convergence log-log fit R-squared",
        measured=fit_r_squared,
        lower_bound=0.999,
        resolution="96 -> 192 -> 384 cells",
        metadata=metadata,
    )


@pytest.mark.simulation
@pytest.mark.parametrize(
    ("anisotropic", "control", "minimum_gain", "maximum_control_order"),
    (
        (True, "farjadpour_diagonal", 10.0, 1.0),
        (False, "volume", 3.0, 1.6),
    ),
)
def test_full_farjadpour_convergence_is_sensitive_to_weaker_material_controls(
    anisotropic,
    control,
    minimum_gain,
    maximum_control_order,
    validation_metrics,
):
    """Weaker material rules lose both fine-grid accuracy and asymptotic order."""

    grids = (48, 96, 192, 384)
    arguments = {"angle_degrees": 27.0, "interface_offset": -0.11 * bz.um}
    full_errors = tuple(
        _analytical_interface_error(cells, anisotropic=anisotropic, **arguments)
        for cells in grids
    )
    control_errors = tuple(
        _analytical_interface_error(
            cells,
            anisotropic=anisotropic,
            smoothing=control,
            **arguments,
        )
        for cells in grids
    )
    control_orders, _, _ = _convergence_statistics(grids, control_errors)
    fine_grid_gain = control_errors[-1] / full_errors[-1]
    metadata = {
        "control": control,
        "anisotropic_material": anisotropic,
        "grid_sizes": list(grids),
        "full_errors": list(full_errors),
        "control_errors": list(control_errors),
        "control_pairwise_orders": list(control_orders),
    }

    validation_metrics.check_lower(
        "full Farjadpour fine-grid accuracy gain",
        measured=fine_grid_gain,
        lower_bound=minimum_gain,
        unit="times",
        resolution="384 cells",
        metadata=metadata,
    )
    validation_metrics.check_upper(
        "weaker-control finest-pair convergence order",
        measured=control_orders[-1],
        upper_bound=maximum_control_order,
        resolution="192 -> 384 cells",
        metadata=metadata,
    )


def _te_energy(state) -> float:
    electric = np.sum(np.asarray(state.ex, dtype=np.float64) ** 2) + np.sum(
        np.asarray(state.ey, dtype=np.float64) ** 2
    )
    magnetic = np.sum(np.asarray(state.hz, dtype=np.float64) ** 2)
    return float(EPS_0 * electric + MU_0 * magnetic)


@pytest.mark.simulation
def test_sloped_interface_full_tensor_update_is_stable_for_four_thousand_steps():
    cells = 48
    spacing = 4.0 * bz.um / cells
    segment_steps = 500
    segments = 8
    dt = 0.2 * spacing / LIGHT_SPEED
    simulation, state = _sloped_interface_simulation(
        cells, np.arange(segment_steps * segments) * dt
    )
    initial_energy = _te_energy(state)
    energies = []

    for _ in range(segments):
        state = simulation.advance(
            state=state,
            num_steps=segment_steps,
            progress=False,
        ).state
        assert all(
            np.all(np.isfinite(np.asarray(field)))
            for field in (state.ex, state.ey, state.hz)
        )
        energies.append(_te_energy(state))

    assert max(energies) < 2.0 * initial_energy
