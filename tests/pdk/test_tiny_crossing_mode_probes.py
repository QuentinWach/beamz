import importlib.util

import numpy as np
import pytest

from beamz import Port, µm
from beamz.design.gds import import_component

pytestmark = [
    pytest.mark.integration,
    pytest.mark.pdk,
    pytest.mark.filterwarnings(
        "ignore:Support for class-based `config` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:Implicitly cleaning up <TemporaryDirectory.*:ResourceWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:unclosed file .*gdsfactory.*:ResourceWarning"
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("gdsfactory") is None,
        reason="gdsfactory not installed",
    ),
]


def _move_along(center: tuple[float, float], direction: str, distance: float):
    x, y = center
    return {
        "+x": (x + distance, y),
        "-x": (x - distance, y),
        "+y": (x, y + distance),
        "-y": (x, y - distance),
    }[str(direction)]


def _port_plane(
    port: Port,
    *,
    span: float,
    z_span: float,
    z_center: float,
    offset: float = 0.0,
):
    direction = port.signed_direction
    cx, cy = _move_along(port.center[:2], direction, offset)
    z0 = float(z_center) - 0.5 * float(z_span)
    z1 = float(z_center) + 0.5 * float(z_span)
    if direction.endswith("x"):
        return (cx, cy - 0.5 * float(span), z0), (cx, cy + 0.5 * float(span), z1)
    return (cx - 0.5 * float(span), cy, z0), (cx + 0.5 * float(span), cy, z1)


def _incoming_wave(direction: str) -> str:
    direction = str(direction)
    if direction.endswith(("x", "y")):
        return "minus"
    return "plus" if direction.startswith("+") else "minus"


def _outgoing_wave(direction: str) -> str:
    return "minus" if _incoming_wave(direction) == "plus" else "plus"


def _plane_clearances_to_active_box(
    plane,
    *,
    world_origin: tuple[float, float, float],
    width: float,
    height: float,
    depth: float,
    pml_xy: float,
    pml_z: float,
):
    active_min = np.asarray(world_origin, dtype=float) + np.asarray(
        [float(pml_xy), float(pml_xy), float(pml_z)], dtype=float
    )
    active_max = np.asarray(world_origin, dtype=float) + np.asarray(
        [
            float(width) - float(pml_xy),
            float(height) - float(pml_xy),
            float(depth) - float(pml_z),
        ],
        dtype=float,
    )
    a = np.asarray(plane[0], dtype=float) + np.asarray(world_origin, dtype=float)
    b = np.asarray(plane[1], dtype=float) + np.asarray(world_origin, dtype=float)
    pmin = np.minimum(a, b)
    pmax = np.maximum(a, b)
    return {
        "left": float(pmin[0] - active_min[0]),
        "right": float(active_max[0] - pmax[0]),
        "bottom": float(pmin[1] - active_min[1]),
        "top": float(active_max[1] - pmax[1]),
        "front": float(pmin[2] - active_min[2]),
        "back": float(active_max[2] - pmax[2]),
    }


def test_tiny_crossing_wave_selectors_match_current_3d_port_convention():
    for direction in ("+x", "-x", "+y", "-y"):
        assert _incoming_wave(direction) == "minus"
        assert _outgoing_wave(direction) == "plus"


def test_tiny_crossing_planes_keep_clearance_from_cpml_in_all_directions():
    n_core, n_clad = 3.47, 1.44
    core_t = 0.22 * µm
    clad_below = 0.50 * µm
    clad_above = 0.50 * µm
    z_padding = 2.00 * µm
    extension = 2.50 * µm
    pml_xy = 1.5 * µm
    pml_z = 1.0 * µm
    port_overlap = 0.10 * µm
    port_margin = 0.50 * µm
    source_monitor_gap = 0.10 * µm

    imported = import_component(
        "crossing",
        layer=(1, 0),
        n_core=n_core,
        n_clad=n_clad,
        core_thickness=core_t,
        clad_below=clad_below,
        clad_above=clad_above,
        xy_padding=extension + pml_xy,
        z_padding=z_padding + pml_z,
        extend_ports=True,
        port_overlap=port_overlap,
    )
    design = imported.design
    ports = {port.name: port for port in imported.ports}
    world_origin = imported.world_origin
    src = ports["o1"]
    width = src.size[1] if src.axis == "x" else src.size[0]
    span = max(width + 2.0 * port_margin, width + 0.1 * µm)
    z_center = src.center[2]
    z_span = clad_below + core_t + clad_above
    source_plane = _port_plane(
        src,
        span=span,
        z_span=z_span,
        z_center=z_center,
        offset=-source_monitor_gap,
    )
    monitor_planes = {
        "o1": _port_plane(src, span=span, z_span=z_span, z_center=z_center),
        "o2": _port_plane(ports["o2"], span=span, z_span=z_span, z_center=z_center),
        "o3": _port_plane(ports["o3"], span=span, z_span=z_span, z_center=z_center),
        "o4": _port_plane(ports["o4"], span=span, z_span=z_span, z_center=z_center),
    }

    source_clearances = _plane_clearances_to_active_box(
        source_plane,
        world_origin=world_origin,
        width=design.width,
        height=design.height,
        depth=design.depth,
        pml_xy=pml_xy,
        pml_z=pml_z,
    )
    assert min(source_clearances.values()) >= 1.95 * µm

    for plane in monitor_planes.values():
        clearances = _plane_clearances_to_active_box(
            plane,
            world_origin=world_origin,
            width=design.width,
            height=design.height,
            depth=design.depth,
            pml_xy=pml_xy,
            pml_z=pml_z,
        )
        assert min(clearances.values()) >= 1.95 * µm


def test_component_import_uses_an_explicit_pdk_layer_stack():
    import gdsfactory as gf

    import_component("straight", layer=(1, 0))
    imported = import_component(
        "straight",
        layer=(1, 0),
        n_core=3.47,
        n_clad=1.44,
        core_thickness=0.22 * µm,
        xy_padding=1.50 * µm,
        z_padding=0.50 * µm,
        layer_stack=gf.get_active_pdk().layer_stack,
    )

    assert np.isclose(imported.design.background.permittivity, 1.0)
    assert imported.ports
    assert any(
        np.isclose(getattr(s, "sidewall_angle", 0.0), 10.0)
        and np.isclose(getattr(s, "depth", 0.0), 0.22 * µm)
        for s in imported.design.structures
    )
