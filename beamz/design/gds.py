"""Import and export GDS layouts as BeamZ designs.

The module has no import-time dependency on gdsfactory. Install ``beamz[gds]``
before calling its public functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from beamz.const import µm
from beamz.design.core import Design
from beamz.design.materials import Material
from beamz.design.structures import Box, Polygon, Rectangle
from beamz.devices.ports import Port

Layer = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ImportedComponent:
    """A layout component converted into simulation-ready BeamZ objects.

    Attributes
    ----------
    design
        Imported and extruded simulation geometry in local design coordinates.
    ports
        Canonical modal ports at the original component port planes.
    component_name
        Name reported by the source layout component.
    world_origin
        Translation from local design coordinates to centered world coordinates.
    """

    design: Design
    ports: tuple[Port, ...]
    component_name: str
    world_origin: tuple[float, float, float]

    def port(self, name: str) -> Port:
        """Return a port by name."""
        for port in self.ports:
            if port.name == name:
                return port
        available = ", ".join(port.name for port in self.ports) or "none"
        raise KeyError(f"Unknown port {name!r}; available ports: {available}.")


@dataclass(frozen=True, slots=True)
class _StackLayer:
    name: str
    zmin: float
    zmax: float
    material: Material
    sidewall_angle: float = 0.0
    width_to_z: float = 0.0


@dataclass(frozen=True, slots=True)
class _VerticalProfile:
    depth: float
    core_zmin: float
    core_zmax: float
    active_zmin: float
    active_zmax: float
    world_z_origin: float
    background: Material
    layers: tuple[_StackLayer, ...] = ()
    sidewall_angle: float = 0.0
    width_to_z: float = 0.0


def _gdsfactory():
    try:
        import gdsfactory as gf
    except ImportError as exc:
        raise ImportError(
            "GDS support requires BeamZ's optional layout dependency. "
            "Install it with `pip install beamz[gds]`."
        ) from exc
    try:
        gf.get_active_pdk()
    except ValueError:
        generic_pdk = getattr(getattr(gf, "gpdk", None), "get_generic_pdk", None)
        if generic_pdk is None:
            raise
        generic_pdk().activate()
    return gf


def _resolve_component(component: Any, settings: Mapping[str, Any] | None):
    gf = _gdsfactory()
    try:
        return gf.get_component(component, settings=dict(settings or {}))
    except Exception as exc:
        raise ValueError(f"Could not resolve layout component {component!r}.") from exc


def _polygons_on_layer(component: Any, layer: Layer) -> tuple[np.ndarray, ...]:
    polygons_by_layer = component.get_polygons_points(by="tuple")
    polygons = tuple(
        np.asarray(points, dtype=float)[:, :2]
        for points in polygons_by_layer.get(tuple(layer), ())
    )
    if not polygons:
        available = sorted(polygons_by_layer)
        raise ValueError(
            f"Layer {tuple(layer)} is absent from component {component.name!r}; "
            f"available layers: {available}."
        )
    return polygons


def _material(value: Material | float) -> Material:
    return value if isinstance(value, Material) else Material(float(value))


def _stack_layers(
    layer_stack: Any,
    *,
    n_core: float,
    n_clad: float,
    material_map: Mapping[str, Material | float] | None,
) -> tuple[_StackLayer, ...]:
    materials: dict[str, Material] = {
        "si": Material(float(n_core) ** 2),
        "silicon": Material(float(n_core) ** 2),
        "sio2": Material(float(n_clad) ** 2),
        "oxide": Material(float(n_clad) ** 2),
        "air": Material(1.0),
    }
    materials.update(
        {
            str(name).strip().lower(): _material(value)
            for name, value in (material_map or {}).items()
        }
    )

    layers = []
    for name, level in getattr(layer_stack, "layers", {}).items():
        material = materials.get(str(getattr(level, "material", "")).lower())
        if material is None or str(name).lower() == "substrate":
            continue
        zmin = float(level.zmin) * µm
        zmax = zmin + float(level.thickness) * µm
        if zmax <= zmin:
            continue
        layers.append(
            _StackLayer(
                name=str(name),
                zmin=zmin,
                zmax=zmax,
                material=material,
                sidewall_angle=float(getattr(level, "sidewall_angle", 0.0) or 0.0),
                width_to_z=float(getattr(level, "width_to_z", 0.0) or 0.0),
            )
        )
    return tuple(layers)


def _vertical_profile(
    *,
    n_core: float,
    n_clad: float,
    core_thickness: float,
    clad_below: float,
    clad_above: float,
    z_padding: float,
    layer_stack: Any | None,
    material_map: Mapping[str, Material | float] | None,
) -> _VerticalProfile:
    if layer_stack is None:
        core_zmin = float(z_padding) + float(clad_below)
        core_zmax = core_zmin + float(core_thickness)
        depth = core_zmax + float(clad_above) + float(z_padding)
        return _VerticalProfile(
            depth=depth,
            core_zmin=core_zmin,
            core_zmax=core_zmax,
            active_zmin=float(z_padding),
            active_zmax=depth - float(z_padding),
            world_z_origin=-core_zmin,
            background=Material(float(n_clad) ** 2),
        )

    layers = _stack_layers(
        layer_stack,
        n_core=n_core,
        n_clad=n_clad,
        material_map=material_map,
    )
    if not layers:
        raise ValueError("The supplied layer stack has no recognized optical layers.")

    core_epsilon = float(n_core) ** 2
    core_candidates = [
        layer
        for layer in layers
        if layer.name.lower() == "core"
        or np.isclose(layer.material.permittivity, core_epsilon)
    ]
    if not core_candidates:
        raise ValueError(
            "The supplied layer stack has no core layer. Add its material to "
            "material_map or use the simple core_thickness/cladding inputs."
        )
    core = min(
        core_candidates,
        key=lambda item: (
            item.name.lower() != "core",
            abs((item.zmax - item.zmin) - float(core_thickness)),
        ),
    )
    stack_zmin = min(layer.zmin for layer in layers)
    stack_zmax = max(layer.zmax for layer in layers)
    z_offset = float(z_padding) - stack_zmin
    shifted_layers = tuple(
        _StackLayer(
            name=layer.name,
            zmin=layer.zmin + z_offset,
            zmax=layer.zmax + z_offset,
            material=layer.material,
            sidewall_angle=layer.sidewall_angle,
            width_to_z=layer.width_to_z,
        )
        for layer in layers
        if not np.isclose(layer.material.permittivity, core_epsilon)
    )
    depth = stack_zmax - stack_zmin + 2.0 * float(z_padding)
    return _VerticalProfile(
        depth=depth,
        core_zmin=core.zmin + z_offset,
        core_zmax=core.zmax + z_offset,
        active_zmin=float(z_padding),
        active_zmax=depth - float(z_padding),
        world_z_origin=-z_offset,
        background=Material(1.0),
        layers=shifted_layers,
        sidewall_angle=core.sidewall_angle,
        width_to_z=core.width_to_z,
    )


def _component_ports(
    component: Any,
    *,
    xmin_um: float,
    ymin_um: float,
    xy_padding: float,
) -> tuple[dict[str, Any], ...]:
    ports = []
    for port in component.ports:
        orientation = float(port.orientation) % 360.0
        rounded = int(round(orientation / 90.0) * 90) % 360
        try:
            direction = {180: "+x", 0: "-x", 90: "-y", 270: "+y"}[rounded]
        except KeyError as exc:
            raise ValueError(
                f"Port {port.name!r} has unsupported orientation {orientation}; "
                "BeamZ requires axis-aligned ports."
            ) from exc
        center_um = getattr(port, "dcenter", port.center)
        width_um = getattr(port, "dwidth", port.width)
        ports.append(
            {
                "name": str(port.name),
                "center": (
                    (float(center_um[0]) - xmin_um) * µm + float(xy_padding),
                    (float(center_um[1]) - ymin_um) * µm + float(xy_padding),
                ),
                "width": float(width_um) * µm,
                "direction": direction,
            }
        )
    return tuple(ports)


def _extend_ports(
    design: Design,
    ports: tuple[dict[str, Any], ...],
    *,
    profile: _VerticalProfile,
    material: Material,
    enabled: bool,
    overlap: float,
) -> Design:
    if not enabled:
        return design
    edges = {"+x": design.width, "-x": 0.0, "+y": design.height, "-y": 0.0}
    for port in ports:
        direction = str(port["direction"])
        outward = ("-" if direction.startswith("+") else "+") + direction[1]
        cx, cy = port["center"]
        shift = -float(overlap)
        sx = cx + shift * ({"+x": 1, "-x": -1}.get(outward, 0))
        sy = cy + shift * ({"+y": 1, "-y": -1}.get(outward, 0))
        width = float(port["width"])
        if outward.endswith("x"):
            edge = edges[outward]
            position = (min(sx, edge), cy - 0.5 * width, profile.core_zmin)
            size = (abs(edge - sx), width)
        else:
            edge = edges[outward]
            position = (cx - 0.5 * width, min(sy, edge), profile.core_zmin)
            size = (width, abs(edge - sy))
        design += Rectangle(
            position=position,
            width=size[0],
            height=size[1],
            depth=profile.core_zmax - profile.core_zmin,
            material=material,
            sidewall_angle=profile.sidewall_angle,
            width_to_z=profile.width_to_z,
        )
    return design


def _beamz_ports(
    ports: tuple[dict[str, Any], ...], profile: _VerticalProfile
) -> tuple[Port, ...]:
    core_center = 0.5 * (profile.core_zmin + profile.core_zmax)
    z_span = 2.0 * max(
        core_center - profile.active_zmin,
        profile.active_zmax - core_center,
    )
    result = []
    for item in ports:
        axis = str(item["direction"])[1]
        direction: Literal["+", "-"] = (
            "+" if str(item["direction"]).startswith("+") else "-"
        )
        width = float(item["width"])
        size = (0.0, width, z_span) if axis == "x" else (width, 0.0, z_span)
        result.append(
            Port(
                center=(*item["center"], core_center),
                size=size,
                name=str(item["name"]),
                direction=direction,
            )
        )
    return tuple(result)


def import_component(
    component: Any = "mmi1x2",
    *,
    layer: Layer = (1, 0),
    n_core: float = 2.0,
    n_clad: float = 1.44,
    core_thickness: float = 0.22e-6,
    clad_below: float = 0.5e-6,
    clad_above: float = 0.5e-6,
    xy_padding: float = 0.0,
    z_padding: float = 0.0,
    extend_ports: bool = False,
    port_overlap: float = 0.0,
    settings: Mapping[str, Any] | None = None,
    layer_stack: Any | None = None,
    material_map: Mapping[str, Material | float] | None = None,
    unify: bool = True,
) -> ImportedComponent:
    """Convert a gdsfactory component or active-PDK cell into a BeamZ design.

    Optional PDK stack interpretation is explicit: pass a gdsfactory
    ``LayerStack`` through ``layer_stack``. Cell names are resolved by
    ``gdsfactory.get_component``, so the active PDK remains the authority for
    component discovery.

    Parameters
    ----------
    component
        Component object, callable, or cell specification understood by
        gdsfactory.
    layer
        GDS layer and datatype containing the imported core geometry.
    n_core, n_clad
        Refractive indices used for core and simple cladding materials.
    core_thickness, clad_below, clad_above
        Simple vertical geometry in metres, used when ``layer_stack`` is absent.
    xy_padding, z_padding
        Domain padding in metres.
    extend_ports
        Extend imported waveguides from their port planes to the domain edges.
    port_overlap
        Inward overlap between imported polygons and port extensions.
    settings
        Settings forwarded to gdsfactory component resolution.
    layer_stack
        Optional explicit gdsfactory PDK layer stack.
    material_map
        Mapping from PDK material names to BeamZ materials or permittivities.
    unify
        Merge compatible touching polygons after import.
    """
    component = _resolve_component(component, settings)
    polygons = _polygons_on_layer(component, layer)
    points = np.vstack(polygons)
    xmin_um, ymin_um = np.min(points, axis=0)
    xmax_um, ymax_um = np.max(points, axis=0)
    profile = _vertical_profile(
        n_core=n_core,
        n_clad=n_clad,
        core_thickness=core_thickness,
        clad_below=clad_below,
        clad_above=clad_above,
        z_padding=z_padding,
        layer_stack=layer_stack,
        material_map=material_map,
    )
    design = Design(
        width=float(xmax_um - xmin_um) * µm + 2.0 * float(xy_padding),
        height=float(ymax_um - ymin_um) * µm + 2.0 * float(xy_padding),
        depth=profile.depth,
        background=profile.background,
    )
    for stack_layer in profile.layers:
        design += Rectangle(
            position=(0.0, 0.0, stack_layer.zmin),
            width=design.width,
            height=design.height,
            depth=stack_layer.zmax - stack_layer.zmin,
            material=stack_layer.material,
        )

    core_material = Material(float(n_core) ** 2)
    for points_um in polygons:
        vertices = tuple(
            (
                (float(x) - xmin_um) * µm + float(xy_padding),
                (float(y) - ymin_um) * µm + float(xy_padding),
                profile.core_zmin,
            )
            for x, y in points_um
        )
        design += Polygon(
            vertices=vertices,
            depth=profile.core_zmax - profile.core_zmin,
            material=core_material,
            sidewall_angle=profile.sidewall_angle,
            width_to_z=profile.width_to_z,
        )

    raw_ports = _component_ports(
        component,
        xmin_um=float(xmin_um),
        ymin_um=float(ymin_um),
        xy_padding=xy_padding,
    )
    design = _extend_ports(
        design,
        raw_ports,
        profile=profile,
        material=core_material,
        enabled=bool(extend_ports),
        overlap=float(port_overlap),
    )
    if unify:
        design = design.unified_polygons()
    return ImportedComponent(
        design=design,
        ports=_beamz_ports(raw_ports, profile),
        component_name=str(component.name),
        world_origin=(
            -0.5 * design.width,
            -0.5 * design.height,
            profile.world_z_origin,
        ),
    )


def import_gds(path: str | Path, **kwargs: Any) -> ImportedComponent:
    """Read a GDS file and convert its top component into a BeamZ design.

    Additional keyword arguments are forwarded to :func:`import_component`.
    """
    gf = _gdsfactory()
    source = Path(path)
    try:
        component = gf.read.import_gds(source)
    except Exception as exc:
        raise ValueError(f"Could not read GDS file {source}.") from exc
    return import_component(component, **kwargs)


def _material_key(structure: Any) -> tuple[float, float, float] | None:
    material = getattr(structure, "material", None)
    if material is None:
        return None
    return (
        float(getattr(material, "permittivity", 1.0)),
        float(getattr(material, "permeability", 1.0)),
        float(getattr(material, "conductivity", 0.0)),
    )


def _add_polygon(component: Any, structure: Any, layer: Layer) -> None:
    gf = _gdsfactory()
    vertices = getattr(structure, "vertices", ())
    if not vertices:
        raise TypeError(f"GDS export does not support {type(structure).__name__}.")
    exterior = [[float(x) / µm, float(y) / µm] for x, y, _ in vertices]
    interiors = [
        [[float(x) / µm, float(y) / µm] for x, y, _ in path]
        for path in getattr(structure, "interiors", ())
    ]
    if not interiors:
        component.add_polygon(exterior, layer=layer)
        return

    polygon = gf.Component()
    polygon.add_polygon(exterior, layer=layer)
    for hole_points in interiors:
        hole = gf.Component()
        hole.add_polygon(hole_points, layer=layer)
        polygon = gf.boolean(polygon, hole, operation="not", layer=layer)
    component.add_ref(polygon)


def export_gds(
    design: Design,
    path: str | Path,
    *,
    cell_name: str = "beamz_design",
) -> Path:
    """Export planar design structures to material-grouped GDS layers.

    Structures sharing electromagnetic material properties share a GDS layer.
    Volumetric structures without a planar projection are rejected explicitly.
    """
    gf = _gdsfactory()
    component = gf.Component(cell_name)
    material_layers: dict[tuple[float, float, float], Layer] = {}
    for structure in design.unified_polygons().structures:
        if getattr(structure, "is_pml", False):
            continue
        if isinstance(structure, Box):
            structure = structure.to_rectangle()
        material_key = _material_key(structure)
        if material_key is None:
            continue
        layer = material_layers.setdefault(material_key, (len(material_layers), 0))
        _add_polygon(component, structure, layer)
    if not material_layers:
        raise ValueError("The design has no planar material structures to export.")
    return Path(component.write_gds(gdspath=path, with_metadata=False))


__all__ = ["ImportedComponent", "export_gds", "import_component", "import_gds"]
