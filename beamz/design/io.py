import gdspy
import numpy as np

from beamz.visual.helpers import display_status


def _orientation_to_inward_direction(orientation_deg: float) -> str:
    """Map gdsfactory port orientation to inward BeamZ launch direction."""
    orientation = int(round(orientation_deg / 90.0) * 90) % 360
    direction_map = {
        180: "+x",
        0: "-x",
        90: "-y",
        270: "+y",
    }
    if orientation not in direction_map:
        raise ValueError(
            f"Unsupported port orientation {orientation_deg}. Expected multiples of 90°."
        )
    return direction_map[orientation]


class _GDSFactoryNamespace:
    """Namespace for gdsfactory-to-BeamZ import helpers."""

    def load(
        self,
        cell: str = "mmi1x2",
        layer: tuple[int, int] = (1, 0),
        n_core: float = 2.0,
        n_clad: float = 1.44,
        padding: float = 3.0,
        component_kwargs: dict | None = None,
    ):
        """Load a gdsfactory cell into a BeamZ design.

        Args:
            cell: Name of a gdsfactory component factory (for example ``mmi1x2``).
            layer: Core geometry layer (layer, datatype) in gdsfactory.
            n_core: Core refractive index.
            n_clad: Cladding refractive index.
            padding: Padding around imported geometry in microns.
            component_kwargs: Optional kwargs forwarded to component factory.

        Returns:
            Tuple ``(design, ports)`` where ``ports`` maps port name to metadata.
        """
        from beamz.design.core import Design
        from beamz.design.materials import Material
        from beamz.design.structures import Polygon

        try:
            import gdsfactory as gf
        except ImportError as exc:
            raise ImportError(
                "gdsfactory is required for design.io.gdsf.load(...). "
                "Install it with `pip install gdsfactory`."
            ) from exc

        gf.gpdk.PDK.activate()
        component_kwargs = component_kwargs or {}

        if isinstance(cell, str):
            factory = getattr(gf.components, cell, None)
            if factory is None:
                raise ValueError(
                    f"Unknown gdsfactory component '{cell}'. "
                    "Expected a factory in gdsfactory.components."
                )
            component = factory(**component_kwargs)
        elif callable(cell):
            component = cell(**component_kwargs)
        else:
            component = cell

        polygons_by_layer = component.get_polygons_points(by="tuple")
        if layer not in polygons_by_layer:
            available = sorted(polygons_by_layer.keys())
            raise ValueError(
                f"Layer {layer} not found in component '{component.name}'. "
                f"Available layers: {available}"
            )

        core_polygons = polygons_by_layer[layer]
        if not core_polygons:
            raise ValueError(
                f"Component '{component.name}' has no polygons on {layer}."
            )

        all_points = np.vstack([np.asarray(poly)[:, :2] for poly in core_polygons])
        xmin, ymin = np.min(all_points, axis=0)
        xmax, ymax = np.max(all_points, axis=0)
        pad_um = float(padding)
        width = float((xmax - xmin + 2.0 * pad_um) * 1e-6)
        height = float((ymax - ymin + 2.0 * pad_um) * 1e-6)

        design = Design(
            width=width, height=height, depth=0, material=Material(n_clad**2)
        )

        for poly_points in core_polygons:
            vertices = [
                (
                    float((point[0] - xmin + pad_um) * 1e-6),
                    float((point[1] - ymin + pad_um) * 1e-6),
                )
                for point in poly_points
            ]
            design += Polygon(vertices=vertices, material=Material(n_core**2), depth=0)

        ports = {}
        for port in component.ports:
            orientation = float(port.orientation) % 360.0
            ports[port.name] = {
                "center": (
                    float((port.center[0] - xmin + pad_um) * 1e-6),
                    float((port.center[1] - ymin + pad_um) * 1e-6),
                ),
                "width": float(port.width) * 1e-6,
                "orientation": orientation,
                "direction": _orientation_to_inward_direction(orientation),
            }

        return design, ports


gdsf = _GDSFactoryNamespace()


def import_gds(gds_file: str, default_depth=1e-6):
    """Import a GDS file and return polygon and layer data.

    Args:
        gds_file (str): Path to the GDS file
        default_depth (float): Default depth/thickness for imported structures in meters
    """
    from beamz.design.core import Design
    from beamz.design.structures import Polygon

    gds_lib = gdspy.GdsLibrary(infile=gds_file)
    design = Design()  # Create Design instance
    cells = gds_lib.cells  # Get all cells from the library
    total_polygons_imported = 0

    # Filter cells to import: skip context/metadata cells and primitive (unreferenced) cells
    # gdsfactory creates hierarchical GDS files where:
    # - $$$CONTEXT_INFO$$$ contains metadata (skip)
    # - Primitive cells (0 references) contain un-transformed geometry (skip)
    # - Composed cells (has references) contain the final transformed geometry (import)
    cells_to_import = []
    for cell_name, cell in cells.items():
        # Skip gdsfactory metadata/context cells
        if cell_name.startswith("$$$") or "CONTEXT" in cell_name.upper():
            continue
        # Only import from composed cells (cells that have references to other cells)
        # These contain the correctly transformed/positioned geometry
        if len(cell.references) > 0:
            cells_to_import.append(cell)

    # If no composed cells found, fall back to importing all non-context cells
    # This handles simple GDS files without hierarchy
    if not cells_to_import:
        cells_to_import = [
            cell
            for name, cell in cells.items()
            if not name.startswith("$$$") and "CONTEXT" not in name.upper()
        ]

    for cell in cells_to_import:
        # Get polygons by spec, which returns a dict: {(layer, datatype): [poly1_points, poly2_points,...]}
        gdspy_polygons_by_spec = cell.get_polygons(by_spec=True)
        for (
            layer_num,
            _datatype,
        ), list_of_polygon_points in gdspy_polygons_by_spec.items():
            if layer_num not in design.layers:
                design.layers[layer_num] = []
            for polygon_points in list_of_polygon_points:
                # Convert points from microns to meters and ensure CCW ordering
                vertices_2d = [
                    (point[0] * 1e-6, point[1] * 1e-6) for point in polygon_points
                ]
                # Create polygon with appropriate depth
                beamz_polygon = Polygon(vertices=vertices_2d, depth=default_depth)
                design.layers[layer_num].append(beamz_polygon)
                design.structures.append(beamz_polygon)
                total_polygons_imported += 1

    # Set 3D flag if we have depth
    if default_depth > 0:
        design.is_3d = True
        design.depth = default_depth

    print(
        f"Imported {total_polygons_imported} polygons from '{gds_file}' into Design object."
    )
    if design.is_3d:
        print(f"3D design with depth: {design.depth:.2e} m")
    return design


def export_gds(self, output_file):
    """Export a BEAMZ design (including only the structures, not sources or monitors) to a GDS file.

    For 3D designs, structures with the same material that touch (in 3D) will be placed in the same layer.
    """
    from beamz.design.structures import (
        Circle,
        CircularBend,
        Polygon,
        Rectangle,
        Ring,
        Taper,
    )
    from beamz.devices.monitors import Monitor
    from beamz.devices.sources import GaussianSource, ModeSource

    # Create library with micron units (1e-6) and nanometer precision (1e-9)
    lib = gdspy.GdsLibrary(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("main")
    # First, we unify the polygons given their material and if they touch
    self.unify_polygons()
    # Scale factor to convert from meters to microns
    scale = 1e6  # 1 meter = 1e6 microns

    # Group structures by material properties
    material_groups = {}
    for structure in self.structures:
        # Skip PML visualizations, sources, monitors
        if hasattr(structure, "is_pml") and structure.is_pml:
            continue
        if isinstance(structure, (ModeSource, GaussianSource, Monitor)):
            continue
        # Create material key based on material properties
        material = getattr(structure, "material", None)
        if material is None:
            continue
        material_key = (
            getattr(material, "permittivity", 1.0),
            getattr(material, "permeability", 1.0),
            getattr(material, "conductivity", 0.0),
        )
        if material_key not in material_groups:
            material_groups[material_key] = []
        material_groups[material_key].append(structure)

    # Export each material group as a separate layer
    for layer_num, (material_key, structures) in enumerate(material_groups.items()):
        for structure in structures:
            # Get vertices based on structure type
            if isinstance(structure, Polygon):
                vertices = structure.vertices
                interiors = (
                    structure.interiors if hasattr(structure, "interiors") else []
                )
            elif isinstance(structure, Rectangle):
                x, y = structure.position[0:2]  # Take only x,y from position
                w, h = structure.width, structure.height
                vertices = [(x, y, 0), (x + w, y, 0), (x + w, y + h, 0), (x, y + h, 0)]
                interiors = []
            elif isinstance(structure, (Circle, Ring, CircularBend, Taper)):
                if hasattr(structure, "to_polygon"):
                    poly = structure.to_polygon()
                    vertices = poly.vertices
                    interiors = getattr(poly, "interiors", [])
                else:
                    continue
            else:
                continue

            # Project vertices to 2D and scale to microns
            vertices_2d = [(x * scale, y * scale) for x, y, _ in vertices]
            if not vertices_2d:
                continue
            # Scale and project interiors if they exist
            interior_2d = []
            if interiors:
                for interior in interiors:
                    interior_2d.append([(x * scale, y * scale) for x, y, _ in interior])
            try:
                # Create gdspy polygon for this layer
                if interior_2d:
                    gdspy_poly = gdspy.Polygon(
                        vertices_2d, layer=layer_num, holes=interior_2d
                    )
                else:
                    gdspy_poly = gdspy.Polygon(vertices_2d, layer=layer_num)
                cell.add(gdspy_poly)
            except Exception as e:
                print(f"Warning: Failed to create GDS polygon: {e}")
                continue

    # Write the GDS file
    lib.write_gds(output_file)
    print(
        f"GDS file saved as '{output_file}' with {len(material_groups)} material-based layers"
    )
    # Print material information for each layer
    for layer_num, (material_key, structures) in enumerate(material_groups.items()):
        print(
            f"Layer {layer_num}: εᵣ={material_key[0]:.1f}, μᵣ={material_key[1]:.1f}, σ={material_key[2]:.2e} S/m"
        )
    display_status(
        f"Created design with size: {self.width:.2e} x {self.height:.2e} x {self.depth:.2e} m"
    )
