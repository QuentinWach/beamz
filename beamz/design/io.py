import gdspy
import numpy as np

from beamz.visual.helpers import display_status


def export_grid_gds(
    grid,
    output_file,
    threshold=None,
    n_core=None,
    n_clad=None,
    dx=None,
    layer=0,
    cell_name="main",
):
    """Export a rasterized permittivity grid to a GDS file using contour extraction.

    Accepts either a RegularGrid object (uses .permittivity and .dx) or a raw
    numpy array (requires the ``dx`` parameter).

    Args:
        grid: A RegularGrid instance or a 2D numpy array of permittivity values.
        output_file (str): Path for the output GDS file.
        threshold (float, optional): Permittivity threshold for contour extraction.
            If not given, computed from n_core/n_clad or from array min/max.
        n_core (float, optional): Core refractive index (used for threshold calculation).
        n_clad (float, optional): Cladding refractive index (used for threshold and padding).
        dx (float, optional): Grid spacing in metres. Required when ``grid`` is a numpy array.
        layer (int): GDS layer number (default 0).
        cell_name (str): GDS cell name (default "main").

    Returns:
        int: Number of polygons written.
    """
    import matplotlib.pyplot as plt

    # Accept either a RegularGrid or a raw numpy array
    if hasattr(grid, "permittivity") and hasattr(grid, "dx"):
        eps = np.asarray(grid.permittivity)
        dx = grid.dx
    else:
        eps = np.asarray(grid)
        if dx is None:
            raise ValueError("dx is required when grid is a raw numpy array")

    # Determine threshold
    if threshold is None:
        if n_core is not None and n_clad is not None:
            threshold = (n_core**2 + n_clad**2) / 2
        else:
            threshold = (float(eps.max()) + float(eps.min())) / 2

    # Determine cladding value for padding
    if n_clad is not None:
        pad_value = n_clad**2
    else:
        pad_value = float(eps.min())

    # Pad the grid with cladding so contours close at boundaries
    eps_padded = np.full((eps.shape[0] + 2, eps.shape[1] + 2), pad_value)
    eps_padded[1:-1, 1:-1] = eps

    # Extract contours
    fig_temp, ax_temp = plt.subplots()
    cs = ax_temp.contour(eps_padded.T, levels=[threshold])
    plt.close(fig_temp)

    # Write GDS
    lib = gdspy.GdsLibrary(unit=1e-6, precision=1e-9)
    cell = lib.new_cell(cell_name)

    n_polys = 0
    for seg in cs.allsegs[0]:
        # Offset by -1 for padding, then convert grid coords to microns
        verts_um = [((x - 1) * dx * 1e6, (y - 1) * dx * 1e6) for x, y in seg]
        if len(verts_um) >= 3:
            cell.add(gdspy.Polygon(verts_um, layer=layer))
            n_polys += 1

    lib.write_gds(output_file)
    return n_polys


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


def export_gds(design, output_file):
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
    design.unify_polygons()
    # Scale factor to convert from meters to microns
    scale = 1e6  # 1 meter = 1e6 microns

    # Group structures by material properties
    material_groups = {}
    for structure in design.structures:
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
        f"Created design with size: {design.width:.2e} x {design.height:.2e} x {design.depth:.2e} m"
    )
