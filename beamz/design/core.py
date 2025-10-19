import random
import numpy as np
from beamz.const import µm
from beamz.design.materials import Material
from beamz.design.structures import Polygon, Rectangle, Circle, Ring, CircularBend, Taper


class Design:
    def __init__(self, width:float=5*µm, height:float=5*µm, depth:float=0, material:Material=None, 
        color:str=None, border_color:str="black"):
                    
        if material is None: material = Material(permittivity=1.0, permeability=1.0, conductivity=0.0)
        self.structures = [Rectangle(position=(0,0,0), width=width, height=height, depth=depth, material=material, color=color)]
        self.sources = []
        self.monitors = []
        self.boundaries = []
        self.width = width
        self.height = height
        self.depth = depth
        self.border_color = border_color
        self.time = 0
        self.is_3d = False if depth is None else True
        self.layers: dict[int, list[Polygon]] = {}
        self.is_3d = False if depth is None or depth == 0 else True


    def __str__(self):
        return f"Design with {len(self.structures)} structures ({'3D' if self.is_3d else '2D'})"

    def __iadd__(self, structure):
        """Implement += operator for adding structures."""
        self.add(structure)
        return self

    def _determine_if_3d(self):
        """Determine if the design should be visualized in 3D (delegated to beamz.viz)."""
        return viz.determine_if_3d(self)

    def _simplify_polygon_for_3d(self, shapely_polygon, tolerance_factor=0.01):
        """Simplify a Shapely polygon to reduce vertex count for better 3D triangulation."""
        try:
            # Calculate appropriate tolerance based on polygon size
            bounds = shapely_polygon.bounds
            size = max(bounds[2] - bounds[0], bounds[3] - bounds[1])  # max of width, height
            tolerance = size * tolerance_factor
            # Apply simplification
            simplified = shapely_polygon.simplify(tolerance, preserve_topology=True)
            # Check if simplification was successful and polygon is still valid
            if simplified.is_valid and not simplified.is_empty:
                # Check if we still have a reasonable number of vertices
                if simplified.geom_type == 'Polygon':
                    exterior_coords = len(list(simplified.exterior.coords))
                    if 3 <= exterior_coords <= 100:  # Reasonable range for 3D visualization
                        return simplified
                    elif exterior_coords > 100:
                        # Try more aggressive simplification
                        more_simplified = shapely_polygon.simplify(tolerance * 2.0, preserve_topology=True)
                        if more_simplified.is_valid and not more_simplified.is_empty:
                            return more_simplified
            
            # If simplification failed or created invalid geometry, return original
            return shapely_polygon
            
        except Exception as e:
            print(f"Warning: Polygon simplification failed ({e}), using original")
            return shapely_polygon
 
    def add(self, structure):
        """Core add function for adding structures on top of the design."""
        # Check for 3D structures - improved detection
        if hasattr(structure, 'depth') and structure.depth != 0: self.is_3d = True
        elif hasattr(structure, 'position') and len(structure.position) > 2 and structure.position[2] != 0: self.is_3d = True
        elif hasattr(structure, 'vertices') and structure.vertices:
            # Check if any vertex has non-zero z coordinate
            for vertex in structure.vertices:
                if len(vertex) > 2 and vertex[2] != 0:
                    self.is_3d = True
                    break

    def scatter(self, structure, n=1000, xyrange=(-5*µm, 5*µm), scale_range=(0.05, 1)):
        """Randomly distribute a given object over the design domain."""
        for _ in range(n):
            new_structure = structure.copy()
            new_structure.shift(random.uniform(xyrange[0], xyrange[1]), random.uniform(xyrange[0], xyrange[1]))
            new_structure.rotate(random.uniform(0, 360))
            new_structure.scale(random.uniform(scale_range[0], scale_range[1]))
            self.add(new_structure)
    
    def unify_polygons(self):
        """If polygons are the same material and overlap spatially, unify them into a single, simplified polygon."""
        try:
            from shapely.geometry import Polygon as ShapelyPolygon
            from shapely.ops import unary_union
        except ImportError:
            display_status("Shapely library is required for polygon unification. \
                            Please install with: pip install shapely", "error")
            return False
            
        # Group structures by material properties
        material_groups = {}
        non_polygon_structures = []
        # Track which structures to remove later
        structures_to_remove = []
        # First pass: group polygons by material
        for structure in self.structures:
            # Skip PML visualizations, sources, monitors
            if hasattr(structure, 'is_pml') and structure.is_pml:
                non_polygon_structures.append(structure)
                continue
            # Import at function level to avoid circular imports
            from beamz.devices.sources import ModeSource, GaussianSource
            from beamz.devices.monitors import Monitor
            if isinstance(structure, (ModeSource, GaussianSource, Monitor)):
                non_polygon_structures.append(structure)
                continue
            # Only process polygon-like structures with vertices
            if not hasattr(structure, 'vertices') or not hasattr(structure, 'material'):
                non_polygon_structures.append(structure)
                continue
                
            # Create a material key based on material properties
            material = structure.material
            if not material:
                non_polygon_structures.append(structure)
                continue
            material_key = (
                getattr(material, 'permittivity', None),
                getattr(material, 'permeability', None),
                getattr(material, 'conductivity', None)
            )
            
            # Add to the appropriate group
            if material_key not in material_groups: material_groups[material_key] = []
            # Convert to Shapely polygon
            try:
                # Handle polygons that might have interiors defined
                if hasattr(structure, 'interiors') and structure.interiors:
                    # Ensure vertices are not empty, and interiors are lists of coordinates
                    valid_interiors = [list(i_path) for i_path in structure.interiors if i_path]
                    if structure.vertices and valid_interiors:
                         shapely_polygon = ShapelyPolygon(shell=structure.vertices, holes=valid_interiors)
                    elif structure.vertices: # Only exterior is valid
                         shapely_polygon = ShapelyPolygon(shell=structure.vertices)
                    else: # No valid exterior
                        display_status(f"Skipping structure with no valid exterior vertices: {structure}", "warning")
                        non_polygon_structures.append(structure)
                        continue
                elif hasattr(structure, 'vertices') and structure.vertices: # Simple polygon with only exterior vertices
                    shapely_polygon = ShapelyPolygon(shell=structure.vertices)
                else: # Not a valid polygon structure
                    non_polygon_structures.append(structure)
                    continue # Skip if no vertices

                if shapely_polygon.is_valid:
                    material_groups[material_key].append((structure, shapely_polygon))
                    structures_to_remove.append(structure)
                else:
                    display_status(f"Skipping invalid polygon: {structure}", "warning")
                    non_polygon_structures.append(structure)
            except Exception as e:
                display_status(f"Error converting structure to Shapely polygon: {e}", "warning")
                non_polygon_structures.append(structure)
        
        # Second pass: Check for Ring objects that don't touch other objects
        rings_to_preserve = []
        for material_key, structure_group in material_groups.items():
            if len(structure_group) <= 1:
                continue  # Skip material groups with only one structure
            rings_in_group = [(idx, s) for idx, s in enumerate(structure_group) if isinstance(s[0], Ring)]
            if not rings_in_group:
                continue  # No rings in this group
            # For each Ring, preserve it unconditionally to maintain hole structure
            for ring_idx, (ring, ring_shapely) in rings_in_group:
                rings_to_preserve.append(ring)
                # Remove it from the material group to prevent it from being unified
                if ring in structures_to_remove:
                    structures_to_remove.remove(ring)
        
        # Third pass: unify polygons within each material group
        new_structures = []
        for material_key, structure_group in material_groups.items():
            # Exclude preserved rings from union to avoid duplicate drawing (simplified + original)
            filtered_group = [s for s in structure_group if s[0] not in rings_to_preserve]
            if len(filtered_group) <= 1:
                # Zero or one non-ring structure left to merge; keep originals (non-rings)
                new_structures.extend([s[0] for s in filtered_group])
                for s in filtered_group:
                    if s[0] in structures_to_remove:
                        structures_to_remove.remove(s[0])
                # Rings are already preserved and excluded from removal above
                continue
                
            # Extract shapely polygons for merging
            shapely_polygons = [p[1] for p in filtered_group]
            # Get the material from the first structure in the group
            material = filtered_group[0][0].material
            try:
                # Unify the polygons
                merged = unary_union(shapely_polygons)
                # The result could be a single polygon or a multipolygon
                if merged.geom_type == 'Polygon':
                    # Simplify the unified polygon to reduce complexity for 3D triangulation
                    simplified = self._simplify_polygon_for_3d(merged)
                    # Don't slice off the last vertex - our add_to_plot method needs complete vertices
                    exterior_coords = list(simplified.exterior.coords[:-1])  # Keep [:-1] to remove duplicate closing vertex from Shapely
                    interior_coords_lists = [list(interior.coords[:-1]) for interior in simplified.interiors]
                    if exterior_coords and len(exterior_coords) >= 3:
                        # Add depth and z information from the first structure
                        first_structure = filtered_group[0][0]
                        depth = getattr(first_structure, 'depth', 0)
                        z = getattr(first_structure, 'z', 0)
                        
                        new_poly = Polygon(vertices=exterior_coords, interiors=interior_coords_lists, 
                                         material=material, depth=depth, z=z)
                        new_structures.append(new_poly)
                        display_status(f"Unified {len(structure_group)} polygons with permittivity={material_key[0]} \
                                        (simplified to {len(exterior_coords)} vertices)", "success")
                    else:
                        display_status(f"Failed to convert merged polygon for material {material_key[0]} \
                            (no exterior or too few vertices), keeping original {len(structure_group)} structures.", "warning")
                        new_structures.extend([s[0] for s in structure_group])
                        for s_tuple in structure_group: # Ensure these are not removed
                            if s_tuple[0] in structures_to_remove:
                                structures_to_remove.remove(s_tuple[0])

                elif merged.geom_type == 'MultiPolygon':
                    all_geoms_converted_successfully = True
                    temp_new_polys_for_multipolygon = []
                    for geom in merged.geoms:
                        # Simplify each polygon in the multipolygon
                        simplified_geom = self._simplify_polygon_for_3d(geom)
                        # Keep [:-1] to remove duplicate closing vertex from Shapely (our add_to_plot will add it back)
                        exterior_coords = list(simplified_geom.exterior.coords[:-1])
                        interior_coords_lists = [list(interior.coords[:-1]) for interior in simplified_geom.interiors]
                        if exterior_coords and len(exterior_coords) >= 3:
                            # Add depth and z information from the first structure
                            first_structure = filtered_group[0][0]
                            depth = getattr(first_structure, 'depth', 0)
                            z = getattr(first_structure, 'z', 0)
                            new_poly = Polygon(vertices=exterior_coords, interiors=interior_coords_lists, 
                                             material=material, depth=depth, z=z)
                            temp_new_polys_for_multipolygon.append(new_poly)
                        else: # A sub-geometry had no exterior or too few vertices
                            all_geoms_converted_successfully = False
                            display_status(f"Failed to convert a geometry (no exterior or too few vertices) \
                                        within MultiPolygon for material {material_key[0]}.", "warning")
                            break 
                    
                    if all_geoms_converted_successfully:
                        new_structures.extend(temp_new_polys_for_multipolygon)
                        total_vertices = sum(len(p.vertices) for p in temp_new_polys_for_multipolygon)
                        display_status(f"Unified into {len(merged.geoms)} separate polygons with \
                                permittivity={material_key[0]} (simplified to {total_vertices} total vertices)", "success")
                    else:
                        display_status(f"Reverting unification for material {material_key[0]} due to conversion error \
                            in MultiPolygon, keeping original {len(structure_group)} structures.", "warning")
                        new_structures.extend([s[0] for s in structure_group])
                        for s_tuple in structure_group: # Ensure these are not removed
                            if s_tuple[0] in structures_to_remove:
                                structures_to_remove.remove(s_tuple[0])
                else:
                    # If the result is something unexpected, keep the original structures
                    display_status(f"Unexpected geometry type after union: {merged.geom_type} for material {material_key[0]}, \
                        keeping original structures", "warning")
                    new_structures.extend([s[0] for s in structure_group])
                    for s in structure_group:
                        if s[0] in structures_to_remove:
                            structures_to_remove.remove(s[0])
            except Exception as e:
                display_status(f"Error unifying polygons: {e}", "error")
                # Keep original structures if unification fails
                new_structures.extend([s[0] for s in structure_group])
                for s in structure_group:
                    if s[0] in structures_to_remove:
                        structures_to_remove.remove(s[0])
        
        # Rebuild structures list preserving original z-order
        # Create a mapping from material groups to their unified replacements
        material_replacements = {}
        
        # Match unified structures to their material groups by material properties
        for new_struct in new_structures:
            if hasattr(new_struct, 'material') and new_struct.material:
                new_material_key = (
                    getattr(new_struct.material, 'permittivity', None),
                    getattr(new_struct.material, 'permeability', None),
                    getattr(new_struct.material, 'conductivity', None)
                )
                # Find the material group that this unified structure belongs to
                for material_key, structure_group in material_groups.items():
                    if len(structure_group) > 1 and material_key == new_material_key:
                        if material_key not in material_replacements:
                            material_replacements[material_key] = []
                        material_replacements[material_key].append(new_struct)
                        break
        
        # Rebuild the structures list in original order
        rebuilt_structures = []
        material_groups_used = set()
        
        for structure in self.structures:
            if structure in structures_to_remove:
                # Find which material group this structure belongs to
                structure_material_key = None
                if hasattr(structure, 'material') and structure.material:
                    structure_material_key = (
                        getattr(structure.material, 'permittivity', None),
                        getattr(structure.material, 'permeability', None),
                        getattr(structure.material, 'conductivity', None)
                    )
                
                # Add the unified replacement(s) for this material group (only once)
                if structure_material_key and structure_material_key not in material_groups_used:
                    if structure_material_key in material_replacements:
                        rebuilt_structures.extend(material_replacements[structure_material_key])
                        material_groups_used.add(structure_material_key)
            else:
                # Keep non-unified structures (includes preserved rings and non-polygon structures)
                rebuilt_structures.append(structure)
        
        # Replace the structures list
        self.structures = rebuilt_structures
        
        # Final report
        display_status(f"Polygon unification complete: {len(structures_to_remove)} structures merged \
            into {len(new_structures)} unified shapes, {len(rings_to_preserve)} isolated rings preserved", "success")
        return True

    ## Rastered Grid Methods =============================================================
    
    def get_material_value(self, x, y, z=0, dx=None, dt=None):
        """Return the material value at a given (x, y, z) coordinate, prioritizing the topmost structure."""
        # First get material values from underlying structures
        # Start with default background material 
        epsilon = 1.0
        mu = 1.0
        sigma_base = 0.0
        # Find the material values from the structures (outside PML calculation)
        for structure in reversed(self.structures):
            if isinstance(structure, Polygon):
                if structure.point_in_polygon(x, y, z):
                    epsilon, mu, sigma_base = structure.material.get_sample()
                    break
            elif isinstance(structure, Rectangle):
                if structure.is_pml:
                    # Skip visual PML structures - they're just for display
                    continue
                if structure.point_in_polygon(x, y, z):
                    epsilon, mu, sigma_base = structure.material.get_sample()
                    break
            elif isinstance(structure, Circle):
                if np.hypot(x - structure.position[0], y - structure.position[1]) <= structure.radius:
                    epsilon, mu, sigma_base = structure.material.get_sample()
                    break
            elif isinstance(structure, Ring):
                distance = np.hypot(x - structure.position[0], y - structure.position[1])
                if structure.inner_radius <= distance <= structure.outer_radius:
                    epsilon, mu, sigma_base = structure.material.get_sample()
                    break
            elif isinstance(structure, CircularBend):
                distance = np.hypot(x - structure.position[0], y - structure.position[1])
                if structure.inner_radius <= distance <= structure.outer_radius:
                    epsilon, mu, sigma_base = structure.material.get_sample()
                    break
        # Calculate PML conductivity based on the UNDERLYING material
        # This is crucial for proper absorption without reflection
        pml_conductivity = 0.0
        if dx is not None:
            eps_avg = epsilon  # Use the actual permittivity at this point
            # Apply all PML boundaries
            for boundary in self.boundaries:
                pml_conductivity += boundary.get_conductivity(x, y, dx=dx, dt=dt, eps_avg=eps_avg)
        
        # Return with the permittivity of the underlying structure plus PML conductivity
        return [epsilon, mu, sigma_base + pml_conductivity]

    def rasterize(self, resolution, grid_type="regular", **kwargs):
        """Rasterize the design into a mesh grid using beamz.simulation.meshing.

        Args:
            resolution: Grid resolution (float or tuple depending on grid type).
            grid_type: "regular" (2D), "regular3d"/"3d", "auto", or a grid class.
            **kwargs: Additional keyword arguments forwarded to the grid constructor
                (e.g., resolution_z for 3D grids, auto_select flags, etc.).

        Returns:
            A mesh grid instance (RegularGrid, RegularGrid3D, or custom grid).
        """
        from beamz.simulation import meshing

        grid_cls = None

        if isinstance(grid_type, str):
            gt = grid_type.lower()
            if gt in {"regular", "regulargrid", "2d"}:
                grid_cls = meshing.RegularGrid
            elif gt in {"regular3d", "3d"}:
                grid_cls = meshing.RegularGrid3D
            elif gt in {"auto", "auto-select", "autoselect"}:
                return meshing.create_mesh(self, resolution, **kwargs)
            else:
                raise ValueError(f"Unknown grid_type '{grid_type}'. Expected 'regular', 'regular3d', or 'auto'.")
        elif isinstance(grid_type, type):
            grid_cls = grid_type
        else:
            raise TypeError("grid_type must be a string or a mesh grid class")

        if grid_cls is meshing.RegularGrid3D:
            # Expect resolution_xy (required) and optional resolution_z in kwargs
            resolution_xy = resolution
            resolution_z = kwargs.pop("resolution_z", None)
            return grid_cls(self, resolution_xy=resolution_xy, resolution_z=resolution_z)

        # Default path: RegularGrid or custom grid class with (design, resolution)
        return grid_cls(self, resolution, **kwargs)

    def get_material_grids(self, resolution):
        """Get rasterized material property arrays at specified resolution.
        Caches the grids in the Design object to avoid recomputation and maintain single source of truth.
        
        Returns tuple of (permittivity, conductivity, permeability) numpy arrays as references.
        """
        # Cache the grid at this resolution to maintain ownership in Design
        if not hasattr(self, '_cached_grid') or self._cached_resolution != resolution:
            self._cached_grid = self.rasterize(resolution)
            self._cached_resolution = resolution
        
        return (self._cached_grid.permittivity, self._cached_grid.conductivity, self._cached_grid.permeability)


    ## GDS Import/Export & Copy Methods ==================================================

    def import_gds(gds_file: str, default_depth=1e-6):
        """Import a GDS file and return polygon and layer data.
        
        Args:
            gds_file (str): Path to the GDS file
            default_depth (float): Default depth/thickness for imported structures in meters
        """
        pass

    def export_gds(self, output_file):
        """Export a BEAMZ design (including only the structures, not sources or monitors) to a GDS file.
        
        For 3D designs, structures with the same material that touch (in 3D) will be placed in the same layer.
        """
        pass
   
    def show(self, unify_structures=True):
        """Display the design visually using 2D matplotlib or 3D plotly (delegated to beamz.viz)."""
        return viz.show_design(self, unify_structures)

    def copy(self):
        """Create a deep copy of the design."""
        # Get background material from the first structure (background rectangle)
        background_material = None
        if self.structures and hasattr(self.structures[0], 'material'):
            background_material = self.structures[0].material
        
        # Create new design without auto_pml since we'll copy boundaries manually
        new_design = Design(width=self.width, height=self.height, material=background_material, 
                           auto_pml=False)
        
        # Clear the automatically created structures and rebuild with deep copies
        new_design.structures = []
        new_design.sources = []
        new_design.monitors = []
        new_design.boundaries = []
        
        # Deep copy all structures in the correct order
        for structure in self.structures:
            if hasattr(structure, 'copy'):
                copied_structure = structure.copy()
                # Deep copy materials if they have a copy method (like CustomMaterial)
                if hasattr(copied_structure, 'material') and copied_structure.material is not None:
                    if hasattr(copied_structure.material, 'copy'): copied_structure.material = copied_structure.material.copy()
                # Update design reference for sources and monitors
                if hasattr(copied_structure, 'design'): copied_structure.design = new_design
                new_design.structures.append(copied_structure)
                # Also add to appropriate lists
                if isinstance(copied_structure, (ModeSource, GaussianSource)):
                    new_design.sources.append(copied_structure)
                elif hasattr(structure, '__class__') and 'Monitor' in structure.__class__.__name__:
                    new_design.monitors.append(copied_structure)
            else:
                # Fallback for structures without copy method
                new_design.structures.append(structure)
        
        # Deep copy boundaries
        for boundary in self.boundaries:
            if hasattr(boundary, 'copy'): new_design.boundaries.append(boundary.copy())
            else: new_design.boundaries.append(boundary)
        
        # Copy other attributes
        new_design.is_3d = self.is_3d
        new_design.depth = self.depth
        new_design.border_color = self.border_color
        new_design.time = self.time
        new_design.layers = self.layers.copy() if hasattr(self, 'layers') else {}
        
        return new_design
