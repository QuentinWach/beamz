"""Boolean merge helpers for design structure lists."""

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from beamz.design.structures import Polygon, Ring


def _material_key(material):
    """Return a hashable key for a material based on its physical properties."""
    return (
        getattr(material, "permittivity", None),
        getattr(material, "permeability", None),
        getattr(material, "conductivity", None),
    )


def _to_shapely(structure):
    """Convert a BEAMZ structure to a Shapely polygon, or None if not possible."""
    if hasattr(structure, "interiors") and structure.interiors:
        valid_interiors = [list(i_path) for i_path in structure.interiors if i_path]
        if structure.vertices and valid_interiors:
            poly = ShapelyPolygon(shell=structure.vertices, holes=valid_interiors)
        elif structure.vertices:
            poly = ShapelyPolygon(shell=structure.vertices)
        else:
            return None
    elif hasattr(structure, "vertices") and structure.vertices:
        poly = ShapelyPolygon(shell=structure.vertices)
    else:
        return None
    return poly if poly.is_valid else None


def _group_by_material(structures):
    """Group structures by material key and convert to Shapely polygons."""
    material_groups = {}
    structures_to_remove = []
    for structure in structures:
        material = getattr(structure, "material", None)
        if not material:
            continue
        key = _material_key(material)
        shapely_poly = _to_shapely(structure)
        if shapely_poly is None:
            continue
        material_groups.setdefault(key, []).append((structure, shapely_poly))
        structures_to_remove.append(structure)
    return material_groups, structures_to_remove


def _find_rings_to_preserve(material_groups, structures_to_remove):
    """Identify Ring structures that should not be merged (they have interiors)."""
    rings_to_preserve = []
    for structure_group in material_groups.values():
        if len(structure_group) <= 1:
            continue
        for _idx, (struct, _shapely) in enumerate(structure_group):
            if isinstance(struct, Ring):
                rings_to_preserve.append(struct)
                if struct in structures_to_remove:
                    structures_to_remove.remove(struct)
    return rings_to_preserve


def _shapely_to_polygons(merged, material, first_structure):
    """Convert a merged Shapely geometry back to BEAMZ Polygon(s)."""
    depth = getattr(first_structure, "depth", 0)
    z = getattr(first_structure, "z", 0)

    def _geom_to_polygon(geom):
        exterior_coords = list(geom.exterior.coords[:-1])
        if not exterior_coords or len(exterior_coords) < 3:
            return None
        interior_coords_lists = [
            list(interior.coords[:-1]) for interior in geom.interiors
        ]
        return Polygon(
            vertices=exterior_coords,
            interiors=interior_coords_lists,
            material=material,
            depth=depth,
            z=z,
        )

    if merged.geom_type == "Polygon":
        poly = _geom_to_polygon(merged)
        return [poly] if poly else None
    if merged.geom_type == "MultiPolygon":
        polys = []
        for geom in merged.geoms:
            poly = _geom_to_polygon(geom)
            if poly is None:
                return None
            polys.append(poly)
        return polys
    return None


def _merge_groups(material_groups, rings_to_preserve, structures_to_remove):
    """Merge each material group using Shapely union."""
    new_structures = []
    for material_key, structure_group in material_groups.items():
        filtered_group = [s for s in structure_group if s[0] not in rings_to_preserve]
        if len(filtered_group) <= 1:
            new_structures.extend([s[0] for s in filtered_group])
            for s in filtered_group:
                if s[0] in structures_to_remove:
                    structures_to_remove.remove(s[0])
            continue

        shapely_polygons = [p[1] for p in filtered_group]
        material = filtered_group[0][0].material
        merged = unary_union(shapely_polygons)
        result = _shapely_to_polygons(merged, material, filtered_group[0][0])

        if result is not None:
            new_structures.extend(result)
        else:
            new_structures.extend([s[0] for s in structure_group])
            for s_tuple in structure_group:
                if s_tuple[0] in structures_to_remove:
                    structures_to_remove.remove(s_tuple[0])

    return new_structures, structures_to_remove


def _rebuild_structure_list(
    original, structures_to_remove, new_structures, material_groups
):
    """Rebuild the structure list, replacing merged groups at their original position."""
    material_replacements = {}
    for new_struct in new_structures:
        if not (hasattr(new_struct, "material") and new_struct.material):
            continue
        key = _material_key(new_struct.material)
        for mat_key, group in material_groups.items():
            if len(group) > 1 and mat_key == key:
                material_replacements.setdefault(mat_key, []).append(new_struct)
                break

    rebuilt, used = [], set()
    for structure in original:
        if structure in structures_to_remove:
            if not (hasattr(structure, "material") and structure.material):
                continue
            key = _material_key(structure.material)
            if key not in used and key in material_replacements:
                rebuilt.extend(material_replacements[key])
                used.add(key)
        else:
            rebuilt.append(structure)
    return rebuilt
