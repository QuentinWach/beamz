from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

from beamz._cache_tokens import cache_token
from beamz.const import µm
from beamz.design.materials import Material, MaterialProtocol
from beamz.design.structures import (
    Polygon,
    Ring,
)

_DEFAULT_DOMAIN_SIZE = object()


def _material_key(material):
    """Return a hashable key for a material based on its physical properties."""
    return (
        getattr(material, "permittivity", None),
        getattr(material, "permeability", None),
        getattr(material, "conductivity", None),
    )


def _merge_group_key(structure):
    """Return a hashable key for planar-union compatibility.

    Structures can only be merged safely when both their material properties and
    their z/depth placement match; the shapely union ignores z entirely.
    """
    return (
        _material_key(getattr(structure, "material", None)),
        getattr(structure, "z", None),
        getattr(structure, "depth", None),
        getattr(structure, "sidewall_angle", None),
        getattr(structure, "width_to_z", None),
    )


def _to_shapely(structure):
    """Convert a beamz structure to a Shapely polygon, or None if not possible."""
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
    """Group structures by material key and convert to Shapely polygons.

    Returns (material_groups, structures_to_remove) where material_groups maps
    material_key -> list of (structure, shapely_polygon) tuples.
    """
    material_groups = {}
    structures_to_remove = []
    for structure in structures:
        material = getattr(structure, "material", None)
        if not material:
            continue
        key = _merge_group_key(structure)
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
    """Convert a merged Shapely geometry back to beamz Polygon(s).

    Returns a list of Polygon objects, or None if conversion fails.
    """
    depth = getattr(first_structure, "depth", 0)
    z = getattr(first_structure, "z", 0)
    sidewall_angle = getattr(first_structure, "sidewall_angle", 0.0)
    width_to_z = getattr(first_structure, "width_to_z", 0.0)

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
            sidewall_angle=sidewall_angle,
            width_to_z=width_to_z,
        )

    if merged.geom_type == "Polygon":
        poly = _geom_to_polygon(merged)
        return [poly] if poly else None
    elif merged.geom_type == "MultiPolygon":
        polys = []
        for geom in merged.geoms:
            poly = _geom_to_polygon(geom)
            if poly is None:
                return None
            polys.append(poly)
        return polys
    return None


def _merge_groups(material_groups, rings_to_preserve, structures_to_remove):
    """Merge each material group using Shapely union.

    Returns (new_structures, updated structures_to_remove).
    """
    new_structures = []
    for structure_group in material_groups.values():
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
            # Fallback: keep originals
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
        key = _merge_group_key(new_struct)
        for mat_key, group in material_groups.items():
            if len(group) > 1 and mat_key == key:
                material_replacements.setdefault(mat_key, []).append(new_struct)
                break

    rebuilt, used = [], set()
    for structure in original:
        if structure in structures_to_remove:
            if not (hasattr(structure, "material") and structure.material):
                continue
            key = _merge_group_key(structure)
            if key not in used and key in material_replacements:
                rebuilt.extend(material_replacements[key])
                used.add(key)
        else:
            rebuilt.append(structure)
    return rebuilt


@dataclass(frozen=True, slots=True, eq=False, init=False)
class Design:
    """Immutable geometry specification with a functional convenience facade.

    ``background`` is the sole background material. ``structures`` contains only
    user geometry, in painter's order. Python's augmented assignment still gives
    a friendly workflow because ``design += structure`` rebinds ``design`` to the
    new value returned by :meth:`with_structure`.

    Parameters
    ----------
    width : float, default=4 um
        Domain extent along x in metres; must be finite and positive.
    height : float, default=4 um
        Domain extent along y in metres; must be finite and positive.
    depth : float, default=0
        Domain extent along z in metres. Zero selects a 2D design.
    material : Material, optional
        Backward-compatible name for the background material.
    background : Material, optional
        Material filling cells not covered by a structure. Do not pass together
        with a different ``material`` value.
    structures : iterable, optional
        Geometry primitives in painter's order; later structures overwrite
        earlier structures during rasterization.

    Raises
    ------
    ValueError
        If an extent is invalid or both background aliases disagree.
    TypeError
        If ``structures`` contains unsupported mutable geometry.

    Notes
    -----
    Public geometry coordinates use ``(x, y, z)`` order and SI units. A design
    is immutable: methods such as ``with_structure`` return a new value.

    Examples
    --------
    >>> import beamz as bz
    >>> design = bz.Design(
    ...     width=6 * bz.um,
    ...     height=3 * bz.um,
    ...     background=bz.Material(permittivity=1.44**2),
    ... )
    >>> design += bz.Rectangle(
    ...     position=(1 * bz.um, 1.25 * bz.um),
    ...     width=4 * bz.um,
    ...     height=0.5 * bz.um,
    ...     material=bz.Material(permittivity=3.48**2),
    ... )
    """

    width: float
    height: float
    depth: float
    background: Any
    structures: tuple[Any, ...]
    _centered_coordinates: bool = field(
        default=False, repr=False, compare=False, metadata={"beamz_cache": False}
    )

    def __init__(
        self,
        width: float | object = _DEFAULT_DOMAIN_SIZE,
        height: float | object = _DEFAULT_DOMAIN_SIZE,
        depth: float | object = _DEFAULT_DOMAIN_SIZE,
        material: Any = None,
        background: Any = None,
        structures=(),
    ):
        """Create a design domain with specified dimensions and background material."""
        width_is_default = width is _DEFAULT_DOMAIN_SIZE
        height_is_default = height is _DEFAULT_DOMAIN_SIZE
        depth_is_default = depth is _DEFAULT_DOMAIN_SIZE
        background_arg = background
        if width_is_default:
            width = 4 * µm
        if height_is_default:
            height = 4 * µm
        if depth_is_default:
            depth = 0
        explicit_size = not (
            width_is_default and height_is_default and depth_is_default
        )
        if (
            background_arg is not None
            and material is not None
            and background_arg is not material
        ):
            raise ValueError("Pass only one of background=... or material=....")
        if background_arg is not None:
            material = background_arg
        if material is None:
            material = Material(permittivity=1.0, permeability=1.0, conductivity=0.0)
        if not isinstance(material, MaterialProtocol):
            raise TypeError("Design background must satisfy MaterialProtocol.")

        width = float(cast(float, width))
        height = float(cast(float, height))
        depth = float(cast(float, depth))
        if not np.isfinite(width) or width <= 0:
            raise ValueError("Design width must be finite and positive.")
        if not np.isfinite(height) or height <= 0:
            raise ValueError("Design height must be finite and positive.")
        if not np.isfinite(depth) or depth < 0:
            raise ValueError("Design depth must be finite and non-negative.")

        normalized_structures = tuple(structures)
        for structure in normalized_structures:
            self._validate_structure(structure)

        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "background", material)
        object.__setattr__(self, "structures", normalized_structures)
        object.__setattr__(
            self,
            "_centered_coordinates",
            bool(background_arg is not None and not explicit_size),
        )

    @staticmethod
    def _validate_structure(structure):
        from beamz.design.structures import Structure

        if not isinstance(structure, Structure):
            raise TypeError(
                "Design only accepts immutable geometry specs; pass sources and "
                "monitors to Simulation instead."
            )

    @property
    def is_3d(self):
        """Return whether the design has a positive z extent."""
        return self.depth > 0

    def __str__(self):
        return f"Design with {len(self.structures)} structures ({'3D' if self.is_3d else '2D'})"

    def canonical_spec(self):
        """Return the immutable values defining physical cache identity."""
        return self.width, self.height, self.depth, self.background, self.structures

    def __eq__(self, other):
        if not isinstance(other, Design):
            return NotImplemented
        return cache_token(self.canonical_spec()) == cache_token(other.canonical_spec())

    def __hash__(self):
        return hash(cache_token(self.canonical_spec()))

    def __iadd__(self, structure):
        return self.with_structure(structure)

    def with_structure(self, structure):
        """Return a design with one geometry appended in painter's order.

        Parameters
        ----------
        structure : Polygon or Box
            Immutable geometry to place above existing structures.
        """
        self._validate_structure(structure)
        return self.updated_copy(structures=(*self.structures, structure))

    def updated_copy(self, **changes):
        """Return a new design with the requested fields replaced.

        Parameters
        ----------
        **changes
            Any of ``width``, ``height``, ``depth``, ``background``, or
            ``structures`` and their replacement values.
        """
        unknown = set(changes) - {
            "width",
            "height",
            "depth",
            "background",
            "structures",
        }
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"Unknown Design field(s): {names}.")
        values = {
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "background": self.background,
            "structures": self.structures,
        }
        values.update(changes)
        result = type(self)(**values)
        if not ({"width", "height", "depth"} & changes.keys()):
            object.__setattr__(
                result, "_centered_coordinates", self._centered_coordinates
            )
        return result

    def unified_polygons(self):
        """Return a design with compatible overlapping polygons merged."""
        material_groups, structures_to_remove = _group_by_material(self.structures)
        rings_to_preserve = _find_rings_to_preserve(
            material_groups, structures_to_remove
        )
        new_structures, structures_to_remove = _merge_groups(
            material_groups, rings_to_preserve, structures_to_remove
        )
        structures = _rebuild_structure_list(
            self.structures, structures_to_remove, new_structures, material_groups
        )
        return self.updated_copy(structures=structures)

    def rasterize(
        self,
        resolution: float,
        grid_type: str = "auto",
        force_recompute: bool = False,
        progress: bool = False,
        **kwargs,
    ):
        """Rasterize this design into the solver's immutable material grid."""
        from beamz.design.raster.integration import _rasterize_design

        return _rasterize_design(
            self,
            resolution,
            grid_type=grid_type,
            force_recompute=force_recompute,
            progress=progress,
            **kwargs,
        )

    def discretize(self, resolution, **kwargs):
        """Return design-owned material grids at the requested resolution.

        Parameters
        ----------
        resolution : float
            In-plane cell spacing in metres.
        **kwargs
            Additional options forwarded to :func:`build_material_grid`.
        """
        from beamz.design.discretization import build_material_grid

        return build_material_grid(self, resolution, **kwargs)

    def plot(self, **kwargs):
        """Plot the design layout using the matplotlib backend.

        Parameters
        ----------
        **kwargs
            Plotting options forwarded to the design plotting backend. ``show``
            defaults to false.
        """
        from beamz.analysis.plotting import plot_design

        kwargs.setdefault("show", False)
        return plot_design(self, **kwargs)

    def show(self, **kwargs):
        """Display the design layout using the matplotlib backend.

        Parameters
        ----------
        **kwargs
            Plotting options forwarded to :meth:`plot`. ``show`` defaults to
            true.
        """
        kwargs.setdefault("show", True)
        return self.plot(**kwargs)

    def copy(self):
        """Return this immutable specification."""
        return self

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        del memo
        return self
