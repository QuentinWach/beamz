"""Lower immutable boundary specifications into grid-aligned numerical arrays."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, MU_0
from beamz.design.discretization import MaterialGrid
from beamz.devices.boundaries import (
    PEC,
    PML,
    Absorber,
    edges_for_dimension,
    normalize_boundaries,
)
from beamz.lattice import component_axis_offsets_3d

_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _rectilinear_geometry(fields):
    geometry = getattr(fields, "geometry", None)
    if geometry is None:
        return None
    axes = ("x", "y", "z") if fields.permittivity.ndim == 3 else ("x", "y")
    return geometry if geometry.metric_kind_for(axes) != "isotropic_uniform" else None


@dataclass(frozen=True, slots=True)
class CpmlDerivativeSpec:
    """One derivative profile that Devices lowers onto a Yee component support."""

    name: str
    target_component: str
    derivative_axis: str


CPML_3D_H_DERIVATIVES = (
    CpmlDerivativeSpec("Hxy", "Hx", "y"),
    CpmlDerivativeSpec("Hxz", "Hx", "z"),
    CpmlDerivativeSpec("Hyz", "Hy", "z"),
    CpmlDerivativeSpec("Hyx", "Hy", "x"),
    CpmlDerivativeSpec("Hzx", "Hz", "x"),
    CpmlDerivativeSpec("Hzy", "Hz", "y"),
)

CPML_3D_E_DERIVATIVES = (
    CpmlDerivativeSpec("Exy", "Ex", "y"),
    CpmlDerivativeSpec("Exz", "Ex", "z"),
    CpmlDerivativeSpec("Eyz", "Ey", "z"),
    CpmlDerivativeSpec("Eyx", "Ey", "x"),
    CpmlDerivativeSpec("Ezx", "Ez", "x"),
    CpmlDerivativeSpec("Ezy", "Ez", "y"),
)


@dataclass(frozen=True, slots=True)
class BoundaryData:
    """Numerical material, absorber, and PEC data produced by boundary lowering."""

    permittivity: Any
    conductivity: Any
    permeability: Any
    yee_materials: Mapping[str, Any]
    profiles: Mapping[str, Any] | None
    masks: Mapping[str, Any]
    metallic_edges: frozenset[str]


class _ComponentSupport:
    """Shape-only component view used while compiling staggered profiles."""

    def __init__(self, shape) -> None:
        self.shape = tuple(int(value) for value in shape)


class _BoundaryGrid:
    """Mutable material workspace without allocating solver field arrays."""

    def __init__(self, material_grid, component_shapes) -> None:
        self.permittivity = jnp.asarray(material_grid.permittivity)
        self.conductivity = jnp.asarray(material_grid.conductivity)
        self.permeability = jnp.asarray(material_grid.permeability)
        self.polarization_2d = material_grid.polarization or "tm"
        # Preserve the established uniform profile algebra exactly; stretched grids
        # opt into physical-coordinate grading.
        self.geometry = (
            material_grid.grid
            if material_grid.metric_kind != "isotropic_uniform"
            else None
        )
        self.yee_materials = {
            name: jnp.asarray(value)
            for name, value in material_grid.yee_materials.items()
        }
        for component, shape in component_shapes.items():
            setattr(self, component, _ComponentSupport(shape))


def resolve_metallic_edges(boundaries, is_3d: bool) -> frozenset[str]:
    """Apply boundary precedence and return walls that remain metallic."""
    metallic: set[str] = set()
    for boundary in normalize_boundaries(boundaries):
        edges = edges_for_dimension(boundary.edges, bool(is_3d))
        if isinstance(boundary, PEC):
            metallic.update(edges)
        else:
            metallic.difference_update(edges)
    return frozenset(metallic)


def compile_metallic_masks(
    component_shapes, material_shape, boundaries, *, polarization_2d: str = "tm"
) -> dict[str, jnp.ndarray]:
    """Compile boundary specifications into component-aligned PEC masks."""
    masks = {
        name: np.zeros(tuple(component_shapes[name]), dtype=bool)
        for name in _COMPONENTS
    }
    is_3d = len(material_shape) == 3
    metallic_edges = resolve_metallic_edges(boundaries, is_3d)
    wall_specs = (
        {
            "front": (0, 0, ("Ex", "Ey", "Hz")),
            "back": (0, -1, ("Ex", "Ey", "Hz")),
            "bottom": (1, 0, ("Ex", "Ez", "Hy")),
            "top": (1, -1, ("Ex", "Ez", "Hy")),
            "left": (2, 0, ("Ey", "Ez", "Hx")),
            "right": (2, -1, ("Ey", "Ez", "Hx")),
        }
        if is_3d
        else (
            {
                "bottom": (0, 0, ("Ez", "Hy")),
                "top": (0, -1, ("Ez", "Hy")),
                "left": (1, 0, ("Ez", "Hx")),
                "right": (1, -1, ("Ez", "Hx")),
            }
            if polarization_2d == "tm"
            else {
                "bottom": (0, 0, ("Ex",)),
                "top": (0, -1, ("Ex",)),
                "left": (1, 0, ("Ey",)),
                "right": (1, -1, ("Ey",)),
            }
        )
    )
    for edge in metallic_edges:
        axis, index, components = wall_specs[edge]
        selection: list[slice | int] = [slice(None)] * len(material_shape)
        selection[axis] = index
        for component in components:
            masks[component][tuple(selection)] = True
    return {name: jnp.asarray(mask) for name, mask in masks.items()}


def _merge_profiles(lhs, rhs, *, key: str | None = None):
    """Merge contributions using the physical operation for each coefficient family."""
    if isinstance(lhs, Mapping) and isinstance(rhs, Mapping):
        merged = dict(lhs)
        for child_key, child_value in rhs.items():
            merged[child_key] = (
                _merge_profiles(merged[child_key], child_value, key=child_key)
                if child_key in merged
                else child_value
            )
        return merged
    key = "" if key is None else str(key)
    if key == "formulation":
        if str(lhs).lower() != str(rhs).lower():
            raise ValueError(
                "Cannot merge boundary profiles with different formulations: "
                f"{lhs!r} != {rhs!r}."
            )
        return str(lhs).lower()
    lhs_arr, rhs_arr = jnp.asarray(lhs), jnp.asarray(rhs)
    if key == "mask":
        return lhs_arr.astype(bool) | rhs_arr.astype(bool)
    if "kappa" in key or "alpha" in key:
        return jnp.maximum(lhs_arr, rhs_arr)
    return lhs_arr + rhs_arr


class _AbsorberCompiler:
    """Lower one immutable Device specification into grid-aware solver arrays."""

    def __init__(self, spec: PML | Absorber):
        self.spec = spec

    def _get_edges_for_dimensionality(self, is_3d):
        return list(edges_for_dimension(self.spec.edges, is_3d))

    def compile_absorber_regions(self, fields, domain_size, resolution, dt):
        """Create grid-aligned masks and graded conductivity profiles."""
        # Resolve geometry-dependent arrays now so runtime code stays branch-free and
        # shape-static.
        profile = self._resolved_profile_boundary(fields, resolution, dt)

        if self.spec.formulation == "cpml":
            if fields.permittivity.ndim == 3:
                out = profile._create_cpml_profiles_3d(fields, domain_size)
            else:
                out = profile._create_cpml_profiles_2d(fields, domain_size)
            self._extend_cpml_materials_to_absorber(fields, out)
            return out

        out = profile._create_sponge_profiles(fields, domain_size)
        self._warn_if_material_not_extruded(fields, out)
        return out

    def _resolved_profile_boundary(self, fields, resolution, dt):
        # Build this profile on its exact staggered support to avoid interpolation
        # inside timestep kernels.
        sigma_max, alpha_max = self._resolved_profile_params(fields, resolution, dt)
        changes: dict[str, float | None] = {"sigma_max": sigma_max}
        if isinstance(self.spec, PML):
            changes["alpha_max"] = alpha_max
        return type(self)(self.spec.updated_copy(**changes))

    def _resolved_profile_params(self, fields, resolution, dt):
        sigma_max = self.spec.sigma_max
        if sigma_max is None:
            eta = np.sqrt(MU_0 / (EPS_0 * 1.0))
            thickness = max(float(self.spec.thickness), float(resolution))
            sigma_max = (
                -(self.spec.m + 1)
                * np.log(max(self.spec.target_reflection, 1e-16))
                / (2.0 * eta * thickness)
            )
        alpha_max = self.spec.alpha_max
        if self.spec.formulation == "cpml" and alpha_max is None:
            # Convert a conservative normalized CFS alpha into the solver's
            # conductivity-like units so default CPML keeps a nonzero CFS shift.
            alpha_normalized = self.spec._DEFAULT_CPML_ALPHA_NORMALIZED
            if getattr(fields.permittivity, "ndim", 0) == 3:
                alpha_normalized = self.spec._DEFAULT_3D_CPML_ALPHA_NORMALIZED
            alpha_max = 2.0 * EPS_0 * alpha_normalized / max(float(dt), 1e-30)
        return float(sigma_max), None if alpha_max is None else float(alpha_max)

    def _pml_material_variation_edges(self, fields, pml_data):
        """Return edges where material changes along the PML normal."""
        # Map storage axes to named physical walls once so every boundary path uses
        # the same convention.
        material = np.asarray(fields.permittivity)
        if material.size == 0:
            return []
        bad_edges = []
        for edge, axis, side, count in self._pml_material_edge_counts(
            material, pml_data
        ):
            if count <= 0 or count >= material.shape[axis]:
                continue
            if self._pml_edge_material_varies(material, axis, side, count):
                bad_edges.append(edge)
        return bad_edges

    def _pml_material_edge_counts(self, material, pml_data):
        """Return active absorber-cell counts for material extrusion checks."""
        # 1. Map public wall names to storage axes for the current dimensionality; 2D
        # planes omit one physical axis and therefore need a plane-specific mapping.
        mat = np.asarray(material)
        axis_map_3d = {"z": 0, "y": 1, "x": 2}
        axis_map_2d = {"y": 0, "x": 1}
        edge_axes = {
            "left": ("x", "low"),
            "right": ("x", "high"),
            "bottom": ("y", "low"),
            "top": ("y", "high"),
            "front": ("z", "low"),
            "back": ("z", "high"),
        }
        is_3d = mat.ndim == 3
        axis_map = axis_map_3d if is_3d else axis_map_2d
        # 2. For each active wall, collapse its sigma grid transversely to a 1D activity
        # profile and count contiguous absorber cells inward from that wall.
        edge_counts = []
        for edge in self._get_edges_for_dimensionality(is_3d):
            axis_name, side = edge_axes.get(edge, (None, None))
            if axis_name not in axis_map:
                continue
            sigma = pml_data.get(f"sigma_{axis_name}")
            if sigma is None:
                continue
            axis = axis_map[axis_name]
            active_1d = np.any(
                np.asarray(sigma) > 0.0,
                axis=tuple(i for i in range(material.ndim) if i != axis),
            )
            if side == "low":
                count = int(np.argmax(~active_1d)) if np.any(~active_1d) else 0
            else:
                rev = active_1d[::-1]
                count = int(np.argmax(~rev)) if np.any(~rev) else 0
            edge_counts.append((edge, axis, side, count))
        return edge_counts

    def _warn_if_material_not_extruded(self, fields, pml_data):
        """Warn when material changes along the PML normal inside the absorber."""
        # Diagnose impedance changes inside the absorber because they create avoidable reflection.
        bad_edges = self._pml_material_variation_edges(fields, pml_data)
        if bad_edges:
            warnings.warn(
                f"{type(self.spec).__name__} material varies along the absorber normal on edges "
                f"{bad_edges}. For lower reflection, extrude the boundary material "
                "profile through the PML or keep geometry clear of the absorber.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _extend_cpml_materials_to_absorber(self, fields, pml_data):
        """Copy first-interior material values through active CPML slabs."""

        # Iterate in deterministic axis order because profile and component tuples
        # must remain aligned.
        for attr in ("permittivity", "conductivity", "permeability"):
            source = getattr(fields, attr)
            source_arr = np.asarray(source)
            if source_arr.ndim == 0:
                continue
            material = np.array(source_arr, copy=True)
            for _edge, axis, side, count in self._pml_material_edge_counts(
                material, pml_data
            ):
                if count <= 0 or count >= material.shape[axis]:
                    continue
                if side == "low":
                    dst_sel = [slice(None)] * material.ndim
                    dst_sel[axis] = slice(0, count)
                    ref_sel = [slice(None)] * material.ndim
                    ref_sel[axis] = count
                else:
                    dst_sel = [slice(None)] * material.ndim
                    dst_sel[axis] = slice(material.shape[axis] - count, None)
                    ref_sel = [slice(None)] * material.ndim
                    ref_sel[axis] = material.shape[axis] - count - 1
                material[tuple(dst_sel)] = np.expand_dims(
                    material[tuple(ref_sel)], axis=axis
                )
            setattr(fields, attr, jnp.asarray(material, dtype=source_arr.dtype))
        for name, source in getattr(fields, "yee_materials", {}).items():
            source_arr = np.asarray(source)
            material = np.array(source_arr, copy=True)
            for _edge, axis, side, count in self._pml_material_edge_counts(
                np.asarray(fields.permittivity), pml_data
            ):
                if count <= 0 or count >= material.shape[axis]:
                    continue
                if side == "low":
                    destination = [slice(None)] * material.ndim
                    destination[axis] = slice(0, count)
                    reference = [slice(None)] * material.ndim
                    reference[axis] = count
                else:
                    destination = [slice(None)] * material.ndim
                    destination[axis] = slice(material.shape[axis] - count, None)
                    reference = [slice(None)] * material.ndim
                    reference[axis] = material.shape[axis] - count - 1
                material[tuple(destination)] = np.expand_dims(
                    material[tuple(reference)], axis=axis
                )
            fields.yee_materials[name] = jnp.asarray(material, dtype=source_arr.dtype)

    @staticmethod
    def _pml_edge_material_varies(material, axis: int, side: str, count: int) -> bool:
        # Map storage axes to named physical walls once so every boundary path uses
        # the same convention.
        mat = np.asarray(material)
        if side == "low":
            slab_sel = [slice(None)] * mat.ndim
            slab_sel[axis] = slice(0, count)
            ref_sel: list[Any] = [slice(None)] * mat.ndim
            ref_sel[axis] = count
        else:
            slab_sel = [slice(None)] * mat.ndim
            slab_sel[axis] = slice(mat.shape[axis] - count, mat.shape[axis])
            ref_sel = [slice(None)] * mat.ndim
            ref_sel[axis] = mat.shape[axis] - count - 1
        slab = mat[tuple(slab_sel)]
        ref = np.expand_dims(mat[tuple(ref_sel)], axis=axis)
        return bool(np.any(np.abs(slab - ref) > 1e-9))

    def _compute_1d_profile(self, coords, length, low_active, high_active):
        """Compute 1D graded-sigma profile along an axis."""
        # Build this profile on its exact staggered support to avoid interpolation
        # inside timestep kernels.
        sigma = jnp.zeros_like(coords)
        thickness = self.spec.thickness
        if self.spec.sigma_max is None:
            raise ValueError("PML profile parameters must be resolved before sampling.")

        if low_active:
            dist = jnp.clip(thickness - coords, 0.0, None)
            sigma = sigma + self.spec.sigma_max * (dist / thickness) ** self.spec.m

        if high_active:
            dist = jnp.clip(coords - (length - thickness), 0.0, None)
            sigma = sigma + self.spec.sigma_max * (dist / thickness) ** self.spec.m

        return sigma

    def _compute_1d_cpml_profile(self, coords, length, low_active, high_active):
        """Compute 1D CPML sigma/kappa/alpha profiles."""

        # Build this profile on its exact staggered support to avoid interpolation
        # inside timestep kernels.
        sigma = jnp.zeros_like(coords)
        kappa = jnp.ones_like(coords)
        alpha = jnp.zeros_like(coords)
        thickness = max(float(self.spec.thickness), 1e-30)
        if self.spec.sigma_max is None or self.spec.alpha_max is None:
            raise ValueError(
                "CPML profile parameters must be resolved before sampling."
            )
        sigma_max = self.spec.sigma_max
        alpha_max = self.spec.alpha_max

        def apply_side(dist):
            # Grade sigma/kappa inward while alpha decreases toward the interior interface.
            graded = jnp.clip(dist / thickness, 0.0, 1.0)
            side_sigma = sigma_max * graded**self.spec.m
            side_kappa = 1.0 + (self.spec.kappa_max - 1.0) * graded**self.spec.m
            side_alpha = alpha_max * (1.0 - graded)
            mask = dist > 0
            return mask, side_sigma, side_kappa, side_alpha

        if low_active:
            dist = jnp.clip(thickness - coords, 0.0, None)
            mask, side_sigma, side_kappa, side_alpha = apply_side(dist)
            sigma = sigma + side_sigma
            kappa = jnp.where(mask, side_kappa, kappa)
            alpha = jnp.where(mask, side_alpha, alpha)

        if high_active:
            dist = jnp.clip(coords - (length - thickness), 0.0, None)
            mask, side_sigma, side_kappa, side_alpha = apply_side(dist)
            sigma = sigma + side_sigma
            kappa = jnp.where(mask, side_kappa, kappa)
            alpha = jnp.where(mask, side_alpha, alpha)

        return sigma, kappa, alpha

    def _compute_fdtdx_staggered_profile_1d(
        self,
        total_samples,
        spacing,
        low_active,
        high_active,
        *,
        sample_kind,
        domain_cells=None,
        sigma_order=None,
        kappa_order=None,
        alpha_order=None,
        sample_coordinates=None,
        domain_bounds=None,
    ):
        """Compute a 1D CPML profile using FDTDX-style discrete Yee offsets."""

        # 1. Start from vacuum CPML values and convert the physical thickness to an integer
        # cell count used by FDTDX's discrete grading convention.
        sigma = jnp.zeros((int(total_samples),), dtype=jnp.float32)
        kappa = jnp.ones((int(total_samples),), dtype=jnp.float32)
        alpha = jnp.zeros((int(total_samples),), dtype=jnp.float32)

        pml_cells = max(
            int(round(float(self.spec.thickness) / max(float(spacing), 1e-30))), 1
        )
        sigma_order = float(self.spec.m if sigma_order is None else sigma_order)
        kappa_order = float(self.spec.m if kappa_order is None else kappa_order)
        alpha_order = float(1.0 if alpha_order is None else alpha_order)

        def apply_u(u):
            # Evaluate normalized polynomial profiles once for either low or high side distance.
            dtype = sigma.dtype
            u = jnp.clip(jnp.asarray(u, dtype=dtype), 0.0, 1.0)
            side_sigma = jnp.asarray(self.spec.sigma_max, dtype=dtype) * jnp.power(
                u, sigma_order
            )
            side_kappa = jnp.asarray(1.0, dtype=dtype) + (
                jnp.asarray(self.spec.kappa_max, dtype=dtype)
                - jnp.asarray(1.0, dtype=dtype)
            ) * jnp.power(u, kappa_order)
            side_alpha = jnp.asarray(self.spec.alpha_max, dtype=dtype) * jnp.power(
                jnp.asarray(1.0, dtype=dtype) - u, alpha_order
            )
            return side_sigma, side_kappa, side_alpha

        # 2. Place the profile on physical Yee coordinates. Complete E supports have
        # wall-aligned samples at i*dx for i=0..N, while H supports have one sample per
        # material cell at (i+1/2)*dx for i=0..N-1. Deriving both absorber sides from
        # these coordinates avoids separate low/high recipes drifting by half a cell.
        if sample_kind == "E":
            offset = 0.0
            inferred_domain_cells = int(total_samples) - 1
        elif sample_kind == "H":
            offset = 0.5
            inferred_domain_cells = int(total_samples)
        else:
            raise ValueError(f"Unsupported CPML sample kind {sample_kind!r}")
        domain_cells = (
            inferred_domain_cells if domain_cells is None else int(domain_cells)
        )
        if domain_cells <= 0:
            raise ValueError("CPML domain_cells must be positive.")
        coords = jnp.arange(int(total_samples), dtype=jnp.float32) + jnp.asarray(
            offset, dtype=jnp.float32
        )

        physical_coordinates = (
            None
            if sample_coordinates is None
            else jnp.asarray(sample_coordinates, dtype=jnp.float32)
        )
        distance_scale = (
            max(float(pml_cells), 1e-30)
            if physical_coordinates is None
            else max(float(self.spec.thickness), 1e-30)
        )

        def apply_side(dist):
            mask = dist > 0.0
            side_sigma, side_kappa, side_alpha = apply_u(dist / distance_scale)
            return mask, side_sigma, side_kappa, side_alpha

        # 3. Evaluate each active side from the same domain coordinate system. The
        # interface sample has zero CPML strength and remains an identity update.
        if physical_coordinates is not None:
            if domain_bounds is None:
                raise ValueError(
                    "Physical CPML sample coordinates require domain_bounds."
                )
            lower_bound, upper_bound = (float(value) for value in domain_bounds)
            coords = physical_coordinates

        if low_active:
            low_dist = (
                jnp.clip(float(pml_cells) - coords, 0.0, float(pml_cells))
                if physical_coordinates is None
                else jnp.clip(
                    float(self.spec.thickness) - (coords - lower_bound),
                    0.0,
                    float(self.spec.thickness),
                )
            )
            mask, side_sigma, side_kappa, side_alpha = apply_side(low_dist)
            sigma = jnp.where(mask, jnp.maximum(sigma, side_sigma), sigma)
            kappa = jnp.where(mask, jnp.maximum(kappa, side_kappa), kappa)
            alpha = jnp.where(mask, jnp.maximum(alpha, side_alpha), alpha)

        if high_active:
            high_dist = (
                jnp.clip(
                    coords - float(domain_cells - pml_cells),
                    0.0,
                    float(pml_cells),
                )
                if physical_coordinates is None
                else jnp.clip(
                    coords - (upper_bound - float(self.spec.thickness)),
                    0.0,
                    float(self.spec.thickness),
                )
            )
            mask, side_sigma, side_kappa, side_alpha = apply_side(high_dist)
            sigma = jnp.where(mask, jnp.maximum(sigma, side_sigma), sigma)
            kappa = jnp.where(mask, jnp.maximum(kappa, side_kappa), kappa)
            alpha = jnp.where(mask, jnp.maximum(alpha, side_alpha), alpha)

        # 5. Return aligned sigma, kappa, and alpha vectors for broadcasting onto the
        # corresponding directional derivative.
        return sigma, kappa, alpha

    def _create_sponge_profiles(self, fields, domain_size):
        """Create sponge profiles from one axis-driven 2D/3D implementation."""
        shape = tuple(fields.permittivity.shape)
        width, height, depth = domain_size
        if len(shape) == 3:
            axes = (
                ("z", depth, 0, "front", "back"),
                ("y", height, 1, "bottom", "top"),
                ("x", width, 2, "left", "right"),
            )
        else:
            axes = (
                ("y", height, 0, "bottom", "top"),
                ("x", width, 1, "left", "right"),
            )

        edges = set(self._get_edges_for_dimensionality(len(shape) == 3))
        profiles = {f"sigma_{name}": jnp.zeros(shape) for name in "xyz"}
        mask = jnp.zeros(shape, dtype=bool)
        for name, length, axis, low_edge, high_edge in axes:
            geometry = _rectilinear_geometry(fields)
            coords = (
                jnp.asarray(geometry.centers(name) - geometry.axis_edges(name)[0])
                if geometry is not None
                else jnp.linspace(0, length, shape[axis])
            )
            sigma = self._compute_1d_profile(
                coords, length, low_edge in edges, high_edge in edges
            )
            broadcast_shape = [1] * len(shape)
            broadcast_shape[axis] = shape[axis]
            grid = jnp.broadcast_to(jnp.reshape(sigma, broadcast_shape), shape)
            profiles[f"sigma_{name}"] = grid
            mask |= grid > 0
        return {"formulation": "sponge", "mask": mask, **profiles}

    def _create_cpml_profiles_2d(self, fields, domain_size):
        """Create sigma/kappa/alpha CPML profiles for 2D."""
        # Public planes are already canonical y/x, so one axis table can drive both
        # cell-centered diagnostics and the selected polarization's Yee profiles.
        ny, nx = fields.permittivity.shape
        width, height, _ = domain_size
        edges = set(self._get_edges_for_dimensionality(False))
        axes = {
            "x": (nx, width, "left", "right"),
            "y": (ny, height, "bottom", "top"),
        }

        def expand(axis, value):
            return value[:, None] if axis == "y" else value[None, :]

        out: dict[str, Any] = {"formulation": "cpml"}
        geometry = _rectilinear_geometry(fields)
        for axis, (cells, length, low, high) in axes.items():
            values = self._compute_1d_cpml_profile(
                (
                    jnp.asarray(geometry.centers(axis) - geometry.axis_edges(axis)[0])
                    if geometry is not None
                    else jnp.linspace(0, length, cells)
                ),
                length,
                low in edges,
                high in edges,
            )
            for family, value in zip(("sigma", "kappa", "alpha"), values, strict=True):
                out[f"{family}_{axis}"] = jnp.broadcast_to(
                    expand(axis, value), (ny, nx)
                )
        out.update(
            sigma_z=jnp.zeros((ny, nx)),
            kappa_z=jnp.ones((ny, nx)),
            alpha_z=jnp.zeros((ny, nx)),
        )
        out["mask"] = (out["sigma_x"] > 0) | (out["sigma_y"] > 0)

        polarization = fields.polarization_2d
        supports = (
            (
                ("Ez", "x", nx + 1, "E", (ny + 1, nx + 1)),
                ("Ez", "y", ny + 1, "E", (ny + 1, nx + 1)),
                ("Hx", "y", ny, "H", (ny, nx + 1)),
                ("Hy", "x", nx, "H", (ny + 1, nx)),
            )
            if polarization == "tm"
            else (
                ("Hz", "x", nx, "H", (ny, nx)),
                ("Hz", "y", ny, "H", (ny, nx)),
                ("Ex", "y", ny + 1, "E", (ny + 1, nx)),
                ("Ey", "x", nx + 1, "E", (ny, nx + 1)),
            )
        )
        staggered = {}
        for component, axis, samples, kind, target_shape in supports:
            cells, length, low, high = axes[axis]
            values = self._compute_fdtdx_staggered_profile_1d(
                samples,
                float(length) / max(cells, 1),
                low in edges,
                high in edges,
                sample_kind=kind,
                domain_cells=cells,
                sample_coordinates=(
                    geometry.axis_edges(axis)
                    if geometry is not None and kind == "E"
                    else (geometry.centers(axis) if geometry is not None else None)
                ),
                domain_bounds=(
                    (geometry.axis_edges(axis)[0], geometry.axis_edges(axis)[-1])
                    if geometry is not None
                    else None
                ),
            )
            for family, value in zip(("sigma", "kappa", "alpha"), values, strict=True):
                staggered[f"{component}_{axis}_{family}"] = jnp.broadcast_to(
                    expand(axis, value), target_shape
                )
        out[f"{polarization}_xy_cpml"] = staggered
        return out

    def _create_cpml_profiles_3d(self, fields, domain_size):
        """Create sigma/kappa/alpha CPML profiles for 3D."""
        # 1. Build cell-centered profiles along x, y, and z for material-level CPML
        # metadata and diagnostics.
        shape = fields.permittivity.shape
        nz, ny, nx = shape
        width, height, depth = domain_size

        edges = set(self._get_edges_for_dimensionality(True))
        axis_index = {"z": 0, "y": 1, "x": 2}
        axes = {
            "x": (nx, width, "left", "right"),
            "y": (ny, height, "bottom", "top"),
            "z": (nz, depth, "front", "back"),
        }
        out: dict[str, Any] = {"formulation": "cpml"}
        geometry = _rectilinear_geometry(fields)
        for axis, (cells, length, low, high) in axes.items():
            values = self._compute_1d_cpml_profile(
                (
                    jnp.asarray(geometry.centers(axis) - geometry.axis_edges(axis)[0])
                    if geometry is not None
                    else jnp.linspace(0, length, cells)
                ),
                length,
                low in edges,
                high in edges,
            )
            broadcast_shape = [1, 1, 1]
            broadcast_shape[axis_index[axis]] = cells
            for family, value in zip(("sigma", "kappa", "alpha"), values, strict=True):
                out[f"{family}_{axis}"] = jnp.reshape(value, broadcast_shape)

        def compact_axis(profile, axis_name):
            # Reshape 1D profiles for broadcasting along the two transverse axes.
            shape_1d = [1, 1, 1]
            shape_1d[axis_index[axis_name]] = profile.shape[0]
            return jnp.reshape(profile, tuple(shape_1d))

        def profile_for_spec(spec):
            # Build this profile on its exact staggered support to avoid interpolation
            # inside timestep kernels.
            target = getattr(fields, spec.target_component)
            target_shape = tuple(int(v) for v in target.shape)
            axis_name = spec.derivative_axis
            cells, length, low_edge, high_edge = axes[axis_name]
            offset = component_axis_offsets_3d(spec.target_component)[axis_name]
            sample_kind = "E" if offset == 0.0 else "H"
            return self._compute_fdtdx_staggered_profile_1d(
                target_shape[axis_index[axis_name]],
                float(length) / max(cells, 1),
                low_edge in edges,
                high_edge in edges,
                sample_kind=sample_kind,
                domain_cells=cells,
                sample_coordinates=(
                    geometry.axis_edges(axis_name)
                    if geometry is not None and sample_kind == "E"
                    else (geometry.centers(axis_name) if geometry is not None else None)
                ),
                domain_bounds=(
                    (
                        geometry.axis_edges(axis_name)[0],
                        geometry.axis_edges(axis_name)[-1],
                    )
                    if geometry is not None
                    else None
                ),
            )

        # 3. Generate sigma/kappa/alpha for every directional curl term independently;
        # each target component has distinct support even along the same physical axis.
        for spec in (*CPML_3D_H_DERIVATIVES, *CPML_3D_E_DERIVATIVES):
            sigma_1d, kappa_1d, alpha_1d = profile_for_spec(spec)
            out[f"cpml3d_{spec.name}_sigma"] = compact_axis(
                sigma_1d, spec.derivative_axis
            )
            out[f"cpml3d_{spec.name}_kappa"] = compact_axis(
                kappa_1d, spec.derivative_axis
            )
            out[f"cpml3d_{spec.name}_alpha"] = compact_axis(
                alpha_1d, spec.derivative_axis
            )
        # 4. Return both compact broadcastable base profiles and all native derivative
        # profiles needed by the compiled 3D CPML kernel.
        return out


def compile_absorber_regions(
    spec, fields, domain_size, resolution, dt
) -> Mapping[str, Any]:
    """Compile an absorbing Device boundary for one discretized field grid."""
    return _AbsorberCompiler(spec).compile_absorber_regions(
        fields, domain_size, resolution, dt
    )


def lower_boundaries(
    material_grid: MaterialGrid,
    component_shapes,
    boundaries,
    domain_size,
    dt: float,
    *,
    polarization_2d: str = "tm",
) -> BoundaryData:
    """Lower the complete boundary tuple once for Simulation compilation."""
    boundaries = normalize_boundaries(boundaries)
    workspace = _BoundaryGrid(material_grid, component_shapes)
    profiles: Mapping[str, Any] | None = None
    for boundary in boundaries:
        if not isinstance(boundary, (PML, Absorber)):
            continue
        contribution = compile_absorber_regions(
            boundary,
            workspace,
            tuple(float(value) for value in domain_size),
            material_grid.resolution,
            dt,
        )
        profiles = (
            dict(contribution)
            if profiles is None
            else cast(Mapping[str, Any], _merge_profiles(profiles, contribution))
        )
    return BoundaryData(
        workspace.permittivity,
        workspace.conductivity,
        workspace.permeability,
        workspace.yee_materials,
        profiles,
        compile_metallic_masks(
            component_shapes,
            material_grid.shape,
            boundaries,
            polarization_2d=polarization_2d,
        ),
        resolve_metallic_edges(boundaries, len(material_grid.shape) == 3),
    )
