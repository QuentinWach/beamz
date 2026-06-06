import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, MU_0, µm
from beamz.shared_kernels import CPML_3D_E_DERIVATIVES, CPML_3D_H_DERIVATIVES
from beamz.simulation.yee import (
    component_axis_offsets_3d,
    sample_voxel_grid_at_component_3d,
    sample_voxel_grid_at_tm_xy_full_component_2d,
)


def _canonicalize_pml_formulation(formulation):
    """Normalize public absorber/PML formulation names."""

    normalized = str(formulation).lower()
    aliases = {
        "sigma": "sponge",
        "sponge": "sponge",
        "cpml": "cpml",
    }
    if normalized not in aliases:
        raise ValueError(
            "Unsupported boundary formulation "
            f"{formulation!r}. Expected one of: 'sponge', 'sigma', 'cpml'."
        )
    return aliases[normalized]


class Boundary:
    """Abstract base class for all boundary conditions."""

    def __init__(self, edges, thickness):
        """
        Args:
            edges: list of edge names or 'all'
                   2D: ['left', 'right', 'top', 'bottom']
                   3D: ['left', 'right', 'top', 'bottom', 'front', 'back']
            thickness: physical thickness of boundary region
        """
        if edges == "all":
            self.edges = "all"
        else:
            self.edges = edges if isinstance(edges, list) else [edges]
        self.thickness = thickness

    def _get_edges_for_dimensionality(self, is_3d):
        """Resolve 'all' edges based on dimensionality."""
        if self.edges == "all":
            return (
                ["left", "right", "top", "bottom", "front", "back"]
                if is_3d
                else ["left", "right", "top", "bottom"]
            )
        return self.edges


class PML(Boundary):
    """Perfectly Matched Layer boundary condition.

    BeamZ keeps this class for backwards compatibility, but the default
    formulation is a graded-conductivity absorbing layer exposed as
    ``formulation="sponge"``. The optional ``formulation="cpml"`` path exposes
    the additional ``sigma/kappa/alpha`` profiles needed for a true
    convolutional PML in the solver update equations.
    """

    _DEFAULT_CPML_ALPHA_NORMALIZED = 0.225
    _DEFAULT_CPML_SIGMA_SCALE = 0.375

    def __init__(
        self,
        edges="all",
        thickness=1 * µm,
        sigma_max=None,
        m=3,
        formulation="sponge",
        kappa_max=2.0,
        alpha_max=None,
        target_reflection=1e-6,
    ):
        """
        Args:
            edges: edges to apply PML
            thickness: PML thickness
            sigma_max: maximum conductivity (auto-calculated if None)
            m: conductivity grading order
            formulation: ``"sponge"`` for the default graded-conductivity
                absorber, ``"sigma"`` as a backwards-compatible alias, or
                ``"cpml"`` for the convolutional PML
        """
        super().__init__(edges, thickness)
        self.sigma_max = sigma_max
        self.m = m
        self.formulation = _canonicalize_pml_formulation(formulation)
        self.kappa_max = float(kappa_max)
        self.alpha_max = None if alpha_max is None else float(alpha_max)
        self.target_reflection = float(target_reflection)

    def create_pml_regions(self, fields, design, resolution, dt, plane_2d="xy"):
        """Create permanent PML region masks and graded-sigma conductivity profiles.

        Returns dict with:
            - mask: boolean array indicating PML cells
            - sigma_x, sigma_y, sigma_z: conductivity profiles
        """
        if self.sigma_max is None:
            eta = np.sqrt(MU_0 / (EPS_0 * 1.0))
            thickness = max(float(self.thickness), float(resolution))
            self.sigma_max = (
                -(self.m + 1)
                * np.log(max(self.target_reflection, 1e-16))
                / (2.0 * eta * thickness)
            )
            if self.formulation == "cpml":
                # The CPML curl correction plus collocated material loss is
                # sensitive to over-damping at the absorber entrance. A softer
                # ramp reduces impedance mismatch on BeamZ's native Yee grid.
                self.sigma_max *= self._DEFAULT_CPML_SIGMA_SCALE
        if self.formulation == "cpml" and self.alpha_max is None:
            # Convert a conservative normalized CFS alpha into the solver's
            # conductivity-like units so the default CPML keeps a nonzero
            # complex-frequency shift instead of silently falling back to alpha=0.
            self.alpha_max = (
                2.0
                * EPS_0
                * self._DEFAULT_CPML_ALPHA_NORMALIZED
                / max(float(dt), 1e-30)
            )

        if self.formulation == "cpml":
            if fields.permittivity.ndim == 3:
                out = self._create_cpml_profiles_3d(fields, design)
            else:
                out = self._create_cpml_profiles_2d(fields, design, plane_2d)
            self._raise_if_cpml_material_not_extruded(fields, out, plane_2d)
            return out

        if fields.permittivity.ndim == 3:
            out = self._create_pml_profiles_3d(fields, design)
        else:
            out = self._create_pml_profiles_2d(fields, design, plane_2d)
        self._warn_if_material_not_extruded(fields, out, plane_2d)
        return out

    def _pml_material_variation_edges(self, fields, pml_data, plane_2d="xy"):
        """Return edges where material changes along the PML normal."""
        material = np.asarray(fields.permittivity)
        if material.size == 0:
            return []
        axis_map_3d = {"z": 0, "y": 1, "x": 2}
        axis_map_2d = {
            "xy": {"y": 0, "x": 1},
            "yz": {"z": 0, "y": 1},
            "xz": {"z": 0, "x": 1},
        }.get(str(plane_2d), {})
        edge_axes = {
            "left": ("x", "low"),
            "right": ("x", "high"),
            "bottom": ("y", "low"),
            "top": ("y", "high"),
            "front": ("z", "low"),
            "back": ("z", "high"),
        }
        is_3d = material.ndim == 3
        axis_map = axis_map_3d if is_3d else axis_map_2d
        bad_edges = []
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
            if count <= 0 or count >= material.shape[axis]:
                continue
            if self._pml_edge_material_varies(material, axis, side, count):
                bad_edges.append(edge)
        return bad_edges

    def _warn_if_material_not_extruded(self, fields, pml_data, plane_2d="xy"):
        """Warn when material changes along the PML normal inside the absorber."""
        bad_edges = self._pml_material_variation_edges(fields, pml_data, plane_2d)
        if bad_edges:
            warnings.warn(
                "PML material varies along the absorber normal on edges "
                f"{bad_edges}. For lower reflection, extrude the boundary material "
                "profile through the PML or keep geometry clear of the absorber.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _raise_if_cpml_material_not_extruded(self, fields, pml_data, plane_2d="xy"):
        """Reject CPML when geometry varies through the absorber."""
        bad_edges = self._pml_material_variation_edges(fields, pml_data, plane_2d)
        if bad_edges:
            raise ValueError(
                "CPML material varies along the absorber normal on edges "
                f"{bad_edges}. Extrude the boundary material profile through the "
                "CPML or keep geometry clear of the absorber."
            )

    @staticmethod
    def _pml_edge_material_varies(material, axis: int, side: str, count: int) -> bool:
        mat = np.asarray(material)
        if side == "low":
            slab_sel = [slice(None)] * mat.ndim
            slab_sel[axis] = slice(0, count)
            ref_sel = [slice(None)] * mat.ndim
            ref_sel[axis] = count
        else:
            slab_sel = [slice(None)] * mat.ndim
            slab_sel[axis] = slice(mat.shape[axis] - count, mat.shape[axis])
            ref_sel = [slice(None)] * mat.ndim
            ref_sel[axis] = mat.shape[axis] - count - 1
        slab = mat[tuple(slab_sel)]
        ref = np.expand_dims(mat[tuple(ref_sel)], axis=axis)
        return bool(np.any(np.abs(slab - ref) > 1e-9))

    def get_conductivity(
        self, x, y, z=0, dx=1e-6, dt=1e-15, eps_avg=1.0, width=0, height=0, depth=0
    ):
        """Calculate PML conductivity at a specific point (x,y,z)."""
        if self.sigma_max is None:
            eta = np.sqrt(MU_0 / (EPS_0 * eps_avg))
            s_max = 0.8 * (self.m + 1) / (eta * dx)
        else:
            s_max = self.sigma_max

        sigma = 0.0
        is_3d = depth > 0
        edges = self._get_edges_for_dimensionality(is_3d)

        for edge in edges:
            dist = -1.0
            if edge == "left" and x < self.thickness:
                dist = self.thickness - x
            elif edge == "right" and x > (width - self.thickness):
                dist = x - (width - self.thickness)
            elif edge == "bottom" and y < self.thickness:
                dist = self.thickness - y
            elif edge == "top" and y > (height - self.thickness):
                dist = y - (height - self.thickness)
            elif edge == "front" and z < self.thickness:
                dist = self.thickness - z
            elif edge == "back" and z > (depth - self.thickness):
                dist = z - (depth - self.thickness)

            if dist > 0:
                sigma += s_max * (dist / self.thickness) ** self.m
        return sigma

    def _compute_1d_profile(self, coords, length, low_active, high_active):
        """Compute 1D graded-sigma profile along an axis.

        Args:
            coords: 1D coordinate array along this axis
            length: physical length along this axis
            low_active: whether the low edge (left/bottom/front) is active
            high_active: whether the high edge (right/top/back) is active
        """
        sigma = jnp.zeros_like(coords)
        thickness = self.thickness

        if low_active:
            dist = jnp.clip(thickness - coords, 0.0, None)
            sigma = sigma + self.sigma_max * (dist / thickness) ** self.m

        if high_active:
            dist = jnp.clip(coords - (length - thickness), 0.0, None)
            sigma = sigma + self.sigma_max * (dist / thickness) ** self.m

        return sigma

    def _compute_1d_cpml_profile(self, coords, length, low_active, high_active):
        """Compute 1D CPML sigma/kappa/alpha profiles."""

        sigma = jnp.zeros_like(coords)
        kappa = jnp.ones_like(coords)
        alpha = jnp.zeros_like(coords)
        thickness = max(float(self.thickness), 1e-30)

        def apply_side(dist):
            graded = jnp.clip(dist / thickness, 0.0, 1.0)
            side_sigma = self.sigma_max * graded**self.m
            side_kappa = 1.0 + (self.kappa_max - 1.0) * graded**self.m
            side_alpha = self.alpha_max * (1.0 - graded)
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
        sigma_order=None,
        kappa_order=None,
        alpha_order=None,
    ):
        """Compute a 1D CPML profile using FDTDX-style discrete Yee offsets.

        FDTDX constructs the PML grading on discrete E/H sample offsets rather than
        from continuous physical coordinates. That changes the exact grading at the
        interface and outer edge, especially for E-node samples. We mirror that here
        for the native 2D full-TM path used by the compiled benchmarks.
        """

        sigma = jnp.zeros((int(total_samples),), dtype=jnp.float32)
        kappa = jnp.ones((int(total_samples),), dtype=jnp.float32)
        alpha = jnp.zeros((int(total_samples),), dtype=jnp.float32)

        pml_cells = max(
            int(round(float(self.thickness) / max(float(spacing), 1e-30))), 1
        )
        sigma_order = float(self.m if sigma_order is None else sigma_order)
        kappa_order = float(self.m if kappa_order is None else kappa_order)
        alpha_order = float(1.0 if alpha_order is None else alpha_order)

        def apply_u(u):
            u = jnp.clip(u, 0.0, 1.0)
            side_sigma = self.sigma_max * jnp.power(u, sigma_order)
            side_kappa = 1.0 + (self.kappa_max - 1.0) * jnp.power(u, kappa_order)
            side_alpha = self.alpha_max * jnp.power(1.0 - u, alpha_order)
            return side_sigma, side_kappa, side_alpha

        def low_distances(count):
            if sample_kind == "E":
                return jnp.arange(count - 1, -1, -1, dtype=jnp.float32)
            if count <= 0:
                return jnp.zeros((0,), dtype=jnp.float32)
            head = jnp.arange(count - 1.5, -0.5, -1.0, dtype=jnp.float32)
            return jnp.concatenate([head, jnp.zeros((1,), dtype=jnp.float32)])[:count]

        def high_distances(count):
            if sample_kind == "E":
                if count <= 0:
                    return jnp.zeros((0,), dtype=jnp.float32)
                tail = jnp.arange(0.5, count - 0.5, 1.0, dtype=jnp.float32)
                return jnp.concatenate([jnp.zeros((1,), dtype=jnp.float32), tail])[
                    :count
                ]
            return jnp.arange(0.0, count, 1.0, dtype=jnp.float32)

        if low_active:
            count = min(int(total_samples), pml_cells)
            d = low_distances(count)
            side_sigma, side_kappa, side_alpha = apply_u(
                d / max(float(pml_cells), 1e-30)
            )
            sigma = sigma.at[:count].set(side_sigma)
            kappa = kappa.at[:count].set(side_kappa)
            alpha = alpha.at[:count].set(side_alpha)

        if high_active:
            count = min(int(total_samples), pml_cells)
            d = high_distances(count)
            side_sigma, side_kappa, side_alpha = apply_u(
                d / max(float(pml_cells), 1e-30)
            )
            sigma = sigma.at[-count:].set(jnp.maximum(sigma[-count:], side_sigma))
            kappa = kappa.at[-count:].set(jnp.maximum(kappa[-count:], side_kappa))
            alpha = alpha.at[-count:].set(jnp.maximum(alpha[-count:], side_alpha))

        return sigma, kappa, alpha

    def _create_pml_profiles_2d(self, fields, design, plane_2d):
        """Create graded-conductivity absorber profiles for a 2D plane."""
        shape = fields.permittivity.shape
        dim1, dim2 = shape

        if plane_2d == "xy":
            len1, len2 = design.height, design.width
            axis1_name, axis2_name = "y", "x"
        elif plane_2d == "yz":
            len1 = design.depth if design.depth else 0
            len2 = design.height
            axis1_name, axis2_name = "z", "y"
        elif plane_2d == "xz":
            len1 = design.depth if design.depth else 0
            len2 = design.width
            axis1_name, axis2_name = "z", "x"

        edges = self._get_edges_for_dimensionality(False)

        coords1 = jnp.linspace(0, len1, dim1)
        coords2 = jnp.linspace(0, len2, dim2)

        sigma1 = self._compute_1d_profile(
            coords1, len1, "bottom" in edges, "top" in edges
        )
        sigma2 = self._compute_1d_profile(
            coords2, len2, "left" in edges, "right" in edges
        )

        sigma_axis1 = jnp.broadcast_to(sigma1[:, None], shape)
        sigma_axis2 = jnp.broadcast_to(sigma2[None, :], shape)

        third_axis_name = ({"x", "y", "z"} - {axis1_name, axis2_name}).pop()

        profiles = {
            f"sigma_{axis1_name}": sigma_axis1,
            f"sigma_{axis2_name}": sigma_axis2,
            f"sigma_{third_axis_name}": jnp.zeros(shape),
        }

        pml_mask = (sigma_axis1 > 0) | (sigma_axis2 > 0)
        return {"formulation": "sponge", "mask": pml_mask, **profiles}

    def _create_pml_profiles_3d(self, fields, design):
        """Create graded-conductivity absorber profiles for 3D."""
        shape = fields.permittivity.shape  # (nz, ny, nx)
        nz, ny, nx = shape
        depth, height, width = design.depth, design.height, design.width

        edges = self._get_edges_for_dimensionality(True)

        coords_x = jnp.linspace(0, width, nx)
        coords_y = jnp.linspace(0, height, ny)
        coords_z = jnp.linspace(0, depth, nz)

        sigma_x_1d = self._compute_1d_profile(
            coords_x, width, "left" in edges, "right" in edges
        )
        sigma_y_1d = self._compute_1d_profile(
            coords_y, height, "bottom" in edges, "top" in edges
        )
        sigma_z_1d = self._compute_1d_profile(
            coords_z, depth, "front" in edges, "back" in edges
        )

        sigma_x = jnp.broadcast_to(sigma_x_1d[None, None, :], shape)
        sigma_y = jnp.broadcast_to(sigma_y_1d[None, :, None], shape)
        sigma_z = jnp.broadcast_to(sigma_z_1d[:, None, None], shape)

        profiles = {
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "sigma_z": sigma_z,
        }

        pml_mask = (sigma_x > 0) | (sigma_y > 0) | (sigma_z > 0)
        return {"formulation": "sponge", "mask": pml_mask, **profiles}

    def _create_cpml_profiles_2d(self, fields, design, plane_2d):
        """Create sigma/kappa/alpha CPML profiles for 2D."""
        shape = fields.permittivity.shape
        dim1, dim2 = shape

        if plane_2d == "xy":
            len1, len2 = design.height, design.width
            axis1_name, axis2_name = "y", "x"
        elif plane_2d == "yz":
            len1 = design.depth if design.depth else 0
            len2 = design.height
            axis1_name, axis2_name = "z", "y"
        elif plane_2d == "xz":
            len1 = design.depth if design.depth else 0
            len2 = design.width
            axis1_name, axis2_name = "z", "x"
        else:
            raise ValueError(f"Unsupported plane {plane_2d!r}")

        edges = self._get_edges_for_dimensionality(False)
        coords1 = jnp.linspace(0, len1, dim1)
        coords2 = jnp.linspace(0, len2, dim2)

        sigma1, kappa1, alpha1 = self._compute_1d_cpml_profile(
            coords1, len1, "bottom" in edges, "top" in edges
        )
        sigma2, kappa2, alpha2 = self._compute_1d_cpml_profile(
            coords2, len2, "left" in edges, "right" in edges
        )

        sigma_axis1 = jnp.broadcast_to(sigma1[:, None], shape)
        sigma_axis2 = jnp.broadcast_to(sigma2[None, :], shape)
        kappa_axis1 = jnp.broadcast_to(kappa1[:, None], shape)
        kappa_axis2 = jnp.broadcast_to(kappa2[None, :], shape)
        alpha_axis1 = jnp.broadcast_to(alpha1[:, None], shape)
        alpha_axis2 = jnp.broadcast_to(alpha2[None, :], shape)

        third_axis_name = ({"x", "y", "z"} - {axis1_name, axis2_name}).pop()
        pml_mask = (sigma_axis1 > 0) | (sigma_axis2 > 0)
        out = {
            "formulation": "cpml",
            "mask": pml_mask,
            f"sigma_{axis1_name}": sigma_axis1,
            f"sigma_{axis2_name}": sigma_axis2,
            f"sigma_{third_axis_name}": jnp.zeros(shape),
            f"kappa_{axis1_name}": kappa_axis1,
            f"kappa_{axis2_name}": kappa_axis2,
            f"kappa_{third_axis_name}": jnp.ones(shape),
            f"alpha_{axis1_name}": alpha_axis1,
            f"alpha_{axis2_name}": alpha_axis2,
            f"alpha_{third_axis_name}": jnp.zeros(shape),
        }
        if plane_2d == "xy":
            ny, nx = shape
            dy = float(len1) / max(ny, 1)
            dx = float(len2) / max(nx, 1)
            sigma_ez_y, kappa_ez_y, alpha_ez_y = (
                self._compute_fdtdx_staggered_profile_1d(
                    ny + 1,
                    dy,
                    "bottom" in edges,
                    "top" in edges,
                    sample_kind="E",
                )
            )
            sigma_ez_x, kappa_ez_x, alpha_ez_x = (
                self._compute_fdtdx_staggered_profile_1d(
                    nx + 1,
                    dx,
                    "left" in edges,
                    "right" in edges,
                    sample_kind="E",
                )
            )
            sigma_hx_y, kappa_hx_y, alpha_hx_y = (
                self._compute_fdtdx_staggered_profile_1d(
                    ny,
                    dy,
                    "bottom" in edges,
                    "top" in edges,
                    sample_kind="H",
                )
            )
            sigma_hy_x, kappa_hy_x, alpha_hy_x = (
                self._compute_fdtdx_staggered_profile_1d(
                    nx,
                    dx,
                    "left" in edges,
                    "right" in edges,
                    sample_kind="H",
                )
            )
            out["tm_xy_cpml"] = {
                "Ez_x_sigma": jnp.broadcast_to(sigma_ez_x[None, :], (ny + 1, nx + 1)),
                "Ez_x_kappa": jnp.broadcast_to(kappa_ez_x[None, :], (ny + 1, nx + 1)),
                "Ez_x_alpha": jnp.broadcast_to(alpha_ez_x[None, :], (ny + 1, nx + 1)),
                "Ez_y_sigma": jnp.broadcast_to(sigma_ez_y[:, None], (ny + 1, nx + 1)),
                "Ez_y_kappa": jnp.broadcast_to(kappa_ez_y[:, None], (ny + 1, nx + 1)),
                "Ez_y_alpha": jnp.broadcast_to(alpha_ez_y[:, None], (ny + 1, nx + 1)),
                "Hx_y_sigma": jnp.broadcast_to(sigma_hx_y[:, None], (ny, nx + 1)),
                "Hx_y_kappa": jnp.broadcast_to(kappa_hx_y[:, None], (ny, nx + 1)),
                "Hx_y_alpha": jnp.broadcast_to(alpha_hx_y[:, None], (ny, nx + 1)),
                "Hy_x_sigma": jnp.broadcast_to(sigma_hy_x[None, :], (ny + 1, nx)),
                "Hy_x_kappa": jnp.broadcast_to(kappa_hy_x[None, :], (ny + 1, nx)),
                "Hy_x_alpha": jnp.broadcast_to(alpha_hy_x[None, :], (ny + 1, nx)),
            }
        return out

    def _create_cpml_profiles_3d(self, fields, design):
        """Create sigma/kappa/alpha CPML profiles for 3D."""
        shape = fields.permittivity.shape
        nz, ny, nx = shape
        depth, height, width = design.depth, design.height, design.width

        edges = self._get_edges_for_dimensionality(True)
        coords_x = jnp.linspace(0, width, nx)
        coords_y = jnp.linspace(0, height, ny)
        coords_z = jnp.linspace(0, depth, nz)

        sigma_x_1d, kappa_x_1d, alpha_x_1d = self._compute_1d_cpml_profile(
            coords_x, width, "left" in edges, "right" in edges
        )
        sigma_y_1d, kappa_y_1d, alpha_y_1d = self._compute_1d_cpml_profile(
            coords_y, height, "bottom" in edges, "top" in edges
        )
        sigma_z_1d, kappa_z_1d, alpha_z_1d = self._compute_1d_cpml_profile(
            coords_z, depth, "front" in edges, "back" in edges
        )

        sigma_x = jnp.broadcast_to(sigma_x_1d[None, None, :], shape)
        sigma_y = jnp.broadcast_to(sigma_y_1d[None, :, None], shape)
        sigma_z = jnp.broadcast_to(sigma_z_1d[:, None, None], shape)
        kappa_x = jnp.broadcast_to(kappa_x_1d[None, None, :], shape)
        kappa_y = jnp.broadcast_to(kappa_y_1d[None, :, None], shape)
        kappa_z = jnp.broadcast_to(kappa_z_1d[:, None, None], shape)
        alpha_x = jnp.broadcast_to(alpha_x_1d[None, None, :], shape)
        alpha_y = jnp.broadcast_to(alpha_y_1d[None, :, None], shape)
        alpha_z = jnp.broadcast_to(alpha_z_1d[:, None, None], shape)

        pml_mask = (sigma_x > 0) | (sigma_y > 0) | (sigma_z > 0)
        out = {
            "formulation": "cpml",
            "mask": pml_mask,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "sigma_z": sigma_z,
            "kappa_x": kappa_x,
            "kappa_y": kappa_y,
            "kappa_z": kappa_z,
            "alpha_x": alpha_x,
            "alpha_y": alpha_y,
            "alpha_z": alpha_z,
        }

        dz = float(depth) / max(nz, 1)
        dy = float(height) / max(ny, 1)
        dx = float(width) / max(nx, 1)

        axis_index = {"z": 0, "y": 1, "x": 2}
        axis_spacing = {"z": dz, "y": dy, "x": dx}
        axis_edges = {
            "z": ("front", "back"),
            "y": ("bottom", "top"),
            "x": ("left", "right"),
        }

        def bcast_axis(profile, axis_name, target_shape):
            shape_1d = [1, 1, 1]
            shape_1d[axis_index[axis_name]] = profile.shape[0]
            return jnp.broadcast_to(jnp.reshape(profile, tuple(shape_1d)), target_shape)

        def profile_for_spec(spec):
            target = getattr(fields, spec.target_component)
            target_shape = tuple(int(v) for v in target.shape)
            axis_name = spec.derivative_axis
            low_edge, high_edge = axis_edges[axis_name]
            offset = component_axis_offsets_3d(spec.target_component)[axis_name]
            sample_kind = "E" if offset == 0.0 else "H"
            return self._compute_fdtdx_staggered_profile_1d(
                target_shape[axis_index[axis_name]],
                axis_spacing[axis_name],
                low_edge in edges,
                high_edge in edges,
                sample_kind=sample_kind,
            )

        for spec in (*CPML_3D_H_DERIVATIVES, *CPML_3D_E_DERIVATIVES):
            target_shape = tuple(
                int(v) for v in getattr(fields, spec.target_component).shape
            )
            sigma_1d, kappa_1d, alpha_1d = profile_for_spec(spec)
            out[f"cpml3d_{spec.name}_sigma"] = bcast_axis(
                sigma_1d, spec.derivative_axis, target_shape
            )
            out[f"cpml3d_{spec.name}_kappa"] = bcast_axis(
                kappa_1d, spec.derivative_axis, target_shape
            )
            out[f"cpml3d_{spec.name}_alpha"] = bcast_axis(
                alpha_1d, spec.derivative_axis, target_shape
            )
        return out


class AbsorbingLayer(PML):
    """Legacy graded-conductivity absorbing layer.

    This boundary adds loss by merging a graded conductivity shell into the
    material conductivity update. It is not a matched-layer formulation.
    """

    def __init__(
        self,
        edges="all",
        thickness=1 * µm,
        sigma_max=None,
        m=3,
        target_reflection=1e-6,
    ):
        super().__init__(
            edges=edges,
            thickness=thickness,
            sigma_max=sigma_max,
            m=m,
            formulation="sponge",
            target_reflection=target_reflection,
        )


class PEC(Boundary):
    """Perfect electric conductor boundary condition."""

    def __init__(self, edges="all"):
        # Thickness is unused for PEC but kept for API parity with other boundaries.
        super().__init__(edges=edges, thickness=0.0)


@dataclass
class FullPec3DState:
    """Symmetric 3D Yee representation for full-domain PEC cavities.

    Beamz's compact 3D storage omits the high-side plane on every axis. This
    state reconstructs the full Yee field representation by adding those missing
    planes explicitly, so low and high PEC walls can be treated symmetrically.
    """

    Ex: jnp.ndarray
    Ey: jnp.ndarray
    Ez: jnp.ndarray
    Hx: jnp.ndarray
    Hy: jnp.ndarray
    Hz: jnp.ndarray
    eps_x_region: jnp.ndarray
    sig_x_region: jnp.ndarray
    eps_y_region: jnp.ndarray
    sig_y_region: jnp.ndarray
    eps_z_region: jnp.ndarray
    sig_z_region: jnp.ndarray
    sigma_m_hx: jnp.ndarray
    sigma_m_hy: jnp.ndarray
    sigma_m_hz: jnp.ndarray
    masks: dict[str, jnp.ndarray]


@dataclass
class Tm2DXYState:
    """Native 2D TMz Yee representation for the xy plane."""

    Ez: jnp.ndarray
    Hx: jnp.ndarray
    Hy: jnp.ndarray
    eps_z_region: jnp.ndarray
    sig_z_region: jnp.ndarray
    sigma_m_hx: jnp.ndarray
    sigma_m_hy: jnp.ndarray
    metallic_edges: frozenset[str]
    ez_mask: jnp.ndarray
    hx_mask: jnp.ndarray
    hy_mask: jnp.ndarray


def _pad_high_planes_3d(arr):
    out = jnp.asarray(arr)
    for axis in range(3):
        tail = jnp.take(out, indices=jnp.array([out.shape[axis] - 1]), axis=axis)
        out = jnp.concatenate([out, tail], axis=axis)
    return out


def _full_pec_mask_for_component(
    component: str, shape: tuple[int, int, int]
) -> jnp.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if component == "Ex":
        mask[0, :, :] = True
        mask[-1, :, :] = True
        mask[:, 0, :] = True
        mask[:, -1, :] = True
    elif component == "Ey":
        mask[0, :, :] = True
        mask[-1, :, :] = True
        mask[:, :, 0] = True
        mask[:, :, -1] = True
    elif component == "Ez":
        mask[:, 0, :] = True
        mask[:, -1, :] = True
        mask[:, :, 0] = True
        mask[:, :, -1] = True
    elif component == "Hx":
        mask[:, :, 0] = True
        mask[:, :, -1] = True
    elif component == "Hy":
        mask[:, 0, :] = True
        mask[:, -1, :] = True
    elif component == "Hz":
        mask[0, :, :] = True
        mask[-1, :, :] = True
    else:
        raise ValueError(f"Unsupported component {component!r}")
    return jnp.asarray(mask)


def _sample_full_pec_e_region_material_3d(
    grid,
    component: str,
    *,
    stored_shape: tuple[int, int, int],
    region: tuple[slice, slice, slice],
) -> jnp.ndarray:
    """Sample material on the full-PEC E update region.

    The full-PEC state updates only the component-owned axis through the PEC
    walls. The two transverse axes are sliced to unconstrained planes first, so
    their material coefficients are sampled directly at those interior planes.
    Along the component axis, average the two adjacent cell-centered voxels.
    """

    if component not in {"Ex", "Ey", "Ez"}:
        raise ValueError(f"Unsupported E component {component!r}")

    grid_shape = tuple(int(v) for v in np.asarray(grid).shape)
    offsets = component_axis_offsets_3d(component)
    sampled = jnp.asarray(grid)

    for axis_index, (axis, dim, grid_dim, axis_region) in enumerate(
        zip(
            ("z", "y", "x"),
            stored_shape,
            grid_shape,
            region,
            strict=False,
        )
    ):
        coord = np.arange(int(dim), dtype=np.float64)
        if offsets[axis] == 0.5:
            lo = np.clip(coord.astype(np.int32), 0, int(grid_dim) - 1)
            hi = np.clip(lo + 1, 0, int(grid_dim) - 1)
            lo = lo[axis_region]
            hi = hi[axis_region]
            sampled_lo = jnp.take(sampled, jnp.asarray(lo), axis=axis_index)
            sampled_hi = jnp.take(sampled, jnp.asarray(hi), axis=axis_index)
            sampled = 0.5 * (sampled_lo + sampled_hi)
        else:
            idx = np.clip(coord.astype(np.int32), 0, int(grid_dim) - 1)
            idx = idx[axis_region]
            sampled = jnp.take(sampled, jnp.asarray(idx), axis=axis_index)

    return sampled


def initialize_full_pec_3d_state(fields) -> FullPec3DState:
    full_shapes = {
        component: tuple(int(v) + 1 for v in getattr(fields, component).shape)
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }
    full_masks = {
        component: _full_pec_mask_for_component(
            component,
            full_shapes[component],
        )
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }

    def _full(component: str) -> jnp.ndarray:
        return _pad_high_planes_3d(getattr(fields, component))

    region_x = (slice(1, -1), slice(1, -1), slice(None))
    region_y = (slice(1, -1), slice(None), slice(1, -1))
    region_z = (slice(None), slice(1, -1), slice(1, -1))

    total_sigma = jnp.asarray(
        getattr(fields, "total_conductivity", fields.conductivity)
    )
    sigma_base = total_sigma * jnp.asarray(fields.permeability) * MU_0 / EPS_0

    state = FullPec3DState(
        Ex=_full("Ex"),
        Ey=_full("Ey"),
        Ez=_full("Ez"),
        Hx=_full("Hx"),
        Hy=_full("Hy"),
        Hz=_full("Hz"),
        eps_x_region=_sample_full_pec_e_region_material_3d(
            fields.permittivity,
            "Ex",
            stored_shape=full_shapes["Ex"],
            region=region_x,
        ),
        sig_x_region=_sample_full_pec_e_region_material_3d(
            total_sigma,
            "Ex",
            stored_shape=full_shapes["Ex"],
            region=region_x,
        ),
        eps_y_region=_sample_full_pec_e_region_material_3d(
            fields.permittivity,
            "Ey",
            stored_shape=full_shapes["Ey"],
            region=region_y,
        ),
        sig_y_region=_sample_full_pec_e_region_material_3d(
            total_sigma,
            "Ey",
            stored_shape=full_shapes["Ey"],
            region=region_y,
        ),
        eps_z_region=_sample_full_pec_e_region_material_3d(
            fields.permittivity,
            "Ez",
            stored_shape=full_shapes["Ez"],
            region=region_z,
        ),
        sig_z_region=_sample_full_pec_e_region_material_3d(
            total_sigma,
            "Ez",
            stored_shape=full_shapes["Ez"],
            region=region_z,
        ),
        sigma_m_hx=sample_voxel_grid_at_component_3d(
            sigma_base,
            "Hx",
            stored_shape=full_shapes["Hx"],
        ),
        sigma_m_hy=sample_voxel_grid_at_component_3d(
            sigma_base,
            "Hy",
            stored_shape=full_shapes["Hy"],
        ),
        sigma_m_hz=sample_voxel_grid_at_component_3d(
            sigma_base,
            "Hz",
            stored_shape=full_shapes["Hz"],
        ),
        masks=full_masks,
    )

    state.Ex = jnp.where(state.masks["Ex"], 0.0, state.Ex)
    state.Ey = jnp.where(state.masks["Ey"], 0.0, state.Ey)
    state.Ez = jnp.where(state.masks["Ez"], 0.0, state.Ez)
    state.Hx = jnp.where(state.masks["Hx"], 0.0, state.Hx)
    state.Hy = jnp.where(state.masks["Hy"], 0.0, state.Hy)
    state.Hz = jnp.where(state.masks["Hz"], 0.0, state.Hz)
    return state


def sync_compact_fields_from_full_pec_3d(fields, state: FullPec3DState) -> None:
    fields.Ex = state.Ex[:-1, :-1, :-1]
    fields.Ey = state.Ey[:-1, :-1, :-1]
    fields.Ez = state.Ez[:-1, :-1, :-1]
    fields.Hx = state.Hx[:-1, :-1, :-1]
    fields.Hy = state.Hy[:-1, :-1, :-1]
    fields.Hz = state.Hz[:-1, :-1, :-1]


def sync_full_pec_3d_from_compact(fields, state: FullPec3DState) -> None:
    """Refresh full-PEC native fields from compact storage after direct injections."""

    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        full = getattr(state, component)
        full = full.at[:-1, :-1, :-1].set(getattr(fields, component))
        setattr(
            state,
            component,
            jnp.where(state.masks[component], 0.0, full),
        )


def full_tm_2d_xy_masks(
    ez_shape: tuple[int, int], metallic_edges: set[str] | frozenset[str]
) -> dict[str, jnp.ndarray]:
    """Return explicit native-TMz PEC masks for the 2D xy Yee fields."""

    ny, nx = (int(v) for v in ez_shape)
    metallic_edges = set(metallic_edges)

    ez_mask = np.zeros((ny + 1, nx + 1), dtype=bool)
    hx_mask = np.zeros((ny, nx + 1), dtype=bool)
    hy_mask = np.zeros((ny + 1, nx), dtype=bool)

    if "bottom" in metallic_edges:
        ez_mask[0, :] = True
        hy_mask[0, :] = True
    if "top" in metallic_edges:
        ez_mask[-1, :] = True
        hy_mask[-1, :] = True
    if "left" in metallic_edges:
        ez_mask[:, 0] = True
        hx_mask[:, 0] = True
    if "right" in metallic_edges:
        ez_mask[:, -1] = True
        hx_mask[:, -1] = True

    return {
        "Ez": jnp.asarray(ez_mask),
        "Hx": jnp.asarray(hx_mask),
        "Hy": jnp.asarray(hy_mask),
    }


def initialize_tm_2d_xy_state(fields) -> Tm2DXYState:
    """Build native 2D TMz Yee metadata for an xy-plane simulation."""

    ny, nx = (int(v) for v in fields.permittivity.shape)
    total_sigma = jnp.asarray(
        getattr(fields, "total_conductivity", fields.conductivity)
    )
    sigma_base = total_sigma * jnp.asarray(fields.permeability) * MU_0 / EPS_0
    metallic_edges = frozenset(
        resolve_metallic_edges(getattr(fields, "boundaries", None), is_3d=False)
    )

    masks = full_tm_2d_xy_masks((ny, nx), metallic_edges)
    ez = jnp.where(masks["Ez"], 0.0, jnp.asarray(fields.Ez))
    hx = jnp.where(masks["Hx"], 0.0, jnp.asarray(fields.Hx))
    hy = jnp.where(masks["Hy"], 0.0, jnp.asarray(fields.Hy))

    return Tm2DXYState(
        Ez=ez,
        Hx=hx,
        Hy=hy,
        eps_z_region=sample_voxel_grid_at_tm_xy_full_component_2d(
            fields.permittivity,
            "Ez",
        ),
        sig_z_region=sample_voxel_grid_at_tm_xy_full_component_2d(
            total_sigma,
            "Ez",
        ),
        sigma_m_hx=sample_voxel_grid_at_tm_xy_full_component_2d(sigma_base, "Hx"),
        sigma_m_hy=sample_voxel_grid_at_tm_xy_full_component_2d(sigma_base, "Hy"),
        metallic_edges=metallic_edges,
        ez_mask=masks["Ez"],
        hx_mask=masks["Hx"],
        hy_mask=masks["Hy"],
    )


def normalize_boundaries(boundaries, *, is_3d):
    """Resolve the simulation boundary list into explicit boundary objects.

    Beamz historically treated the absence of explicit boundaries as metallic
    walls. Keep that behavior, but express it as an explicit ``PEC(edges="all")``
    boundary so the rest of the simulation stack can reason in terms of boundary
    objects rather than hidden defaults.
    """

    resolved = list(boundaries or [])
    if not resolved:
        resolved = [PEC(edges="all")]
    del is_3d
    return resolved


def resolve_metallic_edges(boundaries, is_3d):
    """Return the active metallic cell walls for the simulation domain.

    Beamz historically relied on zero-padded curls, which pins entire outer field
    layers to zero. Meep's default with no `k_point` is different: the boundaries
    are metallic, and only the Yee samples that lie exactly on the wall are forced
    to zero each step. We match that behavior by deriving the active metallic
    walls from the boundary list.

    Policy:
    - With no explicit boundary objects, all domain walls are metallic, matching Meep.
    - PML removes metallic treatment from the walls it occupies.
    """

    metallic = set()

    for boundary in normalize_boundaries(boundaries, is_3d=is_3d):
        if isinstance(boundary, PEC):
            metallic.update(boundary._get_edges_for_dimensionality(is_3d))
        if isinstance(boundary, PML):
            metallic.difference_update(boundary._get_edges_for_dimensionality(is_3d))
    return metallic


def create_metallic_boundary_masks(fields, boundaries, *, is_3d, plane_2d="xy"):
    """Build per-component masks for Yee samples constrained by metallic walls.

    The masks are aligned to Beamz's staggered field storage and are intended to
    mirror Meep's `zero_metal(ft)` logic for metallic boundaries.
    """

    metallic_edges = resolve_metallic_edges(boundaries, is_3d)

    def _empty_like(arr):
        return np.zeros(tuple(int(v) for v in arr.shape), dtype=bool)

    masks = {
        "Ex": _empty_like(fields.Ex),
        "Ey": _empty_like(fields.Ey),
        "Ez": _empty_like(fields.Ez),
        "Hx": _empty_like(fields.Hx),
        "Hy": _empty_like(fields.Hy),
        "Hz": _empty_like(fields.Hz),
    }

    if not metallic_edges:
        return {name: jnp.asarray(mask) for name, mask in masks.items()}

    if is_3d:
        if "bottom" in metallic_edges:
            masks["Ex"][:, 0, :] = True
            masks["Ez"][:, 0, :] = True
            masks["Hy"][:, 0, :] = True

        if "front" in metallic_edges:
            masks["Ex"][0, :, :] = True
            masks["Ey"][0, :, :] = True
            masks["Hz"][0, :, :] = True

        if "left" in metallic_edges:
            masks["Ey"][:, :, 0] = True
            masks["Ez"][:, :, 0] = True
            masks["Hx"][:, :, 0] = True
    else:
        # Keep 2D unchanged for now. The benchmark in question is 3D and the 2D
        # field layouts in Beamz combine TE/TM components in a way that requires a
        # separate derivation.
        del plane_2d

    return {name: jnp.asarray(mask) for name, mask in masks.items()}


def _pad_with_boundary_ghosts(arr, axis, *, low_metallic=False, high_metallic=False):
    """Pad one axis with PEC ghosts on metallic sides and open ghosts otherwise."""

    shape = list(arr.shape)
    shape[axis] = 1
    zero = jnp.zeros(tuple(shape), dtype=arr.dtype)
    low = zero if low_metallic else jnp.take(arr, indices=jnp.array([0]), axis=axis)
    high = (
        zero
        if high_metallic
        else jnp.take(arr, indices=jnp.array([arr.shape[axis] - 1]), axis=axis)
    )
    return jnp.concatenate([low, arr, high], axis=axis)


def _edge_pair_for_axis(axis):
    if axis == 0:
        return "front", "back"
    if axis == 1:
        return "bottom", "top"
    if axis == 2:
        return "left", "right"
    raise ValueError(f"Unsupported axis {axis!r}")


def _scalar_like(value, dtype):
    """Cast scalar inputs to a stable array dtype before JAX arithmetic."""

    return jnp.asarray(value, dtype=dtype)


def _extend_axis_for_pec(arr, axis, source_offset):
    """Extend one axis according to the compact-grid PEC policy.

    Beamz stores only the low-wall sample for integer-aligned axes and omits the
    corresponding high-wall sample. For a PEC cavity:
    - integer-aligned source axes need an explicit zero-valued high-wall plane
    - half-step source axes must use zero ghosts on both sides so the resulting
      adjacent-difference operator is the discrete transpose of Beamz's compact
      interior forward-difference used for the paired H update.
    """

    shape = list(arr.shape)
    shape[axis] = 1
    plane = jnp.zeros(tuple(shape), dtype=arr.dtype)
    if source_offset == 0.0:
        return jnp.concatenate([arr, plane], axis=axis)
    if source_offset == 0.5:
        return jnp.concatenate([plane, arr, plane], axis=axis)
    raise ValueError(f"Unsupported Yee offset {source_offset!r}")


def _adjacent_difference(arr, axis, resolution):
    moved = jnp.moveaxis(arr, axis, 0)
    resolution = _scalar_like(resolution, moved.dtype)
    diff = (moved[1:] - moved[:-1]) / resolution
    return jnp.moveaxis(diff, 0, axis)


def _cpml_ab_from_profiles(sigma, kappa, alpha, dt):
    dtype = jnp.result_type(sigma, kappa, alpha, jnp.float32)
    sigma = jnp.asarray(sigma, dtype=dtype)
    kappa = jnp.asarray(kappa, dtype=dtype)
    alpha = jnp.asarray(alpha, dtype=dtype)
    one = _scalar_like(1.0, dtype)
    zero = _scalar_like(0.0, dtype)
    eps_0 = _scalar_like(EPS_0, dtype)
    dt = _scalar_like(dt, dtype)
    denom_floor = _scalar_like(1e-30, dtype)
    kappa = jnp.maximum(kappa, one)
    decay = (sigma / kappa + alpha) * (dt / eps_0)
    b = jnp.expm1(-decay) + one
    denom = sigma + kappa * alpha
    a = jnp.nan_to_num(
        ((b - one) * sigma) / jnp.maximum(denom * kappa, denom_floor),
        nan=zero,
        posinf=zero,
        neginf=zero,
    )
    return a, b


def _cpml_correct_native_term(derivative, psi, a_term, b_term, inv_kappa_term):
    dtype = psi.dtype
    derivative = jnp.asarray(derivative, dtype=dtype)
    a_term = jnp.asarray(a_term, dtype=dtype)
    b_term = jnp.asarray(b_term, dtype=dtype)
    inv_kappa_term = jnp.asarray(inv_kappa_term, dtype=dtype)
    psi_updated = b_term * psi + a_term * derivative
    corrected = derivative * inv_kappa_term + psi_updated
    return corrected, psi_updated


def _cpml_update_native_term(derivative, psi, a_term, b_term):
    dtype = psi.dtype
    derivative = jnp.asarray(derivative, dtype=dtype)
    a_term = jnp.asarray(a_term, dtype=dtype)
    b_term = jnp.asarray(b_term, dtype=dtype)
    return b_term * psi + a_term * derivative


def _cpml_corrected_update_term(derivative, psi, a_term, b_term, inv_kappa_term):
    psi_updated = _cpml_update_native_term(derivative, psi, a_term, b_term)
    corrected = derivative * jnp.asarray(inv_kappa_term, dtype=psi.dtype) + psi_updated
    return corrected, psi_updated


def build_h_boundary_views_for_e_3d(hx, hy, hz, boundaries):
    """Return H-field views for the 3D E update with boundaries applied outside ops.

    Each returned view is extended only along the derivative axis needed for the
    corresponding curl term. Compact 3D Beamz storage omits one outer Yee plane
    along half-step source axes, so the extension must respect the source
    component's physical offset rather than blindly inserting zero ghosts.

    Keys:
    - ``hz_y``: for ``dHz/dy`` in the ``Ex`` update
    - ``hy_z``: for ``dHy/dz`` in the ``Ex`` update
    - ``hx_z``: for ``dHx/dz`` in the ``Ey`` update
    - ``hz_x``: for ``dHz/dx`` in the ``Ey`` update
    - ``hy_x``: for ``dHy/dx`` in the ``Ez`` update
    - ``hx_y``: for ``dHx/dy`` in the ``Ez`` update
    """

    del (
        boundaries
    )  # Boundary selection lives here; current 3D policy uses compact Yee extension.
    return {
        "hz_y": _extend_axis_for_pec(
            hz, axis=1, source_offset=component_axis_offsets_3d("Hz")["y"]
        ),
        "hy_z": _extend_axis_for_pec(
            hy, axis=0, source_offset=component_axis_offsets_3d("Hy")["z"]
        ),
        "hx_z": _extend_axis_for_pec(
            hx, axis=0, source_offset=component_axis_offsets_3d("Hx")["z"]
        ),
        "hz_x": _extend_axis_for_pec(
            hz, axis=2, source_offset=component_axis_offsets_3d("Hz")["x"]
        ),
        "hy_x": _extend_axis_for_pec(
            hy, axis=2, source_offset=component_axis_offsets_3d("Hy")["x"]
        ),
        "hx_y": _extend_axis_for_pec(
            hx, axis=1, source_offset=component_axis_offsets_3d("Hx")["y"]
        ),
    }


def has_full_pec_3d(boundaries) -> bool:
    return resolve_metallic_edges(boundaries, is_3d=True) == {
        "left",
        "right",
        "bottom",
        "top",
        "front",
        "back",
    }


def has_full_pec_2d_xy(boundaries, plane_2d: str) -> bool:
    return plane_2d == "xy" and resolve_metallic_edges(boundaries, is_3d=False) == {
        "left",
        "right",
        "bottom",
        "top",
    }


def pec_curl_h_to_e_3d(hx, hy, hz, resolution, ex_shape, ey_shape, ez_shape):
    """Compute the 3D E curl for a full-PEC cavity from explicit boundary layers."""

    hz_y = _extend_axis_for_pec(
        hz, axis=1, source_offset=component_axis_offsets_3d("Hz")["y"]
    )
    hy_z = _extend_axis_for_pec(
        hy, axis=0, source_offset=component_axis_offsets_3d("Hy")["z"]
    )
    hx_z = _extend_axis_for_pec(
        hx, axis=0, source_offset=component_axis_offsets_3d("Hx")["z"]
    )
    hz_x = _extend_axis_for_pec(
        hz, axis=2, source_offset=component_axis_offsets_3d("Hz")["x"]
    )
    hy_x = _extend_axis_for_pec(
        hy, axis=2, source_offset=component_axis_offsets_3d("Hy")["x"]
    )
    hx_y = _extend_axis_for_pec(
        hx, axis=1, source_offset=component_axis_offsets_3d("Hx")["y"]
    )

    curl_hx = _adjacent_difference(
        hz_y, axis=1, resolution=resolution
    ) - _adjacent_difference(hy_z, axis=0, resolution=resolution)
    curl_hy = _adjacent_difference(
        hx_z, axis=0, resolution=resolution
    ) - _adjacent_difference(hz_x, axis=2, resolution=resolution)
    curl_hz = _adjacent_difference(
        hy_x, axis=2, resolution=resolution
    ) - _adjacent_difference(hx_y, axis=1, resolution=resolution)

    assert curl_hx.shape == ex_shape, (
        f"curl_hx shape mismatch: {curl_hx.shape} vs {ex_shape}"
    )
    assert curl_hy.shape == ey_shape, (
        f"curl_hy shape mismatch: {curl_hy.shape} vs {ey_shape}"
    )
    assert curl_hz.shape == ez_shape, (
        f"curl_hz shape mismatch: {curl_hz.shape} vs {ez_shape}"
    )
    return curl_hx, curl_hy, curl_hz


def pec_curl_e_to_h_3d(ex, ey, ez, resolution, hx_shape, hy_shape, hz_shape):
    """Compute the 3D H curl for a full-PEC cavity from explicit boundary layers."""
    resolution = _scalar_like(resolution, ex.dtype)
    curl_ex = (ez[:, 1:, :] - ez[:, :-1, :]) / resolution - (
        ey[1:, :, :] - ey[:-1, :, :]
    ) / resolution
    curl_ey = (ex[1:, :, :] - ex[:-1, :, :]) / resolution - (
        ez[:, :, 1:] - ez[:, :, :-1]
    ) / resolution
    curl_ez = (ey[:, :, 1:] - ey[:, :, :-1]) / resolution - (
        ex[:, 1:, :] - ex[:, :-1, :]
    ) / resolution

    assert curl_ex.shape == hx_shape, (
        f"curl_ex shape mismatch: {curl_ex.shape} vs {hx_shape}"
    )
    assert curl_ey.shape == hy_shape, (
        f"curl_ey shape mismatch: {curl_ey.shape} vs {hy_shape}"
    )
    assert curl_ez.shape == hz_shape, (
        f"curl_ez shape mismatch: {curl_ez.shape} vs {hz_shape}"
    )
    return curl_ex, curl_ey, curl_ez


def _full_pec_advance_h_component(field, curl, decay, source, mask):
    if getattr(decay, "size", 0) == 0:
        field = field - source * curl
    else:
        field = decay * field - source * curl
    return jnp.where(mask, jnp.zeros((), dtype=field.dtype), field)


def full_pec_update_h_from_e_3d(
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    resolution,
    *,
    h_decay,
    h_source,
    h_mask,
    source_m=(None, None, None),
):
    """Update full-PEC H components without materializing a curl tuple."""

    resolution = _scalar_like(resolution, ex.dtype)
    h_decay_x, h_decay_y, h_decay_z = h_decay
    h_source_x, h_source_y, h_source_z = h_source
    hx_mask, hy_mask, hz_mask = h_mask
    source_m_x, source_m_y, source_m_z = source_m

    curl_x = ((ez[:, 1:, :] - ez[:, :-1, :]) - (ey[1:, :, :] - ey[:-1, :, :])) / (
        resolution
    )
    if source_m_x is not None:
        curl_x = curl_x + source_m_x
    hx = _full_pec_advance_h_component(
        hx, curl_x, h_decay_x, h_source_x, hx_mask
    )

    curl_y = ((ex[1:, :, :] - ex[:-1, :, :]) - (ez[:, :, 1:] - ez[:, :, :-1])) / (
        resolution
    )
    if source_m_y is not None:
        curl_y = curl_y + source_m_y
    hy = _full_pec_advance_h_component(
        hy, curl_y, h_decay_y, h_source_y, hy_mask
    )

    curl_z = ((ey[:, :, 1:] - ey[:, :, :-1]) - (ex[:, 1:, :] - ex[:, :-1, :])) / (
        resolution
    )
    if source_m_z is not None:
        curl_z = curl_z + source_m_z
    hz = _full_pec_advance_h_component(
        hz, curl_z, h_decay_z, h_source_z, hz_mask
    )

    return hx, hy, hz


def full_pec_h_update_coefficients_3d(state, dt):
    """Return grouped H update coefficients for a full-PEC 3D state."""

    sigma = (state.sigma_m_hx, state.sigma_m_hy, state.sigma_m_hz)
    denom = tuple(1.0 + term * (dt / (2.0 * MU_0)) for term in sigma)
    decay = tuple(
        (1.0 - term * (dt / (2.0 * MU_0))) / den
        for term, den in zip(sigma, denom, strict=True)
    )
    source = tuple((dt / MU_0) / den for den in denom)
    return decay, source


def _full_pec_region_coefficient(coeff, region, field_shape):
    if getattr(coeff, "size", 0) == 0:
        return coeff
    if tuple(getattr(coeff, "shape", ())) == tuple(field_shape):
        return coeff[region]
    return coeff


def _full_pec_advance_e_component(field, curl, decay, source, mask, region):
    if getattr(decay, "size", 0) == 0:
        values = field[region] + source[region] * curl
    else:
        decay = _full_pec_region_coefficient(decay, region, field.shape)
        source = _full_pec_region_coefficient(source, region, field.shape)
        values = decay * field[region] + source * curl
    field = field.at[region].set(values)
    return jnp.where(mask, jnp.zeros((), dtype=field.dtype), field)


def full_pec_update_e_from_h_3d(
    hx,
    hy,
    hz,
    ex,
    ey,
    ez,
    resolution,
    *,
    e_decay,
    e_source,
    e_mask,
    source_j=(None, None, None),
):
    """Update full-PEC E components without building full zero-padded curl grids."""

    resolution = _scalar_like(resolution, hx.dtype)
    e_decay_x, e_decay_y, e_decay_z = e_decay
    e_source_x, e_source_y, e_source_z = e_source
    ex_mask, ey_mask, ez_mask = e_mask
    source_j_x, source_j_y, source_j_z = source_j

    region_x = (slice(1, -1), slice(1, -1), slice(None))
    dHz_dy = (hz[:, 1:, :] - hz[:, :-1, :]) / resolution
    dHy_dz = (hy[1:, :, :] - hy[:-1, :, :]) / resolution
    curl_x = dHz_dy[1:-1, :, :] - dHy_dz[:, 1:-1, :]
    if source_j_x is not None:
        curl_x = curl_x + source_j_x[region_x]
    ex = _full_pec_advance_e_component(
        ex, curl_x, e_decay_x, e_source_x, ex_mask, region_x
    )

    region_y = (slice(1, -1), slice(None), slice(1, -1))
    dHx_dz = (hx[1:, :, :] - hx[:-1, :, :]) / resolution
    dHz_dx = (hz[:, :, 1:] - hz[:, :, :-1]) / resolution
    curl_y = dHx_dz[:, :, 1:-1] - dHz_dx[1:-1, :, :]
    if source_j_y is not None:
        curl_y = curl_y + source_j_y[region_y]
    ey = _full_pec_advance_e_component(
        ey, curl_y, e_decay_y, e_source_y, ey_mask, region_y
    )

    region_z = (slice(None), slice(1, -1), slice(1, -1))
    dHy_dx = (hy[:, :, 1:] - hy[:, :, :-1]) / resolution
    dHx_dy = (hx[:, 1:, :] - hx[:, :-1, :]) / resolution
    curl_z = dHy_dx[:, 1:-1, :] - dHx_dy[:, :, 1:-1]
    if source_j_z is not None:
        curl_z = curl_z + source_j_z[region_z]
    ez = _full_pec_advance_e_component(
        ez, curl_z, e_decay_z, e_source_z, ez_mask, region_z
    )

    return ex, ey, ez


def full_pec_e_update_coefficients_3d(state, dt):
    """Return grouped region-local E update coefficients for a full-PEC 3D state."""

    conductivity = (state.sig_x_region, state.sig_y_region, state.sig_z_region)
    permittivity = (state.eps_x_region, state.eps_y_region, state.eps_z_region)
    denom = tuple(
        1.0 + sigma * (dt / (2.0 * EPS_0 * eps))
        for sigma, eps in zip(conductivity, permittivity, strict=True)
    )
    decay = tuple(
        (1.0 - sigma * (dt / (2.0 * EPS_0 * eps))) / den
        for sigma, eps, den in zip(conductivity, permittivity, denom, strict=True)
    )
    source = tuple(
        (dt / (EPS_0 * eps)) / den
        for eps, den in zip(permittivity, denom, strict=True)
    )
    return decay, source


def cpml_curl_e_to_h_3d(
    ex,
    ey,
    ez,
    resolution,
    *,
    a_h_terms,
    b_h_terms,
    inv_kappa_h_terms,
    psi_h_terms,
):
    """CPML-corrected 3D curl E -> H on native Yee derivative terms."""

    resolution = _scalar_like(resolution, ex.dtype)
    d_ez_dy = (ez[:, 1:, :] - ez[:, :-1, :]) / resolution
    d_ey_dz = (ey[1:, :, :] - ey[:-1, :, :]) / resolution
    d_ex_dz = (ex[1:, :, :] - ex[:-1, :, :]) / resolution
    d_ez_dx = (ez[:, :, 1:] - ez[:, :, :-1]) / resolution
    d_ey_dx = (ey[:, :, 1:] - ey[:, :, :-1]) / resolution
    d_ex_dy = (ex[:, 1:, :] - ex[:, :-1, :]) / resolution

    term0, psi0 = _cpml_correct_native_term(
        d_ez_dy, psi_h_terms[0], a_h_terms[0], b_h_terms[0], inv_kappa_h_terms[0]
    )
    term1, psi1 = _cpml_correct_native_term(
        d_ey_dz, psi_h_terms[1], a_h_terms[1], b_h_terms[1], inv_kappa_h_terms[1]
    )
    term2, psi2 = _cpml_correct_native_term(
        d_ex_dz, psi_h_terms[2], a_h_terms[2], b_h_terms[2], inv_kappa_h_terms[2]
    )
    term3, psi3 = _cpml_correct_native_term(
        d_ez_dx, psi_h_terms[3], a_h_terms[3], b_h_terms[3], inv_kappa_h_terms[3]
    )
    term4, psi4 = _cpml_correct_native_term(
        d_ey_dx, psi_h_terms[4], a_h_terms[4], b_h_terms[4], inv_kappa_h_terms[4]
    )
    term5, psi5 = _cpml_correct_native_term(
        d_ex_dy, psi_h_terms[5], a_h_terms[5], b_h_terms[5], inv_kappa_h_terms[5]
    )

    curl_hx = term0 - term1
    curl_hy = term2 - term3
    curl_hz = term4 - term5
    return curl_hx, curl_hy, curl_hz, (psi0, psi1, psi2, psi3, psi4, psi5)


def cpml_update_h_from_e_3d(
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    h_decay_x,
    h_source_x,
    h_decay_y,
    h_source_y,
    h_decay_z,
    h_source_z,
    resolution,
    *,
    a_h_terms,
    b_h_terms,
    inv_kappa_h_terms,
    psi_h_terms,
):
    """CPML-corrected H update from E without returning full curl arrays."""

    resolution = _scalar_like(resolution, ex.dtype)

    term0, psi0 = _cpml_corrected_update_term(
        (ez[:, 1:, :] - ez[:, :-1, :]) / resolution,
        psi_h_terms[0],
        a_h_terms[0],
        b_h_terms[0],
        inv_kappa_h_terms[0],
    )
    term1, psi1 = _cpml_corrected_update_term(
        (ey[1:, :, :] - ey[:-1, :, :]) / resolution,
        psi_h_terms[1],
        a_h_terms[1],
        b_h_terms[1],
        inv_kappa_h_terms[1],
    )
    hx = h_decay_x * hx - h_source_x * (term0 - term1)

    term2, psi2 = _cpml_corrected_update_term(
        (ex[1:, :, :] - ex[:-1, :, :]) / resolution,
        psi_h_terms[2],
        a_h_terms[2],
        b_h_terms[2],
        inv_kappa_h_terms[2],
    )
    term3, psi3 = _cpml_corrected_update_term(
        (ez[:, :, 1:] - ez[:, :, :-1]) / resolution,
        psi_h_terms[3],
        a_h_terms[3],
        b_h_terms[3],
        inv_kappa_h_terms[3],
    )
    hy = h_decay_y * hy - h_source_y * (term2 - term3)

    term4, psi4 = _cpml_corrected_update_term(
        (ey[:, :, 1:] - ey[:, :, :-1]) / resolution,
        psi_h_terms[4],
        a_h_terms[4],
        b_h_terms[4],
        inv_kappa_h_terms[4],
    )
    term5, psi5 = _cpml_corrected_update_term(
        (ex[:, 1:, :] - ex[:, :-1, :]) / resolution,
        psi_h_terms[5],
        a_h_terms[5],
        b_h_terms[5],
        inv_kappa_h_terms[5],
    )
    hz = h_decay_z * hz - h_source_z * (term4 - term5)

    return hx, hy, hz, (psi0, psi1, psi2, psi3, psi4, psi5)


def apply_lossy_shell_from_lossless_3d(
    updated_lossless,
    old,
    source_lossless,
    slabs,
    decay_slabs,
    source_slabs,
    *,
    source_permittivity=None,
    dt=None,
):
    out = updated_lossless
    for starts, sizes, decay_s, source_s in zip(
        (slab[0] for slab in slabs),
        (slab[1] for slab in slabs),
        decay_slabs,
        source_slabs,
    ):
        old_s = jax.lax.dynamic_slice(old, starts, sizes)
        lossless_s = jax.lax.dynamic_slice(updated_lossless, starts, sizes)
        if getattr(source_lossless, "ndim", 0) == 0:
            source_ll_s = source_lossless
        elif getattr(source_lossless, "size", 0) == 0:
            if source_permittivity is None or dt is None:
                raise ValueError(
                    "source_permittivity and dt are required when "
                    "source_lossless is empty."
                )
            eps_s = jax.lax.dynamic_slice(source_permittivity, starts, sizes)
            source_ll_s = dt / (_scalar_like(EPS_0, eps_s.dtype) * eps_s)
        else:
            source_ll_s = jax.lax.dynamic_slice(source_lossless, starts, sizes)
        beta = source_s / source_ll_s
        lossy_s = (decay_s - beta) * old_s + beta * lossless_s
        out = jax.lax.dynamic_update_slice(out, lossy_s, starts)
    return out


def cpml_update_h_from_e_3d_shell_split(
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    h_source_lossless_x,
    h_source_lossless_y,
    h_source_lossless_z,
    resolution,
    *,
    a_h_terms,
    b_h_terms,
    inv_kappa_h_terms,
    psi_h_terms,
    h_lossy_shell_x,
    h_lossy_shell_y,
    h_lossy_shell_z,
    h_shell_decay_x,
    h_shell_source_x,
    h_shell_decay_y,
    h_shell_source_y,
    h_shell_decay_z,
    h_shell_source_z,
):
    """CPML H update using lossless full-domain coefficients plus shell patches."""

    resolution = _scalar_like(resolution, ex.dtype)

    term0, psi0 = _cpml_corrected_update_term(
        (ez[:, 1:, :] - ez[:, :-1, :]) / resolution,
        psi_h_terms[0],
        a_h_terms[0],
        b_h_terms[0],
        inv_kappa_h_terms[0],
    )
    hx_old = hx
    hx = hx_old - h_source_lossless_x * term0
    term1, psi1 = _cpml_corrected_update_term(
        (ey[1:, :, :] - ey[:-1, :, :]) / resolution,
        psi_h_terms[1],
        a_h_terms[1],
        b_h_terms[1],
        inv_kappa_h_terms[1],
    )
    hx = hx + h_source_lossless_x * term1
    hx = apply_lossy_shell_from_lossless_3d(
        hx,
        hx_old,
        h_source_lossless_x,
        h_lossy_shell_x,
        h_shell_decay_x,
        h_shell_source_x,
    )

    term2, psi2 = _cpml_corrected_update_term(
        (ex[1:, :, :] - ex[:-1, :, :]) / resolution,
        psi_h_terms[2],
        a_h_terms[2],
        b_h_terms[2],
        inv_kappa_h_terms[2],
    )
    hy_old = hy
    hy = hy_old - h_source_lossless_y * term2
    term3, psi3 = _cpml_corrected_update_term(
        (ez[:, :, 1:] - ez[:, :, :-1]) / resolution,
        psi_h_terms[3],
        a_h_terms[3],
        b_h_terms[3],
        inv_kappa_h_terms[3],
    )
    hy = hy + h_source_lossless_y * term3
    hy = apply_lossy_shell_from_lossless_3d(
        hy,
        hy_old,
        h_source_lossless_y,
        h_lossy_shell_y,
        h_shell_decay_y,
        h_shell_source_y,
    )

    term4, psi4 = _cpml_corrected_update_term(
        (ey[:, :, 1:] - ey[:, :, :-1]) / resolution,
        psi_h_terms[4],
        a_h_terms[4],
        b_h_terms[4],
        inv_kappa_h_terms[4],
    )
    hz_old = hz
    hz = hz_old - h_source_lossless_z * term4
    term5, psi5 = _cpml_corrected_update_term(
        (ex[:, 1:, :] - ex[:, :-1, :]) / resolution,
        psi_h_terms[5],
        a_h_terms[5],
        b_h_terms[5],
        inv_kappa_h_terms[5],
    )
    hz = hz + h_source_lossless_z * term5
    hz = apply_lossy_shell_from_lossless_3d(
        hz,
        hz_old,
        h_source_lossless_z,
        h_lossy_shell_z,
        h_shell_decay_z,
        h_shell_source_z,
    )

    return hx, hy, hz, (psi0, psi1, psi2, psi3, psi4, psi5)


def cpml_curl_h_to_e_3d(
    hx,
    hy,
    hz,
    resolution,
    *,
    a_e_terms,
    b_e_terms,
    inv_kappa_e_terms,
    psi_e_terms,
    metallic_edges=frozenset(),
):
    """CPML-corrected 3D curl H -> E on native Yee derivative terms."""

    metallic_edges = frozenset(metallic_edges or ())

    def pad(arr, axis):
        low_edge, high_edge = _edge_pair_for_axis(axis)
        return _pad_with_boundary_ghosts(
            arr,
            axis,
            low_metallic=low_edge in metallic_edges,
            high_metallic=high_edge in metallic_edges,
        )

    d_hz_dy = _adjacent_difference(pad(hz, axis=1), axis=1, resolution=resolution)
    d_hy_dz = _adjacent_difference(pad(hy, axis=0), axis=0, resolution=resolution)
    d_hx_dz = _adjacent_difference(pad(hx, axis=0), axis=0, resolution=resolution)
    d_hz_dx = _adjacent_difference(pad(hz, axis=2), axis=2, resolution=resolution)
    d_hy_dx = _adjacent_difference(pad(hy, axis=2), axis=2, resolution=resolution)
    d_hx_dy = _adjacent_difference(pad(hx, axis=1), axis=1, resolution=resolution)

    term0, psi0 = _cpml_correct_native_term(
        d_hz_dy, psi_e_terms[0], a_e_terms[0], b_e_terms[0], inv_kappa_e_terms[0]
    )
    term1, psi1 = _cpml_correct_native_term(
        d_hy_dz, psi_e_terms[1], a_e_terms[1], b_e_terms[1], inv_kappa_e_terms[1]
    )
    term2, psi2 = _cpml_correct_native_term(
        d_hx_dz, psi_e_terms[2], a_e_terms[2], b_e_terms[2], inv_kappa_e_terms[2]
    )
    term3, psi3 = _cpml_correct_native_term(
        d_hz_dx, psi_e_terms[3], a_e_terms[3], b_e_terms[3], inv_kappa_e_terms[3]
    )
    term4, psi4 = _cpml_correct_native_term(
        d_hy_dx, psi_e_terms[4], a_e_terms[4], b_e_terms[4], inv_kappa_e_terms[4]
    )
    term5, psi5 = _cpml_correct_native_term(
        d_hx_dy, psi_e_terms[5], a_e_terms[5], b_e_terms[5], inv_kappa_e_terms[5]
    )

    curl_ex = term0 - term1
    curl_ey = term2 - term3
    curl_ez = term4 - term5
    return curl_ex, curl_ey, curl_ez, (psi0, psi1, psi2, psi3, psi4, psi5)


def cpml_update_e_from_h_3d(
    hx,
    hy,
    hz,
    ex,
    ey,
    ez,
    e_decay_x,
    e_source_x,
    e_decay_y,
    e_source_y,
    e_decay_z,
    e_source_z,
    resolution,
    *,
    a_e_terms,
    b_e_terms,
    inv_kappa_e_terms,
    psi_e_terms,
    metallic_edges=frozenset(),
):
    """CPML-corrected E update from H without returning full curl arrays."""

    metallic_edges = frozenset(metallic_edges or ())

    def pad(arr, axis):
        low_edge, high_edge = _edge_pair_for_axis(axis)
        return _pad_with_boundary_ghosts(
            arr,
            axis,
            low_metallic=low_edge in metallic_edges,
            high_metallic=high_edge in metallic_edges,
        )

    term0, psi0 = _cpml_corrected_update_term(
        _adjacent_difference(pad(hz, axis=1), axis=1, resolution=resolution),
        psi_e_terms[0],
        a_e_terms[0],
        b_e_terms[0],
        inv_kappa_e_terms[0],
    )
    term1, psi1 = _cpml_corrected_update_term(
        _adjacent_difference(pad(hy, axis=0), axis=0, resolution=resolution),
        psi_e_terms[1],
        a_e_terms[1],
        b_e_terms[1],
        inv_kappa_e_terms[1],
    )
    ex = e_decay_x * ex + e_source_x * (term0 - term1)

    term2, psi2 = _cpml_corrected_update_term(
        _adjacent_difference(pad(hx, axis=0), axis=0, resolution=resolution),
        psi_e_terms[2],
        a_e_terms[2],
        b_e_terms[2],
        inv_kappa_e_terms[2],
    )
    term3, psi3 = _cpml_corrected_update_term(
        _adjacent_difference(pad(hz, axis=2), axis=2, resolution=resolution),
        psi_e_terms[3],
        a_e_terms[3],
        b_e_terms[3],
        inv_kappa_e_terms[3],
    )
    ey = e_decay_y * ey + e_source_y * (term2 - term3)

    term4, psi4 = _cpml_corrected_update_term(
        _adjacent_difference(pad(hy, axis=2), axis=2, resolution=resolution),
        psi_e_terms[4],
        a_e_terms[4],
        b_e_terms[4],
        inv_kappa_e_terms[4],
    )
    term5, psi5 = _cpml_corrected_update_term(
        _adjacent_difference(pad(hx, axis=1), axis=1, resolution=resolution),
        psi_e_terms[5],
        a_e_terms[5],
        b_e_terms[5],
        inv_kappa_e_terms[5],
    )
    ez = e_decay_z * ez + e_source_z * (term4 - term5)

    return ex, ey, ez, (psi0, psi1, psi2, psi3, psi4, psi5)


def cpml_update_e_from_h_3d_shell_split(
    hx,
    hy,
    hz,
    ex,
    ey,
    ez,
    e_source_lossless_x,
    e_source_lossless_y,
    e_source_lossless_z,
    e_permittivity_x,
    e_permittivity_y,
    e_permittivity_z,
    dt,
    resolution,
    *,
    a_e_terms,
    b_e_terms,
    inv_kappa_e_terms,
    psi_e_terms,
    e_lossy_shell_x,
    e_lossy_shell_y,
    e_lossy_shell_z,
    e_shell_decay_x,
    e_shell_source_x,
    e_shell_decay_y,
    e_shell_source_y,
    e_shell_decay_z,
    e_shell_source_z,
    metallic_edges=frozenset(),
):
    """CPML E update using lossless material coefficients plus shell patches."""

    metallic_edges = frozenset(metallic_edges or ())
    dt = _scalar_like(dt, ex.dtype)

    def pad(arr, axis):
        low_edge, high_edge = _edge_pair_for_axis(axis)
        return _pad_with_boundary_ghosts(
            arr,
            axis,
            low_metallic=low_edge in metallic_edges,
            high_metallic=high_edge in metallic_edges,
        )

    def add_e_term(field, source_lossless, permittivity, term):
        if getattr(source_lossless, "size", 0) > 0:
            return field + source_lossless * term
        scale = _scalar_like(dt, field.dtype) / _scalar_like(EPS_0, field.dtype)
        return field + scale * term / permittivity

    def subtract_e_term(field, source_lossless, permittivity, term):
        if getattr(source_lossless, "size", 0) > 0:
            return field - source_lossless * term
        scale = _scalar_like(dt, field.dtype) / _scalar_like(EPS_0, field.dtype)
        return field - scale * term / permittivity

    term0, psi0 = _cpml_corrected_update_term(
        _adjacent_difference(pad(hz, axis=1), axis=1, resolution=resolution),
        psi_e_terms[0],
        a_e_terms[0],
        b_e_terms[0],
        inv_kappa_e_terms[0],
    )
    ex_old = ex
    ex = add_e_term(ex_old, e_source_lossless_x, e_permittivity_x, term0)
    term1, psi1 = _cpml_corrected_update_term(
        _adjacent_difference(pad(hy, axis=0), axis=0, resolution=resolution),
        psi_e_terms[1],
        a_e_terms[1],
        b_e_terms[1],
        inv_kappa_e_terms[1],
    )
    ex = subtract_e_term(ex, e_source_lossless_x, e_permittivity_x, term1)
    ex = apply_lossy_shell_from_lossless_3d(
        ex,
        ex_old,
        e_source_lossless_x,
        e_lossy_shell_x,
        e_shell_decay_x,
        e_shell_source_x,
        source_permittivity=e_permittivity_x,
        dt=dt,
    )

    term2, psi2 = _cpml_corrected_update_term(
        _adjacent_difference(pad(hx, axis=0), axis=0, resolution=resolution),
        psi_e_terms[2],
        a_e_terms[2],
        b_e_terms[2],
        inv_kappa_e_terms[2],
    )
    ey_old = ey
    ey = add_e_term(ey_old, e_source_lossless_y, e_permittivity_y, term2)
    term3, psi3 = _cpml_corrected_update_term(
        _adjacent_difference(pad(hz, axis=2), axis=2, resolution=resolution),
        psi_e_terms[3],
        a_e_terms[3],
        b_e_terms[3],
        inv_kappa_e_terms[3],
    )
    ey = subtract_e_term(ey, e_source_lossless_y, e_permittivity_y, term3)
    ey = apply_lossy_shell_from_lossless_3d(
        ey,
        ey_old,
        e_source_lossless_y,
        e_lossy_shell_y,
        e_shell_decay_y,
        e_shell_source_y,
        source_permittivity=e_permittivity_y,
        dt=dt,
    )

    term4, psi4 = _cpml_corrected_update_term(
        _adjacent_difference(pad(hy, axis=2), axis=2, resolution=resolution),
        psi_e_terms[4],
        a_e_terms[4],
        b_e_terms[4],
        inv_kappa_e_terms[4],
    )
    ez_old = ez
    ez = add_e_term(ez_old, e_source_lossless_z, e_permittivity_z, term4)
    term5, psi5 = _cpml_corrected_update_term(
        _adjacent_difference(pad(hx, axis=1), axis=1, resolution=resolution),
        psi_e_terms[5],
        a_e_terms[5],
        b_e_terms[5],
        inv_kappa_e_terms[5],
    )
    ez = subtract_e_term(ez, e_source_lossless_z, e_permittivity_z, term5)
    ez = apply_lossy_shell_from_lossless_3d(
        ez,
        ez_old,
        e_source_lossless_z,
        e_lossy_shell_z,
        e_shell_decay_z,
        e_shell_source_z,
        source_permittivity=e_permittivity_z,
        dt=dt,
    )

    return ex, ey, ez, (psi0, psi1, psi2, psi3, psi4, psi5)


def full_pec_curl_e_to_h_2d_xy(ez, resolution, hx_shape, hy_shape):
    """Native TMz curl E -> H on the 2D xy PEC representation."""

    resolution = _scalar_like(resolution, ez.dtype)
    curl_hx = (ez[1:, :] - ez[:-1, :]) / resolution
    curl_hy = -(ez[:, 1:] - ez[:, :-1]) / resolution

    assert curl_hx.shape == hx_shape, (
        f"curl_hx shape mismatch: {curl_hx.shape} vs {hx_shape}"
    )
    assert curl_hy.shape == hy_shape, (
        f"curl_hy shape mismatch: {curl_hy.shape} vs {hy_shape}"
    )
    return curl_hx, curl_hy


def tm_xy_curl_e_to_h_2d(ez, resolution, hx_shape, hy_shape, metallic_edges):
    """Native TMz curl E -> H on the 2D xy Yee representation."""

    del hx_shape, hy_shape, metallic_edges
    resolution = _scalar_like(resolution, ez.dtype)
    curl_hx = (ez[1:, :] - ez[:-1, :]) / resolution
    curl_hy = -(ez[:, 1:] - ez[:, :-1]) / resolution
    return curl_hx, curl_hy


def full_pec_curl_h_to_e_2d_xy(hx, hy, resolution, ez_shape):
    """Native TMz curl H -> E on the 2D xy PEC representation."""

    curl_hz = tm_xy_curl_h_to_e_2d(
        hx,
        hy,
        resolution,
        ez_shape,
        frozenset({"left", "right", "top", "bottom"}),
    )
    return curl_hz


def tm_xy_curl_h_to_e_2d(hx, hy, resolution, ez_shape, metallic_edges=frozenset()):
    """Native TMz curl H -> E on the 2D xy Yee representation."""

    resolution = _scalar_like(resolution, hy.dtype)
    left_col = hy[:, :1]
    right_col = hy[:, -1:]
    bottom_row = hx[:1, :]
    top_row = hx[-1:, :]

    if "left" in metallic_edges:
        left_col = jnp.zeros_like(left_col)
    if "right" in metallic_edges:
        right_col = jnp.zeros_like(right_col)
    if "bottom" in metallic_edges:
        bottom_row = jnp.zeros_like(bottom_row)
    if "top" in metallic_edges:
        top_row = jnp.zeros_like(top_row)

    hy_pad_x = jnp.concatenate([left_col, hy, right_col], axis=1)
    hx_pad_y = jnp.concatenate([bottom_row, hx, top_row], axis=0)
    curl_hz = (
        hy_pad_x[:, 1:] - hy_pad_x[:, :-1] - (hx_pad_y[1:, :] - hx_pad_y[:-1, :])
    ) / resolution
    assert curl_hz.shape == ez_shape, (
        f"curl_hz shape mismatch: {curl_hz.shape} vs {ez_shape}"
    )
    return curl_hz


def tm_xy_cpml_curl_e_to_h_2d(
    ez,
    resolution,
    *,
    sigma_h_terms,
    kappa_h_aux_terms,
    alpha_h_terms,
    kappa_h_direct_terms,
    psi_h_terms,
    dt,
):
    """CPML curl E -> H on the native 2D xy TM Yee representation.

    Terms are stored by derivative contribution, mirroring the FDTDX structure:
    - term 0: d/dy Ez contribution to Hx
    - term 1: d/dx Ez contribution to Hy
    """

    dtype = ez.dtype
    resolution = _scalar_like(resolution, dtype)
    one = _scalar_like(1.0, dtype)
    zero = _scalar_like(0.0, dtype)
    eps_0 = _scalar_like(EPS_0, dtype)
    dt = _scalar_like(dt, dtype)
    denom_floor = _scalar_like(1e-30, dtype)

    ez_pad_y = jnp.pad(ez, ((0, 1), (0, 0)))
    ez_pad_x = jnp.pad(ez, ((0, 0), (0, 1)))
    d_ez_dy_full = (ez_pad_y[1:, :] - ez_pad_y[:-1, :]) / resolution
    d_ez_dx_full = (ez_pad_x[:, 1:] - ez_pad_x[:, :-1]) / resolution
    d_ez_dy_full = d_ez_dy_full.at[-1, :].set(zero)
    d_ez_dx_full = d_ez_dx_full.at[:, -1].set(zero)

    d_terms = jnp.stack((d_ez_dy_full, d_ez_dx_full), axis=0)
    kappa_h_aux_terms = jnp.maximum(kappa_h_aux_terms, one)
    decay = (sigma_h_terms / kappa_h_aux_terms + alpha_h_terms) * (dt / eps_0)
    b_terms = jnp.expm1(-decay) + one
    denom = sigma_h_terms + kappa_h_aux_terms * alpha_h_terms
    a_terms = jnp.nan_to_num(
        ((b_terms - one) * sigma_h_terms)
        / jnp.maximum(denom * kappa_h_aux_terms, denom_floor),
        nan=zero,
        posinf=zero,
        neginf=zero,
    )
    psi_h_updated = b_terms * psi_h_terms + a_terms * d_terms
    psi_h_updated = psi_h_updated.at[0, -1, :].set(zero)
    psi_h_updated = psi_h_updated.at[1, :, -1].set(zero)
    curl_hx = (one / jnp.maximum(kappa_h_direct_terms[0, :-1, :], one)) * d_ez_dy_full[
        :-1, :
    ] + psi_h_updated[0, :-1, :]
    curl_hy = -(
        (one / jnp.maximum(kappa_h_direct_terms[1, :, :-1], one)) * d_ez_dx_full[:, :-1]
        + psi_h_updated[1, :, :-1]
    )

    return curl_hx, curl_hy, psi_h_updated


def tm_xy_cpml_curl_h_to_e_2d(
    hx,
    hy,
    resolution,
    ez_shape,
    metallic_edges,
    *,
    sigma_e_terms,
    kappa_e_terms,
    alpha_e_terms,
    psi_e_terms,
    dt,
):
    """CPML curl H -> E on the native 2D xy TM Yee representation.

    Terms are stored by derivative contribution, mirroring the FDTDX structure:
    - term 0: d/dx Hy contribution to Ez
    - term 1: d/dy Hx contribution to Ez
    """

    dtype = hx.dtype
    resolution = _scalar_like(resolution, dtype)
    one = _scalar_like(1.0, dtype)
    zero = _scalar_like(0.0, dtype)
    eps_0 = _scalar_like(EPS_0, dtype)
    dt = _scalar_like(dt, dtype)
    denom_floor = _scalar_like(1e-30, dtype)

    left_col = hy[:, :1]
    right_col = hy[:, -1:]
    bottom_row = hx[:1, :]
    top_row = hx[-1:, :]

    if "left" in metallic_edges:
        left_col = jnp.zeros_like(left_col)
    if "right" in metallic_edges:
        right_col = jnp.zeros_like(right_col)
    if "bottom" in metallic_edges:
        bottom_row = jnp.zeros_like(bottom_row)
    if "top" in metallic_edges:
        top_row = jnp.zeros_like(top_row)

    hy_pad_x = jnp.concatenate([left_col, hy, right_col], axis=1)
    hx_pad_y = jnp.concatenate([bottom_row, hx, top_row], axis=0)
    d_hy_dx = (hy_pad_x[:, 1:] - hy_pad_x[:, :-1]) / resolution
    d_hx_dy = (hx_pad_y[1:, :] - hx_pad_y[:-1, :]) / resolution
    d_terms = jnp.stack((d_hy_dx, d_hx_dy), axis=0)
    kappa_e_terms = jnp.maximum(kappa_e_terms, one)
    decay = (sigma_e_terms / kappa_e_terms + alpha_e_terms) * (dt / eps_0)
    b_terms = jnp.expm1(-decay) + one
    denom = sigma_e_terms + kappa_e_terms * alpha_e_terms
    a_terms = jnp.nan_to_num(
        ((b_terms - one) * sigma_e_terms)
        / jnp.maximum(denom * kappa_e_terms, denom_floor),
        nan=zero,
        posinf=zero,
        neginf=zero,
    )
    psi_e_updated = b_terms * psi_e_terms + a_terms * d_terms
    curl_hz = (
        (one / kappa_e_terms[0]) * d_hy_dx
        + psi_e_updated[0]
        - (one / kappa_e_terms[1]) * d_hx_dy
        - psi_e_updated[1]
    )
    assert curl_hz.shape == ez_shape, (
        f"curl_hz shape mismatch: {curl_hz.shape} vs {ez_shape}"
    )
    return curl_hz, psi_e_updated


def xy_te_curl_e_to_h_2d(ex, ey, resolution, hz_shape):
    """Physical TEz curl E -> H on the 2D xy Yee representation."""

    resolution = _scalar_like(resolution, ex.dtype)
    curl_hz = (ey[:, 1:] - ey[:, :-1] - (ex[1:, :] - ex[:-1, :])) / resolution
    assert curl_hz.shape == hz_shape, (
        f"curl_hz shape mismatch: {curl_hz.shape} vs {hz_shape}"
    )
    return curl_hz


def xy_te_curl_h_to_e_2d(hz, resolution, ex_shape, ey_shape, metallic_edges):
    """Physical TEz curl H -> E on the 2D xy Yee representation.

    Metallic walls use zero-valued ghost samples on the constrained side; open
    sides reuse the nearest stored edge sample to preserve the existing
    non-PEC behavior outside cavity benchmarks.
    """

    del ex_shape, ey_shape
    resolution = _scalar_like(resolution, hz.dtype)

    left_col = hz[:, :1]
    right_col = hz[:, -1:]
    bottom_row = hz[:1, :]
    top_row = hz[-1:, :]

    if "left" in metallic_edges:
        left_col = jnp.zeros_like(left_col)
    if "right" in metallic_edges:
        right_col = jnp.zeros_like(right_col)
    if "bottom" in metallic_edges:
        bottom_row = jnp.zeros_like(bottom_row)
    if "top" in metallic_edges:
        top_row = jnp.zeros_like(top_row)

    hz_pad_x = jnp.concatenate([left_col, hz, right_col], axis=1)
    hz_pad_y = jnp.concatenate([bottom_row, hz, top_row], axis=0)

    curl_ex = (hz_pad_y[1:, :] - hz_pad_y[:-1, :]) / resolution
    curl_ey = -(hz_pad_x[:, 1:] - hz_pad_x[:, :-1]) / resolution
    return curl_ex, curl_ey
