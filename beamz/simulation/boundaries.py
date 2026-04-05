from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, MU_0, µm

_VALID_EDGES = frozenset({"left", "right", "top", "bottom", "front", "back"})


@dataclass(frozen=True, slots=True)
class Boundary:
    """Abstract base class for all boundary conditions."""
    edges: str | tuple[str, ...] = "all"
    thickness: float = 1 * µm

    def __post_init__(self):
        if self.edges != "all":
            edges = self.edges if isinstance(self.edges, (tuple, list)) else (self.edges,)
            normalized = tuple(str(edge).lower() for edge in edges)
            invalid = sorted(set(normalized) - _VALID_EDGES)
            if invalid:
                raise ValueError(f"invalid boundary edges: {invalid}")
            object.__setattr__(self, "edges", normalized)
        object.__setattr__(self, "thickness", float(self.thickness))
        if not np.isfinite(self.thickness) or self.thickness <= 0:
            raise ValueError("thickness must be positive")

    def _get_edges_for_dimensionality(self, is_3d):
        """Resolve 'all' edges based on dimensionality."""
        if self.edges == "all":
            return (
                ["left", "right", "top", "bottom", "front", "back"]
                if is_3d
                else ["left", "right", "top", "bottom"]
            )
        return self.edges

    def to_dict(self):
        return {
            "type": type(self).__name__,
            "edges": self.edges if self.edges == "all" else list(self.edges),
            "thickness": float(self.thickness),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            edges=data.get("edges", "all"),
            thickness=data.get("thickness", 1 * µm),
        )


@dataclass(frozen=True, slots=True)
class PML(Boundary):
    """Perfectly Matched Layer boundary condition using graded-sigma absorption."""
    sigma_max: float | None = None
    m: int = 3

    def __post_init__(self):
        Boundary.__post_init__(self)
        if self.sigma_max is not None:
            object.__setattr__(self, "sigma_max", float(self.sigma_max))
            if not np.isfinite(self.sigma_max) or self.sigma_max <= 0:
                raise ValueError("sigma_max must be a finite positive value when provided")
        object.__setattr__(self, "m", int(self.m))
        if self.m <= 0:
            raise ValueError("m must be positive")

    def _resolved_sigma_max(self, resolution, eps_avg=1.0):
        if self.sigma_max is not None:
            return self.sigma_max
        eta = np.sqrt(MU_0 / (EPS_0 * eps_avg))
        return 0.8 * (self.m + 1) / (eta * resolution)

    def to_dict(self):
        data = Boundary.to_dict(self)
        data.update(
            sigma_max=self.sigma_max,
            m=int(self.m),
        )
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            edges=data.get("edges", "all"),
            thickness=data.get("thickness", 1 * µm),
            sigma_max=data.get("sigma_max"),
            m=data.get("m", 3),
        )

    def create_pml_regions(self, fields, design, resolution, dt, plane_2d="xy"):
        """Create permanent PML region masks and graded-sigma conductivity profiles.

        Returns dict with:
            - mask: boolean array indicating PML cells
            - sigma_x, sigma_y, sigma_z: conductivity profiles
        """
        sigma_max = self._resolved_sigma_max(resolution)

        if fields.permittivity.ndim == 3:
            return self._create_pml_profiles_3d(fields, design, sigma_max)
        else:
            return self._create_pml_profiles_2d(fields, design, plane_2d, sigma_max)

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

    def _compute_1d_profile(self, coords, length, low_active, high_active, sigma_max):
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
            sigma = sigma + sigma_max * (dist / thickness) ** self.m

        if high_active:
            dist = jnp.clip(coords - (length - thickness), 0.0, None)
            sigma = sigma + sigma_max * (dist / thickness) ** self.m

        return sigma

    def _create_pml_profiles_2d(self, fields, design, plane_2d, sigma_max):
        """Create graded-sigma PML profiles for 2D plane."""
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
            coords1, len1, "bottom" in edges, "top" in edges, sigma_max
        )
        sigma2 = self._compute_1d_profile(
            coords2, len2, "left" in edges, "right" in edges, sigma_max
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
        return {"mask": pml_mask, **profiles}

    def _create_pml_profiles_3d(self, fields, design, sigma_max):
        """Create graded-sigma PML profiles for 3D."""
        shape = fields.permittivity.shape  # (nz, ny, nx)
        nz, ny, nx = shape
        depth, height, width = design.depth, design.height, design.width

        edges = self._get_edges_for_dimensionality(True)

        coords_x = jnp.linspace(0, width, nx)
        coords_y = jnp.linspace(0, height, ny)
        coords_z = jnp.linspace(0, depth, nz)

        sigma_x_1d = self._compute_1d_profile(
            coords_x, width, "left" in edges, "right" in edges, sigma_max
        )
        sigma_y_1d = self._compute_1d_profile(
            coords_y, height, "bottom" in edges, "top" in edges, sigma_max
        )
        sigma_z_1d = self._compute_1d_profile(
            coords_z, depth, "front" in edges, "back" in edges, sigma_max
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
        return {"mask": pml_mask, **profiles}


def boundary_spec_to_dict(boundary):
    if isinstance(boundary, (Boundary, PML)):
        return boundary.to_dict()
    raise TypeError(f"unsupported boundary spec type: {type(boundary).__name__}")


def boundary_spec_from_dict(data):
    kind = data.get("type")
    if kind == "Boundary":
        return Boundary.from_dict(data)
    if kind == "PML":
        return PML.from_dict(data)
    raise ValueError(f"unknown boundary spec type: {kind!r}")
