from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from beamz._cache_tokens import cache_token
from beamz.devices._immutable import readonly_array
from beamz.lattice import normalize_polarization_2d


def _tensor_diagonal(values) -> np.ndarray:
    tensor = np.asarray(values)
    if tensor.shape[0] == 1:
        return np.broadcast_to(tensor, (3, *tensor.shape[1:]))
    return tensor[:3]


def _tensor_off_diagonal(values) -> np.ndarray:
    tensor = np.asarray(values)
    return tensor[3:] if tensor.shape[0] == 6 else np.zeros((0,), dtype=tensor.dtype)


def _support_diagonal(values, axis: int) -> np.ndarray:
    tensor = np.asarray(values)
    if tensor.ndim < 2 or tensor.shape[0] not in (1, 3, 6):
        raise ValueError(f"Yee tensor has invalid compact shape {tensor.shape}.")
    return tensor[0 if tensor.shape[0] == 1 else axis]


@dataclass(frozen=True, slots=True, eq=False)
class MaterialGrid:
    """Store immutable cell summaries and optional direct Yee coefficients.

    Parameters
    ----------
    permittivity : array-like
        Relative-permittivity samples in ``(y, x)`` or ``(z, y, x)`` order.
    conductivity : array-like
        Electrical-conductivity samples in siemens per metre.
    permeability : array-like
        Relative-permeability samples matching ``permittivity``.
    resolution : float
        Uniform spatial cell size in metres.
    shape : tuple of int
        Material array shape. It must follow array storage order, not public
        coordinate order.
    yee_materials : mapping, optional
        Componentwise material arrays already sampled at Yee supports.
    tensors : mapping, optional
        Packed symmetric cell tensors retained for compatible mode solving.
    yee_tensors : mapping, optional
        Full symmetric permittivity tensors sampled at electric Yee supports and,
        for cross coupling, at the shared grid nodes.
    smoothing : str, default="volume"
        Raster smoothing policy that produced the coefficients.
    origin : tuple of float, default=(0, 0, 0)
        Physical coordinate of the first x, y, and z grid edges.
    polarization : {"tm", "te"}, optional
        Active component family for a two-dimensional solver grid. Three-dimensional
        grids leave this unset.

    Notes
    -----
    Input arrays are copied and made read-only during construction.
    """

    permittivity: npt.ArrayLike
    conductivity: npt.ArrayLike
    permeability: npt.ArrayLike
    resolution: float
    shape: tuple[int, ...]
    yee_materials: Mapping[str, npt.ArrayLike] = field(default_factory=dict)
    tensors: Mapping[str, npt.ArrayLike] = field(default_factory=dict)
    smoothing: str = "volume"
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    polarization: Literal["tm", "te"] | None = None
    yee_tensors: Mapping[str, npt.ArrayLike] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "permittivity", readonly_array(self.permittivity))
        object.__setattr__(self, "conductivity", readonly_array(self.conductivity))
        object.__setattr__(self, "permeability", readonly_array(self.permeability))
        resolution = float(self.resolution)
        if not np.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("MaterialGrid resolution must be finite and positive.")
        object.__setattr__(self, "resolution", resolution)
        shape = tuple(int(v) for v in self.shape)
        if len(shape) not in (2, 3) or any(value <= 0 for value in shape):
            raise ValueError(
                "MaterialGrid shape must contain two or three positive counts."
            )
        object.__setattr__(self, "shape", shape)
        polarization = self.polarization
        if len(shape) == 2:
            polarization = normalize_polarization_2d(polarization or "tm")
        elif polarization is not None:
            raise ValueError("MaterialGrid polarization applies only to 2D grids.")
        object.__setattr__(self, "polarization", polarization)
        for name in ("permittivity", "conductivity", "permeability"):
            values = np.asarray(getattr(self, name))
            actual = values.shape
            if actual == ():
                continue
            if actual != shape:
                raise ValueError(f"{name} has shape {actual}, expected {shape}.")
        for name in ("yee_materials", "tensors", "yee_tensors"):
            values = {
                str(key): readonly_array(value)
                for key, value in dict(getattr(self, name)).items()
            }
            object.__setattr__(self, name, MappingProxyType(values))
        smoothing = str(self.smoothing).strip().lower()
        if smoothing not in {"volume", "farjadpour_diagonal", "farjadpour_full"}:
            raise ValueError("Unknown material-grid smoothing mode.")
        object.__setattr__(self, "smoothing", smoothing)
        origin = tuple(float(value) for value in self.origin)
        if len(origin) != 3 or not np.all(np.isfinite(origin)):
            raise ValueError("MaterialGrid origin must contain three finite values.")
        object.__setattr__(self, "origin", origin)
        allowed_yee = {
            "eps_x": "Ex",
            "eps_y": "Ey",
            "eps_z": "Ez",
            "sig_x": "Ex",
            "sig_y": "Ey",
            "sig_z": "Ez",
            "mu_hx": "Hx",
            "mu_hy": "Hy",
            "mu_hz": "Hz",
        }
        unknown_yee = set(self.yee_materials) - set(allowed_yee)
        if unknown_yee:
            raise ValueError(
                f"Unknown Yee material fields: {', '.join(sorted(unknown_yee))}."
            )
        if self.yee_materials:
            from beamz.lattice import component_shapes

            shapes = component_shapes(self.shape, self.polarization or "tm")
            for name, values in self.yee_materials.items():
                expected = shapes[allowed_yee[name]]
                if np.asarray(values).shape != expected:
                    raise ValueError(
                        f"{name} has shape {np.asarray(values).shape}, expected {expected}."
                    )
        unknown_tensors = set(self.tensors) - {"epsilon", "mu", "conductivity"}
        if unknown_tensors:
            raise ValueError(
                f"Unknown cell tensors: {', '.join(sorted(unknown_tensors))}."
            )
        for name, values in self.tensors.items():
            shape = np.asarray(values).shape
            if (
                len(shape) != len(self.shape) + 1
                or shape[0] not in (1, 3, 6)
                or shape[1:] != self.shape
            ):
                raise ValueError(f"{name} tensor has invalid compact shape {shape}.")
        unknown_yee_tensors = set(self.yee_tensors) - {
            "eps_x",
            "eps_y",
            "eps_z",
            "eps_node",
        }
        if unknown_yee_tensors:
            raise ValueError(
                "Unknown Yee permittivity tensors: "
                f"{', '.join(sorted(unknown_yee_tensors))}."
            )
        if self.yee_tensors:
            from beamz.lattice import component_shapes

            shapes = component_shapes(self.shape, self.polarization or "tm")
            for name, values in self.yee_tensors.items():
                expected = (
                    tuple(value + 1 for value in self.shape)
                    if name == "eps_node"
                    else shapes[{"eps_x": "Ex", "eps_y": "Ey", "eps_z": "Ez"}[name]]
                )
                actual = np.asarray(values).shape
                if actual[0:1] not in ((1,), (3,), (6,)) or actual[1:] != expected:
                    raise ValueError(
                        f"{name} tensor has shape {actual}, expected compact "
                        f"(1|3|6, {', '.join(map(str, expected))})."
                    )

    @classmethod
    def from_raster_result(
        cls,
        result: Any,
        *,
        dimensions: Literal[2, 3] | None = None,
        polarization: Literal["tm", "te"] = "tm",
    ) -> MaterialGrid:
        """Convert a uniform raster result into the current 2D or 3D solver grid.

        Reduced TMz/TEz output identifies a 2D result automatically. Full output
        remains 3D, including a legitimate one-cell-thick 3D raster; pass
        ``dimensions=2`` explicitly when repurposing full output for 2D.
        """

        support_tensors = dict(getattr(result, "yee_tensors", {}))
        edges = getattr(result, "grid_edges", None)
        if edges is None or len(edges) != 3:
            raise ValueError("RasterResult must retain x, y, and z grid edges.")
        if dimensions is None:
            available = set(support_tensors)
            full_output = {
                "epsilon_ex",
                "epsilon_ey",
                "epsilon_ez",
                "conductivity_ex",
                "conductivity_ey",
                "conductivity_ez",
                "mu_hx",
                "mu_hy",
                "mu_hz",
            }
            tm_output = {"epsilon_ez", "conductivity_ez", "mu_hx", "mu_hy"}
            te_output = {
                "epsilon_ex",
                "epsilon_ey",
                "conductivity_ex",
                "conductivity_ey",
                "mu_hz",
            }
            if full_output <= available:
                dimensions = 3
            elif (tm_output <= available and not (te_output & available)) or (
                te_output <= available and not (tm_output & available)
            ):
                dimensions = 2
            else:
                raise ValueError(
                    "Cannot infer RasterResult dimensionality from its Yee "
                    "components; pass dimensions=2 or dimensions=3 explicitly."
                )
        if dimensions not in (2, 3):
            raise ValueError("dimensions must be 2 or 3.")
        polarization = normalize_polarization_2d(polarization)
        component_specs = (
            (
                ("eps_x", "epsilon_ex", 0),
                ("eps_y", "epsilon_ey", 1),
                ("eps_z", "epsilon_ez", 2),
                ("sig_x", "conductivity_ex", 0),
                ("sig_y", "conductivity_ey", 1),
                ("sig_z", "conductivity_ez", 2),
                ("mu_hx", "mu_hx", 0),
                ("mu_hy", "mu_hy", 1),
                ("mu_hz", "mu_hz", 2),
            )
            if dimensions == 3
            else (
                (
                    ("eps_z", "epsilon_ez", 2),
                    ("sig_z", "conductivity_ez", 2),
                    ("mu_hx", "mu_hx", 0),
                    ("mu_hy", "mu_hy", 1),
                )
                if polarization == "tm"
                else (
                    ("eps_x", "epsilon_ex", 0),
                    ("eps_y", "epsilon_ey", 1),
                    ("sig_x", "conductivity_ex", 0),
                    ("sig_y", "conductivity_ey", 1),
                    ("mu_hz", "mu_hz", 2),
                )
            )
        )
        required = {source for _target, source, _axis in component_specs}
        missing = required - set(support_tensors)
        if missing:
            raise ValueError(
                f"RasterResult omits solver components: {', '.join(sorted(missing))}."
            )
        spacings = tuple(np.diff(np.asarray(axis, dtype=float)) for axis in edges)
        resolution = float(spacings[0][0])
        active_axes = spacings if dimensions == 3 else spacings[:2]
        if any(
            not np.allclose(axis, resolution, rtol=1e-12, atol=0.0)
            for axis in active_axes
        ):
            raise ValueError(
                "BeamZ Simulation requires one uniform spacing on every active axis."
            )
        smoothing = str(getattr(result, "smoothing", "volume"))
        materials = {
            target: _support_diagonal(support_tensors[source], axis)
            for target, source, axis in component_specs
        }
        yee_tensors = {
            target: np.asarray(support_tensors[source])
            for target, source, _axis in component_specs
            if target.startswith("eps_")
        }
        tensors = dict(getattr(result, "tensors", {}))
        missing_tensors = {"epsilon", "mu", "conductivity"} - set(tensors)
        if missing_tensors:
            raise ValueError(
                "RasterResult omits cell tensors: "
                f"{', '.join(sorted(missing_tensors))}."
            )
        epsilon_tensor = np.asarray(tensors["epsilon"])
        mu_tensor = np.asarray(tensors["mu"])
        conductivity_tensor = np.asarray(tensors["conductivity"])
        if np.any(np.abs(_tensor_off_diagonal(conductivity_tensor)) > 1e-10):
            raise ValueError(
                "BeamZ's full-tensor update does not support off-diagonal "
                "conductivity. Use diagonal conductivity coefficients."
            )
        if (
            dimensions == 2
            and epsilon_tensor.shape[0] == 6
            and np.any(np.abs(epsilon_tensor[[4, 5]]) > 1e-10)
        ):
            raise ValueError(
                "A 2D simulation requires permittivity without xz or yz coupling."
            )
        has_full_permittivity = dimensions == 3 or polarization == "te"
        has_full_permittivity = has_full_permittivity and any(
            np.any(np.abs(np.asarray(values)[3:]) > 1e-10)
            for values in yee_tensors.values()
        )
        if has_full_permittivity:
            if "epsilon_node" not in support_tensors:
                raise ValueError(
                    "Full-tensor permittivity requires the node-centered "
                    "off-diagonal raster tensor."
                )
            yee_tensors["eps_node"] = np.asarray(support_tensors["epsilon_node"])
        else:
            yee_tensors = {}
        if has_full_permittivity and np.any(np.abs(conductivity_tensor) > 1e-10):
            raise ValueError(
                "Full-tensor permittivity currently requires zero conductivity."
            )
        mu_diagonal = _tensor_diagonal(mu_tensor)
        if not (
            np.allclose(mu_diagonal, 1.0, rtol=1e-10, atol=1e-12)
            and np.allclose(
                _tensor_off_diagonal(mu_tensor), 0.0, rtol=1e-10, atol=1e-12
            )
        ):
            raise ValueError(
                "BeamZ's current FDTD update supports only unit permeability. "
                "The tensor raster remains available for standalone use."
            )
        epsilon_diagonal = _tensor_diagonal(epsilon_tensor)
        conductivity_diagonal = _tensor_diagonal(conductivity_tensor)
        epsilon = np.mean(epsilon_diagonal, axis=0)
        conductivity = np.mean(conductivity_diagonal, axis=0)
        permeability = np.mean(mu_diagonal, axis=0)
        if dimensions == 2:
            if epsilon.shape[0] != 1:
                raise ValueError("A 2D MaterialGrid requires exactly one z cell.")
            if polarization == "tm":
                epsilon = epsilon_diagonal[2, 0]
                conductivity = conductivity_diagonal[2, 0]
                permeability = 0.5 * (mu_diagonal[0, 0] + mu_diagonal[1, 0])
            else:
                epsilon = 0.5 * (epsilon_diagonal[0, 0] + epsilon_diagonal[1, 0])
                conductivity = 0.5 * (
                    conductivity_diagonal[0, 0] + conductivity_diagonal[1, 0]
                )
                permeability = mu_diagonal[2, 0]
            materials = {
                name: np.asarray(value)[0] for name, value in materials.items()
            }
            tensors = {name: np.asarray(value)[:, 0] for name, value in tensors.items()}
            yee_tensors = {
                name: np.asarray(value)[:, 0] for name, value in yee_tensors.items()
            }
        return cls(
            epsilon,
            conductivity,
            permeability,
            resolution,
            tuple(int(value) for value in epsilon.shape),
            materials,
            tensors,
            smoothing,
            (
                float(np.asarray(edges[0])[0]),
                float(np.asarray(edges[1])[0]),
                float(np.asarray(edges[2])[0]),
            ),
            polarization if dimensions == 2 else None,
            yee_tensors,
        )

    def field_arrays(self) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """Return permittivity, conductivity, and permeability arrays."""
        return self.permittivity, self.conductivity, self.permeability

    @property
    def uses_direct_yee_materials(self) -> bool:
        """Return whether propagation must consume the retained Yee arrays."""

        if not self.yee_materials:
            return False
        if self.smoothing == "farjadpour_diagonal" or self.uses_full_permittivity:
            return True
        for name in ("epsilon", "conductivity"):
            if name not in self.tensors:
                continue
            diagonal = _tensor_diagonal(self.tensors[name])
            if not (
                np.allclose(diagonal[0], diagonal[1], rtol=1e-10, atol=1e-12)
                and np.allclose(diagonal[0], diagonal[2], rtol=1e-10, atol=1e-12)
            ):
                return True
        return False

    @property
    def uses_full_permittivity(self) -> bool:
        """Return whether propagation needs off-diagonal electric coupling."""

        return any(
            np.any(np.abs(np.asarray(values)[3:]) > 1e-10)
            for values in self.yee_tensors.values()
        )

    def canonical_spec(self):
        """Return values defining material-grid cache identity."""
        return (
            *self.field_arrays(),
            self.resolution,
            self.shape,
            self.yee_materials,
            self.tensors,
            self.smoothing,
            self.origin,
            self.polarization,
            self.yee_tensors,
        )

    def __eq__(self, other):
        if not isinstance(other, MaterialGrid):
            return NotImplemented
        return cache_token(self.canonical_spec()) == cache_token(other.canonical_spec())

    def __hash__(self):
        return hash(cache_token(self.canonical_spec()))


def build_material_grid(
    design,
    resolution: float,
    *,
    grid_type: str = "auto",
    force_recompute: bool = False,
    progress: bool = False,
    **kwargs,
) -> MaterialGrid:
    """Discretize a design into the immutable solver material grid.

    Parameters
    ----------
    design : Design
        Immutable geometry and material specification to rasterize.
    resolution : float
        Uniform spatial cell size in metres.
    grid_type : str, default="auto"
        Dimensionality policy: ``"auto"``, ``"2d"``, or ``"3d"``.
    force_recompute : bool, default=False
        Rebuild the grid even when a matching cached discretization exists.
    progress : bool, default=False
        Emit rasterization progress when the design supports it.
    **kwargs
        Additional rasterizer-specific options.

    Returns
    -------
    MaterialGrid
        Read-only cell summaries and compatible direct Yee coefficients.

    """
    return design.rasterize(
        resolution,
        grid_type=grid_type,
        force_recompute=force_recompute,
        progress=progress,
        **kwargs,
    )
