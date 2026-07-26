"""Rasterized mode-solver API for Beamz-style grid inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import xarray as xr

from ._scipy import solve_diagonal_scipy_reference, solve_tensorial_scipy_reference
from .constants import C_0
from .models import BoundaryCondition, BoundarySpec, Materials, PmlSpec
from .result import Result

_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


# Public entry points keep geometry handling out of the solver core: callers
# supply already-rasterized material arrays and grid coordinates.
def solve_grid(
    *,
    eps_xx: np.ndarray,
    eps_yy: np.ndarray | None = None,
    eps_zz: np.ndarray | None = None,
    eps_xy: np.ndarray | None = None,
    eps_xz: np.ndarray | None = None,
    eps_yx: np.ndarray | None = None,
    eps_yz: np.ndarray | None = None,
    eps_zx: np.ndarray | None = None,
    eps_zy: np.ndarray | None = None,
    mu_xx: np.ndarray | None = None,
    mu_yy: np.ndarray | None = None,
    mu_zz: np.ndarray | None = None,
    mu_xy: np.ndarray | None = None,
    mu_xz: np.ndarray | None = None,
    mu_yx: np.ndarray | None = None,
    mu_yz: np.ndarray | None = None,
    mu_zx: np.ndarray | None = None,
    mu_zy: np.ndarray | None = None,
    x_edges: Sequence[float],
    y_edges: Sequence[float],
    freqs: Sequence[float] | None = None,
    wavelength: float | Sequence[float] | None = None,
    num_modes: int = 1,
    target_neff: float | None = None,
    pml: PmlSpec | tuple[int, int] | None = None,
    boundary: BoundarySpec | tuple[str, str] | None = None,
    direction: Literal["+", "-"] = "+",
    components: Sequence[str] | None = None,
    normal_axis: Literal[0, 1, 2] = 2,
    normal_coordinate: float = 0.0,
    krylov_dim: int | None = None,
) -> Result:
    """Solve modes from rasterized material components and grid edges.

    This is the core API Beamz should target: geometry and materials are already
    sampled on a two-dimensional mode-plane grid. Coordinates are in microns and
    frequencies are in Hz, matching the rest of BeamZ.
    """
    material_grid = Materials.from_components(
        eps_xx=eps_xx,
        eps_yy=eps_yy,
        eps_zz=eps_zz,
        eps_xy=eps_xy,
        eps_xz=eps_xz,
        eps_yx=eps_yx,
        eps_yz=eps_yz,
        eps_zx=eps_zx,
        eps_zy=eps_zy,
        mu_xx=mu_xx,
        mu_yy=mu_yy,
        mu_zz=mu_zz,
        mu_xy=mu_xy,
        mu_xz=mu_xz,
        mu_yx=mu_yx,
        mu_yz=mu_yz,
        mu_zx=mu_zx,
        mu_zy=mu_zy,
        x_edges=x_edges,
        y_edges=y_edges,
        normal_axis=normal_axis,
        normal_coordinate=normal_coordinate,
    )

    # Once the component arrays have been normalized into Materials, the full
    # solve path is identical to solve_modes.
    return _solve_materials(
        material_grid=material_grid,
        freqs=freqs,
        wavelength=wavelength,
        num_modes=num_modes,
        target_neff=target_neff,
        pml=pml,
        boundary=boundary,
        direction=direction,
        components=components,
        krylov_dim=krylov_dim,
    )


def _solve_materials(
    *,
    material_grid: Materials,
    freqs: Sequence[float] | None = None,
    wavelength: float | Sequence[float] | None = None,
    num_modes: int = 1,
    target_neff: float | None = None,
    pml: PmlSpec | tuple[int, int] | None = None,
    boundary: BoundarySpec | tuple[str, str] | None = None,
    direction: Literal["+", "-"] = "+",
    components: Sequence[str] | None = None,
    krylov_dim: int | None = None,
) -> Result:
    """Solve modes for an already-rasterized material tensor grid.

    This is the preferred BeamZ integration point. BeamZ owns geometry and
    material rasterization; this package owns the sparse SciPy mode solve and
    field reconstruction on the supplied grid.
    """
    # Main solver orchestration layer. It validates user-facing grid objects,
    # solves one frequency at a time, then wraps flattened solver outputs into
    # coordinate-aware xarray arrays.
    if not isinstance(material_grid, Materials):
        raise TypeError("material_grid must be a Materials")

    # Validate edge arrays against the tensor shape and resolve the frequency
    # input before any expensive sparse work starts.
    shape = material_grid.shape
    x_edges_arr = _validate_edges("x_edges", material_grid.grid.x_edges, shape[0])
    y_edges_arr = _validate_edges("y_edges", material_grid.grid.y_edges, shape[1])
    solve_freqs = _resolve_freqs(freqs=freqs, wavelength=wavelength)

    # Normalize scalar options and compact model objects at the API boundary.
    if num_modes <= 0:
        raise ValueError("num_modes must be positive")
    if direction not in {"+", "-"}:
        raise ValueError("direction must be '+' or '-'")
    pml_spec = _resolve_pml_spec(pml)
    boundary_spec = _resolve_boundary_spec(boundary)
    # Component filtering happens after solving so field reconstruction still has
    # every component needed for normalization and coordinate conversion.
    requested_components = tuple(components or _COMPONENTS)
    unknown = set(requested_components).difference(_COMPONENTS)
    if unknown:
        raise ValueError(f"unknown field component(s): {', '.join(sorted(unknown))}")

    # Accumulate raw NumPy rows first. Building xarray objects once at the end
    # keeps component filtering and frequency stacking simple.
    n_rows = []
    solver_runs = []
    fields_by_component: dict[str, list[np.ndarray]] = {
        component: [] for component in requested_components
    }
    for freq in solve_freqs:
        # The sparse solver is frequency-local; multi-frequency solves are a
        # deterministic loop plus a final stack into xarray dimensions.
        n_complex, fields, solver_info = _solve_one_frequency(
            x_edges=x_edges_arr,
            y_edges=y_edges_arr,
            freq=float(freq),
            num_modes=num_modes,
            target_neff=target_neff,
            pml_spec=pml_spec,
            direction=direction,
            krylov_dim=krylov_dim,
            boundary_spec=boundary_spec,
            material_grid=material_grid,
        )
        # The solver uses local coordinates where local z is the propagation
        # normal. Convert field labels back to the global x/y/z axes requested
        # by the material grid before exposing them.
        fields = _local_fields_to_global(
            fields, normal_axis=material_grid.grid.normal_axis
        )
        n_rows.append(n_complex)
        solver_runs.append(solver_info)
        for component, values in fields.items():
            if component in fields_by_component:
                fields_by_component[component].append(values)

    # Convert accumulated lists into the public Result layout only once.
    n_values = np.asarray(n_rows, dtype=np.complex128)
    field_components = _field_data_arrays(
        fields_by_component,
        x_edges_arr,
        y_edges_arr,
        solve_freqs,
        normal_axis=material_grid.grid.normal_axis,
        normal_coordinate=material_grid.grid.normal_coordinate,
    )
    coords = {"f": np.asarray(solve_freqs), "mode_index": np.arange(n_values.shape[1])}
    return Result(
        n_complex=xr.DataArray(n_values, dims=("f", "mode_index"), coords=coords),
        field_components=field_components,
        solver_info={
            "backend": solver_runs[0].get("backend") if solver_runs else None,
            "runs": solver_runs,
            "pml": pml_spec.as_dict(),
            "boundary": boundary_spec.as_dict(),
            "normal_axis": material_grid.grid.normal_axis,
        },
    )


def _solve_one_frequency(
    *,
    material_grid: Materials,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    freq: float,
    num_modes: int,
    target_neff: float | None,
    pml_spec: PmlSpec,
    direction: str,
    krylov_dim: int | None,
    boundary_spec: BoundarySpec,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    """Select the sparse formulation and solve a single frequency."""
    # Forward and backward spacings represent the local Yee grid. The derivative
    # builders need both because E and H components are staggered.
    dlf = (np.diff(x_edges), np.diff(y_edges))
    dlb = (_dual_steps(dlf[0]), _dual_steps(dlf[1]))

    # Flatten tensors to the layout expected by scipy_reference while preserving
    # the original grid metadata for result coordinates.
    eps_tensor = material_grid.flat_eps_tensor()
    mu_tensor = material_grid.flat_mu_tensor()
    if target_neff is None:
        # Default to the highest local material index. Users can supply a target
        # explicitly when hunting for modes near another branch.
        target_neff = float(np.sqrt(np.max(np.abs(eps_tensor))))
    target_neff = _shift_target_neff(float(target_neff))

    if not material_grid.is_diagonal:
        # User supplied a full tensor grid with no coordinate transform.
        return _solve_one_frequency_scipy_tensorial(
            eps_tensor=eps_tensor,
            mu_tensor=mu_tensor,
            dlf=dlf,
            dlb=dlb,
            freq=freq,
            num_modes=num_modes,
            target_neff=target_neff,
            pml_spec=pml_spec,
            direction=direction,
            krylov_dim=krylov_dim,
            boundary_spec=boundary_spec,
        )
    # Ordinary scalar/diagonal grids use the diagonal sparse formulation.
    return _solve_one_frequency_scipy(
        eps_tensor=eps_tensor,
        mu_tensor=mu_tensor,
        dlf=dlf,
        dlb=dlb,
        freq=freq,
        num_modes=num_modes,
        target_neff=target_neff,
        pml_spec=pml_spec,
        direction=direction,
        krylov_dim=krylov_dim,
        boundary_spec=boundary_spec,
    )


def _solve_one_frequency_scipy(
    *,
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
    dlf: tuple[np.ndarray, np.ndarray],
    dlb: tuple[np.ndarray, np.ndarray],
    freq: float,
    num_modes: int,
    target_neff: float,
    pml_spec: PmlSpec,
    direction: str,
    krylov_dim: int | None,
    boundary_spec: BoundarySpec,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    """Run the diagonal SciPy solver and reshape its fields."""
    # The diagonal solver has two transverse unknowns per cell: Ex and Ey.
    nx = len(dlf[0])
    ny = len(dlf[1])
    actual_krylov_dim = 32 if krylov_dim is None else int(krylov_dim)

    # Convert frequency into the nondimensional derivative scale and angular
    # frequency expected by the sparse reference implementation.
    n_complex, fields, solver_info = solve_diagonal_scipy_reference(
        eps_tensor=eps_tensor,
        mu_tensor=mu_tensor,
        dlf=dlf,
        dlb=dlb,
        num_modes=num_modes,
        neff_guess=target_neff,
        direction=direction,
        derivative_scale=C_0 / (2 * np.pi * freq),
        omega=2 * np.pi * freq,
        num_pml=pml_spec.num_cells,
        pml_profile=pml_spec.profile_dict(),
        dmin_pml=boundary_spec.dmin_pml,
        dmin_pmc=boundary_spec.dmin_pmc,
        krylov_dim=actual_krylov_dim,
        initial_vector=_default_initial_vector(2 * nx * ny, shape=(nx, ny)),
    )

    # Solver fields are flattened by component; reshape them back to grid
    # tensors and attach dispatch metadata for diagnostics.
    return (
        n_complex,
        _fields_to_grid(fields, (nx, ny)),
        _solver_info_with_context(
            solver_info,
            backend_kind="diagonal_scipy_reference",
            shape=(nx, ny),
            krylov_dim=actual_krylov_dim,
        ),
    )


def _solve_one_frequency_scipy_tensorial(
    *,
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
    dlf: tuple[np.ndarray, np.ndarray],
    dlb: tuple[np.ndarray, np.ndarray],
    freq: float,
    num_modes: int,
    target_neff: float,
    pml_spec: PmlSpec,
    direction: str,
    krylov_dim: int | None,
    boundary_spec: BoundarySpec,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    """Run the tensorial SciPy solver and reshape its fields."""
    # The tensorial solver keeps four transverse unknowns per cell:
    # Ex, Ey, Hx, and Hy.
    nx = len(dlf[0])
    ny = len(dlf[1])
    actual_krylov_dim = 32 if krylov_dim is None else int(krylov_dim)

    # Full tensor coupling uses the first-order operator and solves directly for
    # effective index rather than n_eff squared.
    n_complex, fields, solver_info = solve_tensorial_scipy_reference(
        eps_tensor=eps_tensor,
        mu_tensor=mu_tensor,
        dlf=dlf,
        dlb=dlb,
        num_modes=num_modes,
        neff_guess=target_neff,
        direction=direction,
        derivative_scale=C_0 / (2 * np.pi * freq),
        omega=2 * np.pi * freq,
        num_pml=pml_spec.num_cells,
        pml_profile=pml_spec.profile_dict(),
        dmin_pml=boundary_spec.dmin_pml,
        dmin_pmc=boundary_spec.dmin_pmc,
        krylov_dim=actual_krylov_dim,
        initial_vector=_default_initial_vector(4 * nx * ny, shape=(nx, ny)),
    )

    # Keep return structure identical to the diagonal path so solve_modes does
    # not need backend-specific field handling.
    return (
        n_complex,
        _fields_to_grid(fields, (nx, ny)),
        _solver_info_with_context(
            solver_info,
            backend_kind="tensorial_scipy_reference",
            shape=(nx, ny),
            krylov_dim=actual_krylov_dim,
        ),
    )


def _resolve_pml_spec(
    pml: PmlSpec | tuple[int, int] | None,
) -> PmlSpec:
    """Normalize user PML input into a PmlSpec."""
    # The public API accepts compact tuples, but the solver only sees validated
    # PmlSpec objects.
    if pml is None:
        return PmlSpec()
    if isinstance(pml, PmlSpec):
        return pml
    return PmlSpec.from_num_cells(pml)


def _resolve_boundary_spec(
    boundary: BoundarySpec | tuple[str, str] | None,
) -> BoundarySpec:
    """Normalize user boundary input into a BoundarySpec."""
    # BoundarySpec handles case normalization and allowed-value validation.
    if boundary is None:
        return BoundarySpec()
    if isinstance(boundary, BoundarySpec):
        return boundary
    return BoundarySpec(low=cast(tuple[BoundaryCondition, BoundaryCondition], boundary))


def _solver_info_with_context(
    solver_info: dict[str, object],
    *,
    backend_kind: str,
    shape: tuple[int, int],
    krylov_dim: int,
) -> dict[str, object]:
    """Attach Python-side dispatch metadata to solver diagnostics."""
    # Add enough Python-side context for saved Result files and benchmark
    # reports to be self-describing.
    out = dict(solver_info)
    out["backend_kind"] = backend_kind
    out["shape"] = shape
    out["krylov_dim"] = krylov_dim
    out["phase_convention"] = "dominant_e_real_positive"
    out["normalization"] = "lorentz_orthogonal_unit_transverse_power"
    return out


def _shift_target_neff(target_neff: float) -> float:
    """Nudge the shift target away from exact eigenvalues for ARPACK stability."""
    # Exact shifts can make shift-invert factorization singular; the tiny nudge
    # preserves the requested target while avoiding exact equality.
    target_shift = float(10 * np.finfo(np.float32).eps)
    if abs(target_shift) > abs(target_neff * target_shift):
        return target_neff + target_shift
    return target_neff * (1.0 + target_shift)


def _validate_edges(name: str, values: Sequence[float], cell_count: int) -> np.ndarray:
    """Validate mode-plane edge coordinates against a cell count."""
    # Edge validation is repeated here because callers may provide an existing
    # Materials object from outside the dataclass constructors.
    edges = np.asarray(values, dtype=float)
    if edges.shape != (cell_count + 1,):
        raise ValueError(f"{name} must have length {cell_count + 1}")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return edges


def _resolve_freqs(
    *,
    freqs: Sequence[float] | None,
    wavelength: float | Sequence[float] | None,
) -> tuple[float, ...]:
    """Resolve exactly one frequency or wavelength input into frequencies."""
    # Accept either frequencies or wavelengths, never both, so the solve axis is
    # unambiguous.
    if (freqs is None) == (wavelength is None):
        raise ValueError("provide exactly one of freqs or wavelength")
    if freqs is not None:
        values = tuple(float(freq) for freq in freqs)
    else:
        # Wavelengths are in microns because C_0 is stored in um/s.
        wavelengths = np.asarray(wavelength, dtype=float).reshape(-1)
        values = tuple(float(C_0 / value) for value in wavelengths)
    if not values or any(not np.isfinite(freq) or freq <= 0 for freq in values):
        raise ValueError("frequencies must be finite and positive")
    return values


def _dual_steps(primal_steps: np.ndarray) -> np.ndarray:
    """Return backward-grid step sizes from primal cell widths."""
    # Interior backward steps live halfway between adjacent primal cells.
    if len(primal_steps) == 1:
        return primal_steps.copy()
    return np.hstack((primal_steps[0], (primal_steps[:-1] + primal_steps[1:]) / 2))


def _fields_to_grid(
    fields: list[np.ndarray], shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    """Reshape flattened solver fields into x/y/mode arrays."""
    # scipy_reference returns [mode, flattened_cell]; public results use
    # [x, y, mode] per component.
    nx, ny = shape
    mode_count = fields[0].shape[0]
    out = {}
    for component, values_by_mode in zip(_COMPONENTS, fields, strict=True):
        values = np.asarray(values_by_mode).reshape(mode_count, nx, ny)
        out[component] = np.moveaxis(values, 0, -1)
    return out


def _local_fields_to_global(
    fields: dict[str, np.ndarray], *, normal_axis: int
) -> dict[str, np.ndarray]:
    """Map solver-local field components onto global x/y/z component names.

    The SciPy solver uses local coordinates where local z is the
    propagation-normal axis. For x- or y-normal planes, component labels must be
    permuted back to global coordinates before returning a Result.
    """

    axis_names = ("x", "y", "z")
    if normal_axis not in {0, 1, 2}:
        raise ValueError("normal_axis must be 0, 1, or 2")

    # Local solver axes are always tangential-x, tangential-y, normal-z.
    local_to_global = (*(axis for axis in range(3) if axis != normal_axis), normal_axis)
    out: dict[str, np.ndarray] = {}
    for prefix in ("E", "H"):
        for local_axis, global_axis in enumerate(local_to_global):
            local_name = f"{prefix}{axis_names[local_axis]}"
            global_name = f"{prefix}{axis_names[global_axis]}"
            if local_name in fields:
                # For y-normal planes the local (x, z, y) order is left-handed;
                # flipping the local-y tangential component restores physical +y flux.
                sign = -1.0 if normal_axis == 1 and local_axis == 1 else 1.0
                out[global_name] = sign * fields[local_name]
    return out


def _field_data_arrays(
    fields_by_component: dict[str, list[np.ndarray]],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    freqs: Sequence[float],
    *,
    normal_axis: int,
    normal_coordinate: float,
) -> dict[str, xr.DataArray]:
    """Wrap solved field arrays in coordinate-aware xarray objects."""
    axis_names = ("x", "y", "z")
    if normal_axis not in {0, 1, 2}:
        raise ValueError("normal_axis must be 0, 1, or 2")

    # Determine which two global coordinates correspond to the solver's local
    # x/y plane.
    tangential = tuple(axis for axis in range(3) if axis != normal_axis)
    coord0 = (x_edges[:-1] + x_edges[1:]) / 2
    coord1 = (y_edges[:-1] + y_edges[1:]) / 2

    # Store width coordinates beside center coordinates so integrations can use
    # physical cell areas without reconstructing the original edges.
    coords = {
        axis_names[tangential[0]]: coord0,
        f"{axis_names[tangential[0]]}_width": (
            axis_names[tangential[0]],
            np.diff(x_edges),
        ),
        axis_names[tangential[1]]: coord1,
        f"{axis_names[tangential[1]]}_width": (
            axis_names[tangential[1]],
            np.diff(y_edges),
        ),
        axis_names[normal_axis]: np.asarray([normal_coordinate]),
        f"{axis_names[normal_axis]}_width": (axis_names[normal_axis], np.ones(1)),
        "f": np.asarray(freqs),
        "mode_index": None,
    }
    dims = (
        axis_names[tangential[0]],
        axis_names[tangential[1]],
        axis_names[normal_axis],
        "f",
        "mode_index",
    )
    out = {}
    for component, rows in fields_by_component.items():
        # Stack frequencies on axis 2, then insert the singleton normal axis so
        # every component has x/y/z/f/mode_index dimensions.
        values = np.stack(rows, axis=2)[:, :, None, :, :]
        component_coords = dict(coords)
        component_coords["mode_index"] = np.arange(values.shape[-1])
        out[component] = xr.DataArray(
            values,
            dims=dims,
            coords=component_coords,
            attrs={"normal_dim": axis_names[normal_axis]},
        )
    return out


def _default_initial_vector(
    size: int, shape: tuple[int, int] | None = None
) -> np.ndarray:
    """Build a deterministic ARPACK seed vector for reproducible solves."""
    # When grid shape is known, build a seed with the same flattened cell order
    # as the solver state vector.
    if shape is not None and size % (shape[0] * shape[1]) == 0:
        nx, ny = shape
        multiplier = size // (nx * ny)
        rng = np.random.default_rng(0)
        vector = rng.random((nx, ny, multiplier)) + 1j * rng.random(
            (nx, ny, multiplier)
        )

        # Zero low-edge rows when possible so the initial guess is compatible
        # with symmetry-wall derivative stencils.
        if nx > 1:
            vector[0, :, :] = 0
        if ny > 1:
            vector[:, 0, :] = 0
        stacked = np.concatenate(tuple(vector[ix, :, :] for ix in range(nx)), axis=0)
        return stacked.flatten("F")

    # Fallback for callers that only know the operator size.
    index = np.arange(1, size + 1, dtype=float)
    return np.sin(0.37 * index) + 1j * np.cos(0.53 * index)
