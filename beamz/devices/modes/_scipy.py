"""Readable SciPy implementation for the mode solver.

This module assembles the sparse mode-solver paths in plain Python/SciPy. It
keeps the numerical contract inspectable by users who want to audit the
finite-difference operators against SciPy/ARPACK.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

ETA0 = 376.730_313_666_853_5
MU0 = 1.256_637_062_12e-12
C0 = 2.997_924_58e14
EPSILON0 = 1.0 / (MU0 * C0 * C0)


SparseSolveResult = tuple[np.ndarray, list[np.ndarray], dict[str, object]]


# The SciPy reference layer stays deliberately explicit so finite-difference
# assembly can be audited against the physics model.
@dataclass
class _ModeFields:
    """Mutable six-component field container used during normalization."""

    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray
    hx: np.ndarray
    hy: np.ndarray
    hz: np.ndarray

    def components(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return field components in electric-then-magnetic order."""
        return self.ex, self.ey, self.ez, self.hx, self.hy, self.hz

    def add_scaled(self, other: _ModeFields, scale: complex) -> None:
        """Add another mode into this mode with a complex scale factor."""
        # Orthogonalization operates in-place to avoid repeatedly allocating six
        # large component arrays per mode.
        for left, right in zip(self.components(), other.components(), strict=True):
            left += scale * right


def solve_diagonal_scipy_reference(
    *,
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
    dlf: tuple[np.ndarray, np.ndarray],
    dlb: tuple[np.ndarray, np.ndarray],
    num_modes: int,
    neff_guess: float,
    direction: str,
    derivative_scale: float,
    omega: float | None = None,
    num_pml: tuple[int, int] = (0, 0),
    pml_profile: dict[str, float | int] | None = None,
    dmin_pml: tuple[bool, bool] = (True, True),
    dmin_pmc: tuple[bool, bool] = (False, False),
    krylov_dim: int = 32,
    initial_vector: np.ndarray | None = None,
) -> SparseSolveResult:
    """Solve the diagonal sparse eigenproblem with SciPy/ARPACK.

    This is the reduced ``[Ex, Ey]`` transverse eigenproblem for diagonal
    material tensors.
    """

    sparse, spla, scipy_linalg = _import_scipy()

    # The diagonal formulation has two transverse electric unknowns per cell.
    nx = len(dlf[0])
    ny = len(dlf[1])
    n = nx * ny

    # Validate flattened tensor layout before assembling sparse operators.
    if (
        eps_tensor.shape != mu_tensor.shape
        or eps_tensor.shape[:2] != (3, 3)
        or eps_tensor.shape[-1] != n
    ):
        raise ValueError(
            "eps_tensor and mu_tensor must both have shape (3, 3, len(dlf[0]) * len(dlf[1]))"
        )
    if num_modes <= 0:
        raise ValueError("num_modes must be positive")

    # Build finite-difference derivative matrices, including optional PML
    # complex-coordinate stretching.
    derivatives = _create_derivative_matrices(
        sparse,
        eps_tensor=eps_tensor,
        mu_tensor=mu_tensor,
        shape=(nx, ny),
        dlf=dlf,
        dlb=dlb,
        omega=omega,
        num_pml=num_pml,
        pml_profile=pml_profile,
        dmin_pml=dmin_pml,
        dmin_pmc=dmin_pmc,
        scale=float(derivative_scale),
    )

    # Assemble the reduced Ex/Ey eigen-operator for diagonal material tensors.
    operators = _assemble_diagonal_operators(sparse, eps_tensor, mu_tensor, derivatives)
    operator = cast(Any, operators["mat"])
    eig_guess = complex(-(neff_guess * neff_guess), 0.0)

    # ARPACK can solve a real problem faster when the operator and shift are
    # numerically real; otherwise it keeps the complex path.
    operator, arpack_initial_vector, arpack_guess = _real_arpack_problem_if_close(
        operator, initial_vector, eig_guess
    )
    values, vectors = _selected_eigenpairs(
        operator,
        num_modes=num_modes,
        sigma=arpack_guess,
        krylov_dim=krylov_dim,
        initial_vector=arpack_initial_vector,
        spla=spla,
        scipy_linalg=scipy_linalg,
    )

    # Normalize residuals by eigenvector norm so diagnostics are comparable
    # across modes.
    residuals = np.asarray(
        [
            np.linalg.norm(
                operator @ vectors[:, index] - values[index] * vectors[:, index]
            )
            for index in range(len(values))
        ],
        dtype=float,
    )
    vector_norms = np.maximum(np.linalg.norm(vectors, axis=0), np.finfo(float).eps)
    residuals = residuals / vector_norms

    # Convert eigenvalues of -n_eff^2 back to n_eff and sort by real index.
    modes: list[tuple[complex, np.ndarray, float]] = []
    for value, vector, residual in zip(values, vectors.T, residuals, strict=True):
        modes.append(
            (
                complex(np.sqrt(-value + 0j)),
                np.asarray(vector, dtype=np.complex128),
                float(residual),
            )
        )
    modes.sort(key=lambda item: item[0].real, reverse=True)

    # Precompute z-component inverse material matrices used during field
    # reconstruction.
    inv_eps_zz = sparse.diags(1.0 / eps_tensor[2, 2, :], format="csc")
    inv_mu_zz = sparse.diags(1.0 / mu_tensor[2, 2, :], format="csc")
    dxf, dxb, dyf, dyb = derivatives
    cell_areas = np.repeat(np.asarray(dlf[0], dtype=float), ny) * np.tile(
        np.asarray(dlf[1], dtype=float), nx
    )

    # Reconstruct all six field components from the transverse eigenvectors.
    n_complex: list[complex] = []
    mode_fields: list[_ModeFields] = []
    sorted_residuals: list[float] = []
    for mode_n, vector, residual in modes:
        # The eigenvector stores Ex then Ey in flattened cell order.
        ex = vector[:n].copy()
        ey = vector[n:].copy()
        denom = complex(-mode_n.imag, mode_n.real)

        # Magnetic transverse components come from the Q operator divided by the
        # effective-index factor.
        h_field = operators["qmat"] @ vector
        hx = np.asarray(h_field[:n] / denom, dtype=np.complex128)
        hy = np.asarray(h_field[n:] / denom, dtype=np.complex128)

        # Longitudinal H follows from the curl of transverse E.
        hz_source = dxf @ ey - dyf @ ex
        hz = np.asarray(inv_mu_zz @ hz_source, dtype=np.complex128)

        # Longitudinal E follows from the curl of the material electric block.
        h_partial = np.asarray(
            (operators["q_ep"] @ vector) / denom, dtype=np.complex128
        )
        ez_source = dxb @ h_partial[n:] - dyb @ h_partial[:n]
        ez = np.asarray(inv_eps_zz @ ez_source, dtype=np.complex128)

        # Convert magnetic fields to the same physical normalization convention
        # used by Result overlap calculations.
        h_scale = -1j / ETA0
        hx *= h_scale
        hy *= h_scale
        hz *= h_scale

        # Backward propagation flips transverse H and longitudinal E.
        if direction == "-":
            hx *= -1.0
            hy *= -1.0
            ez *= -1.0

        # Accumulate raw modes first; normalization is applied to the full set
        # so modes can be Lorentz-orthogonalized together.
        n_complex.append(mode_n)
        sorted_residuals.append(residual)
        mode_fields.append(_ModeFields(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz))

    # Apply final normalization and package field components by component name.
    orthogonalization = _lorentz_orthogonalize_and_normalize(mode_fields, cell_areas)
    fields = [
        np.asarray(
            [getattr(mode, component) for mode in mode_fields], dtype=np.complex128
        )
        for component in ("ex", "ey", "ez", "hx", "hy", "hz")
    ]
    return (
        np.asarray(n_complex, dtype=np.complex128),
        fields,
        {
            "backend": "scipy_arpack_reference",
            "operator_size": int(operator.shape[0]),
            "operator_nnz": int(operator.nnz),
            "residuals": np.asarray(sorted_residuals, dtype=float),
            "power_norms": orthogonalization["power_norms"],
            "lorentz_norms": orthogonalization["lorentz_norms"],
            "lorentz_orthogonality_error": orthogonalization[
                "lorentz_orthogonality_error"
            ],
        },
    )


def solve_tensorial_scipy_reference(
    *,
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
    dlf: tuple[np.ndarray, np.ndarray],
    dlb: tuple[np.ndarray, np.ndarray],
    num_modes: int,
    neff_guess: float,
    direction: str,
    derivative_scale: float,
    omega: float | None = None,
    num_pml: tuple[int, int] = (0, 0),
    pml_profile: dict[str, float | int] | None = None,
    dmin_pml: tuple[bool, bool] = (True, True),
    dmin_pmc: tuple[bool, bool] = (False, False),
    krylov_dim: int = 32,
    initial_vector: np.ndarray | None = None,
) -> SparseSolveResult:
    """Solve the first-order tensorial sparse eigenproblem with SciPy/ARPACK."""

    sparse, spla, scipy_linalg = _import_scipy()

    # The tensorial formulation keeps Ex, Ey, Hx, and Hy as unknowns per cell.
    nx = len(dlf[0])
    ny = len(dlf[1])
    n = nx * ny

    # Full tensors are already flattened by the raster layer.
    if (
        eps_tensor.shape != mu_tensor.shape
        or eps_tensor.shape[:2] != (3, 3)
        or eps_tensor.shape[-1] != n
    ):
        raise ValueError(
            "eps_tensor and mu_tensor must both have shape (3, 3, len(dlf[0]) * len(dlf[1]))"
        )
    if num_modes <= 0:
        raise ValueError("num_modes must be positive")

    # Reuse the same Yee derivative matrices as the diagonal solver.
    derivatives = _create_derivative_matrices(
        sparse,
        eps_tensor=eps_tensor,
        mu_tensor=mu_tensor,
        shape=(nx, ny),
        dlf=dlf,
        dlb=dlb,
        omega=omega,
        num_pml=num_pml,
        pml_profile=pml_profile,
        dmin_pml=dmin_pml,
        dmin_pmc=dmin_pmc,
        scale=float(derivative_scale),
    )

    # Assemble the first-order operator that directly returns n_eff eigenvalues.
    operator = _assemble_tensorial_operator(sparse, eps_tensor, mu_tensor, derivatives)
    values, vectors = _selected_eigenpairs(
        operator,
        num_modes=num_modes,
        sigma=complex(neff_guess, 0.0),
        krylov_dim=krylov_dim,
        initial_vector=initial_vector,
        spla=spla,
        scipy_linalg=scipy_linalg,
    )

    # Record normalized ARPACK residuals for solver diagnostics.
    residuals = np.asarray(
        [
            np.linalg.norm(
                operator @ vectors[:, index] - values[index] * vectors[:, index]
            )
            for index in range(len(values))
        ],
        dtype=float,
    )
    residuals = residuals / np.maximum(
        np.linalg.norm(vectors, axis=0), np.finfo(float).eps
    )

    # Sort modes in descending real effective index, matching diagonal behavior.
    modes = [
        (complex(value), np.asarray(vector, dtype=np.complex128), float(residual))
        for value, vector, residual in zip(values, vectors.T, residuals, strict=True)
    ]
    modes.sort(key=lambda item: item[0].real, reverse=True)

    # z-component inverses reconstruct Ez and Hz after the transverse solve.
    inv_eps_zz = sparse.diags(1.0 / eps_tensor[2, 2, :], format="csc")
    inv_mu_zz = sparse.diags(1.0 / mu_tensor[2, 2, :], format="csc")
    dxf, dxb, dyf, dyb = derivatives
    cell_areas = np.repeat(np.asarray(dlf[0], dtype=float), ny) * np.tile(
        np.asarray(dlf[1], dtype=float), nx
    )

    n_complex: list[complex] = []
    mode_fields: list[_ModeFields] = []
    sorted_residuals: list[float] = []
    for mode_n, vector, residual in modes:
        # The first-order eigenvector stores transverse E followed by transverse H.
        ex = vector[:n].copy()
        ey = vector[n : 2 * n].copy()
        hx = vector[2 * n : 3 * n].copy()
        hy = vector[3 * n : 4 * n].copy()

        # Longitudinal fields include off-diagonal material coupling through the
        # z-row of the tensor.
        hz_source = (
            dxf @ ey - dyf @ ex - mu_tensor[2, 0, :] * hx - mu_tensor[2, 1, :] * hy
        )
        hz = np.asarray(inv_mu_zz @ hz_source, dtype=np.complex128)

        ez_source = (
            dxb @ hy - dyb @ hx - eps_tensor[2, 0, :] * ex - eps_tensor[2, 1, :] * ey
        )
        ez = np.asarray(inv_eps_zz @ ez_source, dtype=np.complex128)

        # Match the magnetic-field convention used by the diagonal solver.
        h_scale = -1j / ETA0
        hx *= h_scale
        hy *= h_scale
        hz *= h_scale

        # Backward propagation uses the same sign convention in both solvers.
        if direction == "-":
            hx *= -1.0
            hy *= -1.0
            ez *= -1.0

        # Store modes for the shared Lorentz orthogonalization pass.
        n_complex.append(mode_n)
        sorted_residuals.append(residual)
        mode_fields.append(_ModeFields(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz))

    # Normalize the full set, then return component-major field arrays.
    orthogonalization = _lorentz_orthogonalize_and_normalize(mode_fields, cell_areas)
    fields = [
        np.asarray(
            [getattr(mode, component) for mode in mode_fields], dtype=np.complex128
        )
        for component in ("ex", "ey", "ez", "hx", "hy", "hz")
    ]
    return (
        np.asarray(n_complex, dtype=np.complex128),
        fields,
        {
            "backend": "scipy_arpack_reference",
            "operator_size": int(operator.shape[0]),
            "operator_nnz": int(operator.nnz),
            "residuals": np.asarray(sorted_residuals, dtype=float),
            "power_norms": orthogonalization["power_norms"],
            "lorentz_norms": orthogonalization["lorentz_norms"],
            "lorentz_orthogonality_error": orthogonalization[
                "lorentz_orthogonality_error"
            ],
        },
    )


def _import_scipy():
    """Import SciPy modules lazily so package import stays lightweight."""
    try:
        import scipy.linalg as scipy_linalg
        import scipy.sparse as sparse
        import scipy.sparse.linalg as spla
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise ImportError(
            "the native mode solver requires BeamZ's SciPy runtime dependency"
        ) from exc
    return sparse, spla, scipy_linalg


def _create_derivative_matrices(
    sparse,
    *,
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
    shape: tuple[int, int],
    dlf: tuple[np.ndarray, np.ndarray],
    dlb: tuple[np.ndarray, np.ndarray],
    omega: float | None,
    num_pml: tuple[int, int],
    pml_profile: dict[str, float | int] | None,
    dmin_pml: tuple[bool, bool],
    dmin_pmc: tuple[bool, bool],
    scale: float,
):
    """Assemble scaled Yee derivative matrices with optional PML stretching."""
    # Build raw finite-difference stencils first; optional PML scaling is a
    # diagonal pre-multiplication applied afterward.
    matrices = (
        _make_dxf(sparse, np.asarray(dlf[0], dtype=float), shape, dmin_pmc[0]),
        _make_dxb(sparse, np.asarray(dlb[0], dtype=float), shape, dmin_pmc[0]),
        _make_dyf(sparse, np.asarray(dlf[1], dtype=float), shape, dmin_pmc[1]),
        _make_dyb(sparse, np.asarray(dlb[1], dtype=float), shape, dmin_pmc[1]),
    )
    if num_pml[0] > 0 or num_pml[1] > 0:
        # Complex stretch factors require angular frequency to scale sigma.
        if omega is None:
            raise ValueError("omega is required when num_pml is nonzero")
        pml_values = _create_s_diagonal_values(
            shape=shape,
            num_pml=num_pml,
            dlf=dlf,
            dlb=dlb,
            eps_tensor=eps_tensor,
            mu_tensor=mu_tensor,
            dmin_pml=dmin_pml,
            omega=float(omega),
            profile=pml_profile,
        )
        matrices = tuple(
            sparse.diags(values, format="csc") @ matrix
            for values, matrix in zip(pml_values, matrices, strict=True)
        )

    # derivative_scale nondimensionalizes spatial derivatives by wavelength.
    return tuple(matrix * complex(scale, 0.0) for matrix in matrices)


def _make_dxf(sparse, dls: np.ndarray, shape: tuple[int, int], pmc: bool):
    """Build the forward x derivative on the local Yee grid."""
    nx, ny = shape
    if nx == 1:
        return sparse.csc_matrix((ny, ny), dtype=np.complex128)

    # Assemble in coordinate-list form, then convert to CSC for sparse algebra.
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    for ix in range(nx):
        for iy in range(ny):
            row = ix * ny + iy
            value = 1.0 / dls[ix]

            # At the low edge, PEC-style symmetry drops the diagonal term while
            # PMC keeps the one-sided stencil.
            diagonal = 0.0 if ix == 0 and not pmc else -value
            if diagonal != 0.0:
                rows.append(row)
                cols.append(row)
                data.append(diagonal)
            if ix + 1 < nx:
                # Interior forward difference points to the next x cell.
                rows.append(row)
                cols.append((ix + 1) * ny + iy)
                data.append(value)
    return sparse.csc_matrix(
        (data, (rows, cols)), shape=(nx * ny, nx * ny), dtype=np.complex128
    )


def _make_dxb(sparse, dls: np.ndarray, shape: tuple[int, int], pmc: bool):
    """Build the backward x derivative on the local Yee grid."""
    nx, ny = shape
    if nx == 1:
        return sparse.csc_matrix((ny, ny), dtype=np.complex128)

    # Backward stencils share the same flattened row order as forward stencils.
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    for ix in range(nx):
        for iy in range(ny):
            row = ix * ny + iy
            value = 1.0 / dls[ix]

            # PMC at the low boundary mirrors the field and doubles the diagonal
            # contribution; otherwise the first row has no backward neighbor.
            diagonal = 2.0 * value if ix == 0 and pmc else (0.0 if ix == 0 else value)
            if diagonal != 0.0:
                rows.append(row)
                cols.append(row)
                data.append(diagonal)
            if ix > 0:
                # Interior backward difference points to the previous x cell.
                rows.append(row)
                cols.append((ix - 1) * ny + iy)
                data.append(-value)
    return sparse.csc_matrix(
        (data, (rows, cols)), shape=(nx * ny, nx * ny), dtype=np.complex128
    )


def _make_dyf(sparse, dls: np.ndarray, shape: tuple[int, int], pmc: bool):
    """Build the forward y derivative on the local Yee grid."""
    nx, ny = shape
    if ny == 1:
        return sparse.csc_matrix((nx, nx), dtype=np.complex128)

    # y derivatives use the same flat index but move along the inner y axis.
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    for ix in range(nx):
        for iy in range(ny):
            row = ix * ny + iy
            value = 1.0 / dls[iy]

            # Low-edge y boundary uses the same PEC/PMC convention as x.
            diagonal = 0.0 if iy == 0 and not pmc else -value
            if diagonal != 0.0:
                rows.append(row)
                cols.append(row)
                data.append(diagonal)
            if iy + 1 < ny:
                # Interior forward y difference points to the next y cell.
                rows.append(row)
                cols.append(ix * ny + iy + 1)
                data.append(value)
    return sparse.csc_matrix(
        (data, (rows, cols)), shape=(nx * ny, nx * ny), dtype=np.complex128
    )


def _make_dyb(sparse, dls: np.ndarray, shape: tuple[int, int], pmc: bool):
    """Build the backward y derivative on the local Yee grid."""
    nx, ny = shape
    if ny == 1:
        return sparse.csc_matrix((nx, nx), dtype=np.complex128)

    # Backward y differences mirror _make_dxb on the inner axis.
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    for ix in range(nx):
        for iy in range(ny):
            row = ix * ny + iy
            value = 1.0 / dls[iy]

            # A PMC low edge doubles the reflected diagonal contribution.
            diagonal = 2.0 * value if iy == 0 and pmc else (0.0 if iy == 0 else value)
            if diagonal != 0.0:
                rows.append(row)
                cols.append(row)
                data.append(diagonal)
            if iy > 0:
                # Interior backward y difference points to the previous y cell.
                rows.append(row)
                cols.append(ix * ny + iy - 1)
                data.append(-value)
    return sparse.csc_matrix(
        (data, (rows, cols)), shape=(nx * ny, nx * ny), dtype=np.complex128
    )


def _assemble_diagonal_operators(
    sparse, eps: np.ndarray, mu: np.ndarray, der_mats
) -> dict[str, object]:
    """Assemble the reduced diagonal-material eigen-operator blocks."""
    # All blocks operate on flattened cell vectors.
    n = eps.shape[-1]
    zero = sparse.csc_matrix((n, n), dtype=np.complex128)
    dxf, dxb, dyf, dyb = der_mats

    # Longitudinal component elimination requires inverse zz material terms.
    inv_eps_zz = sparse.diags(1.0 / eps[2, 2, :], format="csc")
    inv_mu_zz = sparse.diags(1.0 / mu[2, 2, :], format="csc")

    # The block names mirror docs/physics-model.md:
    # A_diag = P_mu Q + P_partial Q_epsilon, with Q = Q_epsilon + Q_partial.
    p_mu = sparse.bmat(
        [
            [zero, sparse.diags(mu[1, 1, :], format="csc")],
            [-sparse.diags(mu[0, 0, :], format="csc"), zero],
        ],
        format="csc",
    )

    # P_partial captures derivative coupling introduced by eliminating Ez.
    p_partial = sparse.bmat(
        [
            [-(dxf @ inv_eps_zz @ dyb), dxf @ inv_eps_zz @ dxb],
            [-(dyf @ inv_eps_zz @ dyb), dyf @ inv_eps_zz @ dxb],
        ],
        format="csc",
    )
    q_ep = sparse.bmat(
        [
            [zero, sparse.diags(eps[1, 1, :], format="csc")],
            [-sparse.diags(eps[0, 0, :], format="csc"), zero],
        ],
        format="csc",
    )

    # Q_partial captures derivative coupling introduced by eliminating Hz.
    q_partial = sparse.bmat(
        [
            [-(dxb @ inv_mu_zz @ dyf), dxb @ inv_mu_zz @ dxf],
            [-(dyb @ inv_mu_zz @ dyf), dyb @ inv_mu_zz @ dxf],
        ],
        format="csc",
    )
    qmat = q_ep + q_partial

    # The final reduced operator acts on [Ex, Ey].
    mat = p_mu @ qmat + p_partial @ q_ep
    return {
        "q_ep": _canonical_sparse(q_ep),
        "qmat": _canonical_sparse(qmat),
        "mat": _canonical_sparse(mat),
    }


def _assemble_tensorial_operator(sparse, eps: np.ndarray, mu: np.ndarray, der_mats):
    """Assemble the full tensorial first-order eigen-operator."""
    # The tensorial operator acts on [Ex, Ey, Hx, Hy] and leaves Ez/Hz implicit.
    dxf, dxb, dyf, dyb = der_mats
    inv_eps_22 = sparse.diags(1.0 / eps[2, 2, :], format="csc")
    inv_mu_22 = sparse.diags(1.0 / mu[2, 2, :], format="csc")

    def diag(values):
        """Return a sparse diagonal matrix for one flattened tensor component."""
        return sparse.diags(values, format="csc")

    # Precompute tensor ratios that appear repeatedly after eliminating z fields.
    eps_20_over_22 = eps[2, 0, :] / eps[2, 2, :]
    eps_21_over_22 = eps[2, 1, :] / eps[2, 2, :]
    eps_02_over_22 = eps[0, 2, :] / eps[2, 2, :]
    eps_12_over_22 = eps[1, 2, :] / eps[2, 2, :]
    mu_20_over_22 = mu[2, 0, :] / mu[2, 2, :]
    mu_21_over_22 = mu[2, 1, :] / mu[2, 2, :]
    mu_02_over_22 = mu[0, 2, :] / mu[2, 2, :]
    mu_12_over_22 = mu[1, 2, :] / mu[2, 2, :]

    # Schur-complement terms eliminate Ez and Hz before solving the first-order
    # transverse system [Ex, Ey, Hx, Hy]. Suffix "_s" means the component has
    # been corrected by its coupling through the local z component.
    mu_10_s = mu[1, 0, :] - mu[1, 2, :] * mu_20_over_22
    mu_11_s = mu[1, 1, :] - mu[1, 2, :] * mu_21_over_22
    mu_00_s = mu[0, 0, :] - mu[0, 2, :] * mu_20_over_22
    mu_01_s = mu[0, 1, :] - mu[0, 2, :] * mu_21_over_22
    eps_10_s = eps[1, 0, :] - eps[1, 2, :] * eps_20_over_22
    eps_11_s = eps[1, 1, :] - eps[1, 2, :] * eps_21_over_22
    eps_00_s = eps[0, 0, :] - eps[0, 2, :] * eps_20_over_22
    eps_01_s = eps[0, 1, :] - eps[0, 2, :] * eps_21_over_22

    # Block names encode output row and input column:
    # a = electric transverse rows, b = magnetic transverse rows, x/y = local
    # transverse components. For example, axby maps Hy into the Ex equation.
    # Electric rows include derivative terms and magnetic tensor corrections.
    axax = -(dxf @ diag(eps_20_over_22)) - diag(mu_12_over_22) @ dyf
    axay = -(dxf @ diag(eps_21_over_22)) + diag(mu_12_over_22) @ dxf
    axbx = -(dxf @ inv_eps_22 @ dyb) + diag(mu_10_s)
    axby = dxf @ inv_eps_22 @ dxb + diag(mu_11_s)

    ayax = -(dyf @ diag(eps_20_over_22)) + diag(mu_02_over_22) @ dyf
    ayay = -(dyf @ diag(eps_21_over_22)) - diag(mu_02_over_22) @ dxf
    aybx = -(dyf @ inv_eps_22 @ dyb) - diag(mu_00_s)
    ayby = dyf @ inv_eps_22 @ dxb - diag(mu_01_s)

    # Magnetic rows mirror the electric rows with eps/mu roles swapped.
    bxax = -(dxb @ inv_mu_22 @ dyf) + diag(eps_10_s)
    bxay = dxb @ inv_mu_22 @ dxf + diag(eps_11_s)
    bxbx = -(dxb @ diag(mu_20_over_22)) - diag(eps_12_over_22) @ dyb
    bxby = -(dxb @ diag(mu_21_over_22)) + diag(eps_12_over_22) @ dxb

    byax = -(dyb @ inv_mu_22 @ dyf) - diag(eps_00_s)
    byay = dyb @ inv_mu_22 @ dxf - diag(eps_01_s)
    bybx = -(dyb @ diag(mu_20_over_22)) + diag(eps_02_over_22) @ dyb
    byby = -(dyb @ diag(mu_21_over_22)) - diag(eps_02_over_22) @ dxb

    # The leading -i matches the chosen time-harmonic convention.
    return _canonical_sparse(
        -1j
        * sparse.bmat(
            [
                [axax, axay, axbx, axby],
                [ayax, ayay, aybx, ayby],
                [bxax, bxay, bxbx, bxby],
                [byax, byay, bybx, byby],
            ],
            format="csc",
        )
    )


def _canonical_sparse(matrix):
    """Return a cleaned CSC sparse matrix."""
    # CSC is the format used by ARPACK and sparse block algebra here.
    matrix = matrix.tocsc(copy=True)
    matrix.eliminate_zeros()
    return matrix


def _real_arpack_problem_if_close(
    matrix, initial_vector: np.ndarray | None, guess: complex
):
    """Use a real ARPACK problem when the operator and shift are effectively real."""
    # Empty operators cannot be inspected for imaginary scale.
    if matrix.nnz == 0:
        return matrix, initial_vector, guess
    matrix_imag = matrix.data.imag
    matrix_scale = max(float(np.max(np.abs(matrix.data))), 1.0)
    guess_is_real = abs(guess.imag) <= 1e-14 * max(abs(guess), 1.0)

    # If the complex part is numerical noise, cast to real to avoid SciPy's
    # ComplexWarning path and improve reproducibility.
    if np.max(np.abs(matrix_imag)) <= 1e-14 * matrix_scale and guess_is_real:
        real_vector = (
            None
            if initial_vector is None
            else np.asarray(np.real(initial_vector), dtype=float)
        )
        return matrix.real.astype(float), real_vector, float(guess.real)
    return matrix, initial_vector, guess


def _selected_eigenpairs(
    mat,
    *,
    num_modes: int,
    sigma: complex,
    krylov_dim: int,
    initial_vector: np.ndarray | None,
    spla,
    scipy_linalg,
) -> tuple[np.ndarray, np.ndarray]:
    """Select eigenpairs nearest the requested shift."""
    size = mat.shape[0]

    # Very small operators cannot use sparse eigs with k >= N - 1, so fall back
    # to dense eig and select the nearest modes manually.
    if num_modes >= size - 1:
        values, vectors = scipy_linalg.eig(mat.toarray())
        order = np.argsort(np.abs(values - sigma))[:num_modes]
        return values[order], vectors[:, order]

    # Keep ARPACK's Krylov subspace large enough for the requested number of
    # modes but bounded by the matrix size.
    ncv = min(size, max(int(krylov_dim), num_modes + 2))

    # Shift-invert mode returns eigenvalues closest to sigma.
    complex_warning = getattr(
        getattr(np, "exceptions", np), "ComplexWarning", RuntimeWarning
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=complex_warning, module="scipy")
        values, vectors = spla.eigs(
            mat,
            k=num_modes,
            sigma=sigma,
            which="LM",
            v0=None if initial_vector is None else np.asarray(initial_vector),
            ncv=ncv,
            tol=1e-10,
        )
    return np.asarray(values, dtype=np.complex128), np.asarray(
        vectors, dtype=np.complex128
    )


def _create_s_diagonal_values(
    *,
    shape: tuple[int, int],
    num_pml: tuple[int, int],
    dlf: tuple[np.ndarray, np.ndarray],
    dlb: tuple[np.ndarray, np.ndarray],
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
    dmin_pml: tuple[bool, bool],
    omega: float,
    profile: dict[str, float | int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build inverse PML stretch vectors for every derivative stencil."""
    nx, ny = shape

    # Estimate boundary-relative speed so PML sigma scales with local material.
    avg_speed = _average_relative_speed(shape, num_pml, eps_tensor, mu_tensor)

    # Forward and backward derivative stencils sample stretch factors at
    # different Yee locations.
    sx_f = _create_sfactor(
        "f",
        omega,
        np.asarray(dlf[0], dtype=float),
        nx,
        num_pml[0],
        dmin_pml[0],
        avg_speed[:2],
        profile,
    )
    sx_b = _create_sfactor(
        "b",
        omega,
        np.asarray(dlb[0], dtype=float),
        nx,
        num_pml[0],
        dmin_pml[0],
        avg_speed[:2],
        profile,
    )
    sy_f = _create_sfactor(
        "f",
        omega,
        np.asarray(dlf[1], dtype=float),
        ny,
        num_pml[1],
        dmin_pml[1],
        avg_speed[2:],
        profile,
    )
    sy_b = _create_sfactor(
        "b",
        omega,
        np.asarray(dlb[1], dtype=float),
        ny,
        num_pml[1],
        dmin_pml[1],
        avg_speed[2:],
        profile,
    )

    # Expand one-dimensional stretch factors into flattened cell vectors.
    sx_f_vec = np.empty(nx * ny, dtype=np.complex128)
    sx_b_vec = np.empty(nx * ny, dtype=np.complex128)
    sy_f_vec = np.empty(nx * ny, dtype=np.complex128)
    sy_b_vec = np.empty(nx * ny, dtype=np.complex128)
    for ix in range(nx):
        for iy in range(ny):
            index = ix * ny + iy
            sx_f_vec[index] = 1.0 / sx_f[ix]
            sx_b_vec[index] = 1.0 / sx_b[ix]
            sy_f_vec[index] = 1.0 / sy_f[iy]
            sy_b_vec[index] = 1.0 / sy_b[iy]
    return sx_f_vec, sx_b_vec, sy_f_vec, sy_b_vec


def _average_relative_speed(
    shape: tuple[int, int],
    num_pml: tuple[int, int],
    eps_tensor: np.ndarray,
    mu_tensor: np.ndarray,
) -> np.ndarray:
    """Estimate relative wave speed near PML boundaries."""
    # Average diagonal epsilon and mu on each side, then convert to relative
    # speed for damping-strength scaling.
    eps_avg = _pml_average_all_sides(shape, num_pml, eps_tensor)
    mu_avg = _pml_average_all_sides(shape, num_pml, mu_tensor)
    return 1.0 / np.sqrt(eps_avg * mu_avg)


def _pml_average_all_sides(
    shape: tuple[int, int], num_pml: tuple[int, int], tensor: np.ndarray
) -> np.ndarray:
    """Average diagonal material components on each boundary side."""
    nx, ny = shape

    # Region order is xmin, xmax, ymin, ymax.
    regions: list[list[complex]] = [[], [], [], []]
    for comp in range(3):
        for ix in range(nx):
            for iy in range(ny):
                value = complex(tensor[comp, comp, ix * ny + iy])

                # Collect only cells that lie inside the requested PML thickness.
                if ix < num_pml[0]:
                    regions[0].append(value)
                if ix >= max(nx - num_pml[0], 0):
                    regions[1].append(value)
                if iy < num_pml[1]:
                    regions[2].append(value)
                if iy >= max(ny - num_pml[1], 0):
                    regions[3].append(value)
    out = np.ones(4, dtype=np.complex128)

    # If a side has no PML cells, leave its average at one so it does not affect
    # unused stretch factors.
    for index, values in enumerate(regions):
        if values:
            out[index] = sum(values) / len(values)
    return out


def _create_sfactor(
    direction: str,
    omega: float,
    dls: np.ndarray,
    n: int,
    n_pml: int,
    dmin_pml: bool,
    avg_speed: np.ndarray,
    profile: dict[str, float | int] | None,
) -> np.ndarray:
    """Dispatch PML stretch-factor construction for a derivative direction."""
    # No PML means identity stretching along this axis.
    if n_pml == 0:
        return np.ones(n, dtype=np.complex128)

    # Forward and backward stencils use half-cell-shifted PML depths.
    if direction == "f":
        return _create_sfactor_f(omega, dls, n, n_pml, dmin_pml, avg_speed, profile)
    if direction == "b":
        return _create_sfactor_b(omega, dls, n, n_pml, dmin_pml, avg_speed, profile)
    raise ValueError(f"direction value {direction} not recognized")


def _create_sfactor_f(
    omega: float,
    dls: np.ndarray,
    n: int,
    n_pml: int,
    dmin_pml: bool,
    avg_speed: np.ndarray,
    profile: dict[str, float | int] | None,
) -> np.ndarray:
    """Build forward-grid PML stretch factors."""
    # Start with identity stretch and overwrite only PML cells.
    sfactor = np.ones(n, dtype=np.complex128)
    for i in range(n):
        # Low-edge PML can be disabled by symmetry walls.
        if i < n_pml and dmin_pml:
            sfactor[i] = _s_value(
                dls[0], (n_pml - i - 0.5) / n_pml, omega, avg_speed[0], profile
            )
        elif i >= n - n_pml:
            sfactor[i] = _s_value(
                dls[-1], (i - (n - n_pml) + 0.5) / n_pml, omega, avg_speed[1], profile
            )
    return sfactor


def _create_sfactor_b(
    omega: float,
    dls: np.ndarray,
    n: int,
    n_pml: int,
    dmin_pml: bool,
    avg_speed: np.ndarray,
    profile: dict[str, float | int] | None,
) -> np.ndarray:
    """Build backward-grid PML stretch factors."""
    # Backward stencils sample exactly on the low edge, not half a cell inward.
    sfactor = np.ones(n, dtype=np.complex128)
    for i in range(n):
        if i < n_pml and dmin_pml:
            sfactor[i] = _s_value(
                dls[0], (n_pml - i) / n_pml, omega, avg_speed[0], profile
            )
        elif i > n - n_pml:
            sfactor[i] = _s_value(
                dls[-1], (i - (n - n_pml)) / n_pml, omega, avg_speed[1], profile
            )
    return sfactor


def _s_value(
    dl: float,
    step: float,
    omega: float,
    avg_speed: complex,
    profile: dict[str, float | int] | None,
) -> complex:
    """Evaluate one polynomial PML stretch factor."""
    # Defaults match PmlSpec and are overridden by a validated profile dict.
    values = {
        "sigma_max": 2.0,
        "kappa_min": 1.0,
        "kappa_max": 3.0,
        "order": 3,
    }
    if profile is not None:
        values.update(profile)

    # Polynomial depth controls both attenuation sigma and coordinate kappa.
    step_power = step ** int(values["order"])
    kappa = (
        float(values["kappa_min"])
        + (float(values["kappa_max"]) - float(values["kappa_min"])) * step_power
    )
    sigma = avg_speed * (float(values["sigma_max"]) / (ETA0 * dl) * step_power)
    return complex(kappa, 0.0) + 1j * sigma / (omega * EPSILON0)


def _lorentz_orthogonalize_and_normalize(
    modes: list[_ModeFields], cell_areas: np.ndarray
) -> dict[str, object]:
    """Normalize modes and enforce Lorentz orthogonality."""
    # Begin with unit transverse power so projection coefficients are stable.
    for mode in modes:
        _normalize_to_unit_power(mode, cell_areas)

    # Modified Gram-Schmidt using the unconjugated Lorentz overlap.
    for mode_index, mode in enumerate(modes):
        for previous in modes[:mode_index]:
            denom = _lorentz_overlap(previous, previous, cell_areas)
            if abs(denom) <= np.finfo(float).eps:
                continue
            coeff = _lorentz_overlap(previous, mode, cell_areas) / denom
            mode.add_scaled(previous, -coeff)

        # Renormalize and pin phase after each projection step.
        _normalize_to_unit_power(mode, cell_areas)
        _apply_dominant_e_phase_convention(mode)

    # Diagnostics expose both power normalization and residual Lorentz coupling.
    power_norms = np.asarray(
        [abs(_transverse_power(mode, cell_areas)) for mode in modes], dtype=float
    )
    lorentz_norms = np.asarray(
        [_lorentz_overlap(mode, mode, cell_areas) for mode in modes],
        dtype=np.complex128,
    )
    error = 0.0
    for left_index, left in enumerate(modes):
        for right_index, right in enumerate(modes):
            if left_index == right_index:
                continue
            denom = float(
                np.sqrt(
                    abs(lorentz_norms[left_index]) * abs(lorentz_norms[right_index])
                )
            )
            if denom <= np.finfo(float).eps:
                continue
            # Track the largest normalized off-diagonal overlap.
            error = max(error, abs(_lorentz_overlap(left, right, cell_areas)) / denom)
    return {
        "power_norms": power_norms,
        "lorentz_norms": lorentz_norms,
        "lorentz_orthogonality_error": float(error),
    }


def _normalize_to_unit_power(mode: _ModeFields, cell_areas: np.ndarray) -> float:
    """Scale a mode to unit transverse power."""
    # Use magnitude so forward/backward sign conventions do not prevent scaling.
    norm = abs(_transverse_power(mode, cell_areas))
    if norm <= np.finfo(float).eps:
        return 0.0
    scale = 1.0 / np.sqrt(norm)

    # Apply one scalar to all six components to preserve field ratios.
    for component in mode.components():
        component *= scale
    return abs(_transverse_power(mode, cell_areas))


def _transverse_power(mode: _ModeFields, cell_areas: np.ndarray) -> complex:
    """Compute conjugated transverse power flux."""
    # For local z propagation, transverse flux is (E x H*) . z.
    return complex(
        np.sum((mode.ex * np.conj(mode.hy) - mode.ey * np.conj(mode.hx)) * cell_areas)
    )


def _lorentz_overlap(
    left: _ModeFields, right: _ModeFields, cell_areas: np.ndarray
) -> complex:
    """Compute the unconjugated reciprocal-product overlap."""
    # Symmetrize the unconjugated cross product to match reciprocal-mode theory.
    left_cross_right = np.sum((left.ex * right.hy - left.ey * right.hx) * cell_areas)
    right_cross_left = np.sum((right.ex * left.hy - right.ey * left.hx) * cell_areas)
    return complex(0.5 * (left_cross_right + right_cross_left))


def _apply_dominant_e_phase_convention(mode: _ModeFields) -> None:
    """Rotate a mode so its largest electric component is real-positive."""
    # Choose a robust phase anchor from the largest electric-field entry.
    electric = np.concatenate((mode.ex, mode.ey, mode.ez))
    if electric.size == 0:
        return
    anchor = electric[int(np.argmax(np.abs(electric) ** 2))]
    if abs(anchor) <= np.finfo(float).eps:
        return

    # Multiplying by conj(anchor)/|anchor| rotates the anchor onto the positive
    # real axis without changing norms.
    phase = np.conj(anchor) / abs(anchor)
    for component in mode.components():
        component *= phase
