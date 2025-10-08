from beamz.const import LIGHT_SPEED
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spl
import scipy.sparse as sps
import warnings

# Improved mode solver functions
def compute_derivative_matrices(N, dL, npml=0):
    """Compute finite-difference matrices for a 1D waveguide with PML boundaries.
    
    Args:
        N: Number of grid points
        dL: Grid spacing
        npml: Number of PML grid points on each side
    
    Returns:
        tuple: (Dxf, Dxb) forward and backward derivative matrices
    """
    # Basic derivative matrices
    Dxf = sps.diags([-1, 1], [0, 1], shape=(N, N), format='csr') / dL
    Dxb = sps.diags([-1, 1], [-1, 0], shape=(N, N), format='csr') / dL
    # Apply PML if requested
    if npml > 1: # Need at least 2 points for PML to have a non-zero thickness
        # Create PML scaling array
        sc_array = np.ones(N, dtype=complex)
        # Define PML parameters
        amax = 3.0  # PML strength 
        m = 3      # PML power (cubic profile often works well)
        # Calculate scaling factor based on normalized distance into PML (0=interface, 1=outer edge)
        def pml_scale(dist_norm):
            return 1.0 / (1.0 + 1j * amax * dist_norm**m)
        # Left PML: Interface at index npml-1, Outer edge at index 0
        for i in range(npml):
            # Normalized distance from interface: increases from 0 at i=npml-1 to 1 at i=0
            dist_norm = ((npml - 1) - i) / (npml - 1) 
            sc_array[i] = pml_scale(dist_norm)

        # Right PML: Interface at index N-npml, Outer edge at index N-1
        for k in range(npml):
            # Index in array is j = N - npml + k
            # Normalized distance from interface: increases from 0 at k=0 to 1 at k=npml-1
            dist_norm = k / (npml - 1)
            sc_array[N - npml + k] = pml_scale(dist_norm)

        # Create scaling matrices and apply to derivative operators
        sc_x = sps.diags(sc_array, 0)
        inv_sc_x = sps.diags(1/sc_array, 0)
        Dxf = inv_sc_x @ Dxf @ sc_x
        Dxb = inv_sc_x @ Dxb @ sc_x

    return Dxf, Dxb

def solve_modes(eps, omega, dL, npml=0, m=2):
    """Solve for the modes of a simple 1D waveguide.
    
    Args:
        eps: Permittivity profile
        omega: Angular frequency
        dL: Grid spacing
        npml: Number of PML layers
        m: Number of modes to compute
        
    Returns:
        tuple: (vals, vecs) effective indices and mode profiles
    """
    k0 = omega / LIGHT_SPEED
    N = eps.size
    # Compute derivative matrices with PML
    Dxf, Dxb = compute_derivative_matrices(N, dL, npml)
    # Define the eigenvalue problem matrix
    diag_eps = sps.diags(eps.flatten(), 0)
    # Use averaged operator for potentially better symmetry
    A = diag_eps + 0.5 * (Dxf @ Dxb + Dxb @ Dxf) * (1 / k0) ** 2
    # Solve for eigenmodes
    n_max = np.sqrt(np.max(eps))
    vals, vecs = spl.eigs(A, k=m, sigma=n_max**2, which='LM')
    # Normalize mode profiles
    vecs = vecs / np.sqrt(np.sum(np.square(np.abs(vecs)), axis=0))
    # Compute effective indices from eigenvalues
    neff = np.sqrt(vals)
    return neff, vecs
