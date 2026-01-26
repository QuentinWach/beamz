"""Numerical operations for FDTD field updates: curls, field advancement, material handling on staggered Yee grids."""

import jax
import jax.numpy as jnp

from beamz.const import EPS_0, MU_0


def curl_e_to_h_2d(e_fields, resolution, plane="xy"):
    """Compute curl of E-field for H update in 2D on staggered Yee grid for arbitrary plane."""
    # Unpack E-fields based on plane
    if plane == "xy":
        # E = (Ex, Ey, Ez) with ∂/∂z = 0
        Ex, Ey, Ez = e_fields
        # Field shapes: Ez(ny, nx), Ex(ny, nx-1), Ey(ny-1, nx)
        # H-field shapes: Hx(ny, nx-1), Hy(ny-1, nx), Hz(ny-1, nx-1)

        # Match existing convention for backward compatibility:
        # Hx(ny, nx-1) from diff(Ez, axis=1) -> ∂Ez/∂x gives (ny, nx-1) ✓
        # Hy(ny-1, nx) from -diff(Ez, axis=0) -> -∂Ez/∂y gives (ny-1, nx) ✓
        curl_ex = jnp.diff(Ez, axis=1) / resolution  # (ny, nx-1) for Hx
        curl_ey = -jnp.diff(Ez, axis=0) / resolution  # (ny-1, nx) for Hy

        # Hz(ny-1, nx-1) from ∂Ey/∂x - ∂Ex/∂y
        # Ey(ny-1, nx) -> diff(axis=1) gives (ny-1, nx-1)
        # Ex(ny, nx-1) -> diff(axis=0) gives (ny-1, nx-1)
        term1_z = jnp.diff(Ey, axis=1) / resolution  # (ny-1, nx-1)
        term2_z = jnp.diff(Ex, axis=0) / resolution  # (ny-1, nx-1)
        curl_ez = term1_z - term2_z

        return curl_ex, curl_ey, curl_ez

    elif plane == "yz":
        # E = (Ex, Ey, Ez) with ∂/∂x = 0
        # Dimensions (nz, ny). Axis 0=z, Axis 1=y.
        Ex, Ey, Ez = e_fields
        # ∇×E = (∂Ez/∂y - ∂Ey/∂z)x̂ + (∂Ex/∂z)ŷ + (-∂Ex/∂y)ẑ

        # Hx = ∂Ez/∂y - ∂Ey/∂z
        # Ez(nz-1, ny). ∂Ez/∂y -> diff(axis=1). (nz-1, ny-1).
        # Ey(nz, ny-1). ∂Ey/∂z -> diff(axis=0). (nz-1, ny-1).
        curl_ex = jnp.diff(Ez, axis=1) / resolution - jnp.diff(Ey, axis=0) / resolution

        # Hy = ∂Ex/∂z
        # Ex(nz, ny). ∂Ex/∂z -> diff(axis=0). (nz-1, ny).
        curl_ey = jnp.diff(Ex, axis=0) / resolution

        # Hz = -∂Ex/∂y
        # Ex(nz, ny). ∂Ex/∂y -> diff(axis=1). (nz, ny-1).
        curl_ez = -jnp.diff(Ex, axis=1) / resolution

        return curl_ex, curl_ey, curl_ez

    elif plane == "xz":
        # E = (Ex, Ey, Ez) with ∂/∂y = 0
        # Dimensions (nz, nx). Axis 0=z, Axis 1=x.
        Ex, Ey, Ez = e_fields
        # ∇×E = (-∂Ey/∂z)x̂ + (∂Ex/∂z - ∂Ez/∂x)ŷ + (∂Ey/∂x)ẑ

        # Hx = -∂Ey/∂z
        # Ey(nz, nx). ∂Ey/∂z -> diff(axis=0). (nz-1, nx).
        curl_ex = -jnp.diff(Ey, axis=0) / resolution

        # Hy = ∂Ex/∂z - ∂Ez/∂x
        # Ex(nz, nx-1). ∂Ex/∂z -> diff(axis=0). (nz-1, nx-1).
        # Ez(nz-1, nx). ∂Ez/∂x -> diff(axis=1). (nz-1, nx-1).
        curl_ey = jnp.diff(Ex, axis=0) / resolution - jnp.diff(Ez, axis=1) / resolution

        # Hz = ∂Ey/∂x
        # Ey(nz, nx). ∂Ey/∂x -> diff(axis=1). (nz, nx-1).
        curl_ez = jnp.diff(Ey, axis=1) / resolution

        return curl_ex, curl_ey, curl_ez

    return None


def curl_h_to_e_2d(h_fields, resolution, e_shapes, plane="xy"):
    """Compute curl of H-field for E update in 2D for arbitrary plane."""
    # e_shapes is tuple of shapes for (Ex, Ey, Ez) to handle boundary padding

    if plane == "xy":
        # ∂/∂z = 0
        Hx, Hy, Hz = h_fields
        # Field shapes: Hx(ny, nx-1), Hy(ny-1, nx), Hz(ny-1, nx-1)
        # E-field shapes: Ex(ny, nx-1), Ey(ny-1, nx), Ez(ny, nx)

        # Ex update: (∇×H)_x = ∂Hz/∂y - ∂Hy/∂z(0) = ∂Hz/∂y
        # Ex(ny, nx-1), Hz(ny-1, nx-1)
        # ∂Hz/∂y: Hz is (ny-1, nx-1), diff(axis=0) gives (ny-2, nx-1)
        curl_ex = jnp.zeros(e_shapes[0])
        dHz_dy = (Hz[1:, :] - Hz[:-1, :]) / resolution  # (ny-2, nx-1)
        # Use .at[].set() for functional update (JAX immutable arrays)
        curl_ex = curl_ex.at[1:-1, :].set(dHz_dy)

        # Ey update: (∇×H)_y = ∂Hx/∂z(0) - ∂Hz/∂x = -∂Hz/∂x
        # Ey(ny-1, nx), Hz(ny-1, nx-1)
        # -∂Hz/∂x: -diff(Hz, axis=1) gives (ny-1, nx-2)
        # Ey update region is [:, 1:-1] -> (ny-1, nx-2)
        curl_ey = jnp.zeros(e_shapes[1])
        dHz_dx = (Hz[:, 1:] - Hz[:, :-1]) / resolution  # (ny-1, nx-2)
        curl_ey = curl_ey.at[:, 1:-1].set(-dHz_dx)

        # Ez update: (∇×H)_z = ∂Hy/∂x - ∂Hx/∂y
        # Ez(ny, nx), Hx(ny, nx-1), Hy(ny-1, nx)
        # Match original implementation exactly for backward compatibility
        curl_ez = jnp.zeros(e_shapes[2])
        # Original code computed these derivatives at interior Ez points
        dHy_dx = (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) / resolution  # Shape: (ny-2, nx-2)
        dHx_dy = (Hx[1:-1, 1:] - Hx[1:-1, :-1]) / resolution  # Shape: (ny-2, nx-2)
        curl_ez = curl_ez.at[1:-1, 1:-1].set(dHy_dx - dHx_dy)

        return curl_ex, curl_ey, curl_ez

    elif plane == "yz":
        # ∂/∂x = 0
        Hx, Hy, Hz = h_fields
        # Ex ~ ∂Hz/∂y - ∂Hy/∂z
        # Ey ~ ∂Hx/∂z
        # Ez ~ -∂Hx/∂y

        # Ex (nz, ny). Hz(nz-1, ny). Hy(nz, ny-1).
        curl_ex = jnp.zeros(e_shapes[0])
        curl_ex = curl_ex.at[1:-1, 1:-1].set(
            (Hz[1:-1, 1:] - Hz[1:-1, :-1]) / resolution
            - (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) / resolution
        )

        # Ey ~ ∂Hx/∂z
        # Ey (nz, ny-1). Hx (nz-1, ny-1).
        curl_ey = jnp.zeros(e_shapes[1])
        curl_ey = curl_ey.at[1:-1, :].set((Hx[1:, :] - Hx[:-1, :]) / resolution)

        # Ez ~ -∂Hx/∂y
        # Ez (nz-1, ny). Hx (nz-1, ny-1).
        curl_ez = jnp.zeros(e_shapes[2])
        curl_ez = curl_ez.at[:, 1:-1].set(-(Hx[:, 1:] - Hx[:, :-1]) / resolution)

        return curl_ex, curl_ey, curl_ez

    elif plane == "xz":
        # ∂/∂y = 0
        Hx, Hy, Hz = h_fields
        # (∇×H)_x = ∂Hz/∂y - ∂Hy/∂z = -∂Hy/∂z
        # (∇×H)_y = ∂Hx/∂z - ∂Hz/∂x
        # (∇×H)_z = ∂Hy/∂x - ∂Hx/∂y(0) = ∂Hy/∂x

        # Ex ~ -∂Hy/∂z
        # Ex (nz, nx-1). Hy (nz-1, nx-1).
        curl_ex = jnp.zeros(e_shapes[0])
        curl_ex = curl_ex.at[1:-1, :].set(-(Hy[1:, :] - Hy[:-1, :]) / resolution)

        # Ey ~ ∂Hx/∂z - ∂Hz/∂x
        # Ey (nz, nx). Hx (nz-1, nx). Hz (nz, nx-1).
        curl_ey = jnp.zeros(e_shapes[1])
        dHx_dz = (Hx[1:, :] - Hx[:-1, :]) / resolution
        dHz_dx = (Hz[:, 1:] - Hz[:, :-1]) / resolution
        curl_ey = curl_ey.at[1:-1, 1:-1].set(dHx_dz[:, 1:-1] - dHz_dx[1:-1, :])

        # Ez ~ ∂Hy/∂x
        # Ez (nz-1, nx). Hy (nz-1, nx-1).
        curl_ez = jnp.zeros(e_shapes[2])
        curl_ez = curl_ez.at[:, 1:-1].set((Hy[:, 1:] - Hy[:, :-1]) / resolution)

        return curl_ex, curl_ey, curl_ez

    return None


def material_slice_for_e_2d_component(permittivity, conductivity, component, plane):
    """Extract material parameters for a specific E-component in 2D plane."""
    # component: 'x', 'y', or 'z'
    # plane: 'xy', 'yz', 'xz'

    # We need to slice based on where the component is defined in the grid
    # To simplify, we'll take the "interior" valid region for update

    # 3D Shapes: Ex(z, y, x-1/2), Ey(z, y-1/2, x), Ez(z-1/2, y, x)
    # 2D Slices must respect this relative staggering

    slices = [slice(None), slice(None), slice(None)]  # [z, y, x]

    # Default to "exclude boundaries" (1:-1) for active dimensions
    # and "select all" (slice(None)) or specific index for invariant dimension

    if plane == "xy":
        # Invariant z (axis 0 in 3D array if present, or implied)
        # If arrays are 3D (nz, ny, nx): slice z=0 or middle?
        # But permittivity is likely 2D (ny, nx) passed in?
        # Fields.__init__ passes self.permittivity.
        pass  # Handle below

    # Helper to generate the 2D slice tuple for (dim1, dim2) array
    def get_slice(s1, s2):
        return (s1, s2)

    s_mid = slice(1, -1)
    s_all = slice(None)

    if plane == "xy":
        # Grid (ny, nx). Component staggering:
        # Ex (ny, nx-1) - staggered in x
        # Ey (ny-1, nx) - staggered in y
        # Ez (ny, nx) - centered

        if component == "x":
            # Ex (ny, nx-1). Update region: Ex[1:-1, :] -> (ny-2, nx-1)
            # Material at Ex[i, j] positions: use material[i, j] from permittivity[i, j]
            # So material[1:-1, :-1] gives (ny-2, nx-1) from permittivity(ny, nx)
            # Region [1:-1, :] is used for field/curl, material is pre-sliced to match
            region = (s_mid, s_all)  # [1:-1, :] for field update
        elif component == "y":
            # Ey (ny-1, nx). Update region: Ey[:, 1:-1] -> (ny-1, nx-2)
            # Material at Ey[i, j] positions: use material[i, j] from permittivity[i, j]
            # So material[:-1, 1:-1] gives (ny-1, nx-2) from permittivity(ny, nx)
            # Region [:, 1:-1] is used for field/curl, material is pre-sliced to match
            region = (s_all, s_mid)  # [:, 1:-1] for field update
        elif component == "z":
            # Ez (ny, nx). Update region: Ez[1:-1, 1:-1] -> (ny-2, nx-2)
            # Material needs: permittivity[1:-1, 1:-1] -> (ny-2, nx-2)
            region = (s_mid, s_mid)

    elif plane == "yz":
        # Grid (nz, ny)
        if component == "x":
            # Ex (nz, ny) - normal to plane
            region = (s_mid, s_mid)
        elif component == "y":
            # Ey (nz, ny-1) - in plane y
            region = (s_mid, s_all)
        elif component == "z":
            # Ez (nz-1, ny)
            region = (s_all, s_mid)

    elif plane == "xz":
        # Grid (nz, nx)
        if component == "x":
            # Ex (nz, nx-1)
            region = (s_all, s_mid)
        elif component == "y":
            # Ey (nz, nx) - normal
            region = (s_mid, s_mid)
        elif component == "z":
            # Ez (nz-1, nx)
            region = (s_mid, s_all)

    # Slice the material arrays to match the field component shape
    # advance_e_field uses material directly (not indexed by region), so material must match field[region] shape
    if plane == "xy":
        if component == "x":
            # Ex is (ny, nx-1), update region [1:-1, :] gives (ny-2, nx-1)
            # Material at Ex positions: material[1:-1, :-1] -> (ny-2, nx-1)
            eps = permittivity[1:-1, :-1]
            sig = conductivity[1:-1, :-1]
        elif component == "y":
            # Ey is (ny-1, nx), update region [:, 1:-1] gives (ny-1, nx-2)
            # Material at Ey positions: material[:-1, 1:-1] -> (ny-1, nx-2)
            eps = permittivity[:-1, 1:-1]
            sig = conductivity[:-1, 1:-1]
        else:  # component == 'z'
            # Ez is (ny, nx), update region [1:-1, 1:-1] gives (ny-2, nx-2)
            # Material: material[1:-1, 1:-1] -> (ny-2, nx-2)
            eps = permittivity[region]
            sig = conductivity[region]
    else:
        # For yz and xz planes, use standard slicing for now
        eps = permittivity[region]
        sig = conductivity[region]

    return eps, sig, region


def magnetic_conductivity_terms_2d_full(
    conductivity, permeability, hx_shape, hy_shape, hz_shape, plane
):
    """Compute magnetic conductivity for all H-components in 2D."""
    # sigma_m = sigma * mu * MU_0 / EPS_0
    base_term = conductivity * permeability * MU_0 / EPS_0

    if plane == "xy":
        # Hx(ny, nx-1), Hy(ny-1, nx), Hz(ny-1, nx-1)
        # conductivity is (ny, nx) - same as permittivity grid
        # Hx is staggered in x -> slice x: conductivity[:, :-1]
        sigma_m_hx = base_term[:, :-1]
        # Hy is staggered in y -> slice y: conductivity[:-1, :]
        sigma_m_hy = base_term[:-1, :]
        # Hz is staggered in both -> slice both: conductivity[:-1, :-1]
        sigma_m_hz = base_term[:-1, :-1]

    elif plane == "yz":
        # Hx(nz-1, ny-1), Hy(nz, ny-1), Hz(nz-1, ny)
        # conductivity is (nz, ny)
        sigma_m_hx = base_term[:-1, :-1]  # Staggered in both z and y
        sigma_m_hy = base_term[:, :-1]  # Staggered in y
        sigma_m_hz = base_term[:-1, :]  # Staggered in z

    elif plane == "xz":
        # Hx(nz, nx-1), Hy(nz-1, nx-1), Hz(nz-1, nx)
        # conductivity is (nz, nx)
        sigma_m_hx = base_term[:, :-1]  # Staggered in x
        sigma_m_hy = base_term[:-1, :-1]  # Staggered in both z and x
        sigma_m_hz = base_term[:-1, :]  # Staggered in z

    # Ensure shapes match exactly
    sigma_m_hx = (
        jnp.reshape(sigma_m_hx, hx_shape)
        if sigma_m_hx.shape != hx_shape
        else sigma_m_hx
    )
    sigma_m_hy = (
        jnp.reshape(sigma_m_hy, hy_shape)
        if sigma_m_hy.shape != hy_shape
        else sigma_m_hy
    )
    sigma_m_hz = (
        jnp.reshape(sigma_m_hz, hz_shape)
        if sigma_m_hz.shape != hz_shape
        else sigma_m_hz
    )

    return sigma_m_hx, sigma_m_hy, sigma_m_hz


def curl_e_to_h_3d(ex, ey, ez, resolution):
    """Compute curl of E-field for H update in 3D: ∂H/∂t = -∇×E/μ₀."""
    # Full 3D curl: ∇×E = [(∂Ez/∂y - ∂Ey/∂z)x̂ + (∂Ex/∂z - ∂Ez/∂x)ŷ + (∂Ey/∂x - ∂Ex/∂y)ẑ]
    # Field shapes: Ex(nz, ny, nx-1), Ey(nz, ny-1, nx), Ez(nz-1, ny, nx)
    # H-field shapes: Hx(nz-1, ny-1, nx), Hy(nz-1, ny, nx-1), Hz(nz, ny-1, nx-1)

    # Hx update from x-component: (∇×E)_x = ∂Ez/∂y - ∂Ey/∂z
    # Hx is at (z-1/2, y-1/2, x), need curl at that position
    # Ez is at (z-1/2, y, x), ∂Ez/∂y -> diff along y axis: (nz-1, ny-1, nx)
    # Ey is at (z, y-1/2, x), ∂Ey/∂z -> diff along z axis: (nz-1, ny-1, nx)
    dEz_dy = (ez[:, 1:, :] - ez[:, :-1, :]) / resolution  # (nz-1, ny-1, nx)
    dEy_dz = (ey[1:, :, :] - ey[:-1, :, :]) / resolution  # (nz-1, ny-1, nx)
    curl_ex = dEz_dy - dEy_dz  # (nz-1, ny-1, nx) matches Hx shape

    # Hy update from y-component: (∇×E)_y = ∂Ex/∂z - ∂Ez/∂x
    # Hy is at (z-1/2, y, x-1/2), need curl at that position
    # Ex is at (z, y, x-1/2), ∂Ex/∂z -> diff along z axis: (nz-1, ny, nx-1)
    # Ez is at (z-1/2, y, x), ∂Ez/∂x -> diff along x axis: (nz-1, ny, nx-1)
    dEx_dz = (ex[1:, :, :] - ex[:-1, :, :]) / resolution  # (nz-1, ny, nx-1)
    dEz_dx = (ez[:, :, 1:] - ez[:, :, :-1]) / resolution  # (nz-1, ny, nx-1)
    curl_ey = dEx_dz - dEz_dx  # (nz-1, ny, nx-1) matches Hy shape

    # Hz update from z-component: (∇×E)_z = ∂Ey/∂x - ∂Ex/∂y
    # Hz is at (z, y-1/2, x-1/2), need curl at that position
    # Ey is at (z, y-1/2, x), ∂Ey/∂x -> diff along x axis: (nz, ny-1, nx-1)
    # Ex is at (z, y, x-1/2), ∂Ex/∂y -> diff along y axis: (nz, ny-1, nx-1)
    dEy_dx = (ey[:, :, 1:] - ey[:, :, :-1]) / resolution  # (nz, ny-1, nx-1)
    dEx_dy = (ex[:, 1:, :] - ex[:, :-1, :]) / resolution  # (nz, ny-1, nx-1)
    curl_ez = dEy_dx - dEx_dy  # (nz, ny-1, nx-1) matches Hz shape

    return (curl_ex, curl_ey, curl_ez)


def curl_h_to_e_3d(hx, hy, hz, resolution, ex_shape=None, ey_shape=None, ez_shape=None):
    """Compute curl of H-field for E update in 3D: ∂E/∂t = ∇×H/(ε₀εᵣ)."""
    # Full 3D curl: ∇×H = [(∂Hz/∂y - ∂Hy/∂z)x̂ + (∂Hx/∂z - ∂Hz/∂x)ŷ + (∂Hy/∂x - ∂Hx/∂y)ẑ]
    # Field shapes: Hx(nz-1, ny-1, nx), Hy(nz-1, ny, nx-1), Hz(nz, ny-1, nx-1)
    # E-field shapes: Ex(nz, ny, nx-1), Ey(nz, ny-1, nx), Ez(nz-1, ny, nx)

    # Determine target shapes from E-field shapes if provided
    if ex_shape is None:
        ex_shape = (hz.shape[0], hz.shape[1] + 1, hz.shape[2])
    if ey_shape is None:
        ey_shape = (hx.shape[0] + 1, hx.shape[1], hx.shape[2])
    if ez_shape is None:
        ez_shape = (hy.shape[0], hy.shape[1], hy.shape[2] + 1)

    # Ex update: (∇×H)_x = ∂Hz/∂y - ∂Hy/∂z
    curl_hx = jnp.zeros(ex_shape)
    curl_hx = curl_hx.at[:, 1:-1, :].set((hz[:, 1:, :] - hz[:, :-1, :]) / resolution)
    curl_hx = curl_hx.at[1:-1, :, :].add(-(hy[1:, :, :] - hy[:-1, :, :]) / resolution)

    # Ey update: (∇×H)_y = ∂Hx/∂z - ∂Hz/∂x
    curl_hy = jnp.zeros(ey_shape)
    curl_hy = curl_hy.at[1:-1, :, :].set((hx[1:, :, :] - hx[:-1, :, :]) / resolution)
    curl_hy = curl_hy.at[:, :, 1:-1].add(-(hz[:, :, 1:] - hz[:, :, :-1]) / resolution)

    # Ez update: (∇×H)_z = ∂Hy/∂x - ∂Hx/∂y
    curl_hz = jnp.zeros(ez_shape)
    curl_hz = curl_hz.at[:, :, 1:-1].set((hy[:, :, 1:] - hy[:, :, :-1]) / resolution)
    curl_hz = curl_hz.at[:, 1:-1, :].add(-(hx[:, 1:, :] - hx[:, :-1, :]) / resolution)

    return (curl_hx, curl_hy, curl_hz)


def magnetic_conductivity_terms_2d(conductivity, permeability, hx_shape, hy_shape):
    """Compute magnetic conductivity σ_m = σ * μ₀μᵣ/ε₀ for H-field PML absorption in 2D."""
    if conductivity.ndim < 2:
        return (
            jnp.zeros(hx_shape),
            jnp.zeros(hy_shape),
        )  # No PML if conductivity is scalar
    # PML uses magnetic loss: σ_m = σ * (μ₀μᵣ/ε₀) to create matched impedance at boundaries
    sigma_m_x = (
        conductivity[:, :-1] * permeability[:, :-1] * MU_0 / EPS_0
    )  # Slice to Hx position (y, x-1/2)
    sigma_m_y = (
        conductivity[:-1, :] * permeability[:-1, :] * MU_0 / EPS_0
    )  # Slice to Hy position (y-1/2, x)
    return (jnp.reshape(sigma_m_x, hx_shape), jnp.reshape(sigma_m_y, hy_shape))


def magnetic_conductivity_terms_3d(
    conductivity, permeability, hx_shape, hy_shape, hz_shape
):
    """Compute magnetic conductivity σ_m = σ * μ₀μᵣ/ε₀ for H-field PML absorption in 3D."""
    if conductivity.ndim < 3:
        return (jnp.zeros(hx_shape), jnp.zeros(hy_shape), jnp.zeros(hz_shape))
    # Slice arrays to match staggered Yee grid positions of each H-field component
    sigma_m_hx = jnp.reshape(
        conductivity[:-1, :-1, :] * permeability[:-1, :-1, :] * MU_0 / EPS_0, hx_shape
    )  # Hx at (z-1/2, y-1/2, x)
    sigma_m_hy = jnp.reshape(
        conductivity[:-1, :, :-1] * permeability[:-1, :, :-1] * MU_0 / EPS_0, hy_shape
    )  # Hy at (z-1/2, y, x-1/2)
    sigma_m_hz = jnp.reshape(
        conductivity[:, :-1, :-1] * permeability[:, :-1, :-1] * MU_0 / EPS_0, hz_shape
    )  # Hz at (z, y-1/2, x-1/2)
    return (sigma_m_hx, sigma_m_hy, sigma_m_hz)


def advance_h_field(field, curl, sigma_m, dt):
    """Advance H-field one time step via Crank-Nicolson: ∂H/∂t = -∇×E/μ₀ - σ_m*H/μ₀.

    FUNCTIONAL version - returns NEW array instead of mutating input.
    """
    # Faraday's law with magnetic loss: μ₀∂H/∂t = -∇×E - σ_m*H
    # Crank-Nicolson (implicit midpoint): H^(n+1) = [(1 - α)/(1 + α)]H^n - [Δt/μ₀/(1 + α)]∇×E^(n+1/2)
    # where α = σ_m*Δt/(2μ₀) ensures second-order accuracy and unconditional stability
    denom = 1.0 + sigma_m * (dt / (2.0 * MU_0))  # Denominator: 1 + α
    factor = (1.0 - sigma_m * (dt / (2.0 * MU_0))) / denom
    source_coeff = (dt / MU_0) / denom
    # Return NEW array (functional style for JAX)
    return field * factor - source_coeff * curl


def advance_e_field(field, curl, conductivity, permittivity, dt, region):
    """Advance E-field one time step via Crank-Nicolson: ∂E/∂t = ∇×H/(ε₀εᵣ) - σE/(ε₀εᵣ).

    FUNCTIONAL version - returns NEW array using .at[].set() for indexed updates.
    """
    # Ampere's law with electric loss: ε₀εᵣ∂E/∂t = ∇×H - σE
    # Crank-Nicolson: E^(n+1) = [(1 - β)/(1 + β)]E^n + [Δt/(ε₀εᵣ)/(1 + β)]∇×H^(n+1/2)
    # where β = σΔt/(2ε₀εᵣ) for stability and second-order temporal accuracy
    # Note: conductivity and permittivity are already sliced to the interior region
    denom = 1.0 + conductivity * (
        dt / (2.0 * EPS_0 * permittivity)
    )  # Denominator: 1 + β
    factor = (1.0 - conductivity * (dt / (2.0 * EPS_0 * permittivity))) / denom
    source = (dt / (EPS_0 * permittivity)) / denom

    # Compute new values for the interior region
    new_values = field[region] * factor + source * curl[region]

    # Use .at[].set() for functional update (JAX immutable arrays)
    return field.at[region].set(new_values)


def material_slice_for_e_2d(permittivity, conductivity):
    """Extract material parameters at staggered Yee grid positions for E-field in 2D."""
    # Ez is located at (i, j) on Yee grid, interior points exclude boundaries for proper curl computation
    region = (
        slice(1, -1),
        slice(1, -1),
    )  # [1:-1, 1:-1] selects interior, avoiding edges
    return permittivity[region], conductivity[region], region


def material_slice_for_e_3d(permittivity, conductivity, orientation):
    """Extract material parameters at staggered Yee grid positions for E-field components in 3D."""
    # Each E-field component lives at different staggered positions on Yee grid:
    # Ex at (z, y, x-1/2), Ey at (z, y-1/2, x), Ez at (z-1/2, y, x)
    # Slicing [1:-1] along an axis excludes boundaries for that dimension.
    # We must slice the material grid to match the field component's staggered dimension.
    if orientation == "x":
        m_region = (slice(1, -1), slice(1, -1), slice(None, -1))  # Match Ex staggered x
        f_region = (slice(1, -1), slice(1, -1), slice(None))
    elif orientation == "y":
        m_region = (slice(1, -1), slice(None, -1), slice(1, -1))  # Match Ey staggered y
        f_region = (slice(1, -1), slice(None), slice(1, -1))
    else:  # orientation == "z"
        m_region = (slice(None, -1), slice(1, -1), slice(1, -1))  # Match Ez staggered z
        f_region = (slice(None), slice(1, -1), slice(1, -1))

    return permittivity[m_region], conductivity[m_region], f_region


def update_e_field_upml_2d(
    Ez, Ez_x, Ez_y, Hx, Hy, pml_data, permittivity, conductivity, resolution, dt
):
    """Update E field with UPML split-field formulation for 2D TM mode.

    FUNCTIONAL version - returns NEW array.
    """
    # Standard curl calculation
    (curl_h,) = curl_h_to_e_2d(Hx, Hy, resolution, Ez.shape)

    # Get PML parameters
    mask = pml_data["mask"]
    sigma_x = pml_data["sigma_x"]
    sigma_y = pml_data["sigma_y"]
    kappa_x = pml_data["kappa_x"]
    kappa_y = pml_data["kappa_y"]
    alpha_x = pml_data["alpha_x"]
    alpha_y = pml_data["alpha_y"]

    # Simplified UPML implementation - use standard FDTD with PML conductivity
    # This is more stable than the full split-field formulation
    region = (slice(1, -1), slice(1, -1))

    # Add PML conductivity to the existing conductivity
    total_conductivity = conductivity[region] + sigma_x[region] + sigma_y[region]

    # Use standard FDTD update with modified conductivity
    denom = 1.0 + total_conductivity * dt / (2.0 * EPS_0 * permittivity[region])
    factor = (
        1.0 - total_conductivity * dt / (2.0 * EPS_0 * permittivity[region])
    ) / denom
    source = (dt / (EPS_0 * permittivity[region])) / denom

    # Compute new values and use .at[].set() for functional update
    new_values = factor * Ez[region] + source * curl_h[region]
    return Ez.at[region].set(new_values)
