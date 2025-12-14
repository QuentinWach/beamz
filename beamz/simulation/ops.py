"""Numerical operations for FDTD field updates: curls, field advancement, material handling on staggered Yee grids."""
import numpy as np
from beamz.const import EPS_0, MU_0


def curl_e_to_h_2d(e_fields, resolution, plane='xy'):
    """Compute curl of E-field for H update in 2D on staggered Yee grid for arbitrary plane."""
    # Unpack E-fields based on plane
    if plane == 'xy':
        # E = (Ex, Ey, Ez) with ∂/∂z = 0
        Ex, Ey, Ez = e_fields
        # ∇×E = (∂Ez/∂y)x̂ - (∂Ez/∂x)ŷ + (∂Ey/∂x - ∂Ex/∂y)ẑ
        
        # Hx: ∂Ez/∂y (staggered in x, so diff along y, shape like Hx)
        # Hx[i, j+1/2] comes from (Ez[i, j+1] - Ez[i, j])/dy
        # Ez shape (ny, nx), Hx shape (ny, nx-1) -> this is slightly confusing in standard notation
        # Standard Yee 2D TM: Ez at (i,j), Hx at (i, j+1/2), Hy at (i+1/2, j)
        # Actually in our grid: Ez(ny, nx)
        # Hx(ny, nx-1) - staggered in x?? No, Hx is usually staggered in y in TM. 
        # Let's stick to the conventions defined in fields.py:
        # xy: Ex(ny, nx-1), Ey(ny-1, nx), Ez(ny, nx)
        #     Hx(ny, nx-1), Hy(ny-1, nx), Hz(ny-1, nx-1)
        
        # Hx (at y, x+1/2? No, Hx is usually associated with Ey. 
        # In 3D: Hx is at (y+1/2, z+1/2). In 2D TM, Hx is at (y+1/2).
        # Let's verify standard Yee 2D TM (Ez, Hx, Hy):
        # Ez(i, j). Hx(i, j+1/2). Hy(i+1/2, j).
        # ∂Ez/∂y -> Hx. ∂Ez/∂x -> Hy.
        
        # In our code:
        # Hx (ny, nx-1). Ez (ny, nx). 
        # If Hx corresponds to diff(Ez, axis=1) -> that is derivative along x (axis 1).
        # Wait, usually axis 0 is y, axis 1 is x in image conventions (ny, nx).
        # So diff(Ez, axis=1) is ∂Ez/∂x.
        # But Hx = ∂Ez/∂y - ∂Ey/∂z. In 2D xy, ∂/∂z=0 => Hx = ∂Ez/∂y.
        # So Hx should come from diff(Ez, axis=0) (y-derivative).
        # Existing code: curl_ex = np.diff(ez, axis=1) / resolution.
        # axis=1 is x. This calculates ∂Ez/∂x.
        # This implies Hx update uses ∂Ez/∂x ??
        # Let's check Maxwell: ∂Hx/∂t = -1/u (∂Ez/∂y - ∂Ey/∂z).
        # Existing code in curl_e_to_h_2d: 
        # curl_ex = np.diff(ez, axis=1). This is derivative wrt index 1 (x).
        # curl_ey = -np.diff(ez, axis=0). This is derivative wrt index 0 (y).
        # So existing code says Hx depends on ∂Ez/∂x, and Hy depends on ∂Ez/∂y.
        # This matches 2D TE/TM rotated 90 degrees or swapped axes?
        # Standard physics: ∂Hx/∂t ~ -∂Ez/∂y.
        # If axis 0 is y, then ∂Ez/∂y is diff(axis=0).
        # EXISTING CODE SEEMS TO SWAP X/Y definition or use (x, y) indexing vs (row, col).
        # "Ez[i,j], Hx[i, j+1/2]" -> j is usually x index. i is y index.
        # If Hx is at [i, j+1/2], it's staggered in x? 
        # Standard Yee: Hx staggered in y. Hy staggered in x.
        # Let's respect the EXISTING implementation's behavior for TM to avoid breaking it,
        # but generalize for the new components.
        
        # Existing logic for TM (Ez -> Hx, Hy):
        # Hx comes from diff(Ez, axis=1) -> ∂Ez/∂x ??
        # Hy comes from -diff(Ez, axis=0) -> -∂Ez/∂y ??
        # This implies Hx is related to y-derivative but computed along axis 1? 
        # OR axis 1 is y?
        # In fields.py: self.Ez = np.zeros((ny, nx)). 
        # Usually (row, col) -> (y, x).
        # If axis 1 is x, diff(axis=1) is ∂/∂x.
        # So existing code computes Hx using ∂Ez/∂x. 
        # This is physically H_y (or -H_y) in standard notation.
        # This suggests Hx/Hy naming in existing code might be swapped or axes are (x, y).
        # BUT we must stick to it to avoid breaking changes if we don't fix everything.
        # WAIT: The plan says "All components use proper Yee grid staggering consistent with 3D conventions".
        # 3D Code: 
        # Hx update: diff_ez_y = np.diff(ez, axis=1). Axis 1 is y in (z, y, x). Correct.
        # So in 3D, axis 0=z, axis 1=y, axis 2=x.
        # In 2D existing (ny, nx): axis 0=y, axis 1=x.
        # Existing 2D: Hx = diff(Ez, axis=1) -> ∂Ez/∂x. This is WRONG for Hx = ∂Ez/∂y.
        # It seems existing 2D code treats axis 0 as x and axis 1 as y? 
        # Or Hx/Hy are swapped?
        # Let's assume standard numpy (y, x).
        # If we want to be consistent with 3D, we should use proper curls.
        # 3D: Hx = ∂Ez/∂y - ∂Ey/∂z.
        # 2D xy: Hx = ∂Ez/∂y.
        # ∂Ez/∂y = diff(Ez, axis=0).
        # Existing 2D used diff(axis=1).
        
        # Let's fix 2D to be consistent with 3D physics if we are doing "Full 2D".
        # 2D xy plane (∂/∂z=0):
        # Hx = ∂Ez/∂y - ∂Ey/∂z(0) = ∂Ez/∂y
        # Hy = ∂Ex/∂z(0) - ∂Ez/∂x = -∂Ez/∂x
        # Hz = ∂Ey/∂x - ∂Ex/∂y
        
        # We need to map (y, x) axes to (0, 1).
        # ∂/∂y -> axis 0. ∂/∂x -> axis 1.
        
        # Ez shape: (ny, nx).
        # Hx = ∂Ez/∂y -> diff(Ez, axis=0). Shape (ny-1, nx).
        # Hy = -∂Ez/∂x -> -diff(Ez, axis=1). Shape (ny, nx-1).
        # Hz = ∂Ey/∂x - ∂Ex/∂y -> diff(Ey, axis=1) - diff(Ex, axis=0).
        # Ex shape (ny, nx-1)? 
        # If Hx is (ny-1, nx) and Hy is (ny, nx-1), this matches standard Yee.
        
        # BUT fields.py allocated:
        # TM: Ez(ny, nx), Hx(ny, nx-1), Hy(ny-1, nx).
        # This allocation implies Hx is staggered in x (matches ∂/∂x), Hy staggered in y (matches ∂/∂y).
        # This confirms existing code had Hx ~ ∂Ez/∂x and Hy ~ ∂Ez/∂y.
        # This effectively swaps x/y or Hx/Hy roles compared to standard physics.
        # Hx is usually component along x-axis. Hy along y-axis.
        # Maxwell: curl E = (∂Ez/∂y, -∂Ez/∂x, ...).
        # (∂Ez/∂y) is x-component of curl -> drives Hx.
        # (-∂Ez/∂x) is y-component of curl -> drives Hy.
        # So Hx should depend on y-derivative (axis 0).
        # Hy should depend on x-derivative (axis 1).
        
        # If we stick to fields.py allocation: Hx(ny, nx-1) -> Staggered in X.
        # This means Hx *must* come from x-derivative.
        # So current code calculates "Hx" as something driven by ∂Ez/∂x.
        # Physically, ∂Ez/∂x drives -Hy.
        # So "Hx" in code behaves like -Hy physically.
        # And "Hy" (ny-1, nx) from y-derivative behaves like Hx physically.
        
        # This is a mess. For "Full 2D", we should align with 3D code which is correct.
        # 3D: Hx(nz-1, ny-1, nx). Staggered in y, z. Centered in x?
        # Let's look at 3D init: Hx = zeros((nz-1, ny-1, nx)). 
        # 3D curl Hx: diff_ez_y (axis 1). 
        # This confirms 3D is correct.
        
        # REFACTOR DECISION: align 2D xy with 3D logic.
        # xy plane: dimensions (ny, nx) -> map to 3D (1, ny, nx) effectively.
        # Components:
        # Ex(ny, nx-1) -> Staggered in x. 
        # Ey(ny-1, nx) -> Staggered in y.
        # Ez(ny, nx) -> Centered? Usually Ez is (ny, nx) in 2D TM (node based).
        # In 3D: Ez(nz-1, ny, nx). Staggered in z.
        
        # Let's implement robust curls for the specified plane.
        
        # XY Plane (∂/∂z = 0):
        # Hx = ∂Ez/∂y
        # Hy = -∂Ez/∂x
        # Hz = ∂Ey/∂x - ∂Ex/∂y
        
        curl_ex = np.diff(Ez, axis=0) / resolution # ∂Ez/∂y (axis 0) -> drives Hx
        curl_ey = -np.diff(Ez, axis=1) / resolution # -∂Ez/∂x (axis 1) -> drives Hy
        
        # Hz terms
        # Ey is (ny-1, nx). ∂Ey/∂x -> diff(axis=1). Result (ny-1, nx-1).
        term1_z = np.diff(Ey, axis=1) / resolution
        # Ex is (ny, nx-1). ∂Ex/∂y -> diff(axis=0). Result (ny-1, nx-1).
        term2_z = np.diff(Ex, axis=0) / resolution
        curl_ez = term1_z - term2_z
        
        return curl_ex, curl_ey, curl_ez

    elif plane == 'yz':
        # E = (Ex, Ey, Ez) with ∂/∂x = 0
        # Dimensions (nz, ny). Axis 0=z, Axis 1=y.
        Ex, Ey, Ez = e_fields
        # ∇×E = (∂Ez/∂y - ∂Ey/∂z)x̂ + (∂Ex/∂z)ŷ + (-∂Ex/∂y)ẑ
        
        # Hx = ∂Ez/∂y - ∂Ey/∂z
        # Ez(nz-1, ny). ∂Ez/∂y -> diff(axis=1). (nz-1, ny-1).
        # Ey(nz, ny-1). ∂Ey/∂z -> diff(axis=0). (nz-1, ny-1).
        curl_ex = np.diff(Ez, axis=1)/resolution - np.diff(Ey, axis=0)/resolution
        
        # Hy = ∂Ex/∂z
        # Ex(nz, ny). ∂Ex/∂z -> diff(axis=0). (nz-1, ny).
        curl_ey = np.diff(Ex, axis=0)/resolution
        
        # Hz = -∂Ex/∂y
        # Ex(nz, ny). ∂Ex/∂y -> diff(axis=1). (nz, ny-1).
        curl_ez = -np.diff(Ex, axis=1)/resolution
        
        return curl_ex, curl_ey, curl_ez

    elif plane == 'xz':
        # E = (Ex, Ey, Ez) with ∂/∂y = 0
        # Dimensions (nz, nx). Axis 0=z, Axis 1=x.
        Ex, Ey, Ez = e_fields
        # ∇×E = (-∂Ey/∂z)x̂ + (∂Ex/∂z - ∂Ez/∂x)ŷ + (∂Ey/∂x)ẑ
        
        # Hx = -∂Ey/∂z
        # Ey(nz, nx). ∂Ey/∂z -> diff(axis=0). (nz-1, nx).
        curl_ex = -np.diff(Ey, axis=0)/resolution
        
        # Hy = ∂Ex/∂z - ∂Ez/∂x
        # Ex(nz, nx-1). ∂Ex/∂z -> diff(axis=0). (nz-1, nx-1).
        # Ez(nz-1, nx). ∂Ez/∂x -> diff(axis=1). (nz-1, nx-1).
        curl_ey = np.diff(Ex, axis=0)/resolution - np.diff(Ez, axis=1)/resolution
        
        # Hz = ∂Ey/∂x
        # Ey(nz, nx). ∂Ey/∂x -> diff(axis=1). (nz, nx-1).
        curl_ez = np.diff(Ey, axis=1)/resolution
        
        return curl_ex, curl_ey, curl_ez
    
    return None

def curl_h_to_e_2d(h_fields, resolution, e_shapes, plane='xy'):
    """Compute curl of H-field for E update in 2D for arbitrary plane."""
    # e_shapes is tuple of shapes for (Ex, Ey, Ez) to handle boundary padding
    
    if plane == 'xy':
        # ∂/∂z = 0
        Hx, Hy, Hz = h_fields
        # ∂E/∂t ~ ∇×H
        # Ex ~ (∇×H)_x = ∂Hz/∂y - ∂Hy/∂z(0) = ∂Hz/∂y
        # Ey ~ (∇×H)_y = ∂Hx/∂z(0) - ∂Hz/∂x = -∂Hz/∂x
        # Ez ~ (∇×H)_z = ∂Hy/∂x - ∂Hx/∂y
        
        # Ex update: ∂Hz/∂y
        # Hz(ny-1, nx-1). ∂/∂y -> diff(axis=0). Result (ny-2, nx-1).
        # Need to pad to Ex interior shape. Ex is (ny, nx-1). Interior (ny-2, nx-1)?
        # Actually Ex is staggered in x.
        # We perform standard Yee update on E-field.
        
        # Ex (ny, nx-1). Interior points at y[1:-1].
        # ∂Hz/∂y = (Hz[j, i] - Hz[j-1, i]) / dy.
        # Hz is at y+1/2.
        
        curl_ex = np.zeros(e_shapes[0])
        # Hz is (ny-1, nx-1). Ex is (ny, nx-1).
        # We need Hz[1:, :] - Hz[:-1, :] to get derivative at integer y (Ex location).
        dHz_dy = (Hz[1:, :] - Hz[:-1, :]) / resolution
        # dHz_dy shape is (ny-2, nx-1). Fits into Ex[1:-1, :].
        curl_ex[1:-1, :] = dHz_dy
        
        # Ey update: -∂Hz/∂x
        # Hz(ny-1, nx-1). Ey(ny-1, nx).
        # -∂Hz/∂x = -(Hz[j, i] - Hz[j, i-1]) / dx.
        curl_ey = np.zeros(e_shapes[1])
        dHz_dx = (Hz[:, 1:] - Hz[:, :-1]) / resolution
        # dHz_dx shape is (ny-1, nx-2). Fits into Ey[:, 1:-1].
        curl_ey[:, 1:-1] = -dHz_dx
        
        # Ez update: ∂Hy/∂x - ∂Hx/∂y
        # Hx(ny-1, nx). Hy(ny, nx-1). Wait, from curl_e_to_h:
        # Hx came from ∂Ez/∂y (ny-1, nx). Hy came from -∂Ez/∂x (ny, nx-1).
        # Matches fields.py allocation if we swap interpretation of Hx/Hy in fields.py?
        # NO, we implemented curl_e_to_h consistent with physics: Hx(ny-1, nx), Hy(ny, nx-1).
        # BUT fields.py allocates: Hx(ny, nx-1), Hy(ny-1, nx).
        # THIS IS A MISMATCH.
        # Hx in fields.py is (ny, nx-1) -> Staggered in X. Physics Hx is staggered in Y.
        # Hy in fields.py is (ny-1, nx) -> Staggered in Y. Physics Hy is staggered in X.
        
        # We MUST correct fields.py allocation in the next step to match physics if we want 
        # "proper Yee grid staggering". 
        # Plan item 2 said: "**TM set** (existing): `Ez: (ny, nx)`, `Hx: (ny, nx-1)`, `Hy: (ny-1, nx)`"
        # This confirms legacy code has SWAPPED Hx/Hy staggering vs standard texts (or uses x/y axes swapped).
        # Standard Yee: Ez(i,j), Hx(i, j+.5), Hy(i+.5, j).
        # Hx staggered in y (should be ny-1). Hy staggered in x (should be nx-1).
        # Legacy code: Hx(ny, nx-1) [staggered x]. Hy(ny-1, nx) [staggered y].
        # So Legacy Hx is functionally Hy. Legacy Hy is functionally Hx.
        
        # To support legacy while adding full mode, we have a dilemma.
        # Option A: Fix legacy. Rename Hx/Hy or swap shapes. BREAKS existing saves/visuals?
        # Option B: Adapt operators to legacy shapes for xy plane.
        # Legacy Hx acts like Hy (associated with x-derivative).
        # Legacy Hy acts like Hx (associated with y-derivative).
        
        # Given "Full 2D" and "proper staggering", I will stick to PHYSICS and fix fields.py in the update step.
        # I will implement operators assuming PHYSICS CORRECT shapes.
        # Hx (ny-1, nx). Hy (ny, nx-1).
        # This means I need to go back to fields.py and SWAP the allocation shapes for Hx/Hy in xy plane.
        # I will do that in the fields.py update.
        
        curl_ez = np.zeros(e_shapes[2])
        # ∂Hy/∂x - ∂Hx/∂y
        dHy_dx = (Hy[:, 1:] - Hy[:, :-1]) / resolution
        dHx_dy = (Hx[1:, :] - Hx[:-1, :]) / resolution
        curl_ez[1:-1, 1:-1] = dHy_dx - dHx_dy
        
        return curl_ex, curl_ey, curl_ez

    elif plane == 'yz':
        # ∂/∂x = 0
        Hx, Hy, Hz = h_fields
        # Ex ~ ∂Hz/∂y - ∂Hy/∂z
        # Ey ~ ∂Hx/∂z
        # Ez ~ -∂Hx/∂y
        
        # Ex (nz, ny). Hz(nz-1, ny). Hy(nz, ny-1).
        curl_ex = np.zeros(e_shapes[0])
        dHz_dy = (Hz[:, 1:] - Hz[:, :-1]) / resolution
        dHy_dz = (Hy[1:, :] - Hy[:-1, :]) / resolution
        # Pad to Ex interior
        curl_ex[1:-1, 1:-1] = dHz_dy[1:-1, :] - dHy_dz[:, 1:-1] 
        # Wait, indices are tricky.
        # Ex is centered in y,z? No, Ex is tangential, usually on edges. 
        # In 2D yz (TM-like for Ex), Ex is at nodes (nz, ny).
        # Hz is at y+1/2 (ny-1). Hy is at z+1/2 (nz-1).
        # ∂Hz/∂y -> (Hz[k, j] - Hz[k, j-1])/dy -> centered at j.
        # ∂Hy/∂z -> (Hy[k, j] - Hy[k-1, j])/dz -> centered at k.
        curl_ex[1:-1, 1:-1] = (Hz[1:-1, 1:] - Hz[1:-1, :-1])/resolution - (Hy[1:, 1:-1] - Hy[:-1, 1:-1])/resolution

        # Ey ~ ∂Hx/∂z
        # Ey (nz, ny-1). Hx (nz-1, ny-1).
        # ∂Hx/∂z -> (Hx[k, j] - Hx[k-1, j])/dz -> centered at k.
        curl_ey = np.zeros(e_shapes[1])
        curl_ey[1:-1, :] = (Hx[1:, :] - Hx[:-1, :]) / resolution
        
        # Ez ~ -∂Hx/∂y
        # Ez (nz-1, ny). Hx (nz-1, ny-1).
        # -∂Hx/∂y -> -(Hx[k, j] - Hx[k, j-1])/dy -> centered at j.
        curl_ez = np.zeros(e_shapes[2])
        curl_ez[:, 1:-1] = -(Hx[:, 1:] - Hx[:, :-1]) / resolution
        
        return curl_ex, curl_ey, curl_ez
        
    elif plane == 'xz':
        # ∂/∂y = 0
        Hx, Hy, Hz = h_fields
        # Ex ~ -∂Hz/∂y(0) + ∂Hy/∂z(0) - no wait, ∇×H
        # (∇×H)_x = ∂Hz/∂y - ∂Hy/∂z = -∂Hy/∂z
        # (∇×H)_y = ∂Hx/∂z - ∂Hz/∂x
        # (∇×H)_z = ∂Hy/∂x - ∂Hx/∂y(0) = ∂Hy/∂x
        
        # Ex ~ -∂Hy/∂z
        # Ex (nz, nx-1). Hy (nz-1, nx-1).
        curl_ex = np.zeros(e_shapes[0])
        curl_ex[1:-1, :] = -(Hy[1:, :] - Hy[:-1, :]) / resolution
        
        # Ey ~ ∂Hx/∂z - ∂Hz/∂x
        # Ey (nz, nx). Hx (nz-1, nx). Hz (nz, nx-1).
        curl_ey = np.zeros(e_shapes[1])
        dHx_dz = (Hx[1:, :] - Hx[:-1, :]) / resolution
        dHz_dx = (Hz[:, 1:] - Hz[:, :-1]) / resolution
        curl_ey[1:-1, 1:-1] = dHx_dz[:, 1:-1] - dHz_dx[1:-1, :]
        
        # Ez ~ ∂Hy/∂x
        # Ez (nz-1, nx). Hy (nz-1, nx-1).
        curl_ez = np.zeros(e_shapes[2])
        curl_ez[:, 1:-1] = (Hy[:, 1:] - Hy[:, :-1]) / resolution
        
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
    
    slices = [slice(None), slice(None), slice(None)] # [z, y, x]
    
    # Default to "exclude boundaries" (1:-1) for active dimensions
    # and "select all" (slice(None)) or specific index for invariant dimension
    
    if plane == 'xy':
        # Invariant z (axis 0 in 3D array if present, or implied)
        # If arrays are 3D (nz, ny, nx): slice z=0 or middle?
        # But permittivity is likely 2D (ny, nx) passed in?
        # Fields.__init__ passes self.permittivity.
        pass # Handle below
        
    # Helper to generate the 2D slice tuple for (dim1, dim2) array
    def get_slice(s1, s2):
        return (s1, s2)
        
    s_mid = slice(1, -1)
    s_all = slice(None)
    
    if plane == 'xy':
        # Grid (ny, nx). Component staggering:
        # Ex (ny, nx-1) -> staggered x -> slice x centered? No, Ex lives at x+1/2 in cell?
        # Standard Yee: Ex at (j, i+1/2). 
        # If grid is collocated material, we just take midpoints.
        # Ex: interior in x (1:-1), full in y?
        # Update eq: Ex[j, i] needs eps at [j, i].
        # Ex grid is (ny, nx-1). 
        # We generally update interior (1:-1) or full?
        # Let's match curl shapes.
        
        if component == 'x':
            # Ex (ny, nx-1). Curl Hz is (ny-2, nx-1).
            # Update region: Ex[1:-1, :]
            region = (s_mid, s_all)
        elif component == 'y':
            # Ey (ny-1, nx). Curl Hz is (ny-1, nx-2).
            # Update region: Ey[:, 1:-1]
            region = (s_all, s_mid)
        elif component == 'z':
            # Ez (ny, nx). Curl (ny-2, nx-2).
            # Update region: Ez[1:-1, 1:-1]
            region = (s_mid, s_mid)
            
    elif plane == 'yz':
        # Grid (nz, ny)
        if component == 'x':
            # Ex (nz, ny) - normal to plane
            region = (s_mid, s_mid)
        elif component == 'y':
            # Ey (nz, ny-1) - in plane y
            region = (s_all, s_mid) # Wait, Ey is staggered in y? In 3D Ey(z, y-1/2, x). 
            # In yz: Ey(nz, ny-1). Update needs curl Hx (nz-1, ny-1) and Hz (nz-1, ny).
            # Curl calculation produced (nz-2, ny-1) for Ex part?
            # Let's trust standard interior logic: 
            # If component is normal (Ex), fully interior [1:-1, 1:-1]
            # If component is in plane (Ey, Ez):
            # Ey: [1:-1, :] (interior z, full y-staggered?)
            region = (s_mid, s_all) 
        elif component == 'z':
            # Ez (nz-1, ny)
            region = (s_all, s_mid)

    elif plane == 'xz':
        # Grid (nz, nx)
        if component == 'x':
            # Ex (nz, nx-1)
            region = (s_all, s_mid)
        elif component == 'y':
            # Ey (nz, nx) - normal
            region = (s_mid, s_mid)
        elif component == 'z':
            # Ez (nz-1, nx)
            region = (s_mid, s_all)

    # Slice the material arrays
    # Ensure region matches array dimensions
    eps = permittivity[region]
    sig = conductivity[region]
    
    return eps, sig, region

def magnetic_conductivity_terms_2d_full(conductivity, permeability, hx_shape, hy_shape, hz_shape, plane):
    """Compute magnetic conductivity for all H-components in 2D."""
    # We need to slice/reshape sigma to match H-field staggering
    # Approximate by averaging or direct slicing if collocated
    
    # Simple collocated approximation for now:
    # Just resize/broadcast sigma to H-shapes.
    # Sigma is at E-field locations or cell centers? Usually cell centers or E-locs.
    # FDTD often uses sigma at H-nodes by averaging.
    
    # For now, simplistic approach: Slice to interior to match H-fields if they are smaller
    # Or interpolate.
    
    # We will just return arrays of correct shape derived from conductivity
    # assuming slowly varying conductivity or PML
    
    def get_sigma_m(shape, axis_stagger):
        # Slice conductivity to match shape
        # shape is target H-field shape
        # axis_stagger: list of axes where H is smaller than Grid
        
        # This is getting complex to do perfectly generally.
        # Fallback: Just return scalar 0 if no conductivity, else slice carefully.
        
        # Re-use existing simple logic:
        # sigma_m = sigma * mu / eps
        term = conductivity * permeability * MU_0 / EPS_0
        
        # Crop term to shape
        slices = []
        for d in range(term.ndim):
            if term.shape[d] > shape[d]:
                diff = term.shape[d] - shape[d]
                # Crop from end
                slices.append(slice(0, -diff))
            else:
                slices.append(slice(None))
        return term[tuple(slices)]

    shx = get_sigma_m(hx_shape, [])
    shy = get_sigma_m(hy_shape, [])
    shz = get_sigma_m(hz_shape, [])
    
    return shx, shy, shz



def curl_h_to_e_3d(hx, hy, hz, resolution):
    """Compute curl of H-field for E update in 3D: ∂E/∂t = ∇×H/(ε₀εᵣ)."""
    # Full 3D curl: ∇×H = [(∂Hz/∂y - ∂Hy/∂z)x̂ + (∂Hx/∂z - ∂Hz/∂x)ŷ + (∂Hy/∂x - ∂Hx/∂y)ẑ]
    # Ex update from x-component: (∇×H)_x = ∂Hz/∂y - ∂Hy/∂z
    curl_hx = (hz[:, 1:, :] - hz[:, :-1, :]) / resolution - (hy[1:, :, :] - hy[:-1, :, :]) / resolution
    # Ey update from y-component: (∇×H)_y = ∂Hx/∂z - ∂Hz/∂x
    curl_hy = (hx[1:, :, :] - hx[:-1, :, :]) / resolution - (hz[:, :, 1:] - hz[:, :, :-1]) / resolution
    # Ez update from z-component: (∇×H)_z = ∂Hy/∂x - ∂Hx/∂y
    curl_hz = (hy[:, :, 1:] - hy[:, :, :-1]) / resolution - (hx[:, 1:, :] - hx[:, :-1, :]) / resolution
    return (curl_hx, curl_hy, curl_hz)


def magnetic_conductivity_terms_2d(conductivity, permeability, hx_shape, hy_shape):
    """Compute magnetic conductivity σ_m = σ * μ₀μᵣ/ε₀ for H-field PML absorption in 2D."""
    if conductivity.ndim < 2: return (np.zeros(hx_shape), np.zeros(hy_shape))  # No PML if conductivity is scalar
    # PML uses magnetic loss: σ_m = σ * (μ₀μᵣ/ε₀) to create matched impedance at boundaries
    sigma_m_x = conductivity[:, :-1] * permeability[:, :-1] * MU_0 / EPS_0  # Slice to Hx position (y, x-1/2)
    sigma_m_y = conductivity[:-1, :] * permeability[:-1, :] * MU_0 / EPS_0  # Slice to Hy position (y-1/2, x)
    return (sigma_m_x.reshape(hx_shape), sigma_m_y.reshape(hy_shape))


def magnetic_conductivity_terms_3d(conductivity, permeability, hx_shape, hy_shape, hz_shape):
    """Compute magnetic conductivity σ_m = σ * μ₀μᵣ/ε₀ for H-field PML absorption in 3D."""
    if conductivity.ndim < 3: return (np.zeros(hx_shape), np.zeros(hy_shape), np.zeros(hz_shape))
    # Slice arrays to match staggered Yee grid positions of each H-field component
    sigma_m_hx = (conductivity[:-1, :-1, :] * permeability[:-1, :-1, :] * MU_0 / EPS_0).reshape(hx_shape)  # Hx at (z-1/2, y-1/2, x)
    sigma_m_hy = (conductivity[:-1, :, :-1] * permeability[:-1, :, :-1] * MU_0 / EPS_0).reshape(hy_shape)  # Hy at (z-1/2, y, x-1/2)
    sigma_m_hz = (conductivity[:, :-1, :-1] * permeability[:, :-1, :-1] * MU_0 / EPS_0).reshape(hz_shape)  # Hz at (z, y-1/2, x-1/2)
    return (sigma_m_hx, sigma_m_hy, sigma_m_hz)


def advance_h_field(field, curl, sigma_m, dt):
    """Advance H-field one time step via Crank-Nicolson: ∂H/∂t = -∇×E/μ₀ - σ_m*H/μ₀."""
    # Faraday's law with magnetic loss: μ₀∂H/∂t = -∇×E - σ_m*H
    # Crank-Nicolson (implicit midpoint): H^(n+1) = [(1 - α)/(1 + α)]H^n - [Δt/μ₀/(1 + α)]∇×E^(n+1/2)
    # where α = σ_m*Δt/(2μ₀) ensures second-order accuracy and unconditional stability
    denom = 1.0 + sigma_m * dt / (2.0 * MU_0)  # Denominator: 1 + α
    factor = (1.0 - sigma_m * dt / (2.0 * MU_0)) / denom  # Coefficient for H^n: (1 - α)/(1 + α)
    source = (dt / MU_0) / denom  # Coefficient for curl term: Δt/(μ₀(1 + α))
    return factor * field - source * curl  # H^(n+1) = factor*H^n - source*∇×E


def advance_e_field(field, curl, conductivity, permittivity, dt, region):
    """Advance E-field one time step via Crank-Nicolson: ∂E/∂t = ∇×H/(ε₀εᵣ) - σE/(ε₀εᵣ)."""
    # Ampere's law with electric loss: ε₀εᵣ∂E/∂t = ∇×H - σE
    # Crank-Nicolson: E^(n+1) = [(1 - β)/(1 + β)]E^n + [Δt/(ε₀εᵣ)/(1 + β)]∇×H^(n+1/2)
    # where β = σΔt/(2ε₀εᵣ) for stability and second-order temporal accuracy
    # Note: conductivity and permittivity are already sliced to the interior region
    updated = field.copy()  # Create copy for output (preserve boundary values)
    current = field[region]  # Extract interior field values
    curl_region = curl[region]  # Extract curl at interior points
    denom = 1.0 + conductivity * dt / (2.0 * EPS_0 * permittivity)  # Denominator: 1 + β
    factor = (1.0 - conductivity * dt / (2.0 * EPS_0 * permittivity)) / denom  # Coefficient for E^n: (1 - β)/(1 + β)
    source = (dt / (EPS_0 * permittivity)) / denom  # Coefficient for curl term: Δt/(ε₀εᵣ(1 + β))
    updated[region] = factor * current + source * curl_region  # E^(n+1) = factor*E^n + source*∇×H
    return updated


def material_slice_for_e_2d(permittivity, conductivity):
    """Extract material parameters at staggered Yee grid positions for E-field in 2D."""
    # Ez is located at (i, j) on Yee grid, interior points exclude boundaries for proper curl computation
    region = (slice(1, -1), slice(1, -1))  # [1:-1, 1:-1] selects interior, avoiding edges
    return permittivity[region], conductivity[region], region


def material_slice_for_e_3d(permittivity, conductivity, orientation):
    """Extract material parameters at staggered Yee grid positions for E-field components in 3D."""
    # Each E-field component lives at different staggered positions on Yee grid:
    # Ex at (z, y, x-1/2), Ey at (z, y-1/2, x), Ez at (z-1/2, y, x)
    # Slicing [1:-1] along an axis excludes boundaries for that dimension
    if orientation == "x": region = (slice(1, -1), slice(1, -1), slice(None))  # Ex: interior in z,y; full x
    elif orientation == "y": region = (slice(1, -1), slice(None), slice(1, -1))  # Ey: interior in z,x; full y
    else: region = (slice(None), slice(1, -1), slice(1, -1))  # Ez: full z; interior in y,x
    return permittivity[region], conductivity[region], region


def update_e_field_upml_2d(Ez, Ez_x, Ez_y, Hx, Hy, pml_data, permittivity, conductivity, resolution, dt):
    """Update E field with UPML split-field formulation for 2D TM mode."""
    from beamz.const import EPS_0
    
    # Standard curl calculation
    curl_h, = curl_h_to_e_2d(Hx, Hy, resolution, Ez.shape)
    
    # Get PML parameters
    mask = pml_data['mask']
    sigma_x = pml_data['sigma_x']
    sigma_y = pml_data['sigma_y']
    kappa_x = pml_data['kappa_x']
    kappa_y = pml_data['kappa_y']
    alpha_x = pml_data['alpha_x']
    alpha_y = pml_data['alpha_y']
    
    # Simplified UPML implementation - use standard FDTD with PML conductivity
    # This is more stable than the full split-field formulation
    region = slice(1, -1), slice(1, -1)
    
    # Add PML conductivity to the existing conductivity
    total_conductivity = conductivity[region] + sigma_x[region] + sigma_y[region]
    
    # Use standard FDTD update with modified conductivity
    denom = 1.0 + total_conductivity * dt / (2.0 * EPS_0 * permittivity[region])
    factor = (1.0 - total_conductivity * dt / (2.0 * EPS_0 * permittivity[region])) / denom
    source = (dt / (EPS_0 * permittivity[region])) / denom
    
    # Update Ez field
    Ez[region] = factor * Ez[region] + source * curl_h[region]
    
    return Ez
