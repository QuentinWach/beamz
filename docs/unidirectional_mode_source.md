# Unidirectional Mode Source Implementation

## Overview

The `ModeSource` class implements a unidirectional electromagnetic mode source using Huygens surface equivalent currents. This source injects fields that propagate in only one direction (±x), preventing unwanted reflections and backpropagation.

## Physics Background

### Maxwell's Equations with Sources

For 2D simulations with fields varying in the xy-plane, the relevant Maxwell equations with electric current **J** and magnetic current **M** are:

```
∂_t H_x = -(1/μ)(∂_y E_z + M_x)
∂_t H_y = (1/μ)(∂_x E_z - M_y)
∂_t E_z = (1/ε)(∂_x H_y - ∂_y H_x - J_z)
```

### Huygens Surface Equivalent Currents

To inject a mode unidirectionally, we place electric and magnetic surface currents on a virtual surface (Huygens surface) with normal vector **n̂**:

```
J_s = n̂ × H_mode
M_s = -n̂ × E_mode
```

For propagation in the +x direction (n̂ = x̂), with field components E_z and H_y:

```
J_z = H_y^mode
M_y = E_z^mode
```

These currents create a wave that propagates forward (+x) while canceling the backward wave (-x), achieving unidirectional injection.

### Sign Conventions for FDTD Integration

In the FDTD update equations:

- **J_z** is subtracted from the curl term in the E_z update:
  ```
  E_z^(n+1) = ... + (dt/ε)[curlH - J_z]
  ```
  Therefore, we pass J_z with a negative sign.

- **M_y** is added to the curl_E_y term, which is then subtracted in the H_y update:
  ```
  H_y^(n+1) = ... - (dt/μ)[curlE_y + M_y]
  ```
  Therefore, we pass M_y with a positive sign.

## Yee Grid Implementation

### Field Component Placement

In the staggered Yee grid for 2D:

- **E_z**: Cell centers at (i+1/2, j+1/2)
- **H_x**: Cell edges at (i+1/2, j)
- **H_y**: Cell edges at (i, j+1/2)

### Source Placement

For a vertical source plane at x = x_s implementing the Total-Field/Scattered-Field (TFSF) boundary:

- **J_z** is injected at E_z positions: column index `x_ez_idx` where `(x_ez_idx + 0.5) * dx ≈ x_s`
- **M_y** is injected at H_y positions with directional offset:
  - For **+x propagation**: `x_hy_idx = x_ez_idx - 1` (one column to the LEFT)
  - For **-x propagation**: `x_hy_idx = x_ez_idx + 1` (one column to the RIGHT)

This spatial offset is critical for unidirectional behavior. The staggering creates the proper TFSF boundary where:
- E_z at column `i` is at physical position x = `(i + 0.5) * dx`
- H_y at column `i-1` is at physical position x = `(i - 1) * dx`
- The offset enables the forward and backward wave components to interfere destructively in one direction while reinforcing in the desired propagation direction.

## Usage Example

```python
from beamz import *
import numpy as np

# Setup
X, Y = 20*µm, 10*µm
WL = 1.55*µm
TIME = 40*WL/LIGHT_SPEED
N_CORE, N_CLAD = 2.04, 1.444
WG_W = 0.6*µm
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2, points_per_wavelength=10)

# Create waveguide design
design = Design(width=X, height=Y, material=Material(N_CLAD**2))
design += Rectangle(position=(0, Y/2-WG_W/2), width=X, height=WG_W, material=Material(N_CORE**2))

# Create time signal
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=0.1, frequency=LIGHT_SPEED/WL, 
                       ramp_duration=WL*6/LIGHT_SPEED, t_max=TIME/2)

# Create unidirectional mode source
source = ModeSource(
    grid=design.rasterize(resolution=DX),
    center=(3*µm, Y/2),      # Source plane position
    width=2.4*µm,            # Vertical extent
    wavelength=WL,
    pol="tm",                # Polarization (te or tm)
    signal=signal,
    direction="+x"           # Propagation direction (+x or -x)
)

# Run simulation
sim = Simulation(design=design, devices=[source], 
                boundaries=[PML(edges='all', thickness=1.2*WL)], 
                time=time_steps, resolution=DX)
sim.run(animate_live="Ez")
```

## Implementation Details

### Polarization Handling

The mode solver returns fields in a coordinate system where:

- **TE** mode: E transverse (E_x, E_y), H has z-component (H_z)
- **TM** mode: H transverse (H_x, H_y), E has z-component (E_z)

For 2D FDTD with x-propagation:
- `pol="te"` → Uses E_mode[2] (E_z) and H_mode[1] (H_y)
- `pol="tm"` → Uses E_mode[1] (E_y) and H_mode[2] (H_z) mapped as equivalent fields

### Direction Handling

For **-x propagation**:
- Both J_z and M_y are negated relative to +x propagation
- This is automatically handled by checking the Poynting vector direction

### Key Methods

- `initialize(permittivity, resolution)`: Computes the mode and sets up source currents
- `get_source_terms(...)`: Returns (source_j, source_m) dictionaries with current arrays and indices
- `_phase_align(field)`: Aligns field phase to be mostly real at peak amplitude
- `_enforce_propagation_direction(E, H, axis)`: Ensures correct propagation direction via Poynting vector

## Verification

The implementation has been verified to:

1. ✓ Correctly apply J_z = H_y^mode at E_z Yee grid positions
2. ✓ Correctly apply M_y = E_z^mode at H_y Yee grid positions with proper spatial offset
3. ✓ Use proper sign conventions matching Maxwell's equations (both currents subtracted)
4. ✓ Place H_y source one column offset from E_z source for TFSF boundary
5. ✓ Handle both +x and -x propagation directions with correct spatial offsets
6. ✓ Generate non-zero field amplitudes from mode solver with correct impedance relation

## References

- Taflove & Hagness, "Computational Electrodynamics: The Finite-Difference Time-Domain Method"
- Oskooi et al., "MEEP: A flexible free-software package for electromagnetic simulations"
- Schneider, "Understanding the Finite-Difference Time-Domain Method"

## Notes

- The source automatically initializes on first call to `get_source_terms()`
- Mode effective index (neff) is printed during initialization
- Both TE and TM polarizations are supported for 2D simulations
- For proper unidirectional behavior, ensure PML boundaries are used to absorb outgoing waves

