# Native mode solver

`beamz.devices.modes` owns BeamZ's finite-difference frequency-domain mode
solver. It is shared by mode sources, mode monitors, ports, and direct
rasterized mode analysis; it does not participate in FDTD time stepping.

The high-level BeamZ workflow uses `ModeSpec` on a `ModeSource`, `ModeMonitor`,
or `Port`. For direct solver access, provide an already-rasterized transverse
permittivity grid:

```python
import numpy as np

from beamz import LIGHT_SPEED
from beamz.devices.modes import solve_grid

eps = np.ones((40, 30))
eps[15:25, 12:18] = 3.48**2
result = solve_grid(
    eps_xx=eps,
    x_edges=np.linspace(-2.0, 2.0, eps.shape[0] + 1),
    y_edges=np.linspace(-1.5, 1.5, eps.shape[1] + 1),
    freqs=[LIGHT_SPEED / 1.55e-6],
    num_modes=2,
    target_neff=3.0,
)
```

Coordinates passed to the direct raster API are measured in micrometres and
frequencies are measured in hertz. The result contains labeled effective
indices, field components, and solver diagnostics; BeamZ analysis owns plotting
and persistence.

The discrete launch path converts solved modes onto BeamZ's component-specific
Yee supports, normalizes their signed power, and applies guarded refinement
only when field overlap, impedance, energy, power, and discrete-Maxwell
validation all accept the candidate.

See `UPSTREAM.md` for the imported MicroMode revision and source mapping.
