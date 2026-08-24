# Mirror Symmetry

A simulation whose materials, sources, and requested solution are mirror
symmetric can solve only one half of each symmetric axis. This reduces the
number of simulated cells by approximately a factor of two per axis: two active
axes reduce a 2D grid to one quarter, while three active axes reduce a 3D grid
to one eighth.

Set `Simulation.symmetry` with one entry for each physical axis `(x, y, z)`:

- `0` disables reduction on that axis.
- `1` selects even electric-field parity across the center plane.
- `-1` selects odd electric-field parity across the center plane.

For example, this solves the lower `x` and `y` quarters of a centered 2D domain:

```python
simulation = bz.Simulation(
    domain=(4 * bz.um, 4 * bz.um),
    resolution=0.2 * bz.um,
    time=time,
    sources=(centered_source,),
    monitors=(bz.FieldRecorder(("Ez", "Hx", "Hy"), name="fields"),),
    boundaries=(bz.PML(edges="all"),),
    symmetry=(1, 1, 0),
)

results = simulation.run()
full_results = results.symmetry_expanded
ez = full_results["fields"].fields["Ez"]
```

The input domain remains the full physical domain. BEAMZ validates its material
grid, retains the lower half of each selected axis, and replaces the absorbing
boundary at a symmetry cut with the appropriate PEC or PMC condition. Source
and monitor coordinates therefore stay in the same full-domain coordinate
system as an unreduced simulation.

`run()` returns the data actually computed on the reduced grid. Use
`results.symmetry_expanded` (or `results.symmetry_expanded_copy()`) to create an
immutable full-domain view of domain field recordings, retained material data,
and supported full-span 2D frequency monitors. Vector components receive the
correct reflection signs and samples on the symmetry plane are not duplicated.

## Choosing parity

Parity describes the electric vector under reflection, not whether every
component is visually even. For reflection normal to `x`:

| `symmetry[0]` | Tangential `Ey`, `Ez` | Normal `Ex` | Cut boundary |
| --- | --- | --- | --- |
| `1` | even | odd | PMC |
| `-1` | odd | even | PEC |

Magnetic-vector component signs are complementary. The same rule rotates with
the normal axis. A centered scalar `Ez` source in a 2D TM simulation is even;
an odd excitation can be made from mirrored sources with opposite signals.

## Validation and current scope

- Every selected axis must have an even number of material cells, and material
  arrays must be exactly mirror symmetric on that axis.
- The symmetry plane is the center plane of the domain. Arbitrary cut planes,
  translated planes, and rotational or periodic symmetry are not yet supported.
- The geometry, all sources, and the intended mode must share the selected
  parity. Material symmetry is checked automatically; source parity remains the
  user's responsibility.
- In 2D, the inactive physical axis must use parity `0`. Physical-axis tuples
  work consistently for `xy`, `xz`, and `yz` planes.
- Reduced-domain symmetry and standalone PMC boundaries currently execute with
  the JAX backend. `backend="auto"` selects it automatically; requesting a CUDA
  kernel explicitly raises a clear error.
- Off-diagonal anisotropic permittivity is not supported by symmetry reduction.

See the runnable
[`mirror_symmetry.py` example](https://github.com/beamzorg/beamz/blob/main/examples/2D_basics/mirror_symmetry.py)
for a full-versus-reduced comparison.
