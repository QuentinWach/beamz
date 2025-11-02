# Unidirectional Mode Source - Key Fixes

## Critical Issue Fixed: Phase Relationship

### The Problem
The original implementation was **phase-aligning E_z and H_y independently**:

```python
# WRONG - destroys impedance relationship
Ez_mode = self._phase_align(Ez_mode)  
Hy_mode = self._phase_align(Hy_mode)
```

This broke the fundamental electromagnetic impedance relationship between E and H fields, which is **essential** for unidirectional propagation. When E_z and H_y have an incorrect phase relationship, they don't properly cancel the backward wave.

### The Fix
Phase align them **together** to preserve their relative phase:

```python
# CORRECT - preserves impedance relationship
idx_max = np.argmax(np.abs(Ez_mode))
phase_ref = np.angle(Ez_mode[idx_max])
Ez_mode = Ez_mode * np.exp(-1j * phase_ref)
Hy_mode = Hy_mode * np.exp(-1j * phase_ref)
```

Both fields are rotated by the **same** phase angle, maintaining their mutual relationship.

## How Huygens Sources Work

For a unidirectional source propagating in +x direction:

1. **Electric current**: J_z = H_y^mode (placed at E_z column i)
2. **Magnetic current**: M_y = E_z^mode (placed at H_y column i-1, one cell to the LEFT)

The spatial offset between J_z and M_y creates a **Total-Field/Scattered-Field (TFSF) boundary** where:
- Forward waves from J_z and M_y **constructively interfere** → propagate +x
- Backward waves from J_z and M_y **destructively interfere** → canceled

This only works if E_z and H_y maintain their proper **impedance relationship**:
```
H_y = (neff/η₀) * E_z
```

where η₀ ≈ 377 Ω is the impedance of free space and neff is the mode effective index.

## Debug Information Added

The updated code now prints:
```
[DEBUG] Mode profile at peak:
  Ez: real=..., imag=...
  Hy: real=..., imag=...
  Impedance ratio Hy/Ez: ...
  Phase diff: ... degrees
```

**For a proper mode**, you should see:
- Real parts dominant (imaginary ≈ 0 after phase alignment)
- Phase difference ≈ 0 degrees between E_z and H_y
- Impedance ratio ≈ neff/377 (typically 0.002-0.006 for neff ~1-3)

## Mode Profile Visualization

The code now saves a plot to `/tmp/mode_profile.png` showing:
- Left panel: J_z = H_y current profile
- Right panel: M_y = E_z current profile

**What to look for**:
- Single symmetric peak (fundamental mode)
- Real part (solid line) should be dominant
- Imaginary part (dashed line) should be nearly zero
- Smooth Gaussian-like shape

## Sign Conventions Summary

| Current | Maxwell Equation | FDTD Implementation | Net Effect |
|---------|-----------------|---------------------|------------|
| J_z | ∂E_z/∂t = (1/ε)[curlH - J_z] | `jz = -H_y/dx`, added to curlH, then multiplied by +dt/ε | **-J_z** ✓ |
| M_y | ∂H_y/∂t = (1/μ)[∂E_z/∂x - M_y] | `my = +E_z/dx`, added to curlE, then multiplied by -dt/μ | **-M_y** ✓ |

Both currents are effectively **subtracted** in the FDTD updates, which is correct.

## Spatial Offset

For **+x propagation**:
- E_z at column i (physical x = (i+0.5)dx)
- H_y at column i-1 (physical x = (i-1)dx)
- **Offset**: -1 cell to the LEFT

For **-x propagation**:
- E_z at column i
- H_y at column i+1  
- **Offset**: +1 cell to the RIGHT

This directional offset is crucial for unidirectional behavior.

## Testing

Run the example:
```bash
python examples/1_mmi.py
```

**Expected result**:
- Wave propagates **only** in +x direction (to the right)
- **No** backward wave propagating in -x direction (to the left)
- Single mode profile visible in /tmp/mode_profile.png

## If Still Bidirectional

If you still see bidirectional propagation, check:

1. **Mode profile plot**: Are real parts dominant? Is phase diff ≈ 0°?
2. **Impedance ratio**: Is it reasonable (~neff/377)?
3. **Spatial offset**: Is it showing -1 cells for +x?
4. **Terminal output**: Check the debug info for the mode profile

The phase relationship is the most critical - if Ez and Hy are not properly phase-aligned together, the source **will** be bidirectional regardless of everything else being correct.

