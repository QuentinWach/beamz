# CPML Waveguide Benchmark Notes

The 12-cell straight-waveguide benchmark should be interpreted as a guided-mode
CPML reflection test, not as a ModeSource directionality test. The source-side
modal diagnostic separates three effects:

- `back monitor: +x wave toward source` estimates left-CPML return traveling
  back toward the source.
- `back monitor: -x wave toward left CPML` estimates source-side backward launch
  leaving the source region toward the left CPML. It is useful, but it is not a
  pure CPML-return measurement.
- `front monitor: -x wave returning from right CPML` estimates power returned
  from the right CPML after the guided mode reaches the downstream absorber.
- The target for a simple straight guide should be at least below -40 dB across
  the useful band, with center-frequency values closer to -50 dB expected from a
  well-tuned CPML.

Things to check while tuning:

- Compare against absorbing-boundary references, not against source leakage.
  Tidy3D reports near-noise-floor reflection for a comparable silicon strip
  waveguide using 12 PML layers, while Meep recommends increasing PML thickness
  until convergence and warns that waveguides and inhomogeneous media can need
  extra care.
- A 12-cell BeamZ CPML at 15 steps per wavelength in silicon is only about
  0.356 um thick at 1.55 um. That is much thinner than the simple half-vacuum-
  wavelength rule of thumb, so the profile must be discretely well matched.
- CPML-width sweeps should keep the non-PML design region fixed and add CPML
  outside it. If the total simulation box is fixed, increasing CPML cells shrinks
  the physical aperture and changes the source/monitor problem; that is not a
  clean absorber convergence test.
- Material must be extruded through the CPML along each absorber normal. If the
  waveguide, substrate, or cladding changes along the PML normal inside the
  absorber, reflections can be boundary-termination reflections rather than
  profile-parameter reflections.
- The CPML profile must be sampled on the same Yee locations used by the update.
  E and H terms have different offsets, and the interface cell should remain
  well matched instead of starting with a strong conductivity jump.
- Compact 3D Yee arrays do not store every high-side boundary sample. High-side
  CPML grading must therefore use the material-domain cell count and component
  Yee offset, not only the length of the target component array.
- Sigma, kappa, alpha, and polynomial order interact. Too little sigma leaves
  energy unabsorbed before the outer boundary; too much sigma or alpha creates a
  discrete impedance mismatch at the entrance.
- In the straight 3D silicon-waveguide benchmark, the default normalized CFS
  alpha is intentionally conservative at 0.05. A larger value improved some
  source-side numbers but raised downstream CPML return for 15-cell guided-mode
  terminations.
- Monitor planes need enough distance from the ModeSource and from CPMLs to
  avoid measuring TF/SF near fields, source residuals, or absorber near fields.
- The mode/source aperture should use the largest plane that stays outside the
  transverse CPML. A slightly clipped aperture can move the center-bin backward
  modal amplitude by a few tenths of a dB, which is enough to obscure a -40 dB
  CPML tuning check.
- Broadband DFT results depend on run length and pulse bandwidth. Center-bin
  results are the most stable first debugging signal; the full-band target
  should be checked after the center bin is fixed.

Reference points:

- Tidy3D absorbing-boundary waveguide benchmark:
  https://www.flexcompute.com/tidy3d/examples/notebooks/AbsorbingBoundaryReflection/
- Meep PML guidance:
  https://meep.readthedocs.io/en/latest/Perfectly_Matched_Layer/
- Ansys/Lumerical PML guidance:
  https://optics.ansys.com/hc/en-us/articles/360034382674-PML-boundary-conditions-in-FDTD-and-MODE
- Berenger PML survey/report:
  https://cecas.clemson.edu/cvel/pdf/TR94-6-017.pdf
- Waveguide PML termination discussion:
  https://web.stanford.edu/group/fan/publication/Mekis_IEEE_MGWL_9_502_1999.pdf
