# BeamZ Compact-Model S-Parameter Accuracy Plan

## Goal
Match the trusted MEEP baseline for crossing S-parameters (shape + absolute levels), with physically valid normalization and robust monitor workflow.

## Current Problems
- Non-physical S-parameter magnitudes in some runs (`|S| > 1`, poor closure).
- Overly high reflection and low transmission vs reference.
- Sensitivity to monitor placement/time windows/mode index heuristics.
- Straight-waveguide sanity is not enforced before device extraction.

## MEEP Reference Principles to Replicate
- Use a consistent incoming/outgoing convention per port.
- Normalize all outputs by source-port incoming eigenmode coefficient.
- Keep source and source monitor separated in space.
- Use robust stopping criteria (energy decay) rather than hand-tuned short windows.
- Validate with power conservation and coefficient diagnostics per frequency.

## Implementation Phases

### Phase 1: Foundation and Sanity Gates
1. Lock a single port-wave convention for BeamZ and enforce it in source/monitor decomposition.
2. Add automatic straight-waveguide calibration run before crossing extraction.
3. Add pass/fail thresholds:
   - Through near 0 dB over band.
   - Reflection well below through.
   - Closure near unity.
4. Add hard geometry checks:
   - Monitor/source non-PML overlap fraction in z.
   - Minimum source/monitor spacing in straight sections.

### Phase 2: Normalization Upgrade
1. Implement two-run normalization:
   - Reference run (calibration structure / background).
   - Device run (crossing).
2. Compute incident amplitude from reference monitors and reuse for device normalization.
3. Reflection handling:
   - Subtract source/background contribution where needed.
   - Keep reflected/transmitted branch mapping explicit and testable.

### Phase 3: Mode Extraction Stability
1. Replace score-by-transmission heuristics with physically constrained mode selection:
   - Guided `n_eff` threshold.
   - Condition number threshold.
   - Continuity across frequency.
2. Prevent mode hopping with nearest-neighbor tracking in frequency.
3. Export per-port diagnostics (`a+`, `a-`, `n_eff`, condition number) for audit.

### Phase 4: Validation Against Baseline
1. Compare BeamZ results to reference crossing curves (`reference_result.png` and numeric dumps if available).
2. Track errors:
   - Through-port dB RMSE.
   - Reflection dB RMSE.
   - Max closure deviation.
3. Freeze default settings after metrics pass.

## Immediate Next Steps (Now)
1. Implement straight-waveguide calibration gate in `beamz_crossing.py`.
2. Wire calibration metrics + fail-fast behavior.
3. Keep outputs and logs explicit so failures are diagnosable.
