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

### Phase 2: Single-Run Normalization Upgrade
1. Use one dedicated incident monitor on the source arm for normalization (`a_incident = a+` on source-forward monitor).
2. Measure reflection and transmission on separate output monitors in the same run.
3. Make outgoing-wave choice explicit per port (not hard-coded globally to `a-`).
4. Keep optional straight-waveguide calibration as a sanity gate only, not as a required normalization reference.

### Phase 3: Mode Extraction Stability + Convention Audit
1. Replace score-by-transmission heuristics with physically constrained mode selection:
   - Guided `n_eff` threshold.
   - Condition number threshold.
   - Continuity across frequency.
2. Prevent mode hopping with nearest-neighbor tracking in frequency.
3. Export per-port diagnostics (`a+`, `a-`, selected wave key, `n_eff`, condition number) for audit.
4. Add explicit convention checks:
   - On each output monitor, selected outgoing wave should dominate the opposite wave over most bins.
   - On source-forward monitor, selected incident wave should dominate near pulse center.

### Phase 4: Validation Against Baseline
1. Compare BeamZ results to reference crossing curves (`reference_result.png` and numeric dumps if available).
2. Track errors:
   - Through-port dB RMSE.
   - Reflection dB RMSE.
   - Max closure deviation.
3. Freeze default settings after metrics pass.

## Immediate Next Steps (Now)
1. Make `PortSpec`/S-matrix extraction support explicit `incident_wave` and `scattered_wave` selectors.
2. Switch crossing extraction to single-run normalization with dedicated source-forward monitor.
3. Add unit tests that fail when wave-selector mapping is wrong.
