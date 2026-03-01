# Benchmark Artifact Policy

## Goal
Keep the repository lean and reviewable while preserving reproducible benchmark workflows.

## Rules
- Treat `benchmarks/results/` as generated output.
- Do not commit large binary artifacts by default (`.png`, `.npz`, `.csv`, `.mp4`, etc.).
- Commit only small, intentional baseline artifacts when they are required for regression tests or documentation.
- Prefer linking to external artifact storage (release assets, object storage, CI artifacts) for full benchmark outputs.

## Local Workflow
1. Run benchmark scripts locally and inspect outputs in `benchmarks/results/`.
2. Keep generated outputs untracked unless there is a clear reason to version them.
3. If a baseline must be tracked, document why in the PR description and keep the baseline minimal.

## CI Guidance
- Benchmark jobs should publish artifacts through CI artifact storage.
- Regressions should be detected via numeric summaries (JSON/CSV metrics), not by storing full image/video dumps in git.
