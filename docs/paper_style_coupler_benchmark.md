# Paper-Style 3D Coupler Benchmark

This benchmark recreates the performance setup from [arXiv:2412.12360](https://arxiv.org/abs/2412.12360) with:

- simulation box: `6 µm x 4 µm x 1.5 µm`
- runtime: `200 fs`
- FDTDX-style Courant safety factor: `0.99`
- resolution sweep: `25, 20, 10, 5, 2.5 nm`

The original paper references an external silicon coupler geometry that is not included in the repo, so this benchmark uses a plausible synthetic silicon slab coupler instead.

## Meep setup

Create the Meep environment once:

```bash
conda env create -f benchmarks/meep_environment.yml
```

## Run both solvers

```bash
python benchmarks/paper_style_coupler_compare.py --backend both --mode both
```

If `meep` is not importable in the current Python, the script automatically falls back to:

```bash
conda run -n beamz-meep python benchmarks/paper_style_coupler_compare.py --backend meep ...
```

## Accuracy only

Default accuracy capture runs at `20 nm` and compares normalized center-plane `|E|` slices in `xy`, `xz`, and `yz`.

```bash
python benchmarks/paper_style_coupler_compare.py --mode accuracy --accuracy-resolution-nm 20
```

## Performance only

```bash
python benchmarks/paper_style_coupler_compare.py --mode performance --resolutions-nm 25,20,10
```

## Repeated interleaved benchmarking

The performance runner supports repeated interleaved execution to reduce thermal-drift bias. With `--performance-repeats 5`, each resolution is run 5 times per backend and the schedule alternates backend and reverses order every round.

```bash
python benchmarks/paper_style_coupler_compare.py \
  --backend both \
  --mode performance \
  --resolutions-nm 25,20,10 \
  --performance-repeats 5
```

Artifacts are written to a timestamped directory under `benchmarks/results/` unless `--results-dir` is provided.

Saved files:

- `benchmark_manifest.json`
- `performance_raw_runs.csv`
- `performance_summary.csv`
- `performance_paper_table.csv`

The CSV files include:

- raw per-run `setup_s`, `compile_s`, `run_s`, `total_s`
- `gcups_run` and `gcups_total`
- `gcompups_run` and `gcompups_total`
- summary statistics: mean, std, sem, and 95% CI

## Notes

- The paper’s `0.99` Courant value is a safety factor relative to the 3D CFL limit.
- In Meep this is implemented as `Courant = 0.99 / sqrt(3) ~= 0.571577`.
- The `5 nm` and `2.5 nm` cases are very large and may require hardware similar to the paper’s setup.
