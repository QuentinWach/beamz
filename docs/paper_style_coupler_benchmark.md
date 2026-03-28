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

## Notes

- The paper’s `0.99` Courant value is a safety factor relative to the 3D CFL limit.
- In Meep this is implemented as `Courant = 0.99 / sqrt(3) ~= 0.571577`.
- The `5 nm` and `2.5 nm` cases are very large and may require hardware similar to the paper’s setup.
