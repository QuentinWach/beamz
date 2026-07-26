# MicroMode provenance

The numerical mode-solver implementation in this package was imported from
[`beamzorg/micromode`](https://github.com/beamzorg/micromode) at commit
`80c57d86d02be26904bfccaae2aac900a917898e` (the `main` head on
2026-07-26).

Both projects use the Apache License 2.0. The import intentionally keeps the
upstream numerical implementation and public solver behavior together so that
BeamZ can own and validate the complete mode-source pipeline.

The source-file mapping is:

- `micromode/beamz.py` → `beamz/devices/modes/discrete.py`
- `micromode/raster.py` → `beamz/devices/modes/solver.py`
- `micromode/scipy_reference.py` → `beamz/devices/modes/_scipy.py`
- `micromode/yee.py` → `beamz/devices/modes/_yee.py`
- all other Python modules retain their upstream filename

Import-path and module-name edits plus BeamZ's automatic Ruff formatting are
the only intentional changes in the initial snapshot. Subsequent integration
changes are recorded in later commits.
