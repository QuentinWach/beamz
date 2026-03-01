# BEAMZ Production Readiness Audit (2026-03-01)

## Snapshot
- Test suite status: `204 passed` (approx. 6m18s).
- Coverage total: ~55%.
- Lint status: red (Ruff and Flake8 both report many issues; CI currently does not enforce lint/type checks).

## Findings (Severity-Ordered)

1. High: Topology optimization uses unsafe implicit resolution fallback.
   - `TopologyManager` currently infers resolution via `design.rasterize(resolution=0.1)` when `resolution` is omitted.
   - This can silently produce unphysical/coarse grids and non-repeatable optimization behavior.
   - File: `beamz/optimization/topology.py`

2. High: Optimization helper mutates caller-owned geometry.
   - `create_optimization_mask()` can overwrite `region_structure.material` when no copy exists.
   - Side effect can corrupt upstream design state.
   - File: `beamz/optimization/topology.py`

3. High: GDS import domain construction is fragile.
   - `import_gds()` starts from default `Design()` domain and appends imported polygons directly.
   - Domain extents/background assumptions are not derived robustly from imported geometry.
   - File: `beamz/design/io.py`

4. High: Quality gate inconsistency.
   - CI runs tests but not lint/type checks.
   - Tooling mismatch: Black line length 88 vs default Flake8 79 (no harmonized config), causing permanent lint noise.
   - Files: `.github/workflows/tests.yml`, `Makefile`, `pyproject.toml`

5. Medium: Bare exception swallowing in monitor live plotting.
   - `except:` with `pass` drops stack traces and blocks diagnosis.
   - File: `beamz/devices/monitors/monitors.py`

6. Medium: Core runtime prints instead of structured logging.
   - Progress/warnings in core library paths use `print()`, which is hard to route/filter in production.
   - Files: `beamz/simulation/core.py`, `beamz/design/io.py`, others

7. Medium: Heavy top-level import surface.
   - `beamz/__init__.py` eagerly imports large subsystems; increases import cost/coupling.
   - File: `beamz/__init__.py`

8. Medium: Duplicate pytest configuration.
   - Settings live in both `pytest.ini` and `pyproject.toml`, risking drift.
   - Files: `pytest.ini`, `pyproject.toml`

9. Medium: Docs/release process drift.
   - Development docs mention options/files that do not exist anymore.
   - Files: `DEVELOPMENT.md`, `release_version.py`

10. Medium: Low coverage on visualization/runtime plumbing.
    - Very low coverage in visual stack (`runner.py`, `video.py`, `animation.py`, etc.).

11. Medium: 3D PML rasterization performance hotspot.
    - Triple nested Python loops in `_process_3d_pml`.
    - File: `beamz/design/meshing.py`

12. Low/Medium: Invalid `grid_type` can return `None` silently.
    - `Design.rasterize()` returns `None` for unsupported string values instead of failing fast.
    - File: `beamz/design/core.py`

13. Low: Placeholder/stale module content.
    - `design/library.py` is effectively empty while docs imply populated material library.
    - Files: `beamz/design/library.py`, `beamz/design/README.md`

14. Low: Constant/symbol hygiene.
    - Duplicate micro-symbol exports and coarse constants reduce polish.
    - Files: `beamz/__init__.py`, `beamz/const.py`

15. Low: Repository artifact hygiene.
    - `benchmarks/results/` currently tracks many generated files; should be policy-managed.

## Recommended Execution Order
1. Establish quality gate: unify lint stack/config and enforce in CI.
2. Fix correctness risks first: topology resolution fallback, mutable mask helper, GDS import robustness.
3. Replace swallowed exceptions and migrate prints to structured logging.
4. Improve API/import hygiene and docs/release consistency.
5. Address performance hotspots and low-coverage runtime areas.

## Current Work In This Session
- Start with:
  - Issue 5: Monitor bare exception handling.
  - Issue 1: Topology resolution default behavior.
