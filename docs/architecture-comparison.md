# Architecture and comparison

GPU acceleration is not, by itself, BeamZ's architectural differentiator. Other
solvers offer GPU execution, automatic differentiation, or a high-level Python API.
BeamZ combines those capabilities with an intentionally small, inspectable local
solver whose configuration, execution, and analysis layers have explicit contracts:

- **Immutable specification, explicit state.** A `Simulation` describes a run but
  never owns its current fields, clock, monitor buffers, or executable cache. It is
  lowered through a data-only request into an immutable compiled plan;
  `SimulationState` contains everything that evolves, `SimulationResults` is a
  detached read-only value, and `SimulationRun` keeps the two lifecycles separate.
  This makes continuation, caching, parameter sweeps, and reproducibility easier
  to reason about.
- **Compilation is a boundary, not a side effect.** Rasterization, device lowering,
  JAX planning, execution, and derived analysis are separate stages. Numerical
  kernels do not depend on plotting, S-parameter conventions, or notebook state,
  and bounded caches live outside the immutable model.
- **The accelerated solver remains inspectable.** The FDTD kernels, CPML lowering,
  monitor accumulation, and multi-device sharding are available in the Python
  source under Apache-2.0. CPU, one-GPU, and multi-GPU execution use the same public
  simulation and state model.
- **Photonic-chip workflow without a service boundary.** GDSII geometry,
  rasterization, mode sources, DFT monitors, modal projection, S-parameters, and
  inverse-design utilities can all run in a local Python/JAX workflow. Users can
  inspect, test, or replace the numerical implementation rather than only the
  client API.
- **A deliberately narrow public solver API.** Grid-aware plans, Yee helpers,
  update kernels, and source/monitor lowering stay private. There is one supported
  route from a simulation description to results, which reduces the number of
  partially compatible abstractions users and contributors must understand.

The complete internal data flow and ownership rules are documented in the
[simulation architecture](https://github.com/beamzorg/beamz/blob/main/beamz/simulation/README.md).

## Comparison with other FDTD packages

This is an architectural comparison, reviewed against the projects' official
documentation in July 2026. It is not a claim that matching feature names imply
matching numerical accuracy or performance.

| Package | Execution and openness | Architectural emphasis | Where BeamZ is stronger | Where the other package is stronger |
| --- | --- | --- | --- | --- |
| **BeamZ** | Local JAX on CPU, GPU, or multiple devices; complete FDTD solver under Apache-2.0 | Immutable `Simulation -> request -> compiled plan`, continuation state separated from detached results, and analysis outside the solver core | Baseline: permissive and inspectable end-to-end local workflow, explicit ownership, 2D/3D, and a small Python-native codebase | Alpha-stage project; it does not yet claim the breadth, validation history, support organization, or published head-to-head results of the established packages |
| [**Meep**](https://meep.readthedocs.io/en/latest/) | Local C++ core with Python/Scheme/C++ APIs; GPL; official parallel path uses MPI | Mature, highly programmable, stateful `Simulation` runtime with callbacks and access to lower-level field objects | JAX-native GPU execution, permissive licensing, immutable specs, explicit continuation state, detached results, and a narrower photonic-chip API | Far broader material models and analyses, 1D/cylindrical coordinates, symmetry support, MPI portability, and roughly two decades of use and validation |
| [**FDTDX**](https://fdtdx.readthedocs.io/en/stable/) | Local JAX on CPU/GPU/TPU with multi-GPU support; MIT | Functional JAX tree objects, automatic differentiation, and memory-efficient time-reversible gradients for 3D design | First-class 2D as well as 3D FDTD, a more explicit spec/plan/state/result separation, detached analysis layer, and an integrated GDS-to-S-parameter chip workflow | This is BeamZ's closest peer, not a package BeamZ clearly supersedes: FDTDX has a published JOSS paper and a stronger documented differentiable-solver story, especially its time-reversible gradient method |
| [**Tidy3D**](https://docs.flexcompute.com/projects/tidy3d/en/latest/) | Open-source Python client; proprietary GPU solver runs as a billed cloud service | Immutable, serializable simulation models and a polished client/job/result workflow around managed remote execution | The numerical engine is open, locally executable, modifiable, and usable without an account, network transfer, per-run credits, or a service boundary | Highly optimized managed compute, polished web GUI, automatic meshing and a much broader production ecosystem; likely the better choice when turnkey speed and support matter more than solver auditability |
| [**Ansys Lumerical FDTD**](https://www.ansys.com/products/optics/fdtd) | Commercial desktop/HPC/cloud product with CPU and multi-GPU resources | Integrated CAD, solver, scripting, optimization, foundry, and wider Ansys Optics workflows | Apache-licensed source, simpler installation and extension, no license server, and transparent local numerical kernels and compilation | Much more mature industrial tooling, support, foundry interoperability, nonuniform meshing, GUI workflows, and a broader solver suite including RCWA and multilayer tools |

## Bottom line

BeamZ is a better fit when the priority is **an auditable, hackable, local and
GPU-native photonic FDTD engine with explicit functional architecture**. It is not
yet a blanket replacement for these packages. Choose Meep for breadth and mature
open-source CPU/MPI science, FDTDX for the strongest current open JAX differentiable
FDTD story, Tidy3D for managed cloud performance, and Lumerical for an established
commercial photonics environment.

No controlled cross-solver benchmark is presented here, so this table intentionally
does **not** claim that BeamZ is faster or more accurate. Such claims should follow a
reproducible benchmark that compares error at equal scientific targets—not GCUPS
alone—as discussed in [performance and convergence](perf-and-conv.md).
