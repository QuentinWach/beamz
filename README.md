<div align="left">
<img src="docs/assets/BEAMZ_logo.png" alt="BEAMZ" width="150" align="left" hspace="15" vspace="0"/>

BEAMZ is an **[electromagnetic](https://en.wikipedia.org/wiki/Electromagnetism) simulation** package using the [FDTD](https://en.wikipedia.org/wiki/Finite-difference_time-domain_method) method. It features a **high-level API** for fast prototyping with just a few lines of code, an **inverse design module** for topology optimization using the adjoint method with **Jax-based [autodiff](https://en.wikipedia.org/wiki/Automatic_differentiation)** and a thermal solver. Made for (but not limited to) photonic integrated circuits.
</div>

```bash
pip install beamz
```

![License](https://img.shields.io/github/license/QuentinWach/beamz)
![Last Update](https://img.shields.io/github/last-commit/QuentinWach/beamz)
![Stargazers](https://img.shields.io/github/stars/QuentinWach/beamz)


## ✨ Core Features
- **100% Python**, free (MIT license; not just GPL) & open-source.
- Modular architecture with a high-level API.
- **GPU-accelerated** (but CPU capable).
- Built-in layout flow (GDSII import/export).
- FDTD simulation in 2D and 3D.
- PML absorbing boundaries.
- Gaussian and mode sources with TE and TM polarization.
- Custom source time profiles (array/callable signal support).
- Dedicated visualization module for ...almost everything.
- Streamlined parametric design module.
- Thermal workflows (transient coupling + static thermal solves).
- Optimization/autodiff utilities for **inverse-design** with Jax.


## 🚀 Example Library
Read and try out our **[example notebooks](https://quentinwach.com/beamz-notebooks/)** or download and run `examples/` from this repository.


---


## Planned / Work in Progress
- [ ] Native dispersive EM time-domain models (Drude/Lorentz/Sellmeier/Debye).
- [ ] Simulation benchmarks and performance improvements (especially in 3D).
- [ ] Explicit production-grade multi-GPU scaling.
- [ ] Enable full polarization control in mode solving beyond TE and TM.


- [ ] Cylindrical-coordinate simulation mode.
- [ ] Official Conda precompiled package channel.
- [ ] Arbitrary spatial custom-current source profile API parity.
- [ ] Broader material model parity:
  - [ ] Anisotropic epsilon/mu tensors.
  - [ ] Native dispersive epsilon(omega)/mu(omega) updates in FDTD stepping.
  - [ ] Nonlinear Kerr/Pockels material models.
  - [ ] Saturable gain/absorption models.
  - [ ] Gyrotropic (magneto-optical) media.
- [ ] Built-in broadband materials library (predefined complex index datasets).
- [ ] Boundary-condition parity beyond PML:
  - [ ] Bloch-periodic boundaries.
  - [ ] Perfect-conductor boundary conditions.
- [ ] Symmetry-exploitation API (mirror/rotation domain reduction).
- [ ] Full subpixel-smoothing parity for accuracy/shape optimization.
- [ ] Frequency-domain solver (CW response).
- [ ] Frequency-domain eigensolver (resonant modes).
- [ ] HDF5 epsilon/mu and field import/export workflows.
- [ ] Advanced field-analysis parity:
  - [ ] DTFT/FFT field-spectrum monitor APIs.
  - [ ] Mode decomposition / S-parameter pipeline.
  - [ ] Near-to-far-field transforms.
  - [ ] Frequency extraction helpers.
  - [ ] LDOS and modal-volume analysis.
  - [ ] Energy-density spectra APIs.
  - [ ] Maxwell stress tensor analysis.
  - [ ] Absorbed power density analysis API.
  - [ ] Programmable arbitrary field-function analysis parity.
- [ ] Sanitizer CI pipeline parity.
- [ ] Hosted "latest docs" badge/documentation deployment parity.
- [ ] Formal citation guidance section in README/docs.

## FDTDX Feature Coverage (Implemented vs WIP)

This section mirrors the main points from the FDTDX README and marks whether the same capability is already available in BEAMZ.

### Already Implemented in BEAMZ
- [x] JAX-based autodiff utilities for gradient-based inverse-design workflows.
- [x] GPU-capable execution path (JAX backend on supported hardware).
- [x] 3D simulation support (plus 2D support).
- [x] High-level, user-friendly Python API for geometry/source/monitor setup.
- [x] Public docs/examples/notebooks for onboarding and API usage.
- [x] CI and code coverage integrated in the repository.

### TODO / WIP for FDTDX-Level Parity
- [ ] Explicit production-grade multi-GPU scaling workflow and tuning guidance.
- [ ] Memory-efficiency strategy parity for reverse/adjoint gradients at very large scale.
- [ ] "Billions of grid cells" scale claims backed by benchmarks and docs.
- [ ] Installation profile parity with accelerator-specific extras (e.g. CUDA/ROCm install paths).
- [ ] Consolidated "best practices" performance guide for large JAX runs.
- [ ] Formal citation block in README/docs (paper/JOSS style entry).


## About
BEAMZ's goal is to become the **pragmatic** FDTD engine of choice for **photonic chip designers**.

It focuses on **stream-lined workflows** to produce **usefuly results** without tedious setup or configuration files. I.e. this is _not_ a research project with the goal to demo a novel framework we can publish nor a costly, closed API that hides how it works and gives you no ownership. 

We are building in python and choosing a modular architecture that is composible over a brutalist object-oriented architecture to make the code readable and development easy. So that, if there is something that isn't working or missing, you can quickly add it yourself!


## Contributing
If you have any questions, please open an issue on GitHub! And feel free to fork this project, to suggest or contribute new features. The WIP section contains a list of features that are planned to be implemented. Help is very much appreciated! That said, the easiest way to support the project is to **give this repo a ⭐!**

Thank you!
