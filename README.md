<div align="left">
  <strong>BEAMZ</strong> is a <strong>GPU-accelerated</strong> <strong><a href="https://en.wikipedia.org/wiki/Electromagnetism">electromagnetic</a> simulation</strong> package for photonic chip designers using the <strong><a href="https://en.wikipedia.org/wiki/Finite-difference_time-domain_method">FDTD</a> method</strong>. It features a <strong>familiar high-level API</strong> for fast prototyping with just a few lines of code and an <strong>inverse design module</strong> for gradient-based optimization using the adjoint method with <strong><a href="https://en.wikipedia.org/wiki/Automatic_differentiation">autodiff</a></strong>.
</div>


```bash
pip install beamz
```

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/beamzorg/beamz/blob/main/LICENSE)
[![Tests](https://github.com/beamzorg/beamz/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/beamzorg/beamz/actions/workflows/tests.yml)
[![Coverage](https://raw.githubusercontent.com/beamzorg/beamz/main/.github/badges/coverage.svg)](https://github.com/beamzorg/beamz/actions/workflows/tests.yml)


## Core Features
- **100% Python**, free (Apache-2.0 license) & open-source.
- **GPU-accelerated** (but CPU-capable) to achieve high 10-100 GCUPS performance.
- Modular architecture with a **familiar high-level API**.
- FDTD simulation in **2D and 3D**.
- **Native FDFD mode solver - _[micromode](https://github.com/beamzorg/micromode)_**.
- **CPML**, absorbing layers and PEC boundaries.
- **Mode** and other sources  with TE and TM polarization.
- **Sub-pixel averaging** using super-sampling.
- Custom source time profiles.
- Built-in layout flow (GDSII import/export).
- **DFT monitors** and S-parameter extraction workflow for compact modeling.
- Streamlined **parametric design** module and **interactive 3D web-view**.
- Optimization/autodiff utilities for gradient-based **inverse-design** with Jax.


## Examples
Read and try out our **[example notebooks](https://beamzorg.github.io/beamz-notebooks/)** (recommended) or download and run [`examples/` from this repository](https://github.com/beamzorg/beamz/tree/main/examples).


## About
BEAMZ's mission is to be the **pragmatic** FDTD engine of choice for **photonic chip designers**.

It focuses on **streamlined workflows** to produce **useful results** without tedious setup or configuration files. The project is under active development yet this is _not_ a research project with the goal to demo a novel framework we can publish, nor a costly, closed API that hides how it works and gives you no ownership. A **modular architecture** is chosen over a purely object-oriented architecture to **make the code readable and development easy** so that - if there is something that isn't working or missing - you can quickly add it yourself.

If any of this excites you or if have any questions, please open an issue on GitHub. Feel free to fork this project, to suggest or contribute new features, or simply support the project by **giving this repo a star.** Thank you!

---

Copyright © 2026 Quentin Wach — [Apache-2.0](https://github.com/beamzorg/beamz/blob/HEAD/LICENSE)
