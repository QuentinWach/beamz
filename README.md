<div align="center">
  <picture style="padding-right: 23px; padding-bottom: 7px;">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/BEAMZ%20Dark.png">
    <img alt="BEAMZ logo" src="docs/assets/BEAMZ%20Light.png" width="380">
  </picture>

  <strong>BEAMZ</strong> is a <strong>GPU-accelerated</strong> <strong><a href="https://en.wikipedia.org/wiki/Electromagnetism">electromagnetic</a> simulation</strong> framework for photonic chip designers using the <strong><a href="https://en.wikipedia.org/wiki/Finite-difference_time-domain_method">FDTD</a> method</strong>. It features a highly optimized engine enabling fast large-scale simulations with a <strong>familiar high-level API</strong> for fast prototyping with just a few lines of code and an <strong>inverse design module</strong> for gradient-based optimization using the proven <strong>adjoint method</strong> with <strong><a href="https://en.wikipedia.org/wiki/Automatic_differentiation">autodiff</a></strong>.

  <h3>

  [Homepage](https://beamzorg.github.io/beamz-notebooks/docs/index) / [Documentation](https://beamzorg.github.io/beamz-notebooks/docs/index) / [Example Library](https://beamzorg.github.io/beamz-notebooks/examples/)

  </h3>

  [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/beamzorg/beamz/blob/main/LICENSE)
  [![Tests](https://github.com/beamzorg/beamz/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/beamzorg/beamz/actions/workflows/tests.yml)
  [![Coverage](https://raw.githubusercontent.com/beamzorg/beamz/main/.github/badges/coverage.svg)](https://github.com/beamzorg/beamz/actions/workflows/tests.yml)
</div>



## Core Features
- **100% Python**, free (Apache-2.0 license) & open-source.
- FDTD simulation in **2D and 3D**.
- **GPU-accelerated**, achieving extremely high **GCUPS performance**.
- Handling **large-scale simulations** with _billions of cells_.
- CPU-capable for **fast prototyping**, even on your laptop.
- Modular architecture with an **intuitive and familiar, high-level API**.
- **Native FDFD mode solver, _[micromode](https://github.com/beamzorg/micromode)_**.
- **CPML**, absorbing layers and PEC boundaries.
- **Mode** and other sources  with TE and TM polarization.
- **Sub-pixel averaging** using super-sampling.
- Custom source time profiles.
- Built-in layout flow (GDSII import/export).
- **DFT monitors** and S-parameter extraction workflow for compact modeling.
- Streamlined **parametric design** module and **interactive 3D web-view**.
- Optimization/autodiff utilities for gradient-based **inverse-design** with Jax.



## Examples
Try out notebooks from our growing **[example library](https://beamzorg.github.io/beamz-notebooks/)**. 
The repository examples include an [index](examples/README.md) that separates
good first reads from advanced or experimental workflows.


## Installation

Get started with

```bash
pip install beamz
```


## About
BEAMZ's mission is to be the **pragmatic** FDTD engine of choice for **photonic chip designers**.

It focuses on **streamlined workflows** to produce **useful results** without tedious setup or configuration files and bringing GPU-acceleration for **maximum performance in large-scale simulations** to everyone. The project is **actively maintained** and this is _not_ a research project with the goal to demo a novel framework we can publish, nor a costly, closed API that hides how it works and gives you no ownership. A **modular architecture** is chosen over a purely object-oriented architecture to **make the code readable and development easy** so that - if there is something that isn't working or missing - you can quickly add it yourself. The engine is grounded in hundreds of tests, verifiable simulations and benchmarks, replicating known results from the established literature. Rather than just benchmarking for impressive engine stats, we aim to **reduce friction for chip designers at every step** - from installation, to setting up the sim using a familiar API, to optimizing the performance of the rasterizer, mode solver, compiler, the core engine, optimization loop, and integration into the overall chip design workflow.

If any of this excites you or if have any questions, please open an issue on GitHub. Feel free to fork this project, to suggest or contribute new features, or simply support the project by **giving this repo a star.** Thank you!

---

Copyright © 2026 Quentin Wach — [Apache-2.0](https://github.com/beamzorg/beamz/blob/HEAD/LICENSE)
