<div align="left">
<img src="docs/assets/BEAMZ_logo.png" alt="BEAMZ" width="150" align="left" hspace="15" vspace="0"/>

BEAMZ is an **[electromagnetic](https://en.wikipedia.org/wiki/Electromagnetism) simulation** package for photonic chip designers using the **[FDTD](https://en.wikipedia.org/wiki/Finite-difference_time-domain_method) method** written in Jax. It features a **high-level API** for fast prototyping with just a few lines of code, an **inverse design module** for gradient-based optimization using the adjoint method with **[autodiff](https://en.wikipedia.org/wiki/Automatic_differentiation)**.
</div>

```bash
pip install beamz
```

![License](https://img.shields.io/github/license/QuentinWach/beamz)
[![Tests](https://github.com/QuentinWach/beamz/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/QuentinWach/beamz/actions/workflows/tests.yml)
[![Coverage](https://raw.githubusercontent.com/QuentinWach/beamz/main/.github/badges/coverage.svg)](https://github.com/QuentinWach/beamz/actions/workflows/tests.yml)


## Core Features
- **100% Python**, free (Apache-2.0 license) & open-source.
- Modular architecture with a high-level API.
- **GPU-accelerated** (but CPU-capable).
- Built-in layout flow (GDSII import/export).
- FDTD simulation in 2D and **3D**.
- Absorbing Layers, CPML (WIP), and PEC boundaries.
- **Sub-pixel smoothing** (using super-sampling).
- Gaussian and **mode sources** with TE and TM polarization.
- Custom source time profiles.
- **DFT monitors** and S-parameter extraction workflow for compact modeling.
- Streamlined parametric design module and interactive 3D web-view.
- Optimization/autodiff utilities for gradient-based **inverse-design** with Jax.


## Examples
Read and try out our **[example notebooks](https://beamzorg.github.io/beamz-notebooks/)** or download and run [`examples/` from this repository](https://github.com/beamzorg/beamz/tree/main/examples).


## About
BEAMZ's goal is to become the **pragmatic** FDTD engine of choice for **photonic chip designers**.

It focuses on **streamlined workflows** to produce **useful results** without tedious setup or configuration files. While currently still experimental, this is _not_ a research project with the goal to demo a novel framework we can publish, nor a costly, closed API that hides how it works and gives you no ownership. A **modular architecture** is chosen over a purely object-oriented architecture to **make the code readable and development easy** so that, if there is something that isn't working or missing, you can quickly add it yourself.

If any of this excites you or if have any questions, please open an issue on GitHub. Feel free to fork this project, to suggest or contribute new features, or simply support the project by **giving this repo a star.** Thank you!

---

Copyright © 2026 Quentin Wach — [Apache-2.0](https://github.com/beamzorg/beamz/blob/HEAD/LICENSE)