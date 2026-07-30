<div align="center">
  <picture style="padding-right: 0px; padding-bottom: 7px;">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/beamz_logo_black.png">
    <img alt="BeamZ logo" src="docs/assets/beamz_logo_white.png" width="130">
  </picture>

  <strong>BeamZ</strong> is a <strong>GPU-accelerated</strong> <strong><a href="https://en.wikipedia.org/wiki/Electromagnetism">electromagnetic</a> simulation</strong> framework for photonic chip designers using the <strong><a href="https://en.wikipedia.org/wiki/Finite-difference_time-domain_method">FDTD</a> method</strong>. It enables fast, large-scale simulations and offers a <strong>familiar, high-level API</strong> for fast prototyping with just a few lines of code as well as an <strong>inverse design module</strong> for gradient-based optimization using the <strong>adjoint method</strong>.

  <h3>

  [Homepage](https://www.beamz.tech) / [Documentation](https://www.beamz.tech/docs/index) / [Example Library](https://www.beamz.tech/examples)

  </h3>

  [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/beamzorg/beamz/blob/main/LICENSE)
  [![Tests](https://github.com/beamzorg/beamz/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/beamzorg/beamz/actions/workflows/tests.yml)
  [![Coverage](https://raw.githubusercontent.com/beamzorg/beamz/main/.github/badges/coverage.svg)](https://github.com/beamzorg/beamz/actions/workflows/tests.yml)
</div>



## Core Features
- **100% Python**, free (Apache-2.0 license) & open-source.
- FDTD simulation in **2D and 3D**.
- **GPU-accelerated**, achieving high **GCUPS performance**.
- **Multi-GPU** runs, handling **large-scale simulations** with _billions of cells_.
- CPU-capable for **fast prototyping**, even on your laptop.
- Intuitive and **familiar API**.
- Native **FDFD mode solver** with discrete Yee-grid refinement and validation.
- **CPML**, absorbing layers and PEC boundaries.
- Unidirectional **mode sources** (single freq. and broadband, Huygens fields + TFSF, TE/TM).
- **Gaussian sources**, e.g. for grating coupler simulations.
- Integrated **rasterization module**.
- **Sub-pixel averaging** using super-sampling.
- Custom source time profiles.
- Built-in layout flow (GDSII import/export).
- **DFT monitors** and S-parameter extraction workflow for compact modeling.
- Streamlined **parametric design** module.
- Optimization/autodiff utilities for gradient-based **inverse-design** with Jax.


## Examples
Try out notebooks from our growing **[example library](https://beamz.tech/examples/)**. It includes:

- [1) Mode Sources and Monitors](https://beamz.tech/examples/modal_sources_monitors)
- [2) Waveguide Crossing with Cosine Tapers](https://beamz.tech/examples/cosine_waveguide_crossing)
- [3) Topology Optimized 90° Bend (2D)](https://beamz.tech/examples/ceviche_bend)
<!--- [Broadband Mode Sources]() (coming soon)
- [Straight & Curved Waveguide Benchmark]() (coming soon)-->
<!--- [Mode Converter (3D)]() (coming soon)
- [DEMUX]() (coming soon)-->

## Integration 

BeamZ is used by several other OSS packages as an FDTD engine:
+ [SiEPIC's GDS FDTD](https://github.com/SiEPIC/gds_fdtd), an EDA- and solver-agnostic 3D FDTD compact modeling framework.
+ [Lumix](https://github.com/amiskandarmuda/lumix), a research codebase for optical neural networks and matrix inverse design.


## Installation

Install BeamZ using pip:

```bash
pip install beamz
```

Development uses [uv](https://docs.astral.sh/uv/). Clone the repository and sync
the package with its contributor dependencies:

```bash
git clone https://github.com/beamzorg/beamz
cd beamz
uv sync --extra dev --extra test
```

For a ready-to-use CUDA and Jupyter development environment, see the
[Docker and RunPod guide](docker/runpod/README.md).


## About
BeamZ's mission is to be the **pragmatic** FDTD engine of choice for **photonic chip designers**.

It focuses on **streamlined workflows** over **feature bloat** to produce **useful results** without tedious setup or configuration files and bringing GPU-acceleration for **maximum performance in large-scale simulations** to everyone.

The project is **actively maintained**. We aim to keep the code in Python, minimize dependencies, keep the line-count low, commented, and features local within the code to **make the code readable and development easy** so that - if there is something that isn't working or missing - you can quickly add it yourself. The engine is grounded in hundreds of tests, verifiable simulations and benchmarks, replicating known results from the established literature. Beyond benchmarking the core engine stats, we aim to **reduce friction for chip designers at every step** - from installation, to setting up the sim using a familiar API, to optimizing the performance of the rasterizer, mode solver, compiler, optimization loop, and integration into the overall chip design workflow.


## Contributing

**We appreciate all contributions.** If you are planning to contribute bug-fixes, please do so without any further discussion. If you would like to add new features, please first open an issue and discuss the feature with us. There may be ongoing work that could conflict with your changes, or we may be heading in a different direction and we don't want to waste your time working on something that might be rejected. - You can find [more information here](CONTRIBUTING.md).

The simplest way to support the project of course is by **giving this repo a star.** Thank you!

---

Copyright © 2026 Quentin Wach — [Apache-2.0](https://github.com/beamzorg/beamz/blob/HEAD/LICENSE)
