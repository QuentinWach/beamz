<div align="left">
<img src="docs/assets/BEAMZ_logo.png" alt="BEAMZ" width="150" align="left" hspace="15" vspace="0"/>

BEAMZ is an **electromagnetic simulation** package using the FDTD method. It features a **high-level API** for fast prototyping with just a few lines of code, an **inverse design module** for topology optimization using the adjoint method with **Jax-based autodiff** and a thermal solver. Made for (but not limited to) photonic integrated circuits.
</div>

```bash
pip install beamz
```

![PyPI](https://img.shields.io/pypi/v/beamz?color=black)
![License](https://img.shields.io/github/license/QuentinWach/beamz?color=black)
![Last Update](https://img.shields.io/github/last-commit/QuentinWach/beamz?color=black)
![Stargazers](https://img.shields.io/github/stars/QuentinWach/beamz)


## Get Started
Read and try out our **[example notebooks](https://quentinwach.com/beamz-notebooks/)** or download and run `examples/` from this repository.


## Features

- 100% Python, Free & Open-Source.
- Modular Architecture with High-level API.
- GPU accelerated.
- Built-in layout flow (GDS import/export).
- 2D/3D simulation.
- PML boundaries.
- Gaussian and mode sources.
- TE/TM polarization.
- Monitors and field recording/visualization.
- Thermal workflows (transient coupling + static thermal solves).
- Optimization/autodiff utilities for inverse-design with Jax.


## Planned / Work in Progress

- [ ] Native dispersive EM time-domain models (Drude/Lorentz/Sellmeier/Debye).
- [ ] Simulation benchmarks and performance improvements (especially in 3D).
- [ ] Explicit production-grade multi-GPU scaling.
- [ ] Enable full polarization control in mode solving beyond TE and TM.