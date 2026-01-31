<div align="left">
<img src="docs/assets/BEAMZ_logo.png" alt="BEAMZ" width="150" align="left" hspace="15" vspace="0"/>

BEAMZ is an **electromagnetic simulation** package using the FDTD method. It features a **high-level API** for fast prototyping with just a few lines of code as well as an **inverse design module** for topology optimization using the adjoint method with **Jax-based autodiff**. Made for (but not limited to) photonic integrated circuits.
</div>

```bash
uv pip install beamz
```

![PyPI](https://img.shields.io/pypi/v/beamz?color=black)
![License](https://img.shields.io/github/license/QuentinWach/beamz?color=black)
![Last Update](https://img.shields.io/github/last-commit/QuentinWach/beamz?color=black)
![Stargazers](https://img.shields.io/github/stars/QuentinWach/beamz)


---

<div align="left">
  <img src="docs/assets/4_topo.png" alt="Example topology optimization result" width="200" align="right" style="border-radius: 15px; margin-left: 15px;"/>
  Design your first gradient-optimized 90°-bend topology in under 5 min by copying the example script from this repo and running <code>uv run python examples/4_topology.py</code>.
  <span>This design predicts a high broadband transmission of &gt;98% from 1300&nbsp;nm to 1800&nbsp;nm.</span>
</div>