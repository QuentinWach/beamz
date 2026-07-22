# Performance and Convergence

A key metric for any FDTD engine is how many giga-cell updates per second (GUPCS) it capable of. But GCUPS only measures the denominator:

\[
T_{\text{solution}} \approx \frac{N_{\text{cells}}N_{\text{timesteps}}} {\text{GCUPS}\times10^9} +T_{\text{setup}}
\]

It says nothing about whether $S_{21}$, loss, resonance wavelength, $Q$, or an adjoint gradient is accurate. A solver can have excellent GCUPS but still be slower scientifically if it needs a finer grid, longer decay time, thicker PML, or repeated manual reruns.

In 3D, halving the grid spacing creates roughly $8\times$ more cells and $2\times$ more steps for the same physical duration: about $16\times$ more updates. Therefore, a discretization that reaches the same error on a coarser grid can outweigh a large raw-GCUPS advantage.

There are two relevant meanings of convergence:
- **Temporal convergence:** after the source turns off, fields and frequency-domain monitor values have settled.
- **Numerical convergence:** the quantity of interest stops changing materially as resolution, timestep, domain padding, and PML thickness are refined.

FDTD itself does not iteratively converge at each timestep; it evolves Maxwell’s equations explicitly. That's why it is important to optimize the tooling around the core-engine just as much.
