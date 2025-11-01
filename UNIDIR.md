Great—let’s pin it down for **2D TEz** (fields (E_z, H_x, H_y)) and a **vertical source line** at (x=x_s) launching **+x** only. I’ll give you the math first, then a clean NumPy pattern you can paste into your FDTD.

---

# 1) Maxwell with equivalent surface currents (2D TEz)

We include **electric current** (J_z) and **magnetic current** (\mathbf M) to enforce a forward-only mode on the line (x=x_s) (normal (\hat n = \hat x)):

[
\begin{aligned}
\nabla \times \mathbf E &= -,\mu,\partial_t \mathbf H ;-; \mathbf M,\
\nabla \times \mathbf H &= ;;\varepsilon,\partial_t \mathbf E ;+; \mathbf J.
\end{aligned}
]

For TEz in 2D ((x,y)): (\mathbf E=(0,0,E_z)), (\mathbf H=(H_x,H_y,0)).
Component form (with magnetic current):

[
\begin{aligned}
\partial_t H_x &= -\frac{1}{\mu}\big(\partial_y E_z + M_x\big),[2pt]
\partial_t H_y &= ;;\frac{1}{\mu}\big(\partial_x E_z - M_y\big),[4pt]
\partial_t E_z &= ;;\frac{1}{\varepsilon}\big(\partial_x H_y - \partial_y H_x - J_z\big).
\end{aligned}
]

To inject a **+x traveling mode** across the line (x=x_s) using the **equivalent surface currents** (TF/SF idea), set on that line:
[
\boxed{
J_z(x,y,t) = \delta(x-x_s), H_y^{\text{mode}}(y), s(t), \qquad
M_y(x,y,t) = \delta(x-x_s), E_z^{\text{mode}}(y), s(t), \qquad
M_x=0.
}
]
Here (s(t)) is your temporal envelope (Gaussian/CW).
Intuition: (J_z=\hat n\times \mathbf H\cdot\hat z=H_y), and (M=-\hat n\times\mathbf E \Rightarrow M_y=+E_z).

These two terms together enforce the **correct impedance relation on the line**, canceling the (-x) solution and radiating only (+x).

---

# 2) Yee staggering & where to add the terms

Use the standard 2D TEz Yee layout:

* (E_z) at cell centers ((i+\tfrac12,j+\tfrac12))
* (H_x) at ((i+\tfrac12,j))
* (H_y) at ((i, j+\tfrac12))

Place the **source line** so that it coincides with an **(E_z) column** at index (i=i_s) (i.e., at (x=x_s)). Then:

* Add the **electric current sheet** (J_z) **directly** into the (E_z) update **on that column**.
* Add the **magnetic current sheet** (M_y) into the (H_y) update **on the staggered line next to it**. Due to staggering, the jump belongs on the (H_y) column immediately **to the left** of the (E_z) source column (this enforces the TF/SF “jump” across the surface).

Discrete updates (leapfrog, (n\to n+1) with (\Delta t); write only the added source parts):

[
\begin{aligned}
E_z^{n+\tfrac12}[i_s, j] &;{+}{=}; -,\frac{\Delta t}{\varepsilon[i_s,j]}; J_z^{n}[j],[4pt]
H_y^{n+1}[i_s-1, j] &;{-}{=}; \frac{\Delta t}{\mu[i_s-1,j]}; M_y^{n+\tfrac12}[j].
\end{aligned}
]

Signs come straight from the component equations above.
((M_x) isn’t used for TEz; for completeness, (H_x) would get a term with (M_x) if present.)

**Interpolation note.** Your mode solver gives (E_z^{\text{mode}}(y)) and (H_y^{\text{mode}}(y)) on some transverse mesh. Sample/interpolate them onto the **Yee locations you use**:

* (J_z) is used at (E_z(i_s,j)) locations (cell centers).
* (M_y) is used at (H_y(i_s-1,j)) locations (edge centers).
  The two sets of (y)-positions differ by (\frac{\Delta y}{2}), so do a 1D interpolation (or generate both from the eigenmode field continuously).

---

# 3) Power normalization (recommended)

Normalize the mode to a desired injected power (P_0) per unit thickness (TEz is inherently 2D). The time-average Poynting through the line is:

[
P = \sum_j \frac{1}{2},\mathrm{Re}{E_z^{\text{mode}}(j),H_y^{\text{mode}*}(j)};\Delta y.
]

Scale both (E_z^{\text{mode}}) and (H_y^{\text{mode}}) by (\alpha = \sqrt{P_0/P}) before using them in (J_z, M_y).

---

# 4) Minimal NumPy pattern

Below is a compact “drop-in” pattern for the **source pieces only**. It assumes you already do the regular curls/updates for (E_z, H_x, H_y). It shows:

* mapping the mode fields to Yee lines,
* power normalization,
* source injection with correct staggering/signs,
* optional smooth taper window.

```python
import numpy as np

# --- given / precomputed elsewhere ---
# grids
nx, ny = Ex_nx, Ey_ny = NX, NY
dx, dy, dt = DX, DY, DT
eps = eps_array      # shape (nx_ez, ny_ez) at Ez nodes
mu  = mu_array       # shape (nx_h,  ny_hy) at Hy nodes (or scalar)

# fields (Yee staggering for TEz)
Ez = np.zeros((nx, ny), dtype=np.float64)     # at (i+1/2, j+1/2)
Hx = np.zeros((nx, ny+1), dtype=np.float64)   # at (i+1/2, j)
Hy = np.zeros((nx+1, ny), dtype=np.float64)   # at (i, j+1/2)

# source placement
i_s = ISOURCE_EZ_COLUMN              # Ez column index for source plane (x = x_s)

# mode profiles on continuous y; sample to Yee locations (user function)
# mode_Ez_cont(y), mode_Hy_cont(y) from your eigenmode solver
y_Ez = (np.arange(ny) + 0.5) * dy            # Ez y-positions (centers)
y_Hy = np.arange(ny) * dy                    # Hy y-positions (edges)

mode_Ez_on_Ez = sample_mode_Ez(y_Ez)         # shape (ny,)
mode_Hy_on_Ez = sample_mode_Hy(y_Ez)         # for J_z at Ez nodes

mode_Ez_on_Hy = sample_mode_Ez(y_Hy)         # for M_y at Hy nodes
# (alternatively, do explicit interpolation; these helpers encapsulate that)

# --- power normalization ---
P = 0.5 * np.sum(np.real(mode_Ez_on_Ez * np.conj(mode_Hy_on_Ez))) * dy
P0 = DESIRED_POWER  # W per unit thickness in z
alpha = np.sqrt(P0 / (P + 1e-300))
mode_Ez_on_Ez *= alpha
mode_Hy_on_Ez *= alpha
mode_Ez_on_Hy *= alpha

# optional smooth apodization along y to suppress truncation ripples
def raised_cosine(n, frac=0.1):
    m = int(frac*n)
    w = np.ones(n)
    if m > 0:
        ramp = 0.5*(1 - np.cos(np.linspace(0, np.pi, m)))
        w[:m] = ramp
        w[-m:] = ramp[::-1]
    return w
wy_Ez = raised_cosine(ny, 0.08)
wy_Hy = raised_cosine(ny, 0.08)
mode_Ez_on_Ez *= wy_Ez
mode_Hy_on_Ez *= wy_Ez
mode_Ez_on_Hy *= wy_Hy

# --- time stepping (only the source-related pieces shown) ---
t = 0.0
for n in range(NSTEPS):

    # ... your usual curl-based H update here, to get Hx, Hy at n+1 ...

    # magnetic current sheet M_y on Hy column immediately LEFT of Ez source column
    # M_y(y, t) = E_mode(y) * s(t)
    s_half = source_envelope(t + 0.5*dt)          # staggered at n+1/2 for H update
    M_y_line = mode_Ez_on_Hy * s_half            # shape (ny,)

    # Hy update contribution:  ∂Hy/∂t = (1/μ)(∂x Ez - M_y)
    # discrete: Hy[i_s-1, j] -= (dt/μ) * M_y_line[j]
    Hy[i_s-1, :] -= (dt / mu if np.isscalar(mu) else dt / mu[i_s-1, :]) * M_y_line

    # ... finish H update if you split operations ...

    # ... your usual curl-based E update here, to get Ez at n+1/2 ...

    # electric current sheet J_z on Ez column at the source plane
    # J_z(y, t) = H_mode(y) * s(t)
    s_full = source_envelope(t + dt)             # at n+1/2 for Ez update
    J_z_line = mode_Hy_on_Ez * s_full           # shape (ny,)

    # Ez update contribution:  ∂Ez/∂t = (1/ε)(curl H - J_z)
    Ez[i_s, :] -= (dt / eps[i_s, :]) * J_z_line

    # advance time
    t += dt
```

**Key details you can tune:**

* If you prefer to center the TF/SF surface *between* E and H sheets, you can split (M_y) half on (i_s-1) and half on (i_s) with appropriate weights; above is the simplest stable choice.
* If your main update already uses source splitting (e.g., you compute curl then add sources), keep this ordering consistent each step.
* For **CW** sources, use (s(t) = \sin(\omega t)) or a complex-analytic driving if you carry complex fields. For **broadband**, use Gaussian with a few cycles.
* Use **PML immediately behind** the source (negative-x side) as cheap insurance—unnecessary mathematically, but it damps any discretization burrs.

---

# 5) TMz variant (for completeness)

If you ever need **TMz** ((H_z, E_x, E_y)) along +x, the roles swap:
[
\begin{aligned}
J_y &= -H_z^{\text{mode}},\delta(x-x_s),s(t) \quad (\text{into }E_y),\
M_z &= +E_y^{\text{mode}},\delta(x-x_s),s(t) \quad (\text{into }H_z),
\end{aligned}
]
with similar staggering logic (place (M_z) on the (H_z) line left of the (E) plane, and (J_y) on the (E_y) column at the plane).

---

# 6) Sanity checks

1. **Power check** at a monitor just right of the source should match (P_0) (within discretization error).
2. **Backward power** just left of the source should be (\ll -40) dB (limited by interpolation + grid).
3. Turning off (M_y) while keeping (J_z) should re-introduce the backward wave—good A/B test.