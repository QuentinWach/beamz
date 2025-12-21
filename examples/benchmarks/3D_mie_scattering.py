from beamz import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, yv

class UniformSource:
    """Injects a uniform plane wave propagating in the +x and -x directions."""
    def __init__(self, x_pos, signal):
        self.x_pos = x_pos
        self.signal = signal
        self._indices = None

    def inject(self, fields, t, dt, current_step, resolution, design):
        if self._indices is None:
            ix = int(round(self.x_pos / resolution))
            nz, ny, nx = fields.Ez.shape
            self._indices = (slice(0, nz), slice(0, ny), ix)
        idx_float = t / dt
        idx_low = int(np.floor(idx_float))
        if 0 <= idx_low < len(self.signal) - 1:
            frac = idx_float - idx_low
            val = (1.0 - frac) * self.signal[idx_low] + frac * self.signal[idx_low+1]
        else: val = 0.0
        # Soft source Ez injection
        fields.Ez[self._indices] += -val * dt / (8.85e-12 * 1.0)

def get_mie_coefficients(x, m, n_max):
    an, bn = [], []
    def psi(n, z): return z * (np.sqrt(np.pi / (2 * z)) * jv(n + 0.5, z))
    def zeta(n, z): return psi(n, z) - 1j * z * (np.sqrt(np.pi / (2 * z)) * yv(n + 0.5, z))
    def psi_prime(n, z): return (n + 1) * (np.sqrt(np.pi / (2 * z)) * jv(n + 0.5, z)) - z * (np.sqrt(np.pi / (2 * z)) * jv(n + 1.5, z))
    def zeta_prime(n, z):
        return ((n+1)*jv(n+0.5,z) - z*jv(n+1.5,z))*np.sqrt(np.pi/(2*z)) - 1j*((n+1)*yv(n+0.5,z) - z*yv(n+1.5,z))*np.sqrt(np.pi/(2*z))
    for n in range(1, n_max + 1):
        px, pmx, zx = psi(n, x), psi(n, m * x), zeta(n, x)
        ppx, ppmx, zpx = psi_prime(n, x), psi_prime(n, m * x), zeta_prime(n, x)
        an.append((m * pmx * ppx - px * ppmx) / (m * pmx * zpx - zx * ppmx))
        bn.append((pmx * ppx - m * px * ppmx) / (pmx * zpx - m * zx * ppmx))
    return np.array(an), np.array(bn)

def calculate_mie_qext(radius, wavelength, n_sphere, n_medium=1.0):
    k, m = 2 * np.pi * n_medium / wavelength, n_sphere / n_medium
    x = k * radius
    n_max = int(round(x + 4 * x**(1/3) + 2))
    an, bn = get_mie_coefficients(x, m, n_max)
    return (2 / x**2) * np.sum((2 * np.arange(1, n_max + 1) + 1) * np.real(an + bn))

def run_mie_benchmark():
    WL, RADIUS, N_SPHERE, N_CLAD = 0.6*µm, 0.3*µm, 2.0, 1.0
    X, Y, Z = 4.0*µm, 3.0*µm, 3.0*µm
    DX, DT = calc_optimal_fdtd_params(WL, N_SPHERE, dims=3, points_per_wavelength=12)
    TIME = 45.0 * WL / LIGHT_SPEED
    t_steps = np.arange(0, TIME, DT)
    L = RADIUS + 0.3*µm
    cx, cy, cz = X/2 + 0.5*µm, Y/2, Z/2
    
    def create_box_monitors():
        return [
            Monitor(start=(cx-L, cy-L, cz+L), end=(cx+L, cy+L, cz+L), name='z_pos'),
            Monitor(start=(cx-L, cy-L, cz-L), end=(cx+L, cy+L, cz-L), name='z_neg'),
            Monitor(start=(cx-L, cy+L, cz-L), end=(cx+L, cy+L, cz+L), name='y_pos'),
            Monitor(start=(cx-L, cy-L, cz-L), end=(cx+L, cy-L, cz+L), name='y_neg'),
            Monitor(start=(cx+L, cy-L, cz-L), end=(cx+L, cy+L, cz+L), name='x_pos'),
            Monitor(start=(cx-L, cy-L, cz-L), end=(cx-L, cy+L, cz+L), name='x_neg')
        ]

    signal = ramped_cosine(t_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, ramp_duration=5*WL/LIGHT_SPEED, t_max=TIME/2)
    source = UniformSource(x_pos=0.8*µm, signal=signal)
    
    print("Running Total Field Simulation...")
    d_tot = Design(X, Y, Z, material=Material(N_CLAD**2))
    d_tot += Sphere(position=(cx, cy, cz), radius=RADIUS, material=Material(N_SPHERE**2))
    m_tot = create_box_monitors()
    Simulation(d_tot, devices=[source] + m_tot, boundaries=[PML(edges='all', thickness=WL)], time=t_steps, resolution=DX).run()
    
    print("Running Incident Field Simulation...")
    d_inc = Design(X, Y, Z, material=Material(N_CLAD**2))
    m_inc = create_box_monitors()
    Simulation(d_inc, devices=[source] + m_inc, boundaries=[PML(edges='all', thickness=WL)], time=t_steps, resolution=DX).run()
    
    p_scat_total, dt = 0, t_steps[1] - t_steps[0]
    print("\nProcessing Scattered Field Integration...")
    for i in range(6):
        def get_arr(m, comp): return np.array(m.fields[comp])
        ex, ey, ez = [get_arr(m_tot[i], c) - get_arr(m_inc[i], c) for c in ['Ex', 'Ey', 'Ez']]
        hx, hy, hz = [get_arr(m_tot[i], c) - get_arr(m_inc[i], c) for c in ['Hx', 'Hy', 'Hz']]
        name = m_tot[i].name
        if   name == 'x_pos': flux =  (ey * hz - ez * hy)
        elif name == 'x_neg': flux = -(ey * hz - ez * hy)
        elif name == 'y_pos': flux =  (ez * hx - ex * hz)
        elif name == 'y_neg': flux = -(ez * hx - ex * hz)
        elif name == 'z_pos': flux =  (ex * hy - ey * hx)
        elif name == 'z_neg': flux = -(ex * hy - ey * hx)
        p_scat_total += np.sum(flux) * (DX * DX) * dt

    m_in = m_inc[5] # x_neg
    # X-monitor fields are (steps, nz, ny). Index center of NZ, NY.
    ez_vals = np.array(m_in.fields['Ez'])
    hy_vals = np.array(m_in.fields['Hy'])
    iz, iy = ez_vals.shape[1]//2, ez_vals.shape[2]//2
    u_inc_c = np.sum(-ez_vals[:, iz, iy] * hy_vals[:, iz, iy]) * dt
    
    qext_sim = p_scat_total / (u_inc_c * np.pi * RADIUS**2)
    qext_theory = calculate_mie_qext(RADIUS, WL, N_SPHERE)
    print(f"\n--- Mie Scattering (Uniform Source Method) ---")
    print(f"Analytical Qext: {qext_theory:.4f}\nSimulated  Qext: {qext_sim:.4f}")
    print(f"Relative Error:  {abs(qext_sim - qext_theory)/qext_theory*100:.2f}%")

if __name__ == "__main__":
    run_mie_benchmark()
