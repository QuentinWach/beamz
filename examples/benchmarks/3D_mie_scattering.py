from beamz import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, yv

class GaussianBeamSource:
    """Injects a Gaussian beam propagating in the +x direction."""
    def __init__(self, x_pos, signal, w0, center_y, center_z):
        self.x_pos = x_pos
        self.signal = signal
        self.w0 = w0
        self.cy = center_y
        self.cz = center_z
        self._indices = None
        self._profile = None

    def inject(self, fields, t, dt, current_step, resolution, design):
        if self._indices is None:
            ix = int(round(self.x_pos / resolution))
            # Ez shape: (nz-1, ny, nx) -> check Fields class
            # Actually fields.Ez might be (nz, ny, nx) depending on implementation details 
            # but usually Ez is on z-edges.
            # Let's check shape at runtime.
            nz_ez, ny_ez, nx_ez = fields.Ez.shape
            self._indices = (slice(0, nz_ez), slice(0, ny_ez), ix)
            
            # Create coordinate grids for the slice
            # Ez is at (z+0.5, y, x)
            z_coords = (np.arange(nz_ez) + 0.5) * resolution
            y_coords = (np.arange(ny_ez)) * resolution
            
            Z, Y = np.meshgrid(z_coords, y_coords, indexing='ij')
            
            r2 = (Y - self.cy)**2 + (Z - self.cz)**2
            self._profile = np.exp(-r2 / (self.w0**2))
            
        idx_float = t / dt
        idx_low = int(np.floor(idx_float))
        if 0 <= idx_low < len(self.signal) - 1:
            frac = idx_float - idx_low
            val = (1.0 - frac) * self.signal[idx_low] + frac * self.signal[idx_low+1]
        else: val = 0.0
        
        # Soft source injection
        fields.Ez[self._indices] += -val * self._profile * dt / (8.85e-12 * 1.0)

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

def dft_fields(time_data, t_array, freq):
    """Compute single-frequency component from time-domain data."""
    # time_data shape: (steps, ...)
    # t_array shape: (steps,)
    # result shape: (...)
    return np.sum(time_data * np.exp(-1j * 2 * np.pi * freq * t_array)[:, None, None], axis=0)

def run_mie_benchmark():
    WL, RADIUS, N_SPHERE, N_CLAD = 0.6*µm, 0.3*µm, 2.0, 1.0
    # Revert to smaller domain for speed, using narrower beam
    X, Y, Z = 4.0*µm, 3.0*µm, 3.0*µm 
    DX, DT = calc_optimal_fdtd_params(WL, N_SPHERE, dims=3, points_per_wavelength=12)
    TIME = 45.0 * WL / LIGHT_SPEED
    t_steps = np.arange(0, TIME, DT)
    L = RADIUS + 0.3*µm
    cx, cy, cz = X/2 + 0.5*µm, Y/2, Z/2
    FREQ = LIGHT_SPEED / WL
    
    def create_box_monitors():
        return [
            Monitor(start=(cx-L, cy-L, cz+L), end=(cx+L, cy+L, cz+L), name='z_pos'),
            Monitor(start=(cx-L, cy-L, cz-L), end=(cx+L, cy+L, cz-L), name='z_neg'),
            Monitor(start=(cx-L, cy+L, cz-L), end=(cx+L, cy+L, cz+L), name='y_pos'),
            Monitor(start=(cx-L, cy-L, cz-L), end=(cx+L, cy-L, cz+L), name='y_neg'),
            Monitor(start=(cx+L, cy-L, cz-L), end=(cx+L, cy+L, cz+L), name='x_pos'),
            Monitor(start=(cx-L, cy-L, cz-L), end=(cx-L, cy+L, cz+L), name='x_neg')
        ]

    signal = ramped_cosine(t_steps, amplitude=1.0, frequency=FREQ, ramp_duration=5*WL/LIGHT_SPEED, t_max=TIME/2)
    # w0=0.8um is large enough for r=0.3 sphere, small enough to decay before PML at +/- 1.5
    source = GaussianBeamSource(x_pos=0.8*µm, signal=signal, w0=0.8*µm, center_y=cy, center_z=cz)
    
    print("Running Total Field Simulation...")
    d_tot = Design(X, Y, Z, material=Material(N_CLAD**2))
    d_tot += Sphere(position=(cx, cy, cz), radius=RADIUS, material=Material(N_SPHERE**2))
    m_tot = create_box_monitors()
    Simulation(d_tot, devices=[source] + m_tot, boundaries=[PML(edges='all', thickness=WL)], time=t_steps, resolution=DX).run()
    
    print("Running Incident Field Simulation...")
    d_inc = Design(X, Y, Z, material=Material(N_CLAD**2))
    m_inc = create_box_monitors()
    # Add center monitor to measure incident intensity exactly at sphere location
    # Use small finite size to ensure valid slicing
    m_center = Monitor(start=(cx, cy-DX, cz-DX), end=(cx, cy+DX, cz+DX), name='center')
    Simulation(d_inc, devices=[source] + m_inc + [m_center], boundaries=[PML(edges='all', thickness=WL)], time=t_steps, resolution=DX).run()
    
    p_scat_total = 0
    print("\nProcessing Frequency-Domain Scattered Fields...")
    
    for i in range(6):
        def get_dft(m, comp):
            # Convert list to array (steps, n1, n2)
            arr = np.array(m.fields[comp])
            # Perform DFT to get phasor at target frequency
            return dft_fields(arr, t_steps, FREQ)
            
        ex = get_dft(m_tot[i], 'Ex') - get_dft(m_inc[i], 'Ex')
        ey = get_dft(m_tot[i], 'Ey') - get_dft(m_inc[i], 'Ey')
        ez = get_dft(m_tot[i], 'Ez') - get_dft(m_inc[i], 'Ez')
        hx = get_dft(m_tot[i], 'Hx') - get_dft(m_inc[i], 'Hx')
        hy = get_dft(m_tot[i], 'Hy') - get_dft(m_inc[i], 'Hy')
        hz = get_dft(m_tot[i], 'Hz') - get_dft(m_inc[i], 'Hz')
        
        # S = 0.5 * Re(E x H*)
        name = m_tot[i].name
        if   name == 'x_pos': flux =  (ey * np.conj(hz) - ez * np.conj(hy))
        elif name == 'x_neg': flux = -(ey * np.conj(hz) - ez * np.conj(hy))
        elif name == 'y_pos': flux =  (ez * np.conj(hx) - ex * np.conj(hz))
        elif name == 'y_neg': flux = -(ez * np.conj(hx) - ex * np.conj(hz))
        elif name == 'z_pos': flux =  (ex * np.conj(hy) - ey * np.conj(hx))
        elif name == 'z_neg': flux = -(ex * np.conj(hy) - ey * np.conj(hx))
        
        # Integrate flux over area
        p_scat_total += np.sum(0.5 * np.real(flux)) * (DX * DX)

    # Calculate Incident Intensity at sphere center (from m_center)
    # m_center fields are shape (steps, 1, 1) or (steps,) depending on implementation.
    # Usually (steps, nz, ny) with nz=1, ny=1.
    ez_inc_c = dft_fields(np.array(m_center.fields['Ez']), t_steps, FREQ)
    hy_inc_c = dft_fields(np.array(m_center.fields['Hy']), t_steps, FREQ)
    
    # S_inc = 0.5 * Re(-Ez * Hy*)
    # Take first element if array
    if ez_inc_c.ndim > 0: ez_inc_c = ez_inc_c.flat[0]
    if hy_inc_c.ndim > 0: hy_inc_c = hy_inc_c.flat[0]
        
    i_inc = 0.5 * np.real(-ez_inc_c * np.conj(hy_inc_c))
    
    qext_sim = p_scat_total / (i_inc * np.pi * RADIUS**2)
    qext_theory = calculate_mie_qext(RADIUS, WL, N_SPHERE)
    
    print(f"\n--- Mie Scattering (Frequency-Domain) ---")
    print(f"Analytical Qext: {qext_theory:.4f}\nSimulated  Qext: {qext_sim:.4f}")
    print(f"Relative Error:  {abs(qext_sim - qext_theory)/qext_theory*100:.2f}%")
    
    if abs(qext_sim - qext_theory)/qext_theory < 0.3:
        print("✅ Benchmark Passed! (Error < 30%)")
    else:
        print("❌ Benchmark Failed! (Error > 30%)")

if __name__ == "__main__":
    run_mie_benchmark()
