from beamz import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, yv, jvp, yvp

def get_mie_coefficients(x, m, n_max):
    """Calculate Mie coefficients an and bn for a sphere."""
    # x = k * a
    # m = n_sphere / n_medium
    an = []
    bn = []
    
    def riccati_bessel_psi(n, z):
        return z * (np.sqrt(np.pi / (2 * z)) * jv(n + 0.5, z))
    
    def riccati_bessel_zeta(n, z):
        # zeta = psi + i * chi
        psi = z * (np.sqrt(np.pi / (2 * z)) * jv(n + 0.5, z))
        chi = -z * (np.sqrt(np.pi / (2 * z)) * yv(n + 0.5, z))
        return psi + 1j * chi

    def d_riccati_bessel_psi(n, z):
        # derivative of z * jn(z)
        return (n + 1) * (np.sqrt(np.pi / (2 * z)) * jv(n + 0.5, z)) - z * (np.sqrt(np.pi / (2 * z)) * jv(n + 1.5, z))

    def d_riccati_bessel_zeta(n, z):
        psi_prime = (n + 1) * (np.sqrt(np.pi / (2 * z)) * jv(n + 0.5, z)) - z * (np.sqrt(np.pi / (2 * z)) * jv(n + 1.5, z))
        chi_prime = (n + 1) * (np.sqrt(np.pi / (2 * z)) * yv(n + 0.5, z)) - z * (np.sqrt(np.pi / (2 * z)) * yv(n + 1.5, z))
        return psi_prime + 1j * chi_prime

    for n in range(1, n_max + 1):
        # an = (m*psi_n(mx)*psi_n'(x) - psi_n(x)*psi_n'(mx)) / (m*psi_n(mx)*zeta_n'(x) - zeta_n(x)*psi_n'(mx))
        # bn = (psi_n(mx)*psi_n'(x) - m*psi_n(x)*psi_n'(mx)) / (psi_n(mx)*zeta_n'(x) - m*zeta_n(x)*psi_n'(mx))
        
        psi_x = riccati_bessel_psi(n, x)
        psi_mx = riccati_bessel_psi(n, m * x)
        zeta_x = riccati_bessel_zeta(n, x)
        
        d_psi_x = d_riccati_bessel_psi(n, x)
        d_psi_mx = d_riccati_bessel_psi(n, m * x)
        d_zeta_x = d_riccati_bessel_zeta(n, x)
        
        a = (m * psi_mx * d_psi_x - psi_x * d_psi_mx) / (m * psi_mx * d_zeta_x - zeta_x * d_psi_mx)
        b = (psi_mx * d_psi_x - m * psi_x * d_psi_mx) / (psi_mx * d_zeta_x - m * zeta_x * d_psi_mx)
        
        an.append(a)
        bn.append(b)
        
    return np.array(an), np.array(bn)

def calculate_mie_qext(radius, wavelength, n_sphere, n_medium=1.0):
    """Calculate extinction efficiency Qext from Mie theory."""
    k = 2 * np.pi * n_medium / wavelength
    x = k * radius
    m = n_sphere / n_medium
    
    # Heuristic for n_max
    n_max = int(round(x + 4 * x**(1/3) + 2))
    
    an, bn = get_mie_coefficients(x, m, n_max)
    
    n = np.arange(1, n_max + 1)
    qext = (2 / x**2) * np.sum((2 * n + 1) * np.real(an + bn))
    return qext

# --- Benchmark Simulation ---
def run_mie_benchmark():
    WL = 0.6*µm
    RADIUS = 0.3*µm
    N_SPHERE = 2.0
    N_CLAD = 1.0
    
    # Simulation domain
    X = Y = Z = 4.0*µm
    DX, DT = calc_optimal_fdtd_params(WL, N_SPHERE, dims=3)
    TIME = 15.0 * WL / LIGHT_SPEED
    time_steps = np.arange(0, TIME, DT)
    
    # 1. Total Field Run (with sphere)
    design_total = Design(X, Y, Z, material=Material(N_CLAD**2))
    design_total += Sphere(position=(X/2, Y/2, Z/2), radius=RADIUS, material=Material(N_SPHERE**2))
    
    signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, ramp_duration=3*WL/LIGHT_SPEED, t_max=TIME/2)
    # Use a large Gaussian to approximate a plane wave
    source = GaussianSource(position=(X/2, Y/2, 1.0*µm), width=1.5*µm, signal=signal)
    
    # Monitor behind the sphere to measure power
    monitor_pos_z = Z/2 + RADIUS + 0.5*µm
    monitor = Monitor(start=(X/2-1.5*µm, Y/2-1.5*µm, monitor_pos_z), 
                     end=(X/2+1.5*µm, Y/2+1.5*µm, monitor_pos_z), 
                     accumulate_power=True)
    
    sim_total = Simulation(design_total, devices=[source, monitor], boundaries=[PML(edges='all', thickness=WL)], time=time_steps, resolution=DX)
    print("Running Total Field Simulation...")
    sim_total.run()
    power_total = np.array(monitor.power_history)
    
    # 2. Incident Field Run (without sphere)
    design_inc = Design(X, Y, Z, material=Material(N_CLAD**2))
    monitor_inc = Monitor(start=(X/2-1.5*µm, Y/2-1.5*µm, monitor_pos_z), 
                         end=(X/2+1.5*µm, Y/2+1.5*µm, monitor_pos_z), 
                         accumulate_power=True)
    
    sim_inc = Simulation(design_inc, devices=[source, monitor_inc], boundaries=[PML(edges='all', thickness=WL)], time=time_steps, resolution=DX)
    print("Running Incident Field Simulation...")
    sim_inc.run()
    power_inc = np.array(monitor_inc.power_history)
    
    # 3. Analyze Results
    # For a dielectric sphere, Qext = Qsca. 
    # Extinction cross section sigma_ext = (P_inc - P_total) / I_inc
    # But since we use a Gaussian beam, we compare total power passing through the monitor.
    # A more robust way is to look at the peak or steady state.
    
    # Normalized power difference
    p_diff = (np.max(power_inc) - np.max(power_total)) / np.max(power_inc)
    # This is a very rough estimate of Qext for the beam area.
    # For a real Mie benchmark, we'd integrate the scattered field Poynting vector.
    
    qext_analytical = calculate_mie_qext(RADIUS, WL, N_SPHERE)
    
    print(f"\n--- Mie Scattering Results ---")
    print(f"Analytical Qext: {qext_analytical:.4f}")
    print(f"Simulated Power reduction factor: {p_diff:.4f}")
    print(f"Note: This is a qualitative verification. Precise Qext requires Scattered-Field integration.")

if __name__ == "__main__":
    run_mie_benchmark()

