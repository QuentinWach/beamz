from beamz import *
import numpy as np
import matplotlib.pyplot as plt

def run_dipole_benchmark():
    WL = 1.0*µm
    FREQ = LIGHT_SPEED / WL
    OMEGA = 2 * np.pi * FREQ
    
    # Simulation parameters
    SIZE = 4.0*µm
    DX, DT = calc_optimal_fdtd_params(WL, 1.0, dims=3, points_per_wavelength=15)
    TIME = 15 * WL / LIGHT_SPEED
    t = np.arange(0, TIME, DT)
    
    # Dipole parameters
    # J(t) = J0 * sin(omega * t)
    # I(t) = J(t) * V_eff
    # We use GaussianSource which injects E directly.
    # E_inc = -J * dt / (eps0 * eps_r)
    # Signal is our J.
    SIGMA = WL/10
    signal = ramped_cosine(t, amplitude=1.0, frequency=FREQ, ramp_duration=3*WL/LIGHT_SPEED, t_max=TIME/2)
    source = GaussianSource(position=(SIZE/2, SIZE/2, SIZE/2), width=SIGMA, signal=signal)
    
    # Calculate analytical total power
    # I0 = amplitude * (2 * pi * sigma^2)^(3/2)
    I0 = 1.0 * (2 * np.pi * SIGMA**2)**(1.5)
    # p0 = I0 / omega (since J = dP/dt)
    p0 = I0 / OMEGA
    p_analytical = (OMEGA**4 * p0**2) / (12 * np.pi * EPS_0 * LIGHT_SPEED**3)
    
    # Setup design
    design = Design(SIZE, SIZE, SIZE, material=Material(1.0))
    
    # Setup 6 monitors forming a box around the dipole
    L = 0.8*µm
    cx, cy, cz = SIZE/2, SIZE/2, SIZE/2
    
    # Plane monitors
    monitors = [
        Monitor(start=(cx-L, cy-L, cz+L), end=(cx+L, cy+L, cz+L), accumulate_power=True), # +z
        Monitor(start=(cx-L, cy-L, cz-L), end=(cx+L, cy+L, cz-L), accumulate_power=True), # -z
        Monitor(start=(cx-L, cy+L, cz-L), end=(cx+L, cy+L, cz+L), plane_normal='y', plane_position=cy+L, accumulate_power=True), # +y
        Monitor(start=(cx-L, cy-L, cz-L), end=(cx+L, cy-L, cz+L), plane_normal='y', plane_position=cy-L, accumulate_power=True), # -y
        Monitor(start=(cx+L, cy-L, cz-L), end=(cx+L, cy+L, cz+L), plane_normal='x', plane_position=cx+L, accumulate_power=True), # +x
        Monitor(start=(cx-L, cy-L, cz-L), end=(cx-L, cy+L, cz+L), plane_normal='x', plane_position=cx-L, accumulate_power=True), # -x
    ]
    
    sim = Simulation(design, devices=[source] + monitors, boundaries=[PML(edges='all', thickness=WL)], time=t, resolution=DX)
    
    print(f"Running 3D Dipole Benchmark (DX={DX/nm:.1f}nm)...")
    sim.run()
    
    # Total power is sum of peak power from all 6 monitors
    # (Since it's a pulsed-ish source, we look at the peak or integrate)
    total_sim_power = sum(np.max(m.power_history) for m in monitors)
    
    print(f"\n--- 3D Dipole Radiation Results ---")
    print(f"Analytical Power: {p_analytical:.4e}")
    print(f"Simulated Power:  {total_sim_power:.4e}")
    print(f"Relative Error:   {abs(total_sim_power - p_analytical)/p_analytical*100:.2f}%")

if __name__ == "__main__":
    run_dipole_benchmark()

