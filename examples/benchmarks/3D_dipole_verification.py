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
    TIME = 20 * WL / LIGHT_SPEED
    t = np.arange(0, TIME, DT)
    
    # Dipole parameters
    SIGMA = WL/12
    AMPLITUDE = 1.0 
    signal = ramped_cosine(t, amplitude=AMPLITUDE, frequency=FREQ, ramp_duration=5*WL/LIGHT_SPEED, t_max=TIME/2)
    source = GaussianSource(position=(SIZE/2, SIZE/2, SIZE/2), width=SIGMA, signal=signal)
    
    # Calculate analytical total power
    # In beamz, GaussianSource amplitude A is peak current density J0
    # Total integrated current amplitude I0 = J0 * integral(exp(-r^2/2sigma^2) dV)
    # I0 = J0 * (2 * pi * sigma^2)^(1.5)
    vol_int = (2 * np.pi * SIGMA**2)**1.5
    I0 = AMPLITUDE * vol_int
    
    # Dipole moment p0 = I0 / omega
    p0 = I0 / OMEGA
    
    # Radiated Power P = (omega^4 * p0^2) / (12 * pi * eps0 * c^3)
    # P = (omega^4 * (I0/omega)^2) / (12 * pi * eps0 * c^3) = (omega^2 * I0^2) / (12 * pi * eps0 * c^3)
    p_analytical = (OMEGA**2 * I0**2) / (12 * np.pi * EPS_0 * LIGHT_SPEED**3)
    
    # Setup design
    design = Design(SIZE, SIZE, SIZE, material=Material(1.0))
    
    # Setup box of monitors
    L = 0.8*µm
    cx, cy, cz = SIZE/2, SIZE/2, SIZE/2
    monitors = [
        Monitor(start=(cx-L, cy-L, cz+L), end=(cx+L, cy+L, cz+L), name='z_pos', accumulate_power=True),
        Monitor(start=(cx-L, cy-L, cz-L), end=(cx+L, cy+L, cz-L), name='z_neg', accumulate_power=True),
        Monitor(start=(cx-L, cy+L, cz-L), end=(cx+L, cy+L, cz+L), plane_normal='y', plane_position=cy+L, name='y_pos', accumulate_power=True),
        Monitor(start=(cx-L, cy-L, cz-L), end=(cx+L, cy-L, cz+L), plane_normal='y', plane_position=cy-L, name='y_neg', accumulate_power=True),
        Monitor(start=(cx+L, cy-L, cz-L), end=(cx+L, cy+L, cz+L), plane_normal='x', plane_position=cx+L, name='x_pos', accumulate_power=True),
        Monitor(start=(cx-L, cy-L, cz-L), end=(cx-L, cy+L, cz+L), plane_normal='x', plane_position=cx-L, name='x_neg', accumulate_power=True),
    ]
    
    sim = Simulation(design, devices=[source] + monitors, boundaries=[PML(edges='all', thickness=WL)], time=t, resolution=DX)
    print(f"Running 3D Dipole Benchmark (DX={DX/nm:.1f}nm)...")
    sim.run()
    
    # Integrate total power through all faces at peak of the pulse
    # We take the peak total power across the box
    total_power_history = np.zeros_like(monitors[0].power_history)
    for m in monitors:
        total_power_history += np.array(m.power_history)
    
    sim_power = np.max(total_power_history)
    
    print(f"\n--- 3D Dipole Radiation Results ---")
    print(f"Analytical Power: {p_analytical:.4e} W")
    print(f"Simulated Power:  {sim_power:.4e} W")
    print(f"Relative Error:   {abs(sim_power - p_analytical)/p_analytical*100:.2f}%")

if __name__ == "__main__":
    run_dipole_benchmark()
