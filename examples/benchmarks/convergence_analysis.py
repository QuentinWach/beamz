from beamz import *
import numpy as np
import matplotlib.pyplot as plt

def run_convergence_test():
    WL = 1.0*µm
    SIZE = 6.0*µm
    TIME = 10 * WL / LIGHT_SPEED
    
    resolutions = [WL/10, WL/15, WL/20, WL/30]
    errors = []
    
    # Reference high-res simulation (or analytical if available)
    # We'll use a 2D dipole for speed.
    
    print("Starting Convergence Analysis...")
    
    # We will use the power at a certain point as the metric
    results = []
    
    for res in resolutions:
        dx = res
        dt = dx / (LIGHT_SPEED * np.sqrt(2)) * 0.99
        t = np.arange(0, TIME, dt)
        
        design = Design(SIZE, SIZE, material=Material(1.0))
        signal = ramped_cosine(t, amplitude=1.0, frequency=LIGHT_SPEED/WL, ramp_duration=3*WL/LIGHT_SPEED, t_max=TIME/2)
        source = GaussianSource(position=(SIZE/2, SIZE/2), width=WL/8, signal=signal)
        
        # Monitor at a distance
        monitor = Monitor(start=(SIZE/2 + 1.5*µm, SIZE/2 - 1.0*µm), 
                         end=(SIZE/2 + 1.5*µm, SIZE/2 + 1.0*µm), 
                         accumulate_power=True)
        
        sim = Simulation(design, devices=[source, monitor], boundaries=[PML(edges='all', thickness=WL)], time=t, resolution=dx)
        print(f"  Running resolution WL/{WL/res:.1f} (dx={dx/nm:.1f}nm)...")
        sim.run()
        
        results.append(np.max(monitor.power_history))
    
    # Calculate relative error with respect to the highest resolution
    ref_val = results[-1]
    errors = [abs(val - ref_val) / ref_val for val in results[:-1]]
    dx_vals = resolutions[:-1]
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.loglog(dx_vals, errors, 'o-', label='FDTD Error')
    
    # Plot O(dx^2) slope for comparison
    slope_x = np.array([dx_vals[0], dx_vals[-1]])
    slope_y = errors[0] * (slope_x / dx_vals[0])**2
    plt.loglog(slope_x, slope_y, 'k--', label='O(dx^2) slope')
    
    plt.xlabel('Grid Resolution dx (m)')
    plt.ylabel('Relative Error')
    plt.title('FDTD Convergence Analysis (2D Dipole)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig('convergence_plot.png')
    plt.show()
    
    print("\nConvergence analysis complete. Plot saved as 'convergence_plot.png'.")

if __name__ == "__main__":
    run_convergence_test()

