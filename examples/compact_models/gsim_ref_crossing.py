from ubcpdk import PDK, cells
from gsim import meep

# Activate the PDK & import an ebeam crossing cell
PDK.activate()
c = cells.ebeam_crossing4()
# Show the cell
c


# Create an simulation object
sim = meep.Simulation()
# Load the geometry of the cell into the simulation object
sim.geometry(component=c, z_crop="auto")
# Define the materials of the simulation
sim.materials = {"si": 3.47, "SiO2": 1.44}
# Define the source of the simulation
sim.source(port="o1", wavelength=1.55, wavelength_span=0.04, num_freqs=51)
# Place the monitors at the ports of the cell
sim.monitors = ["o1", "o2", "o3", "o4"]
# Define the PML boundaries and margin between the PML and cell
sim.domain(pml=1.0, margin=0.5)
# Plot the simulation domain
sim.plot_2d(slices="xyz")


# Set the solver resolution, and recording interval for animation
sim.solver(resolution=20, save_animation=True, verbose_interval=5.0)
# Set stop time for the simulation
sim.solver.stop_after_sources(time=45)
# Print out the configuration of the simulation
print(sim.validate_config())
#Stack validation: PASSED
#Warnings:
#  - No stack configured. Will use active PDK with defaults.


# Run the simulation on GDSFactory+ cloud
result = sim.run()
#meep-941bee26  completed  4m 40s
# Extracting results.tar.gz...
#Downloaded 252 files to /home/runner/work/gsim/gsim/nbs/sim-data-meep-941bee26


# Plot the results of the simulation
result.plot(db=True)