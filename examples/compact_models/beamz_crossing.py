from ubcpdk import PDK, cells
import numpy as np
import beamz as bz
from beamz.const import *


# Set hyperparameters
WL = 1.55 * µm
N_CORE, N_CLAD = 3.47, 1.44 # SiN, SiO2
DX, DT = bz.dxdt(WL, max(N_CORE, N_CLAD), points_per_wavelength=20)
TIME = 45*WL/LIGHT_SPEED
NUM_FREQS = 51


# Activate the PDK & import an ebeam crossing cell
PDK.activate()
cell = cells.ebeam_crossing4()
# Show the cell
cell.show()
# Import the cell into BeamZ
# What info do we actually get from the cell?
design = bz.Design.from_gdsfactory(cell)
# Create padding around the cell
design.padding(edges='all', thickness=2.0*WL)
# Place straight waveguideguies at the ports of the cell
for port in ["o1", "o2", "o3", "o4"]: # can we do, c.ports.keys()?
    design += bz.Box(
        position=port,
        width=1.0*µm,
        height=1.0*µm,
        length=1.0*µm,
        material=bz.Material(1.44**2)
    )
# Show the design
design.show()


# Define the signal for the source
time_steps = np.arange(0, TIME, DT)
signal = bz.ramped_cosine(
    time_steps,
    amplitude=1.0,
    frequency=LIGHT_SPEED / WL,
    ramp_duration=WL * 20 / LIGHT_SPEED,
    t_max=TIME / 2,
)
# Define the source to sit at the input port
source = bz.ModeSource(
    grid=design.rasterize(resolution=DX),
    center=design.ports["o1"],
    signal=signal,
    wavelength=WL,
    pol="tm",
    direction="+x"
)
# Define the monitors to sit at all the ports
# They need to be configured for S-parameter extraction!
monitors = []
for port in ["o1", "o2", "o3", "o4"]:
    monitor = bz.Monitor(
        grid=design.rasterize(resolution=DX),
        center=design.ports[port],
        size=1.0*µm,
        material=bz.Material(1.44**2)
    )
    monitors.append(monitor)


# Bundle everything into the simulation object
sim = bz.Simulation(
    design=design,
    devices=[source, *monitors],
    boundaries=[bz.PML(edges='all', thickness=1.2*WL)],
    time=time_steps,
    resolution=DX
)
# Run the simulation
results = sim.run(
    animate_live="Ez", 
    animation_interval=15,
    #axis_scale=[-1*DX**2, 1*DX*+2],
    axis_scale=None,
    cmap="twilight_zero", 
    #clean_visualization=True,
    save_video="crossing.mp4",
    video_fps=40)


# Plot the results of the simulation
results.plot(db=True)