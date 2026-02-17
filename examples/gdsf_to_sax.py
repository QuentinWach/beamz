import sys
from pathlib import Path

import numpy as np
import sax

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beamz import *

WL0 = 1.55 * µm
N_CORE, N_CLAD = 2.0, 1.44
TIME = 18 * WL0 / LIGHT_SPEED
DX, DT = dxdt(
    WL0, n_max=N_CORE, safety_factor=0.95, points_per_wavelength=8, dims=2
)
WAVELENGTHS = np.linspace(1.50 * µm, 1.60 * µm, 41)
FREQUENCIES = LIGHT_SPEED / WAVELENGTHS
SOURCE_OFFSET = 0.35 * µm
MONITOR_OFFSET = 0.80 * µm
SPAN_FACTOR = 4.0
MIN_SPAN = 2.0 * µm


def _inward_offset(port, distance):
    """Move a point inward from a port along BeamZ direction."""
    cx, cy = port["center"]
    direction = port["direction"]
    sign = 1.0 if direction.startswith("+") else -1.0
    axis = direction[1]
    if axis == "x":
        return cx + sign * distance, cy
    return cx, cy + sign * distance


def _line_monitor_for_port(port, offset, span_factor=SPAN_FACTOR, min_span=MIN_SPAN):
    """Create a monitor line orthogonal to the local propagation axis."""
    cx, cy = _inward_offset(port, offset)
    span = max(float(min_span), float(span_factor) * float(port["width"]))
    axis = port["direction"][1]
    if axis == "x":
        start = (cx, cy - span / 2)
        end = (cx, cy + span / 2)
    else:
        start = (cx - span / 2, cy)
        end = (cx + span / 2, cy)
    return start, end, span


device_design, ports = design.io.gdsf.load(
    "mmi1x2",
    n_core=N_CORE,
    n_clad=N_CLAD,
    layer=(1, 0),
    padding=3.0,
)
grid = device_design.rasterize(resolution=DX)
time = np.arange(0, TIME, DT)
signal = ramped_cosine(
    time,
    amplitude=1.0,
    frequency=LIGHT_SPEED / WL0,
    ramp_duration=WL0 * 4 / LIGHT_SPEED,
    t_max=TIME / 2,
)

port_names = sorted(ports.keys())
s_full = {}

for source_port in port_names:
    source_meta = ports[source_port]
    src_center = _inward_offset(source_meta, SOURCE_OFFSET)
    _monitor_start, _monitor_end, source_span = _line_monitor_for_port(
        source_meta, SOURCE_OFFSET
    )
    source = ModeSource(
        grid=grid,
        center=src_center,
        width=source_span,
        wavelength=WL0,
        pol="tm",
        signal=signal,
        direction=source_meta["direction"],
    )

    monitors = []
    for port_name in port_names:
        start, end, _ = _line_monitor_for_port(ports[port_name], MONITOR_OFFSET)
        monitors.append(Monitor(start=start, end=end, name=port_name, record_fields=True))

    sim = Simulation(
        design=device_design,
        devices=[source, *monitors],
        boundaries=[PML(edges="all", thickness=1.0 * WL0)],
        time=time,
        resolution=DX,
    )
    sim.run()

    s_column = sim.get_S_matrix(
        input_ports=port_names,
        output_ports=port_names,
        source_port=source_port,
        frequencies=FREQUENCIES,
        field_component="Ez",
        reduction="mean",
        as_sax=False,
    )
    s_full.update(s_column)

s_sax = sax.sdict(s_full)
center_idx = int(np.argmin(np.abs(WAVELENGTHS - WL0)))
print(f"Computed SAX S-matrix entries: {len(s_sax)} at {len(WAVELENGTHS)} wavelengths")
for to_port, from_port in sorted(s_sax.keys()):
    value = np.asarray(s_sax[(to_port, from_port)])[center_idx]
    print(
        f"S[{to_port},{from_port}] @ {WL0 / µm:.3f}um: "
        f"|S|={np.abs(value):.3f}, phase={np.angle(value):.3f} rad"
    )
