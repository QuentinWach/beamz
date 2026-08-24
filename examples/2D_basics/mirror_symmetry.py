"""Solve one quarter of a centered 2D problem and expand its results."""

import os

import numpy as np

import beamz as bz

test_mode = os.environ.get("BEAMZ_DOCS_TEST") == "1"
resolution = (0.5 if test_mode else 0.1) * bz.um
steps = 8 if test_mode else 120
time = np.arange(steps, dtype=float) * 1e-16
signal = np.zeros(steps, dtype=float)
signal[1] = 1.0

source = bz.GaussianSource(
    position=(0.0, 0.0),
    width=0.4 * bz.um,
    signal=signal,
)
recorder = bz.FieldRecorder(("Ez", "Hx", "Hy"), interval=1, name="fields")

common = dict(
    domain=(4 * bz.um, 4 * bz.um),
    resolution=resolution,
    time=time,
    sources=(source,),
    monitors=(recorder,),
    boundaries=(bz.PEC(),),
)
full_simulation = bz.Simulation(**common)
reduced_simulation = bz.Simulation(**common, symmetry=(1, 1, 0))

full_shape = full_simulation.to_request().materials.shape
reduced_shape = reduced_simulation.to_request().materials.shape
assert np.prod(full_shape) == 4 * np.prod(reduced_shape)

full_results = full_simulation.run(progress=False, performance=False)
reduced_results = reduced_simulation.run(progress=False, performance=False)
expanded_results = reduced_results.symmetry_expanded
assert expanded_results.metadata.fields.grid_shape == full_shape
assert expanded_results["fields"].fields["Ez"].shape[1:] == tuple(
    size + 1 for size in full_shape
)
np.testing.assert_allclose(
    expanded_results["fields"].fields["Ez"],
    full_results["fields"].fields["Ez"],
    rtol=2e-6,
    atol=2e-12,
)

print(f"material cells: {full_shape} -> {reduced_shape}")
print(f"expanded Ez frames: {expanded_results['fields'].fields['Ez'].shape}")
