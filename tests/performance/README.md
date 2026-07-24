# Performance evidence

Performance tests record rasterization, tracing, compilation, warm execution,
memory, result extraction, and monitor overhead separately. Hard regression
gates belong on controlled hardware; shared runners should publish measurements
without making noisy pass/fail claims.

`benchmark_schema.py` defines the portable record and comparison policy. Every
record includes the BeamZ/JAX/Python versions, processor or accelerator,
precision, grid, timestep count, boundaries, sources, monitors, compilation
time, multiple warm samples, and peak memory. A comparison deliberately returns
an ungated result unless the caller declares that it ran on controlled
hardware.
