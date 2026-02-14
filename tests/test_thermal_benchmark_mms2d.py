import numpy as np

from beamz import (
    Design,
    Material,
    StaticThermalConfig,
    ThermalScenario,
    ThermalSink,
    ThermalSource,
)


def _build_manufactured_case(nx, ny, dx, t0, k, amplitude):
    x_idx = np.arange(nx, dtype=float) / max(nx - 1, 1)
    y_idx = np.arange(ny, dtype=float) / max(ny - 1, 1)
    X, Y = np.meshgrid(x_idx, y_idx)

    mode = np.sin(np.pi * X) * np.sin(np.pi * Y) + 0.35 * np.sin(
        2.0 * np.pi * X
    ) * np.sin(3.0 * np.pi * Y)
    t_ref = t0 + amplitude * mode

    lap = np.zeros_like(t_ref)
    lap[1:-1, 1:-1] = (t_ref[1:-1, 2:] - 2.0 * t_ref[1:-1, 1:-1] + t_ref[1:-1, :-2]) / (
        dx * dx
    ) + (t_ref[2:, 1:-1] - 2.0 * t_ref[1:-1, 1:-1] + t_ref[:-2, 1:-1]) / (dx * dx)
    source = -k * lap

    fixed_mask = np.zeros_like(t_ref, dtype=bool)
    fixed_mask[0, :] = True
    fixed_mask[-1, :] = True
    fixed_mask[:, 0] = True
    fixed_mask[:, -1] = True
    return t_ref, source, fixed_mask


def test_static_2d_mms_matches_reference_pattern():
    nx, ny = 61, 49
    dx = 1.0
    t0 = 300.0
    k = 12.0
    amplitude = 35.0

    t_ref, source, fixed_mask = _build_manufactured_case(
        nx=nx, ny=ny, dx=dx, t0=t0, k=k, amplitude=amplitude
    )

    design = Design(
        width=float(nx * dx),
        height=float(ny * dx),
        material=Material(permittivity=1.0, k=k, rho=1.0, cp=1.0, T0=t0),
    )
    config = StaticThermalConfig(
        max_iters=20000,
        tol=1e-9,
    )
    scenario = ThermalScenario(
        sources=[
            ThermalSource(
                region=np.ones_like(source, dtype=bool), power_density_w_m3=source
            )
        ],
        sinks=[ThermalSink(region=fixed_mask, temperature_k=t0)],
    )
    result = design.solve_thermal(
        resolution=dx,
        scenario=scenario,
        config=config,
    )

    t_num = np.asarray(result.temperature)
    err = t_num - t_ref
    rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(t_ref - t0), 1e-12))
    max_abs = float(np.max(np.abs(err)))

    assert rel_l2 < 8e-3, f"relative L2 error {rel_l2:.4e} exceeds tolerance"
    assert max_abs < 0.5, f"max absolute error {max_abs:.4f} K exceeds tolerance"
