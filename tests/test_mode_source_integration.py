import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    PML,
    Design,
    Material,
    ModeSource,
    Monitor,
    PortSpec,
    Rectangle,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
)
from tests.test_mode_source import (
    TestModeSourceDirectionality3D,
    TestModeSourceEffectiveIndex,
    TestModeSourcePolarization,
    TestModeSourceProfile,
    TestModeSourcePropagation,
)
from tests.utils import TEST_WAVELENGTH

pytestmark = [pytest.mark.integration, pytest.mark.simulation]

# Disabled from regular collection: these wrapper assignments re-enable large,
# expensive FDTD integration runs from tests/test_mode_source.py. Keep the
# wrapper in place for manual re-enabling when needed.
# TestModeSourceEffectiveIndex.__test__ = True
# TestModeSourceProfile.__test__ = True
# TestModeSourcePropagation.__test__ = True
# TestModeSourcePolarization.__test__ = True
# TestModeSourceDirectionality3D.__test__ = True
TestModeSourceEffectiveIndex.__test__ = False
TestModeSourceProfile.__test__ = False
TestModeSourcePropagation.__test__ = False
TestModeSourcePolarization.__test__ = False
TestModeSourceDirectionality3D.__test__ = False


@pytest.mark.compiled
def test_centered_3d_mode_source_raw_reflection_is_below_minus_40_db():
    """A centered 3D straight-guide ModeSource should be nearly one-way."""
    wavelength = TEST_WAVELENGTH
    n_core = 2.0
    n_clad = 1.0
    guide_width = 0.6 * wavelength
    span = guide_width * 2.5
    long_span = 6.0 * wavelength
    transverse_span = 2.4 * wavelength

    design = Design(
        width=long_span,
        height=transverse_span,
        depth=transverse_span,
        material=Material(permittivity=n_clad**2),
    )
    design += Rectangle(
        position=(
            0.0,
            transverse_span / 2 - guide_width / 2,
            transverse_span / 2 - guide_width / 2,
        ),
        width=long_span,
        height=guide_width,
        depth=guide_width,
        material=Material(permittivity=n_core**2),
    )

    dx, dt = calc_optimal_fdtd_params(
        wavelength,
        n_core,
        dims=3,
        safety_factor=0.9,
        points_per_wavelength=8,
        width=design.width,
        height=design.height,
        depth=design.depth,
    )
    freq = LIGHT_SPEED / wavelength
    t_total = 8.0 / freq
    time = np.arange(0.0, t_total, dt)
    signal = ramped_cosine(
        time,
        amplitude=1.0,
        frequency=freq,
        ramp_duration=1.0 / freq,
        t_max=t_total,
    )

    center = (design.width / 2, design.height / 2, design.depth / 2)
    grid = design.rasterize(resolution=dx)
    source = ModeSource(
        grid=grid,
        center=(1.2 * wavelength, center[1], center[2]),
        width=span,
        height=span,
        wavelength=wavelength,
        pol="te",
        signal=signal,
        direction="+x",
    )
    monitor = Monitor(
        start=(1.8 * wavelength, center[1] - span / 2, center[2] - span / 2),
        end=(1.8 * wavelength, center[1] + span / 2, center[2] + span / 2),
        name="o1",
        record_fields=False,
        dft_enabled=True,
        dft_frequencies=np.array([freq]),
        dft_components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        dft_window="none",
        dft_record_every_step=True,
    )

    sim = Simulation(
        design=design,
        sources=[source],
        monitors=[monitor],
        boundaries=[PML(thickness=0.8 * wavelength)],
        time=time,
        resolution=dx,
    )
    sim.run_compiled(progress=False)

    waves = sim.extract_port_waves_dft(
        ports=[
            PortSpec(
                name="o1",
                monitor_name="o1",
                direction="+x",
                polarization="te",
                incident_wave="auto",
                scattered_wave="auto",
            )
        ],
        frequencies=np.array([freq]),
    )
    a_plus = complex(waves["o1"]["a_plus"][0])
    a_minus = complex(waves["o1"]["a_minus"][0])
    major = max(abs(a_plus), abs(a_minus))
    minor = min(abs(a_plus), abs(a_minus))
    reflection_db = 20.0 * np.log10(max(minor / max(major, 1e-18), 1e-12))
    dominance_db = 20.0 * np.log10(max(abs(a_minus), 1e-18) / max(abs(a_plus), 1e-18))

    assert dominance_db >= 35.0, (
        "Expected the +x source-port decomposition to identify the minus branch "
        f"as incident, got dominance={dominance_db:.2f} dB "
        f"(a_plus={a_plus}, a_minus={a_minus})."
    )
    assert reflection_db <= -40.0, (
        "Expected low raw source-port reflection in a centered 3D straight guide, "
        f"got {reflection_db:.2f} dB "
        f"(a_plus={a_plus}, a_minus={a_minus})."
    )
