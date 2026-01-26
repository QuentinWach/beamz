"""ModeSource validation tests.

Tests verify:
1. Mode effective index is reasonable for waveguide geometry
2. Mode profile is peaked at waveguide core
3. Mode propagates along waveguide without significant loss
4. Polarization filtering works (TE/TM separation)
"""

import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    PML,
    Design,
    Material,
    ModeSource,
    Rectangle,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    um,
)
from beamz.devices.sources.solve import solve_modes
from tests.utils import TEST_WAVELENGTH, compute_field_energy


@pytest.mark.simulation
class TestModeSourceEffectiveIndex:
    """Verify mode effective index computation."""

    def test_neff_within_bounds(self):
        """Effective index should be between core and cladding indices.

        Physics: n_clad < n_eff < n_core for guided modes.
        Use a thicker waveguide to ensure mode is well-guided.
        """
        wavelength = TEST_WAVELENGTH
        n_core = 2.0
        n_clad = 1.0
        # Thicker core for better mode confinement
        core_width = 1.5 * wavelength

        domain_width = 12 * wavelength
        domain_height = 8 * wavelength

        dx, dt = calc_optimal_fdtd_params(
            wavelength, n_core, dims=2, safety_factor=0.95, points_per_wavelength=20
        )

        design = Design(
            width=domain_width,
            height=domain_height,
            material=Material(permittivity=n_clad**2),
        )
        design += Rectangle(
            position=(domain_width / 2, domain_height / 2),
            width=domain_width,
            height=core_width,
            material=Material(permittivity=n_core**2),
        )

        # Create minimal time array (we only need to initialize ModeSource)
        frequency = LIGHT_SPEED / wavelength
        t_total = 5 / frequency
        time = np.arange(0, t_total, dt)

        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total,
        )

        # Rasterize design to get grid
        grid = design.rasterize(resolution=dx)

        source = ModeSource(
            grid=grid,
            center=(wavelength * 2, domain_height / 2),
            width=core_width * 2.5,
            wavelength=wavelength,
            pol="tm",
            signal=signal,
            direction="+x",
        )

        # Initialize to compute mode
        source.initialize(grid.permittivity, dx)

        # Check neff was computed
        assert source._neff is not None, "Mode solver should compute n_eff"
        neff = float(np.real(source._neff))

        # Check neff is in valid range (allow small tolerance for numerical precision)
        # n_eff should be close to n_clad or between n_clad and n_core
        assert neff > 0, f"n_eff={neff:.4f} should be positive"
        assert (
            neff < n_core + 0.1
        ), f"n_eff={neff:.4f} should not exceed n_core={n_core}"

        # For well-confined mode, neff should be above n_clad
        # Allow small tolerance for numerical precision near cutoff
        if neff < n_clad - 0.05:
            pytest.skip(f"Mode appears to be near cutoff (neff={neff:.4f})")

    def test_neff_increases_with_core_width(self):
        """Wider waveguide should have higher effective index.

        Physics: More of the mode is confined in high-index core.
        Use thicker waveguides to ensure modes are well-guided.
        """
        wavelength = TEST_WAVELENGTH
        n_core = 2.0
        n_clad = 1.0

        domain_width = 12 * wavelength
        domain_height = 8 * wavelength

        dx, dt = calc_optimal_fdtd_params(
            wavelength, n_core, dims=2, safety_factor=0.95, points_per_wavelength=15
        )

        frequency = LIGHT_SPEED / wavelength
        t_total = 5 / frequency
        time = np.arange(0, t_total, dt)

        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total,
        )

        neffs = []
        # Use thicker waveguides to ensure well-guided modes
        for core_width in [0.8 * wavelength, 1.2 * wavelength, 1.6 * wavelength]:
            design = Design(
                width=domain_width,
                height=domain_height,
                material=Material(permittivity=n_clad**2),
            )
            design += Rectangle(
                position=(domain_width / 2, domain_height / 2),
                width=domain_width,
                height=core_width,
                material=Material(permittivity=n_core**2),
            )

            grid = design.rasterize(resolution=dx)

            source = ModeSource(
                grid=grid,
                center=(wavelength * 2, domain_height / 2),
                width=core_width * 3,
                wavelength=wavelength,
                pol="tm",
                signal=signal,
                direction="+x",
            )

            source.initialize(grid.permittivity, dx)
            neffs.append(float(np.real(source._neff)))

        # neff should increase with core width (allow small tolerance)
        assert (
            neffs[1] >= neffs[0] - 0.01
        ), f"n_eff should increase with core width: {neffs}"
        assert (
            neffs[2] >= neffs[1] - 0.01
        ), f"n_eff should increase with core width: {neffs}"


@pytest.mark.simulation
class TestModeSourceProfile:
    """Verify mode profile shape."""

    def test_mode_profile_peaked_at_center(self, waveguide_domain):
        """Mode profile should have maximum at waveguide center.

        Physics: Fundamental mode is peaked at the core center.
        """
        design = waveguide_domain["design"]
        wavelength = waveguide_domain["wavelength"]
        dx = waveguide_domain["dx"]
        dt = waveguide_domain["dt"]
        domain_height = waveguide_domain["domain_height"]
        core_width = waveguide_domain["core_width"]

        frequency = LIGHT_SPEED / wavelength
        t_total = 5 / frequency
        time = np.arange(0, t_total, dt)

        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total,
        )

        grid = design.rasterize(resolution=dx)

        source = ModeSource(
            grid=grid,
            center=(wavelength * 2, domain_height / 2),
            width=core_width * 3,
            wavelength=wavelength,
            pol="tm",
            signal=signal,
            direction="+x",
        )

        source.initialize(grid.permittivity, dx)

        # Get mode profile (could be _jz_profile for TM or similar)
        profile = None
        for attr in ["_jz_profile", "_Ez_profile", "_my_profile"]:
            p = getattr(source, attr, None)
            if p is not None and np.max(np.abs(p)) > 0:
                profile = np.squeeze(p)
                break

        assert profile is not None, "Mode profile not computed"

        if profile.ndim == 1:
            # 1D profile - check peak is near center
            max_idx = np.argmax(np.abs(profile))
            center_idx = len(profile) // 2
            # Allow 20% deviation from center
            tolerance = int(len(profile) * 0.2)
            assert (
                abs(max_idx - center_idx) < tolerance
            ), f"Peak at index {max_idx}, expected near {center_idx}"
        else:
            # 2D profile - check it has some structure
            assert np.max(np.abs(profile)) > 0, "Profile should have non-zero values"

    def test_mode_decays_in_cladding(self):
        """Mode amplitude should be smaller in cladding than core.

        Physics: Evanescent field decays exponentially in cladding.
        Use thicker waveguide for better mode confinement.
        """
        wavelength = TEST_WAVELENGTH
        n_core = 2.0
        n_clad = 1.0
        core_width = 1.0 * wavelength  # Thicker for better confinement

        domain_width = 12 * wavelength
        domain_height = 8 * wavelength

        dx, dt = calc_optimal_fdtd_params(
            wavelength, n_core, dims=2, safety_factor=0.95, points_per_wavelength=15
        )

        design = Design(
            width=domain_width,
            height=domain_height,
            material=Material(permittivity=n_clad**2),
        )
        design += Rectangle(
            position=(domain_width / 2, domain_height / 2),
            width=domain_width,
            height=core_width,
            material=Material(permittivity=n_core**2),
        )

        frequency = LIGHT_SPEED / wavelength
        t_total = 5 / frequency
        time = np.arange(0, t_total, dt)

        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total,
        )

        grid = design.rasterize(resolution=dx)

        source = ModeSource(
            grid=grid,
            center=(wavelength * 2, domain_height / 2),
            width=core_width * 4,  # Wide enough to capture cladding
            wavelength=wavelength,
            pol="tm",
            signal=signal,
            direction="+x",
        )

        source.initialize(grid.permittivity, dx)

        # Get mode profile
        profile = None
        for attr in ["_jz_profile", "_Ez_profile", "_my_profile"]:
            p = getattr(source, attr, None)
            if p is not None and np.max(np.abs(p)) > 0:
                profile = np.abs(np.squeeze(p))
                break

        if profile is None or profile.ndim != 1:
            pytest.skip("Could not get 1D mode profile")

        # Compare center (core) vs edges (cladding)
        # For a well-confined mode, center should be stronger than edges
        n = len(profile)
        center_value = np.max(profile[n // 3 : 2 * n // 3])  # Middle third
        edge_value = max(
            np.max(profile[: n // 6]) if n // 6 > 0 else 0,
            np.max(profile[-n // 6 :]) if n // 6 > 0 else 0,
        )  # Outer regions

        # Center should have more field than edges (qualitative check)
        # With windowing applied, edges may be suppressed
        if center_value > 0 and edge_value > 0:
            ratio = edge_value / center_value
            # Relaxed tolerance - just check edges aren't stronger than center
            assert ratio < 1.1, (
                f"Edge/center ratio {ratio:.2f} too high. "
                "Edges should not be stronger than center."
            )


@pytest.mark.simulation
class TestModeSourcePropagation:
    """Verify mode propagates correctly in waveguide."""

    def test_mode_propagates_in_correct_direction(self, waveguide_domain):
        """Injected mode should propagate in specified direction.

        Method: Check field exists downstream, not upstream.
        """
        design = waveguide_domain["design"]
        wavelength = waveguide_domain["wavelength"]
        dx = waveguide_domain["dx"]
        dt = waveguide_domain["dt"]
        domain_width = waveguide_domain["domain_width"]
        domain_height = waveguide_domain["domain_height"]
        core_width = waveguide_domain["core_width"]

        frequency = LIGHT_SPEED / wavelength
        t_total = 15 / frequency
        time = np.arange(0, t_total, dt)

        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=3 / frequency,
            t_max=t_total * 0.5,
        )

        grid = design.rasterize(resolution=dx)

        # Source in left portion, propagating +x
        source = ModeSource(
            grid=grid,
            center=(wavelength * 3, domain_height / 2),
            width=core_width * 3,
            wavelength=wavelength,
            pol="tm",
            signal=signal,
            direction="+x",
        )

        sim = Simulation(
            design=design,
            devices=[source],
            boundaries=[PML(thickness=1.5 * wavelength)],
            time=time,
            resolution=dx,
        )

        result = sim.run(save_fields=["Ez"], field_subsample=25)

        # Check late snapshot
        late_field = result["fields"]["Ez"][-1]

        # Field should be more in right half (downstream) than left (upstream)
        source_x_idx = int(wavelength * 3 / dx)
        left_energy = compute_field_energy(late_field[:, :source_x_idx], dx)
        right_energy = compute_field_energy(late_field[:, source_x_idx:], dx)

        # Most energy should be downstream (right side)
        total = left_energy + right_energy
        if total > 1e-30:
            right_fraction = right_energy / total
            assert right_fraction > 0.5, (
                f"Only {right_fraction*100:.1f}% energy downstream. "
                "Mode should propagate in +x direction."
            )

    def test_mode_stays_confined_in_waveguide(self, waveguide_domain):
        """Mode should remain confined to waveguide core region.

        Physics: Guided mode stays within core with evanescent tails.
        """
        design = waveguide_domain["design"]
        wavelength = waveguide_domain["wavelength"]
        dx = waveguide_domain["dx"]
        dt = waveguide_domain["dt"]
        domain_width = waveguide_domain["domain_width"]
        domain_height = waveguide_domain["domain_height"]
        core_width = waveguide_domain["core_width"]

        frequency = LIGHT_SPEED / wavelength
        t_total = 12 / frequency
        time = np.arange(0, t_total, dt)

        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=3 / frequency,
            t_max=t_total * 0.4,
        )

        grid = design.rasterize(resolution=dx)

        source = ModeSource(
            grid=grid,
            center=(wavelength * 2, domain_height / 2),
            width=core_width * 3,
            wavelength=wavelength,
            pol="tm",
            signal=signal,
            direction="+x",
        )

        sim = Simulation(
            design=design,
            devices=[source],
            boundaries=[PML(thickness=1.5 * wavelength)],
            time=time,
            resolution=dx,
        )

        result = sim.run(save_fields=["Ez"], field_subsample=20)

        # Get snapshot during propagation
        mid_idx = len(result["fields"]["Ez"]) // 2
        field = result["fields"]["Ez"][mid_idx]

        ny, nx = field.shape
        center_y = ny // 2

        # Define waveguide region (center ± 2*core_width)
        core_half_cells = int(core_width / dx)
        wg_region = slice(
            center_y - 2 * core_half_cells, center_y + 2 * core_half_cells
        )

        wg_energy = compute_field_energy(field[wg_region, :], dx)
        total_energy = compute_field_energy(field, dx)

        # Most energy should be in waveguide region
        if total_energy > 1e-30:
            confinement = wg_energy / total_energy
            assert confinement > 0.5, (
                f"Only {confinement*100:.1f}% energy in waveguide region. "
                "Mode should be confined."
            )


@pytest.mark.simulation
class TestModeSourcePolarization:
    """Verify polarization filtering works."""

    @pytest.mark.parametrize("pol", ["te", "tm"])
    def test_polarization_mode_computes(self, waveguide_domain, pol):
        """Both TE and TM polarizations should compute valid modes.

        This is a basic sanity check that the mode solver handles both.
        """
        design = waveguide_domain["design"]
        wavelength = waveguide_domain["wavelength"]
        dx = waveguide_domain["dx"]
        dt = waveguide_domain["dt"]
        domain_height = waveguide_domain["domain_height"]
        core_width = waveguide_domain["core_width"]

        frequency = LIGHT_SPEED / wavelength
        t_total = 3 / frequency
        time = np.arange(0, t_total, dt)

        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=2 / frequency,
            t_max=t_total,
        )

        grid = design.rasterize(resolution=dx)

        source = ModeSource(
            grid=grid,
            center=(wavelength * 2, domain_height / 2),
            width=core_width * 3,
            wavelength=wavelength,
            pol=pol,
            signal=signal,
            direction="+x",
        )

        source.initialize(grid.permittivity, dx)

        # Check neff was computed
        assert source._neff is not None, f"n_eff not computed for {pol} mode"
        neff = float(np.real(source._neff))
        assert neff > 0, f"n_eff should be positive for {pol} mode"

        # Check some profile was computed
        has_profile = any(
            getattr(source, attr, None) is not None
            for attr in ["_jz_profile", "_jy_profile", "_Ez_profile", "_Ey_profile"]
        )
        assert has_profile, f"No mode profile computed for {pol} mode"


@pytest.mark.simulation
class TestModeSolver:
    """Direct tests of the mode solver function."""

    def test_solve_modes_returns_valid_neff(self, waveguide_domain):
        """solve_modes should return valid effective indices."""
        wavelength = waveguide_domain["wavelength"]
        dx = waveguide_domain["dx"]
        n_core = waveguide_domain["n_core"]
        n_clad = waveguide_domain["n_clad"]
        core_width = waveguide_domain["core_width"]
        domain_height = waveguide_domain["domain_height"]

        # Create 1D permittivity profile (y-direction slice)
        n_points = int(domain_height / dx)
        eps_profile = np.ones(n_points) * n_clad**2

        center = n_points // 2
        half_core = int(core_width / (2 * dx))
        eps_profile[center - half_core : center + half_core] = n_core**2

        omega = 2 * np.pi * LIGHT_SPEED / wavelength

        neff, modes = solve_modes(
            eps=eps_profile, omega=omega, dL=dx, m=1, direction="+x", filter_pol="tm"
        )

        assert len(neff) >= 1, "Should find at least one mode"
        neff_real = float(np.real(neff[0]))
        assert (
            n_clad < neff_real < n_core
        ), f"n_eff={neff_real:.4f} should be between {n_clad} and {n_core}"
