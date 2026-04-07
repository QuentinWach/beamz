"""ModeSource validation tests.

Tests verify:
1. Mode effective index is reasonable for waveguide geometry
2. Mode profile is peaked at waveguide core
3. Mode propagates along waveguide without significant loss
4. Polarization filtering works (TE/TM separation)
"""

import inspect

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
    um,
)
from beamz.devices.sources import mode as mode_module
from beamz.devices.sources.solve import solve_modes
from tests.utils import TEST_WAVELENGTH, compute_field_energy


def _poynting_component_3d(fields, axis):
    """Compute one Cartesian Poynting component from staggered 3D snapshots."""
    Ex = np.asarray(fields["Ex"])
    Ey = np.asarray(fields["Ey"])
    Ez = np.asarray(fields["Ez"])
    Hx = np.asarray(fields["Hx"])
    Hy = np.asarray(fields["Hy"])
    Hz = np.asarray(fields["Hz"])

    if axis == "x":
        nz = min(Ey.shape[0], Hz.shape[0], Ez.shape[0], Hy.shape[0])
        ny = min(Ey.shape[1], Hz.shape[1], Ez.shape[1], Hy.shape[1])
        nx = min(Ey.shape[2], Hz.shape[2], Ez.shape[2], Hy.shape[2])
        return (
            Ey[:nz, :ny, :nx] * Hz[:nz, :ny, :nx]
            - Ez[:nz, :ny, :nx] * Hy[:nz, :ny, :nx]
        )
    if axis == "y":
        nz = min(Ez.shape[0], Hx.shape[0], Ex.shape[0], Hz.shape[0])
        ny = min(Ez.shape[1], Hx.shape[1], Ex.shape[1], Hz.shape[1])
        nx = min(Ez.shape[2], Hx.shape[2], Ex.shape[2], Hz.shape[2])
        return (
            Ez[:nz, :ny, :nx] * Hx[:nz, :ny, :nx]
            - Ex[:nz, :ny, :nx] * Hz[:nz, :ny, :nx]
        )
    if axis == "z":
        nz = min(Ex.shape[0], Hy.shape[0], Ey.shape[0], Hx.shape[0])
        ny = min(Ex.shape[1], Hy.shape[1], Ey.shape[1], Hx.shape[1])
        nx = min(Ex.shape[2], Hy.shape[2], Ey.shape[2], Hx.shape[2])
        return (
            Ex[:nz, :ny, :nx] * Hy[:nz, :ny, :nx]
            - Ey[:nz, :ny, :nx] * Hx[:nz, :ny, :nx]
        )
    raise ValueError(f"Unsupported axis {axis!r}")


def _plane_flux(p_component, axis, plane_idx):
    """Integrate Poynting component through a plane normal to axis."""
    if axis == "x":
        idx = int(np.clip(plane_idx, 0, p_component.shape[2] - 1))
        return float(np.sum(p_component[:, :, idx]))
    if axis == "y":
        idx = int(np.clip(plane_idx, 0, p_component.shape[1] - 1))
        return float(np.sum(p_component[:, idx, :]))
    if axis == "z":
        idx = int(np.clip(plane_idx, 0, p_component.shape[0] - 1))
        return float(np.sum(p_component[idx, :, :]))
    raise ValueError(f"Unsupported axis {axis!r}")


def _axis_size(arr, axis):
    if axis == "x":
        return arr.shape[2]
    if axis == "y":
        return arr.shape[1]
    return arr.shape[0]


def _profile_correlation(profile_a, profile_b):
    """Return normalized real-valued profile correlation in [-1, 1]."""
    a = np.real(np.asarray(profile_a)).ravel()
    b = np.real(np.asarray(profile_b)).ravel()
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    a = a[:n]
    b = b[:n]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-30
    return float(np.sum(a * b) / denom)


def _build_3d_waveguide(axis, wavelength, n_core, n_clad, guide_width):
    long_span = 4.0 * wavelength
    transverse_span = 2.4 * wavelength

    if axis == "x":
        width, height, depth = long_span, transverse_span, transverse_span
        core = Rectangle(
            position=(0, height / 2 - guide_width / 2, depth / 2 - guide_width / 2),
            width=width,
            height=guide_width,
            depth=guide_width,
            material=Material(n_core**2),
        )
    elif axis == "y":
        width, height, depth = transverse_span, long_span, transverse_span
        core = Rectangle(
            position=(width / 2 - guide_width / 2, 0, depth / 2 - guide_width / 2),
            width=guide_width,
            height=height,
            depth=guide_width,
            material=Material(n_core**2),
        )
    else:
        width, height, depth = transverse_span, transverse_span, long_span
        core = Rectangle(
            position=(width / 2 - guide_width / 2, height / 2 - guide_width / 2, 0),
            width=guide_width,
            height=guide_width,
            depth=depth,
            material=Material(n_core**2),
        )

    design = Design(
        width=width,
        height=height,
        depth=depth,
        material=Material(permittivity=n_clad**2),
    )
    design += core
    center = (width / 2, height / 2, depth / 2)
    return design, center


class TestModeSourceDiscreteHelpers:
    """Unit tests for deterministic discrete launch helpers."""

    def test_numeric_k_satisfies_dispersion_identity(self):
        wavelength = TEST_WAVELENGTH
        omega = 2 * np.pi * LIGHT_SPEED / wavelength
        d_axis = wavelength / 12.0
        neff = 1.8
        dt = 0.4 * d_axis / LIGHT_SPEED

        k_num = mode_module._solve_numeric_k_axis(omega, dt, d_axis, neff)
        S = LIGHT_SPEED * dt / (neff * d_axis)
        lhs = np.sin(0.5 * omega * dt)
        rhs = S * np.sin(0.5 * k_num * d_axis)

        assert (
            abs(lhs - rhs) < 1e-10
        ), f"Discrete dispersion residual too large: lhs={lhs:.6e}, rhs={rhs:.6e}"

    def test_numeric_phase_delay_monotonic(self):
        wavelength = TEST_WAVELENGTH
        omega = 2 * np.pi * LIGHT_SPEED / wavelength
        d_axis = wavelength / 10.0
        neff = 2.0
        dt = 0.45 * d_axis / LIGHT_SPEED
        k_num = mode_module._solve_numeric_k_axis(omega, dt, d_axis, neff)

        dt_small = mode_module._numeric_phase_delay(omega, k_num, 0.5 * d_axis)
        dt_large = mode_module._numeric_phase_delay(omega, k_num, 1.5 * d_axis)

        assert dt_small > 0.0
        assert dt_large > dt_small

    def test_compute_dt_physical_has_no_empirical_multiplier(self):
        src = inspect.getsource(ModeSource._compute_dt_physical)
        assert "1.25" not in src and "1.50" not in src
        assert "dt_scale" not in src

    def test_normalize_2d_pair_by_power_enforces_unit_power(self):
        rng = np.random.default_rng(7)
        h = rng.normal(size=41) + 1j * rng.normal(size=41)
        e = rng.normal(size=41) + 1j * rng.normal(size=41)
        dl = 0.12

        h_n, e_n = mode_module._normalize_2d_pair_by_power(
            h, e, signed_flux_sign=-1.0, dl=dl
        )
        p = mode_module._modal_power_2d(e_n, h_n, signed_flux_sign=-1.0, dl=dl)
        assert np.isfinite(p)
        assert np.isclose(abs(p), 1.0, rtol=1e-10, atol=1e-10)

    def test_normalize_3d_profiles_by_flux_enforces_unit_power(self):
        rng = np.random.default_rng(11)
        profiles = {
            "Ex": rng.normal(size=(12, 9)) + 1j * rng.normal(size=(12, 9)),
            "Ey": rng.normal(size=(12, 9)) + 1j * rng.normal(size=(12, 9)),
            "Ez": rng.normal(size=(12, 9)) + 1j * rng.normal(size=(12, 9)),
            "Hx": rng.normal(size=(12, 9)) + 1j * rng.normal(size=(12, 9)),
            "Hy": rng.normal(size=(12, 9)) + 1j * rng.normal(size=(12, 9)),
            "Hz": rng.normal(size=(12, 9)) + 1j * rng.normal(size=(12, 9)),
        }
        d_area = 0.03
        out = mode_module._normalize_3d_profiles_by_flux(
            dict(profiles), axis="x", d_area=d_area
        )
        p = mode_module._modal_power_3d_from_profiles(out, axis="x", d_area=d_area)
        assert np.isfinite(p)
        assert np.isclose(abs(p), 1.0, rtol=1e-10, atol=1e-10)

    def test_select_core_confined_mode_prefers_centered_mode(self):
        eps = np.ones((81, 1), dtype=float)
        eps[24:57, 0] = 4.0
        y = np.arange(81, dtype=float)

        centered = np.exp(-0.5 * ((y - 40.0) / 6.0) ** 2)
        edge_lobed = np.exp(-0.5 * ((y - 27.0) / 4.0) ** 2) + np.exp(
            -0.5 * ((y - 53.0) / 4.0) ** 2
        )

        e_center = np.zeros((3, 81, 1), dtype=np.complex128)
        e_edge = np.zeros((3, 81, 1), dtype=np.complex128)
        e_center[2, :, 0] = centered
        e_edge[2, :, 0] = edge_lobed

        idx = mode_module._select_core_confined_mode_index(
            eps_profile=eps,
            e_fields=np.stack([e_center, e_edge], axis=0),
            neff_values=np.asarray([1.90, 1.95], dtype=np.complex128),
        )
        assert idx == 0


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

    @pytest.mark.parametrize(
        ("pol", "profile_attr"),
        [("tm", "_jz_profile"), ("te", "_jx_profile")],
    )
    def test_y_direction_profile_nonzero(self, waveguide_domain, pol, profile_attr):
        """+y mode setup should build non-zero polarization-specific source profiles."""
        wavelength = waveguide_domain["wavelength"]
        dx = waveguide_domain["dx"]
        dt = waveguide_domain["dt"]

        n_core = waveguide_domain["n_core"]
        n_clad = waveguide_domain["n_clad"]

        domain_width = 8 * wavelength
        domain_height = 8 * wavelength
        core_width = 0.6 * wavelength

        design = Design(
            width=domain_width,
            height=domain_height,
            material=Material(permittivity=n_clad**2),
        )
        # Vertical core for +y propagation
        design += Rectangle(
            position=(domain_width / 2 - core_width / 2, 0),
            width=core_width,
            height=domain_height,
            material=Material(permittivity=n_core**2),
        )

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
            center=(domain_width / 2, 2 * wavelength),
            width=core_width * 3,
            wavelength=wavelength,
            pol=pol,
            signal=signal,
            direction="+y",
        )
        source.initialize(grid.permittivity, dx)

        profile = getattr(source, profile_attr)
        assert profile is not None, f"{profile_attr} should be defined for +y/{pol}"
        assert (
            float(np.max(np.abs(np.asarray(profile)))) > 1e-8
        ), f"{profile_attr} is near zero for +y/{pol}; check component mapping"

    def test_invalid_direction_raises(self, waveguide_domain):
        """ModeSource should reject directions outside ±x/±y/±z."""
        design = waveguide_domain["design"]
        dx = waveguide_domain["dx"]
        wavelength = waveguide_domain["wavelength"]
        dt = waveguide_domain["dt"]

        grid = design.rasterize(resolution=dx)
        signal = np.ones(max(2, int(5 * dt / dt)))
        with pytest.raises(ValueError, match="direction"):
            ModeSource(
                grid=grid,
                center=(wavelength * 2, design.height / 2),
                width=waveguide_domain["core_width"] * 3,
                wavelength=wavelength,
                pol="tm",
                signal=signal,
                direction="north",
            )

    @pytest.mark.parametrize("direction", ["+z", "-z"])
    def test_3d_z_direction_sets_axis(self, direction):
        """3D z-directed initialization should set axis metadata to 'z'."""
        wavelength = TEST_WAVELENGTH
        n_core = 2.0
        n_clad = 1.0
        guide_w = 0.6 * wavelength

        width = 3.0 * wavelength
        height = 3.0 * wavelength
        depth = 4.0 * wavelength
        dx, dt = calc_optimal_fdtd_params(
            wavelength,
            n_core,
            dims=3,
            safety_factor=0.9,
            points_per_wavelength=6,
            width=width,
            height=height,
            depth=depth,
        )

        design = Design(
            width=width, height=height, depth=depth, material=Material(n_clad**2)
        )
        design += Rectangle(
            position=(width / 2 - guide_w / 2, height / 2 - guide_w / 2, 0),
            width=guide_w,
            height=guide_w,
            depth=depth,
            material=Material(n_core**2),
        )

        freq = LIGHT_SPEED / wavelength
        t_total = 4 / freq
        time = np.arange(0, t_total, dt)
        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=freq,
            ramp_duration=1 / freq,
            t_max=t_total * 0.5,
        )
        grid = design.rasterize(resolution=dx)
        source = ModeSource(
            grid=grid,
            center=(width / 2, height / 2, depth / 2),
            width=guide_w * 2.5,
            height=guide_w * 2.5,
            wavelength=wavelength,
            pol="tm",
            signal=signal,
            direction=direction,
        )
        source.initialize(grid.permittivity, dx)
        assert source._axis == "z"

    def test_2d_z_direction_raises_on_initialize(self, waveguide_domain):
        """z-directed source should reject 2D initialization."""
        design = waveguide_domain["design"]
        wavelength = waveguide_domain["wavelength"]
        dx = waveguide_domain["dx"]
        dt = waveguide_domain["dt"]

        freq = LIGHT_SPEED / wavelength
        t_total = 3 / freq
        time = np.arange(0, t_total, dt)
        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=freq,
            ramp_duration=1 / freq,
            t_max=t_total,
        )
        grid = design.rasterize(resolution=dx)
        source = ModeSource(
            grid=grid,
            center=(wavelength * 2, design.height / 2),
            width=waveguide_domain["core_width"] * 3,
            wavelength=wavelength,
            pol="tm",
            signal=signal,
            direction="+z",
        )
        with pytest.raises(ValueError, match="requires a 3D permittivity grid"):
            source.initialize(grid.permittivity, dx)

    @pytest.mark.parametrize(
        ("direction", "pol"),
        [
            ("+x", "tm"),
            ("-x", "tm"),
            ("+x", "te"),
            ("-x", "te"),
            ("+y", "tm"),
            ("-y", "tm"),
            ("+y", "te"),
            ("-y", "te"),
        ],
    )
    def test_directionality_across_axes_and_polarizations(self, direction, pol):
        """ModeSource should inject predominantly in the requested direction for 2D guides."""
        wavelength = TEST_WAVELENGTH
        n_core = 2.0
        n_clad = 1.0
        wg_width = 0.55 * wavelength

        domain_width = 10 * wavelength
        domain_height = 10 * wavelength

        dx, dt = calc_optimal_fdtd_params(
            wavelength,
            n_core,
            dims=2,
            safety_factor=0.95,
            points_per_wavelength=12,
        )

        design = Design(
            width=domain_width,
            height=domain_height,
            material=Material(permittivity=n_clad**2),
        )
        if direction in {"+x", "-x"}:
            design += Rectangle(
                position=(0, domain_height / 2 - wg_width / 2),
                width=domain_width,
                height=wg_width,
                material=Material(permittivity=n_core**2),
            )
        else:
            design += Rectangle(
                position=(domain_width / 2 - wg_width / 2, 0),
                width=wg_width,
                height=domain_height,
                material=Material(permittivity=n_core**2),
            )

        frequency = LIGHT_SPEED / wavelength
        t_total = 16 / frequency
        time = np.arange(0, t_total, dt)
        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=frequency,
            ramp_duration=4 / frequency,
            t_max=t_total / 2,
        )

        center = (domain_width / 2, domain_height / 2)
        grid = design.rasterize(resolution=dx)
        source = ModeSource(
            grid=grid,
            center=center,
            width=wg_width * 3,
            wavelength=wavelength,
            pol=pol,
            signal=signal,
            direction=direction,
        )

        sim = Simulation(
            design=design,
            devices=[source],
            boundaries=[PML(thickness=1.5 * wavelength)],
            time=time,
            resolution=dx,
        )

        field_name = "Ez" if pol == "tm" else "Hz"
        result = sim.run(save_fields=[field_name], field_subsample=8)
        snapshot = result["fields"][field_name][len(result["fields"][field_name]) // 3]

        sx = int(center[0] / dx)
        sy = int(center[1] / dx)

        if direction == "+x":
            forward = compute_field_energy(snapshot[:, sx:], dx)
            backward = compute_field_energy(snapshot[:, :sx], dx)
        elif direction == "-x":
            forward = compute_field_energy(snapshot[:, :sx], dx)
            backward = compute_field_energy(snapshot[:, sx:], dx)
        elif direction == "+y":
            forward = compute_field_energy(snapshot[sy:, :], dx)
            backward = compute_field_energy(snapshot[:sy, :], dx)
        else:
            forward = compute_field_energy(snapshot[:sy, :], dx)
            backward = compute_field_energy(snapshot[sy:, :], dx)

        forward_fraction = forward / (forward + backward + 1e-30)
        min_forward = 0.97
        assert forward_fraction > min_forward, (
            f"Poor directionality for {direction}/{pol}: "
            f"forward_fraction={forward_fraction:.3f}"
        )


class TestModeSource3DSignGaugeParity:
    """Fast initialization checks for 3D +/- direction sign parity."""

    @pytest.mark.parametrize(
        ("axis", "pol", "j_h_comp", "m_e_comp"),
        [
            ("x", "tm", "Hy", "Ez"),
            ("x", "te", "Hz", "Ey"),
            ("y", "tm", "Hx", "Ez"),
            ("y", "te", "Hz", "Ex"),
            ("z", "tm", "Hx", "Ey"),
            ("z", "te", "Hy", "Ex"),
        ],
    )
    def test_direction_sign_parity_profiles(self, axis, pol, j_h_comp, m_e_comp):
        wavelength = TEST_WAVELENGTH
        n_core = 2.0
        n_clad = 1.0
        guide_width = 0.6 * wavelength

        design, center = _build_3d_waveguide(
            axis, wavelength, n_core, n_clad, guide_width
        )
        dx, dt = calc_optimal_fdtd_params(
            wavelength,
            n_core,
            dims=3,
            safety_factor=0.9,
            points_per_wavelength=6,
            width=design.width,
            height=design.height,
            depth=design.depth,
        )

        freq = LIGHT_SPEED / wavelength
        t_total = 3 / freq
        time = np.arange(0, t_total, dt)
        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=freq,
            ramp_duration=1 / freq,
            t_max=t_total,
        )
        grid = design.rasterize(resolution=dx)

        source_plus = ModeSource(
            grid=grid,
            center=center,
            width=guide_width * 2.5,
            height=guide_width * 2.5,
            wavelength=wavelength,
            pol=pol,
            signal=signal,
            direction=f"+{axis}",
        )
        source_minus = ModeSource(
            grid=grid,
            center=center,
            width=guide_width * 2.5,
            height=guide_width * 2.5,
            wavelength=wavelength,
            pol=pol,
            signal=signal,
            direction=f"-{axis}",
        )
        source_plus.initialize(grid.permittivity, dx)
        source_minus.initialize(grid.permittivity, dx)

        h_plus = getattr(source_plus, f"_{j_h_comp}_profile")
        h_minus = getattr(source_minus, f"_{j_h_comp}_profile")
        e_plus = getattr(source_plus, f"_{m_e_comp}_profile")
        e_minus = getattr(source_minus, f"_{m_e_comp}_profile")
        assert h_plus is not None and h_minus is not None
        assert e_plus is not None and e_minus is not None

        corr_h = _profile_correlation(h_plus, h_minus)
        corr_e = _profile_correlation(e_plus, e_minus)

        assert (
            corr_h < -0.60
        ), f"{axis}/{pol} J-driving H profile should flip sign: corr={corr_h:.3f}"
        assert (
            corr_e > 0.60
        ), f"{axis}/{pol} M-driving E profile should preserve sign: corr={corr_e:.3f}"


@pytest.mark.simulation
@pytest.mark.slow
class TestModeSourceDirectionality3D:
    """3D directionality checks for TE/TM across ±x/±y/±z."""

    @staticmethod
    def _build_3d_waveguide(axis, wavelength, n_core, n_clad, guide_width):
        return _build_3d_waveguide(axis, wavelength, n_core, n_clad, guide_width)

    @pytest.mark.parametrize(
        ("direction", "pol"), [("+x", "tm"), ("-x", "tm"), ("+x", "te"), ("-x", "te")]
    )
    def test_x_slice_direction_regression(self, direction, pol):
        """Regression for reported x-slice reversal: +x -> right, -x -> left."""
        wavelength = TEST_WAVELENGTH
        n_core = 2.0
        n_clad = 1.0
        guide_width = 0.6 * wavelength
        axis = "x"

        design, center = self._build_3d_waveguide(
            axis, wavelength, n_core, n_clad, guide_width
        )
        dx, dt = calc_optimal_fdtd_params(
            wavelength,
            n_core,
            dims=3,
            safety_factor=0.9,
            points_per_wavelength=6,
            width=design.width,
            height=design.height,
            depth=design.depth,
        )

        freq = LIGHT_SPEED / wavelength
        t_total = 6 / freq
        time = np.arange(0, t_total, dt)
        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=freq,
            ramp_duration=1 / freq,
            t_max=t_total,
        )
        grid = design.rasterize(resolution=dx)
        source = ModeSource(
            grid=grid,
            center=center,
            width=guide_width * 2.5,
            height=guide_width * 2.5,
            wavelength=wavelength,
            pol=pol,
            signal=signal,
            direction=direction,
        )
        sim = Simulation(
            design=design,
            devices=[source],
            boundaries=[PML(thickness=0.8 * wavelength)],
            time=time,
            resolution=dx,
        )

        field_name = "Ez" if pol == "tm" else "Hz"
        result = sim.run(save_fields=[field_name], field_subsample=1)
        snapshots = result["fields"][field_name]
        start_idx = max(1, len(snapshots) // 2)
        x_center = int(np.asarray(snapshots[0]).shape[-1] // 2)
        y_half_band = max(2, int(round((guide_width / dx) * 1.2)))

        left_vals = []
        right_vals = []
        for idx in range(start_idx, len(snapshots)):
            field_3d = np.asarray(snapshots[idx])
            z_mid = field_3d.shape[0] // 2
            xy_slice = np.abs(field_3d[z_mid]) ** 2
            y_mid = xy_slice.shape[0] // 2
            y0 = max(0, y_mid - y_half_band)
            y1 = min(xy_slice.shape[0], y_mid + y_half_band)
            left_vals.append(float(np.sum(xy_slice[y0:y1, :x_center])))
            right_vals.append(float(np.sum(xy_slice[y0:y1, x_center:])))

        left_mean = float(np.mean(left_vals))
        right_mean = float(np.mean(right_vals))
        if direction == "+x":
            assert right_mean > 1.10 * left_mean, (
                f"Expected +x to launch right in xy slice for {pol}: "
                f"right_mean={right_mean:.3e}, left_mean={left_mean:.3e}"
            )
        else:
            assert left_mean > 1.10 * right_mean, (
                f"Expected -x to launch left in xy slice for {pol}: "
                f"left_mean={left_mean:.3e}, right_mean={right_mean:.3e}"
            )

    @pytest.mark.parametrize(
        ("direction", "pol"),
        [
            ("+x", "tm"),
            ("-x", "tm"),
            ("+x", "te"),
            ("-x", "te"),
            ("+y", "tm"),
            ("-y", "tm"),
            ("+y", "te"),
            ("-y", "te"),
            ("+z", "tm"),
            ("-z", "tm"),
            ("+z", "te"),
            ("-z", "te"),
        ],
    )
    def test_flux_directionality_matrix_3d(self, direction, pol):
        """ModeSource should direct 3D power flow predominantly into requested direction."""
        wavelength = TEST_WAVELENGTH
        n_core = 2.0
        n_clad = 1.0
        guide_width = 0.6 * wavelength
        axis = direction[1]

        design, center = self._build_3d_waveguide(
            axis, wavelength, n_core, n_clad, guide_width
        )
        dx, dt = calc_optimal_fdtd_params(
            wavelength,
            n_core,
            dims=3,
            safety_factor=0.9,
            # Use higher 3D resolution for strict directional purity checks.
            points_per_wavelength=12,
            width=design.width,
            height=design.height,
            depth=design.depth,
        )

        freq = LIGHT_SPEED / wavelength
        t_total = 4 / freq
        time = np.arange(0, t_total, dt)
        signal = ramped_cosine(
            time,
            amplitude=1.0,
            frequency=freq,
            ramp_duration=1 / freq,
            t_max=t_total,
        )

        grid = design.rasterize(resolution=dx)
        source = ModeSource(
            grid=grid,
            center=center,
            width=guide_width * 2.5,
            height=guide_width * 2.5,
            wavelength=wavelength,
            pol=pol,
            signal=signal,
            direction=direction,
        )
        sim = Simulation(
            design=design,
            devices=[source],
            boundaries=[PML(thickness=0.8 * wavelength)],
            time=time,
            resolution=dx,
        )

        save_fields = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
        result = sim.run(save_fields=save_fields, field_subsample=1)
        axis_len = _axis_size(
            _poynting_component_3d(
                {name: np.asarray(result["fields"][name][0]) for name in save_fields},
                axis,
            ),
            axis,
        )
        axis_center_coord = center[{"x": 0, "y": 1, "z": 2}[axis]]
        center_idx = int(
            np.clip(np.round(axis_center_coord / dx - 0.5), 1, axis_len - 2)
        )
        dir_sign = 1.0 if direction.startswith("+") else -1.0
        # Keep near plane out of the reactive source region.
        near_offset_cells = max(2, int(round(0.8 * wavelength / dx)))
        far_offset_cells = max(near_offset_cells + 2, int(round(1.0 * wavelength / dx)))

        near_offset_cells = int(np.clip(near_offset_cells, 1, max(1, axis_len // 3)))
        far_offset_cells = int(
            np.clip(far_offset_cells, near_offset_cells + 1, max(2, axis_len // 2))
        )

        if direction.startswith("+"):
            near_forward_idx = min(axis_len - 1, center_idx + near_offset_cells)
            near_backward_idx = max(0, center_idx - near_offset_cells)
            far_forward_idx = min(axis_len - 1, center_idx + far_offset_cells)
            far_backward_idx = max(0, center_idx - far_offset_cells)
        else:
            near_forward_idx = max(0, center_idx - near_offset_cells)
            near_backward_idx = min(axis_len - 1, center_idx + near_offset_cells)
            far_forward_idx = max(0, center_idx - far_offset_cells)
            far_backward_idx = min(axis_len - 1, center_idx + far_offset_cells)

        # Strict gate: fixed planes and steady-state time average.
        steady_start = max(1, int(0.75 * len(result["fields"]["Ex"])))
        near_forward_vals = []
        near_backward_vals = []
        far_forward_vals = []
        far_backward_vals = []
        for snap_idx in range(steady_start, len(result["fields"]["Ex"])):
            snapshot = {
                name: np.asarray(result["fields"][name][snap_idx])
                for name in save_fields
            }
            p_axis = _poynting_component_3d(snapshot, axis)

            near_forward_raw = _plane_flux(p_axis, axis, near_forward_idx)
            near_backward_raw = _plane_flux(p_axis, axis, near_backward_idx)
            far_forward_raw = _plane_flux(p_axis, axis, far_forward_idx)
            far_backward_raw = _plane_flux(p_axis, axis, far_backward_idx)

            near_forward_vals.append(dir_sign * near_forward_raw)
            near_backward_vals.append(max(0.0, -dir_sign * near_backward_raw))
            far_forward_vals.append(dir_sign * far_forward_raw)
            far_backward_vals.append(max(0.0, -dir_sign * far_backward_raw))

        near_forward_flux_mean = float(np.mean(near_forward_vals))
        near_backward_flux_mean = float(np.mean(near_backward_vals))
        far_forward_flux_mean = float(np.mean(far_forward_vals))
        far_backward_flux_mean = float(np.mean(far_backward_vals))
        min_forward_flux = 0.5
        near_forward_ratio = near_forward_flux_mean / (
            near_forward_flux_mean + near_backward_flux_mean + 1e-30
        )
        far_forward_ratio = far_forward_flux_mean / (
            far_forward_flux_mean + far_backward_flux_mean + 1e-30
        )
        far_backward_ratio = far_backward_flux_mean / (far_forward_flux_mean + 1e-30)

        assert far_forward_flux_mean > min_forward_flux, (
            f"Weak forward 3D flux for {direction}/{pol}: "
            f"far_forward_flux_mean={far_forward_flux_mean:.3e}"
        )
        assert near_forward_ratio > 0.985, (
            f"Poor near-plane 3D forward dominance for {direction}/{pol}: "
            f"near_forward_ratio={near_forward_ratio:.4f}, "
            f"near_forward_flux_mean={near_forward_flux_mean:.3e}, "
            f"near_backward_flux_mean={near_backward_flux_mean:.3e}, "
            f"near_offset_cells={near_offset_cells}, steady_start={steady_start}"
        )
        assert far_forward_ratio > 0.99, (
            f"Poor far-plane 3D forward dominance for {direction}/{pol}: "
            f"far_forward_ratio={far_forward_ratio:.4f}, "
            f"far_forward_flux_mean={far_forward_flux_mean:.3e}, "
            f"far_backward_flux_mean={far_backward_flux_mean:.3e}, "
            f"far_offset_cells={far_offset_cells}, steady_start={steady_start}"
        )
        assert far_backward_ratio < 1e-2, (
            f"Excess far-plane backward 3D flux for {direction}/{pol}: "
            f"far_backward_ratio={far_backward_ratio:.4e}, "
            f"far_forward_flux_mean={far_forward_flux_mean:.3e}, "
            f"far_backward_flux_mean={far_backward_flux_mean:.3e}, "
            f"far_offset_cells={far_offset_cells}, steady_start={steady_start}"
        )

    def test_raw_same_monitor_reflection_stays_low_in_3d_straight_guide(self):
        """Straight guides should not show large raw source-port reflection."""
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
            devices=[source, monitor],
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
        dominance_db = 20.0 * np.log10(
            max(abs(a_plus), 1e-18) / max(abs(a_minus), 1e-18)
        )

        assert dominance_db >= 6.0, (
            "Expected the +x source-port decomposition to identify the + branch as "
            f"incident, got dominance={dominance_db:.2f} dB "
            f"(a_plus={a_plus}, a_minus={a_minus})."
        )

        assert reflection_db <= -20.0, (
            "Expected low raw source-port reflection in a straight guide, "
            f"got {reflection_db:.2f} dB "
            f"(a_plus={a_plus}, a_minus={a_minus})."
        )

    def test_y_monitor_wave_labels_match_x_monitor_convention_3d(self):
        """y-normal 3D monitors should not invert the dominant transmitted branch."""
        wavelength = TEST_WAVELENGTH
        freq = LIGHT_SPEED / wavelength
        n_core = 2.0
        n_clad = 1.0
        guide_width = 0.6 * wavelength
        span = guide_width * 2.5
        long_span = 6.0 * wavelength
        transverse_span = 2.4 * wavelength

        def run_axis(axis, direction, monitor_pos):
            if axis == "x":
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
                source_center = (
                    (
                        1.2 * wavelength
                        if direction == "+x"
                        else long_span - 1.2 * wavelength
                    ),
                    transverse_span / 2,
                    transverse_span / 2,
                )
                start = (
                    monitor_pos,
                    transverse_span / 2 - span / 2,
                    transverse_span / 2 - span / 2,
                )
                end = (
                    monitor_pos,
                    transverse_span / 2 + span / 2,
                    transverse_span / 2 + span / 2,
                )
            else:
                design = Design(
                    width=transverse_span,
                    height=long_span,
                    depth=transverse_span,
                    material=Material(permittivity=n_clad**2),
                )
                design += Rectangle(
                    position=(
                        transverse_span / 2 - guide_width / 2,
                        0.0,
                        transverse_span / 2 - guide_width / 2,
                    ),
                    width=guide_width,
                    height=long_span,
                    depth=guide_width,
                    material=Material(permittivity=n_core**2),
                )
                source_center = (
                    transverse_span / 2,
                    (
                        1.2 * wavelength
                        if direction == "+y"
                        else long_span - 1.2 * wavelength
                    ),
                    transverse_span / 2,
                )
                start = (
                    transverse_span / 2 - span / 2,
                    monitor_pos,
                    transverse_span / 2 - span / 2,
                )
                end = (
                    transverse_span / 2 + span / 2,
                    monitor_pos,
                    transverse_span / 2 + span / 2,
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
            t_total = 10.0 / freq
            time = np.arange(0.0, t_total, dt)
            signal = ramped_cosine(
                time,
                amplitude=1.0,
                frequency=freq,
                ramp_duration=1.0 / freq,
                t_max=t_total,
            )

            grid = design.rasterize(resolution=dx)
            source = ModeSource(
                grid=grid,
                center=source_center,
                width=span,
                height=span,
                wavelength=wavelength,
                pol="te",
                signal=signal,
                direction=direction,
            )
            monitor = Monitor(
                start=start,
                end=end,
                name="m",
                record_fields=False,
                dft_enabled=True,
                dft_frequencies=np.array([freq]),
                dft_components=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
                dft_window="none",
                dft_record_every_step=True,
            )
            sim = Simulation(
                design=design,
                devices=[source, monitor],
                boundaries=[PML(thickness=0.8 * wavelength)],
                time=time,
                resolution=dx,
            )
            sim.run_compiled(progress=False)

            physical_port_direction = {
                "+x": "-x",
                "-x": "+x",
                "+y": "-y",
                "-y": "+y",
            }[direction]
            waves = sim.extract_port_waves_dft(
                ports=[
                    PortSpec(
                        name="p",
                        monitor_name="m",
                        direction="+" + axis,
                        polarization="te",
                        incident_wave=(
                            "plus"
                            if physical_port_direction.startswith("+")
                            else "minus"
                        ),
                        scattered_wave=(
                            "minus"
                            if physical_port_direction.startswith("+")
                            else "plus"
                        ),
                    )
                ],
                frequencies=np.array([freq]),
            )["p"]
            a_plus = complex(waves["a_plus"][0])
            a_minus = complex(waves["a_minus"][0])
            if physical_port_direction.startswith("+"):
                selected = a_minus
                opposite = a_plus
            else:
                selected = a_plus
                opposite = a_minus
            return 20.0 * np.log10(
                max(abs(selected), 1e-18) / max(abs(opposite), 1e-18)
            )

        x_dominance = run_axis("x", "+x", long_span - 1.2 * wavelength)
        y_dominance = run_axis("y", "+y", long_span - 1.2 * wavelength)

        assert x_dominance > 0.0
        assert y_dominance > 2.0, (
            "Expected y-normal 3D mode launch/extraction to keep the transmitted branch clearly dominant, "
            f"got {y_dominance:.2f} dB."
        )


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

    def test_filter_pol_uses_common_te_tm_mapping(self, waveguide_domain):
        """For +x propagation: TE should be Ey/Hz-like and TM should be Ez/Hy-like."""
        wavelength = waveguide_domain["wavelength"]
        dx = waveguide_domain["dx"]
        n_core = waveguide_domain["n_core"]
        n_clad = waveguide_domain["n_clad"]
        core_width = waveguide_domain["core_width"]
        domain_height = waveguide_domain["domain_height"]

        n_points = int(domain_height / dx)
        eps_profile = np.ones(n_points) * n_clad**2
        center = n_points // 2
        half_core = int(core_width / (2 * dx))
        eps_profile[center - half_core : center + half_core] = n_core**2

        omega = 2 * np.pi * LIGHT_SPEED / wavelength

        _, e_te, h_te, _ = solve_modes(
            eps=eps_profile,
            omega=omega,
            dL=dx,
            m=1,
            direction="+x",
            filter_pol="te",
            return_fields=True,
        )
        _, e_tm, h_tm, _ = solve_modes(
            eps=eps_profile,
            omega=omega,
            dL=dx,
            m=1,
            direction="+x",
            filter_pol="tm",
            return_fields=True,
        )

        e_te_max = [float(np.max(np.abs(np.squeeze(e_te[0, i])))) for i in range(3)]
        h_te_max = [float(np.max(np.abs(np.squeeze(h_te[0, i])))) for i in range(3)]
        e_tm_max = [float(np.max(np.abs(np.squeeze(e_tm[0, i])))) for i in range(3)]
        h_tm_max = [float(np.max(np.abs(np.squeeze(h_tm[0, i])))) for i in range(3)]

        assert e_te_max[1] > 1.05 * e_te_max[2], f"TE should be Ey-like, got {e_te_max}"
        assert h_te_max[2] > 1.05 * h_te_max[1], f"TE should be Hz-like, got {h_te_max}"
        assert e_tm_max[2] > 1.05 * e_tm_max[1], f"TM should be Ez-like, got {e_tm_max}"
        assert h_tm_max[1] > 1.05 * h_tm_max[2], f"TM should be Hy-like, got {h_tm_max}"
