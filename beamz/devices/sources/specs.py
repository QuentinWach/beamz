"""Canonical source specifications and immutable field-profile data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

import numpy as np

from beamz.devices._immutable import (
    canonical_tuple,
    immutable_snapshot,
    normalize_source_signal,
    readonly_array,
)

from .time import chebyshev_frequency_nodes

Direction3D = Literal["+x", "-x", "+y", "-y", "+z", "-z"]

FieldAxis3D = Literal["x", "y", "z"]


FieldIndex3D = tuple[int | slice, int | slice, int | slice]


@dataclass(frozen=True)
class FieldProfile3D:
    """Discrete 3D E/H profile on BeamZ's Yee component grids."""

    components: Mapping[str, np.ndarray]
    indices: Mapping[str, FieldIndex3D]
    axis: FieldAxis3D
    direction_sign: float
    omega: float
    k_axis: float | None
    phase_ref_coord: float
    phase_plane_coord: float

    def __post_init__(self):
        object.__setattr__(self, "components", immutable_snapshot(self.components))
        object.__setattr__(self, "indices", immutable_snapshot(self.indices))


@dataclass(frozen=True)
class ModeSpec:
    """Configure mode selection for previews, sources, and monitors.

    Parameters
    ----------
    num_modes : int, default=1
        Number of candidate eigenmodes to solve; clamped to at least one.
    mode_index : int, default=0
        Zero-based selected mode after polarization-aware ordering.
    polarization : {"te", "tm"}, optional
        Preferred polarization used to order candidate modes.
    target_neff : float, optional
        Effective index near which the eigensolver should search.
    num_freqs : int, optional
        Number of frequency samples used for a broadband mode representation.

    Examples
    --------
    >>> spec = ModeSpec(num_modes=4, mode_index=1, polarization="te")

    Notes
    -----
    ``num_modes`` is the number of eigensolver candidates, whereas ``mode_index``
    selects the mode launched or projected after polarization-aware ordering.
    ``num_freqs`` controls broadband mode reconstruction and is independent of the
    frequencies configured on a monitor.
    """

    num_modes: int = 1
    mode_index: int = 0
    polarization: str | None = None
    target_neff: float | None = None
    num_freqs: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_modes", max(1, int(self.num_modes)))
        object.__setattr__(self, "mode_index", max(0, int(self.mode_index)))
        if self.num_freqs is not None:
            object.__setattr__(self, "num_freqs", max(1, int(self.num_freqs)))
        if self.polarization is not None:
            pol = str(self.polarization).lower()
            if pol not in {"te", "tm"}:
                raise ValueError(f"polarization must be 'te' or 'tm', got {pol!r}.")
            object.__setattr__(self, "polarization", pol)


@dataclass(frozen=True)
class ModeData:
    """Store immutable modal fields returned by a mode solve.

    Parameters
    ----------
    frequencies : array-like
        Solved frequencies in hertz, shape ``(num_frequencies,)``.
    neffs : array-like
        Complex effective indices, shape ``(num_frequencies, num_modes)``.
    e_fields : array-like
        Complex electric modal fields with frequency and mode leading axes.
    h_fields : array-like
        Complex magnetic modal fields matching ``e_fields``.
    eps_profiles : array-like
        Relative-permittivity cross sections used by the eigensolver.
    resolution : float
        Cross-section cell size in metres.
    solver_direction : str, optional
        Signed propagation direction used by the eigensolver.
    axis : str, optional
        Plane-normal axis, one of ``"x"``, ``"y"``, or ``"z"``.
    center : tuple, optional
        Mode-plane center in ``(x, y, z)`` order, in metres.
    plane : object, optional
        Original public plane specification.
    crop_slices : tuple of slice, optional
        Slices locating the solved aperture inside the full material plane.
    eps_profile_fulls : array-like, optional
        Full uncropped relative-permittivity planes for later analysis.

    Notes
    -----
    Array inputs are copied and made read-only.
    """

    frequencies: np.ndarray
    neffs: np.ndarray
    e_fields: np.ndarray
    h_fields: np.ndarray
    eps_profiles: np.ndarray
    resolution: float
    solver_direction: str | None = None
    axis: str | None = None
    center: tuple[float, float, float] | None = None
    plane: Any = field(default=None, compare=False, hash=False, repr=False)
    crop_slices: tuple[slice, ...] | None = field(
        default=None, compare=False, hash=False
    )
    eps_profile_fulls: np.ndarray | None = field(
        default=None, compare=False, hash=False, repr=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "frequencies", readonly_array(self.frequencies, dtype=float)
        )
        object.__setattr__(
            self, "neffs", readonly_array(self.neffs, dtype=np.complex128)
        )
        object.__setattr__(
            self, "e_fields", readonly_array(self.e_fields, dtype=np.complex128)
        )
        object.__setattr__(
            self, "h_fields", readonly_array(self.h_fields, dtype=np.complex128)
        )
        object.__setattr__(self, "eps_profiles", readonly_array(self.eps_profiles))
        if self.eps_profile_fulls is not None:
            object.__setattr__(
                self, "eps_profile_fulls", readonly_array(self.eps_profile_fulls)
            )
        if self.center is not None:
            object.__setattr__(self, "center", tuple(float(v) for v in self.center[:3]))

    def _frequency_index(self, f=None) -> int:
        if self.frequencies.size == 0:
            raise ValueError("ModeData contains no frequencies.")
        if f is None:
            return 0
        return int(
            np.argmin(np.abs(np.asarray(self.frequencies, dtype=float) - float(f)))
        )

    def selected_mode(self, *, f=None, mode_index=0):
        """Return arrays for the nearest frequency and selected mode index.

        Parameters
        ----------
        f : float, optional
            Requested frequency in hertz. By default, select the first solved
            frequency; otherwise select the nearest available sample.
        mode_index : int, default=0
            Zero-based modal index.
        """
        f_idx = self._frequency_index(f)
        m_idx = int(mode_index)
        neff = np.asarray(self.neffs)[f_idx, m_idx]
        e_field = np.asarray(self.e_fields)[f_idx, m_idx]
        h_field = np.asarray(self.h_fields)[f_idx, m_idx]
        eps_full = (
            np.asarray(self.eps_profile_fulls)[f_idx]
            if self.eps_profile_fulls is not None
            else np.asarray(self.eps_profiles)[f_idx]
        )
        return f_idx, m_idx, neff, e_field, h_field, eps_full


SourceDirection = Literal["+", "-"]


def _center_size(
    value: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    center = tuple(float(item) for item in value.center)
    size = tuple(float(item) for item in value.size)
    if len(center) != 3 or len(size) != 3:
        raise ValueError("Planar objects require three-dimensional center and size.")
    return center, size


def plane_axis_and_spans(plane):
    """Return the normal axis, center, and tangential extents of a plane."""
    center, size = _center_size(plane)
    zero_axes = np.flatnonzero(np.isclose(size, 0.0, rtol=0.0, atol=1e-15))
    if zero_axes.size != 1:
        raise ValueError("A modal plane size must contain exactly one zero extent.")
    normal = int(zero_axes[0])
    axis = ("x", "y", "z")[normal]
    spans = tuple(size[index] for index in range(3) if index != normal)
    return axis, center, spans


@dataclass(frozen=True)
class ModeSource:
    """Inject a solved waveguide mode through a finite plane.

    The zero extent in ``size`` selects the injection axis. ``direction`` is
    ``"+"`` or ``"-"`` along that axis, matching Tidy3D's modal source API.
    Temporal behavior belongs exclusively to ``source_time``; sampled drives
    use :class:`~beamz.SampledSignal`.
    """

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    source_time: Any
    direction: SourceDirection
    mode_spec: ModeSpec = field(default_factory=ModeSpec)
    power: float = 1.0

    def __post_init__(self) -> None:
        center = tuple(float(value) for value in self.center)
        size = tuple(float(value) for value in self.size)
        if len(center) != 3 or len(size) != 3:
            raise ValueError("ModeSource center and size must contain three values.")
        if any(not np.isfinite(value) for value in center):
            raise ValueError("ModeSource center must be finite.")
        if any(value < 0.0 or np.isnan(value) for value in size):
            raise ValueError("ModeSource size must contain non-negative extents.")
        direction = str(self.direction)
        if direction not in {"+", "-"}:
            raise ValueError("ModeSource direction must be '+' or '-'.")
        power = float(self.power)
        if not np.isfinite(power) or power < 0.0:
            raise ValueError("ModeSource power must be non-negative and finite.")
        source_time = immutable_snapshot(self.source_time)
        if source_time is None or not hasattr(source_time, "sample"):
            raise TypeError("ModeSource source_time must provide sample(time).")
        freq0 = getattr(source_time, "freq0", None)
        if freq0 is None or not np.isfinite(float(freq0)) or float(freq0) <= 0.0:
            raise ValueError("ModeSource source_time must define a positive freq0.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "source_time", source_time)
        object.__setattr__(self, "direction", direction)
        mode_spec = immutable_snapshot(self.mode_spec)
        if not isinstance(mode_spec, ModeSpec):
            raise TypeError("ModeSource mode_spec must be a ModeSpec.")
        object.__setattr__(self, "mode_spec", mode_spec)
        object.__setattr__(self, "power", power)
        plane_axis_and_spans(self)

    @property
    def axis(self) -> str:
        """Return the source-plane normal axis."""
        return plane_axis_and_spans(self)[0]

    @property
    def signed_direction(self) -> str:
        """Return the internal signed-axis propagation direction."""
        return f"{self.direction}{self.axis}"

    @property
    def frequency(self) -> float:
        """Return the source-time carrier frequency in hertz."""
        return float(self.source_time.freq0)

    @property
    def transverse_size(self) -> tuple[float, float]:
        """Return the two tangential source-plane extents."""
        spans = plane_axis_and_spans(self)[2]
        return float(spans[0]), float(spans[1])

    def profile_frequencies(self) -> np.ndarray:
        """Return mode-profile frequencies requested by ``mode_spec``."""
        count = int(self.mode_spec.num_freqs or 1)
        fwidth = float(getattr(self.source_time, "fwidth", self.frequency / 10.0))
        return chebyshev_frequency_nodes(self.frequency, fwidth, count)

    def updated_copy(self, **changes):
        """Return a source with selected canonical fields replaced."""
        return replace(self, **changes)

    def shifted(self, offset):
        """Return a translated source."""
        delta = tuple(float(value) for value in offset)
        if len(delta) != 3:
            raise ValueError("ModeSource offsets must contain three values.")
        return replace(
            self,
            center=tuple(
                value + shift for value, shift in zip(self.center, delta, strict=True)
            ),
        )

    def solve_modes(self, simulation, freqs=None, mode_spec=None) -> ModeData:
        """Solve modal fields on this source plane."""
        from .solve import solve_mode_plane

        frequencies = (
            np.asarray([self.frequency], dtype=float)
            if freqs is None
            else np.asarray(freqs, dtype=float).reshape(-1)
        )
        if frequencies.size == 0:
            raise ValueError("ModeSource mode solving requires a frequency.")
        return solve_mode_plane(
            simulation=simulation,
            plane=self,
            mode_spec=self.mode_spec if mode_spec is None else mode_spec,
            freqs=frequencies,
            direction=self.signed_direction,
        )

    def source_spectrum(self, freqs, *, normalize: bool = True) -> np.ndarray | None:
        """Return the complex temporal spectrum at frequencies in hertz."""
        frequency = np.asarray(freqs, dtype=float)
        if normalize and hasattr(self.source_time, "dft_normalization_spectrum"):
            spectrum = self.source_time.dft_normalization_spectrum(frequency)
            return np.asarray(spectrum, dtype=np.complex128) * np.sqrt(
                np.maximum(frequency, 0.0) / self.frequency
            )
        if not hasattr(self.source_time, "spectrum"):
            return None
        return np.asarray(
            self.source_time.spectrum(frequency, normalize=normalize),
            dtype=np.complex128,
        )

    def launch_power_normalization_spectrum(self, freqs, *, fields=None, dt=None):
        """Return frequency-dependent launched-power normalization."""
        del freqs, fields, dt
        return None


@dataclass(frozen=True, slots=True)
class GaussianSource:
    """Inject a Gaussian spatial profile into an FDTD simulation.

    The source primarily excites ``Ez`` in 2D and compatible electric components
    in 3D. Position and width remain differentiable through JAX.

    Parameters
    ----------
    position : tuple of float
        Source center in ``(x, y)`` or ``(x, y, z)`` order, in metres.
    width : float
        Positive Gaussian standard-width scale in metres.
    signal : array-like
        Immutable temporal samples evaluated at the simulation time step.

    Examples
    --------
    >>> import numpy as np
    >>> import beamz as bz
    >>> time = np.linspace(0.0, 1e-13, 100)
    >>> source = bz.GaussianSource(
    ...     position=(1 * bz.um, 1.5 * bz.um),
    ...     width=0.2 * bz.um,
    ...     signal=np.sin(2 * np.pi * 2e14 * time),
    ... )
    """

    position: tuple[float, ...]
    width: float
    signal: Any = field(compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        position = canonical_tuple(self.position, dtype=float)
        if len(position) not in {2, 3}:
            raise ValueError("GaussianSource position must be a 2D or 3D coordinate.")
        width = float(self.width)
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("GaussianSource width must be a positive finite value.")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "width", width)
        object.__setattr__(
            self,
            "signal",
            normalize_source_signal(self.signal, name="GaussianSource.signal"),
        )

    def updated_copy(self, **changes):
        """Return a source with selected configuration fields replaced.

        Parameters
        ----------
        **changes
            Source field names and replacement values.
        """
        return replace(self, **changes)

    def shifted(self, offset):
        """Return the source translated by a physical coordinate offset.

        Parameters
        ----------
        offset : tuple of float
            Cartesian translation in metres.
        """
        offset = tuple(float(v) for v in offset)
        return replace(
            self,
            position=tuple(a + b for a, b in zip(self.position, offset, strict=False)),
        )


_VALID_DIRECTIONS_3D = {"+x", "-x", "+y", "-y", "+z", "-z"}


def _as_xyz(value, *, name: str) -> tuple[float, float, float]:
    values = tuple(float(v) for v in value)
    if len(values) != 3:
        raise ValueError(f"GaussianBeamSource {name} must be a 3D coordinate.")
    return (values[0], values[1], values[2])


def _as_direction(value) -> Direction3D:
    direction = str(value)
    if direction not in _VALID_DIRECTIONS_3D:
        raise ValueError(f"Unsupported GaussianBeamSource direction {direction!r}.")
    return cast(Direction3D, direction)


@dataclass(frozen=True, slots=True)
class GaussianBeamSource:
    """Launch an angled 3D Gaussian beam through a planar TF/SF boundary.

    Parameters
    ----------
    center : tuple of float
        Source-plane center ``(x, y, z)`` in metres.
    size : tuple of float
        Plane extents ``(x, y, z)`` in metres; one extent must be zero.
    source_time : object
        Analytic temporal source providing samples and optionally a spectrum.
    direction : signed axis, default="-z"
        Direction normal to the source plane.
    angle_theta : float, default=0
        Polar incidence angle in radians relative to the plane normal.
    angle_phi : float, default=0
        Azimuthal incidence angle in radians.
    pol_angle : float, default=0
        Linear-polarization angle in radians.
    waist_radius : float, optional
        Beam waist radius in metres; derived from the aperture when omitted.
    waist_distance : float, default=0
        Signed distance from the source plane to the waist, in metres.
    power : float, default=1.0
        Requested non-negative launched power in watts.
    background_index : float, default=1.0
        Refractive index used for Gaussian propagation parameters.
    wavelength : float, optional
        Vacuum wavelength in metres; otherwise derived from ``source_time``.
    max_shift : int, default=1
        Maximum discrete Yee-plane shift used by residual construction.

    Examples
    --------
    >>> import beamz as bz
    >>> pulse = bz.GaussianPulse(freq0=193.5e12, fwidth=20e12)
    >>> source = bz.GaussianBeamSource(
    ...     center=(2 * bz.um, 2 * bz.um, 1 * bz.um),
    ...     size=(2 * bz.um, 2 * bz.um, 0.0),
    ...     source_time=pulse,
    ...     direction="-z",
    ...     waist_radius=0.6 * bz.um,
    ... )
    """

    center: tuple[float, float, float]
    size: Any
    source_time: Any
    direction: Direction3D = "-z"
    angle_theta: float = 0.0
    angle_phi: float = 0.0
    pol_angle: float = 0.0
    waist_radius: float | None = None
    waist_distance: float = 0.0
    power: float = 1.0
    background_index: float = 1.0
    wavelength: float | None = None
    max_shift: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _as_xyz(self.center, name="center"))
        object.__setattr__(self, "size", canonical_tuple(self.size, dtype=float))
        object.__setattr__(self, "source_time", immutable_snapshot(self.source_time))
        object.__setattr__(self, "direction", _as_direction(self.direction))
        object.__setattr__(self, "angle_theta", float(self.angle_theta))
        object.__setattr__(self, "angle_phi", float(self.angle_phi))
        object.__setattr__(self, "pol_angle", float(self.pol_angle))
        object.__setattr__(
            self,
            "waist_radius",
            None if self.waist_radius is None else float(self.waist_radius),
        )
        object.__setattr__(self, "waist_distance", float(self.waist_distance))
        power_value = float(self.power)
        if not np.isfinite(power_value) or power_value < 0.0:
            raise ValueError(
                "GaussianBeamSource power must be a non-negative finite value, "
                f"got {self.power!r}."
            )
        object.__setattr__(self, "power", power_value)
        object.__setattr__(self, "background_index", float(self.background_index))
        object.__setattr__(
            self,
            "wavelength",
            None if self.wavelength is None else float(self.wavelength),
        )
        object.__setattr__(self, "max_shift", int(max(1, self.max_shift)))

    def shifted(self, offset):
        """Return the beam plane translated by an ``(x, y, z)`` offset.

        Parameters
        ----------
        offset : tuple of float
            Cartesian translation in metres.
        """
        return replace(
            self,
            center=tuple(
                a + b for a, b in zip(self.center, tuple(offset), strict=True)
            ),
        )

    def updated_copy(self, **changes):
        """Return a beam with selected configuration fields replaced.

        Parameters
        ----------
        **changes
            Beam-source field names and replacement values.
        """
        return replace(self, **changes)

    def source_spectrum(self, freqs, *, normalize: bool = True) -> np.ndarray | None:
        """Return the complex temporal spectrum at frequencies in hertz.

        Parameters
        ----------
        freqs : array-like
            Frequencies in hertz at which to evaluate the temporal waveform.
        normalize : bool, default=True
            Apply the waveform's DFT normalization when available.
        """
        source_time = self.source_time
        if source_time is None:
            return None
        freq_arr = np.asarray(freqs, dtype=float)
        if normalize and hasattr(source_time, "dft_normalization_spectrum"):
            return np.asarray(
                source_time.dft_normalization_spectrum(freq_arr),
                dtype=np.complex128,
            )
        if hasattr(source_time, "spectrum"):
            try:
                return np.asarray(
                    source_time.spectrum(freq_arr, normalize=normalize),
                    dtype=np.complex128,
                )
            except TypeError:
                return np.asarray(source_time.spectrum(freq_arr), dtype=np.complex128)
        return None


@dataclass(frozen=True, slots=True)
class CustomSource:
    """Describe an immutable precomputed field injection.

    Parameters
    ----------
    component : str
        Target field component, such as ``"Ez"`` or ``"Hy"``.
    timing : {"e", "h"}
        Update half-step on which the source is applied.
    index : tuple
        NumPy-style index selecting target cells in the component array.
    coeff : array-like
        Spatial injection coefficient broadcastable to the indexed region.
    waveform : array-like
        Per-step temporal samples.
    target_shape : tuple of int
        Expected full target component shape used for validation.

    Notes
    -----
    All array-like inputs are recursively snapshotted to prevent runtime
    mutation from changing a compiled source.
    """

    component: str
    timing: str
    index: tuple[Any, ...]
    coeff: Any
    waveform: Any
    target_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "component", str(self.component))
        object.__setattr__(self, "timing", str(self.timing))
        object.__setattr__(self, "index", immutable_snapshot(self.index))
        object.__setattr__(self, "coeff", immutable_snapshot(self.coeff))
        object.__setattr__(self, "waveform", immutable_snapshot(self.waveform))
        object.__setattr__(
            self, "target_shape", tuple(int(v) for v in self.target_shape)
        )

    def updated_copy(self, **changes):
        """Return an injection with selected immutable fields replaced.

        Parameters
        ----------
        **changes
            Injection field names and replacement values.
        """
        return replace(self, **changes)

    def shifted(self, offset):
        """Keep grid-indexed injections unchanged by coordinate normalization."""
        del offset
        return self
