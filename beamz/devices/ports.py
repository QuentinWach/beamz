"""Canonical modal ports."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Iterable, Literal, Sequence

import numpy as np

from beamz.devices._immutable import immutable_snapshot
from beamz.devices.sources.specs import ModeSpec

if TYPE_CHECKING:
    from beamz.devices.monitors import ModeMonitor
    from beamz.devices.sources import ModeSource

PortDirection = Literal["+", "-"]


@dataclass(frozen=True)
class Port:
    """Define a modal S-parameter port on a finite plane.

    A port owns geometry and mode-solver settings, then creates matching
    :class:`ModeSource` and :class:`ModeMonitor` objects as needed. The zero
    extent in ``size`` selects the port axis and ``direction`` points into the
    simulated device.

    Parameters
    ----------
    center, size : tuple of float
        Plane center and extents in public ``(x, y, z)`` order, in metres.
        Exactly one size component must be zero.
    name : str
        Stable port identifier used by analysis.
    direction : {"+", "-"}
        Inward propagation direction along the plane-normal axis.
    mode_spec : ModeSpec, optional
        Modal selection and eigensolver settings.
    monitor_name : str, optional
        Name of the matching mode monitor. Defaults to ``name``.

    Examples
    --------
    >>> port = Port(
    ...     center=(0.0, 1e-6, 0.0),
    ...     size=(0.0, 2e-6, 1e-6),
    ...     name="input",
    ...     direction="+",
    ... )

    Notes
    -----
    ``Port`` is an analysis specification, not a runtime object. Use
    :meth:`to_source` and :meth:`to_monitor` to create matching simulation devices.
    """

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    name: str
    direction: PortDirection
    mode_spec: ModeSpec = field(default_factory=ModeSpec)
    monitor_name: str | None = None

    def __post_init__(self) -> None:
        center = tuple(float(value) for value in self.center)
        size = tuple(float(value) for value in self.size)
        if len(center) != 3 or len(size) != 3:
            raise ValueError("Port center and size must contain three values.")
        if any(not np.isfinite(value) for value in center):
            raise ValueError("Port center must be finite.")
        if any(value < 0.0 or np.isnan(value) for value in size):
            raise ValueError("Port size must contain non-negative extents.")
        zero_axes = np.flatnonzero(np.isclose(size, 0.0, rtol=0.0, atol=1e-15))
        if zero_axes.size != 1:
            raise ValueError("Port size must contain exactly one zero extent.")
        name = str(self.name)
        if not name:
            raise ValueError("Port name cannot be empty.")
        direction = str(self.direction)
        if direction not in {"+", "-"}:
            raise ValueError("Port direction must be '+' or '-'.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "direction", direction)
        monitor_name = name if self.monitor_name is None else str(self.monitor_name)
        if not monitor_name:
            raise ValueError("Port monitor_name cannot be empty.")
        object.__setattr__(self, "monitor_name", monitor_name)
        mode_spec = immutable_snapshot(self.mode_spec)
        if not isinstance(mode_spec, ModeSpec):
            raise TypeError("Port mode_spec must be a ModeSpec.")
        object.__setattr__(self, "mode_spec", mode_spec)

    @property
    def axis(self) -> str:
        """Return the plane-normal axis.

        Returns
        -------
        {"x", "y", "z"}
            Axis selected by the zero extent in :attr:`size`.
        """
        index = int(np.argmin(np.abs(np.asarray(self.size, dtype=float))))
        return ("x", "y", "z")[index]

    @property
    def signed_direction(self) -> str:
        """Return the signed input propagation direction.

        Returns
        -------
        str
            Direction such as ``"+x"`` or ``"-z"``.
        """
        return f"{self.direction}{self.axis}"

    @property
    def num_modes(self) -> int:
        """Return the number of candidate modes represented by this port.

        Returns
        -------
        int
            Positive mode count from :attr:`mode_spec`.
        """
        return int(self.mode_spec.num_modes)

    @property
    def mode_index(self) -> int:
        """Return the zero-based mode selected for modal projection.

        Returns
        -------
        int
            Selected index from :attr:`mode_spec`.
        """
        return int(self.mode_spec.mode_index)

    @property
    def polarization(self) -> str:
        """Return the modal polarization used by analysis.

        Returns
        -------
        {"te", "tm"}
            Configured polarization, defaulting to ``"te"`` when unspecified.
        """
        return str(self.mode_spec.polarization or "te").lower()

    @property
    def projection_direction(self) -> str:
        """Return the positive-axis basis used for modal projection.

        Returns
        -------
        str
            Positive signed axis such as ``"+x"``.

        Notes
        -----
        Incoming and outgoing waves are interpreted relative to this fixed modal
        basis and the port's :attr:`direction`.
        """
        return f"+{self.axis}"

    def updated_copy(self, **changes: Any) -> Port:
        """Return a validated port with selected fields replaced.

        Parameters
        ----------
        **changes : object
            Dataclass fields to replace.

        Returns
        -------
        Port
            New immutable port; the original is unchanged.

        Raises
        ------
        TypeError or ValueError
            If a field is unknown or the new geometry is invalid.
        """
        return replace(self, **changes)

    def shifted(self, offset: Iterable[float]) -> Port:
        """Return a translated copy of this port.

        Parameters
        ----------
        offset : iterable of float
            Three-coordinate ``(dx, dy, dz)`` translation in metres.

        Returns
        -------
        Port
            Port with translated center and unchanged size and modal settings.

        Raises
        ------
        ValueError
            If ``offset`` does not contain exactly three values.
        """
        delta = tuple(float(value) for value in offset)
        if len(delta) != 3:
            raise ValueError("Port offsets must contain three values.")
        return replace(
            self,
            center=tuple(
                value + shift for value, shift in zip(self.center, delta, strict=True)
            ),
        )

    def to_monitor(self, freqs: Sequence[float] | np.ndarray) -> ModeMonitor:
        """Create a mode monitor matching this port.

        Parameters
        ----------
        freqs : sequence of float
            Frequencies to acquire, in hertz.

        Returns
        -------
        ModeMonitor
            Monitor with matching geometry, name, and mode specification.

        Examples
        --------
        >>> monitor = port.to_monitor([193.5e12])
        """
        from beamz.devices.monitors import ModeMonitor

        return ModeMonitor(
            center=self.center,
            size=self.size,
            freqs=np.asarray(freqs, dtype=float),
            mode_spec=self.mode_spec,
            name=self.monitor_name,
        )

    def to_source(
        self,
        freq0: float,
        fwidth: float,
        mode_index: int = 0,
        num_freqs: int = 1,
        *,
        source_time: Any | None = None,
        power: float = 1.0,
    ) -> ModeSource:
        """Create a mode source matching this port.

        Parameters
        ----------
        freq0 : float
            Carrier frequency in hertz.
        fwidth : float
            Gaussian frequency width in hertz when ``source_time`` is omitted.
        mode_index : int, default=0
            Zero-based mode to launch.
        num_freqs : int, default=1
            Frequency samples used for broadband modal reconstruction.
        source_time : source-time specification, optional
            Custom temporal waveform. A ``GaussianPulse`` is created by default.
        power : float, default=1.0
            Requested launched power in watts.

        Returns
        -------
        ModeSource
            Source with matching geometry, inward direction, and modal settings.

        Examples
        --------
        >>> source = port.to_source(freq0=193.5e12, fwidth=20e12)
        """
        from beamz.devices.sources import GaussianPulse, ModeSource

        if source_time is None:
            source_time = GaussianPulse(freq0=float(freq0), fwidth=float(fwidth))
        mode_spec = replace(
            self.mode_spec,
            mode_index=int(mode_index),
            num_freqs=max(1, int(num_freqs)),
        )
        return ModeSource(
            center=self.center,
            size=self.size,
            source_time=source_time,
            direction=self.direction,
            mode_spec=mode_spec,
            power=power,
        )


__all__ = ["Port"]
