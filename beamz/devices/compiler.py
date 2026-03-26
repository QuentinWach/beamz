"""Compiled-device registry and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass

from beamz.devices.monitors.monitors import Monitor
from beamz.devices.sources.gaussian import GaussianSource
from beamz.devices.sources.mode import ModeSource


@dataclass(frozen=True)
class CompiledDeviceRegistry:
    source_types: tuple[type, ...] = (GaussianSource, ModeSource)
    monitor_types: tuple[type, ...] = (Monitor,)

    @property
    def all_types(self) -> tuple[type, ...]:
        return self.source_types + self.monitor_types


COMPILED_DEVICE_REGISTRY = CompiledDeviceRegistry()


def validate_compilable_devices(devices: list) -> None:
    """Reject device objects that have no compiled lowering."""
    unsupported: list[str] = []
    for device in devices:
        if isinstance(device, COMPILED_DEVICE_REGISTRY.all_types):
            continue
        unsupported.append(type(device).__name__)
    if unsupported:
        unsupported_names = ", ".join(sorted(set(unsupported)))
        raise NotImplementedError(
            "run_compiled does not support device types without compiled lowering: "
            f"{unsupported_names}."
        )
