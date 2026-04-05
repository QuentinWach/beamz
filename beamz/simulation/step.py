"""Runtime helpers for stepping a simulation session."""

from __future__ import annotations

from beamz.devices.monitors.monitors import Monitor


def record_monitors(sim) -> None:
    """Record data from Monitor devices during simulation."""
    for device in sim.devices:
        if not isinstance(device, Monitor):
            continue
        should_record = device.should_record(sim.current_step)
        dft_every_step = bool(
            getattr(device, "dft_enabled", False)
            and getattr(device, "dft_record_every_step", True)
        )
        if should_record or dft_every_step:
            if not sim.is_3d:
                device.record_fields_2d(
                    sim.fields.Ez,
                    sim.fields.Hx,
                    sim.fields.Hy,
                    sim.t,
                    sim.resolution,
                    sim.resolution,
                    sim.current_step,
                    Ex=sim.fields.Ex,
                    Ey=sim.fields.Ey,
                    Hz=sim.fields.Hz,
                )
            else:
                device.record_fields(
                    sim.fields.Ex,
                    sim.fields.Ey,
                    sim.fields.Ez,
                    sim.fields.Hx,
                    sim.fields.Hy,
                    sim.fields.Hz,
                    sim.t,
                    sim.resolution,
                    sim.resolution,
                    sim.resolution,
                    sim.current_step,
                )


def inject_h_sources(sim) -> None:
    """Inject magnetic currents (M) into H-fields after H update."""
    for device in sim.devices:
        if hasattr(device, "inject_h"):
            device.inject_h(
                sim.fields,
                sim.t,
                sim.dt,
                sim.current_step,
                sim.resolution,
                sim.design,
            )


def inject_e_sources(sim) -> None:
    """Inject electric currents (J) into E-fields after E update."""
    for device in sim.devices:
        if hasattr(device, "inject_e"):
            device.inject_e(
                sim.fields,
                sim.t,
                sim.dt,
                sim.current_step,
                sim.resolution,
                sim.design,
            )


def inject_legacy_sources(sim) -> None:
    """Inject from devices that only have inject() (no inject_h/inject_e)."""
    for device in sim.devices:
        if hasattr(device, "inject") and not hasattr(device, "inject_h"):
            device.inject(
                sim.fields,
                sim.t,
                sim.dt,
                sim.current_step,
                sim.resolution,
                sim.design,
            )


def collect_source_terms(sim):
    """Collect electric and magnetic current sources from all devices."""
    source_j = {}
    source_m = {}

    for device in sim.devices:
        if hasattr(device, "get_source_terms"):
            j, m = device.get_source_terms(
                sim.fields,
                sim.t,
                sim.dt,
                sim.current_step,
                sim.resolution,
                sim.design,
            )
            for key, val in j.items():
                source_j.setdefault(key, []).append(val)
            for key, val in m.items():
                source_m.setdefault(key, []).append(val)

    return source_j, source_m


def run_step(sim) -> bool:
    """Perform one FDTD time step with the current source and monitor setup."""
    if sim.current_step >= sim.num_steps:
        return False

    inject_legacy_sources(sim)
    source_j, source_m = collect_source_terms(sim)

    sim.fields.update_h(sim.dt, source_m=source_m)
    inject_h_sources(sim)

    sim.fields.update_e(sim.dt, source_j=source_j)
    inject_e_sources(sim)

    record_monitors(sim)

    if sim.thermal is not None and getattr(sim.thermal, "enabled", True):
        sim.thermal.step(sim)

    sim.t += sim.dt
    sim.current_step += 1
    return True
