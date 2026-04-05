import os

from beamz.design.core import material_grids_from_spec
from beamz.devices.monitors.compiler import compile_monitor_specs
from beamz.devices.monitors.monitors import Monitor
from beamz.devices.sources.compiler import compile_source_specs
from beamz.simulation.boundaries import PML, boundary_from_spec
from beamz.simulation.boundary_specs import PMLSpec
from beamz.simulation.fields import Fields
from beamz.simulation import ops, shell
from beamz.simulation.material_models import CompiledMaterialSpec


def _merge_pml_region_data(pml_data, new_data):
    if not pml_data:
        return dict(new_data)

    merged = dict(pml_data)
    if "mask" in new_data and "mask" in merged:
        merged["mask"] = merged["mask"] | new_data["mask"]
    elif "mask" in new_data:
        merged["mask"] = new_data["mask"]

    for key, value in new_data.items():
        if key == "mask":
            continue
        merged[key] = merged[key] + value if key in merged else value
    return merged


def initialize_runtime(sim):
    """Populate field storage and PML state from a simulation spec."""
    spec = sim.spec
    runtime = sim.runtime
    design_model = getattr(sim, "_design", None)
    design = design_model if getattr(design_model, "spec", None) == spec.design else None
    boundaries = tuple(boundary_from_spec(boundary_spec) for boundary_spec in spec.boundaries)

    permittivity, conductivity, permeability = material_grids_from_spec(
        spec.design,
        spec.resolution,
        design_model=design,
    )
    runtime.dt = float(spec.time[1] - spec.time[0])
    runtime.num_steps = len(spec.time)
    runtime.t = float(spec.time[0])
    runtime.current_step = 0

    pml_boundaries = [
        boundary
        for boundary, boundary_spec in zip(boundaries, spec.boundaries)
        if isinstance(boundary_spec, PMLSpec) and isinstance(boundary, PML)
    ]
    runtime.fields = Fields(
        permittivity,
        conductivity,
        permeability,
        spec.resolution,
        plane_2d=spec.plane_2d,
        _init_materials=not pml_boundaries,
    )

    runtime.pml_data = None
    if not pml_boundaries:
        runtime.initialized = True
        return

    pml_data = {}
    for pml in pml_boundaries:
        pml_data = _merge_pml_region_data(
            pml_data,
            pml.create_pml_regions(
                runtime.fields,
                spec.design,
                spec.resolution,
                runtime.dt,
                plane_2d=spec.plane_2d,
            ),
        )
    runtime.pml_data = pml_data
    runtime.fields.set_pml_conductivity(pml_data)
    runtime.initialized = True


def ensure_runtime_initialized(sim):
    if not sim.runtime.initialized:
        initialize_runtime(sim)


def invalidate_runtime(sim):
    runtime = sim.runtime
    runtime.initialized = False
    runtime.fields = None
    runtime.dt = 0.0
    runtime.num_steps = 0
    runtime.t = 0.0
    runtime.current_step = 0
    runtime.pml_data = None


def compile_simulation(design, devices, boundaries, run_cfg, *, compiled_cls, config_cls):
    """Build a compiled simulation program from runtime config and fields."""
    del design, boundaries

    fields = run_cfg.fields
    resolution = float(run_cfg.resolution)
    dt = float(run_cfg.dt)
    num_steps = int(run_cfg.num_steps)
    total_steps = int(getattr(run_cfg, "total_steps", num_steps))
    t0 = float(getattr(run_cfg, "t0", 0.0))

    source_specs = compile_source_specs(
        devices=devices,
        fields=fields,
        dt=dt,
        resolution=resolution,
        num_steps=num_steps,
        t0=t0,
        total_steps=total_steps,
    )

    monitor_specs, _ = compile_monitor_specs(
        devices=devices,
        fields=fields,
        resolution=resolution,
        num_steps=num_steps,
        dt=dt,
    )

    monitor_devices = tuple(device for device in devices if isinstance(device, Monitor))

    loop_kind_raw = str(
        getattr(run_cfg, "loop_kind", os.getenv("BEAMZ_COMPILED_LOOP_KIND", "scan"))
    ).strip().lower()
    if loop_kind_raw in {"fori", "fori_loop", "fori-loop"}:
        loop_kind = "fori_loop"
    elif loop_kind_raw in {"scan"}:
        loop_kind = "scan"
    else:
        raise ValueError("Invalid compiled loop kind. Use one of: scan, fori_loop.")

    source_single_slab_dense = os.getenv(
        "BEAMZ_SOURCE_SINGLE_SLAB_DENSE",
        str(getattr(run_cfg, "source_single_slab_dense", False)),
    ).strip().lower() in {"1", "true", "yes", "on"}

    config = config_cls(
        resolution=resolution,
        dt=dt,
        num_steps=num_steps,
        plane_2d=run_cfg.plane_2d,
        is_3d=bool(run_cfg.is_3d),
        precision=getattr(run_cfg, "precision", "float32"),
        loop_kind=loop_kind,
        source_single_slab_dense=source_single_slab_dense,
    )

    h_decay_x, h_source_x, h_source_lossless_x = ops.precompute_h_update_coefficients(
        fields.sigma_m_hx, dt
    )
    h_decay_y, h_source_y, h_source_lossless_y = ops.precompute_h_update_coefficients(
        fields.sigma_m_hy, dt
    )
    h_decay_z, h_source_z, h_source_lossless_z = ops.precompute_h_update_coefficients(
        fields.sigma_m_hz, dt
    )

    e_decay_x, e_source_x, e_source_lossless_x = ops.precompute_e_update_coefficients(
        shape=fields.Ex.shape,
        conductivity=fields.sig_x,
        permittivity=fields.eps_x,
        dt=dt,
        region=fields.region_x,
    )
    e_decay_y, e_source_y, e_source_lossless_y = ops.precompute_e_update_coefficients(
        shape=fields.Ey.shape,
        conductivity=fields.sig_y,
        permittivity=fields.eps_y,
        dt=dt,
        region=fields.region_y,
    )
    e_decay_z, e_source_z, e_source_lossless_z = ops.precompute_e_update_coefficients(
        shape=fields.Ez.shape,
        conductivity=fields.sig_z,
        permittivity=fields.eps_z,
        dt=dt,
        region=fields.region_z,
    )

    e_shell_frac_threshold = 0.35
    h_shell_frac_threshold = 0.20
    enable_e_shell_split = os.getenv("BEAMZ_ENABLE_E_SHELL_SPLIT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    enable_h_shell_split = os.getenv("BEAMZ_ENABLE_H_SHELL_SPLIT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if bool(run_cfg.is_3d):
        e_use_lossy_shell_x, e_lossy_shell_x = shell.infer_lossy_shell_slabs(
            field_shape=tuple(fields.Ex.shape),
            region=fields.region_x,
            conductivity_region=fields.sig_x,
        )
        e_use_lossy_shell_y, e_lossy_shell_y = shell.infer_lossy_shell_slabs(
            field_shape=tuple(fields.Ey.shape),
            region=fields.region_y,
            conductivity_region=fields.sig_y,
        )
        e_use_lossy_shell_z, e_lossy_shell_z = shell.infer_lossy_shell_slabs(
            field_shape=tuple(fields.Ez.shape),
            region=fields.region_z,
            conductivity_region=fields.sig_z,
        )
        h_use_lossy_shell_x, h_lossy_shell_x = shell.infer_lossy_shell_slabs(
            field_shape=tuple(fields.Hx.shape),
            region=(slice(None), slice(None), slice(None)),
            conductivity_region=fields.sigma_m_hx,
        )
        h_use_lossy_shell_y, h_lossy_shell_y = shell.infer_lossy_shell_slabs(
            field_shape=tuple(fields.Hy.shape),
            region=(slice(None), slice(None), slice(None)),
            conductivity_region=fields.sigma_m_hy,
        )
        h_use_lossy_shell_z, h_lossy_shell_z = shell.infer_lossy_shell_slabs(
            field_shape=tuple(fields.Hz.shape),
            region=(slice(None), slice(None), slice(None)),
            conductivity_region=fields.sigma_m_hz,
        )

        if enable_e_shell_split:
            e_use_lossy_shell_x = e_use_lossy_shell_x and (
                shell.lossy_fraction(tuple(fields.Ex.shape), fields.region_x, fields.sig_x)
                <= e_shell_frac_threshold
            )
            e_use_lossy_shell_y = e_use_lossy_shell_y and (
                shell.lossy_fraction(tuple(fields.Ey.shape), fields.region_y, fields.sig_y)
                <= e_shell_frac_threshold
            )
            e_use_lossy_shell_z = e_use_lossy_shell_z and (
                shell.lossy_fraction(tuple(fields.Ez.shape), fields.region_z, fields.sig_z)
                <= e_shell_frac_threshold
            )
        else:
            e_use_lossy_shell_x, e_use_lossy_shell_y, e_use_lossy_shell_z = False, False, False

        if enable_h_shell_split:
            h_use_lossy_shell_x = h_use_lossy_shell_x and (
                shell.lossy_fraction(
                    tuple(fields.Hx.shape),
                    (slice(None), slice(None), slice(None)),
                    fields.sigma_m_hx,
                )
                <= h_shell_frac_threshold
            )
            h_use_lossy_shell_y = h_use_lossy_shell_y and (
                shell.lossy_fraction(
                    tuple(fields.Hy.shape),
                    (slice(None), slice(None), slice(None)),
                    fields.sigma_m_hy,
                )
                <= h_shell_frac_threshold
            )
            h_use_lossy_shell_z = h_use_lossy_shell_z and (
                shell.lossy_fraction(
                    tuple(fields.Hz.shape),
                    (slice(None), slice(None), slice(None)),
                    fields.sigma_m_hz,
                )
                <= h_shell_frac_threshold
            )
        else:
            h_use_lossy_shell_x, h_use_lossy_shell_y, h_use_lossy_shell_z = False, False, False
    else:
        e_use_lossy_shell_x, e_lossy_shell_x = False, tuple()
        e_use_lossy_shell_y, e_lossy_shell_y = False, tuple()
        e_use_lossy_shell_z, e_lossy_shell_z = False, tuple()
        h_use_lossy_shell_x, h_lossy_shell_x = False, tuple()
        h_use_lossy_shell_y, h_lossy_shell_y = False, tuple()
        h_use_lossy_shell_z, h_lossy_shell_z = False, tuple()

    return compiled_cls(
        config=config,
        material_spec=CompiledMaterialSpec(model_kind="linear"),
        source_specs=source_specs,
        monitor_specs=monitor_specs,
        monitor_devices=monitor_devices,
        h_decay_x=h_decay_x,
        h_source_x=h_source_x,
        h_source_lossless_x=h_source_lossless_x,
        h_decay_y=h_decay_y,
        h_source_y=h_source_y,
        h_source_lossless_y=h_source_lossless_y,
        h_decay_z=h_decay_z,
        h_source_z=h_source_z,
        h_source_lossless_z=h_source_lossless_z,
        e_decay_x=e_decay_x,
        e_source_x=e_source_x,
        e_source_lossless_x=e_source_lossless_x,
        e_decay_y=e_decay_y,
        e_source_y=e_source_y,
        e_source_lossless_y=e_source_lossless_y,
        e_decay_z=e_decay_z,
        e_source_z=e_source_z,
        e_source_lossless_z=e_source_lossless_z,
        e_use_lossy_shell_x=e_use_lossy_shell_x,
        e_lossy_shell_x=e_lossy_shell_x,
        e_use_lossy_shell_y=e_use_lossy_shell_y,
        e_lossy_shell_y=e_lossy_shell_y,
        e_use_lossy_shell_z=e_use_lossy_shell_z,
        e_lossy_shell_z=e_lossy_shell_z,
        h_use_lossy_shell_x=h_use_lossy_shell_x,
        h_lossy_shell_x=h_lossy_shell_x,
        h_use_lossy_shell_y=h_use_lossy_shell_y,
        h_lossy_shell_y=h_lossy_shell_y,
        h_use_lossy_shell_z=h_use_lossy_shell_z,
        h_lossy_shell_z=h_lossy_shell_z,
    )
