import jax
import jax.numpy as jnp

from beamz.devices.monitors.compiler import compile_batched_monitor_data
from beamz.devices.sources.compiler import batch_slab_specs
from beamz.simulation import ops
from beamz.simulation.material_models import create_material_model


def build_scan(compiled):
    material_model = create_material_model(compiled.material_spec)
    material_state0 = material_model.init_state(compiled.material_spec)

    resolution = float(compiled.config.resolution)
    dt = float(compiled.config.dt)
    dt_scalar = jnp.asarray(dt, dtype=jnp.float32)
    plane_2d = compiled.config.plane_2d
    is_3d = compiled.config.is_3d

    pre_e_ex_batch, pre_e_ex_rest = batch_slab_specs(compiled._sources_for("pre_e", "Ex"))
    pre_e_ey_batch, pre_e_ey_rest = batch_slab_specs(compiled._sources_for("pre_e", "Ey"))
    pre_e_ez_batch, pre_e_ez_rest = batch_slab_specs(compiled._sources_for("pre_e", "Ez"))

    h_batch_x, h_rest_x = batch_slab_specs(compiled._sources_for("h", "Hx"))
    h_batch_y, h_rest_y = batch_slab_specs(compiled._sources_for("h", "Hy"))
    h_batch_z, h_rest_z = batch_slab_specs(compiled._sources_for("h", "Hz"))

    e_batch_x, e_rest_x = batch_slab_specs(compiled._sources_for("e", "Ex"))
    e_batch_y, e_rest_y = batch_slab_specs(compiled._sources_for("e", "Ey"))
    e_batch_z, e_rest_z = batch_slab_specs(compiled._sources_for("e", "Ez"))

    batched_mon = None
    monitors_2d = ()
    if compiled.monitor_specs and is_3d:
        has_dft_monitor = any(
            bool(getattr(spec, "dft_enabled", False)) for spec in compiled.monitor_specs
        )
        if has_dft_monitor:
            batched_mon = None
            monitors_2d = tuple(compiled.monitor_specs)
        else:
            field_shapes = {
                "Ex": tuple(compiled.e_source_x.shape),
                "Ey": tuple(compiled.e_source_y.shape),
                "Ez": tuple(compiled.e_source_z.shape),
                "Hx": tuple(compiled.h_source_x.shape),
                "Hy": tuple(compiled.h_source_y.shape),
                "Hz": tuple(compiled.h_source_z.shape),
            }
            batched_mon = compile_batched_monitor_data(compiled.monitor_specs, field_shapes)
            monitors_2d = tuple(spec for spec in compiled.monitor_specs if not spec.is_3d)
    elif compiled.monitor_specs:
        monitors_2d = tuple(compiled.monitor_specs)

    def run_scan(engine_state, monitor_state, coeffs):
        h_decay_x, h_source_x = coeffs.h_decay_x, coeffs.h_source_x
        h_source_lossless_x = coeffs.h_source_lossless_x
        h_decay_y, h_source_y = coeffs.h_decay_y, coeffs.h_source_y
        h_source_lossless_y = coeffs.h_source_lossless_y
        h_decay_z, h_source_z = coeffs.h_decay_z, coeffs.h_source_z
        h_source_lossless_z = coeffs.h_source_lossless_z
        e_decay_x, e_source_x = coeffs.e_decay_x, coeffs.e_source_x
        e_source_lossless_x = coeffs.e_source_lossless_x
        e_decay_y, e_source_y = coeffs.e_decay_y, coeffs.e_source_y
        e_source_lossless_y = coeffs.e_source_lossless_y
        e_decay_z, e_source_z = coeffs.e_decay_z, coeffs.e_source_z
        e_source_lossless_z = coeffs.e_source_lossless_z

        use_lossy_shell_ex = compiled.e_use_lossy_shell_x
        use_lossy_shell_ey = compiled.e_use_lossy_shell_y
        use_lossy_shell_ez = compiled.e_use_lossy_shell_z
        lossy_shell_ex = compiled.e_lossy_shell_x
        lossy_shell_ey = compiled.e_lossy_shell_y
        lossy_shell_ez = compiled.e_lossy_shell_z
        use_lossy_shell_hx = compiled.h_use_lossy_shell_x
        use_lossy_shell_hy = compiled.h_use_lossy_shell_y
        use_lossy_shell_hz = compiled.h_use_lossy_shell_z
        lossy_shell_hx = compiled.h_lossy_shell_x
        lossy_shell_hy = compiled.h_lossy_shell_y
        lossy_shell_hz = compiled.h_lossy_shell_z

        def body_with_coeffs(carry):
            eng, mon, mat = carry
            abs_step = eng.current_step

            ex, ey, ez = eng.ex, eng.ey, eng.ez
            hx, hy, hz = eng.hx, eng.hy, eng.hz

            ex = compiled._apply_source_group(ex, abs_step, pre_e_ex_batch, pre_e_ex_rest)
            ey = compiled._apply_source_group(ey, abs_step, pre_e_ey_batch, pre_e_ey_rest)
            ez = compiled._apply_source_group(ez, abs_step, pre_e_ez_batch, pre_e_ez_rest)

            if is_3d:
                any_h_shell = use_lossy_shell_hx or use_lossy_shell_hy or use_lossy_shell_hz
                if any_h_shell:
                    hx_old, hy_old, hz_old = hx, hy, hz
                    hx, hy, hz = ops.fused_update_h_lossless_3d(
                        ex,
                        ey,
                        ez,
                        hx,
                        hy,
                        hz,
                        h_source_lossless_x,
                        h_source_lossless_y,
                        h_source_lossless_z,
                        resolution,
                    )
                    if use_lossy_shell_hx:
                        hx = compiled._apply_lossy_shell_from_lossless(
                            updated_lossless=hx,
                            old=hx_old,
                            decay=h_decay_x,
                            source=h_source_x,
                            source_lossless=h_source_lossless_x,
                            slabs=lossy_shell_hx,
                        )
                    if use_lossy_shell_hy:
                        hy = compiled._apply_lossy_shell_from_lossless(
                            updated_lossless=hy,
                            old=hy_old,
                            decay=h_decay_y,
                            source=h_source_y,
                            source_lossless=h_source_lossless_y,
                            slabs=lossy_shell_hy,
                        )
                    if use_lossy_shell_hz:
                        hz = compiled._apply_lossy_shell_from_lossless(
                            updated_lossless=hz,
                            old=hz_old,
                            decay=h_decay_z,
                            source=h_source_z,
                            source_lossless=h_source_lossless_z,
                            slabs=lossy_shell_hz,
                        )
                else:
                    hx, hy, hz = ops.fused_update_h_lossy_3d(
                        ex,
                        ey,
                        ez,
                        hx,
                        hy,
                        hz,
                        h_decay_x,
                        h_source_x,
                        h_decay_y,
                        h_source_y,
                        h_decay_z,
                        h_source_z,
                        resolution,
                    )
            else:
                curl_ex, curl_ey, curl_ez = ops.curl_e_to_h_2d(
                    (ex, ey, ez),
                    resolution,
                    plane=plane_2d,
                )

                hx_old, hy_old, hz_old = hx, hy, hz

                if use_lossy_shell_hx:
                    hx = hx_old - h_source_lossless_x * curl_ex
                    hx = compiled._apply_lossy_shell(
                        updated=hx,
                        old=hx_old,
                        curl=curl_ex,
                        decay=h_decay_x,
                        source=-h_source_x,
                        slabs=lossy_shell_hx,
                    )
                else:
                    hx = h_decay_x * hx_old - h_source_x * curl_ex

                if use_lossy_shell_hy:
                    hy = hy_old - h_source_lossless_y * curl_ey
                    hy = compiled._apply_lossy_shell(
                        updated=hy,
                        old=hy_old,
                        curl=curl_ey,
                        decay=h_decay_y,
                        source=-h_source_y,
                        slabs=lossy_shell_hy,
                    )
                else:
                    hy = h_decay_y * hy_old - h_source_y * curl_ey

                if use_lossy_shell_hz:
                    hz = hz_old - h_source_lossless_z * curl_ez
                    hz = compiled._apply_lossy_shell(
                        updated=hz,
                        old=hz_old,
                        curl=curl_ez,
                        decay=h_decay_z,
                        source=-h_source_z,
                        slabs=lossy_shell_hz,
                    )
                else:
                    hz = h_decay_z * hz_old - h_source_z * curl_ez

            hx = compiled._apply_source_group(hx, abs_step, h_batch_x, h_rest_x)
            hy = compiled._apply_source_group(hy, abs_step, h_batch_y, h_rest_y)
            hz = compiled._apply_source_group(hz, abs_step, h_batch_z, h_rest_z)

            if is_3d:
                any_e_shell = use_lossy_shell_ex or use_lossy_shell_ey or use_lossy_shell_ez
                if any_e_shell:
                    ex_old, ey_old, ez_old = ex, ey, ez
                    ex, ey, ez = ops.fused_update_e_lossless_3d(
                        hx,
                        hy,
                        hz,
                        ex,
                        ey,
                        ez,
                        e_source_lossless_x,
                        e_source_lossless_y,
                        e_source_lossless_z,
                        resolution,
                    )
                    if use_lossy_shell_ex:
                        ex = compiled._apply_lossy_shell_from_lossless(
                            updated_lossless=ex,
                            old=ex_old,
                            decay=e_decay_x,
                            source=e_source_x,
                            source_lossless=e_source_lossless_x,
                            slabs=lossy_shell_ex,
                        )
                    if use_lossy_shell_ey:
                        ey = compiled._apply_lossy_shell_from_lossless(
                            updated_lossless=ey,
                            old=ey_old,
                            decay=e_decay_y,
                            source=e_source_y,
                            source_lossless=e_source_lossless_y,
                            slabs=lossy_shell_ey,
                        )
                    if use_lossy_shell_ez:
                        ez = compiled._apply_lossy_shell_from_lossless(
                            updated_lossless=ez,
                            old=ez_old,
                            decay=e_decay_z,
                            source=e_source_z,
                            source_lossless=e_source_lossless_z,
                            slabs=lossy_shell_ez,
                        )
                else:
                    ex, ey, ez = ops.fused_update_e_lossy_3d(
                        hx,
                        hy,
                        hz,
                        ex,
                        ey,
                        ez,
                        e_decay_x,
                        e_source_x,
                        e_decay_y,
                        e_source_y,
                        e_decay_z,
                        e_source_z,
                        resolution,
                    )
            else:
                curl_hx, curl_hy, curl_hz = ops.curl_h_to_e_2d(
                    (hx, hy, hz),
                    resolution,
                    (ex.shape, ey.shape, ez.shape),
                    plane=plane_2d,
                )

                ex_old, ey_old, ez_old = ex, ey, ez

                if use_lossy_shell_ex:
                    ex = ex_old + e_source_lossless_x * curl_hx
                    ex = compiled._apply_lossy_shell(
                        updated=ex,
                        old=ex_old,
                        curl=curl_hx,
                        decay=e_decay_x,
                        source=e_source_x,
                        slabs=lossy_shell_ex,
                    )
                else:
                    ex = e_decay_x * ex_old + e_source_x * curl_hx

                if use_lossy_shell_ey:
                    ey = ey_old + e_source_lossless_y * curl_hy
                    ey = compiled._apply_lossy_shell(
                        updated=ey,
                        old=ey_old,
                        curl=curl_hy,
                        decay=e_decay_y,
                        source=e_source_y,
                        slabs=lossy_shell_ey,
                    )
                else:
                    ey = e_decay_y * ey_old + e_source_y * curl_hy

                if use_lossy_shell_ez:
                    ez = ez_old + e_source_lossless_z * curl_hz
                    ez = compiled._apply_lossy_shell(
                        updated=ez,
                        old=ez_old,
                        curl=curl_hz,
                        decay=e_decay_z,
                        source=e_source_z,
                        slabs=lossy_shell_ez,
                    )
                else:
                    ez = e_decay_z * ez_old + e_source_z * curl_hz

            ex = compiled._apply_source_group(ex, abs_step, e_batch_x, e_rest_x)
            ey = compiled._apply_source_group(ey, abs_step, e_batch_y, e_rest_y)
            ez = compiled._apply_source_group(ez, abs_step, e_batch_z, e_rest_z)

            mat, _ = material_model.update(mat, ex, ey, ez, abs_step)

            mon = compiled._update_monitors(
                mon,
                abs_step,
                eng.t,
                dt_scalar,
                ex,
                ey,
                ez,
                hx,
                hy,
                hz,
                batched_mon=batched_mon,
                monitors_2d=monitors_2d,
            )

            new_eng = type(eng)(
                ex=ex,
                ey=ey,
                ez=ez,
                hx=hx,
                hy=hy,
                hz=hz,
                t=eng.t + dt,
                current_step=eng.current_step + jnp.array(1, dtype=jnp.int32),
            )
            return new_eng, mon, mat

        if compiled.config.loop_kind == "scan":

            def scan_body(carry, _unused):
                return body_with_coeffs(carry), None

            (engine_final, monitor_final, material_final), _ = jax.lax.scan(
                scan_body,
                (engine_state, monitor_state, material_state0),
                xs=None,
                length=compiled.config.num_steps,
            )
        else:
            init_carry = (engine_state, monitor_state, material_state0)
            engine_final, monitor_final, material_final = jax.lax.fori_loop(
                0,
                compiled.config.num_steps,
                lambda _idx, carry: body_with_coeffs(carry),
                init_carry,
            )
        return engine_final, monitor_final, material_final

    compiled._compiled_scan = jax.jit(run_scan, donate_argnums=(0, 1))
    compiled._compile_count += 1
