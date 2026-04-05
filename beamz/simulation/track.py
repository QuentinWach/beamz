import jax
import jax.numpy as jnp
import numpy as np


def monitor_power_2d(compiled, spec, ez, hx, hy):
    del compiled
    power_scale = jnp.asarray(spec.power_scale, dtype=jnp.float32)
    ez_vals = ez[spec.y_ez, spec.x_ez] * spec.valid_ez
    hx_vals = hx[spec.y_hx, spec.x_hx] * spec.valid_hx
    hy_vals = hy[spec.y_hy, spec.x_hy] * spec.valid_hy

    sx = -ez_vals * hy_vals
    sy = ez_vals * hx_vals
    mag = jnp.sqrt(sx * sx + sy * sy)
    return jnp.asarray(jnp.sum(mag), dtype=jnp.float32) * power_scale


def monitor_power_3d(compiled, spec, ex, ey, ez, hx, hy, hz):
    del compiled
    power_scale = jnp.asarray(spec.power_scale, dtype=jnp.float32)
    exs = ex[spec.ex_idx][: spec.min_dim0, : spec.min_dim1]
    eys = ey[spec.ey_idx][: spec.min_dim0, : spec.min_dim1]
    ezs = ez[spec.ez_idx][: spec.min_dim0, : spec.min_dim1]
    hxs = hx[spec.hx_idx][: spec.min_dim0, : spec.min_dim1]
    hys = hy[spec.hy_idx][: spec.min_dim0, : spec.min_dim1]
    hzs = hz[spec.hz_idx][: spec.min_dim0, : spec.min_dim1]

    sx = eys * hzs - ezs * hys
    sy = ezs * hxs - exs * hzs
    sz = exs * hys - eys * hxs
    mag = jnp.sqrt(sx * sx + sy * sy + sz * sz)
    return jnp.asarray(jnp.sum(mag), dtype=jnp.float32) * power_scale


def update_monitors(
    compiled,
    monitor_state,
    abs_step,
    t_phys,
    dt_scalar,
    ex,
    ey,
    ez,
    hx,
    hy,
    hz,
    batched_mon=None,
    monitors_2d=(),
):
    if not compiled.monitor_specs:
        return monitor_state

    powers = monitor_state.powers
    timestamps = monitor_state.timestamps
    counts = monitor_state.counts
    freq_flux_re = monitor_state.freq_flux_re
    freq_flux_im = monitor_state.freq_flux_im
    freq_phase_re = monitor_state.freq_phase_re
    freq_phase_im = monitor_state.freq_phase_im
    dft_vec_re = monitor_state.dft_vec_re
    dft_vec_im = monitor_state.dft_vec_im
    dft_weight_sum = monitor_state.dft_weight_sum
    max_records = powers.shape[1]

    if batched_mon is not None:
        bm = batched_mon
        ex_flat = ex.ravel()
        ey_flat = ey.ravel()
        ez_flat = ez.ravel()
        hx_flat = hx.ravel()
        hy_flat = hy.ravel()
        hz_flat = hz.ravel()

        def mon_body(idx, carry):
            pwr, ts, cnt, f_re, f_im, ph_re, ph_im, d_re, d_im, d_w = carry
            mon_idx = bm.monitor_indices[idx]

            should_record = (abs_step % bm.record_intervals[idx]) == 0
            can_record = cnt[mon_idx] < max_records
            do_record = should_record & can_record & bm.accumulate_flags[idx]

            mask = bm.valid_mask[idx]
            exs = ex_flat[bm.ex_flat_idx[idx]] * mask
            eys = ey_flat[bm.ey_flat_idx[idx]] * mask
            ezs = ez_flat[bm.ez_flat_idx[idx]] * mask
            hxs = hx_flat[bm.hx_flat_idx[idx]] * mask
            hys = hy_flat[bm.hy_flat_idx[idx]] * mask
            hzs = hz_flat[bm.hz_flat_idx[idx]] * mask

            sx = eys * hzs - ezs * hys
            sy = ezs * hxs - exs * hzs
            sz = exs * hys - eys * hxs
            power_val = jnp.sum(jnp.sqrt(sx * sx + sy * sy + sz * sz)) * bm.power_scales[idx]
            axis_idx = bm.normal_axes[idx]
            normal_flux = (
                jnp.sum(jnp.where(axis_idx == 0, sx, jnp.where(axis_idx == 1, sy, sz)))
                * bm.power_scales[idx]
            )
            flux_sample = jnp.where(axis_idx < 0, power_val, normal_flux)

            slot = jnp.minimum(cnt[mon_idx], max_records - 1)
            pwr = pwr.at[mon_idx, slot].set(jnp.where(do_record, power_val, pwr[mon_idx, slot]))
            ts = ts.at[mon_idx, slot].set(jnp.where(do_record, t_phys, ts[mon_idx, slot]))
            cnt = cnt.at[mon_idx].set(cnt[mon_idx] + jnp.where(do_record, 1, 0))

            do_freq = bm.freq_enabled[idx] & ((abs_step % bm.freq_record_intervals[idx]) == 0)
            mask_f = bm.freq_mask[idx]
            row_f_re = f_re[mon_idx]
            row_f_im = f_im[mon_idx]
            row_ph_re = ph_re[mon_idx]
            row_ph_im = ph_im[mon_idx]
            cur_ph_re = row_ph_re
            cur_ph_im = row_ph_im
            delta_re = flux_sample * dt_scalar * row_ph_re * mask_f
            delta_im = flux_sample * dt_scalar * row_ph_im * mask_f
            row_f_re = row_f_re + jnp.where(do_freq, delta_re, 0.0)
            row_f_im = row_f_im + jnp.where(do_freq, delta_im, 0.0)
            rot_re = bm.freq_rot_re[idx]
            rot_im = bm.freq_rot_im[idx]
            next_ph_re = row_ph_re * rot_re - row_ph_im * rot_im
            next_ph_im = row_ph_re * rot_im + row_ph_im * rot_re
            row_ph_re = jnp.where(do_freq, next_ph_re, row_ph_re)
            row_ph_im = jnp.where(do_freq, next_ph_im, row_ph_im)
            f_re = f_re.at[mon_idx].set(row_f_re)
            f_im = f_im.at[mon_idx].set(row_f_im)
            ph_re = ph_re.at[mon_idx].set(row_ph_re)
            ph_im = ph_im.at[mon_idx].set(row_ph_im)

            do_dft = (
                bm.dft_enabled[idx]
                & ((abs_step % bm.dft_record_intervals[idx]) == 0)
                & (t_phys >= bm.dft_t_start[idx])
                & (t_phys <= bm.dft_t_end[idx])
            )
            two_pi = jnp.asarray(2.0 * np.pi, dtype=jnp.float32)
            span = jnp.maximum(bm.dft_t_end[idx] - bm.dft_t_start[idx], 1e-30)
            tau = jnp.clip((t_phys - bm.dft_t_start[idx]) / span, 0.0, 1.0)
            w_hann = 0.5 * (1.0 - jnp.cos(two_pi * tau))
            w = jnp.where(
                bm.dft_window_code[idx] == 1,
                w_hann,
                jnp.asarray(1.0, dtype=jnp.float32),
            )
            w = jnp.where(do_dft, w, jnp.asarray(0.0, dtype=jnp.float32))
            ph_vec_re = cur_ph_re * mask_f
            ph_vec_im = cur_ph_im * mask_f

            vecs = jnp.stack((exs, eys, ezs, hxs, hys, hzs), axis=0)
            comp_mask = bm.dft_component_mask[idx][:, None, None]
            delta_re_3d = w * comp_mask * jnp.einsum("f,cp->cfp", ph_vec_re, vecs)
            delta_im_3d = w * comp_mask * jnp.einsum("f,cp->cfp", ph_vec_im, vecs)
            d_re = d_re.at[mon_idx].add(delta_re_3d)
            d_im = d_im.at[mon_idx].add(delta_im_3d)
            d_w = d_w.at[mon_idx].add(w * mask_f)

            return pwr, ts, cnt, f_re, f_im, ph_re, ph_im, d_re, d_im, d_w

        (
            powers,
            timestamps,
            counts,
            freq_flux_re,
            freq_flux_im,
            freq_phase_re,
            freq_phase_im,
            dft_vec_re,
            dft_vec_im,
            dft_weight_sum,
        ) = jax.lax.fori_loop(
            0,
            bm.n_monitors,
            mon_body,
            (
                powers,
                timestamps,
                counts,
                freq_flux_re,
                freq_flux_im,
                freq_phase_re,
                freq_phase_im,
                dft_vec_re,
                dft_vec_im,
                dft_weight_sum,
            ),
        )

    for mon in monitors_2d:
        should_record = (abs_step % mon.record_interval) == 0
        can_record = counts[mon.monitor_index] < max_records
        do_record = should_record & can_record & mon.accumulate_power
        do_freq = (
            (abs_step % mon.freq_record_interval) == 0
            if mon.accumulate_frequency and mon.freq_count > 0
            else jnp.array(False)
        )
        need_sample = do_record | do_freq

        power_sample = jnp.where(
            need_sample,
            (
                compiled._monitor_power_3d(mon, ex, ey, ez, hx, hy, hz)
                if mon.is_3d
                else compiled._monitor_power_2d(mon, ez, hx, hy)
            ),
            jnp.array(0.0, dtype=jnp.float32),
        )
        power_val = jnp.where(do_record, power_sample, jnp.array(0.0, dtype=jnp.float32))

        slot = jnp.minimum(counts[mon.monitor_index], max_records - 1)
        old_power = powers[mon.monitor_index, slot]
        old_ts = timestamps[mon.monitor_index, slot]

        powers = powers.at[mon.monitor_index, slot].set(jnp.where(do_record, power_val, old_power))
        timestamps = timestamps.at[mon.monitor_index, slot].set(jnp.where(do_record, t_phys, old_ts))
        counts = counts.at[mon.monitor_index].set(
            counts[mon.monitor_index] + jnp.where(do_record, 1, 0)
        )
        if mon.accumulate_frequency and mon.freq_count > 0:
            mon_idx = mon.monitor_index
            row_f_re = freq_flux_re[mon_idx, : mon.freq_count]
            row_f_im = freq_flux_im[mon_idx, : mon.freq_count]
            row_ph_re = freq_phase_re[mon_idx, : mon.freq_count]
            row_ph_im = freq_phase_im[mon_idx, : mon.freq_count]
            cur_ph_re = row_ph_re
            cur_ph_im = row_ph_im
            delta_re = power_sample * dt_scalar * row_ph_re
            delta_im = power_sample * dt_scalar * row_ph_im
            row_f_re = row_f_re + jnp.where(do_freq, delta_re, 0.0)
            row_f_im = row_f_im + jnp.where(do_freq, delta_im, 0.0)
            next_ph_re = row_ph_re * mon.freq_rot_re - row_ph_im * mon.freq_rot_im
            next_ph_im = row_ph_re * mon.freq_rot_im + row_ph_im * mon.freq_rot_re
            row_ph_re = jnp.where(do_freq, next_ph_re, row_ph_re)
            row_ph_im = jnp.where(do_freq, next_ph_im, row_ph_im)
            freq_flux_re = freq_flux_re.at[mon_idx, : mon.freq_count].set(row_f_re)
            freq_flux_im = freq_flux_im.at[mon_idx, : mon.freq_count].set(row_f_im)
            freq_phase_re = freq_phase_re.at[mon_idx, : mon.freq_count].set(row_ph_re)
            freq_phase_im = freq_phase_im.at[mon_idx, : mon.freq_count].set(row_ph_im)
        if mon.dft_enabled and mon.freq_count > 0 and mon.dft_point_count > 0:
            mon_idx = mon.monitor_index
            if mon.accumulate_frequency and mon.freq_count > 0:
                dft_ph_re = cur_ph_re
                dft_ph_im = cur_ph_im
            else:
                dft_ph_re = freq_phase_re[mon_idx, : mon.freq_count]
                dft_ph_im = freq_phase_im[mon_idx, : mon.freq_count]
            do_dft = (
                ((abs_step % mon.dft_record_interval) == 0)
                & (t_phys >= mon.dft_t_start)
                & (t_phys <= mon.dft_t_end)
            )
            two_pi = jnp.asarray(2.0 * np.pi, dtype=jnp.float32)
            span = jnp.maximum(
                mon.dft_t_end - mon.dft_t_start,
                jnp.asarray(1e-30, dtype=jnp.float32),
            )
            tau = jnp.asarray(
                jnp.clip((t_phys - mon.dft_t_start) / span, 0.0, 1.0),
                dtype=jnp.float32,
            )
            w_hann = 0.5 * (1.0 - jnp.cos(two_pi * tau))
            w = jnp.asarray(
                jnp.where(
                    mon.dft_window_code == 1,
                    w_hann,
                    jnp.asarray(1.0, dtype=jnp.float32),
                ),
                dtype=jnp.float32,
            )
            w = jnp.asarray(
                jnp.where(do_dft, w, jnp.asarray(0.0, dtype=jnp.float32)),
                dtype=jnp.float32,
            )

            if mon.is_3d:
                ex_vals = ex[mon.ex_idx][: mon.min_dim0, : mon.min_dim1].reshape(-1)
                ey_vals = ey[mon.ey_idx][: mon.min_dim0, : mon.min_dim1].reshape(-1)
                ez_vals = ez[mon.ez_idx][: mon.min_dim0, : mon.min_dim1].reshape(-1)
                hx_vals = hx[mon.hx_idx][: mon.min_dim0, : mon.min_dim1].reshape(-1)
                hy_vals = hy[mon.hy_idx][: mon.min_dim0, : mon.min_dim1].reshape(-1)
                hz_vals = hz[mon.hz_idx][: mon.min_dim0, : mon.min_dim1].reshape(-1)
            else:
                ex_vals = ex[mon.y_ex, mon.x_ex] * mon.valid_ex
                ey_vals = ey[mon.y_ey, mon.x_ey] * mon.valid_ey
                ez_vals = ez[mon.y_ez, mon.x_ez] * mon.valid_ez
                hx_vals = hx[mon.y_hx, mon.x_hx] * mon.valid_hx
                hy_vals = hy[mon.y_hy, mon.x_hy] * mon.valid_hy
                hz_vals = hz[mon.y_hz, mon.x_hz] * mon.valid_hz
            vecs = jnp.stack((ex_vals, ey_vals, ez_vals, hx_vals, hy_vals, hz_vals), axis=0)
            comp_mask = mon.dft_component_mask[:, None, None]
            delta_re = jnp.asarray(
                w * comp_mask * jnp.einsum("f,cp->cfp", dft_ph_re, vecs),
                dtype=jnp.float32,
            )
            delta_im = jnp.asarray(
                w * comp_mask * jnp.einsum("f,cp->cfp", dft_ph_im, vecs),
                dtype=jnp.float32,
            )
            dft_vec_re = dft_vec_re.at[
                mon_idx, :, : mon.freq_count, : mon.dft_point_count
            ].add(delta_re[:, : mon.freq_count, : mon.dft_point_count])
            dft_vec_im = dft_vec_im.at[
                mon_idx, :, : mon.freq_count, : mon.dft_point_count
            ].add(delta_im[:, : mon.freq_count, : mon.dft_point_count])
            dft_weight_sum = dft_weight_sum.at[mon_idx, : mon.freq_count].add(
                jnp.asarray(w, dtype=jnp.float32)
            )

    return type(monitor_state)(
        powers=powers,
        timestamps=timestamps,
        counts=counts,
        freq_flux_re=freq_flux_re,
        freq_flux_im=freq_flux_im,
        freq_phase_re=freq_phase_re,
        freq_phase_im=freq_phase_im,
        dft_vec_re=dft_vec_re,
        dft_vec_im=dft_vec_im,
        dft_weight_sum=dft_weight_sum,
    )


def monitor_state_size(specs, num_steps: int) -> int:
    if not specs:
        return 0
    return int(
        max(int(np.ceil(num_steps / max(1, int(spec.record_interval)))) for spec in specs)
    )


def monitor_frequency_size(specs) -> int:
    if not specs:
        return 0
    return int(max(int(spec.freq_count) for spec in specs))


def monitor_dft_point_size(specs) -> int:
    if not specs:
        return 0
    return int(max(int(getattr(spec, "dft_point_count", 0)) for spec in specs))


def apply_monitor_state(compiled, monitor_state):
    """Push monitor-state buffers back to Monitor objects."""
    for spec in compiled.monitor_specs:
        dev = compiled.monitor_devices[spec.monitor_index]
        count = int(np.asarray(monitor_state.counts[spec.monitor_index]))
        powers = np.asarray(monitor_state.powers[spec.monitor_index, :count], dtype=float)
        ts = np.asarray(monitor_state.timestamps[spec.monitor_index, :count], dtype=float)

        dev.power_history = list(powers.tolist())
        dev.power_timestamps = list(ts.tolist())
        dev.power_accumulation_count = count
        if spec.freq_count > 0:
            re = np.asarray(
                monitor_state.freq_flux_re[spec.monitor_index, : spec.freq_count],
                dtype=np.float32,
            )
            im = np.asarray(
                monitor_state.freq_flux_im[spec.monitor_index, : spec.freq_count],
                dtype=np.float32,
            )
            dev.frequency_flux_spectrum = (re + 1j * im).astype(np.complex64)
        else:
            dev.frequency_flux_spectrum = np.zeros((0,), dtype=np.complex64)

        if spec.dft_enabled and spec.freq_count > 0 and spec.dft_point_count > 0:
            comp_names = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
            comp_mask = (
                np.asarray(spec.dft_component_mask, dtype=np.float32)
                if spec.dft_component_mask is not None
                else np.ones((6,), dtype=np.float32)
            )
            weight_sum = np.asarray(
                monitor_state.dft_weight_sum[spec.monitor_index, : spec.freq_count],
                dtype=np.float64,
            )
            dev._dft_weight_sum = weight_sum
            dev._dft_accum = {}
            for comp_idx, comp_name in enumerate(comp_names):
                if comp_mask[comp_idx] <= 0.0:
                    continue
                re = np.asarray(
                    monitor_state.dft_vec_re[
                        spec.monitor_index,
                        comp_idx,
                        : spec.freq_count,
                        : spec.dft_point_count,
                    ],
                    dtype=np.float64,
                )
                im = np.asarray(
                    monitor_state.dft_vec_im[
                        spec.monitor_index,
                        comp_idx,
                        : spec.freq_count,
                        : spec.dft_point_count,
                    ],
                    dtype=np.float64,
                )
                dev._dft_accum[comp_name] = re + 1j * im
        else:
            dev._dft_weight_sum = np.zeros((0,), dtype=np.float64)
            dev._dft_accum = {}
