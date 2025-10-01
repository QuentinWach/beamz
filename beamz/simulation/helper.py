import numpy as np
from beamz.design.sources import ModeSource, GaussianSource


def apply_sources(fdtd) -> None:
    """Apply all sources for the current time step to fdtd fields."""
    backend_name = getattr(fdtd.backend, "backend_name", "numpy")
    is_jax_backend = backend_name == "jax"

    for source in fdtd.sources:
        if isinstance(source, ModeSource):
            mode_profile = source.mode_profiles[0]
            modulation = source.signal[fdtd.current_step]
            for point in mode_profile:
                if len(point) == 4:
                    amplitude, x_raw, y_raw, z_raw = point
                else:
                    amplitude, x_raw, y_raw = point
                    z_raw = 0
                x = int(round(x_raw / fdtd.dx))
                y = int(round(y_raw / fdtd.dy))
                if fdtd.is_3d:
                    z = int(round(z_raw / fdtd.dz))
                    if (x < 0 or x >= fdtd.nx or y < 0 or y >= fdtd.ny or z < 0 or z >= fdtd.nz):
                        continue
                    z_target = min(z, fdtd.Ez.shape[0] - 1) if z < fdtd.Ez.shape[0] else fdtd.Ez.shape[0] // 2
                else:
                    if x < 0 or x >= fdtd.nx or y < 0 or y >= fdtd.ny:
                        continue
                    z_target = None

                if hasattr(fdtd, 'is_complex_backend') and not fdtd.is_complex_backend:
                    if isinstance(amplitude * modulation, complex):
                        source_value = np.real(amplitude * modulation)
                    else:
                        source_value = amplitude * modulation
                else:
                    source_value = amplitude * modulation

                if fdtd.is_3d:
                    value = fdtd.Ez.dtype.type(source_value)
                    if is_jax_backend:
                        fdtd.Ez = fdtd.Ez.at[z_target, y, x].add(value)
                    else:
                        fdtd.Ez[z_target, y, x] += value

                    zero_val = fdtd.Ez.dtype.type(0)
                    if source.direction == "+x" and x > 0:
                        if is_jax_backend:
                            fdtd.Ez = fdtd.Ez.at[z_target, y, x-1].set(zero_val)
                        else:
                            fdtd.Ez[z_target, y, x-1] = zero_val
                    elif source.direction == "-x" and x < fdtd.nx-1:
                        if is_jax_backend:
                            fdtd.Ez = fdtd.Ez.at[z_target, y, x+1].set(zero_val)
                        else:
                            fdtd.Ez[z_target, y, x+1] = zero_val
                    elif source.direction == "+y" and y > 0:
                        if is_jax_backend:
                            fdtd.Ez = fdtd.Ez.at[z_target, y-1, x].set(zero_val)
                        else:
                            fdtd.Ez[z_target, y-1, x] = zero_val
                    elif source.direction == "-y" and y < fdtd.ny-1:
                        if is_jax_backend:
                            fdtd.Ez = fdtd.Ez.at[z_target, y+1, x].set(zero_val)
                        else:
                            fdtd.Ez[z_target, y+1, x] = zero_val
                    elif source.direction == "+z" and z_target > 0:
                        if is_jax_backend:
                            fdtd.Ez = fdtd.Ez.at[z_target-1, y, x].set(zero_val)
                        else:
                            fdtd.Ez[z_target-1, y, x] = zero_val
                    elif source.direction == "-z" and z_target < fdtd.Ez.shape[0]-1:
                        if is_jax_backend:
                            fdtd.Ez = fdtd.Ez.at[z_target+1, y, x].set(zero_val)
                        else:
                            fdtd.Ez[z_target+1, y, x] = zero_val
                else:
                    value = fdtd.Ez.dtype.type(source_value)
                    if is_jax_backend:
                        fdtd.Ez = fdtd.Ez.at[y, x].add(value)
                    else:
                        fdtd.Ez[y, x] += value

                    zero_val = fdtd.Ez.dtype.type(0)
                    if source.direction == "+x" and x > 0:
                        if is_jax_backend:
                            fdtd.Ez = fdtd.Ez.at[y, x-1].set(zero_val)
                        else:
                            fdtd.Ez[y, x-1] = zero_val
                    elif source.direction == "-x" and x < fdtd.nx-1:
                        if is_jax_backend:
                            fdtd.Ez = fdtd.Ez.at[y, x+1].set(zero_val)
                        else:
                            fdtd.Ez[y, x+1] = zero_val
                    elif source.direction == "+y" and y > 0:
                        if is_jax_backend:
                            fdtd.Ez = fdtd.Ez.at[y-1, x].set(zero_val)
                        else:
                            fdtd.Ez[y-1, x] = zero_val
                    elif source.direction == "-y" and y < fdtd.ny-1:
                        if is_jax_backend:
                            fdtd.Ez = fdtd.Ez.at[y+1, x].set(zero_val)
                        else:
                            fdtd.Ez[y+1, x] = zero_val

        elif isinstance(source, GaussianSource):
            modulation = source.signal[fdtd.current_step]
            center_x_phys, center_y_phys, center_z_phys = source.position
            width_phys = source.width
            center_x_grid = center_x_phys / fdtd.dx
            center_y_grid = center_y_phys / fdtd.dy
            if fdtd.is_3d:
                center_z_grid = center_z_phys / fdtd.dz
                width_x_grid = width_phys / fdtd.dx
                width_y_grid = width_phys / fdtd.dy
                width_z_grid = width_phys / fdtd.dz
                wx_grid_cells = max(1, int(round(3 * width_x_grid)))
                wy_grid_cells = max(1, int(round(3 * width_y_grid)))
                wz_grid_cells = max(1, int(round(3 * width_z_grid)))
                x_center_idx = int(round(center_x_grid))
                y_center_idx = int(round(center_y_grid))
                z_center_idx = int(round(center_z_grid))
                x_start = max(0, x_center_idx - wx_grid_cells)
                x_end = min(fdtd.nx, x_center_idx + wx_grid_cells + 1)
                y_start = max(0, y_center_idx - wy_grid_cells)
                y_end = min(fdtd.ny, y_center_idx + wy_grid_cells + 1)
                z_start = max(0, z_center_idx - wz_grid_cells)
                z_end = min(fdtd.nz, z_center_idx + wz_grid_cells + 1)
                z_end = min(z_end, fdtd.Ez.shape[0])
                z_indices = np.arange(z_start, z_end)
                y_indices = np.arange(y_start, y_end)
                x_indices = np.arange(x_start, x_end)
                z_grid, y_grid, x_grid = np.meshgrid(z_indices, y_indices, x_indices, indexing='ij')
                dist_x_sq = (x_grid - center_x_grid)**2
                dist_y_sq = (y_grid - center_y_grid)**2
                dist_z_sq = (z_grid - center_z_grid)**2
                epsilon = 1e-9
                sigma_x_sq = width_x_grid**2 + epsilon
                sigma_y_sq = width_y_grid**2 + epsilon
                sigma_z_sq = width_z_grid**2 + epsilon
                exponent = -(dist_x_sq / (2 * sigma_x_sq) + dist_y_sq / (2 * sigma_y_sq) + dist_z_sq / (2 * sigma_z_sq))
                gaussian_amp = np.exp(exponent)
                gaussian_amp = fdtd.backend.from_numpy(gaussian_amp)
                z_ez_idx = max(0, min(fdtd.Ez.shape[0]-1, z_start))
                z_slice = slice(z_ez_idx, z_ez_idx + (z_end - z_start))
                y_slice = slice(y_start, y_end)
                x_slice = slice(x_start, x_end)
                increment = (gaussian_amp[:(z_end - z_start), :, :] * modulation).astype(fdtd.Ez.dtype)
                if is_jax_backend:
                    fdtd.Ez = fdtd.Ez.at[z_slice, y_slice, x_slice].add(increment)
                else:
                    fdtd.Ez[z_slice, y_slice, x_slice] += increment
            else:
                width_x_grid = width_phys / fdtd.dx
                width_y_grid = width_phys / fdtd.dy
                wx_grid_cells = max(1, int(round(3 * width_x_grid)))
                wy_grid_cells = max(1, int(round(3 * width_y_grid)))
                x_center_idx = int(round(center_x_grid))
                y_center_idx = int(round(center_y_grid))
                x_start = max(0, x_center_idx - wx_grid_cells)
                x_end = min(fdtd.nx, x_center_idx + wx_grid_cells + 1)
                y_start = max(0, y_center_idx - wy_grid_cells)
                y_end = min(fdtd.ny, y_center_idx + wy_grid_cells + 1)
                y_indices = np.arange(y_start, y_end)
                x_indices = np.arange(x_start, x_end)
                y_grid, x_grid = np.meshgrid(y_indices, x_indices, indexing='ij')
                dist_x_sq = (x_grid - center_x_grid)**2
                dist_y_sq = (y_grid - center_y_grid)**2
                epsilon = 1e-9
                sigma_x_sq = width_x_grid**2 + epsilon
                sigma_y_sq = width_y_grid**2 + epsilon
                exponent = -(dist_x_sq / (2 * sigma_x_sq) + dist_y_sq / (2 * sigma_y_sq))
                gaussian_amp = np.exp(exponent) / 4
                gaussian_amp = fdtd.backend.from_numpy(gaussian_amp)
                increment = (gaussian_amp * modulation).astype(fdtd.Ez.dtype)
                if is_jax_backend:
                    fdtd.Ez = fdtd.Ez.at[y_start:y_end, x_start:x_end].add(increment)
                else:
                    fdtd.Ez[y_start:y_end, x_start:x_end] += increment

def accumulate_power(fdtd) -> None:
    """Accumulate power for current step if requested (updates fdtd.power_accumulated)."""
    if not fdtd.accumulate_power:
        return
    if fdtd.is_3d:
        Ex_np = fdtd.backend.to_numpy(fdtd.Ex)
        Ey_np = fdtd.backend.to_numpy(fdtd.Ey)
        Ez_np = fdtd.backend.to_numpy(fdtd.Ez)
        Hx_np = fdtd.backend.to_numpy(fdtd.Hx)
        Hy_np = fdtd.backend.to_numpy(fdtd.Hy)
        Hz_np = fdtd.backend.to_numpy(fdtd.Hz)
        min_z = min(Ex_np.shape[0], Ey_np.shape[0], Ez_np.shape[0], Hx_np.shape[0], Hy_np.shape[0], Hz_np.shape[0])
        min_y = min(Ex_np.shape[1], Ey_np.shape[1], Ez_np.shape[1], Hx_np.shape[1], Hy_np.shape[1], Hz_np.shape[1])
        min_x = min(Ex_np.shape[2], Ey_np.shape[2], Ez_np.shape[2], Hx_np.shape[2], Hy_np.shape[2], Hz_np.shape[2])
        Ex_center = Ex_np[:min_z, :min_y, :min_x]
        Ey_center = Ey_np[:min_z, :min_y, :min_x]
        Ez_center = Ez_np[:min_z, :min_y, :min_x]
        Hx_center = Hx_np[:min_z, :min_y, :min_x]
        Hy_center = Hy_np[:min_z, :min_y, :min_x]
        Hz_center = Hz_np[:min_z, :min_y, :min_x]
        Sx = np.real(Ey_center * np.conj(Hz_center) - Ez_center * np.conj(Hy_center))
        Sy = np.real(Ez_center * np.conj(Hx_center) - Ex_center * np.conj(Hz_center))
        Sz = np.real(Ex_center * np.conj(Hy_center) - Ey_center * np.conj(Hx_center))
        power_mag = np.sqrt(Sx**2 + Sy**2 + Sz**2)
        if fdtd.power_accumulated is None:
            fdtd.power_accumulated = power_mag.copy()
        else:
            if fdtd.power_accumulated.shape != power_mag.shape:
                fdtd.power_accumulated = power_mag.copy()
                fdtd.power_accumulation_count = 0
            fdtd.power_accumulated += power_mag
        fdtd.power_accumulation_count += 1
    else:
        Ez_np = fdtd.backend.to_numpy(fdtd.Ez)
        Hx_np = fdtd.backend.to_numpy(fdtd.Hx)
        Hy_np = fdtd.backend.to_numpy(fdtd.Hy)
        is_complex = np.iscomplexobj(Ez_np) or np.iscomplexobj(Hx_np) or np.iscomplexobj(Hy_np)
        if np.iscomplexobj(Ez_np):
            Ez_real = np.real(Ez_np)
            Ez_imag = np.imag(Ez_np)
        else:
            Ez_real = Ez_np
            Ez_imag = np.zeros_like(Ez_np)
        if is_complex:
            Hx_full = np.zeros_like(Ez_np, dtype=np.complex128)
            Hy_full = np.zeros_like(Ez_np, dtype=np.complex128)
        else:
            Hx_full = np.zeros_like(Ez_real)
            Hy_full = np.zeros_like(Ez_real)
        Hx_full[:, :-1] = Hx_np
        Hy_full[:-1, :] = Hy_np
        if is_complex:
            Hx_real = np.real(Hx_full); Hx_imag = np.imag(Hx_full)
            Hy_real = np.real(Hy_full); Hy_imag = np.imag(Hy_full)
            Sx = -Ez_real * Hy_real - Ez_imag * Hy_imag
            Sy = Ez_real * Hx_real + Ez_imag * Hx_imag
        else:
            Sx = -Ez_real * Hy_full
            Sy = Ez_real * Hx_full
        power_mag = Sx**2 + Sy**2
        if fdtd.power_accumulated is None:
            fdtd.power_accumulated = power_mag.copy()
        else:
            if fdtd.power_accumulated.shape != power_mag.shape:
                fdtd.power_accumulated = power_mag.copy()
                fdtd.power_accumulation_count = 0
            fdtd.power_accumulated += power_mag
        fdtd.power_accumulation_count += 1

def save_step_results(fdtd) -> None:
    """Save results for this time step if requested and at the right frequency."""
    if (fdtd._save_results and not fdtd.save_memory_mode and 
        (fdtd.current_step % fdtd._effective_save_freq == 0 or fdtd.current_step == fdtd.num_steps - 1)):
        fdtd.results['t'].append(fdtd.t)
        for field in fdtd._save_fields:
            arr = getattr(fdtd, field)
            arr_np = fdtd.backend.to_numpy(fdtd.backend.copy(arr))
            if np.iscomplexobj(arr_np):
                arr_np = np.abs(arr_np)
            fdtd.results[field].append(arr_np)

def record_monitor_data(fdtd, step: int) -> None:
    """Record field data at monitor locations for current step."""
    if not fdtd.design.monitors:
        return
    if fdtd.is_3d:
        Ex_np = fdtd.backend.to_numpy(fdtd.Ex)
        Ey_np = fdtd.backend.to_numpy(fdtd.Ey)
        Ez_np = fdtd.backend.to_numpy(fdtd.Ez)
        Hx_np = fdtd.backend.to_numpy(fdtd.Hx)
        Hy_np = fdtd.backend.to_numpy(fdtd.Hy)
        Hz_np = fdtd.backend.to_numpy(fdtd.Hz)
        for monitor in fdtd.design.monitors:
            if hasattr(monitor, 'record_fields') and callable(monitor.record_fields):
                monitor.record_fields(Ex_np, Ey_np, Ez_np, Hx_np, Hy_np, Hz_np, fdtd.t, fdtd.dx, fdtd.dy, fdtd.dz, step=step)
    else:
        Ez_np = fdtd.backend.to_numpy(fdtd.Ez)
        Hx_np = fdtd.backend.to_numpy(fdtd.Hx)
        Hy_np = fdtd.backend.to_numpy(fdtd.Hy)
        for monitor in fdtd.design.monitors:
            if hasattr(monitor, 'record_fields') and callable(monitor.record_fields):
                if hasattr(monitor, 'is_3d'):
                    original_is_3d = monitor.is_3d
                    monitor.is_3d = False
                monitor.record_fields(Ez_np, Hx_np, Hy_np, fdtd.t, fdtd.dx, fdtd.dy, step=step)
                if hasattr(monitor, 'is_3d'):
                    monitor.is_3d = original_is_3d

def estimate_memory_usage(fdtd, time_steps=None, save_fields=None):
    """Estimate memory usage of the simulation with current settings (returns dict)."""
    if time_steps is None:
        time_steps = fdtd.num_steps
    if save_fields is None:
        save_fields = ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'] if fdtd.is_3d else ['Ez', 'Hx', 'Hy']
    bytes_per_value = np.float64(0).nbytes
    field_sizes = {}
    if fdtd.is_3d:
        field_sizes['Ex'] = (fdtd.nz * fdtd.ny * (fdtd.nx-1)) * bytes_per_value
        field_sizes['Ey'] = (fdtd.nz * (fdtd.ny-1) * fdtd.nx) * bytes_per_value
        field_sizes['Ez'] = ((fdtd.nz-1) * fdtd.ny * fdtd.nx) * bytes_per_value
        field_sizes['Hx'] = ((fdtd.nz-1) * (fdtd.ny-1) * fdtd.nx) * bytes_per_value
        field_sizes['Hy'] = ((fdtd.nz-1) * fdtd.ny * (fdtd.nx-1)) * bytes_per_value
        field_sizes['Hz'] = (fdtd.nz * (fdtd.ny-1) * (fdtd.nx-1)) * bytes_per_value
    else:
        field_sizes['Ez'] = fdtd.nx * fdtd.ny * bytes_per_value
        field_sizes['Hx'] = fdtd.nx * (fdtd.ny-1) * bytes_per_value
        field_sizes['Hy'] = (fdtd.nx-1) * fdtd.ny * bytes_per_value
    t_size = time_steps * bytes_per_value
    total_size = t_size
    single_step_size = 0
    for field in save_fields:
        if field in field_sizes:
            field_size = field_sizes[field]
            total_size += field_size * time_steps
            single_step_size += field_size
    kb = 1024
    mb = kb * 1024
    gb = mb * 1024
    result = {
        'Single timestep': {
            **{field: field_sizes.get(field, 0) / mb for field in save_fields},
            'Total': single_step_size / mb
        },
        'Full simulation': {
            'Total memory (MB)': total_size / mb,
            'Total memory (GB)': total_size / gb,
            'Time steps': time_steps,
            'Grid size': f"{fdtd.nx} x {fdtd.ny}" + (f" x {fdtd.nz}" if fdtd.is_3d else ""),
            'Fields saved': ', '.join(save_fields),
            'Dimensionality': '3D' if fdtd.is_3d else '2D'
        }
    }
    return result
