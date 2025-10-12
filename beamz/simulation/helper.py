import numpy as np
from beamz.devices.sources import ModeSource, GaussianSource


def _get_tfsf_boundary_info(source, fdtd):
    """Get TFSF boundary information for a mode source.
    
    Returns a dict with:
    - 'axis': propagation axis (0=x, 1=y, 2=z)
    - 'direction': +1 or -1 for forward/backward propagation
    - 'boundary_position': position of the TFSF boundary in grid coordinates
    - 'transverse_indices': list of (coord, index) tuples for transverse plane
    """
    from beamz.devices.sources import _direction_to_axis
    
    axis = _direction_to_axis(source.direction)
    direction_sign = 1 if source.direction.startswith("+") else -1
    
    # Get source center position
    center_x = (source.start[0] + source.end[0]) / 2
    center_y = (source.start[1] + source.end[1]) / 2
    center_z = (source.start[2] + source.end[2]) / 2 if fdtd.is_3d else 0
    
    # Convert to grid indices
    center_idx_x = int(round(center_x / fdtd.dx))
    center_idx_y = int(round(center_y / fdtd.dy))
    center_idx_z = int(round(center_z / fdtd.dz)) if fdtd.is_3d else 0
    
    return {
        'axis': axis,
        'direction': direction_sign,
        'center_idx': (center_idx_x, center_idx_y, center_idx_z),
        'source': source
    }


def apply_sources(fdtd) -> None:
    """Apply all sources for the current time step to fdtd fields using unidirectional TFSF."""
    for source in fdtd.sources:
        if isinstance(source, ModeSource):
            if not source.mode_profiles:
                source.compute_modes()

            mode_profile = source.mode_profiles[0]

            e_modulation = source.signal[min(fdtd.current_step, source.signal.size - 1)]

            if fdtd.current_step > 0:
                prev_idx = min(fdtd.current_step - 1, source.signal.size - 1)
                curr_idx = min(fdtd.current_step, source.signal.size - 1)
                h_modulation = 0.5 * (source.signal[prev_idx] + source.signal[curr_idx])
            else:
                h_modulation = source.signal[0]

            from beamz.devices.sources import _direction_to_axis

            scale_factor = 1.0
            for point in mode_profile:
                field_amplitudes = {}
                for field_name in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']:
                    amp = point.get(field_name, None)
                    if amp is not None:
                        field_amplitudes[field_name] = amp / scale_factor

                x_raw = point.get("x", source.center[0])
                y_raw = point.get("y", source.center[1])
                z_raw = point.get("z", source.center[2])

                x = int(round(x_raw / fdtd.dx))
                y = int(round(y_raw / fdtd.dy))
                
                if fdtd.is_3d:
                    z = int(round(z_raw / fdtd.dz))
                    if (x < 0 or x >= fdtd.nx or y < 0 or y >= fdtd.ny or z < 0 or z >= fdtd.nz):
                        continue
                    z_target = min(z, fdtd.Ez.shape[0] - 1) if z < fdtd.Ez.shape[0] else fdtd.Ez.shape[0] // 2
                    # Skip injection inside PML: identify PML by sigma > 0 around boundaries
                    try:
                        # 3D sigma aligned with Ez shape (nz-1, ny, nx)
                        sigma_slice = fdtd.sigma
                        if sigma_slice.ndim == 3:
                            sigma_here = float(sigma_slice[z_target, y, x])
                        else:
                            sigma_here = 0.0
                        if sigma_here > 0:
                            continue
                    except Exception:
                        pass
                    
                    # Hard source: dynamically inject all field components present
                    for field_name, amplitude in field_amplitudes.items():
                        if not hasattr(fdtd, field_name):
                            continue
                        
                        field_array = getattr(fdtd, field_name)
                        if field_array is None or field_array.size == 0:
                            continue
                        
                        # Use appropriate modulation (E-fields at integer steps, H-fields at half-steps)
                        modulation = e_modulation if field_name.startswith('E') else h_modulation
                        
                        # Check bounds and inject (soft source: add instead of overwrite)
                        if z_target < field_array.shape[0] and y < field_array.shape[1] and x < field_array.shape[2]:
                            field_array[z_target, y, x] += amplitude * modulation
                else:
                    # 2D case
                    if x < 0 or x >= fdtd.nx or y < 0 or y >= fdtd.ny:
                        continue
                    # Skip injection inside PML: sigma > 0 near boundaries
                    try:
                        sigma_here = float(fdtd.sigma[y, x]) if fdtd.sigma.ndim == 2 else 0.0
                        if sigma_here > 0:
                            continue
                    except Exception:
                        pass
                    
                    # Hard source for 2D: Inject E and H fields for unidirectional propagation
                    # Only inject components that exist in 2D TE (Ez, Hx, Hy)
                    # Skip Hz which doesn't exist in 2D
                    for field_name, amplitude in field_amplitudes.items():
                        # Skip Hz in 2D (doesn't exist in TE mode)
                        if field_name == 'Hz':
                            continue
                        
                        # Skip if amplitude is zero, NaN, or Inf
                        if amplitude == 0.0 or not np.isfinite(amplitude):
                            continue
                        
                        if not hasattr(fdtd, field_name):
                            continue
                        
                        field_array = getattr(fdtd, field_name)
                        if field_array is None or field_array.size == 0:
                            continue
                        
                        # Use appropriate modulation (E-fields at integer steps, H-fields at half-steps)
                        modulation = e_modulation if field_name.startswith('E') else h_modulation
                        
                        # Check bounds and inject (soft source: add instead of overwrite)
                        if y < field_array.shape[0] and x < field_array.shape[1]:
                            field_array[y, x] += amplitude * modulation

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
                fdtd.Ez[z_ez_idx:z_ez_idx + (z_end - z_start), y_start:y_end, x_start:x_end] += gaussian_amp[:(z_end - z_start), :, :] * modulation
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
                fdtd.Ez[y_start:y_end, x_start:x_end] += gaussian_amp * modulation

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
    should_save_full = (
        fdtd._save_results
        and not fdtd.save_memory_mode
        and (fdtd.current_step % fdtd._effective_save_freq == 0 or fdtd.current_step == fdtd.num_steps - 1)
    )

    cache_frequency = getattr(fdtd, "_cache_frequency", fdtd._effective_save_freq)
    should_cache = (
        fdtd._save_results
        and fdtd._cache_fields
        and (
            fdtd.current_step % cache_frequency == 0
            or fdtd.current_step == fdtd.num_steps - 1
        )
    )

    if not should_save_full and not should_cache:
        return

    if 't' not in fdtd.results:
        fdtd.results['t'] = []
    fdtd.results['t'].append(fdtd.t)

    fields_to_store = []
    if should_save_full:
        fields_to_store.extend(fdtd._save_fields)
    if should_cache:
        for field in fdtd._cache_fields:
            if field not in fields_to_store:
                fields_to_store.append(field)

    for field in fields_to_store:
        arr = getattr(fdtd, field)
        arr_np = fdtd.backend.to_numpy(fdtd.backend.copy(arr))
        if np.iscomplexobj(arr_np) and (field not in fdtd._cache_fields):
            arr_np = np.abs(arr_np)
        if field not in fdtd.results:
            fdtd.results[field] = []
        fdtd.results[field].append(arr_np)

def record_monitor_data(fdtd, step: int) -> None:
    """Record field data at monitor locations for current step."""
    if not fdtd.monitors:
        return
    if fdtd.is_3d:
        Ex_np = fdtd.backend.to_numpy(fdtd.Ex)
        Ey_np = fdtd.backend.to_numpy(fdtd.Ey)
        Ez_np = fdtd.backend.to_numpy(fdtd.Ez)
        Hx_np = fdtd.backend.to_numpy(fdtd.Hx)
        Hy_np = fdtd.backend.to_numpy(fdtd.Hy)
        Hz_np = fdtd.backend.to_numpy(fdtd.Hz)
        for monitor in fdtd.monitors:
            if hasattr(monitor, 'record_fields') and callable(monitor.record_fields):
                monitor.record_fields(Ex_np, Ey_np, Ez_np, Hx_np, Hy_np, Hz_np, fdtd.t, fdtd.dx, fdtd.dy, fdtd.dz, step=step)
    else:
        Ez_np = fdtd.backend.to_numpy(fdtd.Ez)
        Hx_np = fdtd.backend.to_numpy(fdtd.Hx)
        Hy_np = fdtd.backend.to_numpy(fdtd.Hy)
        for monitor in fdtd.monitors:
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
