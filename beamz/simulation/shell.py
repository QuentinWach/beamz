import numpy as np


def apply_lossy_shell(
    compiled,
    updated,
    old,
    curl,
    decay,
    source,
    slabs,
):
    """Apply lossy E/H update only on precomputed shell slabs."""
    del compiled
    out = updated
    for starts, sizes in slabs:
        old_s = compiled_jax_dynamic_slice(old, starts, sizes)
        curl_s = compiled_jax_dynamic_slice(curl, starts, sizes)
        decay_s = compiled_jax_dynamic_slice(decay, starts, sizes)
        source_s = compiled_jax_dynamic_slice(source, starts, sizes)
        lossy_s = decay_s * old_s + source_s * curl_s
        out = compiled_jax_dynamic_update_slice(out, lossy_s, starts)
    return out


def apply_lossy_shell_from_lossless(
    compiled,
    updated_lossless,
    old,
    decay,
    source,
    source_lossless,
    slabs,
):
    """Apply lossy correction on shell slabs using lossless updates."""
    del compiled
    out = updated_lossless
    for starts, sizes in slabs:
        old_s = compiled_jax_dynamic_slice(old, starts, sizes)
        lossless_s = compiled_jax_dynamic_slice(updated_lossless, starts, sizes)
        decay_s = compiled_jax_dynamic_slice(decay, starts, sizes)
        source_s = compiled_jax_dynamic_slice(source, starts, sizes)
        source_ll_s = compiled_jax_dynamic_slice(source_lossless, starts, sizes)
        beta = source_s / source_ll_s
        lossy_s = (decay_s - beta) * old_s + beta * lossless_s
        out = compiled_jax_dynamic_update_slice(out, lossy_s, starts)
    return out


def edge_full_thickness(mask: np.ndarray, axis: int) -> tuple[int, int]:
    """Count leading and trailing planes that are fully lossy."""
    other_axes = tuple(i for i in range(mask.ndim) if i != axis)
    plane_all = mask.all(axis=other_axes)

    left = 0
    size = plane_all.shape[0]
    while left < size and bool(plane_all[left]):
        left += 1

    right = 0
    while right < (size - left) and bool(plane_all[size - 1 - right]):
        right += 1
    return left, right


def region_offsets_and_sizes(
    field_shape: tuple[int, ...],
    region: tuple[slice, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """Return region starts and sizes for slice-only regions with unit strides."""
    if len(field_shape) != len(region):
        return None

    starts: list[int] = []
    sizes: list[int] = []
    for dim, key in zip(field_shape, region):
        if not isinstance(key, slice):
            return None
        start, stop, step = key.indices(dim)
        if step != 1:
            return None
        starts.append(int(start))
        sizes.append(int(max(stop - start, 0)))
    return tuple(starts), tuple(sizes)


def infer_lossy_shell_slabs(
    field_shape: tuple[int, ...],
    region: tuple[slice, ...],
    conductivity_region,
) -> tuple[bool, tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]]:
    """Infer disjoint boundary-shell slabs from a conductivity mask."""
    if len(field_shape) != 3:
        return False, tuple()

    region_layout = region_offsets_and_sizes(field_shape, region)
    if region_layout is None:
        return False, tuple()
    region_starts, region_sizes = region_layout
    if len(region_sizes) != 3 or any(size <= 0 for size in region_sizes):
        return False, tuple()

    local_mask = np.asarray(conductivity_region) > 0.0
    if tuple(local_mask.shape) != tuple(region_sizes):
        return False, tuple()
    if not local_mask.any():
        return False, tuple()

    z_left, z_right = edge_full_thickness(local_mask, axis=0)
    y_left, y_right = edge_full_thickness(local_mask, axis=1)
    x_left, x_right = edge_full_thickness(local_mask, axis=2)

    nz, ny, nx = region_sizes
    z0, z1 = z_left, nz - z_right
    y0, y1 = y_left, ny - y_right
    x0, x1 = x_left, nx - x_right
    if z0 > z1 or y0 > y1 or x0 > x1:
        return False, tuple()

    slabs: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []

    def add_slab(starts: tuple[int, int, int], sizes: tuple[int, int, int]):
        if all(size > 0 for size in sizes):
            slabs.append((starts, sizes))

    add_slab((0, 0, 0), (z_left, ny, nx))
    add_slab((z1, 0, 0), (z_right, ny, nx))
    add_slab((z0, 0, 0), (max(z1 - z0, 0), y_left, nx))
    add_slab((z0, y1, 0), (max(z1 - z0, 0), y_right, nx))
    add_slab((z0, y0, 0), (max(z1 - z0, 0), max(y1 - y0, 0), x_left))
    add_slab((z0, y0, x1), (max(z1 - z0, 0), max(y1 - y0, 0), x_right))

    if not slabs:
        return False, tuple()

    recon = np.zeros(region_sizes, dtype=bool)
    for starts, sizes in slabs:
        z, y, x = starts
        dz, dy, dx = sizes
        recon[z : z + dz, y : y + dy, x : x + dx] = True

    if not np.array_equal(recon, local_mask):
        return False, tuple()

    z_off, y_off, x_off = region_starts
    global_slabs = tuple(
        (
            (starts[0] + z_off, starts[1] + y_off, starts[2] + x_off),
            sizes,
        )
        for starts, sizes in slabs
    )
    return True, global_slabs


def lossy_fraction(
    field_shape: tuple[int, ...],
    region: tuple[slice, ...],
    conductivity_region,
) -> float:
    """Fraction of lossy voxels for a field component."""
    full_mask = np.zeros(field_shape, dtype=bool)
    full_mask[region] = np.asarray(conductivity_region) > 0.0
    return float(full_mask.mean())


def compiled_jax_dynamic_slice(arr, starts, sizes):
    import jax

    return jax.lax.dynamic_slice(arr, starts, sizes)


def compiled_jax_dynamic_update_slice(arr, patch, starts):
    import jax

    return jax.lax.dynamic_update_slice(arr, patch, starts)
