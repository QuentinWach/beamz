import jax
import jax.numpy as jnp

from beamz.devices.sources.compiler import (
    BatchedSlabGroup,
    CompiledSourceSpec,
)


def sources_for(compiled, timing: str, component: str) -> tuple[CompiledSourceSpec, ...]:
    return tuple(
        spec
        for spec in compiled.source_specs
        if spec.timing == timing and spec.component == component
    )


def apply_specs(
    compiled,
    arr: jnp.ndarray,
    abs_step: jnp.ndarray,
    specs: tuple[CompiledSourceSpec, ...],
) -> jnp.ndarray:
    out = arr
    for spec in specs:
        safe_idx = jnp.clip(abs_step, 0, spec.waveform.shape[0] - 1)
        amp = spec.waveform[safe_idx]
        if spec.is_slab and spec.slab_starts is not None and spec.slab_sizes is not None:
            patch = spec.coeff * amp
            cur = jax.lax.dynamic_slice(out, spec.slab_starts, spec.slab_sizes)
            out = jax.lax.dynamic_update_slice(out, cur + patch, spec.slab_starts)
        else:
            out = out.at[spec.index].add(spec.coeff * amp)
    return out


def apply_batched_slabs(
    compiled,
    arr: jnp.ndarray,
    abs_step: jnp.ndarray,
    group: BatchedSlabGroup,
) -> jnp.ndarray:
    """Apply stacked slab sources via fori_loop."""
    safe_idx = jnp.clip(abs_step, 0, group.waveforms.shape[1] - 1)
    ndim = len(group.max_sizes)

    if group.n == 1:
        amp = group.waveforms[0, safe_idx]
        starts_0 = group.starts_tuple[0]
        if compiled.config.source_single_slab_dense:
            pad_width = tuple(
                (
                    starts_0[axis],
                    int(arr.shape[axis]) - starts_0[axis] - group.max_sizes[axis],
                )
                for axis in range(ndim)
            )
            dense_coeff = jnp.pad(group.coeffs[0], pad_width)
            return arr + dense_coeff * amp
        patch = group.coeffs[0] * amp
        cur = jax.lax.dynamic_slice(arr, starts_0, group.max_sizes)
        return jax.lax.dynamic_update_slice(arr, cur + patch, starts_0)

    if group.n == 2:

        def apply_one(out, idx: int):
            amp_i = group.waveforms[idx, safe_idx]
            patch_i = group.coeffs[idx] * amp_i
            starts_i = group.starts_tuple[idx]
            cur_i = jax.lax.dynamic_slice(out, starts_i, group.max_sizes)
            return jax.lax.dynamic_update_slice(out, cur_i + patch_i, starts_i)

        return apply_one(apply_one(arr, 0), 1)

    def body(idx, out):
        amp = group.waveforms[idx, safe_idx]
        patch = group.coeffs[idx] * amp
        starts_i = [group.starts[idx, axis] for axis in range(ndim)]
        cur = jax.lax.dynamic_slice(out, starts_i, group.max_sizes)
        return jax.lax.dynamic_update_slice(out, cur + patch, starts_i)

    return jax.lax.fori_loop(0, group.n, body, arr)


def apply_group(
    compiled,
    arr: jnp.ndarray,
    abs_step: jnp.ndarray,
    batch: BatchedSlabGroup | None,
    rest: tuple[CompiledSourceSpec, ...],
) -> jnp.ndarray:
    """Apply batched slab sources then remaining non-slab sources."""
    if batch is not None:
        arr = compiled._apply_batched_slabs(arr, abs_step, batch)
    if rest:
        arr = compiled._apply_specs(arr, abs_step, rest)
    return arr
