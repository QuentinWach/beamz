"""Strict brush-based fabrication constraints for binary topology design.

This module implements the conditional feasible-design generator introduced by
Schubert et al., *Inverse Design of Photonic Devices with Strict Foundry
Fabrication Constraints* (ACS Photonics, 2022).  A design is feasible for a
brush when both its solid and void phases are invariant under morphological
opening by that brush.  The generator constructs only such designs, while a
smooth convolutional transform supplies a straight-through gradient estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import convolve2d


@dataclass(frozen=True, slots=True)
class Brush2D:
    """A binary, centrosymmetric structuring element for 2D fabrication rules."""

    mask: np.ndarray
    name: str = "custom"

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2 or not mask.size or not np.any(mask):
            raise ValueError("A brush mask must be a nonempty 2D boolean array.")
        if not np.array_equal(mask, np.flip(mask, axis=(0, 1))):
            raise ValueError("A brush mask must be centrosymmetric.")
        immutable = mask.copy()
        immutable.setflags(write=False)
        object.__setattr__(self, "mask", immutable)
        object.__setattr__(self, "name", str(self.name))

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.mask.shape)

    @property
    def area(self) -> int:
        return int(np.sum(self.mask))


def circular_brush(diameter: int) -> Brush2D:
    """Return the paper's pixel-centred circular brush.

    ``diameter`` is the physical brush diameter in pixels, not an odd support
    size.  For an even diameter the circle is centred between the four central
    pixels.  Thus the 100 nm brush on the paper's 10 nm grid is a 10-by-10
    raster, rather than an 11-by-11 raster with a 100 nm radius test.
    """

    diameter = int(diameter)
    if diameter < 1:
        raise ValueError("Circular brush diameter must be at least one pixel.")
    radius = 0.5 * float(diameter)
    coordinates = np.arange(diameter, dtype=float) - 0.5 * (diameter - 1)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    mask = xx**2 + yy**2 <= radius**2
    return Brush2D(mask=mask, name=f"circle-{diameter}")


def notched_square_brush(width: int) -> Brush2D:
    """Return the paper's square brush with its four corner pixels removed."""

    width = int(width)
    if width < 3:
        raise ValueError("Notched-square brush width must be at least three pixels.")
    mask = np.ones((width, width), dtype=bool)
    mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = False
    return Brush2D(mask=mask, name=f"notched-square-{width}")


def _footprint_convolution(
    values: np.ndarray,
    brush: Brush2D,
    *,
    transpose: bool = False,
) -> np.ndarray:
    """Apply the brush footprint with an explicit discrete anchor.

    Forward convolution maps touches to covered pixels. Transposed convolution
    maps pixels to the touches whose footprints overlap them. These operations
    coincide for odd brushes, but differ by one cell for even brushes such as
    the paper's 12-pixel notched square.
    """

    values = np.asarray(values)
    kernel = np.asarray(brush.mask, dtype=values.dtype)
    height, width = kernel.shape
    anchor = (height // 2, width // 2)
    if transpose:
        kernel = np.flip(kernel, axis=(0, 1))
        anchor = (height - 1 - anchor[0], width - 1 - anchor[1])
    full = convolve2d(values, kernel, mode="full", boundary="fill")
    return full[
        anchor[0] : anchor[0] + values.shape[0],
        anchor[1] : anchor[1] + values.shape[1],
    ]


def _dilate(
    values: np.ndarray,
    brush: Brush2D,
    *,
    transpose: bool = False,
) -> np.ndarray:
    """Binary footprint dilation, optionally mapped from pixels to touches."""

    # ndimage's binary implementation is roughly an order of magnitude faster
    # than a full integer convolution for the small brushes used here.  Its
    # default anchor matches `_footprint_convolution`; the transposed operator
    # needs a one-cell anchor shift along each even-sized brush axis.
    origin = tuple(-1 if transpose and size % 2 == 0 else 0 for size in brush.shape)
    return binary_dilation(
        np.asarray(values, dtype=bool),
        structure=brush.mask,
        origin=origin,
        border_value=0,
    )


def _erode(values: np.ndarray, brush: Brush2D) -> np.ndarray:
    """Binary erosion with out-of-domain brush pixels ignored.

    Clipping the brush at the boundary makes uniform all-solid and all-void seed
    designs feasible, as required by the generator and the paper's initialization.
    """

    values = np.asarray(values, dtype=bool)
    return ~_dilate(~values, brush, transpose=True)


def morphological_opening(values: np.ndarray, brush: Brush2D) -> np.ndarray:
    """Return binary erosion followed by dilation using ``brush``."""

    return _dilate(_erode(values, brush), brush)


def brush_feasibility_errors(
    density: np.ndarray,
    brush: Brush2D,
) -> tuple[np.ndarray, np.ndarray]:
    """Return solid- and void-phase pixels violating brush feasibility."""

    solid = np.asarray(density) > 0.5
    solid_error = np.logical_xor(morphological_opening(solid, brush), solid)
    void = ~solid
    void_error = np.logical_xor(morphological_opening(void, brush), void)
    return solid_error, void_error


def is_brush_feasible(density: np.ndarray, brush: Brush2D) -> bool:
    """Whether both solid and void phases are invariant under brush opening."""

    solid_error, void_error = brush_feasibility_errors(density, brush)
    return not bool(np.any(solid_error) or np.any(void_error))


@dataclass(frozen=True, slots=True)
class GeneratedDesign:
    """Result and diagnostics from conditional feasible-design generation."""

    density: np.ndarray
    solid_touches: np.ndarray
    void_touches: np.ndarray
    steps: int


@dataclass(frozen=True, slots=True)
class GeneratorState:
    """Pixel and touch states from equations (3)--(9) of the paper."""

    existing_solid: np.ndarray
    existing_void: np.ndarray
    valid_solid: np.ndarray
    valid_void: np.ndarray
    possible_solid: np.ndarray
    possible_void: np.ndarray
    required_solid: np.ndarray
    required_void: np.ndarray
    resolving_solid: np.ndarray
    resolving_void: np.ndarray
    free_solid: np.ndarray
    free_void: np.ndarray


def _fixed_mask(
    value: np.ndarray | None,
    shape: tuple[int, int],
    *,
    name: str,
) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=bool)
    mask = np.asarray(value, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"{name} shape {mask.shape} does not match reward {shape}.")
    return mask


def generator_state(
    solid_touches: np.ndarray,
    void_touches: np.ndarray,
    brush: Brush2D,
    *,
    fixed_solid: np.ndarray | None = None,
    fixed_void: np.ndarray | None = None,
) -> GeneratorState:
    """Compute all conditional-generator states for existing brush touches.

    This is a direct implementation of equations (3)--(9).  Exposing the state
    calculation makes the traversal auditable against Figures S1 and S2 of the
    paper's Supporting Information.
    """

    solid_touches = np.asarray(solid_touches, dtype=bool)
    void_touches = np.asarray(void_touches, dtype=bool)
    if solid_touches.ndim != 2 or solid_touches.shape != void_touches.shape:
        raise ValueError("solid_touches and void_touches must be matching 2D arrays.")
    shape = tuple(int(value) for value in solid_touches.shape)
    fixed_solid = _fixed_mask(fixed_solid, shape, name="fixed_solid")
    fixed_void = _fixed_mask(fixed_void, shape, name="fixed_void")
    if np.any(fixed_solid & fixed_void):
        raise ValueError("A pixel cannot be both fixed solid and fixed void.")
    if np.any(solid_touches & void_touches):
        raise ValueError("A touch cannot be both solid and void.")

    # Fixed pixels are pre-existing material, not unresolved pixels that the
    # traversal must redundantly repaint with brush touches.
    existing_solid = _dilate(solid_touches, brush) | fixed_solid
    existing_void = _dilate(void_touches, brush) | fixed_void
    if np.any(existing_solid & existing_void):
        raise ValueError("Solid and void touches color overlapping pixels.")

    impossible_solid = _dilate(existing_void | fixed_void, brush, transpose=True)
    impossible_void = _dilate(existing_solid | fixed_solid, brush, transpose=True)
    valid_solid = ~impossible_solid & ~solid_touches
    valid_void = ~impossible_void & ~void_touches

    possible_solid = _dilate(solid_touches | valid_solid, brush) & ~fixed_void
    possible_void = _dilate(void_touches | valid_void, brush) & ~fixed_solid
    required_solid = ~existing_solid & ~possible_void
    required_void = ~existing_void & ~possible_solid
    resolving_solid = _dilate(required_solid, brush, transpose=True) & valid_solid
    resolving_void = _dilate(required_void, brush, transpose=True) & valid_void
    free_solid = (
        ~_dilate(possible_void | existing_void, brush, transpose=True) & valid_solid
    )
    free_void = (
        ~_dilate(possible_solid | existing_solid, brush, transpose=True) & valid_void
    )
    return GeneratorState(
        existing_solid=existing_solid,
        existing_void=existing_void,
        valid_solid=valid_solid,
        valid_void=valid_void,
        possible_solid=possible_solid,
        possible_void=possible_void,
        required_solid=required_solid,
        required_void=required_void,
        resolving_solid=resolving_solid,
        resolving_void=resolving_void,
        free_solid=free_solid,
        free_void=free_void,
    )


@lru_cache(maxsize=32)
def _compiled_conditional_generator(
    shape: tuple[int, int],
    brush_shape: tuple[int, int],
    brush_bytes: bytes,
    max_steps: int,
    diagonal_symmetry: bool,
    reflection_symmetry: Literal["x", "xy"] | None,
):
    """Build and cache a compiled full-array implementation of Algorithm 1."""

    brush_mask = np.frombuffer(brush_bytes, dtype=np.uint8).reshape(brush_shape)
    kernel = jnp.asarray(brush_mask, dtype=jnp.float32)[None, None, :, :]

    def reflect_touches(values: jax.Array, axis: int) -> jax.Array:
        """Reflect touch centers, including the half-cell even-brush offset."""

        flipped = jnp.flip(values, axis=axis)
        if brush_shape[axis] % 2:
            return flipped
        # An even brush is centred on the dual lattice. With the forward
        # dilation anchor used below, t reflects to N - t rather than
        # N - 1 - t. Touch zero therefore has no in-domain partner and must be
        # excluded from a reflection orbit.
        if axis == 0:
            return jnp.concatenate(
                (jnp.zeros_like(flipped[:1, :]), flipped[:-1, :]),
                axis=0,
            )
        return jnp.concatenate(
            (jnp.zeros_like(flipped[:, :1]), flipped[:, :-1]),
            axis=1,
        )

    def orbit_any(values: jax.Array) -> jax.Array:
        reflected = values
        if reflection_symmetry in {"x", "xy"}:
            reflected = reflected | reflect_touches(values, axis=0)
        if reflection_symmetry == "xy":
            reflected = reflected | reflect_touches(reflected, axis=1)
        return reflected

    def orbit_all(values: jax.Array) -> jax.Array:
        reflected = values
        if reflection_symmetry in {"x", "xy"}:
            reflected = reflected & reflect_touches(values, axis=0)
        if reflection_symmetry == "xy":
            reflected = reflected & reflect_touches(reflected, axis=1)
        return reflected

    def orbit_sum(values: jax.Array) -> jax.Array:
        reflected = values
        if reflection_symmetry in {"x", "xy"}:
            reflected = reflected + reflect_touches(values, axis=0)
        if reflection_symmetry == "xy":
            reflected = reflected + reflect_touches(reflected, axis=1)
        return reflected

    def dilation(values: jax.Array, *, transpose: bool = False) -> jax.Array:
        padding = tuple(
            (
                size // 2 if transpose else (size - 1) // 2,
                (size - 1) // 2 if transpose else size // 2,
            )
            for size in brush_shape
        )
        covered = jax.lax.conv_general_dilated(
            values.astype(jnp.float32)[None, None, :, :],
            kernel,
            window_strides=(1, 1),
            padding=padding,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        return covered[0, 0] > 0.0

    def compile_target(
        reward: jax.Array,
        fixed_solid: jax.Array,
        fixed_void: jax.Array,
        initial_solid_touches: jax.Array,
        initial_void_touches: jax.Array,
    ):
        touch_reward = jax.lax.conv_general_dilated(
            reward[None, None, :, :],
            kernel,
            window_strides=(1, 1),
            padding=tuple((size // 2, (size - 1) // 2) for size in brush_shape),
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )[0, 0]

        def condition(carry):
            step, _, _, done = carry
            return (step < max_steps) & ~done

        def body(carry):
            step, solid_touches, void_touches, _ = carry
            existing_solid = dilation(solid_touches) | fixed_solid
            existing_void = dilation(void_touches) | fixed_void
            complete = jnp.all(existing_solid | existing_void)

            def finish(_):
                return step, solid_touches, void_touches, jnp.asarray(True)

            def update(_):
                impossible_solid = dilation(existing_void | fixed_void, transpose=True)
                impossible_void = dilation(existing_solid | fixed_solid, transpose=True)
                valid_solid = ~impossible_solid & ~solid_touches
                valid_void = ~impossible_void & ~void_touches
                possible_solid = dilation(solid_touches | valid_solid) & ~fixed_void
                possible_void = dilation(void_touches | valid_void) & ~fixed_solid
                required_solid = ~existing_solid & ~possible_void
                required_void = ~existing_void & ~possible_solid
                resolving_solid = dilation(required_solid, transpose=True) & valid_solid
                resolving_void = dilation(required_void, transpose=True) & valid_void
                free_solid = (
                    ~dilation(
                        possible_void | existing_void,
                        transpose=True,
                    )
                    & valid_solid
                )
                free_void = (
                    ~dilation(
                        possible_solid | existing_solid,
                        transpose=True,
                    )
                    & valid_void
                )
                valid_orbit_solid = orbit_all(valid_solid)
                valid_orbit_void = orbit_all(valid_void)
                free_orbit_solid = (
                    orbit_any(free_solid) & valid_orbit_solid
                    if reflection_symmetry is not None
                    else free_solid
                )
                free_orbit_void = (
                    orbit_any(free_void) & valid_orbit_void
                    if reflection_symmetry is not None
                    else free_void
                )
                bulk_free_solid = orbit_all(free_solid)
                bulk_free_void = orbit_all(free_void)
                has_bulk_free = (jnp.any(bulk_free_solid) | jnp.any(bulk_free_void)) & (
                    reflection_symmetry is not None
                )
                has_free = jnp.any(free_orbit_solid) | jnp.any(free_orbit_void)

                def take_free(_):
                    return solid_touches | free_solid, void_touches | free_void

                def take_bulk_free_orbits(_):
                    return (
                        solid_touches | bulk_free_solid,
                        void_touches | bulk_free_void,
                    )

                def take_single(_):
                    resolving_orbit_solid = (
                        orbit_any(resolving_solid) & valid_orbit_solid
                    )
                    resolving_orbit_void = orbit_any(resolving_void) & valid_orbit_void
                    has_resolving = jnp.any(resolving_orbit_solid) | jnp.any(
                        resolving_orbit_void
                    )
                    candidate_solid = jnp.where(
                        has_resolving, resolving_orbit_solid, valid_orbit_solid
                    )
                    candidate_void = jnp.where(
                        has_resolving, resolving_orbit_void, valid_orbit_void
                    )
                    orbit_reward = orbit_sum(touch_reward)
                    solid_scores = jnp.where(candidate_solid, orbit_reward, -jnp.inf)
                    void_scores = jnp.where(candidate_void, -orbit_reward, -jnp.inf)
                    solid_index = jnp.argmax(solid_scores)
                    void_index = jnp.argmax(void_scores)
                    solid_best = solid_scores.reshape(-1)[solid_index]
                    void_best = void_scores.reshape(-1)[void_index]
                    choose_solid = solid_best >= void_best
                    selected_solid = (
                        jnp.arange(reward.size).reshape(shape) == solid_index
                    )
                    selected_void = jnp.arange(reward.size).reshape(shape) == void_index
                    if diagonal_symmetry:
                        selected_solid = selected_solid | selected_solid.T
                        selected_void = selected_void | selected_void.T
                    if reflection_symmetry is not None:
                        selected_solid = orbit_any(selected_solid)
                        selected_void = orbit_any(selected_void)
                    return (
                        solid_touches | (selected_solid & choose_solid),
                        void_touches | (selected_void & ~choose_solid),
                    )

                if reflection_symmetry is None:
                    next_solid, next_void = jax.lax.cond(
                        has_free,
                        take_free,
                        take_single,
                        operand=None,
                    )
                else:
                    # A free touch's reflected partners need only be valid, not
                    # themselves free. Select one orbit atomically so opposite
                    # phases cannot claim intersecting reflected orbits in the
                    # same update.
                    free_reward = orbit_sum(touch_reward)
                    free_solid_scores = jnp.where(
                        free_orbit_solid, free_reward, -jnp.inf
                    )
                    free_void_scores = jnp.where(
                        free_orbit_void, -free_reward, -jnp.inf
                    )
                    free_solid_index = jnp.argmax(free_solid_scores)
                    free_void_index = jnp.argmax(free_void_scores)
                    free_solid_best = free_solid_scores.reshape(-1)[free_solid_index]
                    free_void_best = free_void_scores.reshape(-1)[free_void_index]
                    choose_free_solid = free_solid_best >= free_void_best
                    selected_free_solid = orbit_any(
                        jnp.arange(reward.size).reshape(shape) == free_solid_index
                    )
                    selected_free_void = orbit_any(
                        jnp.arange(reward.size).reshape(shape) == free_void_index
                    )

                    def take_free_orbit(_):
                        return (
                            solid_touches | (selected_free_solid & choose_free_solid),
                            void_touches | (selected_free_void & ~choose_free_solid),
                        )

                    next_solid, next_void = jax.lax.cond(
                        has_bulk_free,
                        take_bulk_free_orbits,
                        lambda _: jax.lax.cond(
                            has_free,
                            take_free_orbit,
                            take_single,
                            operand=None,
                        ),
                        operand=None,
                    )
                return step + 1, next_solid, next_void, jnp.asarray(False)

            return jax.lax.cond(complete, finish, update, operand=None)

        step, solid_touches, void_touches, done = jax.lax.while_loop(
            condition,
            body,
            (
                jnp.asarray(0),
                initial_solid_touches,
                initial_void_touches,
                jnp.asarray(False),
            ),
        )
        existing_solid = dilation(solid_touches) | fixed_solid
        existing_void = dilation(void_touches) | fixed_void
        complete = jnp.all(existing_solid | existing_void)
        return existing_solid, solid_touches, void_touches, step, complete | done

    return jax.jit(compile_target)


def _conditional_generator_jax(
    reward: np.ndarray,
    brush: Brush2D,
    *,
    fixed_solid: np.ndarray,
    fixed_void: np.ndarray,
    initial_solid_touches: np.ndarray,
    initial_void_touches: np.ndarray,
    max_steps: int,
    diagonal_symmetry: bool,
    reflection_symmetry: Literal["x", "xy"] | None,
) -> GeneratedDesign:
    compiled = _compiled_conditional_generator(
        tuple(int(value) for value in reward.shape),
        brush.shape,
        np.asarray(brush.mask, dtype=np.uint8).tobytes(),
        int(max_steps),
        bool(diagonal_symmetry),
        reflection_symmetry,
    )
    density, solid_touches, void_touches, steps, complete = compiled(
        jnp.asarray(reward, dtype=jnp.float32),
        jnp.asarray(fixed_solid),
        jnp.asarray(fixed_void),
        jnp.asarray(initial_solid_touches),
        jnp.asarray(initial_void_touches),
    )
    if not bool(complete):
        raise RuntimeError(
            f"Conditional generator did not complete within {max_steps} steps."
        )
    return GeneratedDesign(
        density=np.asarray(density, dtype=np.float32),
        solid_touches=np.asarray(solid_touches, dtype=bool),
        void_touches=np.asarray(void_touches, dtype=bool),
        steps=int(steps),
    )


def conditional_generator(  # noqa: C901
    reward: np.ndarray,
    brush: Brush2D,
    *,
    fixed_solid: np.ndarray | None = None,
    fixed_void: np.ndarray | None = None,
    initial_solid_touches: np.ndarray | None = None,
    initial_void_touches: np.ndarray | None = None,
    max_steps: int | None = None,
    verify: bool = True,
    backend: Literal["auto", "scipy", "jax"] = "auto",
    diagonal_symmetry: bool = False,
    reflection_symmetry: Literal["x", "xy"] | None = None,
) -> GeneratedDesign:
    """Greedily generate a strictly brush-feasible design conditioned on reward.

    This follows Algorithm 1 and equations (2)--(9) of Schubert et al. Free
    touches are taken together. Otherwise, the highest-reward resolving touch is
    selected, falling back to the highest-reward valid touch. Solid-touch reward
    is the sum of covered reward pixels and void-touch reward is its negative.

    ``fixed_solid`` and ``fixed_void`` are pre-existing boundary context. Their
    values constrain every generated touch but are exempt from the final
    feasibility check themselves; only pixels outside the fixed masks are
    generated and certified. With ``diagonal_symmetry=True``, every selected
    touch is paired atomically with its transpose, imposing exact diagonal
    symmetry rather than merely supplying a symmetric reward.
    ``reflection_symmetry="x"`` pairs touches across the horizontal x-axis;
    ``"xy"`` pairs the complete four-touch reflection orbit. Reflection
    symmetries require the reward and fixed context to have the same symmetry.
    ``initial_solid_touches`` and ``initial_void_touches`` may seed a compatible
    partial design, which is useful when refining a feasible design on a finer
    grid. The normal generator traversal completes all remaining pixels.
    """

    reward_shape = getattr(reward, "shape", None)
    if reward_shape is None or len(reward_shape) != 2:
        raise ValueError("Conditional-generator reward must be a finite 2D array.")
    shape = tuple(int(value) for value in reward_shape)
    reward_size = int(np.prod(shape))
    if backend not in {"auto", "scipy", "jax"}:
        raise ValueError("backend must be 'auto', 'scipy', or 'jax'.")
    if backend == "auto":
        backend = "jax" if reward_size >= 4096 else "scipy"
    if backend == "jax":
        reward_array = jnp.asarray(reward, dtype=jnp.float32)
        if not bool(jnp.all(jnp.isfinite(reward_array))):
            raise ValueError("Conditional-generator reward must be a finite 2D array.")
    else:
        reward_array = np.asarray(reward, dtype=float)
        if not np.all(np.isfinite(reward_array)):
            raise ValueError("Conditional-generator reward must be a finite 2D array.")
    fixed_solid = _fixed_mask(fixed_solid, shape, name="fixed_solid")
    fixed_void = _fixed_mask(fixed_void, shape, name="fixed_void")
    initial_solid_touches = _fixed_mask(
        initial_solid_touches,
        shape,
        name="initial_solid_touches",
    )
    initial_void_touches = _fixed_mask(
        initial_void_touches,
        shape,
        name="initial_void_touches",
    )
    if np.any(fixed_solid & fixed_void):
        raise ValueError("A pixel cannot be both fixed solid and fixed void.")
    if np.any(initial_solid_touches & initial_void_touches):
        raise ValueError("A touch cannot be both initially solid and void.")
    # Validate the partial coloring before entering either implementation.
    generator_state(
        initial_solid_touches,
        initial_void_touches,
        brush,
        fixed_solid=fixed_solid,
        fixed_void=fixed_void,
    )
    if reflection_symmetry not in {None, "x", "xy"}:
        raise ValueError("reflection_symmetry must be None, 'x', or 'xy'.")
    if diagonal_symmetry and reflection_symmetry is not None:
        raise ValueError(
            "diagonal_symmetry and reflection_symmetry are mutually exclusive."
        )
    if diagonal_symmetry:
        if shape[0] != shape[1]:
            raise ValueError("Diagonal symmetry requires a square reward array.")
        if not bool(jnp.array_equal(reward_array, reward_array.T)):
            raise ValueError(
                "A diagonally symmetric generator requires symmetric reward."
            )
        if not np.array_equal(fixed_solid, fixed_solid.T) or not np.array_equal(
            fixed_void, fixed_void.T
        ):
            raise ValueError("Fixed context must respect diagonal symmetry.")
    if reflection_symmetry in {"x", "xy"}:
        if not bool(
            jnp.allclose(
                reward_array,
                jnp.flip(reward_array, axis=0),
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise ValueError(
                "Reflection symmetry requires reward symmetry about the x-axis."
            )
        if not np.array_equal(fixed_solid, np.flip(fixed_solid, axis=0)) or not (
            np.array_equal(fixed_void, np.flip(fixed_void, axis=0))
        ):
            raise ValueError("Fixed context must respect x reflection symmetry.")
    if reflection_symmetry == "xy":
        if not bool(
            jnp.allclose(
                reward_array,
                jnp.flip(reward_array, axis=1),
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise ValueError(
                "XY reflection symmetry requires reward symmetry about the y-axis."
            )
        if not np.array_equal(fixed_solid, np.flip(fixed_solid, axis=1)) or not (
            np.array_equal(fixed_void, np.flip(fixed_void, axis=1))
        ):
            raise ValueError("Fixed context must respect y reflection symmetry.")

    solid_touches = initial_solid_touches.copy()
    void_touches = initial_void_touches.copy()
    max_steps = reward_size if max_steps is None else int(max_steps)
    if max_steps < 1:
        raise ValueError("max_steps must be positive.")
    if backend == "jax":
        generated = _conditional_generator_jax(
            reward_array,
            brush,
            fixed_solid=fixed_solid,
            fixed_void=fixed_void,
            initial_solid_touches=initial_solid_touches,
            initial_void_touches=initial_void_touches,
            max_steps=max_steps,
            diagonal_symmetry=diagonal_symmetry,
            reflection_symmetry=reflection_symmetry,
        )
        if not np.all(generated.density[fixed_solid] == 1.0):
            raise RuntimeError("Generated design violated fixed-solid pixels.")
        if not np.all(generated.density[fixed_void] == 0.0):
            raise RuntimeError("Generated design violated fixed-void pixels.")
        if verify:
            solid_error, void_error = brush_feasibility_errors(generated.density, brush)
            generated_pixels = ~(fixed_solid | fixed_void)
            if np.any((solid_error | void_error) & generated_pixels):
                raise RuntimeError(
                    "Conditional generator returned an infeasible generated region."
                )
        if reflection_symmetry is not None:
            axes = (0,) if reflection_symmetry == "x" else (0, 1)
            if not np.array_equal(
                generated.density,
                np.flip(generated.density, axis=axes),
            ):
                raise RuntimeError(
                    "Conditional generator returned a non-symmetric density."
                )
        return generated

    reward_array = np.asarray(reward_array)
    touch_reward = _footprint_convolution(reward_array, brush, transpose=True)
    for step in range(1, max_steps + 1):
        state = generator_state(
            solid_touches,
            void_touches,
            brush,
            fixed_solid=fixed_solid,
            fixed_void=fixed_void,
        )
        existing_solid = state.existing_solid
        existing_void = state.existing_void
        if np.all(existing_solid | existing_void):
            density = existing_solid.astype(np.float32)
            if not np.all(density[fixed_solid] == 1.0):
                raise RuntimeError("Generated design violated fixed-solid pixels.")
            if not np.all(density[fixed_void] == 0.0):
                raise RuntimeError("Generated design violated fixed-void pixels.")
            if verify:
                solid_error, void_error = brush_feasibility_errors(density, brush)
                generated_pixels = ~(fixed_solid | fixed_void)
                if np.any((solid_error | void_error) & generated_pixels):
                    raise RuntimeError(
                        "Conditional generator returned an infeasible generated region."
                    )
            if reflection_symmetry is not None:
                axes = (0,) if reflection_symmetry == "x" else (0, 1)
                if not np.array_equal(density, np.flip(density, axis=axes)):
                    raise RuntimeError(
                        "Conditional generator returned a non-symmetric density."
                    )
            return GeneratedDesign(
                density=density,
                solid_touches=solid_touches,
                void_touches=void_touches,
                steps=step - 1,
            )

        if reflection_symmetry is not None:
            axes = (0,) if reflection_symmetry == "x" else (0, 1)

            def reflect_touches(values, axis):
                flipped = np.flip(values, axis=axis)
                if brush.shape[axis] % 2:
                    return flipped
                reflected = np.zeros_like(values)
                if axis == 0:
                    reflected[1:, :] = flipped[:-1, :]
                else:
                    reflected[:, 1:] = flipped[:, :-1]
                return reflected

            def orbit_any(values, axes=axes):
                result = values.copy()
                for axis in axes:
                    result |= reflect_touches(result, axis)
                return result

            def orbit_all(values, axes=axes):
                result = values.copy()
                for axis in axes:
                    result &= reflect_touches(result, axis)
                return result

            def orbit_sum(values, axes=axes):
                result = values.copy()
                for axis in axes:
                    result += reflect_touches(result, axis)
                return result

            bulk_free_solid = orbit_all(state.free_solid)
            bulk_free_void = orbit_all(state.free_void)
            if np.any(bulk_free_solid) or np.any(bulk_free_void):
                solid_touches |= bulk_free_solid
                void_touches |= bulk_free_void
                continue

        if reflection_symmetry is None and (
            np.any(state.free_solid) or np.any(state.free_void)
        ):
            solid_touches |= state.free_solid
            void_touches |= state.free_void
            continue

        if reflection_symmetry is None:
            candidates_solid, candidates_void = (
                (state.resolving_solid, state.resolving_void)
                if np.any(state.resolving_solid) or np.any(state.resolving_void)
                else (state.valid_solid, state.valid_void)
            )
            orbit_reward = touch_reward
        else:
            valid_solid = orbit_all(state.valid_solid)
            valid_void = orbit_all(state.valid_void)
            free_solid = orbit_any(state.free_solid) & valid_solid
            free_void = orbit_any(state.free_void) & valid_void
            if np.any(free_solid) or np.any(free_void):
                candidates_solid, candidates_void = free_solid, free_void
            else:
                resolving_solid = orbit_any(state.resolving_solid) & valid_solid
                resolving_void = orbit_any(state.resolving_void) & valid_void
                candidates_solid, candidates_void = (
                    (resolving_solid, resolving_void)
                    if np.any(resolving_solid) or np.any(resolving_void)
                    else (valid_solid, valid_void)
                )
            orbit_reward = orbit_sum(touch_reward)
        solid_scores = np.where(candidates_solid, orbit_reward, -np.inf)
        void_scores = np.where(candidates_void, -orbit_reward, -np.inf)
        solid_flat = int(np.argmax(solid_scores))
        void_flat = int(np.argmax(void_scores))
        solid_best = float(solid_scores.flat[solid_flat])
        void_best = float(void_scores.flat[void_flat])
        if not np.isfinite(max(solid_best, void_best)):
            raise RuntimeError("Conditional generator stalled with no valid touches.")
        if solid_best >= void_best:
            if reflection_symmetry is not None:
                selected = np.zeros(shape, dtype=bool)
                selected.flat[solid_flat] = True
                solid_touches |= orbit_any(selected)
            else:
                solid_touches.flat[solid_flat] = True
            if diagonal_symmetry:
                row, column = np.unravel_index(solid_flat, shape)
                solid_touches[column, row] = True
        else:
            if reflection_symmetry is not None:
                selected = np.zeros(shape, dtype=bool)
                selected.flat[void_flat] = True
                void_touches |= orbit_any(selected)
            else:
                void_touches.flat[void_flat] = True
            if diagonal_symmetry:
                row, column = np.unravel_index(void_flat, shape)
                void_touches[column, row] = True

    raise RuntimeError(
        f"Conditional generator did not complete within {max_steps} steps."
    )


def filtered_reward(
    latent: jax.Array,
    brush: Brush2D,
    *,
    beta: float = 4.0,
    diagonal_symmetry: bool = False,
    normalize_kernel: bool = False,
) -> jax.Array:
    """Paper estimator ``tanh(beta * (latent convolved with brush))``."""

    latent = jnp.asarray(latent)
    kernel = jnp.asarray(brush.mask, dtype=latent.dtype)
    if normalize_kernel:
        kernel /= jnp.sum(kernel)
    reward = jax.scipy.signal.convolve2d(latent, kernel, mode="same")
    reward = jnp.tanh(float(beta) * reward)
    if diagonal_symmetry:
        if reward.shape[0] != reward.shape[1]:
            raise ValueError("Diagonal symmetry requires a square reward array.")
        reward = 0.5 * (reward + reward.T)
    return reward


def straight_through_gradient(
    latent: jax.Array,
    density_gradient: jax.Array,
    brush: Brush2D,
    *,
    beta: float = 4.0,
    diagonal_symmetry: bool = False,
    normalize_kernel: bool = False,
) -> jax.Array:
    """Backpropagate a density gradient through the paper's smooth estimator."""

    latent = jnp.asarray(latent)
    density_gradient = jnp.asarray(density_gradient)
    if latent.shape != density_gradient.shape:
        raise ValueError("latent and density_gradient must have matching shapes.")

    def estimator(value: jax.Array) -> jax.Array:
        # Convert the estimator's (-1, +1) range to material density (0, 1).
        return 0.5 * (
            filtered_reward(
                value,
                brush,
                beta=beta,
                diagonal_symmetry=diagonal_symmetry,
                normalize_kernel=normalize_kernel,
            )
            + 1.0
        )

    _, pullback = jax.vjp(estimator, latent)
    return pullback(density_gradient)[0]


__all__ = [
    "Brush2D",
    "GeneratedDesign",
    "GeneratorState",
    "brush_feasibility_errors",
    "circular_brush",
    "conditional_generator",
    "filtered_reward",
    "generator_state",
    "is_brush_feasible",
    "morphological_opening",
    "notched_square_brush",
    "straight_through_gradient",
]
