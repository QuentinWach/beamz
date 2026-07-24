"""Differentiable material inputs for compiled BeamZ simulations.

This module keeps geometry, sources, monitors, boundaries, and grid shapes static
while passing permittivity values through the compiled FDTD scan as dynamic JAX
inputs.  A compiled executable can therefore be reused across inverse-design
iterations and differentiated without rebuilding the public :class:`Simulation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np

from beamz.lattice import (
    canonical_component_2d,
    sample_voxel_grid_at_component_2d,
    sample_voxel_grid_at_e_component_3d_centered,
)
from beamz.simulation import kernels
from beamz.simulation.api import Simulation
from beamz.simulation.execute import build_scan, initial_program_state
from beamz.simulation.model import CompiledProgram, SimulationState, UpdateCoefficients

Symmetry2D = Literal["x", "xy"] | None


def _aligned_index(value: float, resolution: float, *, name: str) -> int:
    scaled = float(value) / float(resolution)
    rounded = int(round(scaled))
    if not np.isclose(scaled, rounded, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"{name}={value!r} must align to the simulation resolution {resolution!r}."
        )
    return rounded


@dataclass(frozen=True, slots=True)
class DesignRegion:
    """Map a compact density variable into a rectangular material-grid region.

    Bounds use public Cartesian order and metres. Density arrays use BeamZ storage
    order: ``(y, x)`` in 2D and ``(z, y, x)`` in 3D. The optional 2D symmetries
    match the Ceviche challenge convention: ``"x"`` mirrors across the horizontal
    axis and ``"xy"`` mirrors across both axes.
    """

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    eps_min: float
    eps_max: float
    symmetry: Symmetry2D = None

    def __post_init__(self) -> None:
        lower = tuple(float(value) for value in self.lower)
        upper = tuple(float(value) for value in self.upper)
        if len(lower) not in {2, 3} or len(lower) != len(upper):
            raise ValueError("DesignRegion bounds must both be 2D or both be 3D.")
        if any(not np.isfinite(value) for value in (*lower, *upper)):
            raise ValueError("DesignRegion bounds must be finite.")
        if any(high <= low for low, high in zip(lower, upper, strict=True)):
            raise ValueError(
                "Every DesignRegion upper bound must exceed its lower bound."
            )
        eps_min, eps_max = float(self.eps_min), float(self.eps_max)
        if not np.isfinite(eps_min) or not np.isfinite(eps_max) or eps_max <= eps_min:
            raise ValueError("DesignRegion requires finite eps_max > eps_min.")
        if self.symmetry not in {None, "x", "xy"}:
            raise ValueError("DesignRegion symmetry must be None, 'x', or 'xy'.")
        if len(lower) == 3 and self.symmetry is not None:
            raise ValueError("DesignRegion symmetry is currently defined only for 2D.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "eps_min", eps_min)
        object.__setattr__(self, "eps_max", eps_max)

    def grid_slices(
        self, resolution: float, grid_shape: tuple[int, ...]
    ) -> tuple[slice, ...]:
        """Return storage-order slices for this region on a uniform material grid."""

        if len(grid_shape) != len(self.lower):
            raise ValueError(
                f"A {len(self.lower)}D region cannot index grid shape {grid_shape}."
            )
        lower = tuple(
            _aligned_index(value, resolution, name=f"lower[{axis}]")
            for axis, value in enumerate(self.lower)
        )
        upper = tuple(
            _aligned_index(value, resolution, name=f"upper[{axis}]")
            for axis, value in enumerate(self.upper)
        )
        # Public axes are (x, y, z); material storage is (y, x) or (z, y, x).
        storage_lower = tuple(reversed(lower))
        storage_upper = tuple(reversed(upper))
        if any(
            low < 0 or high > size
            for low, high, size in zip(
                storage_lower, storage_upper, grid_shape, strict=True
            )
        ):
            raise ValueError(
                f"DesignRegion bounds resolve outside material grid {grid_shape}."
            )
        return tuple(
            slice(low, high)
            for low, high in zip(storage_lower, storage_upper, strict=True)
        )

    def full_shape(
        self, resolution: float, grid_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        slices = self.grid_slices(resolution, grid_shape)
        return tuple(int(region.stop - region.start) for region in slices)

    def variable_shape(
        self, resolution: float, grid_shape: tuple[int, ...]
    ) -> tuple[int, ...]:
        shape = self.full_shape(resolution, grid_shape)
        if self.symmetry is None:
            return shape
        ny, nx = shape
        if self.symmetry == "x":
            return ((ny + 1) // 2, nx)
        return ((ny + 1) // 2, (nx + 1) // 2)

    def expand(self, density: jax.Array, full_shape: tuple[int, ...]) -> jax.Array:
        """Expand a possibly symmetry-reduced density into the full region."""

        density = jnp.asarray(density)
        expected = full_shape
        if self.symmetry == "x":
            expected = ((full_shape[0] + 1) // 2, full_shape[1])
        elif self.symmetry == "xy":
            expected = ((full_shape[0] + 1) // 2, (full_shape[1] + 1) // 2)
        if tuple(density.shape) != tuple(expected):
            raise ValueError(
                f"Density shape {tuple(density.shape)} does not match expected {expected}."
            )
        if self.symmetry in {"x", "xy"}:
            reflected = density[:-1] if full_shape[0] % 2 else density
            density = jnp.concatenate((density, jnp.flip(reflected, axis=0)), axis=0)
        if self.symmetry == "xy":
            reflected = density[:, :-1] if full_shape[1] % 2 else density
            density = jnp.concatenate((density, jnp.flip(reflected, axis=1)), axis=1)
        return density

    def materialize(
        self,
        base_permittivity: jax.Array,
        density: jax.Array,
        resolution: float,
    ) -> jax.Array:
        """Return base permittivity with this region replaced by mapped density."""

        base = jnp.asarray(base_permittivity)
        slices = self.grid_slices(float(resolution), tuple(base.shape))
        full = self.expand(
            jnp.asarray(density), self.full_shape(resolution, base.shape)
        )
        epsilon = self.eps_min + full * (self.eps_max - self.eps_min)
        return base.at[slices].set(epsilon.astype(base.dtype))


def coefficients_for_permittivity(
    program: CompiledProgram, permittivity: jax.Array
) -> UpdateCoefficients:
    """Rebuild only electric update terms affected by dynamic permittivity.

    Conductivity, permeability, PML plans, sources, and monitor geometry remain
    static. This function supports the canonical 2D TM lattice and the full 3D
    Yee lattice.
    """

    eps = jnp.asarray(permittivity)
    expected = tuple(int(value) for value in program.grid.permittivity.shape)
    if tuple(eps.shape) != expected:
        raise ValueError(
            f"Permittivity shape {tuple(eps.shape)} does not match grid {expected}."
        )
    coeffs = program.coefficients
    if not program.config.is_3d:
        eps_z = sample_voxel_grid_at_component_2d(eps, "Ez", "xy")
        e_decay_z, e_source_z = kernels.precompute_e_update_coefficients(
            shape=program.grid.Ez.shape,
            conductivity=program.grid.sig_z,
            permittivity=eps_z,
            dt=program.config.dt,
            region=program.grid.region_z,
        )
        return coeffs._replace(e_decay_z=e_decay_z, e_source_z=e_source_z)

    component_shapes = {
        name: tuple(getattr(program.grid, name).shape) for name in ("Ex", "Ey", "Ez")
    }
    sampled = {
        name: sample_voxel_grid_at_e_component_3d_centered(
            eps, name, stored_shape=component_shapes[name]
        )
        for name in component_shapes
    }
    return coeffs._replace(
        e_permittivity_x=sampled["Ex"],
        e_permittivity_y=sampled["Ey"],
        e_permittivity_z=sampled["Ez"],
    )


@dataclass(frozen=True, slots=True)
class DifferentiableResult:
    """JAX-backed result view returned inside differentiable objectives."""

    state: SimulationState
    program: CompiledProgram = field(repr=False, compare=False)

    def _monitor(self, name: str):
        try:
            return next(
                spec for spec in self.program.monitors if spec.name == str(name)
            )
        except StopIteration as exc:
            available = ", ".join(spec.name for spec in self.program.monitors)
            raise KeyError(f"Unknown monitor {name!r}. Available: {available}") from exc

    def frequencies(self, name: str) -> jax.Array:
        spec = self._monitor(name)
        return jnp.asarray(spec.freq_hz[: spec.freq_count])

    def field(self, name: str, component: str) -> jax.Array:
        """Return normalized complex DFT samples with frequency on axis zero."""

        spec = self._monitor(name)
        canonical, sign = str(component), 1.0
        if not self.program.config.is_3d:
            canonical = canonical_component_2d(component, self.program.config.plane_2d)
            if canonical is None:
                raise ValueError(
                    f"Component {component!r} is inactive in the "
                    f"{self.program.config.plane_2d!r} 2D plane."
                )
            from beamz.lattice import public_component_2d

            _, sign = public_component_2d(canonical, self.program.config.plane_2d)
        component_index = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz").index(canonical)
        fc, pc, mi = spec.freq_count, spec.dft_point_count, spec.monitor_index
        if not spec.dft_component_enabled[component_index]:
            raise ValueError(
                f"Monitor {name!r} did not acquire component {component!r}."
            )
        values = (
            self.state.dft_vec_re[mi, component_index, :fc, :pc]
            + 1j * self.state.dft_vec_im[mi, component_index, :fc, :pc]
        )
        weights = jnp.maximum(self.state.dft_weight_sum[mi, :fc], 1e-18)[:, None]
        return jnp.asarray(sign, dtype=values.real.dtype) * (2.0 / weights) * values

    def flux(self, name: str) -> jax.Array:
        """Return differentiable signed frequency-domain power flux."""

        spec = self._monitor(name)
        values = {}
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            component_index = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz").index(component)
            if spec.dft_component_enabled[component_index]:
                values[component] = self.field(name, component)
        template = next(iter(values.values()))

        def component(name_: str):
            value = values.get(name_, jnp.zeros_like(template))
            if name_.startswith("H"):
                phase = jnp.exp(
                    -1j * jnp.pi * self.frequencies(name) * self.program.config.dt
                )
                value = value * phase[:, None]
            return value

        ex, ey, ez = (component(name_) for name_ in ("Ex", "Ey", "Ez"))
        hx, hy, hz = (component(name_) for name_ in ("Hx", "Hy", "Hz"))
        axis = int(spec.normal_axis)
        flux_density = (
            ey * jnp.conj(hz) - ez * jnp.conj(hy),
            ez * jnp.conj(hx) - ex * jnp.conj(hz),
            ex * jnp.conj(hy) - ey * jnp.conj(hx),
        )[axis]
        return (
            0.5
            * float(spec.normal_sign)
            * float(spec.power_scale)
            * jnp.real(jnp.sum(flux_density, axis=1))
        )


class DifferentiableSimulation:
    """Reuse one compiled FDTD program across differentiable material updates."""

    def __init__(
        self,
        simulation: Simulation,
        design_region: DesignRegion,
        *,
        rematerialize: bool = True,
    ) -> None:
        if not isinstance(simulation, Simulation):
            raise TypeError("DifferentiableSimulation requires a Simulation.")
        if not isinstance(design_region, DesignRegion):
            raise TypeError("design_region must be a DesignRegion.")
        self.simulation = simulation
        self.design_region = design_region
        self.program = simulation.compile()
        if self.program.sharding.layout.enabled:
            raise NotImplementedError(
                "DifferentiableSimulation currently requires unsharded execution."
            )
        if len(design_region.lower) != (3 if simulation.is_3d else 2):
            raise ValueError("DesignRegion dimensionality must match the simulation.")
        self.base_permittivity = jnp.asarray(self.program.grid.permittivity)
        self.initial_state = initial_program_state(
            self.program,
            t=float(simulation.time[0]),
            current_step=0,
        )
        self._scan = build_scan(
            self.program,
            donate_state=False,
            rematerialize=bool(rematerialize),
        )

    @property
    def variable_shape(self) -> tuple[int, ...]:
        return self.design_region.variable_shape(
            self.simulation.resolution, tuple(self.base_permittivity.shape)
        )

    def permittivity(self, density: jax.Array) -> jax.Array:
        return self.design_region.materialize(
            self.base_permittivity, density, self.simulation.resolution
        )

    def run(self, density: jax.Array) -> DifferentiableResult:
        permittivity = self.permittivity(density)
        coeffs = coefficients_for_permittivity(self.program, permittivity)
        state = self._scan(self.initial_state, coeffs)
        return DifferentiableResult(state=state, program=self.program)

    def run_results(self, density: jax.Array):
        """Execute dynamic material values and detach normal BeamZ results.

        This boundary is intentionally non-differentiable. Use :meth:`run` inside
        objectives and this method for plotting, modal analysis, and serialization.
        """

        state = self.run(density).state
        return self._results_from_state(state, density)

    def calibration_results(self, density: jax.Array):
        """Build detached static analysis metadata without running FDTD.

        Modal port bases depend on the compiled monitor geometry and material
        cross sections, not on accumulated time-domain fields.  Returning a
        zero-acquisition result avoids an otherwise redundant full simulation
        when constructing a fixed differentiable port projector.
        """

        return self._results_from_state(self.initial_state, density)

    def _results_from_state(
        self,
        state: SimulationState,
        density: jax.Array,
    ):
        from beamz.simulation.execute import (
            _compiled_source_launch_powers,
            _decode_monitor_results,
        )
        from beamz.simulation.results import SimulationResults

        runtime_grid = replace(
            self.program.grid,
            permittivity=self.permittivity(density),
        )
        monitor_results = _decode_monitor_results(self.simulation, self.program, state)
        return SimulationResults.from_run(
            self.simulation,
            runtime_fields=runtime_grid,
            monitor_results=monitor_results,
            source_launch_powers=_compiled_source_launch_powers(
                self.program, len(self.simulation.sources)
            ),
        )

    def value_and_grad(
        self,
        density: jax.Array,
        objective: Callable[[DifferentiableResult], jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate a scalar monitor objective and its density gradient."""

        def evaluate(value):
            result = objective(self.run(value))
            if jnp.ndim(result) != 0:
                raise ValueError("Differentiable objective must return a scalar.")
            return result

        return jax.value_and_grad(evaluate)(jnp.asarray(density))

    def compile_value_and_grad(
        self,
        objective: Callable[[DifferentiableResult], jax.Array],
    ) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
        """Compile a scalar objective and density gradient for repeated use.

        Constructing :func:`jax.value_and_grad` inside an optimization loop can
        retrace a newly created Python closure at every iteration.  This method
        creates and JIT-compiles the transformed objective once, so all later
        material updates reuse the same executable.
        """

        def evaluate(value):
            result = objective(self.run(value))
            if jnp.ndim(result) != 0:
                raise ValueError("Differentiable objective must return a scalar.")
            return result

        return jax.jit(jax.value_and_grad(evaluate))


__all__ = [
    "DesignRegion",
    "DifferentiableResult",
    "DifferentiableSimulation",
    "coefficients_for_permittivity",
]
