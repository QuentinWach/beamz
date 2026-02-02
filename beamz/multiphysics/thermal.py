from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class ThermalParams:
    thermal_dt: float
    tau_avg: float
    k: float = 0.0
    rho: float = 0.0
    cp: float = 0.0
    dn_dT: float = 0.0
    T0: float = 300.0


def _laplacian_neumann(field, dx, dy=None, dz=None):
    """Compute Laplacian with zero-flux Neumann boundaries using edge padding."""
    if dy is None:
        dy = dx
    if field.ndim == 2:
        pad = jnp.pad(field, ((1, 1), (1, 1)), mode="edge")
        lap_x = (pad[1:-1, 2:] - 2.0 * pad[1:-1, 1:-1] + pad[1:-1, :-2]) / (
            dx * dx
        )
        lap_y = (pad[2:, 1:-1] - 2.0 * pad[1:-1, 1:-1] + pad[:-2, 1:-1]) / (
            dy * dy
        )
        return lap_x + lap_y
    elif field.ndim == 3:
        if dz is None:
            dz = dx
        pad = jnp.pad(field, ((1, 1), (1, 1), (1, 1)), mode="edge")
        lap_x = (
            pad[1:-1, 1:-1, 2:]
            - 2.0 * pad[1:-1, 1:-1, 1:-1]
            + pad[1:-1, 1:-1, :-2]
        ) / (dx * dx)
        lap_y = (
            pad[1:-1, 2:, 1:-1]
            - 2.0 * pad[1:-1, 1:-1, 1:-1]
            + pad[1:-1, :-2, 1:-1]
        ) / (dy * dy)
        lap_z = (
            pad[2:, 1:-1, 1:-1]
            - 2.0 * pad[1:-1, 1:-1, 1:-1]
            + pad[:-2, 1:-1, 1:-1]
        ) / (dz * dz)
        return lap_x + lap_y + lap_z
    raise ValueError(f"Unsupported field dimension: {field.ndim}")


def _to_center(component, target_shape):
    """Map staggered component arrays to cell centers via edge-padding averages."""
    if component is None:
        return None
    if component.shape == target_shape:
        return component

    centered = component
    for axis, (cur, target) in enumerate(zip(component.shape, target_shape)):
        if cur == target:
            continue
        if cur != target - 1:
            raise ValueError(
                f"Cannot center component with shape {component.shape} to {target_shape}"
            )
        pad_before = [0] * component.ndim
        pad_after = [0] * component.ndim
        pad_before[axis] = 1
        pad_after[axis] = 0
        pad_left = jnp.pad(centered, list(zip(pad_before, pad_after)), mode="edge")
        pad_before[axis] = 0
        pad_after[axis] = 1
        pad_right = jnp.pad(centered, list(zip(pad_before, pad_after)), mode="edge")
        centered = 0.5 * (pad_left + pad_right)

    return centered


class ThermoPhysics:
    def __init__(self, params: ThermalParams, enabled: bool = True):
        self.params = params
        self.enabled = enabled
        self.initialized = False

        self.base_eps_r = None
        self.k = None
        self.rho = None
        self.cp = None
        self.dn_dT = None
        self.T0 = None

        self.T = None
        self.E2_avg = None
        self.t_accum = 0.0

        self.dx = None
        self.dy = None
        self.dz = None

    def initialize(self, sim):
        """Initialize thermal grids and state from the simulation."""
        if not self.enabled:
            return

        self.base_eps_r = jnp.asarray(sim.fields.permittivity)
        thermal_grids = sim.design.get_thermal_grids(sim.resolution)
        if thermal_grids is None:
            raise ValueError("Thermal grids not available on design.")

        k, rho, cp, dn_dT, T0 = thermal_grids
        self.k = self._apply_default(jnp.asarray(k), self.params.k)
        self.rho = self._apply_default(jnp.asarray(rho), self.params.rho)
        self.cp = self._apply_default(jnp.asarray(cp), self.params.cp)
        self.dn_dT = self._apply_default(jnp.asarray(dn_dT), self.params.dn_dT)
        self.T0 = self._apply_default(jnp.asarray(T0), self.params.T0)

        self.T = jnp.array(self.T0)
        self.E2_avg = jnp.zeros_like(self.T)
        self.t_accum = 0.0

        self.dx = sim.resolution
        self.dy = sim.resolution
        self.dz = getattr(sim.fields, "resolution", sim.resolution)
        self.initialized = True

    def _apply_default(self, grid, default):
        if default is None:
            return grid
        if default == 0.0:
            return grid
        return jnp.where(grid == 0, default, grid)

    def step(self, sim):
        """Advance thermal state and update permittivity."""
        if not self.enabled:
            return
        if not self.initialized:
            self.initialize(sim)

        dt = sim.dt
        tau_avg = self.params.tau_avg
        if tau_avg <= 0:
            tau_avg = None

        E2 = self._compute_e2(sim.fields, self.T.shape)
        if tau_avg is None:
            self.E2_avg = E2
        else:
            alpha = dt / tau_avg
            self.E2_avg = self.E2_avg + alpha * (E2 - self.E2_avg)

        sigma = sim.fields.conductivity
        Q = sigma * self.E2_avg

        thermal_dt = self.params.thermal_dt
        if thermal_dt <= 0:
            return

        self.t_accum += dt
        num_steps = int(self.t_accum // thermal_dt)
        if num_steps <= 0:
            return

        for _ in range(num_steps):
            lap = _laplacian_neumann(self.T, self.dx, self.dy, self.dz)
            source = self.k * lap + Q
            denom = self.rho * self.cp
            update = jnp.where(denom > 0, (thermal_dt / denom) * source, 0.0)
            self.T = self.T + update

        self.t_accum -= num_steps * thermal_dt

        n0 = jnp.sqrt(self.base_eps_r)
        n = n0 + self.dn_dT * (self.T - self.T0)
        eps_r = n * n
        sim.fields.update_materials(permittivity=eps_r)

    def _compute_e2(self, fields, target_shape):
        """Compute |E|^2 on the cell-centered grid."""
        components = []
        for name in ("Ex", "Ey", "Ez"):
            if hasattr(fields, name):
                components.append(_to_center(getattr(fields, name), target_shape))

        if not components:
            return jnp.zeros(target_shape)

        E2 = jnp.zeros(target_shape)
        for comp in components:
            if comp is not None:
                E2 = E2 + comp * comp
        return E2
