from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass
class ThermalParams:
    thermal_dt: float
    tau_avg: float
    k: float = 0.0
    rho: float = 0.0
    cp: float = 0.0
    dn_dT: float = 0.0
    T0: float = 300.0
    steady_state: bool = False
    max_iters: int = 5000
    tol: float = 1e-6


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

        thermal_grids = sim.design.get_thermal_grids(sim.resolution)
        if thermal_grids is None:
            raise ValueError("Thermal grids not available on design.")

        k, rho, cp, dn_dT, T0 = thermal_grids
        props = sim.design.evaluate_materials(sim.resolution, None)
        self.base_eps_r = props["permittivity"]
        self.k = self._apply_default(props["k"], self.params.k)
        self.rho = self._apply_default(props["rho"], self.params.rho)
        self.cp = self._apply_default(props["cp"], self.params.cp)
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

        props = sim.design.evaluate_materials(sim.resolution, self.T)
        sigma = props["conductivity"]
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
            props = sim.design.evaluate_materials(sim.resolution, self.T)
            k = props["k"]
            rho = props["rho"]
            cp = props["cp"]
            source = k * lap + Q
            denom = rho * cp
            update = jnp.where(denom > 0, (thermal_dt / denom) * source, 0.0)
            self.T = self.T + update

        self.t_accum -= num_steps * thermal_dt

        props_end = sim.design.evaluate_materials(sim.resolution, self.T)
        if jnp.all(self.dn_dT == 0):
            eps_r = props_end["permittivity"]
        else:
            n0 = jnp.sqrt(self.base_eps_r)
            n = n0 + self.dn_dT * (self.T - self.T0)
            eps_r = n * n
        sim.fields.update_materials(
            permittivity=eps_r,
            conductivity=props_end["conductivity"],
            permeability=props_end["permeability"],
        )

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


class StaticThermalSolve:
    def __init__(
        self,
        params: ThermalParams,
        heater_mask=None,
        heater_power=0.0,
        fixed_temp_mask=None,
        fixed_temp_value=None,
    ):
        self.params = params
        self.heater_mask = heater_mask
        self.heater_power = heater_power
        self.fixed_temp_mask = fixed_temp_mask
        self.fixed_temp_value = fixed_temp_value

    def solve(self, design, resolution):
        thermal_grids = design.get_thermal_grids(resolution)
        if thermal_grids is None:
            raise ValueError("Thermal grids not available on design.")

        k_grid, rho_grid, cp_grid, dn_dT_grid, T0_grid = thermal_grids
        k_grid = self._apply_default(np.asarray(k_grid), self.params.k)
        dn_dT_grid = self._apply_default(np.asarray(dn_dT_grid), self.params.dn_dT)
        T0_grid = self._apply_default(np.asarray(T0_grid), self.params.T0)

        Q = self._build_heat_source(design, resolution, k_grid.shape)
        fixed_mask = self._build_mask(design, resolution, k_grid.shape, self.fixed_temp_mask)
        fixed_value = self.fixed_temp_value
        T = np.array(T0_grid, dtype=float)
        if fixed_mask is not None and fixed_value is not None:
            T = np.where(fixed_mask, fixed_value, T)

        max_iters = int(self.params.max_iters)
        tol = float(self.params.tol)

        for _ in range(max_iters):
            T_new = self._steady_state_step(T, k_grid, Q, resolution)
            if fixed_mask is not None and fixed_value is not None:
                T_new = np.where(fixed_mask, fixed_value, T_new)
            delta = np.max(np.abs(T_new - T))
            T = T_new
            if delta < tol:
                break

        # Build permittivity from temperature-dependent materials to avoid sqrt on metals
        props = design.evaluate_materials(resolution, T)
        eps_r = np.asarray(props["permittivity"])
        return eps_r, T

    def _apply_default(self, grid, default):
        if default is None:
            return grid
        if default == 0.0:
            return grid
        return np.where(grid == 0, default, grid)

    def _build_heat_source(self, design, resolution, shape):
        if self.heater_mask is None:
            return np.zeros(shape)
        mask = self._build_mask(design, resolution, shape, self.heater_mask)
        return self.heater_power * mask

    def _build_mask(self, design, resolution, shape, mask_def):
        if mask_def is None:
            return None
        if callable(mask_def):
            if len(shape) == 2:
                ny, nx = shape
                x_centers = (np.arange(nx) + 0.5) * resolution
                y_centers = (np.arange(ny) + 0.5) * resolution
                mask = np.zeros(shape, dtype=bool)
                for i, y in enumerate(y_centers):
                    for j, x in enumerate(x_centers):
                        mask[i, j] = bool(mask_def(x, y, 0.0))
            else:
                nz, ny, nx = shape
                x_centers = (np.arange(nx) + 0.5) * resolution
                y_centers = (np.arange(ny) + 0.5) * resolution
                z_centers = (np.arange(nz) + 0.5) * resolution
                mask = np.zeros(shape, dtype=bool)
                for k, z in enumerate(z_centers):
                    for i, y in enumerate(y_centers):
                        for j, x in enumerate(x_centers):
                            mask[k, i, j] = bool(mask_def(x, y, z))
        else:
            mask = np.asarray(mask_def).astype(bool)
            if mask.shape != shape:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match grid shape {shape}"
                )
        return mask

    def _steady_state_step(self, T, k_grid, Q, dx):
        if T.ndim == 2:
            pad = np.pad(T, ((1, 1), (1, 1)), mode="edge")
            neighbor_sum = (
                pad[1:-1, 2:] + pad[1:-1, :-2] + pad[2:, 1:-1] + pad[:-2, 1:-1]
            )
            denom = np.where(k_grid > 0, 4.0, 1.0)
            rhs = np.zeros_like(Q, dtype=float)
            np.divide(Q * dx * dx, k_grid, out=rhs, where=k_grid > 0)
            updated = (neighbor_sum + rhs) / denom
            return np.where(k_grid > 0, updated, T)
        if T.ndim == 3:
            pad = np.pad(T, ((1, 1), (1, 1), (1, 1)), mode="edge")
            neighbor_sum = (
                pad[1:-1, 1:-1, 2:]
                + pad[1:-1, 1:-1, :-2]
                + pad[1:-1, 2:, 1:-1]
                + pad[1:-1, :-2, 1:-1]
                + pad[2:, 1:-1, 1:-1]
                + pad[:-2, 1:-1, 1:-1]
            )
            denom = np.where(k_grid > 0, 6.0, 1.0)
            rhs = np.zeros_like(Q, dtype=float)
            np.divide(Q * dx * dx, k_grid, out=rhs, where=k_grid > 0)
            updated = (neighbor_sum + rhs) / denom
            return np.where(k_grid > 0, updated, T)
        raise ValueError(f"Unsupported temperature grid dimension: {T.ndim}")


def apply_static_thermal(
    design,
    resolution,
    params,
    heater_mask,
    heater_power,
    fixed_temp_mask=None,
    fixed_temp_value=None,
):
    solver = StaticThermalSolve(
        params=params,
        heater_mask=heater_mask,
        heater_power=heater_power,
        fixed_temp_mask=fixed_temp_mask,
        fixed_temp_value=fixed_temp_value,
    )
    return solver.solve(design, resolution)
