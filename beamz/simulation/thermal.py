from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

import jax.numpy as jnp
import numpy as np


@dataclass
class ThermalConfig:
    thermal_dt: float
    tau_avg: float
    k: float = 0.0
    rho: float = 0.0
    cp: float = 0.0
    dn_dT: float = 0.0
    T0: float = 300.0
    max_iters: int = 5000
    tol: float = 1e-6
    robin_h: float = 0.0
    robin_T_ambient: float = 300.0
    robin_sides: tuple[str, ...] = ()


@dataclass
class StaticThermalResult:
    permittivity: np.ndarray
    temperature: np.ndarray


@dataclass
class StaticThermalConfig:
    k: float = 0.0
    rho: float = 0.0
    cp: float = 0.0
    dn_dT: float = 0.0
    T0: float = 300.0
    max_iters: int = 5000
    tol: float = 1e-6


@dataclass
class ThermalSource:
    region: object
    power_w: float | None = None
    power_density_w_m3: float | np.ndarray | None = None
    name: str | None = None


@dataclass
class ThermalSink:
    region: object
    temperature_k: float
    name: str | None = None


@dataclass
class ConvectionBC:
    h_w_m2_k: float
    ambient_temp_k: float = 300.0
    sides: tuple[str, ...] = ("top",)


@dataclass
class ThermalBoundaryProfile:
    sinks: list[ThermalSink] = field(default_factory=list)
    convection: ConvectionBC | None = None

    @classmethod
    def photonic_chip(
        cls,
        sink_thickness_m: float = 0.2e-6,
        sink_temperature_k: float = 300.0,
        top_h_w_m2_k: float = 10.0,
        ambient_temp_k: float = 300.0,
    ) -> "ThermalBoundaryProfile":
        def substrate_sink(x, y, z):
            _ = x, z
            return 0.0 <= y <= sink_thickness_m

        return cls(
            sinks=[
                ThermalSink(
                    region=substrate_sink,
                    temperature_k=sink_temperature_k,
                    name="substrate_sink",
                )
            ],
            convection=ConvectionBC(
                h_w_m2_k=top_h_w_m2_k,
                ambient_temp_k=ambient_temp_k,
                sides=("top",),
            ),
        )


@dataclass
class ThermalScenario:
    sources: list[ThermalSource] = field(default_factory=list)
    sinks: list[ThermalSink] = field(default_factory=list)
    convection: ConvectionBC | None = None
    boundary_profile: ThermalBoundaryProfile | None = None
    extrusion_depth_m: float | None = None

    def iter_sources(self):
        for source in self.sources:
            if not isinstance(source, ThermalSource):
                raise TypeError(
                    "ThermalScenario.sources must contain ThermalSource objects."
                )
            yield source

    def iter_sinks(self):
        profile_sinks = (
            self.boundary_profile.sinks if self.boundary_profile is not None else []
        )
        for sink in [*profile_sinks, *self.sinks]:
            if not isinstance(sink, ThermalSink):
                raise TypeError(
                    "ThermalScenario.sinks must contain ThermalSink objects."
                )
            yield sink

    def get_convection(self):
        if self.convection is not None:
            return self.convection
        if self.boundary_profile is not None:
            return self.boundary_profile.convection
        return None


@dataclass
class MZITuningResult:
    power_w: np.ndarray
    delta_t_eff_k: np.ndarray
    delta_n_eff: np.ndarray
    delta_phi_rad: np.ndarray
    p_pi_w: float
    temperature_at_power: dict[float, np.ndarray] | None = None


def _harmonic_mean_jax(left, right):
    denom = left + right
    valid = (left > 0) & (right > 0) & (denom > 0)
    return jnp.where(valid, 2.0 * left * right / denom, 0.0)


def _harmonic_mean_np(left, right):
    denom = left + right
    out = np.zeros_like(denom, dtype=float)
    valid = (left > 0) & (right > 0) & (denom > 0)
    np.divide(2.0 * left * right, denom, out=out, where=valid)
    return out


def _div_k_grad_neumann(field, k, dx, dy=None, dz=None):
    """Compute div(k grad T) with zero-flux Neumann boundary conditions."""
    if dy is None:
        dy = dx
    if field.ndim == 2:
        kx_face = _harmonic_mean_jax(k[:, :-1], k[:, 1:])
        ky_face = _harmonic_mean_jax(k[:-1, :], k[1:, :])

        grad_x_face = kx_face * (field[:, 1:] - field[:, :-1]) / dx
        grad_y_face = ky_face * (field[1:, :] - field[:-1, :]) / dy

        div = jnp.zeros_like(field)
        div = div.at[:, :-1].add(grad_x_face / dx)
        div = div.at[:, 1:].add(-grad_x_face / dx)
        div = div.at[:-1, :].add(grad_y_face / dy)
        div = div.at[1:, :].add(-grad_y_face / dy)
        return div
    if field.ndim == 3:
        if dz is None:
            dz = dx
        kx_face = _harmonic_mean_jax(k[:, :, :-1], k[:, :, 1:])
        ky_face = _harmonic_mean_jax(k[:, :-1, :], k[:, 1:, :])
        kz_face = _harmonic_mean_jax(k[:-1, :, :], k[1:, :, :])

        grad_x_face = kx_face * (field[:, :, 1:] - field[:, :, :-1]) / dx
        grad_y_face = ky_face * (field[:, 1:, :] - field[:, :-1, :]) / dy
        grad_z_face = kz_face * (field[1:, :, :] - field[:-1, :, :]) / dz

        div = jnp.zeros_like(field)
        div = div.at[:, :, :-1].add(grad_x_face / dx)
        div = div.at[:, :, 1:].add(-grad_x_face / dx)
        div = div.at[:, :-1, :].add(grad_y_face / dy)
        div = div.at[:, 1:, :].add(-grad_y_face / dy)
        div = div.at[:-1, :, :].add(grad_z_face / dz)
        div = div.at[1:, :, :].add(-grad_z_face / dz)
        return div
    raise ValueError(f"Unsupported field dimension: {field.ndim}")


def _steady_state_jacobi_step(
    T,
    k_grid,
    Q,
    dx,
    dy=None,
    dz=None,
    robin_h=0.0,
    robin_t_ambient=300.0,
    robin_sides=(),
):
    """One Jacobi iteration of div(k grad T) + Q = 0 with Neumann boundaries."""
    if dy is None:
        dy = dx

    if T.ndim == 2:
        kx_face = _harmonic_mean_np(k_grid[:, :-1], k_grid[:, 1:]) / (dx * dx)
        ky_face = _harmonic_mean_np(k_grid[:-1, :], k_grid[1:, :]) / (dy * dy)

        numer = np.array(Q, dtype=float, copy=True)
        denom = np.zeros_like(T, dtype=float)

        numer[:, :-1] += kx_face * T[:, 1:]
        numer[:, 1:] += kx_face * T[:, :-1]
        denom[:, :-1] += kx_face
        denom[:, 1:] += kx_face

        numer[:-1, :] += ky_face * T[1:, :]
        numer[1:, :] += ky_face * T[:-1, :]
        denom[:-1, :] += ky_face
        denom[1:, :] += ky_face

        numer, denom = _apply_robin_static_np(
            numer=numer,
            denom=denom,
            T=T,
            h=robin_h,
            t_ambient=robin_t_ambient,
            dx=dx,
            dy=dy,
            sides=robin_sides,
        )
        return np.where(denom > 0, numer / denom, T)

    if T.ndim == 3:
        if dz is None:
            dz = dx
        kx_face = _harmonic_mean_np(k_grid[:, :, :-1], k_grid[:, :, 1:]) / (dx * dx)
        ky_face = _harmonic_mean_np(k_grid[:, :-1, :], k_grid[:, 1:, :]) / (dy * dy)
        kz_face = _harmonic_mean_np(k_grid[:-1, :, :], k_grid[1:, :, :]) / (dz * dz)

        numer = np.array(Q, dtype=float, copy=True)
        denom = np.zeros_like(T, dtype=float)

        numer[:, :, :-1] += kx_face * T[:, :, 1:]
        numer[:, :, 1:] += kx_face * T[:, :, :-1]
        denom[:, :, :-1] += kx_face
        denom[:, :, 1:] += kx_face

        numer[:, :-1, :] += ky_face * T[:, 1:, :]
        numer[:, 1:, :] += ky_face * T[:, :-1, :]
        denom[:, :-1, :] += ky_face
        denom[:, 1:, :] += ky_face

        numer[:-1, :, :] += kz_face * T[1:, :, :]
        numer[1:, :, :] += kz_face * T[:-1, :, :]
        denom[:-1, :, :] += kz_face
        denom[1:, :, :] += kz_face

        numer, denom = _apply_robin_static_np(
            numer=numer,
            denom=denom,
            T=T,
            h=robin_h,
            t_ambient=robin_t_ambient,
            dx=dx,
            dy=dy,
            dz=dz,
            sides=robin_sides,
        )
        return np.where(denom > 0, numer / denom, T)

    raise ValueError(f"Unsupported temperature grid dimension: {T.ndim}")


def _validate_robin_sides(sides, ndim):
    valid = {"left", "right", "bottom", "top"}
    if ndim == 3:
        valid = valid | {"front", "back"}
    invalid = [s for s in sides if s not in valid]
    if invalid:
        raise ValueError(f"Unsupported robin_sides {invalid} for {ndim}D thermal grid")


def _apply_robin_source_jax(source, T, h, t_ambient, dx, dy=None, dz=None, sides=()):
    if h <= 0 or not sides:
        return source
    if dy is None:
        dy = dx
    _validate_robin_sides(sides, T.ndim)

    out = source
    if T.ndim == 2:
        if "left" in sides:
            out = out.at[:, 0].add(-(h / dx) * (T[:, 0] - t_ambient))
        if "right" in sides:
            out = out.at[:, -1].add(-(h / dx) * (T[:, -1] - t_ambient))
        if "bottom" in sides:
            out = out.at[0, :].add(-(h / dy) * (T[0, :] - t_ambient))
        if "top" in sides:
            out = out.at[-1, :].add(-(h / dy) * (T[-1, :] - t_ambient))
        return out

    if T.ndim == 3:
        if dz is None:
            dz = dx
        if "left" in sides:
            out = out.at[:, :, 0].add(-(h / dx) * (T[:, :, 0] - t_ambient))
        if "right" in sides:
            out = out.at[:, :, -1].add(-(h / dx) * (T[:, :, -1] - t_ambient))
        if "bottom" in sides:
            out = out.at[:, 0, :].add(-(h / dy) * (T[:, 0, :] - t_ambient))
        if "top" in sides:
            out = out.at[:, -1, :].add(-(h / dy) * (T[:, -1, :] - t_ambient))
        if "front" in sides:
            out = out.at[0, :, :].add(-(h / dz) * (T[0, :, :] - t_ambient))
        if "back" in sides:
            out = out.at[-1, :, :].add(-(h / dz) * (T[-1, :, :] - t_ambient))
        return out

    raise ValueError(f"Unsupported temperature grid dimension: {T.ndim}")


def _apply_robin_static_np(
    numer, denom, T, h, t_ambient, dx, dy=None, dz=None, sides=()
):
    if h <= 0 or not sides:
        return numer, denom
    if dy is None:
        dy = dx
    _validate_robin_sides(sides, T.ndim)

    if T.ndim == 2:
        if "left" in sides:
            denom[:, 0] += h / dx
            numer[:, 0] += (h / dx) * t_ambient
        if "right" in sides:
            denom[:, -1] += h / dx
            numer[:, -1] += (h / dx) * t_ambient
        if "bottom" in sides:
            denom[0, :] += h / dy
            numer[0, :] += (h / dy) * t_ambient
        if "top" in sides:
            denom[-1, :] += h / dy
            numer[-1, :] += (h / dy) * t_ambient
        return numer, denom

    if T.ndim == 3:
        if dz is None:
            dz = dx
        if "left" in sides:
            denom[:, :, 0] += h / dx
            numer[:, :, 0] += (h / dx) * t_ambient
        if "right" in sides:
            denom[:, :, -1] += h / dx
            numer[:, :, -1] += (h / dx) * t_ambient
        if "bottom" in sides:
            denom[:, 0, :] += h / dy
            numer[:, 0, :] += (h / dy) * t_ambient
        if "top" in sides:
            denom[:, -1, :] += h / dy
            numer[:, -1, :] += (h / dy) * t_ambient
        if "front" in sides:
            denom[0, :, :] += h / dz
            numer[0, :, :] += (h / dz) * t_ambient
        if "back" in sides:
            denom[-1, :, :] += h / dz
            numer[-1, :, :] += (h / dz) * t_ambient
        return numer, denom

    raise ValueError(f"Unsupported temperature grid dimension: {T.ndim}")


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


class ThermalCoupling:
    def __init__(self, config: ThermalConfig, enabled: bool = True):
        self.config = config
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
        self.robin_h = 0.0
        self.robin_t_ambient = 300.0
        self.robin_sides = ()

    def initialize(self, sim):
        """Initialize thermal grids and state from the simulation."""
        if not self.enabled:
            return

        self.base_eps_r = jnp.asarray(sim.fields.permittivity)
        thermal_grids = sim.design.get_thermal_grids(sim.resolution)
        if thermal_grids is None:
            raise ValueError("Thermal grids not available on design.")

        k, rho, cp, dn_dT, T0 = thermal_grids
        self.k = self._apply_default(jnp.asarray(k), self.config.k)
        self.rho = self._apply_default(jnp.asarray(rho), self.config.rho)
        self.cp = self._apply_default(jnp.asarray(cp), self.config.cp)
        self.dn_dT = self._apply_default(jnp.asarray(dn_dT), self.config.dn_dT)
        self.T0 = self._apply_default(jnp.asarray(T0), self.config.T0)

        self.T = jnp.array(self.T0)
        self.E2_avg = jnp.zeros_like(self.T)
        self.t_accum = 0.0

        self.dx = sim.resolution
        self.dy = sim.resolution
        self.dz = getattr(sim.fields, "resolution", sim.resolution)
        self.robin_h = float(getattr(self.config, "robin_h", 0.0))
        self.robin_t_ambient = float(
            getattr(self.config, "robin_T_ambient", self.config.T0)
        )
        self.robin_sides = tuple(getattr(self.config, "robin_sides", ()))
        _validate_robin_sides(self.robin_sides, self.T.ndim)
        self.initialized = True

    def _apply_default(self, grid, default):
        if default is None:
            return grid
        if default == 0.0:
            return grid
        return jnp.where(grid == 0, default, grid)

    def _stable_substep_dt(self):
        denom = self.rho * self.cp
        alpha = jnp.where(denom > 0, self.k / denom, 0.0)
        alpha_max = float(jnp.max(alpha))
        if alpha_max <= 0:
            return None

        inv_dx2 = 1.0 / (self.dx * self.dx)
        inv_dy2 = 1.0 / (self.dy * self.dy)
        inv_sum = inv_dx2 + inv_dy2
        if self.T.ndim == 3:
            inv_dz2 = 1.0 / (self.dz * self.dz)
            inv_sum += inv_dz2

        if inv_sum <= 0:
            return None

        return 0.5 / (alpha_max * inv_sum)

    def step(self, sim):
        """Advance thermal state and update permittivity."""
        if not self.enabled:
            return
        if not self.initialized:
            self.initialize(sim)

        dt = sim.dt
        tau_avg = self.config.tau_avg
        if tau_avg <= 0:
            tau_avg = None

        E2 = self._compute_e2(sim.fields, self.T.shape)
        if tau_avg is None:
            self.E2_avg = E2
        else:
            beta = 1.0 - math.exp(-dt / tau_avg)
            self.E2_avg = self.E2_avg + beta * (E2 - self.E2_avg)

        sigma = sim.fields.conductivity
        Q = sigma * self.E2_avg

        thermal_dt = self.config.thermal_dt
        if thermal_dt <= 0:
            return

        self.t_accum += dt
        num_steps = int(self.t_accum // thermal_dt)
        if num_steps <= 0:
            return

        stable_dt = self._stable_substep_dt()
        if stable_dt is None:
            substeps = 1
        else:
            substeps = max(1, int(math.ceil(thermal_dt / stable_dt)))
        sub_dt = thermal_dt / substeps

        for _ in range(num_steps):
            for _ in range(substeps):
                div_k_grad = _div_k_grad_neumann(
                    self.T, self.k, self.dx, self.dy, self.dz
                )
                source = div_k_grad + Q
                source = _apply_robin_source_jax(
                    source=source,
                    T=self.T,
                    h=self.robin_h,
                    t_ambient=self.robin_t_ambient,
                    dx=self.dx,
                    dy=self.dy,
                    dz=self.dz,
                    sides=self.robin_sides,
                )
                denom = self.rho * self.cp
                update = jnp.where(denom > 0, (sub_dt / denom) * source, 0.0)
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


def _apply_default_np(grid, default):
    if default is None:
        return grid
    if default == 0.0:
        return grid
    return np.where(grid == 0, default, grid)


def _iter_cell_centers(shape, resolution):
    if len(shape) == 2:
        ny, nx = shape
        for i in range(ny):
            y = (i + 0.5) * resolution
            for j in range(nx):
                x = (j + 0.5) * resolution
                yield (i, j), (x, y, 0.0)
    elif len(shape) == 3:
        nz, ny, nx = shape
        for k in range(nz):
            z = (k + 0.5) * resolution
            for i in range(ny):
                y = (i + 0.5) * resolution
                for j in range(nx):
                    x = (j + 0.5) * resolution
                    yield (k, i, j), (x, y, z)
    else:
        raise ValueError(f"Unsupported thermal grid shape: {shape}")


def _sample_region_mask(shape, resolution, predicate):
    mask = np.zeros(shape, dtype=bool)
    for idx, xyz in _iter_cell_centers(shape, resolution):
        mask[idx] = bool(predicate(*xyz))
    return mask


def _resolve_region_mask(design, resolution, shape, region):
    _ = design
    if region is None:
        return np.zeros(shape, dtype=bool)

    if isinstance(region, np.ndarray):
        mask = np.asarray(region).astype(bool)
        if mask.shape != shape:
            raise ValueError(
                f"Region shape {mask.shape} does not match grid shape {shape}"
            )
        return mask

    if hasattr(region, "point_in_polygon"):
        return _sample_region_mask(
            shape,
            resolution,
            lambda x, y, z: region.point_in_polygon(x, y, z),
        )

    if callable(region):
        return _sample_region_mask(shape, resolution, region)

    if isinstance(region, Iterable) and not isinstance(region, (str, bytes)):
        mask = np.zeros(shape, dtype=bool)
        for part in region:
            mask |= _resolve_region_mask(design, resolution, shape, part)
        return mask

    raise TypeError(
        "Thermal region must be a structure reference, callable, bool array, "
        "or iterable of those."
    )


def _resolve_weight_array(shape, mode_weight):
    if mode_weight is None:
        return np.ones(shape, dtype=float)
    weights = np.asarray(mode_weight, dtype=float)
    if weights.shape != shape:
        raise ValueError(
            f"mode_weight shape {weights.shape} does not match thermal grid shape {shape}"
        )
    return weights


def _source_power_density(
    source, active_cells, resolution, ndim, extrusion_depth_m, shape
):
    if source.power_density_w_m3 is not None and source.power_w is not None:
        raise ValueError(
            "ThermalSource can specify either power_w or power_density_w_m3, not both."
        )

    if source.power_density_w_m3 is not None:
        density = np.asarray(source.power_density_w_m3, dtype=float)
        if density.ndim == 0:
            return float(density)
        if density.shape != shape:
            raise ValueError(
                f"Power density shape {density.shape} does not match thermal grid shape {shape}"
            )
        return density

    if source.power_w is None:
        return None

    if active_cells <= 0:
        return None

    if ndim == 2:
        if extrusion_depth_m is None or extrusion_depth_m <= 0:
            raise ValueError(
                "2D total-power sources require scenario.extrusion_depth_m > 0. "
                "Set extrusion depth to convert heater power (W) to volumetric source (W/m^3)."
            )
        cell_volume = resolution * resolution * extrusion_depth_m
    else:
        cell_volume = resolution**3

    return float(source.power_w) / (active_cells * cell_volume)


def _copy_scenario_with_sources(scenario, sources):
    return ThermalScenario(
        sources=list(sources),
        sinks=list(scenario.sinks),
        convection=scenario.convection,
        boundary_profile=scenario.boundary_profile,
        extrusion_depth_m=scenario.extrusion_depth_m,
    )


class StaticThermalSolver:
    def __init__(self, config: StaticThermalConfig | None = None):
        self.config = config or StaticThermalConfig()

    def solve(
        self, design, resolution, scenario: ThermalScenario
    ) -> StaticThermalResult:
        if not isinstance(scenario, ThermalScenario):
            raise TypeError("solve_thermal expects a ThermalScenario object.")

        thermal_grids = design.get_thermal_grids(resolution)
        if thermal_grids is None:
            raise ValueError("Thermal grids not available on design.")

        k_grid, _rho_grid, _cp_grid, dn_dT_grid, T0_grid = thermal_grids
        k_grid = _apply_default_np(np.asarray(k_grid), self.config.k)
        dn_dT_grid = _apply_default_np(np.asarray(dn_dT_grid), self.config.dn_dT)
        T0_grid = _apply_default_np(np.asarray(T0_grid), self.config.T0)

        Q = self._build_heat_source(design, resolution, k_grid.shape, scenario)
        fixed_mask, fixed_values = self._build_fixed_temperature(
            design, resolution, k_grid.shape, scenario
        )
        convection = scenario.get_convection()
        robin_h = 0.0
        robin_t_ambient = float(self.config.T0)
        robin_sides = ()
        if convection is not None:
            robin_h = float(convection.h_w_m2_k)
            robin_t_ambient = float(convection.ambient_temp_k)
            robin_sides = tuple(convection.sides)
            _validate_robin_sides(robin_sides, k_grid.ndim)

        T = np.array(T0_grid, dtype=float)
        if fixed_mask is not None:
            T = np.where(fixed_mask, fixed_values, T)

        max_iters = int(self.config.max_iters)
        tol = float(self.config.tol)

        for _ in range(max_iters):
            T_new = _steady_state_jacobi_step(
                T=T,
                k_grid=k_grid,
                Q=Q,
                dx=resolution,
                robin_h=robin_h,
                robin_t_ambient=robin_t_ambient,
                robin_sides=robin_sides,
            )
            if fixed_mask is not None:
                T_new = np.where(fixed_mask, fixed_values, T_new)
            delta = np.max(np.abs(T_new - T))
            T = T_new
            if delta < tol:
                break

        n0 = np.sqrt(np.asarray(design.get_material_grids(resolution)[0]))
        n = n0 + dn_dT_grid * (T - T0_grid)
        eps_r = n * n
        return StaticThermalResult(permittivity=eps_r, temperature=T)

    def _build_heat_source(self, design, resolution, shape, scenario):
        source_total = np.zeros(shape, dtype=float)
        for source in scenario.iter_sources():
            mask = _resolve_region_mask(design, resolution, shape, source.region)
            active_cells = int(np.count_nonzero(mask))
            density = _source_power_density(
                source=source,
                active_cells=active_cells,
                resolution=resolution,
                ndim=len(shape),
                extrusion_depth_m=scenario.extrusion_depth_m,
                shape=shape,
            )
            if density is None:
                continue
            source_total = source_total + density * mask.astype(float)
        return source_total

    def _build_fixed_temperature(self, design, resolution, shape, scenario):
        fixed_mask = np.zeros(shape, dtype=bool)
        fixed_values = np.zeros(shape, dtype=float)
        has_sink = False
        for sink in scenario.iter_sinks():
            sink_mask = _resolve_region_mask(design, resolution, shape, sink.region)
            if not np.any(sink_mask):
                continue
            fixed_mask |= sink_mask
            fixed_values[sink_mask] = float(sink.temperature_k)
            has_sink = True
        if not has_sink:
            return None, None
        return fixed_mask, fixed_values


def solve_static_thermal(
    design,
    resolution: float,
    scenario: ThermalScenario,
    config: StaticThermalConfig | None = None,
) -> StaticThermalResult:
    solver = StaticThermalSolver(config=config)
    return solver.solve(design=design, resolution=resolution, scenario=scenario)


def solve_thermal(
    design,
    resolution: float,
    scenario: ThermalScenario,
    config: StaticThermalConfig | None = None,
) -> StaticThermalResult:
    return solve_static_thermal(
        design=design, resolution=resolution, scenario=scenario, config=config
    )


def sweep_mzi_heater(
    design,
    resolution: float,
    powers_w: np.ndarray,
    heater: ThermalSource | list[ThermalSource],
    optical_region,
    arm_length_m: float,
    wavelength_m: float,
    group_index: float,
    scenario_base: ThermalScenario,
    mode_weight: np.ndarray | None = None,
    config: StaticThermalConfig | None = None,
) -> MZITuningResult:
    powers = np.asarray(powers_w, dtype=float).reshape(-1)
    if powers.size == 0:
        raise ValueError("powers_w must contain at least one power point.")
    if wavelength_m <= 0 or arm_length_m <= 0 or group_index <= 0:
        raise ValueError("wavelength_m, arm_length_m and group_index must be > 0.")
    if not isinstance(scenario_base, ThermalScenario):
        raise TypeError("scenario_base must be a ThermalScenario.")

    if isinstance(heater, ThermalSource):
        heater_sources = [heater]
    elif isinstance(heater, Iterable):
        heater_sources = list(heater)
        if not heater_sources or not all(
            isinstance(source, ThermalSource) for source in heater_sources
        ):
            raise TypeError("heater list must contain ThermalSource objects.")
    else:
        raise TypeError("heater must be a ThermalSource or list[ThermalSource].")

    thermal_grids = design.get_thermal_grids(resolution)
    if thermal_grids is None:
        raise ValueError("Thermal grids not available on design.")
    k_grid, _rho, _cp, dn_dT_grid_raw, _T0 = thermal_grids
    grid_shape = np.asarray(k_grid).shape

    dn_dT_grid = np.asarray(dn_dT_grid_raw)
    cfg = config or StaticThermalConfig()
    dn_dT_grid = _apply_default_np(dn_dT_grid, cfg.dn_dT)

    optical_mask = _resolve_region_mask(design, resolution, grid_shape, optical_region)
    if not np.any(optical_mask):
        raise ValueError("optical_region resolved to an empty mask.")
    weights = _resolve_weight_array(grid_shape, mode_weight)
    weights = np.where(optical_mask, weights, 0.0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        raise ValueError("optical_region/mode_weight produced zero effective weight.")

    temperatures = []
    temperature_at_power = {}
    n_heaters = len(heater_sources)
    for idx, power in enumerate(powers):
        power_per_heater = float(power) / n_heaters
        dynamic_sources = []
        for src in heater_sources:
            dynamic_sources.append(
                replace(src, power_w=power_per_heater, power_density_w_m3=None)
            )
        scenario_iter = _copy_scenario_with_sources(
            scenario_base, [*scenario_base.sources, *dynamic_sources]
        )
        result = solve_static_thermal(
            design=design,
            resolution=resolution,
            scenario=scenario_iter,
            config=cfg,
        )
        T = np.asarray(result.temperature)
        temperatures.append(T)
        if idx in {0, len(powers) - 1}:
            temperature_at_power[float(power)] = T

    baseline = temperatures[0]
    delta_t_eff = np.zeros_like(powers, dtype=float)
    delta_n_eff = np.zeros_like(powers, dtype=float)
    delta_phi = np.zeros_like(powers, dtype=float)

    phase_scale = (2.0 * np.pi / wavelength_m) * group_index * arm_length_m
    for i, temp in enumerate(temperatures):
        delta_t = temp - baseline
        delta_t_eff[i] = float(np.sum(delta_t * weights) / weight_sum)
        delta_n = dn_dT_grid * delta_t
        delta_n_eff[i] = float(np.sum(delta_n * weights) / weight_sum)
        delta_phi[i] = phase_scale * delta_n_eff[i]

    p_pi = float("nan")
    if np.any(np.diff(delta_phi) != 0):
        slope = float(np.polyfit(powers, delta_phi, 1)[0])
        if slope > 0:
            p_pi = float(np.pi / slope)
        if np.all(np.diff(delta_phi) >= 0) and delta_phi[-1] >= np.pi:
            p_pi = float(np.interp(np.pi, delta_phi, powers))

    return MZITuningResult(
        power_w=powers,
        delta_t_eff_k=delta_t_eff,
        delta_n_eff=delta_n_eff,
        delta_phi_rad=delta_phi,
        p_pi_w=p_pi,
        temperature_at_power=temperature_at_power or None,
    )


__all__ = [
    "ThermalConfig",
    "ThermalCoupling",
    "StaticThermalConfig",
    "StaticThermalResult",
    "ThermalSource",
    "ThermalSink",
    "ConvectionBC",
    "ThermalBoundaryProfile",
    "ThermalScenario",
    "MZITuningResult",
    "StaticThermalSolver",
    "solve_thermal",
    "solve_static_thermal",
    "sweep_mzi_heater",
]
