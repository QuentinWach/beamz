from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class MaterialEvaluation:
    permittivity: jnp.ndarray
    permeability: jnp.ndarray
    conductivity: jnp.ndarray
    k: jnp.ndarray
    rho: jnp.ndarray
    cp: jnp.ndarray
    T0: jnp.ndarray


class MaterialGridEvaluator:
    """Evaluate temperature-dependent material properties on a material-id grid."""

    def __init__(self, material_id, material_table, base_grids=None):
        self.material_id = jnp.asarray(material_id)
        self.material_table = material_table or []
        self.base_grids = base_grids or {}

    def _init_grid(self, key, default_value):
        if key in self.base_grids:
            return jnp.asarray(self.base_grids[key])
        return jnp.full_like(self.material_id, default_value, dtype=jnp.float32)

    def get_T0_grid(self):
        grid = self._init_grid("T0", 300.0)
        for idx, mat in enumerate(self.material_table):
            t0 = getattr(mat, "T_ref", 300.0)
            grid = jnp.where(self.material_id == idx, t0, grid)
        return grid

    def _evaluate_property(self, T, getter, base_key, default_value):
        grid = self._init_grid(base_key, default_value)
        for idx, mat in enumerate(self.material_table):
            mask = self.material_id == idx
            values = getter(mat, T)
            if not hasattr(values, "shape") or values.shape == ():
                values = jnp.full_like(T, values)
            grid = jnp.where(mask, values, grid)
        return grid

    def evaluate(self, T, omega=None) -> MaterialEvaluation:
        T = jnp.asarray(T)
        permittivity = self._evaluate_property(
            T,
            lambda m, t: m.epsilon_r(t, omega),
            "permittivity",
            1.0,
        )
        permeability = self._evaluate_property(
            T,
            lambda m, t: m.permeability(t, omega),
            "permeability",
            1.0,
        )
        conductivity = self._evaluate_property(
            T,
            lambda m, t: m.conductivity(t, omega),
            "conductivity",
            0.0,
        )
        k = self._evaluate_property(T, lambda m, t: m.thermal_k(t), "k", 0.0)
        rho = self._evaluate_property(T, lambda m, t: m.density(t), "rho", 0.0)
        cp = self._evaluate_property(
            T, lambda m, t: m.heat_capacity(t), "cp", 0.0
        )
        T0 = self.get_T0_grid()
        return MaterialEvaluation(
            permittivity=permittivity,
            permeability=permeability,
            conductivity=conductivity,
            k=k,
            rho=rho,
            cp=cp,
            T0=T0,
        )
