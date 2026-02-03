import numpy as np

import jax.numpy as jnp

from beamz.design.material_evaluator import MaterialGridEvaluator
from beamz.design.materials import (
    CustomMaterial,
    LinearThermoOpticMaterial,
    Material,
    as_material_model,
)


def test_material_model_linear_thermooptic():
    mat = LinearThermoOpticMaterial(n0=2.0, dn_dT=1.0e-3, T_ref=300.0)
    eps = mat.epsilon_r(jnp.array(310.0))
    expected_n = 2.0 + 1.0e-3 * 10.0
    assert np.isclose(eps, expected_n**2)


def test_material_model_thermal_k_rho_cp_linear():
    mat = LinearThermoOpticMaterial(
        n0=1.5,
        dn_dT=0.0,
        k0=10.0,
        dk_dT=0.1,
        rho0=2.0,
        drho_dT=0.2,
        cp0=5.0,
        dcp_dT=0.5,
        T_ref=300.0,
    )
    T = jnp.array(310.0)
    assert np.isclose(mat.thermal_k(T), 11.0)
    assert np.isclose(mat.density(T), 4.0)
    assert np.isclose(mat.heat_capacity(T), 10.0)


def test_evaluator_applies_material_ids_correctly():
    mat_a = LinearThermoOpticMaterial(n0=2.0, dn_dT=0.0)
    mat_b = LinearThermoOpticMaterial(n0=3.0, dn_dT=0.0)
    material_id = jnp.array([[0, 1], [1, 0]])
    evaluator = MaterialGridEvaluator(material_id, [mat_a, mat_b])
    T = jnp.full((2, 2), 300.0)
    props = evaluator.evaluate(T)
    assert np.isclose(props.permittivity[0, 0], 4.0)
    assert np.isclose(props.permittivity[0, 1], 9.0)


def test_backward_compat_constant_material():
    legacy = Material(permittivity=4.0, permeability=1.0, conductivity=0.5)
    model = as_material_model(legacy)
    evaluator = MaterialGridEvaluator(jnp.zeros((2, 2), dtype=jnp.int32), [model])
    T = jnp.full((2, 2), 300.0)
    props = evaluator.evaluate(T)
    assert np.allclose(props.permittivity, 4.0)
    assert np.allclose(props.conductivity, 0.5)


def test_custom_material_adapter_defaults():
    custom = CustomMaterial(permittivity_grid=np.ones((2, 2)) * 2.0, bounds=((0, 1), (0, 1)))
    model = as_material_model(custom)
    eps = model.epsilon_r(jnp.array(300.0))
    assert np.isclose(eps, 1.0)
