use serde::{Deserialize, Serialize};

use crate::{Geometry, RasterError, Result, Vec3};

/// Symmetric tensor packed as xx, yy, zz, xy, xz, yz.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(transparent)]
pub struct SymmetricTensor(pub [f64; 6]);

impl SymmetricTensor {
    pub const fn isotropic(value: f64) -> Self {
        Self([value, value, value, 0.0, 0.0, 0.0])
    }

    pub fn matrix(self) -> [[f64; 3]; 3] {
        let [xx, yy, zz, xy, xz, yz] = self.0;
        [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
    }

    pub fn from_matrix(value: [[f64; 3]; 3]) -> Self {
        Self([
            value[0][0],
            value[1][1],
            value[2][2],
            0.5 * (value[0][1] + value[1][0]),
            0.5 * (value[0][2] + value[2][0]),
            0.5 * (value[1][2] + value[2][1]),
        ])
    }

    pub fn diagonal(self, axis: usize) -> f64 {
        self.0[axis]
    }

    pub fn scaled_add(&mut self, other: Self, weight: f64) {
        for (target, source) in self.0.iter_mut().zip(other.0) {
            *target += source * weight;
        }
    }

    fn validate(self, name: &str, positive_definite: bool) -> Result<()> {
        if self.0.iter().any(|value| !value.is_finite()) {
            return Err(RasterError::InvalidMaterial(format!(
                "{name} must contain finite values"
            )));
        }
        let [xx, yy, zz, xy, xz, yz] = self.0;
        let scale = self
            .0
            .iter()
            .fold(1.0_f64, |current, value| current.max(value.abs()));
        let tolerance = 64.0 * f64::EPSILON * scale;
        let minors = [
            xx,
            yy,
            zz,
            xx * yy - xy * xy,
            xx * zz - xz * xz,
            yy * zz - yz * yz,
            determinant(self.matrix()),
        ];
        let valid = if positive_definite {
            // Sylvester's criterion is sufficient for a symmetric matrix.
            xx > tolerance
                && xx * yy - xy * xy > tolerance * scale
                && minors[6] > tolerance * scale * scale
        } else {
            minors
                .iter()
                .all(|value| *value >= -tolerance * scale * scale)
        };
        if !valid {
            let qualifier = if positive_definite {
                "positive definite"
            } else {
                "positive semidefinite"
            };
            return Err(RasterError::InvalidMaterial(format!(
                "{name} must be {qualifier}"
            )));
        }
        Ok(())
    }
}

fn determinant(value: [[f64; 3]; 3]) -> f64 {
    value[0][0] * (value[1][1] * value[2][2] - value[1][2] * value[2][1])
        - value[0][1] * (value[1][0] * value[2][2] - value[1][2] * value[2][0])
        + value[0][2] * (value[1][0] * value[2][1] - value[1][1] * value[2][0])
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Material {
    pub epsilon_r: SymmetricTensor,
    pub mu_r: SymmetricTensor,
    pub conductivity: SymmetricTensor,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            epsilon_r: SymmetricTensor::isotropic(1.0),
            mu_r: SymmetricTensor::isotropic(1.0),
            conductivity: SymmetricTensor::isotropic(0.0),
        }
    }
}

impl Material {
    pub fn new(epsilon_r: f64, mu_r: f64, conductivity: f64) -> Result<Self> {
        Self::tensor(
            SymmetricTensor::isotropic(epsilon_r),
            SymmetricTensor::isotropic(mu_r),
            SymmetricTensor::isotropic(conductivity),
        )
    }

    pub fn tensor(
        epsilon_r: SymmetricTensor,
        mu_r: SymmetricTensor,
        conductivity: SymmetricTensor,
    ) -> Result<Self> {
        let material = Self {
            epsilon_r,
            mu_r,
            conductivity,
        };
        material.validate()?;
        Ok(material)
    }

    pub fn validate(&self) -> Result<()> {
        self.epsilon_r.validate("epsilon_r", true)?;
        self.mu_r.validate("mu_r", true)?;
        self.conductivity.validate("conductivity", false)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Object {
    pub id: u64,
    pub material_id: usize,
    pub priority: i32,
    pub geometry: Geometry,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Scene {
    pub materials: Vec<Material>,
    pub objects: Vec<Object>,
    pub background_material: usize,
}

impl Scene {
    pub fn new(
        materials: Vec<Material>,
        objects: Vec<Object>,
        background_material: usize,
    ) -> Result<Self> {
        let scene = Self {
            materials,
            objects,
            background_material,
        };
        scene.validate()?;
        Ok(scene)
    }

    pub fn validate(&self) -> Result<()> {
        if self.materials.is_empty() {
            return Err(RasterError::InvalidScene(
                "scene requires at least one material".into(),
            ));
        }
        for material in &self.materials {
            material.validate()?;
        }
        if self.background_material >= self.materials.len() {
            return Err(RasterError::InvalidScene(
                "background material ID is out of range".into(),
            ));
        }
        let mut ids: Vec<u64> = self.objects.iter().map(|object| object.id).collect();
        ids.sort_unstable();
        if ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(RasterError::InvalidScene(
                "object IDs must be unique".into(),
            ));
        }
        if self
            .objects
            .iter()
            .any(|object| object.material_id >= self.materials.len())
        {
            return Err(RasterError::InvalidScene(
                "object material ID is out of range".into(),
            ));
        }
        for object in &self.objects {
            object.geometry.validate()?;
        }
        Ok(())
    }

    pub fn owner_at(&self, point: Vec3) -> usize {
        self.objects
            .iter()
            .filter(|object| object.geometry.contains_half_open(point))
            .max_by_key(|object| (object.priority, object.id))
            .map_or(self.background_material, |object| object.material_id)
    }

    pub fn stable_hash(&self) -> Result<String> {
        let bytes = serde_json::to_vec(self)?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}

#[cfg(test)]
mod tests {
    use crate::Aabb;

    use super::*;

    #[test]
    fn priority_then_id_controls_owner() {
        let objects = vec![
            Object {
                id: 1,
                material_id: 1,
                priority: 0,
                geometry: Geometry::Box {
                    bounds: Aabb::new([0.0; 3], [1.0; 3]).unwrap(),
                },
            },
            Object {
                id: 2,
                material_id: 2,
                priority: 0,
                geometry: Geometry::Box {
                    bounds: Aabb::new([0.0; 3], [1.0; 3]).unwrap(),
                },
            },
        ];
        let scene = Scene::new(
            vec![
                Material::default(),
                Material::new(2.0, 1.0, 0.0).unwrap(),
                Material::new(3.0, 1.0, 0.0).unwrap(),
            ],
            objects,
            0,
        )
        .unwrap();
        assert_eq!(scene.owner_at([0.5; 3]), 2);
    }

    #[test]
    fn rejects_indefinite_tensor() {
        assert!(
            Material::tensor(
                SymmetricTensor([1.0, 1.0, 1.0, 2.0, 0.0, 0.0]),
                SymmetricTensor::isotropic(1.0),
                SymmetricTensor::isotropic(0.0),
            )
            .is_err()
        );
    }
}
