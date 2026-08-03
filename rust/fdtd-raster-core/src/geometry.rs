use geo::{Area, BooleanOps, Coord, LineString, Polygon as GeoPolygon, Rect};
use serde::{Deserialize, Serialize};

use crate::{RasterError, Result, TriangleMesh};

pub type Vec3 = [f64; 3];

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum InterfaceAssessment {
    Missing,
    Laminar(Vec3),
    MultipleOrientations,
}

/// Sign-invariant evidence about the surface patches crossing one support.
///
/// Interface normals are axes for constitutive homogenization: `n` and `-n`
/// describe the same lamination. Patches are aligned to the first reliable
/// normal before accumulation. Every distinct direction is checked against
/// every earlier one so a broad fan cannot be hidden by a central first patch.
#[derive(Clone, Debug)]
pub(crate) struct InterfaceEvidence {
    reference: Option<Vec3>,
    directions: Vec<Vec3>,
    aligned_sum: Vec3,
    total_weight: f64,
    minimum_alignment: f64,
    ambiguous: bool,
}

impl InterfaceEvidence {
    pub(crate) fn new(minimum_alignment: f64) -> Self {
        Self {
            reference: None,
            directions: Vec::new(),
            aligned_sum: [0.0; 3],
            total_weight: 0.0,
            minimum_alignment,
            ambiguous: false,
        }
    }

    pub(crate) fn add(&mut self, normal: Vec3, weight: f64) {
        if self.ambiguous {
            return;
        }
        let length = dot3(normal, normal).sqrt();
        if !weight.is_finite() || weight <= 0.0 || !length.is_finite() || length == 0.0 {
            return;
        }
        let mut unit = normal.map(|value| value / length);
        if self
            .directions
            .iter()
            .any(|existing| dot3(*existing, unit).abs() < self.minimum_alignment)
        {
            self.ambiguous = true;
            return;
        }
        if let Some(reference) = self.reference {
            let alignment = dot3(reference, unit);
            if alignment < 0.0 {
                unit = unit.map(|value| -value);
            }
        } else {
            self.reference = Some(unit);
        }
        if !self
            .directions
            .iter()
            .any(|existing| dot3(*existing, unit).abs() >= 1.0 - 64.0 * f64::EPSILON)
        {
            self.directions.push(unit);
        }
        for (target, value) in self.aligned_sum.iter_mut().zip(unit) {
            *target += weight * value;
        }
        self.total_weight += weight;
    }

    pub(crate) fn mark_ambiguous(&mut self) {
        self.ambiguous = true;
    }

    pub(crate) fn assess(self) -> InterfaceAssessment {
        if self.ambiguous {
            return InterfaceAssessment::MultipleOrientations;
        }
        if self.reference.is_none() || self.total_weight <= 0.0 {
            return InterfaceAssessment::Missing;
        }
        let length = dot3(self.aligned_sum, self.aligned_sum).sqrt();
        if !length.is_finite() || length == 0.0 {
            InterfaceAssessment::Missing
        } else {
            InterfaceAssessment::Laminar(self.aligned_sum.map(|value| value / length))
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Result<Self> {
        if min.iter().chain(max.iter()).any(|v| !v.is_finite())
            || (0..3).any(|axis| max[axis] < min[axis])
        {
            return Err(RasterError::InvalidScene(format!(
                "invalid bounds {min:?}..{max:?}"
            )));
        }
        Ok(Self { min, max })
    }

    pub fn volume(&self) -> f64 {
        (self.max[0] - self.min[0]) * (self.max[1] - self.min[1]) * (self.max[2] - self.min[2])
    }

    pub fn intersects(&self, other: &Self) -> bool {
        (0..3).all(|axis| self.max[axis] > other.min[axis] && other.max[axis] > self.min[axis])
    }

    pub fn contains_half_open(&self, point: Vec3) -> bool {
        (0..3).all(|axis| point[axis] >= self.min[axis] && point[axis] < self.max[axis])
    }

    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let min = [
            self.min[0].max(other.min[0]),
            self.min[1].max(other.min[1]),
            self.min[2].max(other.min[2]),
        ];
        let max = [
            self.max[0].min(other.max[0]),
            self.max[1].min(other.max[1]),
            self.max[2].min(other.max[2]),
        ];
        if (0..3).all(|axis| max[axis] > min[axis]) {
            Some(Self { min, max })
        } else {
            None
        }
    }

    pub fn center(&self) -> Vec3 {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }

    pub fn corners(&self) -> [Vec3; 8] {
        std::array::from_fn(|index| {
            [
                if index & 1 == 0 {
                    self.min[0]
                } else {
                    self.max[0]
                },
                if index & 2 == 0 {
                    self.min[1]
                } else {
                    self.max[1]
                },
                if index & 4 == 0 {
                    self.min[2]
                } else {
                    self.max[2]
                },
            ]
        })
    }

    pub fn child(&self, index: usize) -> Self {
        let mid = self.center();
        let mut min = self.min;
        let mut max = self.max;
        for axis in 0..3 {
            if index & (1 << axis) == 0 {
                max[axis] = mid[axis];
            } else {
                min[axis] = mid[axis];
            }
        }
        Self { min, max }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Polygon2 {
    pub exterior: Vec<[f64; 2]>,
    #[serde(default)]
    pub holes: Vec<Vec<[f64; 2]>>,
}

impl Polygon2 {
    pub fn new(exterior: Vec<[f64; 2]>, holes: Vec<Vec<[f64; 2]>>) -> Result<Self> {
        let exterior = normalize_ring(exterior, true, "exterior")?;
        let mut normalized_holes = Vec::with_capacity(holes.len());
        for (index, hole) in holes.into_iter().enumerate() {
            normalized_holes.push(normalize_ring(hole, false, &format!("hole {index}"))?);
        }
        let result = Self {
            exterior,
            holes: normalized_holes,
        };
        if result.holes.iter().any(|hole| {
            let point = hole[0];
            !point_in_ring(point, &result.exterior)
        }) {
            return Err(RasterError::InvalidPolygon(
                "every hole must lie inside the exterior ring".into(),
            ));
        }
        Ok(result)
    }

    pub fn quantized(mut self, database_unit: f64) -> Result<Self> {
        if !database_unit.is_finite() || database_unit <= 0.0 {
            return Err(RasterError::InvalidPolygon(
                "database unit must be finite and positive".into(),
            ));
        }
        for point in self
            .exterior
            .iter_mut()
            .chain(self.holes.iter_mut().flatten())
        {
            point[0] = (point[0] / database_unit).round() * database_unit;
            point[1] = (point[1] / database_unit).round() * database_unit;
        }
        Self::new(self.exterior, self.holes)
    }

    pub fn bounds(&self) -> [[f64; 2]; 2] {
        let mut min = [f64::INFINITY; 2];
        let mut max = [f64::NEG_INFINITY; 2];
        for point in &self.exterior {
            for axis in 0..2 {
                min[axis] = min[axis].min(point[axis]);
                max[axis] = max[axis].max(point[axis]);
            }
        }
        [min, max]
    }

    pub fn contains_half_open(&self, point: [f64; 2]) -> bool {
        point_in_ring(point, &self.exterior)
            && !self.holes.iter().any(|hole| point_in_ring(point, hole))
    }

    pub fn intersection_area(&self, rect: &Aabb) -> f64 {
        let [minimum, maximum] = self.bounds();
        if rect.max[0] <= minimum[0]
            || rect.min[0] >= maximum[0]
            || rect.max[1] <= minimum[1]
            || rect.min[1] >= maximum[1]
        {
            return 0.0;
        }
        if rect.min[0] <= minimum[0]
            && rect.max[0] >= maximum[0]
            && rect.min[1] <= minimum[1]
            && rect.max[1] >= maximum[1]
        {
            return self.area();
        }
        let polygon = self.to_geo();
        let clip = Rect::new(
            Coord {
                x: rect.min[0],
                y: rect.min[1],
            },
            Coord {
                x: rect.max[0],
                y: rect.max[1],
            },
        )
        .to_polygon();
        polygon.intersection(&clip).unsigned_area()
    }

    pub fn area(&self) -> f64 {
        signed_area(&self.exterior).abs()
            - self
                .holes
                .iter()
                .map(|hole| signed_area(hole).abs())
                .sum::<f64>()
    }

    fn signed_distance(&self, point: [f64; 2]) -> f64 {
        let distance = std::iter::once(&self.exterior)
            .chain(self.holes.iter())
            .flat_map(|ring| {
                (0..ring.len()).map(move |index| {
                    point_segment_distance(point, ring[index], ring[(index + 1) % ring.len()])
                })
            })
            .fold(f64::INFINITY, f64::min);
        if self.contains_half_open(point) {
            -distance
        } else {
            distance
        }
    }

    fn boundary_may_intersect_rect(&self, rect: &Aabb, padding: f64) -> bool {
        std::iter::once(&self.exterior)
            .chain(self.holes.iter())
            .any(|ring| {
                (0..ring.len()).any(|index| {
                    let a = ring[index];
                    let b = ring[(index + 1) % ring.len()];
                    let min_x = a[0].min(b[0]) - padding;
                    let max_x = a[0].max(b[0]) + padding;
                    let min_y = a[1].min(b[1]) - padding;
                    let max_y = a[1].max(b[1]) + padding;
                    max_x > rect.min[0]
                        && min_x < rect.max[0]
                        && max_y > rect.min[1]
                        && min_y < rect.max[1]
                })
            })
    }

    fn to_geo(&self) -> GeoPolygon<f64> {
        fn line(ring: &[[f64; 2]]) -> LineString<f64> {
            let mut coords: Vec<Coord<f64>> =
                ring.iter().map(|p| Coord { x: p[0], y: p[1] }).collect();
            coords.push(coords[0]);
            LineString::new(coords)
        }
        GeoPolygon::new(
            line(&self.exterior),
            self.holes.iter().map(|ring| line(ring)).collect(),
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExtrudedPolygon {
    pub polygon: Polygon2,
    pub z_min: f64,
    pub z_max: f64,
}

impl ExtrudedPolygon {
    pub fn new(polygon: Polygon2, z_min: f64, z_max: f64) -> Result<Self> {
        if !z_min.is_finite() || !z_max.is_finite() || z_max <= z_min {
            return Err(RasterError::InvalidPolygon(
                "extrusion z bounds must be finite and increasing".into(),
            ));
        }
        Ok(Self {
            polygon,
            z_min,
            z_max,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TaperedExtrudedPolygon {
    pub polygon: Polygon2,
    pub z_min: f64,
    pub z_max: f64,
    pub sidewall_angle_degrees: f64,
    pub width_to_z: f64,
}

impl TaperedExtrudedPolygon {
    pub fn new(
        polygon: Polygon2,
        z_min: f64,
        z_max: f64,
        sidewall_angle_degrees: f64,
        width_to_z: f64,
    ) -> Result<Self> {
        ExtrudedPolygon::new(polygon.clone(), z_min, z_max)?;
        if !sidewall_angle_degrees.is_finite()
            || sidewall_angle_degrees.abs() >= 89.0
            || !width_to_z.is_finite()
            || !(0.0..=1.0).contains(&width_to_z)
        {
            return Err(RasterError::InvalidPolygon(
                "sidewall angle must be finite with magnitude below 89 degrees and width_to_z must lie in [0, 1]".into(),
            ));
        }
        Ok(Self {
            polygon,
            z_min,
            z_max,
            sidewall_angle_degrees,
            width_to_z,
        })
    }

    fn offset_at(&self, z: f64) -> f64 {
        let reference = self.z_min + self.width_to_z * (self.z_max - self.z_min);
        -(z - reference) * self.sidewall_angle_degrees.to_radians().tan()
    }

    fn contains_half_open(&self, point: Vec3) -> bool {
        if point[2] < self.z_min || point[2] >= self.z_max {
            return false;
        }
        self.polygon.signed_distance([point[0], point[1]]) <= self.offset_at(point[2])
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Geometry {
    Box {
        bounds: Aabb,
    },
    Sphere {
        center: Vec3,
        radius: f64,
    },
    Cylinder {
        center: [f64; 2],
        radius: f64,
        z_min: f64,
        z_max: f64,
    },
    ExtrudedPolygon(ExtrudedPolygon),
    TaperedExtrudedPolygon(TaperedExtrudedPolygon),
    TriangleMesh(TriangleMesh),
}

impl Geometry {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Box { bounds } => {
                Aabb::new(bounds.min, bounds.max)?;
            }
            Self::Sphere { center, radius } => {
                if !center.iter().all(|value| value.is_finite())
                    || !radius.is_finite()
                    || *radius <= 0.0
                {
                    return Err(RasterError::InvalidScene(
                        "sphere center must be finite and radius positive".into(),
                    ));
                }
            }
            Self::Cylinder {
                center,
                radius,
                z_min,
                z_max,
            } => {
                if !center.iter().all(|value| value.is_finite())
                    || !radius.is_finite()
                    || *radius <= 0.0
                    || !z_min.is_finite()
                    || !z_max.is_finite()
                    || z_max <= z_min
                {
                    return Err(RasterError::InvalidScene(
                        "cylinder parameters must be finite with positive radius and height".into(),
                    ));
                }
            }
            Self::ExtrudedPolygon(extrusion) => {
                Polygon2::new(
                    extrusion.polygon.exterior.clone(),
                    extrusion.polygon.holes.clone(),
                )?;
                ExtrudedPolygon::new(extrusion.polygon.clone(), extrusion.z_min, extrusion.z_max)?;
            }
            Self::TaperedExtrudedPolygon(extrusion) => {
                Polygon2::new(
                    extrusion.polygon.exterior.clone(),
                    extrusion.polygon.holes.clone(),
                )?;
                TaperedExtrudedPolygon::new(
                    extrusion.polygon.clone(),
                    extrusion.z_min,
                    extrusion.z_max,
                    extrusion.sidewall_angle_degrees,
                    extrusion.width_to_z,
                )?;
            }
            Self::TriangleMesh(mesh) => mesh.validate()?,
        }
        Ok(())
    }

    pub fn bounds(&self) -> Aabb {
        match self {
            Self::Box { bounds } => *bounds,
            Self::Sphere { center, radius } => Aabb {
                min: [center[0] - radius, center[1] - radius, center[2] - radius],
                max: [center[0] + radius, center[1] + radius, center[2] + radius],
            },
            Self::Cylinder {
                center,
                radius,
                z_min,
                z_max,
            } => Aabb {
                min: [center[0] - radius, center[1] - radius, *z_min],
                max: [center[0] + radius, center[1] + radius, *z_max],
            },
            Self::ExtrudedPolygon(extrusion) => {
                let [min, max] = extrusion.polygon.bounds();
                Aabb {
                    min: [min[0], min[1], extrusion.z_min],
                    max: [max[0], max[1], extrusion.z_max],
                }
            }
            Self::TaperedExtrudedPolygon(extrusion) => {
                let [min, max] = extrusion.polygon.bounds();
                let expansion = extrusion
                    .offset_at(extrusion.z_min)
                    .max(extrusion.offset_at(extrusion.z_max))
                    .max(0.0);
                Aabb {
                    min: [min[0] - expansion, min[1] - expansion, extrusion.z_min],
                    max: [max[0] + expansion, max[1] + expansion, extrusion.z_max],
                }
            }
            Self::TriangleMesh(mesh) => mesh.bounds(),
        }
    }

    pub fn contains_half_open(&self, point: Vec3) -> bool {
        match self {
            Self::Box { bounds } => bounds.contains_half_open(point),
            Self::Sphere { center, radius } => squared_distance(point, *center) < radius * radius,
            Self::Cylinder {
                center,
                radius,
                z_min,
                z_max,
            } => {
                point[2] >= *z_min
                    && point[2] < *z_max
                    && squared_distance_2d([point[0], point[1]], *center) < radius * radius
            }
            Self::ExtrudedPolygon(extrusion) => {
                point[2] >= extrusion.z_min
                    && point[2] < extrusion.z_max
                    && extrusion.polygon.contains_half_open([point[0], point[1]])
            }
            Self::TaperedExtrudedPolygon(extrusion) => extrusion.contains_half_open(point),
            Self::TriangleMesh(mesh) => mesh.contains(point),
        }
    }

    pub fn exact_overlap_volume(&self, volume: &Aabb) -> Option<f64> {
        match self {
            Self::Box { bounds } => Some(
                bounds
                    .intersection(volume)
                    .map_or(0.0, |intersection| intersection.volume()),
            ),
            Self::ExtrudedPolygon(extrusion) => {
                let z_overlap = (extrusion.z_max.min(volume.max[2])
                    - extrusion.z_min.max(volume.min[2]))
                .max(0.0);
                Some(extrusion.polygon.intersection_area(volume) * z_overlap)
            }
            Self::Sphere { .. }
            | Self::Cylinder { .. }
            | Self::TaperedExtrudedPolygon(_)
            | Self::TriangleMesh(_) => None,
        }
    }

    pub fn surface_may_intersect(&self, volume: &Aabb) -> bool {
        if !self.bounds().intersects(volume) {
            return false;
        }
        match self {
            Self::Box { bounds } => {
                let overlap = bounds.intersection(volume).map_or(0.0, |v| v.volume());
                overlap > 0.0 && overlap < volume.volume()
            }
            Self::Sphere { center, radius } => {
                let radius_squared = radius * radius;
                minimum_squared_distance(*center, volume) < radius_squared
                    && maximum_squared_distance(*center, volume) >= radius_squared
            }
            Self::Cylinder {
                center,
                radius,
                z_min,
                z_max,
            } => {
                let overlaps_z = *z_min < volume.max[2] && *z_max > volume.min[2];
                let radius_squared = radius * radius;
                let side_crosses = minimum_squared_distance_2d(*center, volume) < radius_squared
                    && maximum_squared_distance_2d(*center, volume) >= radius_squared;
                let cap_crosses = (*z_min > volume.min[2] && *z_min < volume.max[2])
                    || (*z_max > volume.min[2] && *z_max < volume.max[2]);
                overlaps_z && (side_crosses || cap_crosses)
            }
            Self::ExtrudedPolygon(_) => {
                let overlap = self.exact_overlap_volume(volume).unwrap_or(0.0);
                overlap > 0.0 && overlap < volume.volume()
            }
            Self::TaperedExtrudedPolygon(extrusion) => {
                let inside = volume
                    .corners()
                    .into_iter()
                    .filter(|point| self.contains_half_open(*point))
                    .count();
                let padding = extrusion
                    .offset_at(volume.min[2])
                    .abs()
                    .max(extrusion.offset_at(volume.max[2]).abs());
                if inside == 8
                    && !extrusion
                        .polygon
                        .boundary_may_intersect_rect(volume, padding)
                {
                    false
                } else {
                    self.bounds().intersection(volume).is_some()
                }
            }
            Self::TriangleMesh(mesh) => mesh.surface_may_intersect(volume),
        }
    }

    pub(crate) fn interface_evidence(
        &self,
        volume: &Aabb,
        minimum_alignment: f64,
    ) -> InterfaceEvidence {
        let mut evidence = InterfaceEvidence::new(minimum_alignment);
        match self {
            Self::Box { bounds } => add_box_evidence(&mut evidence, bounds, volume),
            Self::Sphere { center, radius } => {
                let point = volume.center();
                evidence.add(
                    [
                        point[0] - center[0],
                        point[1] - center[1],
                        point[2] - center[2],
                    ],
                    1.0,
                );
                let half_diagonal = 0.5
                    * (0..3)
                        .map(|axis| (volume.max[axis] - volume.min[axis]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                if half_diagonal / radius > 0.15 {
                    evidence.mark_ambiguous();
                }
            }
            Self::Cylinder {
                center,
                radius,
                z_min,
                z_max,
            } => {
                let point = volume.center();
                let cap_area = rectangle_circle_overlap_upper_bound(volume, *center, *radius);
                if *z_min > volume.min[2] && *z_min < volume.max[2] {
                    evidence.add([0.0, 0.0, -1.0], cap_area);
                }
                if *z_max > volume.min[2] && *z_max < volume.max[2] {
                    evidence.add([0.0, 0.0, 1.0], cap_area);
                }
                let radial = [point[0] - center[0], point[1] - center[1], 0.0];
                if minimum_squared_distance_2d(*center, volume) < radius * radius
                    && maximum_squared_distance_2d(*center, volume) >= radius * radius
                {
                    let z_overlap = (z_max.min(volume.max[2]) - z_min.max(volume.min[2])).max(0.0);
                    evidence.add(
                        radial,
                        z_overlap
                            * (volume.max[0] - volume.min[0]).max(volume.max[1] - volume.min[1]),
                    );
                    let half_diagonal_xy = 0.5
                        * ((volume.max[0] - volume.min[0]).powi(2)
                            + (volume.max[1] - volume.min[1]).powi(2))
                        .sqrt();
                    if half_diagonal_xy / radius > 0.15 {
                        evidence.mark_ambiguous();
                    }
                }
            }
            Self::ExtrudedPolygon(extrusion) => {
                add_extrusion_evidence(
                    &mut evidence,
                    &extrusion.polygon,
                    extrusion.z_min,
                    extrusion.z_max,
                    0.0,
                    0.0,
                    volume,
                );
            }
            Self::TaperedExtrudedPolygon(extrusion) => {
                let slope = extrusion.sidewall_angle_degrees.to_radians().tan();
                let padding = extrusion
                    .offset_at(volume.min[2])
                    .abs()
                    .max(extrusion.offset_at(volume.max[2]).abs());
                add_extrusion_evidence(
                    &mut evidence,
                    &extrusion.polygon,
                    extrusion.z_min,
                    extrusion.z_max,
                    slope,
                    padding,
                    volume,
                );
            }
            Self::TriangleMesh(mesh) => {
                return mesh.interface_evidence(volume, minimum_alignment);
            }
        }
        evidence
    }
}

fn add_box_evidence(evidence: &mut InterfaceEvidence, bounds: &Aabb, volume: &Aabb) {
    for axis in 0..3 {
        let transverse = [(axis + 1) % 3, (axis + 2) % 3];
        let weight = transverse
            .iter()
            .map(|other| {
                (bounds.max[*other].min(volume.max[*other])
                    - bounds.min[*other].max(volume.min[*other]))
                .max(0.0)
            })
            .product::<f64>();
        if weight <= 0.0 {
            continue;
        }
        for (coordinate, sign) in [(bounds.min[axis], -1.0), (bounds.max[axis], 1.0)] {
            if coordinate > volume.min[axis] && coordinate < volume.max[axis] {
                let mut normal = [0.0; 3];
                normal[axis] = sign;
                evidence.add(normal, weight);
            }
        }
    }
}

fn add_extrusion_evidence(
    evidence: &mut InterfaceEvidence,
    polygon: &Polygon2,
    z_min: f64,
    z_max: f64,
    slope: f64,
    padding: f64,
    volume: &Aabb,
) {
    let cap_area = polygon.intersection_area(volume);
    if z_min > volume.min[2] && z_min < volume.max[2] {
        evidence.add([0.0, 0.0, -1.0], cap_area);
    }
    if z_max > volume.min[2] && z_max < volume.max[2] {
        evidence.add([0.0, 0.0, 1.0], cap_area);
    }
    let z_overlap = (z_max.min(volume.max[2]) - z_min.max(volume.min[2])).max(0.0);
    if z_overlap <= 0.0 {
        return;
    }
    let rect = Aabb {
        min: [
            volume.min[0] - padding,
            volume.min[1] - padding,
            volume.min[2],
        ],
        max: [
            volume.max[0] + padding,
            volume.max[1] + padding,
            volume.max[2],
        ],
    };
    for ring in std::iter::once(&polygon.exterior).chain(polygon.holes.iter()) {
        for index in 0..ring.len() {
            let a = ring[index];
            let b = ring[(index + 1) % ring.len()];
            let edge = [b[0] - a[0], b[1] - a[1]];
            let Some(clipped_length) = clipped_segment_length(a, b, &rect) else {
                continue;
            };
            evidence.add(sidewall_normal(edge, slope), clipped_length * z_overlap);
        }
    }
}

fn clipped_segment_length(a: [f64; 2], b: [f64; 2], rect: &Aabb) -> Option<f64> {
    let delta = [b[0] - a[0], b[1] - a[1]];
    let mut low: f64 = 0.0;
    let mut high: f64 = 1.0;
    for (p, q) in [
        (-delta[0], a[0] - rect.min[0]),
        (delta[0], rect.max[0] - a[0]),
        (-delta[1], a[1] - rect.min[1]),
        (delta[1], rect.max[1] - a[1]),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
            continue;
        }
        let ratio = q / p;
        if p < 0.0 {
            low = low.max(ratio);
        } else {
            high = high.min(ratio);
        }
        if low >= high {
            return None;
        }
    }
    Some((high - low) * (delta[0] * delta[0] + delta[1] * delta[1]).sqrt())
}

fn hypot2(value: [f64; 2]) -> f64 {
    value[0].hypot(value[1])
}

fn sidewall_normal(edge: [f64; 2], slope: f64) -> Vec3 {
    normalize([edge[1], -edge[0], slope * hypot2(edge)])
}

fn rectangle_circle_overlap_upper_bound(volume: &Aabb, center: [f64; 2], radius: f64) -> f64 {
    if minimum_squared_distance_2d(center, volume) >= radius * radius {
        return 0.0;
    }
    (volume.max[0] - volume.min[0]) * (volume.max[1] - volume.min[1])
}

fn squared_distance(left: Vec3, right: Vec3) -> f64 {
    (0..3).map(|axis| (left[axis] - right[axis]).powi(2)).sum()
}

fn squared_distance_2d(left: [f64; 2], right: [f64; 2]) -> f64 {
    (left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2)
}

fn minimum_squared_distance(point: Vec3, bounds: &Aabb) -> f64 {
    (0..3)
        .map(|axis| {
            if point[axis] < bounds.min[axis] {
                (bounds.min[axis] - point[axis]).powi(2)
            } else if point[axis] > bounds.max[axis] {
                (point[axis] - bounds.max[axis]).powi(2)
            } else {
                0.0
            }
        })
        .sum()
}

fn maximum_squared_distance(point: Vec3, bounds: &Aabb) -> f64 {
    (0..3)
        .map(|axis| {
            (point[axis] - bounds.min[axis])
                .abs()
                .max((point[axis] - bounds.max[axis]).abs())
                .powi(2)
        })
        .sum()
}

fn minimum_squared_distance_2d(point: [f64; 2], bounds: &Aabb) -> f64 {
    minimum_squared_distance([point[0], point[1], bounds.min[2]], bounds)
}

fn maximum_squared_distance_2d(point: [f64; 2], bounds: &Aabb) -> f64 {
    (0..2)
        .map(|axis| {
            (point[axis] - bounds.min[axis])
                .abs()
                .max((point[axis] - bounds.max[axis]).abs())
                .powi(2)
        })
        .sum()
}

fn dot3(left: Vec3, right: Vec3) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn normalize(value: Vec3) -> Vec3 {
    let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if length == 0.0 {
        value
    } else {
        [value[0] / length, value[1] / length, value[2] / length]
    }
}

fn normalize_ring(mut ring: Vec<[f64; 2]>, ccw: bool, name: &str) -> Result<Vec<[f64; 2]>> {
    if ring.iter().flatten().any(|value| !value.is_finite()) {
        return Err(RasterError::InvalidPolygon(format!(
            "{name} coordinates must be finite"
        )));
    }
    if ring.first() == ring.last() {
        ring.pop();
    }
    ring.dedup();
    loop {
        let mut changed = false;
        if ring.len() >= 3 {
            for i in 0..ring.len() {
                let a = ring[(i + ring.len() - 1) % ring.len()];
                let b = ring[i];
                let c = ring[(i + 1) % ring.len()];
                let cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
                let scale = (b[0] - a[0]).abs()
                    + (b[1] - a[1]).abs()
                    + (c[0] - b[0]).abs()
                    + (c[1] - b[1]).abs();
                if cross.abs() <= f64::EPSILON * scale.max(1.0) {
                    ring.remove(i);
                    changed = true;
                    break;
                }
            }
        }
        if !changed {
            break;
        }
    }
    if ring.len() < 3 {
        return Err(RasterError::InvalidPolygon(format!(
            "{name} requires at least three non-collinear points"
        )));
    }
    if ring_self_intersects(&ring) {
        return Err(RasterError::InvalidPolygon(format!(
            "{name} is self-intersecting"
        )));
    }
    let area = signed_area(&ring);
    if area == 0.0 {
        return Err(RasterError::InvalidPolygon(format!("{name} has zero area")));
    }
    if (area > 0.0) != ccw {
        ring.reverse();
    }
    Ok(ring)
}

fn signed_area(ring: &[[f64; 2]]) -> f64 {
    0.5 * (0..ring.len())
        .map(|i| {
            let a = ring[i];
            let b = ring[(i + 1) % ring.len()];
            a[0] * b[1] - b[0] * a[1]
        })
        .sum::<f64>()
}

fn point_in_ring(point: [f64; 2], ring: &[[f64; 2]]) -> bool {
    let mut inside = false;
    for i in 0..ring.len() {
        let a = ring[i];
        let b = ring[(i + 1) % ring.len()];
        let crosses = (a[1] > point[1]) != (b[1] > point[1]);
        if crosses {
            let x = a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
            if point[0] < x {
                inside = !inside;
            }
        }
    }
    inside
}

fn point_segment_distance(point: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let edge = [b[0] - a[0], b[1] - a[1]];
    let length2 = edge[0] * edge[0] + edge[1] * edge[1];
    if length2 == 0.0 {
        return ((point[0] - a[0]).powi(2) + (point[1] - a[1]).powi(2)).sqrt();
    }
    let t = (((point[0] - a[0]) * edge[0] + (point[1] - a[1]) * edge[1]) / length2).clamp(0.0, 1.0);
    let closest = [a[0] + t * edge[0], a[1] + t * edge[1]];
    ((point[0] - closest[0]).powi(2) + (point[1] - closest[1]).powi(2)).sqrt()
}

fn ring_self_intersects(ring: &[[f64; 2]]) -> bool {
    let n = ring.len();
    for i in 0..n {
        let a = ring[i];
        let b = ring[(i + 1) % n];
        for j in (i + 1)..n {
            if j == i || j == (i + 1) % n || (j + 1) % n == i {
                continue;
            }
            let c = ring[j];
            let d = ring[(j + 1) % n];
            if segments_cross(a, b, c, d) {
                return true;
            }
        }
    }
    false
}

fn segments_cross(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    fn orient(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    }
    let ab_c = orient(a, b, c);
    let ab_d = orient(a, b, d);
    let cd_a = orient(c, d, a);
    let cd_b = orient(c, d, b);
    ab_c * ab_d < 0.0 && cd_a * cd_b < 0.0
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn normalizes_winding_and_removes_collinear_points() {
        let polygon = Polygon2::new(
            vec![[0.0, 0.0], [0.0, 1.0], [0.5, 1.0], [1.0, 1.0], [1.0, 0.0]],
            vec![],
        )
        .unwrap();
        assert_eq!(polygon.exterior.len(), 4);
        assert!(signed_area(&polygon.exterior) > 0.0);
    }

    #[test]
    fn exact_overlap_respects_holes() {
        let polygon = Polygon2::new(
            vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
            vec![vec![[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]]],
        )
        .unwrap();
        let bounds = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [2.0, 2.0, 1.0],
        };
        assert_abs_diff_eq!(polygon.intersection_area(&bounds), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn interface_evidence_accepts_one_planar_face() {
        let geometry = Geometry::Box {
            bounds: Aabb::new([0.0, -1.0, -1.0], [0.5, 2.0, 2.0]).unwrap(),
        };
        let support = Aabb::new([0.0; 3], [1.0; 3]).unwrap();
        assert!(matches!(
            geometry.interface_evidence(&support, 0.995).assess(),
            InterfaceAssessment::Laminar(normal) if normal[0].abs() == 1.0
        ));
    }

    #[test]
    fn antiparallel_faces_form_one_lamination_axis() {
        let geometry = Geometry::Box {
            bounds: Aabb::new([0.25, -1.0, -1.0], [0.75, 2.0, 2.0]).unwrap(),
        };
        let support = Aabb::new([0.0; 3], [1.0; 3]).unwrap();
        assert!(matches!(
            geometry.interface_evidence(&support, 0.995).assess(),
            InterfaceAssessment::Laminar(normal) if normal[0].abs() == 1.0
        ));
    }

    #[test]
    fn box_corner_is_not_reduced_to_an_average_normal() {
        let geometry = Geometry::Box {
            bounds: Aabb::new([0.0, 0.0, -1.0], [0.5, 0.5, 2.0]).unwrap(),
        };
        let support = Aabb::new([0.0; 3], [1.0; 3]).unwrap();
        assert_eq!(
            geometry.interface_evidence(&support, 0.995).assess(),
            InterfaceAssessment::MultipleOrientations
        );
    }

    #[test]
    fn interface_evidence_checks_every_pair_of_directions() {
        let mut evidence = InterfaceEvidence::new(3.0_f64.to_radians().cos());
        for angle in [0.0_f64, 2.0, -2.0] {
            let radians = angle.to_radians();
            evidence.add([radians.cos(), radians.sin(), 0.0], 1.0);
        }
        assert_eq!(evidence.assess(), InterfaceAssessment::MultipleOrientations);
    }

    #[test]
    fn tapered_sidewall_normal_is_scale_invariant() {
        let slope = 30.0_f64.to_radians().tan();
        let base = sidewall_normal([0.6, 0.8], slope);
        let micrometres = sidewall_normal([0.6e-6, 0.8e-6], slope);
        for axis in 0..3 {
            assert_abs_diff_eq!(base[axis], micrometres[axis], epsilon = 1e-12);
        }
    }

    #[test]
    fn curved_surface_requires_a_locally_planar_support() {
        let geometry = Geometry::Sphere {
            center: [0.5; 3],
            radius: 0.4,
        };
        let coarse = Aabb::new([0.0; 3], [1.0; 3]).unwrap();
        let fine = Aabb::new([0.49, 0.49, 0.89], [0.51, 0.51, 0.91]).unwrap();
        assert_eq!(
            geometry.interface_evidence(&coarse, 0.995).assess(),
            InterfaceAssessment::MultipleOrientations
        );
        assert!(matches!(
            geometry.interface_evidence(&fine, 0.995).assess(),
            InterfaceAssessment::Laminar(normal) if normal[2] > 0.99
        ));
    }

    #[test]
    fn rejects_bow_tie() {
        assert!(
            Polygon2::new(vec![[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], vec![]).is_err()
        );
    }

    proptest! {
        #[test]
        fn rectangle_overlap_is_translation_invariant(
            x in -1e3f64..1e3,
            y in -1e3f64..1e3,
            width in 1e-6f64..10.0,
            height in 1e-6f64..10.0,
        ) {
            let polygon = Polygon2::new(
                vec![
                    [x, y],
                    [x + width, y],
                    [x + width, y + height],
                    [x, y + height],
                ],
                vec![],
            ).unwrap();
            let bounds = Aabb {
                min: [x, y, 0.0],
                max: [x + width, y + height, 1.0],
            };
            prop_assert!((polygon.intersection_area(&bounds) - width * height).abs()
                <= 1e-8 * (width * height).max(1.0));
        }
    }
}
