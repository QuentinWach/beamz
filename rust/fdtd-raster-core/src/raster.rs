use std::time::Instant;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::geometry::InterfaceAssessment;
use crate::grid::SupportSpec;
use crate::{Aabb, Grid, Material, Result, Scene, SymmetricTensor};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Quality {
    Fast,
    #[default]
    Balanced,
    Reference,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SmoothingMode {
    #[default]
    Volume,
    FarjadpourDiagonal,
    FarjadpourFull,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutputComponents {
    #[default]
    All,
    TwoDimensionalTm,
    TwoDimensionalTe,
}

impl OutputComponents {
    fn includes(self, component: Component) -> bool {
        match self {
            Self::All => true,
            Self::TwoDimensionalTm => {
                matches!(component, Component::Ez | Component::Hx | Component::Hy)
            }
            Self::TwoDimensionalTe => {
                matches!(component, Component::Ex | Component::Ey | Component::Hz)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct IntegrationOptions {
    pub smoothing: SmoothingMode,
    pub output_components: OutputComponents,
    minimum_depth: u8,
    max_depth: u8,
    fraction_error_tolerance: f64,
    minimum_normal_alignment: f64,
}

impl Default for IntegrationOptions {
    fn default() -> Self {
        Self::for_quality(Quality::Balanced)
    }
}

impl IntegrationOptions {
    pub fn for_quality(quality: Quality) -> Self {
        let (minimum_depth, max_depth, fraction_error_tolerance, minimum_normal_alignment) =
            match quality {
                Quality::Fast => (0, 0, 1.25e-1, 0.98),
                Quality::Balanced => (1, 2, 1e-2, 0.995),
                Quality::Reference => (3, 5, 1e-4, 0.999),
            };
        Self {
            smoothing: SmoothingMode::Volume,
            output_components: OutputComponents::All,
            minimum_depth,
            max_depth,
            fraction_error_tolerance,
            minimum_normal_alignment,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TensorArray {
    /// Compact constitutive components followed by storage order `(c, z, y, x)`.
    /// `c=1` is isotropic, `c=3` diagonal, and `c=6` symmetric packed.
    pub shape: [usize; 4],
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct DiagnosticSummary {
    pub samples: usize,
    pub background_samples: usize,
    pub uniform_samples: usize,
    pub exact_samples: usize,
    pub adaptive_samples: usize,
    pub unresolved_samples: usize,
    pub estimated_fraction_error_sum: f64,
    pub maximum_estimated_fraction_error: f64,
    pub smoothed_samples: usize,
    pub ambiguous_interface_samples: usize,
    pub fallback_multiple_orientations: usize,
    pub fallback_multiple_objects: usize,
    pub fallback_unresolved_geometry: usize,
    pub fallback_missing_surface_evidence: usize,
    pub candidate_object_tests: u64,
    pub validation_seconds: f64,
    pub indexing_seconds: f64,
    pub integration_seconds: f64,
    pub hashing_seconds: f64,
    pub elapsed_seconds: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RasterResult {
    /// Compact epsilon tensors at Ex, Ey, and Ez supports.
    pub epsilon: [TensorArray; 3],
    /// Compact epsilon tensor on the dual-cell centers shared by cross terms.
    pub node_epsilon: TensorArray,
    /// Compact mu tensors at Hx, Hy, and Hz supports.
    pub mu: [TensorArray; 3],
    /// Compact electric-conductivity tensors at Ex, Ey, and Ez supports.
    pub conductivity: [TensorArray; 3],
    /// Authoritative compact cell tensors with 1, 3, or 6 leading components.
    pub cell_epsilon: TensorArray,
    pub cell_mu: TensorArray,
    pub cell_conductivity: TensorArray,
    pub diagnostics: DiagnosticSummary,
    pub scene_hash: String,
}

#[derive(Clone, Copy)]
enum Component {
    Ex,
    Ey,
    Ez,
    Hx,
    Hy,
    Hz,
    Node,
    Cell,
}

impl Component {
    fn support(self) -> SupportSpec {
        match self {
            Self::Ex => SupportSpec::EX,
            Self::Ey => SupportSpec::EY,
            Self::Ez => SupportSpec::EZ,
            Self::Hx => SupportSpec::HX,
            Self::Hy => SupportSpec::HY,
            Self::Hz => SupportSpec::HZ,
            Self::Node => SupportSpec::NODE,
            Self::Cell => SupportSpec::CELL,
        }
    }
}

struct ObjectIndex {
    domain: Aabb,
    dimensions: [usize; 3],
    bins: Vec<Vec<usize>>,
}

impl ObjectIndex {
    fn build(scene: &Scene, grid: &Grid) -> Self {
        let [nx, ny, nz] = grid.shape();
        let dimensions = [
            nx.div_ceil(8).clamp(1, 64),
            ny.div_ceil(8).clamp(1, 64),
            nz.div_ceil(8).clamp(1, 64),
        ];
        let domain_values = grid.domain();
        let domain = Aabb {
            min: [
                domain_values[0][0],
                domain_values[1][0],
                domain_values[2][0],
            ],
            max: [
                domain_values[0][1],
                domain_values[1][1],
                domain_values[2][1],
            ],
        };
        let mut result = Self {
            domain,
            dimensions,
            bins: vec![Vec::new(); dimensions.iter().product()],
        };
        for (object_index, object) in scene.objects.iter().enumerate() {
            if let Some(bounds) = object.geometry.bounds().intersection(&domain) {
                let ranges = result.bin_ranges(&bounds);
                for z in ranges[2].0..=ranges[2].1 {
                    for y in ranges[1].0..=ranges[1].1 {
                        for x in ranges[0].0..=ranges[0].1 {
                            let flat = result.flat(x, y, z);
                            result.bins[flat].push(object_index);
                        }
                    }
                }
            }
        }
        for bin in &mut result.bins {
            bin.sort_unstable();
            bin.dedup();
        }
        result
    }

    fn query(&self, bounds: &Aabb) -> Vec<usize> {
        let Some(bounds) = bounds.intersection(&self.domain) else {
            return Vec::new();
        };
        let ranges = self.bin_ranges(&bounds);
        let mut result = Vec::new();
        for z in ranges[2].0..=ranges[2].1 {
            for y in ranges[1].0..=ranges[1].1 {
                for x in ranges[0].0..=ranges[0].1 {
                    result.extend_from_slice(&self.bins[self.flat(x, y, z)]);
                }
            }
        }
        result.sort_unstable();
        result.dedup();
        result
    }

    fn bin_ranges(&self, bounds: &Aabb) -> [(usize, usize); 3] {
        std::array::from_fn(|axis| {
            let extent = self.domain.max[axis] - self.domain.min[axis];
            let count = self.dimensions[axis];
            let coordinate = |value: f64| {
                (((value - self.domain.min[axis]) / extent) * count as f64)
                    .floor()
                    .clamp(0.0, (count - 1) as f64) as usize
            };
            (coordinate(bounds.min[axis]), coordinate(bounds.max[axis]))
        })
    }

    fn flat(&self, x: usize, y: usize, z: usize) -> usize {
        x + self.dimensions[0] * (y + self.dimensions[1] * z)
    }
}

struct Sample {
    epsilon: f32,
    mu: f32,
    conductivity: f32,
    error: f32,
    path: SamplePath,
    candidate_tests: u64,
    smoothed: bool,
    fallback: Option<FallbackReason>,
}

struct CellSample {
    material: Material,
    meta: SampleMeta,
}

struct SampleMeta {
    error: f32,
    path: SamplePath,
    candidate_tests: u64,
    smoothed: bool,
    fallback: Option<FallbackReason>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FallbackReason {
    MultipleOrientations,
    MultipleObjects,
    UnresolvedGeometry,
    MissingSurfaceEvidence,
}

#[derive(Clone, Copy, Debug)]
enum InterfaceClass {
    None,
    Laminar([f64; 3]),
    Ambiguous(FallbackReason),
}

trait HasMeta {
    fn error(&self) -> f32;
    fn path(&self) -> SamplePath;
    fn candidate_tests(&self) -> u64;
    fn smoothed(&self) -> bool {
        false
    }
    fn fallback(&self) -> Option<FallbackReason> {
        None
    }
}

impl HasMeta for Sample {
    fn error(&self) -> f32 {
        self.error
    }

    fn path(&self) -> SamplePath {
        self.path
    }

    fn candidate_tests(&self) -> u64 {
        self.candidate_tests
    }

    fn smoothed(&self) -> bool {
        self.smoothed
    }

    fn fallback(&self) -> Option<FallbackReason> {
        self.fallback
    }
}

impl HasMeta for CellSample {
    fn error(&self) -> f32 {
        self.meta.error
    }

    fn path(&self) -> SamplePath {
        self.meta.path
    }

    fn candidate_tests(&self) -> u64 {
        self.meta.candidate_tests
    }

    fn smoothed(&self) -> bool {
        self.meta.smoothed
    }

    fn fallback(&self) -> Option<FallbackReason> {
        self.meta.fallback
    }
}

#[derive(Clone, Copy)]
enum SamplePath {
    Background,
    Uniform,
    Exact,
    Adaptive,
}

pub fn rasterize(scene: &Scene, grid: &Grid, options: &IntegrationOptions) -> Result<RasterResult> {
    let start = Instant::now();
    scene.validate()?;
    grid.validate()?;
    let validation_seconds = start.elapsed().as_secs_f64();
    rasterize_impl(scene, grid, options, start, validation_seconds)
}

/// Rasterize a scene whose caller has already run [`Scene::validate`].
///
/// This entry point is intended for compiled language bindings that retain a
/// validated scene between grid sweeps.
pub fn rasterize_prevalidated(
    scene: &Scene,
    grid: &Grid,
    options: &IntegrationOptions,
) -> Result<RasterResult> {
    let start = Instant::now();
    grid.validate()?;
    rasterize_impl(scene, grid, options, start, 0.0)
}

fn rasterize_impl(
    scene: &Scene,
    grid: &Grid,
    options: &IntegrationOptions,
    start: Instant,
    validation_seconds: f64,
) -> Result<RasterResult> {
    if matches!(
        options.output_components,
        OutputComponents::TwoDimensionalTm | OutputComponents::TwoDimensionalTe
    ) && grid.shape()[2] != 1
    {
        return Err(crate::RasterError::InvalidGrid(
            "two-dimensional output requires exactly one z cell".into(),
        ));
    }
    let index_start = Instant::now();
    let index = ObjectIndex::build(scene, grid);
    let indexing_seconds = index_start.elapsed().as_secs_f64();
    let integration_start = Instant::now();
    let mut diagnostics = DiagnosticSummary::default();
    let epsilon_components = material_component_count(scene, |material| material.epsilon_r);
    let mu_components = material_component_count(scene, |material| material.mu_r);
    let conductivity_components = material_component_count(scene, |material| material.conductivity);
    let scalar_volume = options.smoothing == SmoothingMode::Volume
        && epsilon_components == 1
        && mu_components == 1
        && conductivity_components == 1;
    let maximum_components = if options.smoothing == SmoothingMode::FarjadpourDiagonal {
        3
    } else {
        6
    };
    let mut epsilon = Vec::with_capacity(3);
    let mut conductivity = Vec::with_capacity(3);
    for component in [Component::Ex, Component::Ey, Component::Ez] {
        if options.output_components.includes(component) {
            if scalar_volume {
                let (shape, samples) =
                    raster_scalar_support(scene, &index, grid, options, component);
                record_diagnostics(&mut diagnostics, &samples);
                epsilon.push(scalar_tensor_from_samples(shape, &samples, |sample| {
                    sample.epsilon
                }));
                conductivity.push(scalar_tensor_from_samples(shape, &samples, |sample| {
                    sample.conductivity
                }));
            } else {
                let (shape, samples) =
                    raster_material_support(scene, &index, grid, options, component);
                record_diagnostics(&mut diagnostics, &samples);
                epsilon.push(tensor_from_samples(
                    shape,
                    &samples,
                    |sample| sample.material.epsilon_r,
                    maximum_components,
                ));
                conductivity.push(tensor_from_samples(
                    shape,
                    &samples,
                    |sample| sample.material.conductivity,
                    maximum_components,
                ));
            }
        } else {
            epsilon.push(omitted_tensor(component, grid));
            conductivity.push(omitted_tensor(component, grid));
        }
    }
    let mut mu = Vec::with_capacity(3);
    for component in [Component::Hx, Component::Hy, Component::Hz] {
        if options.output_components.includes(component) {
            if scalar_volume {
                let (shape, samples) =
                    raster_scalar_support(scene, &index, grid, options, component);
                record_diagnostics(&mut diagnostics, &samples);
                mu.push(scalar_tensor_from_samples(shape, &samples, |sample| {
                    sample.mu
                }));
            } else {
                let (shape, samples) =
                    raster_material_support(scene, &index, grid, options, component);
                record_diagnostics(&mut diagnostics, &samples);
                mu.push(tensor_from_samples(
                    shape,
                    &samples,
                    |sample| sample.material.mu_r,
                    maximum_components,
                ));
            }
        } else {
            mu.push(omitted_tensor(component, grid));
        }
    }
    let node_epsilon = if options.smoothing == SmoothingMode::FarjadpourFull
        && options.output_components != OutputComponents::TwoDimensionalTm
    {
        let (shape, samples) =
            raster_material_support(scene, &index, grid, options, Component::Node);
        record_diagnostics(&mut diagnostics, &samples);
        tensor_from_samples(
            shape,
            &samples,
            |sample| sample.material.epsilon_r,
            maximum_components,
        )
    } else {
        omitted_tensor(Component::Node, grid)
    };
    let (cell_epsilon, cell_mu, cell_conductivity) = if scalar_volume {
        let (cell_shape, cell_samples) =
            raster_scalar_support(scene, &index, grid, options, Component::Cell);
        record_diagnostics(&mut diagnostics, &cell_samples);
        (
            scalar_tensor_from_samples(cell_shape, &cell_samples, |sample| sample.epsilon),
            scalar_tensor_from_samples(cell_shape, &cell_samples, |sample| sample.mu),
            scalar_tensor_from_samples(cell_shape, &cell_samples, |sample| sample.conductivity),
        )
    } else {
        let (cell_shape, cell_samples) =
            raster_material_support(scene, &index, grid, options, Component::Cell);
        record_diagnostics(&mut diagnostics, &cell_samples);
        (
            tensor_from_samples(
                cell_shape,
                &cell_samples,
                |sample| sample.material.epsilon_r,
                maximum_components,
            ),
            tensor_from_samples(
                cell_shape,
                &cell_samples,
                |sample| sample.material.mu_r,
                maximum_components,
            ),
            tensor_from_samples(
                cell_shape,
                &cell_samples,
                |sample| sample.material.conductivity,
                maximum_components,
            ),
        )
    };
    diagnostics.validation_seconds = validation_seconds;
    diagnostics.indexing_seconds = indexing_seconds;
    diagnostics.integration_seconds = integration_start.elapsed().as_secs_f64();
    let hash_start = Instant::now();
    let scene_hash = scene.stable_hash()?;
    diagnostics.hashing_seconds = hash_start.elapsed().as_secs_f64();
    diagnostics.elapsed_seconds = start.elapsed().as_secs_f64();
    Ok(RasterResult {
        epsilon: epsilon.try_into().unwrap(),
        node_epsilon,
        mu: mu.try_into().unwrap(),
        conductivity: conductivity.try_into().unwrap(),
        cell_epsilon,
        cell_mu,
        cell_conductivity,
        diagnostics,
        scene_hash,
    })
}

fn record_diagnostics<T: HasMeta>(diagnostics: &mut DiagnosticSummary, samples: &[T]) {
    diagnostics.samples += samples.len();
    for sample in samples {
        diagnostics.candidate_object_tests += sample.candidate_tests();
        match sample.path() {
            SamplePath::Background => diagnostics.background_samples += 1,
            SamplePath::Uniform => diagnostics.uniform_samples += 1,
            SamplePath::Exact => diagnostics.exact_samples += 1,
            SamplePath::Adaptive => diagnostics.adaptive_samples += 1,
        }
        let estimated_error = f64::from(sample.error());
        diagnostics.unresolved_samples += usize::from(estimated_error > 0.0);
        diagnostics.estimated_fraction_error_sum += estimated_error;
        diagnostics.maximum_estimated_fraction_error = diagnostics
            .maximum_estimated_fraction_error
            .max(estimated_error);
        diagnostics.smoothed_samples += usize::from(sample.smoothed());
        if let Some(reason) = sample.fallback() {
            diagnostics.ambiguous_interface_samples += 1;
            match reason {
                FallbackReason::MultipleOrientations => {
                    diagnostics.fallback_multiple_orientations += 1;
                }
                FallbackReason::MultipleObjects => {
                    diagnostics.fallback_multiple_objects += 1;
                }
                FallbackReason::UnresolvedGeometry => {
                    diagnostics.fallback_unresolved_geometry += 1;
                }
                FallbackReason::MissingSurfaceEvidence => {
                    diagnostics.fallback_missing_surface_evidence += 1;
                }
            }
        }
    }
}

fn omitted_tensor(component: Component, grid: &Grid) -> TensorArray {
    let logical_shape = component.support().logical_shape(grid);
    TensorArray {
        shape: [0, logical_shape[2], logical_shape[1], logical_shape[0]],
        values: Vec::new(),
    }
}

fn tensor_from_samples(
    shape: [usize; 3],
    samples: &[CellSample],
    property: impl Fn(&CellSample) -> SymmetricTensor,
    maximum_components: usize,
) -> TensorArray {
    let mut components = 1;
    for sample in samples {
        let tensor = property(sample).0;
        let scale = tensor
            .iter()
            .fold(1.0_f64, |current, value| current.max(value.abs()));
        let tolerance = 256.0 * f64::EPSILON * scale;
        if maximum_components == 6 && tensor[3..].iter().any(|value| value.abs() > tolerance) {
            components = 6;
            break;
        }
        if (tensor[0] - tensor[1]).abs() > tolerance || (tensor[0] - tensor[2]).abs() > tolerance {
            components = 3;
        }
    }
    let mut values = Vec::with_capacity(components * samples.len());
    for component in 0..components {
        values.extend(
            samples
                .iter()
                .map(|sample| property(sample).0[component] as f32),
        );
    }
    TensorArray {
        shape: [components, shape[0], shape[1], shape[2]],
        values,
    }
}

fn scalar_tensor_from_samples(
    shape: [usize; 3],
    samples: &[Sample],
    property: impl Fn(&Sample) -> f32,
) -> TensorArray {
    TensorArray {
        shape: [1, shape[0], shape[1], shape[2]],
        values: samples.iter().map(property).collect(),
    }
}

fn material_component_count(
    scene: &Scene,
    property: impl Fn(&Material) -> SymmetricTensor,
) -> usize {
    let mut components = 1;
    for material in &scene.materials {
        let tensor = property(material).0;
        if tensor[3..].iter().any(|value| *value != 0.0) {
            return 6;
        }
        if tensor[0] != tensor[1] || tensor[0] != tensor[2] {
            components = 3;
        }
    }
    components
}

fn raster_scalar_support(
    scene: &Scene,
    index: &ObjectIndex,
    grid: &Grid,
    options: &IntegrationOptions,
    component: Component,
) -> ([usize; 3], Vec<Sample>) {
    raster_support(scene, index, grid, component, |volume, candidates| {
        integrate_scalar(scene, volume, candidates, options)
    })
}

fn raster_material_support(
    scene: &Scene,
    index: &ObjectIndex,
    grid: &Grid,
    options: &IntegrationOptions,
    component: Component,
) -> ([usize; 3], Vec<CellSample>) {
    raster_support(scene, index, grid, component, |volume, candidates| {
        let (material, meta) = integrate_material(scene, volume, candidates, options);
        CellSample { material, meta }
    })
}

fn raster_support<T: Send>(
    scene: &Scene,
    index: &ObjectIndex,
    grid: &Grid,
    component: Component,
    integrate: impl Fn(&Aabb, &[usize]) -> T + Sync,
) -> ([usize; 3], Vec<T>) {
    let logical_shape = component.support().logical_shape(grid);
    let shape_zyx = [logical_shape[2], logical_shape[1], logical_shape[0]];
    let length = logical_shape.iter().product();
    let samples = (0..length)
        .into_par_iter()
        .map(|flat| {
            let x = flat % logical_shape[0];
            let rest = flat / logical_shape[0];
            let y = rest % logical_shape[1];
            let z = rest / logical_shape[1];
            let volume = component.support().volume(grid, [x, y, z]);
            let mut candidates = index.query(&volume);
            candidates.retain(|candidate| {
                scene.objects[*candidate]
                    .geometry
                    .bounds()
                    .intersects(&volume)
            });
            integrate(&volume, &candidates)
        })
        .collect();
    (shape_zyx, samples)
}

fn integrate_scalar(
    scene: &Scene,
    volume: &Aabb,
    candidates: &[usize],
    options: &IntegrationOptions,
) -> Sample {
    let mixture = integrate_mixture(scene, volume, candidates, options);
    let [epsilon, mu, conductivity] = if let Some(owner) = mixture.uniform_owner {
        let material = scene.materials[owner];
        [
            material.epsilon_r.0[0],
            material.mu_r.0[0],
            material.conductivity.0[0],
        ]
    } else {
        let mut values = [0.0; 3];
        for (material, fraction) in scene
            .materials
            .iter()
            .zip(mixture.fractions.as_ref().unwrap())
        {
            values[0] += material.epsilon_r.0[0] * fraction;
            values[1] += material.mu_r.0[0] * fraction;
            values[2] += material.conductivity.0[0] * fraction;
        }
        values
    };
    Sample {
        epsilon: epsilon as f32,
        mu: mu as f32,
        conductivity: conductivity as f32,
        error: mixture.error as f32,
        path: mixture.path,
        candidate_tests: mixture.candidate_tests,
        smoothed: false,
        fallback: None,
    }
}

fn integrate_material(
    scene: &Scene,
    volume: &Aabb,
    candidates: &[usize],
    options: &IntegrationOptions,
) -> (Material, SampleMeta) {
    let mixture = integrate_mixture(scene, volume, candidates, options);
    let (material, smoothed, fallback) = if let Some(owner) = mixture.uniform_owner {
        (scene.materials[owner], false, None)
    } else {
        effective_material(
            scene,
            mixture.fractions.as_ref().unwrap(),
            options.smoothing,
            mixture.interface,
        )
    };
    (
        material,
        SampleMeta {
            error: mixture.error as f32,
            path: mixture.path,
            candidate_tests: mixture.candidate_tests,
            smoothed,
            fallback,
        },
    )
}

/// Geometry-only integration result consumed by every constitutive rule.
struct Mixture {
    fractions: Option<Vec<f64>>,
    interface: InterfaceClass,
    error: f64,
    path: SamplePath,
    candidate_tests: u64,
    uniform_owner: Option<usize>,
}

fn uniform_mixture(owner: usize, path: SamplePath, candidate_tests: u64) -> Mixture {
    Mixture {
        fractions: None,
        interface: InterfaceClass::None,
        error: 0.0,
        path,
        candidate_tests,
        uniform_owner: Some(owner),
    }
}

fn integrate_mixture(
    scene: &Scene,
    volume: &Aabb,
    candidates: &[usize],
    options: &IntegrationOptions,
) -> Mixture {
    let mut candidate_tests = candidates.len() as u64;
    if candidates.is_empty() {
        return uniform_mixture(
            scene.background_material,
            SamplePath::Background,
            candidate_tests,
        );
    }
    let surfaces = candidates
        .iter()
        .any(|index| scene.objects[*index].geometry.surface_may_intersect(volume));
    if !surfaces {
        let owner = owner_at(scene, candidates, volume.center());
        return uniform_mixture(owner, SamplePath::Uniform, candidate_tests);
    }
    if candidates.len() == 1 {
        let object = &scene.objects[candidates[0]];
        if let Some(overlap) = object.geometry.exact_overlap_volume(volume) {
            let fraction = (overlap / volume.volume()).clamp(0.0, 1.0);
            let mut fractions = vec![0.0; scene.materials.len()];
            fractions[scene.background_material] = 1.0 - fraction;
            fractions[object.material_id] += fraction;
            return Mixture {
                fractions: Some(fractions),
                interface: classify_interface(scene, volume, candidates, options, 0.0),
                error: 0.0,
                path: SamplePath::Exact,
                candidate_tests,
                uniform_owner: None,
            };
        }
    }

    let mut volumes = vec![0.0; scene.materials.len()];
    let mut estimated_error_volume = 0.0;
    let mut adaptive_output = AdaptiveOutput {
        accumulated: &mut volumes,
        estimated_error_volume: &mut estimated_error_volume,
        candidate_tests: &mut candidate_tests,
    };
    adaptive_integrate(
        scene,
        volume,
        candidates,
        volume.volume(),
        0,
        options,
        &mut adaptive_output,
    );
    let total = volume.volume();
    let fractions: Vec<f64> = volumes.iter().map(|occupied| occupied / total).collect();
    let error = (estimated_error_volume / total).min(1.0);
    Mixture {
        fractions: Some(fractions),
        interface: classify_interface(scene, volume, candidates, options, error),
        error,
        path: SamplePath::Adaptive,
        candidate_tests,
        uniform_owner: None,
    }
}

fn classify_interface(
    scene: &Scene,
    volume: &Aabb,
    candidates: &[usize],
    options: &IntegrationOptions,
    unresolved_fraction: f64,
) -> InterfaceClass {
    if options.smoothing == SmoothingMode::Volume {
        return InterfaceClass::None;
    }
    if candidates.len() != 1 {
        return InterfaceClass::Ambiguous(FallbackReason::MultipleObjects);
    }
    // The coarse/fine fraction disagreement is an a-posteriori estimate, not a
    // strict mathematical bound. Only reject smoothing when it says most of
    // the support remains unresolved.
    // The estimator is assembled from recursively scaled volumes.  Values at
    // exactly one half can land one ulp above or below the threshold when the
    // same geometry is expressed in another length unit.  Treat that boundary
    // consistently; only a materially larger unresolved share disables the
    // interface model.
    if unresolved_fraction > 0.5 + 256.0 * f64::EPSILON {
        return InterfaceClass::Ambiguous(FallbackReason::UnresolvedGeometry);
    }
    match scene.objects[candidates[0]]
        .geometry
        .interface_evidence(volume, options.minimum_normal_alignment)
        .assess()
    {
        InterfaceAssessment::Laminar(normal) => InterfaceClass::Laminar(normal),
        InterfaceAssessment::MultipleOrientations => {
            InterfaceClass::Ambiguous(FallbackReason::MultipleOrientations)
        }
        InterfaceAssessment::Missing => {
            InterfaceClass::Ambiguous(FallbackReason::MissingSurfaceEvidence)
        }
    }
}

struct AdaptiveOutput<'a> {
    accumulated: &'a mut [f64],
    estimated_error_volume: &'a mut f64,
    candidate_tests: &'a mut u64,
}

fn adaptive_integrate(
    scene: &Scene,
    volume: &Aabb,
    candidates: &[usize],
    root_volume: f64,
    depth: u8,
    options: &IntegrationOptions,
    output: &mut AdaptiveOutput<'_>,
) {
    let intersecting: Vec<usize> = candidates
        .iter()
        .copied()
        .filter(|index| scene.objects[*index].geometry.bounds().intersects(volume))
        .collect();
    let has_surface = intersecting
        .iter()
        .any(|index| scene.objects[*index].geometry.surface_may_intersect(volume));
    if !has_surface {
        output.accumulated[owner_at_counted(
            scene,
            &intersecting,
            volume.center(),
            output.candidate_tests,
        )] += volume.volume();
        return;
    }
    let (mut fine, mut coarse) =
        octant_material_volumes(scene, volume, &intersecting, output.candidate_tests);
    let mut estimated_error = mixture_difference(&coarse, &fine);
    let mut occupied = fine
        .iter()
        .filter(|occupied_volume| **occupied_volume > 32.0 * f64::EPSILON * volume.volume())
        .count();
    if occupied <= 1 {
        let center_owner = owner_at_counted(
            scene,
            &intersecting,
            volume.center(),
            output.candidate_tests,
        );
        coarse.fill(0.0);
        coarse[center_owner] = volume.volume();
        estimated_error = mixture_difference(&coarse, &fine);
        // A surface is known to cross this node. Agreement on one owner can be
        // aliasing rather than convergence, so force refinement. At the depth
        // limit only these aliased nodes pay for a denser estimate.
        if estimated_error <= 256.0 * f64::EPSILON * volume.volume() {
            estimated_error = volume.volume();
        }
    }
    let at_depth_limit = depth >= options.max_depth;
    if at_depth_limit && occupied <= 1 {
        fine = if options.max_depth == 0 {
            sparse_stratified_material_volumes(scene, volume, &intersecting, output.candidate_tests)
        } else {
            stratified_material_volumes(scene, volume, &intersecting, 3, output.candidate_tests)
        };
        estimated_error = mixture_difference(&coarse, &fine);
        occupied = fine
            .iter()
            .filter(|occupied_volume| **occupied_volume > 32.0 * f64::EPSILON * volume.volume())
            .count();
        if occupied <= 1 {
            estimated_error = volume.volume();
        }
    }
    if at_depth_limit
        || (depth >= options.minimum_depth
            && estimated_error / root_volume <= options.fraction_error_tolerance)
    {
        for (target, occupied_volume) in output.accumulated.iter_mut().zip(fine) {
            *target += occupied_volume;
        }
        *output.estimated_error_volume += estimated_error;
        return;
    }
    for child in 0..8 {
        adaptive_integrate(
            scene,
            &volume.child(child),
            &intersecting,
            root_volume,
            depth + 1,
            options,
            output,
        );
    }
}

fn octant_material_volumes(
    scene: &Scene,
    volume: &Aabb,
    candidates: &[usize],
    candidate_tests: &mut u64,
) -> (Vec<f64>, Vec<f64>) {
    let mut all = vec![0.0; scene.materials.len()];
    let mut checkerboard = vec![0.0; scene.materials.len()];
    for child in 0_usize..8 {
        let owner = owner_at_counted(
            scene,
            candidates,
            volume.child(child).center(),
            candidate_tests,
        );
        all[owner] += volume.volume() / 8.0;
        if child.count_ones() % 2 == 0 {
            checkerboard[owner] += volume.volume() / 4.0;
        }
    }
    (all, checkerboard)
}

fn stratified_material_volumes(
    scene: &Scene,
    volume: &Aabb,
    candidates: &[usize],
    divisions: usize,
    candidate_tests: &mut u64,
) -> Vec<f64> {
    let mut result = vec![0.0; scene.materials.len()];
    let sample_count = divisions.pow(3) as f64;
    let sample_volume = volume.volume() / sample_count;
    let width: [f64; 3] = std::array::from_fn(|axis| volume.max[axis] - volume.min[axis]);
    for z in 0..divisions {
        for y in 0..divisions {
            for x in 0..divisions {
                let indices = [x, y, z];
                let point = std::array::from_fn(|axis| {
                    volume.min[axis] + (indices[axis] as f64 + 0.5) * width[axis] / divisions as f64
                });
                result[owner_at_counted(scene, candidates, point, candidate_tests)] +=
                    sample_volume;
            }
        }
    }
    result
}

fn sparse_stratified_material_volumes(
    scene: &Scene,
    volume: &Aabb,
    candidates: &[usize],
    candidate_tests: &mut u64,
) -> Vec<f64> {
    let mut result = vec![0.0; scene.materials.len()];
    let sample_volume = volume.volume() / 9.0;
    let width: [f64; 3] = std::array::from_fn(|axis| volume.max[axis] - volume.min[axis]);
    // A 3x3 Latin pattern covers every low/middle/high stratum on every axis
    // with nine points. It catches thin axis-aligned layers without making the
    // fast preset pay for the full 27-point reference fallback.
    for index in 0..9 {
        let strata = [index % 3, index / 3, (index % 3 + index / 3) % 3];
        let point = std::array::from_fn(|axis| {
            volume.min[axis] + (strata[axis] as f64 + 0.5) * width[axis] / 3.0
        });
        result[owner_at_counted(scene, candidates, point, candidate_tests)] += sample_volume;
    }
    result
}

fn mixture_difference(lhs: &[f64], rhs: &[f64]) -> f64 {
    0.5 * lhs
        .iter()
        .zip(rhs)
        .map(|(left, right)| (left - right).abs())
        .sum::<f64>()
}

fn owner_at(scene: &Scene, candidates: &[usize], point: [f64; 3]) -> usize {
    candidates
        .iter()
        .map(|index| &scene.objects[*index])
        .filter(|object| object.geometry.contains_half_open(point))
        .max_by_key(|object| (object.priority, object.id))
        .map_or(scene.background_material, |object| object.material_id)
}

fn owner_at_counted(
    scene: &Scene,
    candidates: &[usize],
    point: [f64; 3],
    candidate_tests: &mut u64,
) -> usize {
    *candidate_tests += candidates.len() as u64;
    owner_at(scene, candidates, point)
}

fn effective_material(
    scene: &Scene,
    fractions: &[f64],
    smoothing: SmoothingMode,
    interface: InterfaceClass,
) -> (Material, bool, Option<FallbackReason>) {
    let occupied = fractions
        .iter()
        .filter(|fraction| **fraction > 32.0 * f64::EPSILON)
        .count();
    let normal = match interface {
        InterfaceClass::Laminar(normal) => Some(normal),
        InterfaceClass::None | InterfaceClass::Ambiguous(_) => None,
    };
    let can_smooth = smoothing != SmoothingMode::Volume && occupied > 1 && normal.is_some();
    let epsilon_r = if can_smooth {
        farjadpour_average(scene, fractions, normal.unwrap(), |material| {
            material.epsilon_r
        })
    } else {
        volume_average(scene, fractions, |material| material.epsilon_r)
    };
    let mu_r = if can_smooth {
        farjadpour_average(scene, fractions, normal.unwrap(), |material| material.mu_r)
    } else {
        volume_average(scene, fractions, |material| material.mu_r)
    };
    let conductivity = volume_average(scene, fractions, |material| material.conductivity);
    (
        Material {
            epsilon_r,
            mu_r,
            conductivity,
        },
        can_smooth,
        if smoothing == SmoothingMode::Volume || occupied <= 1 || can_smooth {
            None
        } else {
            Some(match interface {
                InterfaceClass::Ambiguous(reason) => reason,
                InterfaceClass::None | InterfaceClass::Laminar(_) => {
                    FallbackReason::MissingSurfaceEvidence
                }
            })
        },
    )
}

fn volume_average(
    scene: &Scene,
    fractions: &[f64],
    property: impl Fn(&Material) -> SymmetricTensor,
) -> SymmetricTensor {
    let mut result = SymmetricTensor([0.0; 6]);
    for (material, fraction) in scene.materials.iter().zip(fractions) {
        result.scaled_add(property(material), *fraction);
    }
    result
}

fn farjadpour_average(
    scene: &Scene,
    fractions: &[f64],
    normal: [f64; 3],
    property: impl Fn(&Material) -> SymmetricTensor,
) -> SymmetricTensor {
    let mut participating = scene
        .materials
        .iter()
        .zip(fractions)
        .filter(|(_, fraction)| **fraction > 32.0 * f64::EPSILON)
        .map(|(material, _)| property(material));
    if let Some(first) = participating.next() {
        if participating.all(|value| value == first) {
            return first;
        }
    }
    let basis = interface_basis(normal);
    let mut averaged_tau = [[0.0; 3]; 3];
    for (material, fraction) in scene.materials.iter().zip(fractions) {
        if *fraction == 0.0 {
            continue;
        }
        let local = rotate_to_local(property(material).matrix(), basis);
        let tau = tau_transform(local);
        for row in 0..3 {
            for col in 0..3 {
                averaged_tau[row][col] += fraction * tau[row][col];
            }
        }
    }
    let local = inverse_tau_transform(averaged_tau);
    SymmetricTensor::from_matrix(rotate_to_global(local, basis))
}

fn interface_basis(normal: [f64; 3]) -> [[f64; 3]; 3] {
    let length = normal.iter().map(|value| value * value).sum::<f64>().sqrt();
    let n = normal.map(|value| value / length);
    let reference = if n[0].abs() <= n[1].abs() && n[0].abs() <= n[2].abs() {
        [1.0, 0.0, 0.0]
    } else if n[1].abs() <= n[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let tangent = normalize(cross(reference, n));
    let second = cross(n, tangent);
    [n, tangent, second]
}

fn normalize(value: [f64; 3]) -> [f64; 3] {
    let length = value.iter().map(|entry| entry * entry).sum::<f64>().sqrt();
    value.map(|entry| entry / length)
}

fn cross(lhs: [f64; 3], rhs: [f64; 3]) -> [f64; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

fn rotate_to_local(value: [[f64; 3]; 3], basis: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    // Basis vectors are rows, so local = B * global * B^T.
    multiply(multiply(basis, value), transpose(basis))
}

fn rotate_to_global(value: [[f64; 3]; 3], basis: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    multiply(multiply(transpose(basis), value), basis)
}

fn transpose(value: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|col| value[col][row]))
}

fn multiply(lhs: [[f64; 3]; 3], rhs: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| {
        std::array::from_fn(|col| (0..3).map(|inner| lhs[row][inner] * rhs[inner][col]).sum())
    })
}

fn tau_transform(epsilon: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let pivot = epsilon[0][0];
    let mut tau = [[0.0; 3]; 3];
    tau[0][0] = -1.0 / pivot;
    for tangent in 1..3 {
        tau[0][tangent] = epsilon[0][tangent] / pivot;
        tau[tangent][0] = epsilon[tangent][0] / pivot;
    }
    for row in 1..3 {
        for col in 1..3 {
            tau[row][col] = epsilon[row][col] - epsilon[row][0] * epsilon[0][col] / pivot;
        }
    }
    tau
}

fn inverse_tau_transform(tau: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let pivot = tau[0][0];
    let mut epsilon = [[0.0; 3]; 3];
    epsilon[0][0] = -1.0 / pivot;
    for tangent in 1..3 {
        epsilon[0][tangent] = -tau[0][tangent] / pivot;
        epsilon[tangent][0] = -tau[tangent][0] / pivot;
    }
    for row in 1..3 {
        for col in 1..3 {
            epsilon[row][col] = tau[row][col] - tau[row][0] * tau[0][col] / pivot;
        }
    }
    epsilon
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::{Geometry, Material, Object, Polygon2, UniformGrid};

    use super::*;

    fn box_scene(bounds: Aabb, epsilon: f64) -> Scene {
        Scene::new(
            vec![
                Material::default(),
                Material::new(epsilon, 1.0, 0.0).unwrap(),
            ],
            vec![Object {
                id: 1,
                material_id: 1,
                priority: 0,
                geometry: Geometry::Box { bounds },
            }],
            0,
        )
        .unwrap()
    }

    #[test]
    fn yee_shapes_are_component_specific() {
        let scene = box_scene(Aabb::new([0.0; 3], [1.0; 3]).unwrap(), 4.0);
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [2, 3, 4],
        }
        .build()
        .unwrap();
        let result = rasterize(&scene, &grid, &IntegrationOptions::default()).unwrap();
        assert_eq!(result.epsilon[0].shape, [1, 5, 4, 2]);
        assert_eq!(result.epsilon[1].shape, [1, 5, 3, 3]);
        assert_eq!(result.epsilon[2].shape, [1, 4, 4, 3]);
        assert_eq!(result.mu[0].shape, [1, 4, 3, 3]);
        assert_eq!(result.mu[1].shape, [1, 4, 4, 2]);
        assert_eq!(result.mu[2].shape, [1, 5, 3, 2]);
        assert!(result.epsilon[0].values.iter().all(|value| *value == 4.0));
    }

    #[test]
    fn exact_box_fraction_is_volume_weighted() {
        let scene = box_scene(Aabb::new([0.25, 0.0, 0.0], [0.75, 1.0, 1.0]).unwrap(), 5.0);
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [1, 1, 1],
        }
        .build()
        .unwrap();
        let result = rasterize(&scene, &grid, &IntegrationOptions::default()).unwrap();
        assert_abs_diff_eq!(result.epsilon[1].values[0], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn curved_fraction_quality_is_estimate_driven_and_never_disappears() {
        let scene = Scene::new(
            vec![Material::default(), Material::new(2.0, 1.0, 0.0).unwrap()],
            vec![Object {
                id: 1,
                material_id: 1,
                priority: 0,
                geometry: Geometry::Sphere {
                    center: [0.5; 3],
                    radius: 0.4,
                },
            }],
            0,
        )
        .unwrap();
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [1; 3],
        }
        .build()
        .unwrap();
        let exact_fraction = 4.0 * std::f64::consts::PI * 0.4_f64.powi(3) / 3.0;
        let fast_options = IntegrationOptions::for_quality(Quality::Fast);
        let fast_mixture = integrate_mixture(
            &scene,
            &Aabb::new([0.0; 3], [1.0; 3]).unwrap(),
            &[0],
            &fast_options,
        );
        // One candidate lookup, eight octants, one alias check, and the
        // nine-point depth-limit fallback. This deterministic work budget guards
        // the fast preset against accidental sampling explosions in CI.
        assert_eq!(fast_mixture.candidate_tests, 19);
        let fast = rasterize(&scene, &grid, &fast_options).unwrap();
        let reference = rasterize(
            &scene,
            &grid,
            &IntegrationOptions::for_quality(Quality::Reference),
        )
        .unwrap();
        let fast_fraction = f64::from(fast.cell_epsilon.values[0] - 1.0);
        let reference_fraction = f64::from(reference.cell_epsilon.values[0] - 1.0);
        assert!(fast_fraction > 0.2);
        assert!(
            (reference_fraction - exact_fraction).abs() < (fast_fraction - exact_fraction).abs()
        );
        assert!(
            reference.diagnostics.maximum_estimated_fraction_error
                < fast.diagnostics.maximum_estimated_fraction_error
        );
    }

    #[test]
    fn tapered_mixture_is_scale_invariant() {
        fn scene(scale: f64) -> Scene {
            let polygon = Polygon2::new(
                vec![
                    [0.173 * scale, 0.193 * scale],
                    [0.817 * scale, 0.223 * scale],
                    [0.791 * scale, 0.843 * scale],
                    [0.201 * scale, 0.779 * scale],
                ],
                vec![],
            )
            .unwrap();
            Scene::new(
                vec![Material::default(), Material::new(4.0, 1.0, 0.0).unwrap()],
                vec![Object {
                    id: 1,
                    material_id: 1,
                    priority: 0,
                    geometry: Geometry::TaperedExtrudedPolygon(
                        crate::TaperedExtrudedPolygon::new(
                            polygon,
                            0.041 * scale,
                            0.937 * scale,
                            17.3,
                            0.37,
                        )
                        .unwrap(),
                    ),
                }],
                0,
            )
            .unwrap()
        }
        let volume = Aabb::new([0.4, 4.0 / 6.0, 4.0 / 7.0], [0.6, 5.0 / 6.0, 5.0 / 7.0]).unwrap();
        let scaled = Aabb::new(
            volume.min.map(|value| value * 1e-6),
            volume.max.map(|value| value * 1e-6),
        )
        .unwrap();
        let options = IntegrationOptions {
            smoothing: SmoothingMode::FarjadpourFull,
            ..IntegrationOptions::for_quality(Quality::Reference)
        };
        let unit_scene = scene(1.0);
        let micro_scene = scene(1e-6);
        let unit = integrate_mixture(&unit_scene, &volume, &[0], &options);
        let micro = integrate_mixture(&micro_scene, &scaled, &[0], &options);
        assert_abs_diff_eq!(unit.error, micro.error, epsilon = 1e-12);
        assert_abs_diff_eq!(
            unit.fractions.as_ref().unwrap()[1],
            micro.fractions.as_ref().unwrap()[1],
            epsilon = 1e-12
        );
        assert!(matches!(
            unit.interface,
            InterfaceClass::Ambiguous(FallbackReason::UnresolvedGeometry)
        ));
        assert!(matches!(
            micro.interface,
            InterfaceClass::Ambiguous(FallbackReason::UnresolvedGeometry)
        ));
    }

    #[test]
    fn fast_quality_does_not_erase_a_thin_tapered_layer() {
        let polygon =
            Polygon2::new(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], vec![]).unwrap();
        let scene = Scene::new(
            vec![Material::default(), Material::new(4.0, 1.0, 0.0).unwrap()],
            vec![Object {
                id: 1,
                material_id: 1,
                priority: 0,
                geometry: Geometry::TaperedExtrudedPolygon(
                    crate::TaperedExtrudedPolygon::new(polygon, 0.0, 0.22, 10.0, 0.0).unwrap(),
                ),
            }],
            0,
        )
        .unwrap();
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [1; 3],
        }
        .build()
        .unwrap();
        let result = rasterize(
            &scene,
            &grid,
            &IntegrationOptions::for_quality(Quality::Fast),
        )
        .unwrap();

        assert!(result.cell_epsilon.values[0] > 1.0);
    }

    #[test]
    fn extruded_polygon_overlap_is_exact() {
        let polygon =
            Polygon2::new(vec![[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]], vec![]).unwrap();
        let scene = Scene::new(
            vec![Material::default(), Material::new(9.0, 1.0, 0.0).unwrap()],
            vec![Object {
                id: 1,
                material_id: 1,
                priority: 0,
                geometry: Geometry::ExtrudedPolygon(
                    crate::ExtrudedPolygon::new(polygon, 0.0, 1.0).unwrap(),
                ),
            }],
            0,
        )
        .unwrap();
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [1, 1, 1],
        }
        .build()
        .unwrap();
        let result = rasterize(&scene, &grid, &IntegrationOptions::default()).unwrap();
        assert_abs_diff_eq!(result.cell_epsilon.values[0], 5.0, epsilon = 1e-6);
    }

    #[test]
    fn two_dimensional_tm_only_emits_solver_components() {
        let scene = box_scene(Aabb::new([0.2, 0.2, 0.0], [0.8, 0.8, 1.0]).unwrap(), 12.0);
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [12, 11, 1],
        }
        .build()
        .unwrap();
        let options = IntegrationOptions {
            output_components: OutputComponents::TwoDimensionalTm,
            ..IntegrationOptions::default()
        };
        let result = rasterize(&scene, &grid, &options).unwrap();
        assert!(result.epsilon[0].values.is_empty());
        assert!(result.epsilon[1].values.is_empty());
        assert!(!result.epsilon[2].values.is_empty());
        assert!(!result.mu[0].values.is_empty());
        assert!(!result.mu[1].values.is_empty());
        assert!(result.mu[2].values.is_empty());
        assert!(!result.cell_epsilon.values.is_empty());
    }

    #[test]
    fn two_dimensional_te_only_emits_solver_components() {
        let scene = box_scene(Aabb::new([0.2, 0.2, 0.0], [0.8, 0.8, 1.0]).unwrap(), 12.0);
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [12, 11, 1],
        }
        .build()
        .unwrap();
        let options = IntegrationOptions {
            output_components: OutputComponents::TwoDimensionalTe,
            ..IntegrationOptions::default()
        };
        let result = rasterize(&scene, &grid, &options).unwrap();
        assert!(!result.epsilon[0].values.is_empty());
        assert!(!result.epsilon[1].values.is_empty());
        assert!(result.epsilon[2].values.is_empty());
        assert!(result.mu[0].values.is_empty());
        assert!(result.mu[1].values.is_empty());
        assert!(!result.mu[2].values.is_empty());
        assert!(result.node_epsilon.values.is_empty());
        assert!(!result.cell_epsilon.values.is_empty());
    }

    #[test]
    fn diagonal_mode_is_harmonic_normal_to_axis_aligned_interface() {
        let scene = box_scene(Aabb::new([0.0, 0.0, 0.0], [0.5, 1.0, 1.0]).unwrap(), 4.0);
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [1, 1, 1],
        }
        .build()
        .unwrap();
        let options = IntegrationOptions {
            smoothing: SmoothingMode::FarjadpourDiagonal,
            ..IntegrationOptions::default()
        };
        let result = rasterize(&scene, &grid, &options).unwrap();
        assert_abs_diff_eq!(result.epsilon[0].values[0], 1.6, epsilon = 1e-6);
        let effective = farjadpour_average(&scene, &[0.5, 0.5], [1.0, 0.0, 0.0], |m| m.epsilon_r);
        assert_abs_diff_eq!(effective.0[0], 1.6, epsilon = 1e-12);
        assert_abs_diff_eq!(effective.0[1], 2.5, epsilon = 1e-12);
    }

    #[test]
    fn tau_transform_round_trips_symmetric_tensor() {
        let tensor = [[4.0, 0.2, -0.1], [0.2, 3.0, 0.3], [-0.1, 0.3, 2.0]];
        let recovered = inverse_tau_transform(tau_transform(tensor));
        for row in 0..3 {
            for col in 0..3 {
                assert_abs_diff_eq!(recovered[row][col], tensor[row][col], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn farjadpour_is_rotation_covariant_for_isotropic_media() {
        let scene = Scene::new(
            vec![Material::default(), Material::new(4.0, 1.0, 0.0).unwrap()],
            vec![],
            0,
        )
        .unwrap();
        let fractions = [0.5, 0.5];
        let effective = farjadpour_average(&scene, &fractions, [1.0, 1.0, 0.0], |m| m.epsilon_r);
        assert_abs_diff_eq!(effective.0[0], 2.05, epsilon = 1e-12);
        assert_abs_diff_eq!(effective.0[1], 2.05, epsilon = 1e-12);
        assert_abs_diff_eq!(effective.0[3], -0.45, epsilon = 1e-12);
        assert_abs_diff_eq!(effective.0[2], 2.5, epsilon = 1e-12);
    }

    #[test]
    fn full_tensors_are_integrated_independently_at_each_constitutive_support() {
        let polygon = Polygon2::new(vec![[-1.0, -1.0], [2.0, -1.0], [-1.0, 2.0]], vec![]).unwrap();
        let geometry =
            Geometry::ExtrudedPolygon(crate::ExtrudedPolygon::new(polygon, -1.0, 2.0).unwrap());
        let scene = Scene::new(
            vec![Material::default(), Material::new(4.0, 1.0, 0.0).unwrap()],
            vec![Object {
                id: 1,
                material_id: 1,
                priority: 0,
                geometry: geometry.clone(),
            }],
            0,
        )
        .unwrap();
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [1; 3],
        }
        .build()
        .unwrap();
        let options = IntegrationOptions {
            smoothing: SmoothingMode::FarjadpourFull,
            ..IntegrationOptions::default()
        };
        let result = rasterize(&scene, &grid, &options).unwrap();

        for (component, tensor) in [Component::Ex, Component::Ey, Component::Ez, Component::Node]
            .into_iter()
            .zip(result.epsilon.iter().chain([&result.node_epsilon]))
        {
            assert_eq!(tensor.shape[0], 6);
            let logical_shape = component.support().logical_shape(&grid);
            let sample_count: usize = logical_shape.iter().product();
            for flat in 0..sample_count {
                let x = flat % logical_shape[0];
                let rest = flat / logical_shape[0];
                let y = rest % logical_shape[1];
                let z = rest / logical_shape[1];
                let support = component.support().volume(&grid, [x, y, z]);
                let fraction = geometry.exact_overlap_volume(&support).unwrap() / support.volume();
                let arithmetic = 1.0 + 3.0 * fraction;
                let harmonic = 1.0 / (1.0 - fraction + fraction / 4.0);
                let expected = [
                    0.5 * (arithmetic + harmonic),
                    0.5 * (arithmetic + harmonic),
                    arithmetic,
                    0.5 * (harmonic - arithmetic),
                    0.0,
                    0.0,
                ];
                for (axis, value) in expected.into_iter().enumerate() {
                    assert_abs_diff_eq!(
                        tensor.values[axis * sample_count + flat],
                        value as f32,
                        epsilon = 2e-6
                    );
                }
            }
        }
        assert!(result.mu.iter().all(|tensor| tensor.shape[0] == 1));
        assert!(
            result
                .conductivity
                .iter()
                .all(|tensor| tensor.shape[0] == 1)
        );
    }

    #[test]
    fn nonuniform_dual_volume_controls_support_tensor_fraction() {
        let scene = box_scene(Aabb::new([0.0, -1.0, -1.0], [1.0, 3.0, 5.0]).unwrap(), 4.0);
        let grid = Grid::new(vec![0.0, 1.0, 3.0], vec![0.0, 2.0], vec![0.0, 4.0]).unwrap();
        let options = IntegrationOptions {
            smoothing: SmoothingMode::FarjadpourFull,
            ..IntegrationOptions::default()
        };
        let result = rasterize(&scene, &grid, &options).unwrap();

        // Ey at x-edge 1 owns the nonuniform dual interval [0.5, 2.0]. The
        // high-index material occupies one third of it, so epsilon normal to
        // the interface is harmonic and the tangential value is arithmetic.
        let tensor = &result.epsilon[1];
        let samples = tensor.shape[1..].iter().product::<usize>();
        let flat = 1;
        assert_abs_diff_eq!(tensor.values[flat], 4.0 / 3.0, epsilon = 2e-6);
        assert_abs_diff_eq!(tensor.values[samples + flat], 2.0, epsilon = 2e-6);
    }

    #[test]
    fn planar_and_parallel_interfaces_are_smoothed() {
        for bounds in [
            Aabb::new([0.0, -1.0, -1.0], [0.5, 2.0, 2.0]).unwrap(),
            Aabb::new([0.25, -1.0, -1.0], [0.75, 2.0, 2.0]).unwrap(),
        ] {
            let scene = box_scene(bounds, 4.0);
            let grid = UniformGrid {
                min: [0.0; 3],
                max: [1.0; 3],
                shape: [1; 3],
            }
            .build()
            .unwrap();
            let options = IntegrationOptions {
                smoothing: SmoothingMode::FarjadpourFull,
                ..IntegrationOptions::default()
            };
            let result = rasterize(&scene, &grid, &options).unwrap();
            assert!(result.diagnostics.smoothed_samples > 0);
            assert_eq!(result.diagnostics.ambiguous_interface_samples, 0);
        }
    }

    #[test]
    fn multi_orientation_support_falls_back_to_volume_average() {
        let scene = box_scene(Aabb::new([0.0, 0.0, -1.0], [0.5, 0.5, 2.0]).unwrap(), 4.0);
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [1; 3],
        }
        .build()
        .unwrap();
        let options = IntegrationOptions {
            smoothing: SmoothingMode::FarjadpourFull,
            ..IntegrationOptions::default()
        };
        let result = rasterize(&scene, &grid, &options).unwrap();
        assert_abs_diff_eq!(result.cell_epsilon.values[0], 1.75, epsilon = 1e-6);
        assert!(result.diagnostics.fallback_multiple_orientations > 0);
    }

    #[test]
    fn multiple_candidate_objects_have_an_explicit_fallback_reason() {
        let scene = Scene::new(
            vec![
                Material::default(),
                Material::new(2.0, 1.0, 0.0).unwrap(),
                Material::new(4.0, 1.0, 0.0).unwrap(),
            ],
            vec![
                Object {
                    id: 1,
                    material_id: 1,
                    priority: 0,
                    geometry: Geometry::Box {
                        bounds: Aabb::new([0.0, -1.0, -1.0], [0.6, 2.0, 2.0]).unwrap(),
                    },
                },
                Object {
                    id: 2,
                    material_id: 2,
                    priority: 1,
                    geometry: Geometry::Box {
                        bounds: Aabb::new([0.4, -1.0, -1.0], [1.0, 2.0, 2.0]).unwrap(),
                    },
                },
            ],
            0,
        )
        .unwrap();
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [1; 3],
        }
        .build()
        .unwrap();
        let options = IntegrationOptions {
            smoothing: SmoothingMode::FarjadpourFull,
            ..IntegrationOptions::default()
        };
        let result = rasterize(&scene, &grid, &options).unwrap();
        assert!(result.diagnostics.fallback_multiple_objects > 0);
    }

    #[test]
    fn undefined_interface_normal_falls_back_without_nan() {
        let scene = Scene::new(
            vec![Material::default(), Material::new(4.0, 1.0, 0.0).unwrap()],
            vec![Object {
                id: 1,
                material_id: 1,
                priority: 0,
                geometry: Geometry::Sphere {
                    center: [0.5; 3],
                    radius: 0.4,
                },
            }],
            0,
        )
        .unwrap();
        let grid = UniformGrid {
            min: [0.0; 3],
            max: [1.0; 3],
            shape: [1; 3],
        }
        .build()
        .unwrap();
        let options = IntegrationOptions {
            smoothing: SmoothingMode::FarjadpourFull,
            ..IntegrationOptions::default()
        };
        let result = rasterize(&scene, &grid, &options).unwrap();
        assert!(
            result
                .cell_epsilon
                .values
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(result.diagnostics.ambiguous_interface_samples > 0);
    }
}
