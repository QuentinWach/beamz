//! Geometry integration and tensor constitutive lowering for Cartesian FDTD.

mod error;
mod geometry;
mod grid;
mod mesh;
mod raster;
mod scene;

pub use error::{RasterError, Result};
pub use geometry::{Aabb, ExtrudedPolygon, Geometry, Polygon2, TaperedExtrudedPolygon, Vec3};
pub use grid::{Grid, UniformGrid};
pub use mesh::{MeshReport, TriangleMesh};
pub use raster::{
    DiagnosticSummary, IntegrationOptions, OutputComponents, Quality, RasterResult, SmoothingMode,
    TensorArray, rasterize, rasterize_prevalidated,
};
pub use scene::{Material, Object, Scene, SymmetricTensor};
