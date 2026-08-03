use thiserror::Error;

pub type Result<T> = std::result::Result<T, RasterError>;

#[derive(Debug, Error)]
pub enum RasterError {
    #[error("invalid grid: {0}")]
    InvalidGrid(String),
    #[error("invalid polygon: {0}")]
    InvalidPolygon(String),
    #[error("invalid mesh: {0}")]
    InvalidMesh(String),
    #[error("invalid material: {0}")]
    InvalidMaterial(String),
    #[error("invalid scene: {0}")]
    InvalidScene(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
