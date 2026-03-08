/// Typed errors for vctrs operations.

use std::fmt;

#[derive(Debug)]
pub enum VctrsError {
    /// Vector dimension doesn't match the index.
    DimensionMismatch { expected: usize, got: usize },
    /// ID already exists (use upsert instead).
    DuplicateId(String),
    /// ID not found in the database.
    NotFound(String),
    /// Database not found at path.
    DatabaseNotFound(String),
    /// IO error during save/load.
    Io(std::io::Error),
    /// Invalid metric string.
    InvalidMetric(String),
    /// Data corruption or invalid format.
    CorruptData(String),
    /// Invalid filter value or operator.
    InvalidFilter(String),
}

impl fmt::Display for VctrsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VctrsError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, got)
            }
            VctrsError::DuplicateId(id) => {
                write!(f, "id '{}' already exists, use upsert instead", id)
            }
            VctrsError::NotFound(id) => write!(f, "id '{}' not found", id),
            VctrsError::DatabaseNotFound(path) => {
                write!(f, "database not found at '{}'", path)
            }
            VctrsError::Io(e) => write!(f, "io error: {}", e),
            VctrsError::InvalidMetric(m) => {
                write!(f, "invalid metric '{}': use 'cosine', 'euclidean'/'l2', or 'dot'/'dot_product'", m)
            }
            VctrsError::CorruptData(msg) => write!(f, "corrupt data: {}", msg),
            VctrsError::InvalidFilter(msg) => write!(f, "invalid filter: {}", msg),
        }
    }
}

impl std::error::Error for VctrsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VctrsError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for VctrsError {
    fn from(e: std::io::Error) -> Self {
        VctrsError::Io(e)
    }
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, VctrsError>;
