/// JSON export/import for backup and migration.
///
/// Format:
/// ```json
/// {
///   "version": 1,
///   "dim": 384,
///   "metric": "cosine",
///   "vectors": [
///     {"id": "doc1", "vector": [0.1, 0.2, ...], "metadata": {"key": "val"}},
///     ...
///   ]
/// }
/// ```

use crate::db::Database;
use crate::distance::Metric;
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

#[derive(Serialize, Deserialize)]
struct ExportFormat {
    version: u32,
    dim: usize,
    metric: String,
    vectors: Vec<ExportVector>,
}

#[derive(Serialize, Deserialize)]
struct ExportVector {
    id: String,
    vector: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
}

impl Database {
    /// Export all vectors and metadata to a JSON writer.
    pub fn export_json<W: Write>(&self, writer: W) -> Result<()> {
        let ids = self.ids();
        let mut vectors = Vec::with_capacity(ids.len());

        for id in &ids {
            let (vec, meta) = self.get(id)?;
            vectors.push(ExportVector {
                id: id.clone(),
                vector: vec,
                metadata: meta,
            });
        }

        let export = ExportFormat {
            version: 1,
            dim: self.dim(),
            metric: match self.metric() {
                Metric::Cosine => "cosine".to_string(),
                Metric::Euclidean => "euclidean".to_string(),
                Metric::DotProduct => "dot_product".to_string(),
            },
            vectors,
        };

        serde_json::to_writer(writer, &export)?;
        Ok(())
    }

    /// Export all vectors and metadata to a pretty-printed JSON writer.
    pub fn export_json_pretty<W: Write>(&self, writer: W) -> Result<()> {
        let ids = self.ids();
        let mut vectors = Vec::with_capacity(ids.len());

        for id in &ids {
            let (vec, meta) = self.get(id)?;
            vectors.push(ExportVector {
                id: id.clone(),
                vector: vec,
                metadata: meta,
            });
        }

        let export = ExportFormat {
            version: 1,
            dim: self.dim(),
            metric: match self.metric() {
                Metric::Cosine => "cosine".to_string(),
                Metric::Euclidean => "euclidean".to_string(),
                Metric::DotProduct => "dot_product".to_string(),
            },
            vectors,
        };

        serde_json::to_writer_pretty(writer, &export)?;
        Ok(())
    }

    /// Import vectors from a JSON reader into a new database at the given path.
    pub fn import_json<R: Read>(reader: R, path: &str) -> Result<Database> {
        let export: ExportFormat = serde_json::from_reader(reader)
            .map_err(|e| crate::error::VctrsError::CorruptData(format!("invalid JSON: {}", e)))?;

        let metric = match export.metric.as_str() {
            "cosine" => Metric::Cosine,
            "euclidean" | "l2" => Metric::Euclidean,
            "dot" | "dot_product" => Metric::DotProduct,
            other => return Err(crate::error::VctrsError::InvalidMetric(other.to_string())),
        };

        let db = Database::open_or_create(path, export.dim, metric)?;

        let items: Vec<_> = export
            .vectors
            .into_iter()
            .map(|v| (v.id, v.vector, v.metadata))
            .collect();

        if !items.is_empty() {
            db.add_many(items)?;
        }

        Ok(db)
    }

    /// Import vectors from a JSON reader into an existing database.
    /// Uses upsert semantics (updates existing IDs, inserts new ones).
    pub fn import_json_into<R: Read>(&self, reader: R) -> Result<()> {
        let export: ExportFormat = serde_json::from_reader(reader)
            .map_err(|e| crate::error::VctrsError::CorruptData(format!("invalid JSON: {}", e)))?;

        if export.dim != self.dim() {
            return Err(crate::error::VctrsError::DimensionMismatch {
                expected: self.dim(),
                got: export.dim,
            });
        }

        let items: Vec<_> = export
            .vectors
            .into_iter()
            .map(|v| (v.id, v.vector, v.metadata))
            .collect();

        if !items.is_empty() {
            self.upsert_many(items)?;
        }

        Ok(())
    }
}

// serde_json errors -> VctrsError
impl From<serde_json::Error> for crate::error::VctrsError {
    fn from(e: serde_json::Error) -> Self {
        crate::error::VctrsError::Io(std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;

    #[test]
    fn test_export_import_roundtrip() {
        let dir1 = tempfile::tempdir().unwrap();
        let path1 = dir1.path().to_str().unwrap();
        let db = Database::open_or_create(path1, 3, Metric::Cosine).unwrap();
        db.add("a", vec![1.0, 0.0, 0.0], Some(serde_json::json!({"tag": "first"}))).unwrap();
        db.add("b", vec![0.0, 1.0, 0.0], None).unwrap();
        db.add("c", vec![0.0, 0.0, 1.0], Some(serde_json::json!({"tag": "third"}))).unwrap();

        let mut buf = Vec::new();
        db.export_json(&mut buf).unwrap();
        drop(db);

        let dir2 = tempfile::tempdir().unwrap();
        let path2 = dir2.path().join("imported").to_str().unwrap().to_string();
        let db2 = Database::import_json(&buf[..], &path2).unwrap();

        assert_eq!(db2.len(), 3);
        assert_eq!(db2.dim(), 3);

        let (vec_a, meta_a) = db2.get("a").unwrap();
        assert_eq!(vec_a, vec![1.0, 0.0, 0.0]);
        assert_eq!(meta_a, Some(serde_json::json!({"tag": "first"})));

        let (_, meta_b) = db2.get("b").unwrap();
        assert!(meta_b.is_none());
    }

    #[test]
    fn test_import_into_existing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        let db = Database::open_or_create(path, 3, Metric::Cosine).unwrap();
        db.add("a", vec![1.0, 0.0, 0.0], None).unwrap();

        let json = serde_json::json!({
            "version": 1,
            "dim": 3,
            "metric": "cosine",
            "vectors": [
                {"id": "a", "vector": [0.5, 0.5, 0.0], "metadata": {"updated": true}},
                {"id": "b", "vector": [0.0, 1.0, 0.0]}
            ]
        });
        let json_bytes = serde_json::to_vec(&json).unwrap();

        db.import_json_into(&json_bytes[..]).unwrap();
        assert_eq!(db.len(), 2);

        let (vec_a, meta_a) = db.get("a").unwrap();
        assert_eq!(vec_a[0], 0.5); // Updated.
        assert_eq!(meta_a, Some(serde_json::json!({"updated": true})));
    }

    #[test]
    fn test_export_pretty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        let db = Database::open_or_create(path, 2, Metric::Cosine).unwrap();
        db.add("x", vec![1.0, 0.0], None).unwrap();

        let mut buf = Vec::new();
        db.export_json_pretty(&mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains('\n')); // Pretty-printed.
        assert!(s.contains("\"dim\": 2"));
    }

    #[test]
    fn test_import_dimension_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap();
        let db = Database::open_or_create(path, 3, Metric::Cosine).unwrap();

        let json = serde_json::json!({
            "version": 1,
            "dim": 5,
            "metric": "cosine",
            "vectors": []
        });
        let json_bytes = serde_json::to_vec(&json).unwrap();

        let err = db.import_json_into(&json_bytes[..]).unwrap_err();
        assert!(matches!(err, crate::error::VctrsError::DimensionMismatch { .. }));
    }
}
