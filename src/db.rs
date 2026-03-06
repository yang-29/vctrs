/// The main database: ties together HNSW index, storage, and id mapping.

use crate::distance::Metric;
use crate::hnsw::HnswIndex;
use crate::storage::{Record, Storage};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::PathBuf;

pub struct Database {
    /// HNSW index for fast search.
    index: RwLock<HnswIndex>,
    /// Map from string id -> internal id.
    id_map: RwLock<HashMap<String, u32>>,
    /// Map from internal id -> string id.
    reverse_map: RwLock<Vec<String>>,
    /// Metadata for each vector, indexed by internal id.
    metadata: RwLock<Vec<Option<serde_json::Value>>>,
    /// Persistent storage.
    storage: Storage,
    /// Metric used.
    metric: Metric,
    /// Dimensionality.
    dim: usize,
}

/// A search result returned to the user.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub metadata: Option<serde_json::Value>,
}

impl Database {
    /// Open or create a database at the given path.
    pub fn open(path: &str, dim: usize, metric: Metric) -> Result<Self, String> {
        let db_path = PathBuf::from(path);

        // Create directory if needed.
        std::fs::create_dir_all(&db_path).map_err(|e| e.to_string())?;

        let data_path = db_path.join("data.vctrs");
        let storage = Storage::new(&data_path, dim);

        let mut index = HnswIndex::new(dim, metric, 16, 200);
        let mut id_map = HashMap::new();
        let mut reverse_map = Vec::new();
        let mut meta = Vec::new();

        // Load existing data if present.
        let records = storage.load().map_err(|e| e.to_string())?;
        for record in &records {
            let internal_id = index.insert(record.vector.clone());
            id_map.insert(record.string_id.clone(), internal_id);
            reverse_map.push(record.string_id.clone());
            meta.push(record.metadata.clone());
        }

        Ok(Database {
            index: RwLock::new(index),
            id_map: RwLock::new(id_map),
            reverse_map: RwLock::new(reverse_map),
            metadata: RwLock::new(meta),
            storage,
            metric,
            dim,
        })
    }

    /// Add a vector with a string id and optional metadata.
    pub fn add(
        &self,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<(), String> {
        if vector.len() != self.dim {
            return Err(format!(
                "dimension mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            ));
        }

        let mut id_map = self.id_map.write();
        if id_map.contains_key(id) {
            return Err(format!("id '{}' already exists, use update instead", id));
        }

        let mut index = self.index.write();
        let internal_id = index.insert(vector);

        id_map.insert(id.to_string(), internal_id);
        self.reverse_map.write().push(id.to_string());
        self.metadata.write().push(metadata);

        Ok(())
    }

    /// Search for the k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, String> {
        if query.len() != self.dim {
            return Err(format!(
                "dimension mismatch: expected {}, got {}",
                self.dim,
                query.len()
            ));
        }

        let index = self.index.read();
        let reverse_map = self.reverse_map.read();
        let metadata = self.metadata.read();

        let ef_search = (k * 2).max(50);
        let raw_results = index.search(query, k, ef_search);

        let results = raw_results
            .into_iter()
            .map(|(internal_id, dist)| SearchResult {
                id: reverse_map[internal_id as usize].clone(),
                distance: dist,
                metadata: metadata[internal_id as usize].clone(),
            })
            .collect();

        Ok(results)
    }

    /// Get a vector and its metadata by string id.
    pub fn get(&self, id: &str) -> Result<(Vec<f32>, Option<serde_json::Value>), String> {
        let id_map = self.id_map.read();
        let internal_id = id_map
            .get(id)
            .ok_or_else(|| format!("id '{}' not found", id))?;

        let index = self.index.read();
        let vector = index
            .get_vector(*internal_id)
            .ok_or_else(|| "internal error".to_string())?
            .to_vec();

        let metadata = self.metadata.read();
        let meta = metadata[*internal_id as usize].clone();

        Ok((vector, meta))
    }

    /// Number of vectors in the database.
    pub fn len(&self) -> usize {
        self.index.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Persist current state to disk.
    pub fn save(&self) -> Result<(), String> {
        let index = self.index.read();
        let reverse_map = self.reverse_map.read();
        let metadata = self.metadata.read();

        let mut records = Vec::with_capacity(reverse_map.len());
        for (i, string_id) in reverse_map.iter().enumerate() {
            let vector = index
                .get_vector(i as u32)
                .ok_or_else(|| "internal error".to_string())?
                .to_vec();
            records.push(Record {
                string_id: string_id.clone(),
                vector,
                metadata: metadata[i].clone(),
            });
        }

        self.storage.save(&records).map_err(|e| e.to_string())
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open(path.to_str().unwrap(), 3, Metric::Cosine).unwrap();

        db.add("a", vec![1.0, 0.0, 0.0], Some(serde_json::json!({"label": "x-axis"}))).unwrap();
        db.add("b", vec![0.0, 1.0, 0.0], None).unwrap();
        db.add("c", vec![0.0, 0.0, 1.0], None).unwrap();

        let results = db.search(&[0.9, 0.1, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        {
            let db = Database::open(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();
            db.add("x", vec![1.0, 2.0], Some(serde_json::json!({"n": 1}))).unwrap();
            db.add("y", vec![3.0, 4.0], None).unwrap();
            db.save().unwrap();
        }

        {
            let db = Database::open(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();
            assert_eq!(db.len(), 2);
            let (vec, meta) = db.get("x").unwrap();
            assert_eq!(vec, vec![1.0, 2.0]);
            assert!(meta.is_some());
        }
    }

    #[test]
    fn test_duplicate_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        db.add("a", vec![1.0, 0.0], None).unwrap();
        let result = db.add("a", vec![0.0, 1.0], None);
        assert!(result.is_err());
    }
}
