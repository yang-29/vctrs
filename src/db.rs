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
    /// Map from internal id -> string id (empty string = deleted).
    reverse_map: RwLock<Vec<String>>,
    /// Metadata for each vector, indexed by internal id.
    metadata: RwLock<Vec<Option<serde_json::Value>>>,
    /// Persistent storage.
    storage: Storage,
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

    /// Add multiple vectors at once. More efficient than calling add() in a loop
    /// because it holds the write lock for the entire batch.
    pub fn add_many(
        &self,
        items: Vec<(String, Vec<f32>, Option<serde_json::Value>)>,
    ) -> Result<(), String> {
        // Validate all dimensions first.
        for (id, vector, _) in &items {
            if vector.len() != self.dim {
                return Err(format!(
                    "dimension mismatch for '{}': expected {}, got {}",
                    id, self.dim, vector.len()
                ));
            }
        }

        let mut id_map = self.id_map.write();
        let mut index = self.index.write();
        let mut reverse_map = self.reverse_map.write();
        let mut metadata = self.metadata.write();

        for (id, vector, meta) in items {
            if id_map.contains_key(&id) {
                return Err(format!("id '{}' already exists", id));
            }
            let internal_id = index.insert(vector);
            id_map.insert(id.clone(), internal_id);
            reverse_map.push(id);
            metadata.push(meta);
        }

        Ok(())
    }

    /// Delete a vector by string id. Returns true if it existed.
    pub fn delete(&self, id: &str) -> Result<bool, String> {
        let mut id_map = self.id_map.write();
        let internal_id = match id_map.remove(id) {
            Some(iid) => iid,
            None => return Ok(false),
        };

        let mut index = self.index.write();
        index.mark_deleted(internal_id);

        // Clear the reverse map entry (keep slot to preserve indices).
        self.reverse_map.write()[internal_id as usize] = String::new();
        self.metadata.write()[internal_id as usize] = None;

        Ok(true)
    }

    /// Update a vector's data and/or metadata. The id must already exist.
    pub fn update(
        &self,
        id: &str,
        vector: Option<Vec<f32>>,
        metadata: Option<Option<serde_json::Value>>,
    ) -> Result<(), String> {
        let id_map = self.id_map.read();
        let internal_id = *id_map
            .get(id)
            .ok_or_else(|| format!("id '{}' not found", id))?;

        if let Some(vec) = vector {
            if vec.len() != self.dim {
                return Err(format!(
                    "dimension mismatch: expected {}, got {}",
                    self.dim,
                    vec.len()
                ));
            }
            self.index.write().update_vector(internal_id, vec);
        }

        if let Some(meta) = metadata {
            self.metadata.write()[internal_id as usize] = meta;
        }

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

    /// Check if an id exists in the database.
    pub fn contains(&self, id: &str) -> bool {
        self.id_map.read().contains_key(id)
    }

    /// Number of vectors in the database.
    pub fn len(&self) -> usize {
        self.index.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Persist current state to disk. Only saves non-deleted vectors.
    pub fn save(&self) -> Result<(), String> {
        let index = self.index.read();
        let id_map = self.id_map.read();
        let metadata = self.metadata.read();

        let mut records = Vec::with_capacity(id_map.len());
        for (string_id, &internal_id) in id_map.iter() {
            let vector = index
                .get_vector(internal_id)
                .ok_or_else(|| "internal error".to_string())?
                .to_vec();
            records.push(Record {
                string_id: string_id.clone(),
                vector,
                metadata: metadata[internal_id as usize].clone(),
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

    #[test]
    fn test_delete() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open(path.to_str().unwrap(), 3, Metric::Cosine).unwrap();

        db.add("a", vec![1.0, 0.0, 0.0], None).unwrap();
        db.add("b", vec![0.0, 1.0, 0.0], None).unwrap();
        db.add("c", vec![0.0, 0.0, 1.0], None).unwrap();

        assert_eq!(db.len(), 3);
        assert!(db.delete("b").unwrap());
        assert_eq!(db.len(), 2);
        assert!(!db.contains("b"));

        // Search should not return deleted vector.
        let results = db.search(&[0.0, 1.0, 0.0], 3).unwrap();
        assert!(results.iter().all(|r| r.id != "b"));
    }

    #[test]
    fn test_update() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"v": 1}))).unwrap();

        // Update vector only.
        db.update("a", Some(vec![0.0, 1.0]), None).unwrap();
        let (vec, meta) = db.get("a").unwrap();
        assert_eq!(vec, vec![0.0, 1.0]);
        assert_eq!(meta.unwrap()["v"], 1); // metadata unchanged

        // Update metadata only.
        db.update("a", None, Some(Some(serde_json::json!({"v": 2})))).unwrap();
        let (_, meta) = db.get("a").unwrap();
        assert_eq!(meta.unwrap()["v"], 2);
    }

    #[test]
    fn test_batch_insert() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        let items = vec![
            ("a".to_string(), vec![1.0, 0.0], None),
            ("b".to_string(), vec![0.0, 1.0], Some(serde_json::json!({"x": 1}))),
            ("c".to_string(), vec![1.0, 1.0], None),
        ];
        db.add_many(items).unwrap();
        assert_eq!(db.len(), 3);

        let results = db.search(&[1.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_delete_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        {
            let db = Database::open(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();
            db.add("a", vec![1.0, 0.0], None).unwrap();
            db.add("b", vec![0.0, 1.0], None).unwrap();
            db.delete("a").unwrap();
            db.save().unwrap();
        }

        {
            let db = Database::open(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();
            assert_eq!(db.len(), 1);
            assert!(!db.contains("a"));
            assert!(db.contains("b"));
        }
    }
}
