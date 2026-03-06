/// The main database: ties together HNSW index, storage, and id mapping.

use crate::distance::Metric;
use crate::hnsw::HnswIndex;
use crate::storage::{MetaRecord, Storage};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::PathBuf;

pub struct Database {
    index: RwLock<HnswIndex>,
    id_map: RwLock<HashMap<String, u32>>,
    reverse_map: RwLock<Vec<String>>,
    metadata: RwLock<Vec<Option<serde_json::Value>>>,
    storage: Storage,
    dim: usize,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub metadata: Option<serde_json::Value>,
}

impl Database {
    pub fn open(path: &str, dim: usize, metric: Metric) -> Result<Self, String> {
        let db_path = PathBuf::from(path);
        std::fs::create_dir_all(&db_path).map_err(|e| e.to_string())?;

        let storage = Storage::new(&db_path, dim);

        if storage.exists() {
            // Fast path: load pre-built graph (no rebuild needed).
            let index = storage.load_graph().map_err(|e| e.to_string())?;
            let meta_records = storage.load_meta().map_err(|e| e.to_string())?;

            let mut id_map = HashMap::with_capacity(meta_records.len());
            let total_slots = index.total_slots();
            let mut reverse_map = vec![String::new(); total_slots];
            let mut metadata = vec![None; total_slots];

            for rec in meta_records {
                id_map.insert(rec.string_id.clone(), rec.internal_id);
                reverse_map[rec.internal_id as usize] = rec.string_id;
                metadata[rec.internal_id as usize] = rec.metadata;
            }

            return Ok(Database {
                index: RwLock::new(index),
                id_map: RwLock::new(id_map),
                reverse_map: RwLock::new(reverse_map),
                metadata: RwLock::new(metadata),
                storage,
                dim,
            });
        }

        // Try legacy v1 format.
        let legacy_path = db_path.join("data.vctrs");
        let legacy_records = storage.load_legacy(&legacy_path).unwrap_or_default();

        let mut index = HnswIndex::new(dim, metric, 16, 200);
        let mut id_map = HashMap::new();
        let mut reverse_map = Vec::new();
        let mut meta = Vec::new();

        if !legacy_records.is_empty() {
            // Batch insert for speed.
            let vecs: Vec<Vec<f32>> = legacy_records.iter().map(|r| r.vector.clone()).collect();
            let ids = index.batch_insert(vecs);
            for (i, record) in legacy_records.iter().enumerate() {
                id_map.insert(record.string_id.clone(), ids[i]);
                reverse_map.push(record.string_id.clone());
                meta.push(record.metadata.clone());
            }
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

    /// Batch insert — uses parallel HNSW construction.
    pub fn add_many(
        &self,
        items: Vec<(String, Vec<f32>, Option<serde_json::Value>)>,
    ) -> Result<(), String> {
        for (id, vector, _) in &items {
            if vector.len() != self.dim {
                return Err(format!(
                    "dimension mismatch for '{}': expected {}, got {}",
                    id, self.dim, vector.len()
                ));
            }
        }

        let mut id_map = self.id_map.write();
        for (id, _, _) in &items {
            if id_map.contains_key(id) {
                return Err(format!("id '{}' already exists", id));
            }
        }

        let (ids_vec, vecs, metas): (Vec<_>, Vec<_>, Vec<_>) = items
            .into_iter()
            .map(|(id, vec, meta)| (id, vec, meta))
            .fold(
                (Vec::new(), Vec::new(), Vec::new()),
                |(mut ids, mut vecs, mut metas), (id, vec, meta)| {
                    ids.push(id);
                    vecs.push(vec);
                    metas.push(meta);
                    (ids, vecs, metas)
                },
            );

        let mut index = self.index.write();
        let internal_ids = index.batch_insert(vecs);

        let mut reverse_map = self.reverse_map.write();
        let mut metadata = self.metadata.write();

        for (i, id) in ids_vec.into_iter().enumerate() {
            id_map.insert(id.clone(), internal_ids[i]);
            // Ensure reverse_map is big enough.
            while reverse_map.len() <= internal_ids[i] as usize {
                reverse_map.push(String::new());
            }
            reverse_map[internal_ids[i] as usize] = id;
            while metadata.len() <= internal_ids[i] as usize {
                metadata.push(None);
            }
            metadata[internal_ids[i] as usize] = metas[i].clone();
        }

        Ok(())
    }

    pub fn delete(&self, id: &str) -> Result<bool, String> {
        let mut id_map = self.id_map.write();
        let internal_id = match id_map.remove(id) {
            Some(iid) => iid,
            None => return Ok(false),
        };

        let mut index = self.index.write();
        index.mark_deleted(internal_id);

        self.reverse_map.write()[internal_id as usize] = String::new();
        self.metadata.write()[internal_id as usize] = None;

        Ok(true)
    }

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
    /// ef_search controls the search quality/speed tradeoff. Higher = better recall, slower.
    /// If None, uses an adaptive formula based on k.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<SearchResult>, String> {
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

        // Adaptive ef: generous at low k for high recall, sublinear at high k for speed.
        // k=1: ef=200, k=10: ef=200, k=100: ef=200, k=500: ef=724
        let ef = ef_search.unwrap_or_else(|| {
            let base = 200usize;
            let extra = ((k as f64).sqrt() * 10.0) as usize;
            base.max(k + extra)
        });
        let raw_results = index.search(query, k, ef);

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

    pub fn contains(&self, id: &str) -> bool {
        self.id_map.read().contains_key(id)
    }

    pub fn len(&self) -> usize {
        self.index.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Persist to disk — saves the full HNSW graph so load is instant.
    pub fn save(&self) -> Result<(), String> {
        let index = self.index.read();
        let id_map = self.id_map.read();
        let metadata = self.metadata.read();

        let meta_records: Vec<MetaRecord> = id_map
            .iter()
            .map(|(string_id, &internal_id)| MetaRecord {
                internal_id,
                string_id: string_id.clone(),
                metadata: metadata[internal_id as usize].clone(),
            })
            .collect();

        self.storage
            .save_full(&index, &meta_records)
            .map_err(|e| e.to_string())
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

        let results = db.search(&[0.9, 0.1, 0.0], 2, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_persistence_with_graph() {
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

            // Search should work on loaded graph.
            let results = db.search(&[1.0, 2.0], 1, None).unwrap();
            assert_eq!(results[0].id, "x");
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

        let results = db.search(&[0.0, 1.0, 0.0], 3, None).unwrap();
        assert!(results.iter().all(|r| r.id != "b"));
    }

    #[test]
    fn test_update() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"v": 1}))).unwrap();

        db.update("a", Some(vec![0.0, 1.0]), None).unwrap();
        let (vec, meta) = db.get("a").unwrap();
        assert_eq!(vec, vec![0.0, 1.0]);
        assert_eq!(meta.unwrap()["v"], 1);

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

        let results = db.search(&[1.0, 0.0], 1, None).unwrap();
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
