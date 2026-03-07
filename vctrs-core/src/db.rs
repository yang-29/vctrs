/// The main database: ties together HNSW index, storage, and id mapping.

use crate::distance::Metric;
use crate::hnsw::HnswIndex;
use crate::quantize::ScalarQuantizer;
use crate::storage::{MetaRecord, Storage};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::PathBuf;

/// Database configuration options.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Number of bi-directional links per node. Higher = better recall, more memory. Default: 16.
    pub m: usize,
    /// Size of the dynamic candidate list during construction. Higher = better recall, slower build. Default: 200.
    pub ef_construction: usize,
    /// Enable scalar quantization (SQ8) for ~4x smaller disk storage.
    /// Vectors are stored as u8 on disk and dequantized to f32 on load.
    pub quantize: bool,
}

impl Default for HnswConfig {
    fn default() -> Self {
        HnswConfig {
            m: 16,
            ef_construction: 200,
            quantize: false,
        }
    }
}

pub struct Database {
    index: RwLock<HnswIndex>,
    id_map: RwLock<HashMap<String, u32>>,
    reverse_map: RwLock<Vec<String>>,
    metadata: RwLock<Vec<Option<serde_json::Value>>>,
    storage: Storage,
    dim: usize,
    quantizer: RwLock<Option<ScalarQuantizer>>,
    quantize_on_save: bool,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub metadata: Option<serde_json::Value>,
}

/// Filter predicate for metadata-filtered search.
#[derive(Debug, Clone)]
pub enum Filter {
    /// Field equals value.
    Eq(String, serde_json::Value),
    /// Field not equals value.
    Ne(String, serde_json::Value),
    /// Field is in list of values.
    In(String, Vec<serde_json::Value>),
    /// All sub-filters must match.
    And(Vec<Filter>),
    /// Any sub-filter must match.
    Or(Vec<Filter>),
}

impl Filter {
    pub fn matches(&self, metadata: &Option<serde_json::Value>) -> bool {
        let obj = match metadata {
            Some(serde_json::Value::Object(m)) => m,
            _ => return false,
        };

        match self {
            Filter::Eq(key, val) => obj.get(key).map_or(false, |v| v == val),
            Filter::Ne(key, val) => obj.get(key).map_or(true, |v| v != val),
            Filter::In(key, vals) => obj.get(key).map_or(false, |v| vals.contains(v)),
            Filter::And(filters) => filters.iter().all(|f| f.matches(metadata)),
            Filter::Or(filters) => filters.iter().any(|f| f.matches(metadata)),
        }
    }
}

impl Database {
    /// Open an existing database. Reads dim and metric from the saved file.
    /// Returns Err if the database doesn't exist — use `open_or_create` instead.
    pub fn open(path: &str) -> Result<Self, String> {
        let db_path = PathBuf::from(path);
        let storage = Storage::new(&db_path, 0);

        if storage.exists() {
            let has_quantized = storage.has_quantized();
            let (index, meta_records) = storage.load().map_err(|e| e.to_string())?;
            let dim = index.dim();

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
                quantizer: RwLock::new(None),
                quantize_on_save: has_quantized,
            });
        }

        Err(format!("database not found at '{}'", path))
    }

    /// Open an existing database or create a new one.
    /// dim and metric are only used when creating — ignored when opening an existing db.
    pub fn open_or_create(path: &str, dim: usize, metric: Metric) -> Result<Self, String> {
        Self::open_or_create_with_config(path, dim, metric, HnswConfig::default())
    }

    /// Open an existing database or create a new one with custom HNSW parameters.
    pub fn open_or_create_with_config(
        path: &str,
        dim: usize,
        metric: Metric,
        config: HnswConfig,
    ) -> Result<Self, String> {
        let db_path = PathBuf::from(path);
        std::fs::create_dir_all(&db_path).map_err(|e| e.to_string())?;

        let storage = Storage::new(&db_path, dim);

        if storage.exists() {
            let has_quantized = storage.has_quantized();
            let (index, meta_records) = storage.load().map_err(|e| e.to_string())?;
            let loaded_dim = index.dim();

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
                dim: loaded_dim,
                quantizer: RwLock::new(None),
                quantize_on_save: config.quantize || has_quantized,
            });
        }

        Ok(Database {
            index: RwLock::new(HnswIndex::new(dim, metric, config.m, config.ef_construction)),
            id_map: RwLock::new(HashMap::new()),
            reverse_map: RwLock::new(Vec::new()),
            metadata: RwLock::new(Vec::new()),
            storage,
            dim,
            quantizer: RwLock::new(None),
            quantize_on_save: config.quantize,
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
            return Err(format!("id '{}' already exists, use upsert instead", id));
        }

        let mut index = self.index.write();
        let internal_id = index.insert(vector);

        id_map.insert(id.to_string(), internal_id);
        self.reverse_map.write().push(id.to_string());
        self.metadata.write().push(metadata);

        Ok(())
    }

    /// Add or update a vector. If the id exists, updates it. If not, inserts it.
    pub fn upsert(
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

        let id_map = self.id_map.read();
        if let Some(&internal_id) = id_map.get(id) {
            drop(id_map);
            self.index.write().update_vector(internal_id, vector);
            self.metadata.write()[internal_id as usize] = metadata;
            return Ok(());
        }
        drop(id_map);

        // Not found — insert.
        let mut id_map = self.id_map.write();
        // Double-check after acquiring write lock.
        if let Some(&internal_id) = id_map.get(id) {
            self.index.write().update_vector(internal_id, vector);
            self.metadata.write()[internal_id as usize] = metadata;
            return Ok(());
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

    /// Search for the k nearest neighbors, optionally filtered by metadata.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
        filter: Option<&Filter>,
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

        let ef = ef_search.unwrap_or_else(|| {
            let base = 200usize;
            let extra = ((k as f64).sqrt() * 10.0) as usize;
            base.max(k + extra)
        });

        if let Some(f) = filter {
            let total = index.len();

            if index.uses_brute_force() {
                // Brute-force: scan everything once, filter inline.
                let raw_results = index.search(query, total, ef);
                let mut results: Vec<SearchResult> = raw_results
                    .into_iter()
                    .filter_map(|(internal_id, dist)| {
                        let meta = &metadata[internal_id as usize];
                        if !f.matches(meta) {
                            return None;
                        }
                        Some(SearchResult {
                            id: reverse_map[internal_id as usize].clone(),
                            distance: dist,
                            metadata: meta.clone(),
                        })
                    })
                    .collect();
                results.truncate(k);
                return Ok(results);
            }

            // HNSW: adaptive over-fetch, escalate if too few results match.
            let mut multiplier = 4usize;
            loop {
                let fetch_k = k * multiplier;
                let search_ef = ef.max(fetch_k);
                let raw_results = index.search(query, fetch_k, search_ef);

                let mut results: Vec<SearchResult> = raw_results
                    .into_iter()
                    .filter_map(|(internal_id, dist)| {
                        let meta = &metadata[internal_id as usize];
                        if !f.matches(meta) {
                            return None;
                        }
                        Some(SearchResult {
                            id: reverse_map[internal_id as usize].clone(),
                            distance: dist,
                            metadata: meta.clone(),
                        })
                    })
                    .collect();

                results.truncate(k);

                if results.len() >= k || multiplier >= 64 || fetch_k >= total {
                    return Ok(results);
                }
                multiplier *= 4;
            }
        } else {
            let raw_results = index.search(query, k, ef);

            let results: Vec<SearchResult> = raw_results
                .into_iter()
                .map(|(internal_id, dist)| SearchResult {
                    id: reverse_map[internal_id as usize].clone(),
                    distance: dist,
                    metadata: metadata[internal_id as usize].clone(),
                })
                .collect();

            Ok(results)
        }
    }

    /// Search multiple queries in parallel. Returns one result Vec per query.
    pub fn search_many(
        &self,
        queries: &[&[f32]],
        k: usize,
        ef_search: Option<usize>,
        filter: Option<&Filter>,
    ) -> Result<Vec<Vec<SearchResult>>, String> {
        for q in queries {
            if q.len() != self.dim {
                return Err(format!(
                    "dimension mismatch: expected {}, got {}",
                    self.dim,
                    q.len()
                ));
            }
        }

        let index = self.index.read();
        let reverse_map = self.reverse_map.read();
        let metadata = self.metadata.read();

        let ef = ef_search.unwrap_or_else(|| {
            let base = 200usize;
            let extra = ((k as f64).sqrt() * 10.0) as usize;
            base.max(k + extra)
        });

        if filter.is_some() {
            // For filtered search, fall back to per-query search.
            drop(index);
            drop(reverse_map);
            drop(metadata);
            return queries
                .iter()
                .map(|q| self.search(q, k, ef_search, filter))
                .collect();
        }

        let raw_results = index.search_many(queries, k, ef);

        Ok(raw_results
            .into_iter()
            .map(|results| {
                results
                    .into_iter()
                    .map(|(internal_id, dist)| SearchResult {
                        id: reverse_map[internal_id as usize].clone(),
                        distance: dist,
                        metadata: metadata[internal_id as usize].clone(),
                    })
                    .collect()
            })
            .collect())
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

    /// Get all ids in the database.
    pub fn ids(&self) -> Vec<String> {
        self.id_map.read().keys().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.index.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> Metric {
        self.index.read().metric()
    }

    /// Persist to disk.
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
            .save(&index, &meta_records, self.quantize_on_save)
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 3, Metric::Cosine).unwrap();

        db.add("a", vec![1.0, 0.0, 0.0], Some(serde_json::json!({"label": "x-axis"}))).unwrap();
        db.add("b", vec![0.0, 1.0, 0.0], None).unwrap();
        db.add("c", vec![0.0, 0.0, 1.0], None).unwrap();

        let results = db.search(&[0.9, 0.1, 0.0], 2, None, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_auto_detect_on_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        {
            let db = Database::open_or_create(path.to_str().unwrap(), 3, Metric::Euclidean).unwrap();
            db.add("x", vec![1.0, 2.0, 3.0], None).unwrap();
            db.save().unwrap();
        }

        // Reopen with just the path — should auto-detect dim and metric.
        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            assert_eq!(db.dim(), 3);
            assert_eq!(db.metric(), Metric::Euclidean);
            let (vec, _) = db.get("x").unwrap();
            assert_eq!(vec, vec![1.0, 2.0, 3.0]);
        }
    }

    #[test]
    fn test_upsert() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        // Insert via upsert.
        db.upsert("a", vec![1.0, 0.0], Some(serde_json::json!({"v": 1}))).unwrap();
        assert_eq!(db.len(), 1);
        let (vec, meta) = db.get("a").unwrap();
        assert_eq!(vec, vec![1.0, 0.0]);
        assert_eq!(meta.unwrap()["v"], 1);

        // Update via upsert.
        db.upsert("a", vec![0.0, 1.0], Some(serde_json::json!({"v": 2}))).unwrap();
        assert_eq!(db.len(), 1);
        let (vec, meta) = db.get("a").unwrap();
        assert_eq!(vec, vec![0.0, 1.0]);
        assert_eq!(meta.unwrap()["v"], 2);
    }

    #[test]
    fn test_filtered_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "sci"}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"cat": "art"}))).unwrap();
        db.add("c", vec![0.8, 0.2], Some(serde_json::json!({"cat": "sci"}))).unwrap();

        // Unfiltered: a is closest.
        let results = db.search(&[1.0, 0.0], 1, None, None).unwrap();
        assert_eq!(results[0].id, "a");

        // Filtered to cat=art: b is the only match.
        let filter = Filter::Eq("cat".to_string(), serde_json::json!("art"));
        let results = db.search(&[1.0, 0.0], 1, None, Some(&filter)).unwrap();
        assert_eq!(results[0].id, "b");

        // Filtered to cat=sci: a is closest.
        let filter = Filter::Eq("cat".to_string(), serde_json::json!("sci"));
        let results = db.search(&[1.0, 0.0], 2, None, Some(&filter)).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
        assert_eq!(results[1].id, "c");
    }

    #[test]
    fn test_ids() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        db.add("x", vec![1.0, 0.0], None).unwrap();
        db.add("y", vec![0.0, 1.0], None).unwrap();

        let mut ids = db.ids();
        ids.sort();
        assert_eq!(ids, vec!["x", "y"]);
    }

    #[test]
    fn test_persistence_with_graph() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        {
            let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();
            db.add("x", vec![1.0, 2.0], Some(serde_json::json!({"n": 1}))).unwrap();
            db.add("y", vec![3.0, 4.0], None).unwrap();
            db.save().unwrap();
        }

        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            assert_eq!(db.len(), 2);
            let (vec, meta) = db.get("x").unwrap();
            assert_eq!(vec, vec![1.0, 2.0]);
            assert!(meta.is_some());

            let results = db.search(&[1.0, 2.0], 1, None, None).unwrap();
            assert_eq!(results[0].id, "x");
        }
    }

    #[test]
    fn test_duplicate_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        db.add("a", vec![1.0, 0.0], None).unwrap();
        let result = db.add("a", vec![0.0, 1.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_delete() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 3, Metric::Cosine).unwrap();

        db.add("a", vec![1.0, 0.0, 0.0], None).unwrap();
        db.add("b", vec![0.0, 1.0, 0.0], None).unwrap();
        db.add("c", vec![0.0, 0.0, 1.0], None).unwrap();

        assert_eq!(db.len(), 3);
        assert!(db.delete("b").unwrap());
        assert_eq!(db.len(), 2);
        assert!(!db.contains("b"));

        let results = db.search(&[0.0, 1.0, 0.0], 3, None, None).unwrap();
        assert!(results.iter().all(|r| r.id != "b"));
    }

    #[test]
    fn test_update() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

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
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        let items = vec![
            ("a".to_string(), vec![1.0, 0.0], None),
            ("b".to_string(), vec![0.0, 1.0], Some(serde_json::json!({"x": 1}))),
            ("c".to_string(), vec![1.0, 1.0], None),
        ];
        db.add_many(items).unwrap();
        assert_eq!(db.len(), 3);

        let results = db.search(&[1.0, 0.0], 1, None, None).unwrap();
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_delete_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        {
            let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();
            db.add("a", vec![1.0, 0.0], None).unwrap();
            db.add("b", vec![0.0, 1.0], None).unwrap();
            db.delete("a").unwrap();
            db.save().unwrap();
        }

        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            assert_eq!(db.len(), 1);
            assert!(!db.contains("a"));
            assert!(db.contains("b"));
        }
    }
}
