/// The main database: ties together HNSW index, storage, and id mapping.

use crate::distance::Metric;
use crate::error::{VctrsError, Result};
use crate::hnsw::{GraphStats, HnswIndex};
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
    /// Field greater than value (numeric).
    Gt(String, f64),
    /// Field greater than or equal to value (numeric).
    Gte(String, f64),
    /// Field less than value (numeric).
    Lt(String, f64),
    /// Field less than or equal to value (numeric).
    Lte(String, f64),
    /// All sub-filters must match.
    And(Vec<Filter>),
    /// Any sub-filter must match.
    Or(Vec<Filter>),
}

/// Extract a JSON value as f64 for numeric comparison.
fn as_f64(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        _ => None,
    }
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
            Filter::Gt(key, threshold) => {
                obj.get(key).and_then(as_f64).map_or(false, |v| v > *threshold)
            }
            Filter::Gte(key, threshold) => {
                obj.get(key).and_then(as_f64).map_or(false, |v| v >= *threshold)
            }
            Filter::Lt(key, threshold) => {
                obj.get(key).and_then(as_f64).map_or(false, |v| v < *threshold)
            }
            Filter::Lte(key, threshold) => {
                obj.get(key).and_then(as_f64).map_or(false, |v| v <= *threshold)
            }
            Filter::And(filters) => filters.iter().all(|f| f.matches(metadata)),
            Filter::Or(filters) => filters.iter().any(|f| f.matches(metadata)),
        }
    }
}

impl Database {
    /// Open an existing database. Reads dim and metric from the saved file.
    /// Returns Err if the database doesn't exist — use `open_or_create` instead.
    pub fn open(path: &str) -> Result<Self> {
        let db_path = PathBuf::from(path);
        let storage = Storage::new(&db_path, 0);

        if storage.exists() {
            let has_quantized = storage.has_quantized();
            let (index, meta_records) = storage.load()?;
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

        Err(VctrsError::DatabaseNotFound(path.to_string()))
    }

    /// Open an existing database or create a new one.
    /// dim and metric are only used when creating — ignored when opening an existing db.
    pub fn open_or_create(path: &str, dim: usize, metric: Metric) -> Result<Self> {
        Self::open_or_create_with_config(path, dim, metric, HnswConfig::default())
    }

    /// Open an existing database or create a new one with custom HNSW parameters.
    pub fn open_or_create_with_config(
        path: &str,
        dim: usize,
        metric: Metric,
        config: HnswConfig,
    ) -> Result<Self> {
        let db_path = PathBuf::from(path);
        std::fs::create_dir_all(&db_path)?;

        let storage = Storage::new(&db_path, dim);

        if storage.exists() {
            let has_quantized = storage.has_quantized();
            let (index, meta_records) = storage.load()?;
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
    ) -> Result<()> {
        if vector.len() != self.dim {
            return Err(VctrsError::DimensionMismatch { expected: self.dim, got: vector.len() });
        }

        let mut id_map = self.id_map.write();
        if id_map.contains_key(id) {
            return Err(VctrsError::DuplicateId(id.to_string()));
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
    ) -> Result<()> {
        if vector.len() != self.dim {
            return Err(VctrsError::DimensionMismatch { expected: self.dim, got: vector.len() });
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
    ) -> Result<()> {
        for (id, vector, _) in &items {
            if vector.len() != self.dim {
                return Err(VctrsError::DimensionMismatch { expected: self.dim, got: vector.len() });
            }
        }

        let mut id_map = self.id_map.write();
        for (id, _, _) in &items {
            if id_map.contains_key(id) {
                return Err(VctrsError::DuplicateId(id.to_string()));
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

    pub fn delete(&self, id: &str) -> Result<bool> {
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
    ) -> Result<()> {
        let id_map = self.id_map.read();
        let internal_id = *id_map
            .get(id)
            .ok_or_else(|| VctrsError::NotFound(id.to_string()))?;

        if let Some(vec) = vector {
            if vec.len() != self.dim {
                return Err(VctrsError::DimensionMismatch { expected: self.dim, got: vec.len() });
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
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            return Err(VctrsError::DimensionMismatch { expected: self.dim, got: query.len() });
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
            // In-graph filtering: the predicate is checked during HNSW traversal.
            // Non-matching vectors still participate in graph navigation but are
            // excluded from results. This is much more efficient than over-fetching.
            let raw_results = index.search_filtered(query, k, ef, |id| {
                f.matches(&metadata[id as usize])
            });

            let results: Vec<SearchResult> = raw_results
                .into_iter()
                .map(|(internal_id, dist)| SearchResult {
                    id: reverse_map[internal_id as usize].clone(),
                    distance: dist,
                    metadata: metadata[internal_id as usize].clone(),
                })
                .collect();

            return Ok(results);
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
    ) -> Result<Vec<Vec<SearchResult>>> {
        for q in queries {
            if q.len() != self.dim {
                return Err(VctrsError::DimensionMismatch { expected: self.dim, got: q.len() });
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

    pub fn get(&self, id: &str) -> Result<(Vec<f32>, Option<serde_json::Value>)> {
        let id_map = self.id_map.read();
        let internal_id = id_map
            .get(id)
            .ok_or_else(|| VctrsError::NotFound(id.to_string()))?;

        let index = self.index.read();
        let vector = index
            .get_vector(*internal_id)
            .ok_or_else(|| VctrsError::CorruptData("vector slot missing".to_string()))?
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

    /// Enable quantized search: SQ8 quantized vectors for faster HNSW traversal
    /// with full-precision re-ranking. Uses ~25% of the memory for vector data.
    pub fn enable_quantized_search(&self) {
        self.index.write().enable_quantized_search();
    }

    /// Disable quantized search (frees quantized memory).
    pub fn disable_quantized_search(&self) {
        self.index.write().disable_quantized_search();
    }

    /// Whether quantized search is currently enabled.
    pub fn has_quantized_search(&self) -> bool {
        self.index.read().has_quantized_search()
    }

    /// Number of deleted slots that haven't been reclaimed.
    pub fn deleted_count(&self) -> usize {
        self.index.read().deleted_ids().len()
    }

    /// Total allocated slots (active + deleted).
    pub fn total_slots(&self) -> usize {
        self.index.read().total_slots()
    }

    /// Get graph-level statistics for diagnostics and monitoring.
    pub fn stats(&self) -> GraphStats {
        self.index.read().graph_stats()
    }

    /// Rebuild the index with only live vectors, reclaiming deleted slots.
    /// This is O(n log n) — it re-inserts all live vectors into a fresh HNSW graph.
    pub fn compact(&self) -> Result<()> {
        let old_index = self.index.read();
        if old_index.deleted_ids().is_empty() {
            return Ok(()); // Nothing to compact.
        }

        let (new_index, old_to_new) = old_index.compact();
        drop(old_index);

        // Remap id_map, reverse_map, and metadata.
        let mut id_map = self.id_map.write();
        let mut reverse_map = self.reverse_map.write();
        let old_metadata = self.metadata.read();

        let new_len = old_to_new.len();
        let mut new_reverse_map = vec![String::new(); new_len];
        let mut new_metadata: Vec<Option<serde_json::Value>> = vec![None; new_len];
        let mut new_id_map = HashMap::with_capacity(new_len);

        for (string_id, &old_internal) in id_map.iter() {
            if let Some(&new_internal) = old_to_new.get(&old_internal) {
                new_id_map.insert(string_id.clone(), new_internal);
                new_reverse_map[new_internal as usize] = string_id.clone();
                new_metadata[new_internal as usize] = old_metadata[old_internal as usize].clone();
            }
        }

        drop(old_metadata);

        *id_map = new_id_map;
        *reverse_map = new_reverse_map;
        *self.metadata.write() = new_metadata;
        *self.index.write() = new_index;

        Ok(())
    }

    /// Persist to disk.
    pub fn save(&self) -> Result<()> {
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
            .save(&index, &meta_records, self.quantize_on_save)?;
        Ok(())
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
    fn test_compact_reclaims_slots() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 3, Metric::Cosine).unwrap();

        db.add("a", vec![1.0, 0.0, 0.0], Some(serde_json::json!({"v": 1}))).unwrap();
        db.add("b", vec![0.0, 1.0, 0.0], Some(serde_json::json!({"v": 2}))).unwrap();
        db.add("c", vec![0.0, 0.0, 1.0], Some(serde_json::json!({"v": 3}))).unwrap();
        db.add("d", vec![0.5, 0.5, 0.0], None).unwrap();

        // Delete two vectors.
        db.delete("b").unwrap();
        db.delete("d").unwrap();
        assert_eq!(db.len(), 2);
        assert_eq!(db.total_slots(), 4);
        assert_eq!(db.deleted_count(), 2);

        // Compact.
        db.compact().unwrap();
        assert_eq!(db.len(), 2);
        assert_eq!(db.total_slots(), 2); // Slots reclaimed.
        assert_eq!(db.deleted_count(), 0);

        // Search still works.
        let results = db.search(&[1.0, 0.0, 0.0], 1, None, None).unwrap();
        assert_eq!(results[0].id, "a");

        // Metadata preserved.
        let (_, meta) = db.get("a").unwrap();
        assert_eq!(meta.unwrap()["v"], 1);
        let (_, meta) = db.get("c").unwrap();
        assert_eq!(meta.unwrap()["v"], 3);

        // Deleted ids are gone.
        assert!(!db.contains("b"));
        assert!(!db.contains("d"));
    }

    #[test]
    fn test_compact_then_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        {
            let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();
            db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"k": "a"}))).unwrap();
            db.add("b", vec![0.0, 1.0], None).unwrap();
            db.add("c", vec![1.0, 1.0], Some(serde_json::json!({"k": "c"}))).unwrap();
            db.delete("b").unwrap();
            db.compact().unwrap();
            db.save().unwrap();
        }

        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            assert_eq!(db.len(), 2);
            assert_eq!(db.total_slots(), 2);
            assert!(db.contains("a"));
            assert!(!db.contains("b"));
            assert!(db.contains("c"));

            let results = db.search(&[1.0, 0.0], 1, None, None).unwrap();
            assert_eq!(results[0].id, "a");

            let (_, meta) = db.get("c").unwrap();
            assert_eq!(meta.unwrap()["k"], "c");
        }
    }

    #[test]
    fn test_compact_noop_when_no_deletes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        db.add("a", vec![1.0, 0.0], None).unwrap();
        db.add("b", vec![0.0, 1.0], None).unwrap();

        db.compact().unwrap(); // Should be a no-op.
        assert_eq!(db.len(), 2);
        assert_eq!(db.total_slots(), 2);
    }

    #[test]
    fn test_compact_all_deleted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        db.add("a", vec![1.0, 0.0], None).unwrap();
        db.add("b", vec![0.0, 1.0], None).unwrap();
        db.delete("a").unwrap();
        db.delete("b").unwrap();

        db.compact().unwrap();
        assert_eq!(db.len(), 0);
        assert_eq!(db.total_slots(), 0);
        assert_eq!(db.deleted_count(), 0);
    }

    #[test]
    fn test_compact_then_insert() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], None).unwrap();
        db.add("b", vec![0.0, 1.0], None).unwrap();
        db.delete("a").unwrap();
        db.compact().unwrap();

        // Insert after compact.
        db.add("c", vec![0.5, 0.5], Some(serde_json::json!({"new": true}))).unwrap();
        assert_eq!(db.len(), 2);
        assert_eq!(db.total_slots(), 2);

        let results = db.search(&[0.5, 0.5], 1, None, None).unwrap();
        assert_eq!(results[0].id, "c");
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

    // -- Edge case tests for Database layer -----------------------------------

    #[test]
    fn test_add_wrong_dimension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 3, Metric::Cosine).unwrap();

        let result = db.add("a", vec![1.0, 0.0], None);
        assert!(matches!(result.unwrap_err(), VctrsError::DimensionMismatch { expected: 3, got: 2 }));
    }

    #[test]
    fn test_search_wrong_dimension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 3, Metric::Cosine).unwrap();
        db.add("a", vec![1.0, 0.0, 0.0], None).unwrap();

        let result = db.search(&[1.0, 0.0], 1, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        let result = db.get("missing");
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        assert!(!db.delete("missing").unwrap());
    }

    #[test]
    fn test_update_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        let result = db.update("missing", Some(vec![1.0, 0.0]), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_update_wrong_dimension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();
        db.add("a", vec![1.0, 0.0], None).unwrap();

        let result = db.update("a", Some(vec![1.0, 0.0, 0.0]), None);
        assert!(matches!(result.unwrap_err(), VctrsError::DimensionMismatch { expected: 2, got: 3 }));
    }

    #[test]
    fn test_upsert_wrong_dimension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        let result = db.upsert("a", vec![1.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_many_duplicate_in_batch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        // Existing id.
        db.add("a", vec![1.0, 0.0], None).unwrap();
        let items = vec![
            ("b".to_string(), vec![0.0, 1.0], None),
            ("a".to_string(), vec![0.5, 0.5], None), // duplicate
        ];
        let result = db.add_many(items);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_many_wrong_dimension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        let items = vec![
            ("a".to_string(), vec![1.0, 0.0], None),
            ("b".to_string(), vec![1.0], None), // wrong dim
        ];
        let result = db.add_many(items);
        assert!(result.is_err());
    }

    #[test]
    fn test_open_nonexistent() {
        let result = Database::open("/tmp/vctrs_definitely_not_here_12345");
        assert!(result.is_err());
    }

    #[test]
    fn test_search_empty_db() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        let results = db.search(&[1.0, 0.0], 10, None, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_filtered_search_no_matches() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "x"}))).unwrap();
        db.add("b", vec![0.0, 1.0], Some(serde_json::json!({"cat": "y"}))).unwrap();

        let filter = Filter::Eq("cat".to_string(), serde_json::json!("z"));
        let results = db.search(&[1.0, 0.0], 10, None, Some(&filter)).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_filtered_search_with_ne() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "x"}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"cat": "y"}))).unwrap();
        db.add("c", vec![0.0, 1.0], Some(serde_json::json!({"cat": "x"}))).unwrap();

        let filter = Filter::Ne("cat".to_string(), serde_json::json!("x"));
        let results = db.search(&[1.0, 0.0], 10, None, Some(&filter)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }

    #[test]
    fn test_filtered_search_with_in() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "x"}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"cat": "y"}))).unwrap();
        db.add("c", vec![0.0, 1.0], Some(serde_json::json!({"cat": "z"}))).unwrap();

        let filter = Filter::In("cat".to_string(), vec![serde_json::json!("x"), serde_json::json!("z")]);
        let results = db.search(&[1.0, 0.0], 10, None, Some(&filter)).unwrap();
        assert_eq!(results.len(), 2);
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"c"));
    }

    #[test]
    fn test_filtered_search_and_combinator() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "x", "val": 1}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"cat": "x", "val": 2}))).unwrap();
        db.add("c", vec![0.0, 1.0], Some(serde_json::json!({"cat": "y", "val": 1}))).unwrap();

        let filter = Filter::And(vec![
            Filter::Eq("cat".to_string(), serde_json::json!("x")),
            Filter::Eq("val".to_string(), serde_json::json!(2)),
        ]);
        let results = db.search(&[1.0, 0.0], 10, None, Some(&filter)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }

    #[test]
    fn test_filtered_search_or_combinator() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "x"}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"cat": "y"}))).unwrap();
        db.add("c", vec![0.0, 1.0], Some(serde_json::json!({"cat": "z"}))).unwrap();

        let filter = Filter::Or(vec![
            Filter::Eq("cat".to_string(), serde_json::json!("x")),
            Filter::Eq("cat".to_string(), serde_json::json!("z")),
        ]);
        let results = db.search(&[1.0, 0.0], 10, None, Some(&filter)).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_filter_no_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        // Vectors with no metadata should not match any Eq filter.
        db.add("a", vec![1.0, 0.0], None).unwrap();
        db.add("b", vec![0.0, 1.0], Some(serde_json::json!({"cat": "x"}))).unwrap();

        let filter = Filter::Eq("cat".to_string(), serde_json::json!("x"));
        let results = db.search(&[1.0, 0.0], 10, None, Some(&filter)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }

    #[test]
    fn test_compact_with_metadata_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "x"}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"cat": "y"}))).unwrap();
        db.add("c", vec![0.0, 1.0], Some(serde_json::json!({"cat": "x"}))).unwrap();
        db.delete("b").unwrap();
        db.compact().unwrap();

        // Filtered search should work after compact.
        let filter = Filter::Eq("cat".to_string(), serde_json::json!("x"));
        let results = db.search(&[1.0, 0.0], 10, None, Some(&filter)).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_upsert_then_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.upsert("a", vec![1.0, 0.0], None).unwrap();
        db.upsert("b", vec![0.0, 1.0], None).unwrap();

        // Move "a" to be near "b".
        db.upsert("a", vec![0.0, 1.0], None).unwrap();

        let results = db.search(&[0.0, 1.0], 2, None, None).unwrap();
        // Both should be found at [0,1].
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
    }

    #[test]
    fn test_search_many_filtered() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "x"}))).unwrap();
        db.add("b", vec![0.0, 1.0], Some(serde_json::json!({"cat": "y"}))).unwrap();
        db.add("c", vec![0.5, 0.5], Some(serde_json::json!({"cat": "x"}))).unwrap();

        let filter = Filter::Eq("cat".to_string(), serde_json::json!("x"));
        let results = db.search_many(
            &[&[1.0, 0.0], &[0.0, 1.0]],
            10, None, Some(&filter),
        ).unwrap();

        assert_eq!(results.len(), 2);
        // Both query results should only contain cat=x items.
        for batch in &results {
            for r in batch {
                assert!(r.id == "a" || r.id == "c");
            }
        }
    }

    #[test]
    fn test_save_load_with_quantized_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        {
            let db = Database::open_or_create_with_config(
                path.to_str().unwrap(), 2, Metric::Euclidean,
                HnswConfig { m: 16, ef_construction: 200, quantize: true },
            ).unwrap();
            db.add("a", vec![1.0, 0.0], None).unwrap();
            db.add("b", vec![0.0, 1.0], None).unwrap();
            db.save().unwrap();
        }

        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            assert!(db.has_quantized_search());
            let results = db.search(&[1.0, 0.0], 1, None, None).unwrap();
            assert_eq!(results[0].id, "a");
        }
    }

    #[test]
    fn test_compact_then_save_load_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        {
            let db = Database::open_or_create(path.to_str().unwrap(), 3, Metric::Cosine).unwrap();
            db.add("a", vec![1.0, 0.0, 0.0], Some(serde_json::json!({"k": 1}))).unwrap();
            db.add("b", vec![0.0, 1.0, 0.0], None).unwrap();
            db.add("c", vec![0.0, 0.0, 1.0], Some(serde_json::json!({"k": 3}))).unwrap();
            db.add("d", vec![0.5, 0.5, 0.0], None).unwrap();
            db.delete("b").unwrap();
            db.delete("d").unwrap();
            db.compact().unwrap();
            db.save().unwrap();
        }

        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            assert_eq!(db.len(), 2);
            assert_eq!(db.total_slots(), 2);
            assert!(db.contains("a"));
            assert!(db.contains("c"));
            assert!(!db.contains("b"));
            assert!(!db.contains("d"));

            let (_, meta) = db.get("a").unwrap();
            assert_eq!(meta.unwrap()["k"], 1);

            let results = db.search(&[1.0, 0.0, 0.0], 1, None, None).unwrap();
            assert_eq!(results[0].id, "a");

            // Can insert new items after load.
            db.add("e", vec![0.0, 0.0, 1.0], None).unwrap();
            assert_eq!(db.len(), 3);
        }
    }

    #[test]
    fn test_ids_after_operations() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        db.add("x", vec![1.0, 0.0], None).unwrap();
        db.add("y", vec![0.0, 1.0], None).unwrap();
        db.add("z", vec![0.5, 0.5], None).unwrap();

        db.delete("y").unwrap();
        let mut ids = db.ids();
        ids.sort();
        assert_eq!(ids, vec!["x", "z"]);

        db.compact().unwrap();
        let mut ids = db.ids();
        ids.sort();
        assert_eq!(ids, vec!["x", "z"]);
    }

    #[test]
    fn test_len_consistency() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        assert_eq!(db.len(), 0);
        assert!(db.is_empty());

        db.add("a", vec![1.0, 0.0], None).unwrap();
        assert_eq!(db.len(), 1);

        db.add("b", vec![0.0, 1.0], None).unwrap();
        assert_eq!(db.len(), 2);

        db.delete("a").unwrap();
        assert_eq!(db.len(), 1);
        assert_eq!(db.deleted_count(), 1);
        assert_eq!(db.total_slots(), 2);

        db.compact().unwrap();
        assert_eq!(db.len(), 1);
        assert_eq!(db.deleted_count(), 0);
        assert_eq!(db.total_slots(), 1);
    }

    #[test]
    fn test_stats() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 3, Metric::Cosine).unwrap();

        // Empty database stats.
        let s = db.stats();
        assert_eq!(s.num_vectors, 0);
        assert_eq!(s.num_deleted, 0);
        assert!(s.uses_brute_force);
        assert!(!s.uses_quantized_search);

        // Add some vectors.
        db.add("a", vec![1.0, 0.0, 0.0], None).unwrap();
        db.add("b", vec![0.0, 1.0, 0.0], None).unwrap();
        db.add("c", vec![0.0, 0.0, 1.0], None).unwrap();

        let s = db.stats();
        assert_eq!(s.num_vectors, 3);
        assert_eq!(s.num_deleted, 0);
        assert!(s.memory_vectors_bytes > 0);
        assert!(s.num_layers >= 1);

        // Delete one.
        db.delete("b").unwrap();
        let s = db.stats();
        assert_eq!(s.num_vectors, 2);
        assert_eq!(s.num_deleted, 1);

        // Enable quantized search.
        db.enable_quantized_search();
        let s = db.stats();
        assert!(s.uses_quantized_search);
        assert!(s.memory_quantized_bytes > 0);
    }

    #[test]
    fn test_filter_gt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"score": 10}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"score": 20}))).unwrap();
        db.add("c", vec![0.0, 1.0], Some(serde_json::json!({"score": 30}))).unwrap();

        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Gt("score".into(), 15.0))).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.id != "a"));
    }

    #[test]
    fn test_filter_gte() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"score": 10}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"score": 20}))).unwrap();

        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Gte("score".into(), 20.0))).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }

    #[test]
    fn test_filter_lt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"score": 10}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"score": 20}))).unwrap();
        db.add("c", vec![0.0, 1.0], Some(serde_json::json!({"score": 30}))).unwrap();

        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Lt("score".into(), 25.0))).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.id != "c"));
    }

    #[test]
    fn test_filter_lte() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"score": 10}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"score": 20}))).unwrap();

        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Lte("score".into(), 10.0))).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_filter_range_combined() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        for i in 0..10 {
            db.add(
                &format!("v{}", i),
                vec![i as f32, 0.0],
                Some(serde_json::json!({"val": i})),
            ).unwrap();
        }

        // 3 <= val < 7
        let filter = Filter::And(vec![
            Filter::Gte("val".into(), 3.0),
            Filter::Lt("val".into(), 7.0),
        ]);
        let results = db.search(&[5.0, 0.0], 10, None, Some(&filter)).unwrap();
        assert_eq!(results.len(), 4); // 3, 4, 5, 6
        for r in &results {
            let val = r.metadata.as_ref().unwrap()["val"].as_i64().unwrap();
            assert!(val >= 3 && val < 7, "val {} not in [3, 7)", val);
        }
    }

    #[test]
    fn test_filter_gt_non_numeric_field_excluded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"name": "alice"}))).unwrap();
        db.add("b", vec![0.0, 1.0], Some(serde_json::json!({"score": 10}))).unwrap();

        // $gt on a string field should return no match for that doc.
        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Gt("name".into(), 5.0))).unwrap();
        assert!(results.iter().all(|r| r.id != "a"), "string field matched numeric $gt");
    }

    #[test]
    fn test_filter_gt_missing_field_excluded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], None).unwrap();
        db.add("b", vec![0.0, 1.0], Some(serde_json::json!({"score": 10}))).unwrap();

        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Gt("score".into(), 5.0))).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }
}
