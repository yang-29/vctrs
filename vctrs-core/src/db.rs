/// The main database: ties together HNSW index, storage, and id mapping.

use crate::distance::Metric;
use crate::error::{VctrsError, Result};
use crate::filter::MetadataIndex;
use crate::hnsw::{GraphStats, HnswIndex};
use crate::storage::{MetaRecord, Storage};
use crate::wal::{Wal, WalEntry};
use parking_lot::{Mutex, RwLock};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

// Re-export Filter so `vctrs_core::db::Filter` still works.
pub use crate::filter::Filter;

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
    meta_index: RwLock<MetadataIndex>,
    storage: Storage,
    dim: usize,
    quantize_on_save: bool,
    wal: Mutex<Wal>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub metadata: Option<serde_json::Value>,
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

            let mut mi = MetadataIndex::new();
            let deleted = index.deleted_ids().clone();
            mi.rebuild(&metadata, &deleted);

            let wal = Wal::new(&db_path);

            let mut db = Database {
                index: RwLock::new(index),
                id_map: RwLock::new(id_map),
                reverse_map: RwLock::new(reverse_map),
                metadata: RwLock::new(metadata),
                meta_index: RwLock::new(mi),
                storage,
                dim,
                quantize_on_save: has_quantized,
                wal: Mutex::new(wal),
            };

            // Replay WAL entries on top of the snapshot.
            db.replay_wal()?;

            return Ok(db);
        }

        Err(VctrsError::DatabaseNotFound(path.to_string()))
    }

    /// Create a purely in-memory database (no filesystem, no WAL).
    /// Useful for WASM and ephemeral use cases.
    pub fn in_memory(dim: usize, metric: Metric) -> Self {
        Database {
            index: RwLock::new(HnswIndex::new(dim, metric, 16, 200)),
            id_map: RwLock::new(HashMap::new()),
            reverse_map: RwLock::new(Vec::new()),
            metadata: RwLock::new(Vec::new()),
            meta_index: RwLock::new(MetadataIndex::new()),
            storage: Storage::new(&PathBuf::new(), dim),
            dim,
            quantize_on_save: false,
            wal: Mutex::new(Wal::noop()),
        }
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

            let mut mi = MetadataIndex::new();
            let deleted = index.deleted_ids().clone();
            mi.rebuild(&metadata, &deleted);

            let wal = Wal::new(&db_path);

            let mut db = Database {
                index: RwLock::new(index),
                id_map: RwLock::new(id_map),
                reverse_map: RwLock::new(reverse_map),
                metadata: RwLock::new(metadata),
                meta_index: RwLock::new(mi),
                storage,
                dim: loaded_dim,
                quantize_on_save: config.quantize || has_quantized,
                wal: Mutex::new(wal),
            };

            db.replay_wal()?;

            return Ok(db);
        }

        let wal = Wal::new(&db_path);

        Ok(Database {
            index: RwLock::new(HnswIndex::new(dim, metric, config.m, config.ef_construction)),
            id_map: RwLock::new(HashMap::new()),
            reverse_map: RwLock::new(Vec::new()),
            metadata: RwLock::new(Vec::new()),
            meta_index: RwLock::new(MetadataIndex::new()),
            storage,
            dim,
            quantize_on_save: config.quantize,
            wal: Mutex::new(wal),
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

        // Quick duplicate check with read lock (fast path for non-conflicting adds).
        if self.id_map.read().contains_key(id) {
            return Err(VctrsError::DuplicateId(id.to_string()));
        }

        // Write WAL entry before mutating in-memory state (outside id_map lock).
        self.wal.lock().append(&WalEntry::Add {
            id: id.to_string(),
            vector: vector.clone(),
            metadata: metadata.clone(),
        })?;

        // Acquire write locks for mutation.
        let mut id_map = self.id_map.write();
        // Double-check after acquiring write lock.
        if id_map.contains_key(id) {
            return Err(VctrsError::DuplicateId(id.to_string()));
        }

        // Hold index write lock while pushing metadata so search threads
        // can't see the vector before its metadata is ready.
        let mut index = self.index.write();
        let internal_id = index.insert(vector);

        self.reverse_map.write().push(id.to_string());
        self.meta_index.write().index(internal_id, &metadata);
        self.metadata.write().push(metadata);

        id_map.insert(id.to_string(), internal_id);
        drop(index);
        drop(id_map);

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

        // WAL: upsert is logged as an Add (replay handles dedup via upsert logic).
        self.wal.lock().append(&WalEntry::Add {
            id: id.to_string(),
            vector: vector.clone(),
            metadata: metadata.clone(),
        })?;

        // Fast path: check with read lock first.
        {
            let id_map = self.id_map.read();
            if let Some(&internal_id) = id_map.get(id) {
                drop(id_map);
                self.index.write().update_vector(internal_id, vector);
                let mut meta_store = self.metadata.write();
                let mut mi = self.meta_index.write();
                mi.remove(internal_id, &meta_store[internal_id as usize]);
                mi.index(internal_id, &metadata);
                meta_store[internal_id as usize] = metadata;
                return Ok(());
            }
        }

        // Not found — insert with write lock.
        let mut id_map = self.id_map.write();
        // Double-check after acquiring write lock.
        if let Some(&internal_id) = id_map.get(id) {
            drop(id_map);
            self.index.write().update_vector(internal_id, vector);
            let mut meta_store = self.metadata.write();
            let mut mi = self.meta_index.write();
            mi.remove(internal_id, &meta_store[internal_id as usize]);
            mi.index(internal_id, &metadata);
            meta_store[internal_id as usize] = metadata;
            return Ok(());
        }

        // Hold index write lock while pushing metadata so search threads
        // can't see the vector before its metadata is ready.
        let mut index = self.index.write();
        let internal_id = index.insert(vector);

        self.reverse_map.write().push(id.to_string());
        self.meta_index.write().index(internal_id, &metadata);
        self.metadata.write().push(metadata);

        id_map.insert(id.to_string(), internal_id);
        drop(index);
        drop(id_map);

        Ok(())
    }

    /// Batch insert — uses parallel HNSW construction.
    pub fn add_many(
        &self,
        items: Vec<(String, Vec<f32>, Option<serde_json::Value>)>,
    ) -> Result<()> {
        for (_id, vector, _) in &items {
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

        // Write WAL entries for each item before mutating in-memory state.
        {
            let mut wal = self.wal.lock();
            for (id, vector, metadata) in &items {
                wal.append(&WalEntry::Add {
                    id: id.clone(),
                    vector: vector.clone(),
                    metadata: metadata.clone(),
                })?;
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
        let mut mi = self.meta_index.write();

        for (i, id) in ids_vec.into_iter().enumerate() {
            id_map.insert(id.clone(), internal_ids[i]);
            while reverse_map.len() <= internal_ids[i] as usize {
                reverse_map.push(String::new());
            }
            reverse_map[internal_ids[i] as usize] = id;
            while metadata.len() <= internal_ids[i] as usize {
                metadata.push(None);
            }
            mi.index(internal_ids[i], &metas[i]);
            metadata[internal_ids[i] as usize] = metas[i].clone();
        }

        Ok(())
    }

    /// Batch upsert — inserts new vectors, updates existing ones.
    pub fn upsert_many(
        &self,
        items: Vec<(String, Vec<f32>, Option<serde_json::Value>)>,
    ) -> Result<()> {
        for (_, vector, _) in &items {
            if vector.len() != self.dim {
                return Err(VctrsError::DimensionMismatch { expected: self.dim, got: vector.len() });
            }
        }

        // WAL: log all items as Add entries (replay uses upsert semantics).
        {
            let mut wal = self.wal.lock();
            for (id, vector, metadata) in &items {
                wal.append(&WalEntry::Add {
                    id: id.clone(),
                    vector: vector.clone(),
                    metadata: metadata.clone(),
                })?;
            }
        }

        // Partition into updates (existing ids) and inserts (new ids).
        let mut to_insert = Vec::new();
        {
            let id_map = self.id_map.read();
            for (id, vector, metadata) in items {
                if let Some(&internal_id) = id_map.get(&id) {
                    // Update existing.
                    self.index.write().update_vector(internal_id, vector);
                    let mut meta_store = self.metadata.write();
                    let mut mi = self.meta_index.write();
                    mi.remove(internal_id, &meta_store[internal_id as usize]);
                    mi.index(internal_id, &metadata);
                    meta_store[internal_id as usize] = metadata;
                } else {
                    to_insert.push((id, vector, metadata));
                }
            }
        }

        if to_insert.is_empty() {
            return Ok(());
        }

        // Batch insert the new items.
        let mut id_map = self.id_map.write();
        // Re-check for any that got inserted between the read and write lock.
        let mut final_insert = Vec::new();
        let mut final_ids = Vec::new();
        let mut final_metas = Vec::new();
        for (id, vector, metadata) in to_insert {
            if let Some(&internal_id) = id_map.get(&id) {
                self.index.write().update_vector(internal_id, vector);
                let mut meta_store = self.metadata.write();
                let mut mi = self.meta_index.write();
                mi.remove(internal_id, &meta_store[internal_id as usize]);
                mi.index(internal_id, &metadata);
                meta_store[internal_id as usize] = metadata;
            } else {
                final_ids.push(id);
                final_insert.push(vector);
                final_metas.push(metadata);
            }
        }

        if final_insert.is_empty() {
            return Ok(());
        }

        // Hold index write lock while pushing metadata so search threads
        // can't see vectors before their metadata is ready.
        let mut index = self.index.write();
        let internal_ids = index.batch_insert(final_insert);

        let mut reverse_map = self.reverse_map.write();
        let mut metadata = self.metadata.write();
        let mut mi = self.meta_index.write();

        for (i, id) in final_ids.into_iter().enumerate() {
            id_map.insert(id.clone(), internal_ids[i]);
            while reverse_map.len() <= internal_ids[i] as usize {
                reverse_map.push(String::new());
            }
            reverse_map[internal_ids[i] as usize] = id;
            while metadata.len() <= internal_ids[i] as usize {
                metadata.push(None);
            }
            mi.index(internal_ids[i], &final_metas[i]);
            metadata[internal_ids[i] as usize] = final_metas[i].clone();
        }

        drop(index);
        Ok(())
    }

    pub fn delete(&self, id: &str) -> Result<bool> {
        let internal_id = {
            let mut id_map = self.id_map.write();
            match id_map.remove(id) {
                Some(iid) => iid,
                None => return Ok(false),
            }
        }; // id_map write lock released here.

        self.wal.lock().append(&WalEntry::Delete {
            id: id.to_string(),
        })?;

        self.index.write().mark_deleted(internal_id);
        self.reverse_map.write()[internal_id as usize] = String::new();
        let mut meta_store = self.metadata.write();
        self.meta_index.write().remove(internal_id, &meta_store[internal_id as usize]);
        meta_store[internal_id as usize] = None;

        Ok(true)
    }

    /// Delete multiple vectors by ID. Returns the number of vectors actually deleted.
    pub fn delete_many(&self, ids: &[&str]) -> Result<usize> {
        let mut id_map = self.id_map.write();
        let mut wal = self.wal.lock();
        let mut index = self.index.write();
        let mut reverse_map = self.reverse_map.write();
        let mut meta_store = self.metadata.write();
        let mut mi = self.meta_index.write();

        let mut deleted = 0;
        for id in ids {
            if let Some(internal_id) = id_map.remove(*id) {
                wal.append(&WalEntry::Delete { id: id.to_string() })?;
                index.mark_deleted(internal_id);
                reverse_map[internal_id as usize] = String::new();
                mi.remove(internal_id, &meta_store[internal_id as usize]);
                meta_store[internal_id as usize] = None;
                deleted += 1;
            }
        }

        Ok(deleted)
    }

    pub fn update(
        &self,
        id: &str,
        vector: Option<Vec<f32>>,
        metadata: Option<Option<serde_json::Value>>,
    ) -> Result<()> {
        let internal_id = {
            let id_map = self.id_map.read();
            *id_map
                .get(id)
                .ok_or_else(|| VctrsError::NotFound(id.to_string()))?
        }; // id_map read lock released here.

        if let Some(ref vec) = vector {
            if vec.len() != self.dim {
                return Err(VctrsError::DimensionMismatch { expected: self.dim, got: vec.len() });
            }
        }

        self.wal.lock().append(&WalEntry::Update {
            id: id.to_string(),
            vector: vector.clone(),
            metadata: metadata.clone(),
        })?;

        if let Some(vec) = vector {
            self.index.write().update_vector(internal_id, vec);
        }

        if let Some(meta) = metadata {
            let mut meta_store = self.metadata.write();
            let mut mi = self.meta_index.write();
            mi.remove(internal_id, &meta_store[internal_id as usize]);
            mi.index(internal_id, &meta);
            meta_store[internal_id as usize] = meta;
        }

        Ok(())
    }

    /// Search for the k nearest neighbors, optionally filtered by metadata.
    /// If `max_distance` is set, results beyond that distance are discarded.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
        filter: Option<&Filter>,
        max_distance: Option<f32>,
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

        let raw_results = if let Some(f) = filter {
            let mi = self.meta_index.read();
            let resolved = f.resolve_from_index(&mi);
            drop(mi);

            if let Some(ref match_set) = resolved {
                index.search_filtered(query, k, ef, |id| {
                    match_set.contains(&id) && f.matches(&metadata[id as usize])
                })
            } else {
                index.search_filtered(query, k, ef, |id| {
                    f.matches(&metadata[id as usize])
                })
            }
        } else {
            index.search(query, k, ef)
        };

        let results: Vec<SearchResult> = raw_results
            .into_iter()
            .filter(|(_, dist)| max_distance.map_or(true, |max| *dist <= max))
            .map(|(internal_id, dist)| SearchResult {
                id: reverse_map[internal_id as usize].clone(),
                distance: dist,
                metadata: metadata[internal_id as usize].clone(),
            })
            .collect();

        Ok(results)
    }

    /// Search multiple queries in parallel. Returns one result Vec per query.
    pub fn search_many(
        &self,
        queries: &[&[f32]],
        k: usize,
        ef_search: Option<usize>,
        filter: Option<&Filter>,
        max_distance: Option<f32>,
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

        let to_results = |raw: Vec<(u32, f32)>| -> Vec<SearchResult> {
            raw.into_iter()
                .filter(|(_, dist)| max_distance.map_or(true, |max| *dist <= max))
                .map(|(internal_id, dist)| SearchResult {
                    id: reverse_map[internal_id as usize].clone(),
                    distance: dist,
                    metadata: metadata[internal_id as usize].clone(),
                })
                .collect()
        };

        if let Some(f) = filter {
            let mi = self.meta_index.read();
            let resolved = f.resolve_from_index(&mi);
            drop(mi);

            #[cfg(feature = "parallel")]
            let iter = queries.par_iter();
            #[cfg(not(feature = "parallel"))]
            let iter = queries.iter();

            let results: Vec<Vec<SearchResult>> = iter
                .map(|query| {
                    let raw = if let Some(ref match_set) = resolved {
                        index.search_filtered(query, k, ef, |id| {
                            match_set.contains(&id) && f.matches(&metadata[id as usize])
                        })
                    } else {
                        index.search_filtered(query, k, ef, |id| {
                            f.matches(&metadata[id as usize])
                        })
                    };
                    to_results(raw)
                })
                .collect();

            return Ok(results);
        }

        let raw_results = index.search_many(queries, k, ef);
        Ok(raw_results.into_iter().map(to_results).collect())
    }

    pub fn get(&self, id: &str) -> Result<(Vec<f32>, Option<serde_json::Value>)> {
        let internal_id = {
            let id_map = self.id_map.read();
            *id_map
                .get(id)
                .ok_or_else(|| VctrsError::NotFound(id.to_string()))?
        };

        let vector = {
            let index = self.index.read();
            index
                .get_vector(internal_id)
                .ok_or_else(|| VctrsError::CorruptData("vector slot missing".to_string()))?
                .to_vec()
        };

        let meta = self.metadata.read()[internal_id as usize].clone();

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

    /// Count vectors matching a filter, or all vectors if no filter is provided.
    pub fn count(&self, filter: Option<&Filter>) -> usize {
        let f = match filter {
            Some(f) => f,
            None => return self.len(),
        };

        // Try fast path via inverted index.
        let mi = self.meta_index.read();
        if let Some(match_set) = f.resolve_from_index(&mi) {
            drop(mi);
            // For pure $eq/$in filters, the index gives us the exact count.
            let index = self.index.read();
            let deleted = index.deleted_ids();
            return match_set.iter().filter(|id| !deleted.contains(id)).count();
        }
        drop(mi);

        // Fallback: scan all live vectors.
        let index = self.index.read();
        let metadata = self.metadata.read();
        let deleted = index.deleted_ids();
        let total = index.total_slots();

        (0..total as u32)
            .filter(|id| !deleted.contains(id) && f.matches(&metadata[*id as usize]))
            .count()
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

        // Rebuild metadata index from scratch with the new metadata.
        let empty_deleted = HashSet::new();
        let mut mi = MetadataIndex::new();
        mi.rebuild(&new_metadata, &empty_deleted);

        *id_map = new_id_map;
        *reverse_map = new_reverse_map;
        *self.metadata.write() = new_metadata;
        *self.meta_index.write() = mi;
        *self.index.write() = new_index;

        Ok(())
    }

    /// Replay WAL entries on top of the current in-memory state.
    /// Called on startup after loading a snapshot.
    fn replay_wal(&mut self) -> Result<()> {
        let wal = self.wal.lock();
        let entries = wal.read_entries()?;
        drop(wal);

        if entries.is_empty() {
            return Ok(());
        }

        for entry in entries {
            match entry {
                WalEntry::Add { id, vector, metadata } => {
                    // Use upsert semantics: if the id already exists (e.g. partial replay),
                    // update it rather than failing on duplicate.
                    let id_map = self.id_map.read();
                    if let Some(&internal_id) = id_map.get(&id) {
                        drop(id_map);
                        self.index.write().update_vector(internal_id, vector);
                        let mut meta_store = self.metadata.write();
                        let mut mi = self.meta_index.write();
                        mi.remove(internal_id, &meta_store[internal_id as usize]);
                        mi.index(internal_id, &metadata);
                        meta_store[internal_id as usize] = metadata;
                    } else {
                        drop(id_map);
                        if vector.len() != self.dim {
                            continue; // Skip corrupt entries.
                        }
                        let mut index = self.index.write();
                        let internal_id = index.insert(vector);
                        let mut id_map = self.id_map.write();
                        id_map.insert(id.clone(), internal_id);
                        let mut reverse_map = self.reverse_map.write();
                        while reverse_map.len() <= internal_id as usize {
                            reverse_map.push(String::new());
                        }
                        reverse_map[internal_id as usize] = id;
                        let mut meta_store = self.metadata.write();
                        while meta_store.len() <= internal_id as usize {
                            meta_store.push(None);
                        }
                        self.meta_index.write().index(internal_id, &metadata);
                        meta_store[internal_id as usize] = metadata;
                    }
                }
                WalEntry::Delete { id } => {
                    let mut id_map = self.id_map.write();
                    if let Some(internal_id) = id_map.remove(&id) {
                        self.index.write().mark_deleted(internal_id);
                        self.reverse_map.write()[internal_id as usize] = String::new();
                        let mut meta_store = self.metadata.write();
                        self.meta_index.write().remove(internal_id, &meta_store[internal_id as usize]);
                        meta_store[internal_id as usize] = None;
                    }
                }
                WalEntry::Update { id, vector, metadata } => {
                    let id_map = self.id_map.read();
                    if let Some(&internal_id) = id_map.get(&id) {
                        drop(id_map);
                        if let Some(vec) = vector {
                            if vec.len() == self.dim {
                                self.index.write().update_vector(internal_id, vec);
                            }
                        }
                        if let Some(meta) = metadata {
                            let mut meta_store = self.metadata.write();
                            let mut mi = self.meta_index.write();
                            mi.remove(internal_id, &meta_store[internal_id as usize]);
                            mi.index(internal_id, &meta);
                            meta_store[internal_id as usize] = meta;
                        }
                    }
                    // If the id doesn't exist, skip (it may have been deleted later in the WAL).
                }
            }
        }

        Ok(())
    }

    /// Persist to disk. Writes a full snapshot and truncates the WAL.
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

        // Snapshot written successfully — truncate the WAL.
        self.wal.lock().truncate()?;

        Ok(())
    }
}

#[cfg(test)]
#[path = "db_tests.rs"]
mod tests;
