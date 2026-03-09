use napi::bindgen_prelude::*;
use napi::Task;
use napi_derive::napi;
use std::sync::Arc;
use vctrs_core::db::{Database, HnswConfig};
use vctrs_core::distance::Metric;
use vctrs_core::filter::Filter;

/// Convert JS f64 vector to Rust f32.
fn to_f32(v: Vec<f64>) -> Vec<f32> {
    v.into_iter().map(|x| x as f32).collect()
}

/// Convert JS f64 vectors to Rust f32.
fn to_f32_batch(vs: Vec<Vec<f64>>) -> Vec<Vec<f32>> {
    vs.into_iter().map(to_f32).collect()
}

/// Convert Rust f32 vector to JS f64.
fn to_f64(v: Vec<f32>) -> Vec<f64> {
    v.into_iter().map(|x| x as f64).collect()
}

#[napi(object)]
pub struct SearchResult {
    pub id: String,
    pub distance: f64,
    pub metadata: Option<serde_json::Value>,
}

#[napi(object)]
pub struct GetResult {
    pub vector: Vec<f64>,
    pub metadata: Option<serde_json::Value>,
}

fn convert_results(results: Vec<vctrs_core::db::SearchResult>) -> Vec<SearchResult> {
    results.into_iter().map(|r| SearchResult {
        id: r.id,
        distance: r.distance as f64,
        metadata: r.metadata,
    }).collect()
}

#[napi]
pub struct VctrsDatabase {
    inner: Arc<Database>,
}

#[napi]
impl VctrsDatabase {
    /// Open or create a database.
    /// If the database exists, dim and metric are auto-detected (pass null).
    /// Optional: `hnswM` (default 16), `efConstruction` (default 200), `quantize` (default false).
    #[napi(constructor)]
    pub fn new(
        path: String,
        dim: Option<u32>,
        metric: Option<String>,
        hnsw_m: Option<u32>,
        ef_construction: Option<u32>,
        quantize: Option<bool>,
    ) -> Result<Self> {
        if dim.is_none() {
            let db = Database::open(&path)
                .map_err(|e| Error::from_reason(e.to_string()))?;
            return Ok(VctrsDatabase { inner: Arc::new(db) });
        }

        let dim = dim.unwrap();
        let m = Metric::from_str(metric.as_deref().unwrap_or("cosine"))
            .map_err(|e| Error::from_reason(e.to_string()))?;

        let config = HnswConfig {
            m: hnsw_m.unwrap_or(16) as usize,
            ef_construction: ef_construction.unwrap_or(200) as usize,
            quantize: quantize.unwrap_or(false),
        };

        let db = Database::open_or_create_with_config(&path, dim as usize, m, config)
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(VctrsDatabase { inner: Arc::new(db) })
    }

    // -- Sync methods ---------------------------------------------------------

    #[napi]
    pub fn add(&self, id: String, vector: Vec<f64>, metadata: Option<serde_json::Value>) -> Result<()> {
        self.inner.add(&id, to_f32(vector), metadata)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Add or update a vector.
    #[napi]
    pub fn upsert(&self, id: String, vector: Vec<f64>, metadata: Option<serde_json::Value>) -> Result<()> {
        self.inner.upsert(&id, to_f32(vector), metadata)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    #[napi]
    pub fn add_many(
        &self,
        ids: Vec<String>,
        vectors: Vec<Vec<f64>>,
        metadatas: Option<Vec<Option<serde_json::Value>>>,
    ) -> Result<()> {
        let metas = metadatas.unwrap_or_else(|| vec![None; ids.len()]);

        if ids.len() != vectors.len() {
            return Err(Error::from_reason(format!(
                "ids length ({}) != vectors length ({})", ids.len(), vectors.len()
            )));
        }

        let items: Vec<_> = ids.into_iter()
            .zip(to_f32_batch(vectors))
            .zip(metas)
            .map(|((id, vec), meta)| (id, vec, meta))
            .collect();

        self.inner.add_many(items)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Search for k nearest neighbors.
    /// `whereFilter`: optional object like { field: "value" }, { field: { $ne: "value" } },
    /// { field: { $in: ["a", "b"] } }, or { field: { $gt: 10, $lte: 20 } }
    /// `maxDistance`: optional number — discard results beyond this distance.
    #[napi]
    pub fn search(
        &self,
        vector: Vec<f64>,
        k: Option<u32>,
        ef_search: Option<u32>,
        where_filter: Option<serde_json::Value>,
        max_distance: Option<f64>,
    ) -> Result<Vec<SearchResult>> {
        let vec_f32 = to_f32(vector);
        let k = k.unwrap_or(10) as usize;
        let ef = ef_search.map(|e| e as usize);
        let filter = where_filter.as_ref().map(parse_js_filter).transpose()?;
        let max_dist = max_distance.map(|d| d as f32);

        let results = self.inner.search(&vec_f32, k, ef, filter.as_ref(), max_dist)
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(convert_results(results))
    }

    /// Search multiple queries in parallel. Returns array of result arrays.
    /// `maxDistance`: optional number — discard results beyond this distance.
    #[napi]
    pub fn search_many(
        &self,
        vectors: Vec<Vec<f64>>,
        k: Option<u32>,
        ef_search: Option<u32>,
        where_filter: Option<serde_json::Value>,
        max_distance: Option<f64>,
    ) -> Result<Vec<Vec<SearchResult>>> {
        let vecs_f32 = to_f32_batch(vectors);
        let queries: Vec<&[f32]> = vecs_f32.iter().map(|v| v.as_slice()).collect();
        let k = k.unwrap_or(10) as usize;
        let ef = ef_search.map(|e| e as usize);
        let filter = where_filter.as_ref().map(parse_js_filter).transpose()?;
        let max_dist = max_distance.map(|d| d as f32);

        let results = self.inner.search_many(&queries, k, ef, filter.as_ref(), max_dist)
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(results.into_iter().map(convert_results).collect())
    }

    /// Batch upsert — inserts new vectors, updates existing ones.
    #[napi]
    pub fn upsert_many(
        &self,
        ids: Vec<String>,
        vectors: Vec<Vec<f64>>,
        metadatas: Option<Vec<Option<serde_json::Value>>>,
    ) -> Result<()> {
        let metas = metadatas.unwrap_or_else(|| vec![None; ids.len()]);

        if ids.len() != vectors.len() {
            return Err(Error::from_reason(format!(
                "ids length ({}) != vectors length ({})", ids.len(), vectors.len()
            )));
        }

        let items: Vec<_> = ids.into_iter()
            .zip(to_f32_batch(vectors))
            .zip(metas)
            .map(|((id, vec), meta)| (id, vec, meta))
            .collect();

        self.inner.upsert_many(items)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Count vectors matching a filter, or all vectors if no filter.
    #[napi]
    pub fn count(&self, where_filter: Option<serde_json::Value>) -> Result<u32> {
        let filter = where_filter.as_ref().map(parse_js_filter).transpose()?;
        Ok(self.inner.count(filter.as_ref()) as u32)
    }

    #[napi]
    pub fn get(&self, id: String) -> Result<GetResult> {
        let (vector, metadata) = self.inner.get(&id)
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(GetResult {
            vector: to_f64(vector),
            metadata,
        })
    }

    #[napi]
    pub fn delete(&self, id: String) -> Result<bool> {
        self.inner.delete(&id).map_err(|e| Error::from_reason(e.to_string()))
    }

    #[napi]
    pub fn delete_many(&self, ids: Vec<String>) -> Result<u32> {
        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        self.inner.delete_many(&id_refs)
            .map(|n| n as u32)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    #[napi]
    pub fn update(
        &self,
        id: String,
        vector: Option<Vec<f64>>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        let vec = vector.map(to_f32);
        let meta = metadata.map(Some);
        self.inner.update(&id, vec, meta)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    #[napi]
    pub fn contains(&self, id: String) -> bool {
        self.inner.contains(&id)
    }

    #[napi]
    pub fn ids(&self) -> Vec<String> {
        self.inner.ids()
    }

    #[napi]
    pub fn save(&self) -> Result<()> {
        self.inner.save().map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Close the database and save to disk.
    #[napi]
    pub fn close(&self) -> Result<()> {
        self.inner.save().map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Get graph-level statistics for diagnostics.
    #[napi]
    pub fn stats(&self) -> serde_json::Value {
        let s = self.inner.stats();
        serde_json::json!({
            "numVectors": s.num_vectors,
            "numDeleted": s.num_deleted,
            "numLayers": s.num_layers,
            "avgDegreeLayer0": s.avg_degree_layer0,
            "maxDegreeLayer0": s.max_degree_layer0,
            "minDegreeLayer0": s.min_degree_layer0,
            "memoryVectorsBytes": s.memory_vectors_bytes,
            "memoryGraphBytes": s.memory_graph_bytes,
            "memoryQuantizedBytes": s.memory_quantized_bytes,
            "usesBruteForce": s.uses_brute_force,
            "usesQuantizedSearch": s.uses_quantized_search,
        })
    }

    /// Rebuild the index with only live vectors, reclaiming deleted slots.
    #[napi]
    pub fn compact(&self) -> Result<()> {
        self.inner.compact().map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Enable quantized search: uses SQ8 quantized vectors for faster HNSW traversal
    /// with full-precision re-ranking.
    #[napi]
    pub fn enable_quantized_search(&self) {
        self.inner.enable_quantized_search();
    }

    /// Disable quantized search.
    #[napi]
    pub fn disable_quantized_search(&self) {
        self.inner.disable_quantized_search();
    }

    /// Whether quantized search is currently enabled.
    #[napi(getter)]
    pub fn quantized_search(&self) -> bool {
        self.inner.has_quantized_search()
    }

    /// Number of deleted slots that haven't been reclaimed.
    #[napi(getter)]
    pub fn deleted_count(&self) -> u32 {
        self.inner.deleted_count() as u32
    }

    /// Total allocated slots (active + deleted).
    #[napi(getter)]
    pub fn total_slots(&self) -> u32 {
        self.inner.total_slots() as u32
    }

    #[napi(getter)]
    pub fn length(&self) -> u32 {
        self.inner.len() as u32
    }

    #[napi(getter)]
    pub fn dim(&self) -> u32 {
        self.inner.dim() as u32
    }

    #[napi(getter)]
    pub fn metric(&self) -> String {
        match self.inner.metric() {
            Metric::Cosine => "cosine".to_string(),
            Metric::Euclidean => "euclidean".to_string(),
            Metric::DotProduct => "dot_product".to_string(),
        }
    }

    // -- Async methods (return Promises, run on libuv thread pool) -------------

    /// Async version of add. Returns a Promise.
    #[napi]
    pub fn add_async(&self, id: String, vector: Vec<f64>, metadata: Option<serde_json::Value>) -> AsyncTask<AddTask> {
        AsyncTask::new(AddTask {
            db: Arc::clone(&self.inner),
            id,
            vector: to_f32(vector),
            metadata,
        })
    }

    /// Async version of upsert. Returns a Promise.
    #[napi]
    pub fn upsert_async(&self, id: String, vector: Vec<f64>, metadata: Option<serde_json::Value>) -> AsyncTask<UpsertTask> {
        AsyncTask::new(UpsertTask {
            db: Arc::clone(&self.inner),
            id,
            vector: to_f32(vector),
            metadata,
        })
    }

    /// Async version of addMany. Returns a Promise.
    #[napi]
    pub fn add_many_async(
        &self,
        ids: Vec<String>,
        vectors: Vec<Vec<f64>>,
        metadatas: Option<Vec<Option<serde_json::Value>>>,
    ) -> Result<AsyncTask<AddManyTask>> {
        let metas = metadatas.unwrap_or_else(|| vec![None; ids.len()]);
        if ids.len() != vectors.len() {
            return Err(Error::from_reason(format!(
                "ids length ({}) != vectors length ({})", ids.len(), vectors.len()
            )));
        }
        let items: Vec<_> = ids.into_iter()
            .zip(to_f32_batch(vectors))
            .zip(metas)
            .map(|((id, vec), meta)| (id, vec, meta))
            .collect();

        Ok(AsyncTask::new(AddManyTask {
            db: Arc::clone(&self.inner),
            items,
        }))
    }

    /// Async version of search. Returns a Promise<SearchResult[]>.
    #[napi]
    pub fn search_async(
        &self,
        vector: Vec<f64>,
        k: Option<u32>,
        ef_search: Option<u32>,
        where_filter: Option<serde_json::Value>,
        max_distance: Option<f64>,
    ) -> Result<AsyncTask<SearchTask>> {
        let k = k.unwrap_or(10) as usize;
        let ef = ef_search.map(|e| e as usize);
        let filter = where_filter.as_ref().map(parse_js_filter).transpose()?;
        let max_dist = max_distance.map(|d| d as f32);

        Ok(AsyncTask::new(SearchTask {
            db: Arc::clone(&self.inner),
            query: to_f32(vector),
            k,
            ef,
            filter,
            max_distance: max_dist,
        }))
    }

    /// Async version of searchMany. Returns a Promise<SearchResult[][]>.
    #[napi]
    pub fn search_many_async(
        &self,
        vectors: Vec<Vec<f64>>,
        k: Option<u32>,
        ef_search: Option<u32>,
        where_filter: Option<serde_json::Value>,
        max_distance: Option<f64>,
    ) -> Result<AsyncTask<SearchManyTask>> {
        let vecs_f32 = to_f32_batch(vectors);
        let k = k.unwrap_or(10) as usize;
        let ef = ef_search.map(|e| e as usize);
        let filter = where_filter.as_ref().map(parse_js_filter).transpose()?;
        let max_dist = max_distance.map(|d| d as f32);

        Ok(AsyncTask::new(SearchManyTask {
            db: Arc::clone(&self.inner),
            queries: vecs_f32,
            k,
            ef,
            filter,
            max_distance: max_dist,
        }))
    }

    /// Async version of upsertMany. Returns a Promise.
    #[napi]
    pub fn upsert_many_async(
        &self,
        ids: Vec<String>,
        vectors: Vec<Vec<f64>>,
        metadatas: Option<Vec<Option<serde_json::Value>>>,
    ) -> Result<AsyncTask<UpsertManyTask>> {
        let metas = metadatas.unwrap_or_else(|| vec![None; ids.len()]);
        if ids.len() != vectors.len() {
            return Err(Error::from_reason(format!(
                "ids length ({}) != vectors length ({})", ids.len(), vectors.len()
            )));
        }
        let items: Vec<_> = ids.into_iter()
            .zip(to_f32_batch(vectors))
            .zip(metas)
            .map(|((id, vec), meta)| (id, vec, meta))
            .collect();

        Ok(AsyncTask::new(UpsertManyTask {
            db: Arc::clone(&self.inner),
            items,
        }))
    }

    /// Async version of save. Returns a Promise.
    #[napi]
    pub fn save_async(&self) -> AsyncTask<SaveTask> {
        AsyncTask::new(SaveTask { db: Arc::clone(&self.inner) })
    }

    /// Async version of delete. Returns a Promise<boolean>.
    #[napi]
    pub fn delete_async(&self, id: String) -> AsyncTask<DeleteTask> {
        AsyncTask::new(DeleteTask { db: Arc::clone(&self.inner), id })
    }
}

// -- AsyncTask implementations ------------------------------------------------

macro_rules! async_task {
    // Simple task: compute returns (), resolve returns ()
    ($name:ident { $($field:ident : $type:ty),* $(,)? } => $compute:expr) => {
        pub struct $name { $($field: $type),* }

        #[napi]
        impl Task for $name {
            type Output = ();
            type JsValue = ();

            fn compute(&mut self) -> Result<Self::Output> {
                $compute(self)
            }

            fn resolve(&mut self, _env: Env, _output: Self::Output) -> Result<Self::JsValue> {
                Ok(())
            }
        }
    };
    // Task with typed output
    ($name:ident { $($field:ident : $type:ty),* $(,)? } => $output:ty, $jsval:ty, $compute:expr, $resolve:expr) => {
        pub struct $name { $($field: $type),* }

        #[napi]
        impl Task for $name {
            type Output = $output;
            type JsValue = $jsval;

            fn compute(&mut self) -> Result<Self::Output> {
                $compute(self)
            }

            fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
                $resolve(output)
            }
        }
    };
}

async_task!(AddTask {
    db: Arc<Database>,
    id: String,
    vector: Vec<f32>,
    metadata: Option<serde_json::Value>,
} => |t: &mut AddTask| {
    t.db.add(&t.id, std::mem::take(&mut t.vector), t.metadata.take())
        .map_err(|e| Error::from_reason(e.to_string()))
});

async_task!(UpsertTask {
    db: Arc<Database>,
    id: String,
    vector: Vec<f32>,
    metadata: Option<serde_json::Value>,
} => |t: &mut UpsertTask| {
    t.db.upsert(&t.id, std::mem::take(&mut t.vector), t.metadata.take())
        .map_err(|e| Error::from_reason(e.to_string()))
});

async_task!(AddManyTask {
    db: Arc<Database>,
    items: Vec<(String, Vec<f32>, Option<serde_json::Value>)>,
} => |t: &mut AddManyTask| {
    t.db.add_many(std::mem::take(&mut t.items))
        .map_err(|e| Error::from_reason(e.to_string()))
});

async_task!(UpsertManyTask {
    db: Arc<Database>,
    items: Vec<(String, Vec<f32>, Option<serde_json::Value>)>,
} => |t: &mut UpsertManyTask| {
    t.db.upsert_many(std::mem::take(&mut t.items))
        .map_err(|e| Error::from_reason(e.to_string()))
});

async_task!(SaveTask {
    db: Arc<Database>,
} => |t: &mut SaveTask| {
    t.db.save().map_err(|e| Error::from_reason(e.to_string()))
});

async_task!(SearchTask {
    db: Arc<Database>,
    query: Vec<f32>,
    k: usize,
    ef: Option<usize>,
    filter: Option<Filter>,
    max_distance: Option<f32>,
} => Vec<vctrs_core::db::SearchResult>, Vec<SearchResult>,
    |t: &mut SearchTask| {
        t.db.search(&t.query, t.k, t.ef, t.filter.as_ref(), t.max_distance)
            .map_err(|e| Error::from_reason(e.to_string()))
    },
    |output| Ok(convert_results(output))
);

async_task!(SearchManyTask {
    db: Arc<Database>,
    queries: Vec<Vec<f32>>,
    k: usize,
    ef: Option<usize>,
    filter: Option<Filter>,
    max_distance: Option<f32>,
} => Vec<Vec<vctrs_core::db::SearchResult>>, Vec<Vec<SearchResult>>,
    |t: &mut SearchManyTask| {
        let refs: Vec<&[f32]> = t.queries.iter().map(|v| v.as_slice()).collect();
        t.db.search_many(&refs, t.k, t.ef, t.filter.as_ref(), t.max_distance)
            .map_err(|e| Error::from_reason(e.to_string()))
    },
    |output| Ok(output.into_iter().map(convert_results).collect())
);

async_task!(DeleteTask {
    db: Arc<Database>,
    id: String,
} => bool, bool,
    |t: &mut DeleteTask| {
        t.db.delete(&t.id).map_err(|e| Error::from_reason(e.to_string()))
    },
    |output| Ok(output)
);

// -- Export / Import ----------------------------------------------------------

#[napi]
impl VctrsDatabase {
    /// Export all vectors and metadata to a JSON file.
    #[napi]
    pub fn export_json(&self, path: String, pretty: Option<bool>) -> Result<()> {
        let file = std::fs::File::create(&path)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let writer = std::io::BufWriter::new(file);
        if pretty.unwrap_or(false) {
            self.inner.export_json_pretty(writer)
        } else {
            self.inner.export_json(writer)
        }
        .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Import vectors from a JSON file into this database (upsert semantics).
    #[napi]
    pub fn import_json(&self, path: String) -> Result<()> {
        let file = std::fs::File::open(&path)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let reader = std::io::BufReader::new(file);
        self.inner.import_json_into(reader)
            .map_err(|e| Error::from_reason(e.to_string()))
    }
}

// -- Client (multi-collection) ------------------------------------------------

#[napi]
pub struct VctrsClient {
    inner: vctrs_core::client::Client,
}

#[napi]
impl VctrsClient {
    #[napi(constructor)]
    pub fn new(path: String) -> Result<Self> {
        let client = vctrs_core::client::Client::new(&path)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(VctrsClient { inner: client })
    }

    /// Create a new collection. Fails if it already exists.
    #[napi]
    pub fn create_collection(
        &self,
        name: String,
        dim: u32,
        metric: Option<String>,
        hnsw_m: Option<u32>,
        ef_construction: Option<u32>,
        quantize: Option<bool>,
    ) -> Result<VctrsDatabase> {
        let m = parse_metric_str(metric.as_deref().unwrap_or("cosine"))?;
        let config = HnswConfig {
            m: hnsw_m.unwrap_or(16) as usize,
            ef_construction: ef_construction.unwrap_or(200) as usize,
            quantize: quantize.unwrap_or(false),
        };
        let db = self.inner.create_collection_with_config(&name, dim as usize, m, config)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(VctrsDatabase { inner: Arc::new(db) })
    }

    /// Get an existing collection by name.
    #[napi]
    pub fn get_collection(&self, name: String) -> Result<VctrsDatabase> {
        let db = self.inner.get_collection(&name)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(VctrsDatabase { inner: Arc::new(db) })
    }

    /// Get or create a collection.
    #[napi]
    pub fn get_or_create_collection(
        &self,
        name: String,
        dim: u32,
        metric: Option<String>,
    ) -> Result<VctrsDatabase> {
        let m = parse_metric_str(metric.as_deref().unwrap_or("cosine"))?;
        let db = self.inner.get_or_create_collection(&name, dim as usize, m)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(VctrsDatabase { inner: Arc::new(db) })
    }

    /// Delete a collection and all its data.
    #[napi]
    pub fn delete_collection(&self, name: String) -> Result<bool> {
        self.inner.delete_collection(&name)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// List all collection names.
    #[napi]
    pub fn list_collections(&self) -> Result<Vec<String>> {
        self.inner.list_collections()
            .map_err(|e| Error::from_reason(e.to_string()))
    }
}

fn parse_metric_str(s: &str) -> Result<Metric> {
    Metric::from_str(s).map_err(|e| Error::from_reason(e.to_string()))
}

// -- Filter parsing -----------------------------------------------------------

fn parse_js_filter(value: &serde_json::Value) -> Result<Filter> {
    vctrs_core::filter::parse_json_filter(value)
        .map_err(|e| Error::from_reason(e.to_string()))
}
