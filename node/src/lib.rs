use napi::bindgen_prelude::*;
use napi_derive::napi;
use vctrs_core::db::{Database, Filter, HnswConfig};
use vctrs_core::distance::Metric;

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

#[napi]
pub struct VctrsDatabase {
    inner: Database,
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
            return Ok(VctrsDatabase { inner: db });
        }

        let dim = dim.unwrap();
        let m = match metric.as_deref().unwrap_or("cosine") {
            "cosine" => Metric::Cosine,
            "euclidean" | "l2" => Metric::Euclidean,
            "dot" | "dot_product" => Metric::DotProduct,
            other => return Err(Error::from_reason(format!(
                "metric must be 'cosine', 'euclidean'/'l2', or 'dot'/'dot_product', got '{}'", other
            ))),
        };

        let config = HnswConfig {
            m: hnsw_m.unwrap_or(16) as usize,
            ef_construction: ef_construction.unwrap_or(200) as usize,
            quantize: quantize.unwrap_or(false),
        };

        let db = Database::open_or_create_with_config(&path, dim as usize, m, config)
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(VctrsDatabase { inner: db })
    }

    #[napi]
    pub fn add(&self, id: String, vector: Vec<f64>, metadata: Option<serde_json::Value>) -> Result<()> {
        let vec_f32: Vec<f32> = vector.iter().map(|&v| v as f32).collect();
        self.inner.add(&id, vec_f32, metadata)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Add or update a vector.
    #[napi]
    pub fn upsert(&self, id: String, vector: Vec<f64>, metadata: Option<serde_json::Value>) -> Result<()> {
        let vec_f32: Vec<f32> = vector.iter().map(|&v| v as f32).collect();
        self.inner.upsert(&id, vec_f32, metadata)
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
            .zip(vectors.into_iter().map(|v| v.iter().map(|&x| x as f32).collect::<Vec<f32>>()))
            .zip(metas)
            .map(|((id, vec), meta)| (id, vec, meta))
            .collect();

        self.inner.add_many(items)
            .map_err(|e| Error::from_reason(e.to_string()))
    }

    /// Search for k nearest neighbors.
    /// `whereFilter`: optional object like { field: "value" } or { field: { $ne: "value" } }
    #[napi]
    pub fn search(
        &self,
        vector: Vec<f64>,
        k: Option<u32>,
        ef_search: Option<u32>,
        where_filter: Option<serde_json::Value>,
    ) -> Result<Vec<SearchResult>> {
        let vec_f32: Vec<f32> = vector.iter().map(|&v| v as f32).collect();
        let k = k.unwrap_or(10) as usize;
        let ef = ef_search.map(|e| e as usize);
        let filter = where_filter.as_ref().map(parse_js_filter).transpose()?;

        let results = self.inner.search(&vec_f32, k, ef, filter.as_ref())
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(results.into_iter().map(|r| SearchResult {
            id: r.id,
            distance: r.distance as f64,
            metadata: r.metadata,
        }).collect())
    }

    /// Search multiple queries in parallel. Returns array of result arrays.
    /// `whereFilter`: optional object like { field: "value" } or { field: { $ne: "value" } }
    #[napi]
    pub fn search_many(
        &self,
        vectors: Vec<Vec<f64>>,
        k: Option<u32>,
        ef_search: Option<u32>,
        where_filter: Option<serde_json::Value>,
    ) -> Result<Vec<Vec<SearchResult>>> {
        let vecs_f32: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| v.iter().map(|&x| x as f32).collect())
            .collect();
        let queries: Vec<&[f32]> = vecs_f32.iter().map(|v| v.as_slice()).collect();
        let k = k.unwrap_or(10) as usize;
        let ef = ef_search.map(|e| e as usize);
        let filter = where_filter.as_ref().map(parse_js_filter).transpose()?;

        let results = self.inner.search_many(&queries, k, ef, filter.as_ref())
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(results
            .into_iter()
            .map(|batch| {
                batch
                    .into_iter()
                    .map(|r| SearchResult {
                        id: r.id,
                        distance: r.distance as f64,
                        metadata: r.metadata,
                    })
                    .collect()
            })
            .collect())
    }

    #[napi]
    pub fn get(&self, id: String) -> Result<GetResult> {
        let (vector, metadata) = self.inner.get(&id)
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(GetResult {
            vector: vector.iter().map(|&v| v as f64).collect(),
            metadata,
        })
    }

    #[napi]
    pub fn delete(&self, id: String) -> Result<bool> {
        self.inner.delete(&id).map_err(|e| Error::from_reason(e.to_string()))
    }

    #[napi]
    pub fn update(
        &self,
        id: String,
        vector: Option<Vec<f64>>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        let vec = vector.map(|v| v.iter().map(|&x| x as f32).collect());
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
    /// Call this after many deletes to reduce memory usage and disk size.
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
}

/// Parse a JS filter object into a Filter.
fn parse_js_filter(value: &serde_json::Value) -> Result<Filter> {
    let obj = value.as_object()
        .ok_or_else(|| Error::from_reason("where_filter must be an object"))?;

    let mut filters = Vec::new();

    for (key, val) in obj {
        if let Some(op_obj) = val.as_object() {
            for (op, op_val) in op_obj {
                match op.as_str() {
                    "$ne" => filters.push(Filter::Ne(key.clone(), op_val.clone())),
                    "$in" => {
                        let arr = op_val.as_array()
                            .ok_or_else(|| Error::from_reason("$in value must be an array"))?;
                        filters.push(Filter::In(key.clone(), arr.clone()));
                    }
                    "$gt" => {
                        let n = op_val.as_f64()
                            .ok_or_else(|| Error::from_reason("$gt value must be a number"))?;
                        filters.push(Filter::Gt(key.clone(), n));
                    }
                    "$gte" => {
                        let n = op_val.as_f64()
                            .ok_or_else(|| Error::from_reason("$gte value must be a number"))?;
                        filters.push(Filter::Gte(key.clone(), n));
                    }
                    "$lt" => {
                        let n = op_val.as_f64()
                            .ok_or_else(|| Error::from_reason("$lt value must be a number"))?;
                        filters.push(Filter::Lt(key.clone(), n));
                    }
                    "$lte" => {
                        let n = op_val.as_f64()
                            .ok_or_else(|| Error::from_reason("$lte value must be a number"))?;
                        filters.push(Filter::Lte(key.clone(), n));
                    }
                    _ => return Err(Error::from_reason(format!("unknown operator: {}", op))),
                }
            }
        } else {
            filters.push(Filter::Eq(key.clone(), val.clone()));
        }
    }

    if filters.len() == 1 {
        Ok(filters.into_iter().next().unwrap())
    } else {
        Ok(Filter::And(filters))
    }
}
