use napi::bindgen_prelude::*;
use napi_derive::napi;
use vctrs_core::db::Database;
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
    #[napi(constructor)]
    pub fn new(path: String, dim: u32, metric: Option<String>) -> Result<Self> {
        let m = match metric.as_deref().unwrap_or("cosine") {
            "cosine" => Metric::Cosine,
            "euclidean" | "l2" => Metric::Euclidean,
            "dot" | "dot_product" => Metric::DotProduct,
            other => return Err(Error::from_reason(format!(
                "metric must be 'cosine', 'euclidean'/'l2', or 'dot'/'dot_product', got '{}'", other
            ))),
        };

        let db = Database::open(&path, dim as usize, m)
            .map_err(|e| Error::from_reason(e))?;

        Ok(VctrsDatabase { inner: db })
    }

    /// Add a single vector.
    #[napi]
    pub fn add(&self, id: String, vector: Vec<f64>, metadata: Option<serde_json::Value>) -> Result<()> {
        let vec_f32: Vec<f32> = vector.iter().map(|&v| v as f32).collect();
        self.inner.add(&id, vec_f32, metadata)
            .map_err(|e| Error::from_reason(e))
    }

    /// Batch insert vectors.
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
            .map_err(|e| Error::from_reason(e))
    }

    /// Search for k nearest neighbors.
    #[napi]
    pub fn search(&self, vector: Vec<f64>, k: Option<u32>, ef_search: Option<u32>) -> Result<Vec<SearchResult>> {
        let vec_f32: Vec<f32> = vector.iter().map(|&v| v as f32).collect();
        let k = k.unwrap_or(10) as usize;
        let ef = ef_search.map(|e| e as usize);

        let results = self.inner.search(&vec_f32, k, ef)
            .map_err(|e| Error::from_reason(e))?;

        Ok(results.into_iter().map(|r| SearchResult {
            id: r.id,
            distance: r.distance as f64,
            metadata: r.metadata,
        }).collect())
    }

    /// Get a vector and its metadata by id.
    #[napi]
    pub fn get(&self, id: String) -> Result<GetResult> {
        let (vector, metadata) = self.inner.get(&id)
            .map_err(|e| Error::from_reason(e))?;

        Ok(GetResult {
            vector: vector.iter().map(|&v| v as f64).collect(),
            metadata,
        })
    }

    /// Delete a vector by id.
    #[napi]
    pub fn delete(&self, id: String) -> Result<bool> {
        self.inner.delete(&id)
            .map_err(|e| Error::from_reason(e))
    }

    /// Update a vector and/or metadata.
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
            .map_err(|e| Error::from_reason(e))
    }

    /// Check if an id exists.
    #[napi]
    pub fn contains(&self, id: String) -> bool {
        self.inner.contains(&id)
    }

    /// Persist to disk.
    #[napi]
    pub fn save(&self) -> Result<()> {
        self.inner.save().map_err(|e| Error::from_reason(e))
    }

    /// Number of vectors.
    #[napi(getter)]
    pub fn length(&self) -> u32 {
        self.inner.len() as u32
    }

    /// Vector dimensionality.
    #[napi(getter)]
    pub fn dim(&self) -> u32 {
        self.inner.dim() as u32
    }
}
