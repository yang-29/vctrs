use napi::bindgen_prelude::*;
use napi_derive::napi;
use vctrs_core::db::{Database, Filter};
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
    #[napi(constructor)]
    pub fn new(path: String, dim: Option<u32>, metric: Option<String>) -> Result<Self> {
        if dim.is_none() {
            let db = Database::open(&path)
                .map_err(|e| Error::from_reason(e))?;
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

        let db = Database::open_or_create(&path, dim as usize, m)
            .map_err(|e| Error::from_reason(e))?;

        Ok(VctrsDatabase { inner: db })
    }

    #[napi]
    pub fn add(&self, id: String, vector: Vec<f64>, metadata: Option<serde_json::Value>) -> Result<()> {
        let vec_f32: Vec<f32> = vector.iter().map(|&v| v as f32).collect();
        self.inner.add(&id, vec_f32, metadata)
            .map_err(|e| Error::from_reason(e))
    }

    /// Add or update a vector.
    #[napi]
    pub fn upsert(&self, id: String, vector: Vec<f64>, metadata: Option<serde_json::Value>) -> Result<()> {
        let vec_f32: Vec<f32> = vector.iter().map(|&v| v as f32).collect();
        self.inner.upsert(&id, vec_f32, metadata)
            .map_err(|e| Error::from_reason(e))
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
            .map_err(|e| Error::from_reason(e))
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
            .map_err(|e| Error::from_reason(e))?;

        Ok(results.into_iter().map(|r| SearchResult {
            id: r.id,
            distance: r.distance as f64,
            metadata: r.metadata,
        }).collect())
    }

    #[napi]
    pub fn get(&self, id: String) -> Result<GetResult> {
        let (vector, metadata) = self.inner.get(&id)
            .map_err(|e| Error::from_reason(e))?;

        Ok(GetResult {
            vector: vector.iter().map(|&v| v as f64).collect(),
            metadata,
        })
    }

    #[napi]
    pub fn delete(&self, id: String) -> Result<bool> {
        self.inner.delete(&id).map_err(|e| Error::from_reason(e))
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
            .map_err(|e| Error::from_reason(e))
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
        self.inner.save().map_err(|e| Error::from_reason(e))
    }

    /// Close the database and save to disk.
    #[napi]
    pub fn close(&self) -> Result<()> {
        self.inner.save().map_err(|e| Error::from_reason(e))
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
