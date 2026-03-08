use vctrs_core::db::Database;
use vctrs_core::distance::Metric;
use wasm_bindgen::prelude::*;

fn js_to_json(val: &JsValue) -> Option<serde_json::Value> {
    if val.is_null() || val.is_undefined() {
        return None;
    }
    let json_str = js_sys::JSON::stringify(val).ok()?.as_string()?;
    serde_json::from_str(&json_str).ok()
}

fn json_to_js(val: &serde_json::Value) -> JsValue {
    let s = serde_json::to_string(val).unwrap_or_default();
    js_sys::JSON::parse(&s).unwrap_or(JsValue::NULL)
}

#[wasm_bindgen]
pub struct VctrsDatabase {
    inner: Database,
}

#[wasm_bindgen]
pub struct SearchResult {
    #[wasm_bindgen(readonly)]
    pub distance: f64,
    id: String,
    metadata: Option<serde_json::Value>,
}

#[wasm_bindgen]
impl SearchResult {
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.id.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn metadata(&self) -> JsValue {
        match &self.metadata {
            Some(v) => json_to_js(v),
            None => JsValue::NULL,
        }
    }
}

fn parse_metric(s: &str) -> Result<Metric, JsValue> {
    match s {
        "cosine" => Ok(Metric::Cosine),
        "euclidean" | "l2" => Ok(Metric::Euclidean),
        "dot" | "dot_product" => Ok(Metric::DotProduct),
        _ => Err(JsValue::from_str(&format!("unknown metric: {}", s))),
    }
}

#[wasm_bindgen]
impl VctrsDatabase {
    /// Create a new in-memory vector database.
    #[wasm_bindgen(constructor)]
    pub fn new(dim: usize, metric: &str) -> Result<VctrsDatabase, JsValue> {
        let m = parse_metric(metric)?;
        Ok(VctrsDatabase { inner: Database::in_memory(dim, m) })
    }

    /// Add a vector with a unique string ID and optional metadata.
    pub fn add(&self, id: &str, vector: &[f32], metadata: JsValue) -> Result<(), JsValue> {
        let meta = js_to_json(&metadata);
        self.inner
            .add(id, vector.to_vec(), meta)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Insert or update a vector.
    pub fn upsert(&self, id: &str, vector: &[f32], metadata: JsValue) -> Result<(), JsValue> {
        let meta = js_to_json(&metadata);
        self.inner
            .upsert(id, vector.to_vec(), meta)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Find the k nearest neighbors to a query vector.
    pub fn search(&self, vector: &[f32], k: usize) -> Result<Vec<SearchResult>, JsValue> {
        let results = self
            .inner
            .search(vector, k, None, None, None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(results
            .into_iter()
            .map(|r| SearchResult {
                id: r.id,
                distance: r.distance as f64,
                metadata: r.metadata,
            })
            .collect())
    }

    /// Delete a vector by ID. Returns true if found and deleted.
    pub fn delete(&self, id: &str) -> Result<bool, JsValue> {
        self.inner
            .delete(id)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Check if a vector ID exists.
    pub fn contains(&self, id: &str) -> bool {
        self.inner.contains(id)
    }

    /// Get the number of vectors in the database.
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    /// Get the vector dimensionality.
    #[wasm_bindgen(getter)]
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }
}
