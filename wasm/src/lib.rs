use vctrs_core::db::{Database, Filter};
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

    /// Batch add multiple vectors.
    pub fn add_many(
        &self,
        ids: Vec<String>,
        vectors: &[f32],
        dim: usize,
        metadatas: JsValue,
    ) -> Result<(), JsValue> {
        let n = ids.len();
        if vectors.len() != n * dim {
            return Err(JsValue::from_str(&format!(
                "vectors length ({}) != ids ({}) * dim ({})", vectors.len(), n, dim
            )));
        }

        let metas: Vec<Option<serde_json::Value>> = if metadatas.is_null() || metadatas.is_undefined() {
            vec![None; n]
        } else {
            let arr = js_sys::Array::from(&metadatas);
            (0..arr.length())
                .map(|i| js_to_json(&arr.get(i)))
                .collect()
        };

        let items: Vec<_> = ids.into_iter()
            .enumerate()
            .map(|(i, id)| {
                let vec = vectors[i * dim..(i + 1) * dim].to_vec();
                (id, vec, metas[i].clone())
            })
            .collect();

        self.inner.add_many(items)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Find the k nearest neighbors to a query vector.
    /// Optional: pass a filter object for metadata filtering.
    pub fn search(
        &self,
        vector: &[f32],
        k: usize,
        where_filter: JsValue,
        max_distance: JsValue,
    ) -> Result<Vec<SearchResult>, JsValue> {
        let filter = if where_filter.is_null() || where_filter.is_undefined() {
            None
        } else {
            let json = js_to_json(&where_filter)
                .ok_or_else(|| JsValue::from_str("invalid filter"))?;
            Some(parse_json_filter(&json)?)
        };

        let max_dist = if max_distance.is_null() || max_distance.is_undefined() {
            None
        } else {
            max_distance.as_f64().map(|d| d as f32)
        };

        let results = self
            .inner
            .search(vector, k, None, filter.as_ref(), max_dist)
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

    /// Retrieve a vector and its metadata by ID.
    pub fn get(&self, id: &str) -> Result<JsValue, JsValue> {
        let (vector, metadata) = self.inner.get(id)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let obj = js_sys::Object::new();
        let vec_arr = js_sys::Float32Array::from(vector.as_slice());
        js_sys::Reflect::set(&obj, &"vector".into(), &vec_arr)?;
        let meta_js = match &metadata {
            Some(v) => json_to_js(v),
            None => JsValue::NULL,
        };
        js_sys::Reflect::set(&obj, &"metadata".into(), &meta_js)?;
        Ok(obj.into())
    }

    /// Return all vector IDs.
    pub fn ids(&self) -> Vec<String> {
        self.inner.ids()
    }

    /// Count vectors, optionally with a metadata filter.
    pub fn count(&self, where_filter: JsValue) -> Result<usize, JsValue> {
        let filter = if where_filter.is_null() || where_filter.is_undefined() {
            None
        } else {
            let json = js_to_json(&where_filter)
                .ok_or_else(|| JsValue::from_str("invalid filter"))?;
            Some(parse_json_filter(&json)?)
        };
        Ok(self.inner.count(filter.as_ref()))
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

    /// Export all vectors and metadata as a JSON string.
    pub fn export_json(&self) -> Result<String, JsValue> {
        let mut buf = Vec::new();
        self.inner.export_json(&mut buf)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        String::from_utf8(buf)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Import vectors from a JSON string (upsert semantics).
    pub fn import_json(&self, json: &str) -> Result<(), JsValue> {
        self.inner.import_json_into(json.as_bytes())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// Parse a JS filter object into a Filter.
fn parse_json_filter(value: &serde_json::Value) -> Result<Filter, JsValue> {
    let obj = value.as_object()
        .ok_or_else(|| JsValue::from_str("filter must be an object"))?;

    let mut filters = Vec::new();

    for (key, val) in obj {
        if let Some(op_obj) = val.as_object() {
            for (op, op_val) in op_obj {
                match op.as_str() {
                    "$eq" => filters.push(Filter::Eq(key.clone(), op_val.clone())),
                    "$ne" => filters.push(Filter::Ne(key.clone(), op_val.clone())),
                    "$in" => {
                        let arr = op_val.as_array()
                            .ok_or_else(|| JsValue::from_str("$in value must be an array"))?;
                        filters.push(Filter::In(key.clone(), arr.clone()));
                    }
                    "$gt" => {
                        let n = op_val.as_f64()
                            .ok_or_else(|| JsValue::from_str("$gt must be a number"))?;
                        filters.push(Filter::Gt(key.clone(), n));
                    }
                    "$gte" => {
                        let n = op_val.as_f64()
                            .ok_or_else(|| JsValue::from_str("$gte must be a number"))?;
                        filters.push(Filter::Gte(key.clone(), n));
                    }
                    "$lt" => {
                        let n = op_val.as_f64()
                            .ok_or_else(|| JsValue::from_str("$lt must be a number"))?;
                        filters.push(Filter::Lt(key.clone(), n));
                    }
                    "$lte" => {
                        let n = op_val.as_f64()
                            .ok_or_else(|| JsValue::from_str("$lte must be a number"))?;
                        filters.push(Filter::Lte(key.clone(), n));
                    }
                    _ => return Err(JsValue::from_str(&format!("unknown operator: {}", op))),
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
