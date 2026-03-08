use vctrs_core::db::{Database, Filter, HnswConfig};
use vctrs_core::distance::Metric;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyDict, PyList};

#[pyclass(name = "SearchResult")]
struct PySearchResult {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    distance: f32,
    meta_json: Option<String>,
}

#[pymethods]
impl PySearchResult {
    #[getter]
    fn metadata(&self, py: Python<'_>) -> Option<PyObject> {
        self.meta_json.as_ref().map(|s| {
            let json_module = py.import_bound("json").unwrap();
            json_module.call_method1("loads", (s.as_str(),)).unwrap().to_object(py)
        })
    }

    fn __repr__(&self) -> String {
        format!("SearchResult(id='{}', distance={:.4})", self.id, self.distance)
    }

    fn __getitem__(&self, py: Python<'_>, idx: usize) -> PyResult<PyObject> {
        match idx {
            0 => Ok(self.id.to_object(py)),
            1 => Ok(self.distance.to_object(py)),
            2 => Ok(self.metadata(py).unwrap_or_else(|| py.None())),
            _ => Err(PyValueError::new_err("index out of range")),
        }
    }
}

#[pyclass(name = "Database")]
struct PyDatabase {
    inner: Database,
}

#[pymethods]
impl PyDatabase {
    /// Open or create a database.
    /// When opening an existing database, dim and metric are auto-detected.
    #[new]
    #[pyo3(signature = (path, dim = None, metric = None, m = None, ef_construction = None, quantize = false))]
    fn new(
        path: &str,
        dim: Option<usize>,
        metric: Option<&str>,
        m: Option<usize>,
        ef_construction: Option<usize>,
        quantize: bool,
    ) -> PyResult<Self> {
        // Try opening existing database first.
        if dim.is_none() {
            match Database::open(path) {
                Ok(db) => return Ok(PyDatabase { inner: db }),
                Err(_) => {
                    return Err(PyValueError::new_err(
                        "database not found — specify dim to create a new one"
                    ));
                }
            }
        }

        let dim = dim.unwrap();
        let metric_enum = match metric.unwrap_or("cosine") {
            "cosine" => Metric::Cosine,
            "euclidean" | "l2" => Metric::Euclidean,
            "dot" | "dot_product" => Metric::DotProduct,
            _ => return Err(PyValueError::new_err(
                "metric must be 'cosine', 'euclidean'/'l2', or 'dot'/'dot_product'"
            )),
        };

        let config = HnswConfig {
            m: m.unwrap_or(16),
            ef_construction: ef_construction.unwrap_or(200),
            quantize,
        };

        let db = Database::open_or_create_with_config(path, dim, metric_enum, config)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PyDatabase { inner: db })
    }

    #[pyo3(signature = (id, vector, metadata = None))]
    fn add(&self, py: Python<'_>, id: &str, vector: VectorInput<'_>, metadata: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let vec = vector.to_vec()?;
        let meta = metadata.map(pythonize_dict).transpose()?;
        let id = id.to_string();
        py.allow_threads(|| {
            self.inner.add(&id, vec, meta)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    /// Add or update a vector.
    #[pyo3(signature = (id, vector, metadata = None))]
    fn upsert(&self, py: Python<'_>, id: &str, vector: VectorInput<'_>, metadata: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let vec = vector.to_vec()?;
        let meta = metadata.map(pythonize_dict).transpose()?;
        let id = id.to_string();
        py.allow_threads(|| {
            self.inner.upsert(&id, vec, meta)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    #[pyo3(signature = (ids, vectors, metadatas = None))]
    fn add_many(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        vectors: BatchVectorInput<'_>,
        metadatas: Option<&Bound<'_, PyList>>,
    ) -> PyResult<()> {
        let vecs = vectors.to_vecs()?;

        if ids.len() != vecs.len() {
            return Err(PyValueError::new_err(format!(
                "ids length ({}) != vectors length ({})",
                ids.len(), vecs.len()
            )));
        }

        let metas: Vec<Option<serde_json::Value>> = match metadatas {
            Some(list) => {
                if list.len() != ids.len() {
                    return Err(PyValueError::new_err(format!(
                        "metadatas length ({}) != ids length ({})",
                        list.len(), ids.len()
                    )));
                }
                list.iter()
                    .map(|item| {
                        if item.is_none() {
                            Ok(None)
                        } else {
                            let dict = item.downcast::<PyDict>()
                                .map_err(|_| PyValueError::new_err("metadatas must be list of dicts or None"))?;
                            pythonize_dict(dict).map(Some)
                        }
                    })
                    .collect::<PyResult<Vec<_>>>()?
            }
            None => vec![None; ids.len()],
        };

        let items: Vec<_> = ids.into_iter().zip(vecs).zip(metas)
            .map(|((id, vec), meta)| (id, vec, meta))
            .collect();

        py.allow_threads(|| {
            self.inner.add_many(items)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    /// Search for the k nearest neighbors.
    /// Returns list of SearchResult objects with .id, .distance, .metadata attributes.
    /// `where_filter`: optional dict like {"field": "value"} or {"field": {"$ne": "value"}}
    #[pyo3(signature = (vector, k = 10, ef_search = None, where_filter = None))]
    fn search(
        &self,
        py: Python<'_>,
        vector: VectorInput<'_>,
        k: usize,
        ef_search: Option<usize>,
        where_filter: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<PySearchResult>> {
        let vec = vector.to_vec()?;
        let filter = where_filter.map(parse_filter).transpose()?;

        let results = py.allow_threads(|| {
            self.inner.search(&vec, k, ef_search, filter.as_ref())
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;

        Ok(results.into_iter()
            .map(|r| PySearchResult {
                id: r.id,
                distance: r.distance,
                meta_json: r.metadata.map(|m| serde_json::to_string(&m).unwrap()),
            })
            .collect())
    }

    /// Search multiple queries in parallel. Returns list of list of SearchResult.
    #[pyo3(signature = (vectors, k = 10, ef_search = None))]
    fn search_many(
        &self,
        py: Python<'_>,
        vectors: BatchVectorInput<'_>,
        k: usize,
        ef_search: Option<usize>,
    ) -> PyResult<Vec<Vec<PySearchResult>>> {
        let vecs = vectors.to_vecs()?;
        let queries: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

        let results = py.allow_threads(|| {
            self.inner.search_many(&queries, k, ef_search, None)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })?;

        Ok(results
            .into_iter()
            .map(|batch| {
                batch
                    .into_iter()
                    .map(|r| PySearchResult {
                        id: r.id,
                        distance: r.distance,
                        meta_json: r.metadata.map(|m| serde_json::to_string(&m).unwrap()),
                    })
                    .collect()
            })
            .collect())
    }

    fn get(&self, id: &str) -> PyResult<(Vec<f32>, Option<PyObject>)> {
        let (vector, meta) = self.inner.get(id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Python::with_gil(|py| {
            let py_meta = meta.map(|m| json_to_pyobject(py, &m));
            Ok((vector, py_meta))
        })
    }

    fn delete(&self, py: Python<'_>, id: &str) -> PyResult<bool> {
        let id = id.to_string();
        py.allow_threads(|| {
            self.inner.delete(&id).map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    #[pyo3(signature = (id, vector = None, metadata = None))]
    fn update(
        &self,
        py: Python<'_>,
        id: &str,
        vector: Option<VectorInput<'_>>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let vec = vector.map(|v| v.to_vec()).transpose()?;
        let meta = metadata.map(|d| pythonize_dict(d).map(Some)).transpose()?;
        let id = id.to_string();
        py.allow_threads(|| {
            self.inner.update(&id, vec, meta)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    fn __contains__(&self, id: &str) -> bool {
        self.inner.contains(id)
    }

    fn ids(&self) -> Vec<String> {
        self.inner.ids()
    }

    fn save(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            self.inner.save().map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    /// Rebuild the index with only live vectors, reclaiming deleted slots.
    /// Call this after many deletes to reduce memory usage and disk size.
    fn compact(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            self.inner.compact().map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    /// Enable quantized search: uses SQ8 quantized vectors for faster HNSW traversal
    /// with full-precision re-ranking. Automatically enabled when quantize=True is set.
    fn enable_quantized_search(&self) {
        self.inner.enable_quantized_search();
    }

    /// Disable quantized search.
    fn disable_quantized_search(&self) {
        self.inner.disable_quantized_search();
    }

    /// Whether quantized search is currently enabled.
    #[getter]
    fn quantized_search(&self) -> bool {
        self.inner.has_quantized_search()
    }

    /// Number of deleted slots that haven't been reclaimed.
    #[getter]
    fn deleted_count(&self) -> usize {
        self.inner.deleted_count()
    }

    /// Total allocated slots (active + deleted).
    #[getter]
    fn total_slots(&self) -> usize {
        self.inner.total_slots()
    }

    /// Get graph-level statistics for diagnostics.
    /// Returns a dict with keys: num_vectors, num_deleted, num_layers, avg_degree_layer0,
    /// max_degree_layer0, min_degree_layer0, memory_vectors_bytes, memory_graph_bytes,
    /// memory_quantized_bytes, uses_brute_force, uses_quantized_search.
    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let s = self.inner.stats();
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("num_vectors", s.num_vectors)?;
        dict.set_item("num_deleted", s.num_deleted)?;
        dict.set_item("num_layers", s.num_layers)?;
        dict.set_item("avg_degree_layer0", s.avg_degree_layer0)?;
        dict.set_item("max_degree_layer0", s.max_degree_layer0)?;
        dict.set_item("min_degree_layer0", s.min_degree_layer0)?;
        dict.set_item("memory_vectors_bytes", s.memory_vectors_bytes)?;
        dict.set_item("memory_graph_bytes", s.memory_graph_bytes)?;
        dict.set_item("memory_quantized_bytes", s.memory_quantized_bytes)?;
        dict.set_item("uses_brute_force", s.uses_brute_force)?;
        dict.set_item("uses_quantized_search", s.uses_quantized_search)?;
        Ok(dict.into())
    }

    /// Context manager: auto-save on exit.
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, pyo3::types::PyAny>>,
        _exc_val: Option<&Bound<'_, pyo3::types::PyAny>>,
        _exc_tb: Option<&Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<bool> {
        self.inner.save().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(false)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn metric(&self) -> &str {
        match self.inner.metric() {
            Metric::Cosine => "cosine",
            Metric::Euclidean => "euclidean",
            Metric::DotProduct => "dot_product",
        }
    }
}

// -- Filter parsing from Python dicts -----------------------------------------

/// Parse a Python filter dict into a Filter.
/// Supports: {"field": "value"}, {"field": {"$ne": "value"}}, {"field": {"$in": [...]}}
/// Multiple keys = AND.
fn parse_filter(dict: &Bound<'_, PyDict>) -> PyResult<Filter> {
    let mut filters = Vec::new();

    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;

        // Check if value is a dict with operators.
        if let Ok(op_dict) = value.downcast::<PyDict>() {
            for (op_key, op_val) in op_dict.iter() {
                let op: String = op_key.extract()?;
                match op.as_str() {
                    "$ne" => {
                        let val = py_to_json(&op_val)?;
                        filters.push(Filter::Ne(key_str.clone(), val));
                    }
                    "$in" => {
                        let list = op_val.downcast::<PyList>()
                            .map_err(|_| PyValueError::new_err("$in value must be a list"))?;
                        let vals: Vec<serde_json::Value> = list.iter()
                            .map(|item| py_to_json(&item))
                            .collect::<PyResult<_>>()?;
                        filters.push(Filter::In(key_str.clone(), vals));
                    }
                    "$gt" => {
                        let n: f64 = op_val.extract()
                            .map_err(|_| PyValueError::new_err("$gt value must be a number"))?;
                        filters.push(Filter::Gt(key_str.clone(), n));
                    }
                    "$gte" => {
                        let n: f64 = op_val.extract()
                            .map_err(|_| PyValueError::new_err("$gte value must be a number"))?;
                        filters.push(Filter::Gte(key_str.clone(), n));
                    }
                    "$lt" => {
                        let n: f64 = op_val.extract()
                            .map_err(|_| PyValueError::new_err("$lt value must be a number"))?;
                        filters.push(Filter::Lt(key_str.clone(), n));
                    }
                    "$lte" => {
                        let n: f64 = op_val.extract()
                            .map_err(|_| PyValueError::new_err("$lte value must be a number"))?;
                        filters.push(Filter::Lte(key_str.clone(), n));
                    }
                    _ => return Err(PyValueError::new_err(format!("unknown operator: {}", op))),
                }
            }
        } else {
            let val = py_to_json(&value)?;
            filters.push(Filter::Eq(key_str, val));
        }
    }

    if filters.len() == 1 {
        Ok(filters.into_iter().next().unwrap())
    } else {
        Ok(Filter::And(filters))
    }
}

fn py_to_json(obj: &Bound<'_, pyo3::types::PyAny>) -> PyResult<serde_json::Value> {
    if let Ok(s) = obj.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(serde_json::Value::Number(i.into()))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(serde_json::json!(f))
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(serde_json::Value::Bool(b))
    } else if obj.is_none() {
        Ok(serde_json::Value::Null)
    } else {
        Err(PyValueError::new_err("unsupported filter value type"))
    }
}

// -- Vector input helpers -----------------------------------------------------

#[derive(FromPyObject)]
enum VectorInput<'py> {
    List(Vec<f32>),
    Numpy(PyReadonlyArray1<'py, f32>),
}

impl VectorInput<'_> {
    fn to_vec(self) -> PyResult<Vec<f32>> {
        match self {
            VectorInput::List(v) => Ok(v),
            VectorInput::Numpy(arr) => Ok(arr.as_slice()?.to_vec()),
        }
    }
}

#[derive(FromPyObject)]
enum BatchVectorInput<'py> {
    Numpy(PyReadonlyArray2<'py, f32>),
    Lists(Vec<Vec<f32>>),
}

impl BatchVectorInput<'_> {
    fn to_vecs(self) -> PyResult<Vec<Vec<f32>>> {
        match self {
            BatchVectorInput::Lists(v) => Ok(v),
            BatchVectorInput::Numpy(arr) => {
                let shape = arr.shape();
                let n = shape[0];
                let dim = shape[1];
                let slice = arr.as_slice()?;
                Ok((0..n).map(|i| slice[i * dim..(i + 1) * dim].to_vec()).collect())
            }
        }
    }
}

fn pythonize_dict(dict: &Bound<'_, PyDict>) -> PyResult<serde_json::Value> {
    let py = dict.py();
    let json_module = py.import_bound("json")?;
    let json_str: String = json_module.call_method1("dumps", (dict,))?.extract()?;
    serde_json::from_str(&json_str)
        .map_err(|e| PyValueError::new_err(format!("invalid metadata: {}", e)))
}

fn json_to_pyobject(py: Python<'_>, value: &serde_json::Value) -> PyObject {
    let json_module = py.import_bound("json").unwrap();
    let json_str = serde_json::to_string(value).unwrap();
    json_module.call_method1("loads", (json_str,)).unwrap().to_object(py)
}

#[pymodule]
fn _vctrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatabase>()?;
    m.add_class::<PySearchResult>()?;
    Ok(())
}
