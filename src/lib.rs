pub mod distance;
pub mod hnsw;
pub mod storage;
pub mod db;

use db::Database;
use distance::Metric;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;

#[pyclass]
struct PyDatabase {
    inner: Database,
}

#[pymethods]
impl PyDatabase {
    #[new]
    #[pyo3(signature = (path, dim, metric = "cosine"))]
    fn new(path: &str, dim: usize, metric: &str) -> PyResult<Self> {
        let m = match metric {
            "cosine" => Metric::Cosine,
            "euclidean" | "l2" => Metric::Euclidean,
            "dot" | "dot_product" => Metric::DotProduct,
            _ => return Err(PyValueError::new_err(
                "metric must be 'cosine', 'euclidean'/'l2', or 'dot'/'dot_product'"
            )),
        };

        let db = Database::open(path, dim, m)
            .map_err(|e| PyValueError::new_err(e))?;

        Ok(PyDatabase { inner: db })
    }

    /// Add a vector with a string id and optional metadata dict.
    #[pyo3(signature = (id, vector, metadata = None))]
    fn add(&self, id: &str, vector: Vec<f32>, metadata: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let meta = match metadata {
            Some(dict) => {
                let json_str = pythonize_dict(dict)?;
                Some(json_str)
            }
            None => None,
        };

        self.inner.add(id, vector, meta)
            .map_err(|e| PyValueError::new_err(e))
    }

    /// Search for the k nearest neighbors. Returns list of (id, distance, metadata).
    #[pyo3(signature = (vector, k = 10))]
    fn search(&self, vector: Vec<f32>, k: usize) -> PyResult<Vec<(String, f32, Option<PyObject>)>> {
        let results = self.inner.search(&vector, k)
            .map_err(|e| PyValueError::new_err(e))?;

        Python::with_gil(|py| {
            results
                .into_iter()
                .map(|r| {
                    let meta = r.metadata.map(|m| json_to_pyobject(py, &m));
                    Ok((r.id, r.distance, meta))
                })
                .collect()
        })
    }

    /// Get a vector and its metadata by id.
    fn get(&self, id: &str) -> PyResult<(Vec<f32>, Option<PyObject>)> {
        let (vector, meta) = self.inner.get(id)
            .map_err(|e| PyValueError::new_err(e))?;

        Python::with_gil(|py| {
            let py_meta = meta.map(|m| json_to_pyobject(py, &m));
            Ok((vector, py_meta))
        })
    }

    /// Persist the database to disk.
    fn save(&self) -> PyResult<()> {
        self.inner.save().map_err(|e| PyValueError::new_err(e))
    }

    /// Number of vectors in the database.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Dimensionality of vectors.
    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }
}

/// Convert a Python dict to a serde_json::Value.
fn pythonize_dict(dict: &Bound<'_, PyDict>) -> PyResult<serde_json::Value> {
    // Simple approach: serialize to JSON string via Python, then parse.
    let py = dict.py();
    let json_module = py.import_bound("json")?;
    let json_str: String = json_module
        .call_method1("dumps", (dict,))?
        .extract()?;
    serde_json::from_str(&json_str)
        .map_err(|e| PyValueError::new_err(format!("invalid metadata: {}", e)))
}

/// Convert a serde_json::Value to a Python object.
fn json_to_pyobject(py: Python<'_>, value: &serde_json::Value) -> PyObject {
    let json_module = py.import_bound("json").unwrap();
    let json_str = serde_json::to_string(value).unwrap();
    json_module
        .call_method1("loads", (json_str,))
        .unwrap()
        .to_object(py)
}

#[pymodule]
fn _vctrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatabase>()?;
    Ok(())
}
