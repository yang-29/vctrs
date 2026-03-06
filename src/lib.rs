use vctrs_core::db::Database;
use vctrs_core::distance::Metric;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyDict, PyList};

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
    fn add(&self, id: &str, vector: VectorInput<'_>, metadata: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let vec = vector.to_vec()?;
        let meta = metadata.map(pythonize_dict).transpose()?;

        self.inner.add(id, vec, meta)
            .map_err(|e| PyValueError::new_err(e))
    }

    /// Batch insert.
    #[pyo3(signature = (ids, vectors, metadatas = None))]
    fn add_many(
        &self,
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

        self.inner.add_many(items)
            .map_err(|e| PyValueError::new_err(e))
    }

    /// Search for the k nearest neighbors.
    #[pyo3(signature = (vector, k = 10, ef_search = None))]
    fn search(&self, vector: VectorInput<'_>, k: usize, ef_search: Option<usize>) -> PyResult<Vec<(String, f32, Option<PyObject>)>> {
        let vec = vector.to_vec()?;
        let results = self.inner.search(&vec, k, ef_search)
            .map_err(|e| PyValueError::new_err(e))?;

        Python::with_gil(|py| {
            results.into_iter()
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

    /// Delete a vector by id.
    fn delete(&self, id: &str) -> PyResult<bool> {
        self.inner.delete(id).map_err(|e| PyValueError::new_err(e))
    }

    /// Update a vector's data and/or metadata.
    #[pyo3(signature = (id, vector = None, metadata = None))]
    fn update(
        &self,
        id: &str,
        vector: Option<VectorInput<'_>>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let vec = vector.map(|v| v.to_vec()).transpose()?;
        let meta = metadata.map(|d| pythonize_dict(d).map(Some)).transpose()?;

        self.inner.update(id, vec, meta)
            .map_err(|e| PyValueError::new_err(e))
    }

    fn __contains__(&self, id: &str) -> bool {
        self.inner.contains(id)
    }

    fn save(&self) -> PyResult<()> {
        self.inner.save().map_err(|e| PyValueError::new_err(e))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }
}

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
    Ok(())
}
