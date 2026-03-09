/// Distance/similarity metrics for vector comparison.
/// Uses SimSIMD for hardware-accelerated SIMD operations (ARM NEON, x86 AVX2/AVX-512).
/// Falls back to scalar for very short vectors where SIMD can't operate.
///
/// Batch operations use platform BLAS when available:
/// - macOS: Apple Accelerate (always available)
/// - Linux: OpenBLAS/BLAS (install libopenblas-dev)
/// - Windows: OpenBLAS (install via vcpkg or conda)
/// Without BLAS, falls back to per-vector SimSIMD (still fast, but ~4x slower than BLAS).

#[cfg(feature = "simd")]
use simsimd::SpatialSimilarity;
use std::sync::atomic::{AtomicBool, Ordering};

// --- BLAS FFI (Accelerate on macOS, OpenBLAS on Linux/Windows) ---

#[cfg(has_blas)]
extern "C" {
    fn cblas_sgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );
}

/// Returns true if this build was compiled with BLAS support.
pub fn has_blas() -> bool {
    cfg!(has_blas)
}

/// One-time flag: have we already printed the BLAS hint?
static BLAS_HINT_SHOWN: AtomicBool = AtomicBool::new(false);

/// Print a one-time hint about installing BLAS for better performance.
/// Called from brute-force search path when BLAS is not available.
pub fn maybe_print_blas_hint() {
    if cfg!(has_blas) {
        return;
    }
    if BLAS_HINT_SHOWN.swap(true, Ordering::Relaxed) {
        return; // already shown
    }
    let msg = if cfg!(target_os = "linux") {
        "vctrs: brute-force search would be ~5x faster with BLAS. Install it with:\n  \
         sudo apt install libopenblas-dev   # Debian/Ubuntu\n  \
         sudo dnf install openblas-devel    # Fedora/RHEL\n  \
         Then rebuild: pip install --force-reinstall vctrs"
    } else if cfg!(target_os = "windows") {
        "vctrs: brute-force search would be ~5x faster with BLAS. Install OpenBLAS:\n  \
         conda install -c conda-forge openblas   # or via vcpkg\n  \
         Then rebuild: pip install --force-reinstall vctrs"
    } else {
        // macOS always has Accelerate — this shouldn't be reached.
        return;
    };
    eprintln!("{}", msg);
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    Cosine,
    Euclidean,
    DotProduct,
}

impl Metric {
    /// Parse a metric name from a string.
    /// Accepts: "cosine", "euclidean"/"l2", "dot"/"dot_product".
    pub fn from_str(s: &str) -> Result<Self, crate::VctrsError> {
        match s {
            "cosine" => Ok(Metric::Cosine),
            "euclidean" | "l2" => Ok(Metric::Euclidean),
            "dot" | "dot_product" => Ok(Metric::DotProduct),
            _ => Err(crate::VctrsError::InvalidMetric(s.to_string())),
        }
    }
}

/// Scalar fallback for cosine distance.
fn cosine_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        norm_a += a[i] as f64 * a[i] as f64;
        norm_b += b[i] as f64 * b[i] as f64;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 1.0;
    }
    (1.0 - dot / denom) as f32
}

/// Scalar fallback for squared euclidean distance.
fn euclidean_distance_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    for i in 0..a.len() {
        let d = a[i] as f64 - b[i] as f64;
        sum += d * d;
    }
    sum as f32
}

/// Scalar fallback for dot product.
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
    }
    dot as f32
}

/// Cosine distance: 1 - cosine_similarity(a, b)
/// Returns 0.0 for identical directions, up to 2.0 for opposite.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "simd")]
    {
        match f32::cosine(a, b) {
            Some(d) => return d as f32,
            None => {}
        }
    }
    cosine_distance_scalar(a, b)
}

/// Squared euclidean distance via SIMD.
pub fn euclidean_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "simd")]
    {
        match f32::sqeuclidean(a, b) {
            Some(d) => return d as f32,
            None => {}
        }
    }
    euclidean_distance_sq_scalar(a, b)
}

/// Negative dot product (lower = more similar, consistent with distance ordering).
pub fn neg_dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "simd")]
    {
        match SpatialSimilarity::dot(a, b) {
            Some(d) => return -(d as f32),
            None => {}
        }
    }
    -dot_scalar(a, b)
}

/// Compute distance between two vectors using the given metric.
pub fn distance(a: &[f32], b: &[f32], metric: Metric) -> f32 {
    match metric {
        Metric::Cosine => cosine_distance(a, b),
        Metric::Euclidean => euclidean_distance_sq(a, b),
        Metric::DotProduct => neg_dot_product(a, b),
    }
}

/// Batch distance computation: compute distances from `query` to N vectors stored
/// contiguously in `vectors` (flat layout: vectors[i*dim..(i+1)*dim]).
///
/// With BLAS: uses cblas_sgemv for all dot products in one kernel call (~numpy speed).
/// Without BLAS: uses per-vector SimSIMD (~4x slower but still SIMD-accelerated).
///
/// `norms`: pre-computed L2 norms for each vector (used by cosine and euclidean).
pub fn batch_distances(
    query: &[f32],
    vectors: &[f32],
    dim: usize,
    metric: Metric,
    norms: &[f32],
) -> Vec<f32> {
    let n = vectors.len() / dim;
    match metric {
        Metric::Cosine => batch_cosine(query, vectors, dim, n, norms),
        Metric::Euclidean => batch_euclidean(query, vectors, dim, n, norms),
        Metric::DotProduct => batch_neg_dot(query, vectors, dim, n),
    }
}

// --- Dot product implementations ---

/// BLAS path: one cblas_sgemv call for all N dot products.
#[cfg(has_blas)]
fn gemv_dot_products(query: &[f32], vectors: &[f32], dim: usize, n: usize) -> Vec<f32> {
    let mut dots = vec![0.0f32; n];
    unsafe {
        cblas_sgemv(
            101,                    // CblasRowMajor
            111,                    // CblasNoTrans
            n as i32,
            dim as i32,
            1.0,
            vectors.as_ptr(),
            dim as i32,
            query.as_ptr(),
            1,
            0.0,
            dots.as_mut_ptr(),
            1,
        );
    }
    dots
}

/// Fallback: per-vector dot products (SimSIMD when available, scalar otherwise).
#[cfg(not(has_blas))]
fn gemv_dot_products(query: &[f32], vectors: &[f32], dim: usize, n: usize) -> Vec<f32> {
    let mut dots = Vec::with_capacity(n);
    for i in 0..n {
        let v = &vectors[i * dim..(i + 1) * dim];
        #[cfg(feature = "simd")]
        {
            match SpatialSimilarity::dot(query, v) {
                Some(d) => { dots.push(d as f32); continue; }
                None => {}
            }
        }
        dots.push(dot_scalar(query, v));
    }
    dots
}

// --- Batch metric implementations ---

fn batch_cosine(query: &[f32], vectors: &[f32], dim: usize, n: usize, norms: &[f32]) -> Vec<f32> {
    let dots = gemv_dot_products(query, vectors, dim, n);

    let query_norm_sq: f32 = query.iter().map(|x| x * x).sum();
    let query_norm = query_norm_sq.sqrt();
    if query_norm == 0.0 {
        return vec![1.0f32; n];
    }

    let mut dists = Vec::with_capacity(n);
    for i in 0..n {
        let v_norm = norms[i];
        if v_norm == 0.0 {
            dists.push(1.0);
        } else {
            dists.push(1.0 - dots[i] / (query_norm * v_norm));
        }
    }
    dists
}

fn batch_euclidean(query: &[f32], vectors: &[f32], dim: usize, n: usize, norms: &[f32]) -> Vec<f32> {
    // ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a·b)
    let dots = gemv_dot_products(query, vectors, dim, n);
    let query_norm_sq: f32 = query.iter().map(|x| x * x).sum();

    let mut dists = Vec::with_capacity(n);
    for i in 0..n {
        let v_norm_sq = norms[i] * norms[i];
        dists.push(query_norm_sq + v_norm_sq - 2.0 * dots[i]);
    }
    dists
}

fn batch_neg_dot(query: &[f32], vectors: &[f32], dim: usize, n: usize) -> Vec<f32> {
    let mut dots = gemv_dot_products(query, vectors, dim, n);
    for d in dots.iter_mut() {
        *d = -*d;
    }
    dots
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let d = cosine_distance(&v, &v);
        assert!(d.abs() < 1e-5, "expected ~0, got {}", d);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let d = cosine_distance(&a, &b);
        assert!((d - 2.0).abs() < 1e-5, "expected ~2, got {}", d);
    }

    #[test]
    fn test_euclidean() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let d = euclidean_distance_sq(&a, &b);
        assert!((d - 25.0).abs() < 1e-4, "expected ~25, got {}", d);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let d = neg_dot_product(&a, &b);
        assert!((d - (-32.0)).abs() < 1e-4, "expected ~-32, got {}", d);
    }

    #[test]
    fn test_cosine_high_dim() {
        let dim = 128;
        let a: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..dim).map(|i| (dim - i) as f32).collect();
        let d_simd = cosine_distance(&a, &b);
        let d_scalar = cosine_distance_scalar(&a, &b);
        assert!(
            (d_simd - d_scalar).abs() < 1e-3,
            "SIMD ({}) and scalar ({}) mismatch",
            d_simd,
            d_scalar
        );
    }

    #[test]
    fn test_batch_distances_cosine() {
        let dim = 128;
        let n = 100;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
        let mut vectors = Vec::with_capacity(n * dim);
        let mut norms = Vec::with_capacity(n);
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|j| ((i * dim + j) as f32) * 0.001).collect();
            norms.push(v.iter().map(|x| x * x).sum::<f32>().sqrt());
            vectors.extend_from_slice(&v);
        }

        let batch = batch_distances(&query, &vectors, dim, Metric::Cosine, &norms);
        assert_eq!(batch.len(), n);
        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            let single = cosine_distance(&query, v);
            assert!(
                (batch[i] - single).abs() < 1e-3,
                "mismatch at {}: batch={}, single={}",
                i, batch[i], single
            );
        }
    }

    #[test]
    fn test_batch_distances_euclidean() {
        let dim = 64;
        let n = 50;
        let query: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let mut vectors = Vec::with_capacity(n * dim);
        let mut norms = Vec::with_capacity(n);
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|j| ((i + j) as f32) * 0.5).collect();
            norms.push(v.iter().map(|x| x * x).sum::<f32>().sqrt());
            vectors.extend_from_slice(&v);
        }

        let batch = batch_distances(&query, &vectors, dim, Metric::Euclidean, &norms);
        assert_eq!(batch.len(), n);
        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            let single = euclidean_distance_sq(&query, v);
            assert!(
                (batch[i] - single).abs() < 1.0,
                "mismatch at {}: batch={}, single={}",
                i, batch[i], single
            );
        }
    }

    #[test]
    fn test_batch_distances_dot() {
        let dim = 64;
        let n = 50;
        let query: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let mut vectors = Vec::with_capacity(n * dim);
        let norms = vec![0.0f32; n]; // unused for dot
        for i in 0..n {
            let v: Vec<f32> = (0..dim).map(|j| ((i + j) as f32) * 0.1).collect();
            vectors.extend_from_slice(&v);
        }

        let batch = batch_distances(&query, &vectors, dim, Metric::DotProduct, &norms);
        assert_eq!(batch.len(), n);
        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            let single = neg_dot_product(&query, v);
            assert!(
                (batch[i] - single).abs() < 1.0,
                "mismatch at {}: batch={}, single={}",
                i, batch[i], single
            );
        }
    }
}
