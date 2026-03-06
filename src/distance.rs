/// Distance/similarity metrics for vector comparison.
/// Uses SimSIMD for hardware-accelerated SIMD operations (ARM NEON, x86 AVX2/AVX-512).
/// Falls back to scalar for very short vectors where SIMD can't operate.

use simsimd::SpatialSimilarity;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    Cosine,
    Euclidean,
    DotProduct,
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
    // simsimd cosine returns distance (1 - similarity) directly.
    match f32::cosine(a, b) {
        Some(d) => d as f32,
        None => cosine_distance_scalar(a, b),
    }
}

/// Squared euclidean distance via SIMD.
pub fn euclidean_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    match f32::sqeuclidean(a, b) {
        Some(d) => d as f32,
        None => euclidean_distance_sq_scalar(a, b),
    }
}

/// Negative dot product (lower = more similar, consistent with distance ordering).
pub fn neg_dot_product(a: &[f32], b: &[f32]) -> f32 {
    match SpatialSimilarity::dot(a, b) {
        Some(d) => -(d as f32),
        None => -dot_scalar(a, b),
    }
}

/// Compute distance between two vectors using the given metric.
pub fn distance(a: &[f32], b: &[f32], metric: Metric) -> f32 {
    match metric {
        Metric::Cosine => cosine_distance(a, b),
        Metric::Euclidean => euclidean_distance_sq(a, b),
        Metric::DotProduct => neg_dot_product(a, b),
    }
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
        // High-dimensional vectors should use SIMD path.
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
}
