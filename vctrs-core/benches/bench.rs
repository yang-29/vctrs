use std::time::Instant;
use vctrs_core::db::Database;
use vctrs_core::distance::Metric;

fn gen_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            (0..dim)
                .map(|d| {
                    let h = ((i as u64).wrapping_mul(2654435761) ^ (d as u64).wrapping_mul(40503))
                        % 100000;
                    h as f32 / 100000.0
                })
                .collect()
        })
        .collect()
}

fn bench_insert(n: usize, dim: usize) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench");
    let db = Database::open_or_create(path.to_str().unwrap(), dim, Metric::Cosine).unwrap();

    let vectors = gen_vectors(n, dim);
    let ids: Vec<String> = (0..n).map(|i| format!("v{}", i)).collect();
    let items: Vec<(String, Vec<f32>, Option<serde_json::Value>)> = ids
        .into_iter()
        .zip(vectors)
        .map(|(id, vec)| (id, vec, None))
        .collect();

    let start = Instant::now();
    db.add_many(items).unwrap();
    let elapsed = start.elapsed();

    println!(
        "  insert {}x{}: {:.2}ms ({:.0} vec/s)",
        n,
        dim,
        elapsed.as_secs_f64() * 1000.0,
        n as f64 / elapsed.as_secs_f64()
    );
}

fn bench_search(n: usize, dim: usize, k: usize) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench");
    let db = Database::open_or_create(path.to_str().unwrap(), dim, Metric::Cosine).unwrap();

    let vectors = gen_vectors(n, dim);
    let ids: Vec<String> = (0..n).map(|i| format!("v{}", i)).collect();
    let items: Vec<(String, Vec<f32>, Option<serde_json::Value>)> = ids
        .into_iter()
        .zip(vectors.clone())
        .map(|(id, vec)| (id, vec, None))
        .collect();
    db.add_many(items).unwrap();

    let query = &vectors[0];
    let n_queries = 1000;

    let start = Instant::now();
    for _ in 0..n_queries {
        db.search(query, k, None, None, None).unwrap();
    }
    let elapsed = start.elapsed();

    println!(
        "  search {}x{} k={}: {:.3}ms/query ({:.0} qps)",
        n,
        dim,
        k,
        elapsed.as_secs_f64() * 1000.0 / n_queries as f64,
        n_queries as f64 / elapsed.as_secs_f64()
    );
}

fn bench_batch_search(n: usize, dim: usize, k: usize, n_queries: usize) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench");
    let db = Database::open_or_create(path.to_str().unwrap(), dim, Metric::Cosine).unwrap();

    let vectors = gen_vectors(n, dim);
    let ids: Vec<String> = (0..n).map(|i| format!("v{}", i)).collect();
    let items: Vec<(String, Vec<f32>, Option<serde_json::Value>)> = ids
        .into_iter()
        .zip(vectors.clone())
        .map(|(id, vec)| (id, vec, None))
        .collect();
    db.add_many(items).unwrap();

    let queries: Vec<&[f32]> = vectors.iter().take(n_queries).map(|v| v.as_slice()).collect();

    let start = Instant::now();
    db.search_many(&queries, k, None, None, None).unwrap();
    let elapsed = start.elapsed();

    println!(
        "  search_many {}x{} k={} batch={}: {:.3}ms/query ({:.0} qps)",
        n,
        dim,
        k,
        n_queries,
        elapsed.as_secs_f64() * 1000.0 / n_queries as f64,
        n_queries as f64 / elapsed.as_secs_f64()
    );
}

fn bench_save_load(n: usize, dim: usize) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench");
    let db = Database::open_or_create(path.to_str().unwrap(), dim, Metric::Cosine).unwrap();

    let vectors = gen_vectors(n, dim);
    let ids: Vec<String> = (0..n).map(|i| format!("v{}", i)).collect();
    let items: Vec<(String, Vec<f32>, Option<serde_json::Value>)> = ids
        .into_iter()
        .zip(vectors)
        .map(|(id, vec)| (id, vec, None))
        .collect();
    db.add_many(items).unwrap();

    let start = Instant::now();
    db.save().unwrap();
    let save_elapsed = start.elapsed();

    drop(db);

    let start = Instant::now();
    let db2 = Database::open(path.to_str().unwrap()).unwrap();
    let load_elapsed = start.elapsed();

    assert_eq!(db2.len(), n);

    println!(
        "  save {}x{}: {:.2}ms | load: {:.2}ms",
        n,
        dim,
        save_elapsed.as_secs_f64() * 1000.0,
        load_elapsed.as_secs_f64() * 1000.0,
    );
}

fn main() {
    println!("vctrs benchmarks");
    println!("================");
    println!();

    println!("Insert:");
    bench_insert(1_000, 384);
    bench_insert(10_000, 384);
    bench_insert(10_000, 768);
    println!();

    println!("Search (single query):");
    bench_search(1_000, 384, 10);
    bench_search(10_000, 384, 10);
    bench_search(10_000, 768, 10);
    println!();

    println!("Search (batch):");
    bench_batch_search(10_000, 384, 10, 100);
    bench_batch_search(10_000, 768, 10, 100);
    println!();

    println!("Save / Load:");
    bench_save_load(10_000, 384);
    bench_save_load(10_000, 768);
}
