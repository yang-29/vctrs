"""Benchmark quantized search vs full-precision search.

Measures:
  - Search latency (with and without quantized search)
  - Recall@k (how many of the true top-k are found)
  - Disk usage (vectors.bin vs vectors.sq8)
  - Memory estimate (f32 vs u8 vector storage)
"""
import time
import tempfile
import os
import numpy as np


def bench(name, fn, iterations=1):
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    median = sorted(times)[len(times) // 2]
    print(f"  {name}: {median*1000:.3f}ms (median of {iterations})")
    return median, result


def recall_at_k(results, ground_truth, k):
    """Fraction of true top-k found in results."""
    found = set(r.id for r in results[:k])
    truth = set(ground_truth[:k])
    return len(found & truth) / k


def main():
    from vctrs import Database

    dim = 384
    n = 50_000
    k_values = [1, 10, 100]
    n_queries = 50

    print(f"Quantized Search Benchmark | dim={dim}, n={n:,}")
    print(f"{'='*70}")

    # Generate random vectors and queries.
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)
    ids = [f"v{i}" for i in range(n)]

    # --- Build database ---
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bench")
        db = Database(path, dim=dim, metric="cosine")
        db.add_many(ids, vectors)

        # --- Ground truth (brute-force via numpy) ---
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vectors_normed = vectors / norms

        def brute_force_topk(query, k):
            qn = query / np.linalg.norm(query)
            sims = vectors_normed @ qn
            top_idx = np.argpartition(-sims, k)[:k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            return [f"v{i}" for i in top_idx]

        # --- Full-precision search ---
        print(f"\nFull-precision search (f32):")
        fp_results = {}
        fp_recalls = {}
        for k in k_values:
            def search_fp(k=k):
                return [db.search(q, k=k) for q in queries]
            t, results = bench(f"search k={k} ({n_queries} queries)", search_fp, iterations=5)
            fp_results[k] = (t, results)

            # Compute recall.
            recalls = []
            for i, q in enumerate(queries):
                gt = brute_force_topk(q, k)
                recalls.append(recall_at_k(results[i], gt, k))
            fp_recalls[k] = np.mean(recalls)
            print(f"    recall@{k}: {fp_recalls[k]:.4f}")

        # --- Enable quantized search ---
        db.enable_quantized_search()
        assert db.quantized_search

        print(f"\nQuantized search (SQ8 traversal + f32 re-rank):")
        qs_results = {}
        qs_recalls = {}
        for k in k_values:
            def search_qs(k=k):
                return [db.search(q, k=k) for q in queries]
            t, results = bench(f"search k={k} ({n_queries} queries)", search_qs, iterations=5)
            qs_results[k] = (t, results)

            recalls = []
            for i, q in enumerate(queries):
                gt = brute_force_topk(q, k)
                recalls.append(recall_at_k(results[i], gt, k))
            qs_recalls[k] = np.mean(recalls)
            print(f"    recall@{k}: {qs_recalls[k]:.4f}")

        # --- Disk usage ---
        db.save()

        # Save with quantization for disk comparison.
        path_q = os.path.join(tmp, "bench_q")
        db_q = Database(path_q, dim=dim, metric="cosine", quantize=True)
        db_q.add_many(ids, vectors)
        db_q.save()

        vec_size = os.path.getsize(os.path.join(path, "vectors.bin"))
        graph_size = os.path.getsize(os.path.join(path, "graph.vctrs"))
        sq8_size = os.path.getsize(os.path.join(path_q, "vectors.sq8"))

        # --- Memory estimate ---
        f32_mem = n * dim * 4  # bytes
        u8_mem = n * dim * 1   # bytes

        # --- Summary ---
        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY (dim={dim}, n={n:,}, {n_queries} queries)")
        print(f"{'='*70}")

        print(f"\nSearch latency (median):")
        print(f"  {'k':>6}  {'f32':>12}  {'SQ8+rerank':>12}  {'speedup':>10}")
        for k in k_values:
            fp_t = fp_results[k][0] * 1000
            qs_t = qs_results[k][0] * 1000
            speedup = fp_t / qs_t if qs_t > 0 else float('inf')
            print(f"  {k:>6}  {fp_t:>10.3f}ms  {qs_t:>10.3f}ms  {speedup:>9.2f}x")

        print(f"\nRecall:")
        print(f"  {'k':>6}  {'f32':>12}  {'SQ8+rerank':>12}  {'delta':>10}")
        for k in k_values:
            delta = qs_recalls[k] - fp_recalls[k]
            print(f"  {k:>6}  {fp_recalls[k]:>11.4f}  {qs_recalls[k]:>11.4f}  {delta:>+9.4f}")

        print(f"\nStorage:")
        print(f"  vectors.bin (f32):  {vec_size / 1024 / 1024:.1f} MB")
        print(f"  vectors.sq8 (u8):   {sq8_size / 1024 / 1024:.1f} MB  ({vec_size / sq8_size:.1f}x smaller)")
        print(f"  graph.vctrs:        {graph_size / 1024 / 1024:.1f} MB")

        print(f"\nVector memory footprint:")
        print(f"  f32 vectors: {f32_mem / 1024 / 1024:.1f} MB")
        print(f"  u8 vectors:  {u8_mem / 1024 / 1024:.1f} MB  ({f32_mem / u8_mem:.0f}x smaller)")
        print(f"  Both (for quantized search): {(f32_mem + u8_mem) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
