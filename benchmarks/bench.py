"""Benchmark vctrs vs ChromaDB vs numpy brute-force."""
import time
import tempfile
import os
import numpy as np


def bench(name, fn, iterations=1):
    start = time.perf_counter()
    for _ in range(iterations):
        result = fn()
    elapsed = (time.perf_counter() - start) / iterations
    print(f"  {name}: {elapsed*1000:.2f}ms")
    return elapsed


def main():
    from vctrs import Database
    import chromadb

    dim = 384
    n = 10_000
    vectors = np.random.rand(n, dim).astype(np.float32)
    query = np.random.rand(dim).astype(np.float32)
    ids = [f"v{i}" for i in range(n)]

    print(f"Benchmark | dim={dim}, n={n:,}")
    print(f"{'='*60}")

    k_values = [10, 100, 500]

    # --- vctrs ---
    print(f"\nvctrs:")
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(os.path.join(tmp, "bench"), dim=dim, metric="cosine")
        vctrs_insert = bench("insert 10k", lambda: db.add_many(ids, vectors))
        vctrs_times = {}
        for k in k_values:
            vctrs_times[k] = bench(f"search k={k}", lambda k=k: db.search(query, k=k), iterations=50)
        bench("get by id", lambda: db.get("v42"), iterations=500)
        bench("save to disk", lambda: db.save())
        path = os.path.join(tmp, "bench")
        bench("load from disk", lambda: Database(path, dim=dim, metric="cosine"))

    # --- ChromaDB ---
    print(f"\nChromaDB:")
    with tempfile.TemporaryDirectory() as tmp:
        client = chromadb.PersistentClient(path=os.path.join(tmp, "chroma"))
        collection = client.create_collection("bench", metadata={"hnsw:space": "cosine"})

        def chroma_insert():
            for i in range(0, n, 5000):
                end = min(i + 5000, n)
                collection.add(ids=ids[i:end], embeddings=vectors[i:end].tolist())

        chroma_insert_time = bench("insert 10k", chroma_insert)

        query_list = query.tolist()
        chroma_times = {}
        for k in k_values:
            chroma_times[k] = bench(
                f"search k={k}",
                lambda k=k: collection.query(query_embeddings=[query_list], n_results=k),
                iterations=50,
            )

    # --- numpy brute-force ---
    print(f"\nnumpy brute-force:")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    vectors_normed = vectors / norms
    query_normed = query / np.linalg.norm(query)

    bf_times = {}
    for k in k_values:
        def brute_force(k=k):
            sims = vectors_normed @ query_normed
            return np.argpartition(-sims, k)[:k]
        bf_times[k] = bench(f"search k={k}", brute_force, iterations=50)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"RESULTS (dim={dim}, n={n:,})")
    print(f"{'='*60}")

    print(f"\nSearch latency:")
    print(f"  {'k':>6}  {'vctrs':>10}  {'ChromaDB':>10}  {'numpy bf':>10}  {'vs Chroma':>10}  {'vs bf':>10}")
    for k in k_values:
        v = vctrs_times[k] * 1000
        c = chroma_times[k] * 1000
        b = bf_times[k] * 1000
        print(f"  {k:>6}  {v:>9.2f}ms  {c:>9.2f}ms  {b:>9.2f}ms  {c/v:>9.1f}x  {b/v:>9.1f}x")

    print(f"\nInsert 10k:")
    print(f"  vctrs:    {vctrs_insert*1000:.0f}ms")
    print(f"  ChromaDB: {chroma_insert_time*1000:.0f}ms  ({chroma_insert_time/vctrs_insert:.1f}x slower)")


if __name__ == "__main__":
    main()
