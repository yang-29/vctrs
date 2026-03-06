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

    # --- vctrs ---
    print(f"\nvctrs:")
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(os.path.join(tmp, "bench"), dim=dim, metric="cosine")
        vctrs_insert = bench("insert 10k", lambda: db.add_many(ids, vectors))
        vctrs_search_10 = bench("search k=10", lambda: db.search(query, k=10), iterations=50)
        vctrs_search_100 = bench("search k=100", lambda: db.search(query, k=100), iterations=50)
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
        chroma_search_10 = bench(
            "search k=10",
            lambda: collection.query(query_embeddings=[query_list], n_results=10),
            iterations=50,
        )
        chroma_search_100 = bench(
            "search k=100",
            lambda: collection.query(query_embeddings=[query_list], n_results=100),
            iterations=50,
        )
        bench("get by id", lambda: collection.get(ids=["v42"]), iterations=500)

    # --- numpy brute-force ---
    print(f"\nnumpy brute-force:")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    vectors_normed = vectors / norms
    query_normed = query / np.linalg.norm(query)

    def brute_force():
        sims = vectors_normed @ query_normed
        return np.argpartition(-sims, 10)[:10]

    bf_search = bench("search k=10", brute_force, iterations=50)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"RESULTS (dim={dim}, n={n:,})")
    print(f"{'='*60}")
    print(f"\nSearch k=10:")
    print(f"  vctrs:        {vctrs_search_10*1000:.2f}ms")
    print(f"  ChromaDB:     {chroma_search_10*1000:.2f}ms  ({chroma_search_10/vctrs_search_10:.1f}x slower)")
    print(f"  numpy bf:     {bf_search*1000:.2f}ms  ({bf_search/vctrs_search_10:.1f}x slower)")
    print(f"\nSearch k=100:")
    print(f"  vctrs:        {vctrs_search_100*1000:.2f}ms")
    print(f"  ChromaDB:     {chroma_search_100*1000:.2f}ms  ({chroma_search_100/vctrs_search_100:.1f}x slower)")
    print(f"\nInsert 10k:")
    print(f"  vctrs:        {vctrs_insert*1000:.0f}ms")
    print(f"  ChromaDB:     {chroma_insert_time*1000:.0f}ms  ({chroma_insert_time/vctrs_insert:.1f}x slower)" if vctrs_insert < chroma_insert_time else f"  ChromaDB:     {chroma_insert_time*1000:.0f}ms  ({vctrs_insert/chroma_insert_time:.1f}x faster)")


if __name__ == "__main__":
    main()
