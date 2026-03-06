"""Compare search quality: vctrs vs ChromaDB vs brute-force ground truth."""
import tempfile
import os
import numpy as np
from vctrs import Database
import chromadb


def brute_force_cosine(vectors, query, k):
    """Ground truth: exact nearest neighbors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    sims = (vectors / norms) @ (query / np.linalg.norm(query))
    top_k = np.argsort(-sims)[:k]
    return set(top_k)


def recall_at_k(retrieved_ids, true_ids):
    """What fraction of true top-k did we find?"""
    return len(retrieved_ids & true_ids) / len(true_ids)


def main():
    dims = [128, 384, 768]
    ns = [1_000, 10_000]
    ks = [1, 10, 50, 100]
    num_queries = 50

    for dim in dims:
        for n in ns:
            print(f"\n{'='*60}")
            print(f"dim={dim}, n={n:,}, {num_queries} queries")
            print(f"{'='*60}")

            vectors = np.random.rand(n, dim).astype(np.float32)
            queries = np.random.rand(num_queries, dim).astype(np.float32)
            ids = [f"v{i}" for i in range(n)]

            # Build vctrs
            with tempfile.TemporaryDirectory() as tmp:
                vdb = Database(os.path.join(tmp, "vctrs"), dim=dim, metric="cosine")
                vdb.add_many(ids, vectors)

                # Build ChromaDB
                client = chromadb.Client()
                col = client.create_collection(
                    f"bench_{dim}_{n}",
                    metadata={"hnsw:space": "cosine"},
                )
                for i in range(0, n, 5000):
                    end = min(i + 5000, n)
                    col.add(ids=ids[i:end], embeddings=vectors[i:end].tolist())

                print(f"\n  {'k':>5}  {'vctrs recall':>14}  {'chroma recall':>14}  {'agree':>8}")

                for k in ks:
                    vctrs_recalls = []
                    chroma_recalls = []
                    agree_count = 0

                    for q in range(num_queries):
                        query = queries[q]
                        truth = brute_force_cosine(vectors, query, k)

                        # vctrs
                        vctrs_results = vdb.search(query, k=k)
                        vctrs_ids = set(int(r[0][1:]) for r in vctrs_results)
                        vctrs_recalls.append(recall_at_k(vctrs_ids, truth))

                        # ChromaDB
                        chroma_results = col.query(
                            query_embeddings=[query.tolist()], n_results=k
                        )
                        chroma_ids = set(int(x[1:]) for x in chroma_results["ids"][0])
                        chroma_recalls.append(recall_at_k(chroma_ids, truth))

                        # Do they agree with each other?
                        if vctrs_ids == chroma_ids:
                            agree_count += 1

                    vr = np.mean(vctrs_recalls)
                    cr = np.mean(chroma_recalls)
                    agree = agree_count / num_queries
                    print(f"  {k:>5}  {vr:>13.1%}  {cr:>13.1%}  {agree:>7.0%}")


if __name__ == "__main__":
    main()
