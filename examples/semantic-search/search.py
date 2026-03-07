"""Semantic search over a collection of texts.

Usage:
    pip install vctrs sentence-transformers
    python search.py
"""

import sys
from sentence_transformers import SentenceTransformer
from vctrs import Database

DOCS = [
    "Python is a high-level programming language known for its readability",
    "Rust provides memory safety without garbage collection",
    "JavaScript runs in the browser and on the server with Node.js",
    "PostgreSQL is a powerful open-source relational database",
    "Redis is an in-memory data store used for caching",
    "Docker containers package applications with their dependencies",
    "Kubernetes orchestrates container deployment at scale",
    "Git tracks changes in source code during development",
    "Linux is an open-source operating system kernel",
    "TCP/IP is the fundamental protocol suite of the internet",
    "Machine learning models learn patterns from training data",
    "Neural networks are inspired by biological brain structure",
    "GraphQL provides a query language for APIs",
    "WebAssembly enables near-native performance in web browsers",
    "SSH provides secure remote access to servers",
]


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()

    db = Database("./search_db", dim=dim, metric="cosine")

    if len(db) == 0:
        print(f"Indexing {len(DOCS)} documents...")
        vectors = model.encode(DOCS)
        ids = [f"doc{i}" for i in range(len(DOCS))]
        metadata = [{"text": doc} for doc in DOCS]
        db.add_many(ids, vectors, metadata)
        db.save()
        print("Done.\n")

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "database systems"
    print(f'Search: "{query}"\n')

    query_vec = model.encode(query)
    results = db.search(query_vec, k=5)

    for rank, (id, dist, meta) in enumerate(results, 1):
        print(f"  {rank}. [{dist:.3f}] {meta['text']}")


if __name__ == "__main__":
    main()
