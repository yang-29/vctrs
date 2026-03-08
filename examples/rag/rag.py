"""Minimal RAG (Retrieval-Augmented Generation) with vctrs + OpenAI.

Index documents, retrieve relevant context, generate an answer.

Usage:
    pip install vctrs sentence-transformers openai
    export OPENAI_API_KEY=sk-...
    python rag.py "What is HNSW?"
"""

import os
import sys
from sentence_transformers import SentenceTransformer
from vctrs import Database

# Chunks of text to index. In a real app these come from PDFs, web pages, etc.
CHUNKS = [
    "HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for approximate nearest neighbor search. It builds a multi-layer graph where each layer is progressively sparser. Search starts at the top layer and greedily descends.",
    "Cosine similarity measures the angle between two vectors. It ranges from -1 (opposite) to 1 (identical). Cosine distance is 1 minus cosine similarity, so 0 means identical.",
    "Vector embeddings represent text, images, or other data as dense floating-point vectors. Similar items have similar embeddings. Models like sentence-transformers produce 384-dimensional embeddings.",
    "Brute-force search compares the query against every vector in the dataset. It gives perfect recall but is O(n). For large datasets, approximate methods like HNSW are much faster.",
    "Memory-mapped files let the OS manage which pages of a file are in RAM. This enables working with datasets larger than available memory. The OS pages data in and out as needed.",
    "RAG (Retrieval-Augmented Generation) combines a retriever (vector search) with a generator (LLM). The retriever finds relevant documents, which are passed as context to the LLM to generate accurate answers.",
]


def build_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()
    db = Database("./rag_db", dim=dim, metric="cosine")
    if len(db) == 0:
        print("Indexing chunks...")
        vectors = model.encode(CHUNKS)
        ids = [f"chunk{i}" for i in range(len(CHUNKS))]
        metadata = [{"text": chunk} for chunk in CHUNKS]
        db.add_many(ids, vectors, metadata)
        db.save()

    return db, model


def retrieve(db, model, query, k=3):
    query_vec = model.encode(query)
    results = db.search(query_vec, k=k)
    return [meta["text"] for _, _, meta in results]
    

def generate(query, context_chunks):
    # Try OpenAI. If no API key, just print the retrieved context.
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("(No OPENAI_API_KEY set — showing retrieved context only)\n")
        print(f"Query: {query}\n")
        print("Retrieved context:")
        for i, chunk in enumerate(context_chunks, 1):
            print(f"  {i}. {chunk}")
        return

    from openai import OpenAI
    client = OpenAI()

    context = "\n\n".join(context_chunks)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Answer based on this context:\n\n{context}"},
            {"role": "user", "content": query},
        ],
    )
    print(f"Q: {query}\n")
    print(f"A: {response.choices[0].message.content}")


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is HNSW and how does it work?"
    db, model = build_index()
    chunks = retrieve(db, model, query)
    generate(query, chunks)


if __name__ == "__main__":
    main()
