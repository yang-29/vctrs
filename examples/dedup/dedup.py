"""Find near-duplicate items in a dataset using vector similarity.

This is where vctrs shines — batch search in tight loops. Comparing every
item against every other item would be O(n²). With HNSW we get O(n log n).

Usage:
    pip install vctrs sentence-transformers
    python dedup.py
"""

from sentence_transformers import SentenceTransformer
from vctrs import Database

ITEMS = [
    "How to reset my password",
    "I forgot my password, how do I reset it",
    "Password reset instructions",
    "How to change my email address",
    "Update my email address",
    "I want to change the email on my account",
    "What are your business hours",
    "When are you open",
    "What time do you close",
    "How do I cancel my subscription",
    "Cancel my account",
    "I want to stop my subscription",
    "Do you offer refunds",
    "Can I get my money back",
    "Refund policy",
    "How to contact support",
    "I need help from a human",
    "The app crashes when I open settings",
    "Settings page is broken, app crashes",
]


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(ITEMS)
    dim = vectors.shape[1]

    db = Database("./dedup_db", dim=dim, metric="cosine")
    ids = [f"item{i}" for i in range(len(ITEMS))]
    metadata = [{"text": item} for item in ITEMS]
    db.add_many(ids, vectors, metadata)

    # Find clusters of near-duplicates.
    threshold = 0.3  # cosine distance — lower means more similar
    seen = set()
    clusters = []

    for i, item in enumerate(ITEMS):
        if i in seen:
            continue
        results = db.search(vectors[i], k=5)
        cluster = []
        for id, dist, meta in results:
            j = int(id.replace("item", ""))
            if j != i and dist < threshold and j not in seen:
                cluster.append((j, dist))
                seen.add(j)
        if cluster:
            seen.add(i)
            clusters.append((i, cluster))

    print(f"Found {len(clusters)} duplicate clusters:\n")
    for i, dupes in clusters:
        print(f'  "{ITEMS[i]}"')
        for j, dist in dupes:
            print(f'    ≈ "{ITEMS[j]}" (distance: {dist:.3f})')
        print()


if __name__ == "__main__":
    main()
