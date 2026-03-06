# vctrs

A fast embedded vector database. Rust core, Python API.

Built for apps that need vector search without running a separate server — RAG pipelines, desktop apps, CLI tools, notebooks.

## Install

```bash
pip install vctrs
```

## Usage

```python
import numpy as np
from vctrs import Database

# Create or open a database
db = Database("./mydb", dim=384, metric="cosine")

# Add vectors (accepts lists or numpy arrays)
db.add("doc1", np.random.rand(384).astype(np.float32), {"title": "hello"})

# Batch insert (much faster)
ids = [f"doc{i}" for i in range(10000)]
vectors = np.random.rand(10000, 384).astype(np.float32)
db.add_many(ids, vectors)

# Search
results = db.search(query_vector, k=10)
for id, distance, metadata in results:
    print(f"{id}: {distance:.4f}")

# Update
db.update("doc1", vector=new_vector, metadata={"title": "updated"})

# Delete
db.delete("doc1")

# Check membership
"doc1" in db
len(db)

# Persist to disk (graph structure saved — instant reload)
db.save()
```

### Metrics

- `"cosine"` (default) — cosine distance
- `"euclidean"` / `"l2"` — squared L2 distance
- `"dot"` / `"dot_product"` — negative dot product

### Tuning search

```python
# ef_search controls recall vs speed. Higher = better recall, slower.
results = db.search(query, k=10, ef_search=300)
```

## Benchmarks

100,000 vectors, 384 dimensions, Apple M-series:

### Speed

| Operation | vctrs | ChromaDB | numpy brute-force |
|-----------|-------|----------|-------------------|
| **Insert 100k** | 39s | 70s | — |
| **Search k=10** | **0.89ms** | 2.04ms | 3.44ms |
| **Search k=100** | **0.72ms** | 2.49ms | — |
| **Load from disk** | 323ms | — | — |
| **Get by id** | <0.01ms | — | — |

3-4x faster than ChromaDB on search, 4x faster than numpy brute-force.

### Recall (search quality)

Measured against brute-force ground truth at 10k vectors (higher is better):

| k | vctrs | ChromaDB |
|---|-------|----------|
| 1 | **92%** | 76% |
| 10 | **91%** | 76% |
| 50 | **85%** | 68% |

vctrs is both faster and more accurate than ChromaDB out of the box.

## How it works

- **HNSW** index for O(log n) approximate nearest neighbor search
- **SimSIMD** for hardware-accelerated distance computation (ARM NEON, x86 AVX2/512)
- **Rayon** for parallel index construction
- **Graph serialization** — saves the full HNSW structure so loading is instant (no rebuild)
- **PyO3** + **maturin** for zero-copy Python bindings with numpy support

## Building from source

```bash
pip install maturin
git clone https://github.com/yang-29/vctrs.git
cd vctrs
python -m venv .venv && source .venv/bin/activate
pip install numpy maturin
maturin develop --release
```

## License

MIT
