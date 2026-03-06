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
results = db.search(query, k=10, ef_search=200)
```

## Benchmarks

10,000 vectors, 384 dimensions (typical embedding size):

| Operation | vctrs | ChromaDB | Speedup |
|-----------|-------|----------|---------|
| Insert 10k | 568ms | 2,610ms | **4.6x** |
| Search k=10 | 0.12ms | 1.16ms | **9.8x** |
| Search k=100 | 0.31ms | 1.27ms | **4.1x** |
| Search k=500 | 0.67ms | 4.28ms | **6.4x** |
| Load from disk | 18ms | — | instant |
| Get by id | <0.01ms | 0.12ms | **~100x** |

## How it works

- **HNSW** index for O(log n) approximate nearest neighbor search
- **SimSIMD** for hardware-accelerated distance computation (ARM NEON, x86 AVX2/512)
- **Rayon** for parallel index construction
- **Graph serialization** — saves the full HNSW structure so loading is instant (no rebuild)
- **PyO3** + **maturin** for zero-copy Python bindings with numpy support

## Building from source

```bash
# Rust + Python
pip install maturin
git clone https://github.com/yang-29/vctrs.git
cd vctrs
python -m venv .venv && source .venv/bin/activate
pip install numpy maturin
maturin develop --release
```

## License

MIT
