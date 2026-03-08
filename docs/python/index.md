# Python — Getting Started

## Install

```bash
pip install vctrs
```

## Basic Usage

```python
from vctrs import Database

# Create a new database (or open existing)
db = Database("./mydb", dim=384, metric="cosine")

# Add vectors
db.add("doc1", [0.1, 0.2, ...], {"title": "Hello"})

# Batch insert
db.add_many(ids, vectors, metadatas)

# Search
results = db.search(query_vector, k=10)
for r in results:
    print(r.id, r.distance, r.metadata)

# Filtered search
results = db.search(query, k=10, where_filter={"category": "science"})

# Distance threshold
results = db.search(query, k=10, max_distance=0.5)

# Save and close
db.save()
```

## Context Manager

```python
with Database("./mydb", dim=384) as db:
    db.add("doc1", vector)
    # auto-saves on exit
```

## Async Usage

```python
from vctrs import AsyncDatabase

async with AsyncDatabase("./mydb", dim=384) as db:
    await db.add("doc1", vector, {"title": "Hello"})
    results = await db.search(query_vector, k=10)
```

## Numpy Support

Vectors can be passed as Python lists or numpy arrays:

```python
import numpy as np

embedding = np.random.randn(384).astype(np.float32)
db.add("doc1", embedding)

# Batch with 2D numpy array
batch = np.random.randn(100, 384).astype(np.float32)
db.add_many([f"doc{i}" for i in range(100)], batch)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | required | Vector dimensionality |
| `metric` | `"cosine"` | `"cosine"`, `"euclidean"`/`"l2"`, `"dot"`/`"dot_product"` |
| `m` | `16` | HNSW edges per node (higher = better recall, more memory) |
| `ef_construction` | `200` | Build-time search width (higher = better index quality) |
| `quantize` | `False` | Enable SQ8 scalar quantization |
