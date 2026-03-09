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

## Collections

Manage multiple isolated indexes under one directory:

```python
from vctrs import Client

client = Client("./data")

# Create collections with different configs
movies = client.create_collection("movies", dim=384)
docs = client.create_collection("docs", dim=768, metric="dot")

# Use them like regular databases
movies.add("m1", vector, {"title": "Alien"})
movies.save()

# List, get, delete
client.list_collections()        # → ["docs", "movies"]
db = client.get_collection("movies")
db = client.get_or_create_collection("items", dim=384)
client.delete_collection("docs")
```

## Export / Import

Backup and restore databases as JSON:

```python
# Export to file
db.export_json("backup.json")
db.export_json("backup.json", pretty=True)  # human-readable

# Import into existing database (upsert semantics)
db.import_json("backup.json")
```

The JSON format includes dimension, metric, and all vectors with metadata.
