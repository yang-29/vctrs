# vctrs

A fast embedded vector database. Rust core with Python and Node bindings.

Vector search as a library, not a service. No server, no config. `pip install vctrs` and go.

## Why vctrs

ChromaDB, Pinecone, and Qdrant are databases you run. vctrs is a library you embed. Use it when:

- You're building a **tool or library** that needs vector search without inflicting a 200MB dependency tree on users
- You need **high-throughput batch search** — deduplication, clustering, similarity joins in tight loops
- You want **sub-millisecond search** without the overhead of client-server round trips
- You're building in **Rust** and need a native vector search crate

## Install

```bash
pip install vctrs        # Python
npm install @yang-29/vctrs  # Node
```

## Python

```python
from vctrs import Database

db = Database("./mydb", dim=384, metric="cosine")

db.add("doc1", vector, {"title": "hello"})
db.add_many(ids, vectors)              # batch insert (parallel HNSW)

results = db.search(query_vector, k=10)  # → [(id, distance, metadata), ...]

db.upsert("doc1", new_vector, metadata)
db.delete("doc1")
db.get("doc1")                          # → (vector, metadata)
"doc1" in db                            # → True
db.save()                               # persist to disk
```

## Node

```javascript
const { VctrsDatabase } = require("@yang-29/vctrs");

const db = new VctrsDatabase("./mydb", 384, "cosine");

db.add("doc1", vector, { title: "hello" });
db.addMany(ids, vectors);

const results = db.search(queryVector, 10); // → [{ id, distance, metadata }, ...]

db.save();
```

Metrics: `"cosine"` (default), `"euclidean"`, `"dot"`.

## Performance

10,000 vectors, 384 dimensions, cosine, Apple M-series:

| | vctrs | ChromaDB | numpy |
|---|---|---|---|
| Search k=10 | **0.14ms** | 0.93ms | 0.16ms |
| Insert 10k | **518ms** | 2367ms | — |
| Load from disk | **1.2ms** | 1.6ms | — |

**6-7x faster than ChromaDB.** Matches raw numpy. Uses Apple Accelerate / OpenBLAS for batch distance computation, mmap for instant loads.

<details>
<summary>How it works</summary>

- HNSW index with flat contiguous vector storage for cache locality
- Auto brute-force for small datasets (100% recall, BLAS-accelerated)
- Memory-mapped vectors — zero-copy load, OS-managed paging
- SimSIMD for per-vector SIMD (ARM NEON, x86 AVX2/512)
- Rayon for parallel index construction
- PyO3 + maturin for zero-copy Python/numpy bindings

</details>

## Rust

```toml
[dependencies]
vctrs-core = "0.1"
```

```rust
use vctrs_core::db::Database;
use vctrs_core::distance::Metric;

let db = Database::open_or_create("./mydb", 384, Metric::Cosine)?;
db.add("doc1", embedding, Some(json!({"title": "hello"})))?;
let results = db.search(&query, 10, None, None)?;
```

## Examples

See [`examples/`](./examples) for complete applications:

- **[Semantic Search](./examples/semantic-search/)** — search over documents with sentence embeddings
- **[Deduplication](./examples/dedup/)** — find near-duplicate items in a dataset
- **[RAG](./examples/rag/)** — retrieval-augmented generation with local LLM

## Build from source

```bash
git clone https://github.com/yang-29/vctrs.git && cd vctrs
python -m venv .venv && source .venv/bin/activate
pip install numpy maturin
maturin develop --release
```

## License

MIT
