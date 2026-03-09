# vctrs

A fast embedded vector database. Rust core with Python, Node.js, and WebAssembly bindings.

Vector search as a library, not a service. No server, no config. `pip install vctrs` and go.

## Why vctrs

ChromaDB, Pinecone, and Qdrant are databases you run. vctrs is a library you embed. Use it when:

- You're building a **tool or library** that needs vector search without inflicting a 200MB dependency tree on users
- You need **high-throughput batch search** — deduplication, clustering, similarity joins in tight loops
- You want **sub-millisecond search** without the overhead of client-server round trips
- You're building in **Rust** and need a native vector search crate

## Install

```bash
pip install vctrs              # Python
npm install @yang-29/vctrs     # Node
npm install @yang-29/vctrs-wasm  # Browser (WASM)
```

## Python

```python
from vctrs import Database

db = Database("./mydb", dim=384, metric="cosine")

# CRUD
db.add("doc1", vector, {"title": "hello"})
db.upsert("doc1", new_vector, {"title": "updated"})
db.update("doc1", metadata={"title": "changed"})   # update metadata only
db.update("doc1", vector=new_vector)                # update vector only
db.delete("doc1")
db.get("doc1")                          # → (vector, metadata)
"doc1" in db                            # → True

# Batch insert (parallel HNSW construction)
db.add_many(ids, vectors, metadatas)

# Search
results = db.search(query_vector, k=10)  # → [SearchResult(id, distance, metadata), ...]
batch = db.search_many(query_vectors, k=10)  # parallel multi-query search

# Filtered search
results = db.search(query, k=10, where_filter={"category": "science"})
results = db.search(query, k=10, where_filter={"category": {"$ne": "sports"}})
results = db.search(query, k=10, where_filter={"category": {"$in": ["sci", "tech"]}})
results = db.search(query, k=10, where_filter={"score": {"$gte": 0.5, "$lt": 0.9}})

# Maintenance
db.compact()                # reclaim deleted vector slots
db.enable_quantized_search()  # SQ8 quantized HNSW traversal + f32 re-ranking
db.save()                   # persist to disk

# Diagnostics
stats = db.stats()          # → dict with graph metrics, memory usage, etc.

# Context manager (auto-save on exit)
with Database("./mydb", dim=384) as db:
    db.add("doc1", vector)
```

Options: `m=16` (HNSW links), `ef_construction=200` (build quality), `quantize=True` (SQ8, ~4x smaller on disk).

### Collections

```python
from vctrs import Client

client = Client("./data")

# Create isolated collections with different configs
movies = client.create_collection("movies", dim=384)
docs = client.create_collection("docs", dim=768, metric="dot")

movies.add("m1", vector, {"title": "Alien"})
movies.save()

# List, get, delete
client.list_collections()       # → ["docs", "movies"]
db = client.get_collection("movies")
client.delete_collection("docs")
```

### Export / Import

```python
# Backup to JSON
db.export_json("backup.json", pretty=True)

# Restore into an existing database (upsert semantics)
db.import_json("backup.json")
```

## Node

```javascript
const { VctrsDatabase } = require("@yang-29/vctrs");

const db = new VctrsDatabase("./mydb", 384, "cosine");

// CRUD
db.add("doc1", vector, { title: "hello" });
db.upsert("doc1", newVector, { title: "updated" });
db.update("doc1", null, { title: "changed" }); // metadata only
db.delete("doc1");
db.get("doc1"); // → { vector, metadata }
db.contains("doc1"); // → true

// Batch
db.addMany(ids, vectors, metadatas);

// Search
const results = db.search(queryVector, 10); // → [{ id, distance, metadata }, ...]
const batch = db.searchMany(queryVectors, 10); // parallel multi-query search

// Filtered search
db.search(query, 10, null, { category: "science" });
db.search(query, 10, null, { score: { $gte: 0.5, $lt: 0.9 } });

// Maintenance
db.compact();
db.enableQuantizedSearch();
db.save();

// Diagnostics
const stats = db.stats(); // → { numVectors, numDeleted, avgDegreeLayer0, ... }
```

Metrics: `"cosine"` (default), `"euclidean"` / `"l2"`, `"dot"` / `"dot_product"`.

### Collections

```javascript
const { VctrsClient } = require("@yang-29/vctrs");

const client = new VctrsClient("./data");

const movies = client.createCollection("movies", 384);
const docs = client.getOrCreateCollection("docs", 768, "dot");

client.listCollections(); // → ["docs", "movies"]
client.deleteCollection("docs");
```

### Export / Import

```javascript
db.exportJson("backup.json", true); // pretty-print
db.importJson("backup.json");       // upsert semantics
```

### Filter operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` (default) | Equals | `{ field: "value" }` |
| `$ne` | Not equals | `{ field: { $ne: "value" } }` |
| `$in` | In list | `{ field: { $in: ["a", "b"] } }` |
| `$gt` | Greater than | `{ field: { $gt: 10 } }` |
| `$gte` | Greater than or equal | `{ field: { $gte: 10 } }` |
| `$lt` | Less than | `{ field: { $lt: 20 } }` |
| `$lte` | Less than or equal | `{ field: { $lte: 20 } }` |

Multiple keys in a filter object are ANDed together.

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
- Optional scalar quantization (SQ8) for ~4x smaller on-disk storage with full-precision re-ranking
- In-graph filtered search (no over-fetch retry loop)
- Auto brute-force for small datasets (100% recall, BLAS-accelerated)
- Memory-mapped vectors — zero-copy load, OS-managed paging
- GC/compaction to reclaim soft-deleted vector slots
- SimSIMD for per-vector SIMD (ARM NEON, x86 AVX2/512)
- Rayon for parallel index construction and batch search
- PyO3 + maturin for zero-copy Python/numpy bindings

</details>

## WebAssembly

Runs in the browser — no server, no backend. ~220KB.

```javascript
import init, { VctrsDatabase } from "@yang-29/vctrs-wasm";

await init();

const db = new VctrsDatabase(384, "cosine");

db.add("doc1", new Float32Array(vector), { title: "hello" });
const results = db.search(new Float32Array(query), 10);
for (const r of results) {
  console.log(r.id, r.distance, r.metadata);
  r.free(); // free WASM memory
}
db.free();
```

In-memory only (no persistence). Call `.free()` on results and the database to avoid leaks.

## Rust

```toml
[dependencies]
vctrs-core = "0.2"
```

```rust
use vctrs_core::db::{Database, Filter, HnswConfig};
use vctrs_core::distance::Metric;

let db = Database::open_or_create("./mydb", 384, Metric::Cosine)?;
db.add("doc1", embedding, Some(json!({"title": "hello"})))?;

// Search
let results = db.search(&query, 10, None, None)?;

// Filtered search
let filter = Filter::Gte("score".into(), 0.5);
let results = db.search(&query, 10, None, Some(&filter))?;

// Batch search (parallel)
let batch = db.search_many(&[&q1, &q2], 10, None, None)?;

// Maintenance
db.compact()?;
db.enable_quantized_search();
let stats = db.stats();

// Custom HNSW config + quantization
let config = HnswConfig { m: 32, ef_construction: 400, quantize: true };
let db = Database::open_or_create_with_config("./mydb", 384, Metric::Cosine, config)?;
```

Errors are typed via `VctrsError` enum (`DimensionMismatch`, `DuplicateId`, `NotFound`, `Io`, etc.).

### Collections

```rust
use vctrs_core::client::Client;
use vctrs_core::distance::Metric;

let client = Client::new("./data")?;
let movies = client.create_collection("movies", 384, Metric::Cosine)?;
let docs = client.get_or_create_collection("docs", 768, Metric::DotProduct)?;

client.list_collections()?;       // → ["docs", "movies"]
client.delete_collection("docs")?;
```

### Export / Import

```rust
// Export
let file = std::fs::File::create("backup.json")?;
db.export_json(file)?;

// Import into new database
let file = std::fs::File::open("backup.json")?;
let db = Database::import_json(file, "./restored")?;

// Import into existing database (upsert)
let file = std::fs::File::open("backup.json")?;
db.import_json_into(file)?;
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
