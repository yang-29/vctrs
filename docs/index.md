# vctrs

A fast embedded vector database. Rust core with Python, Node.js, and WebAssembly bindings.

Vector search as a library, not a service. No server, no config.

## Install

=== "Python"

    ```bash
    pip install vctrs
    ```

=== "Node.js"

    ```bash
    npm install @yang-29/vctrs
    ```

=== "WebAssembly"

    ```bash
    npm install @yang-29/vctrs-wasm
    ```

=== "Rust"

    ```toml
    [dependencies]
    vctrs-core = "0.2"
    ```

## Quick Start

=== "Python"

    ```python
    from vctrs import Database

    db = Database("./mydb", dim=384, metric="cosine")

    db.add("doc1", vector, {"title": "hello"})
    results = db.search(query_vector, k=10)

    for r in results:
        print(r.id, r.distance, r.metadata)
    ```

=== "Node.js"

    ```typescript
    import { VctrsDatabase } from "@yang-29/vctrs";

    const db = new VctrsDatabase("./mydb", 384, "cosine");

    db.add("doc1", vector, { title: "hello" });
    const results = db.search(queryVector, 10);

    for (const r of results) {
      console.log(r.id, r.distance, r.metadata);
    }
    ```

=== "WebAssembly"

    ```javascript
    import init, { VctrsDatabase } from "@yang-29/vctrs-wasm";

    await init();
    const db = new VctrsDatabase(384, "cosine");

    db.add("doc1", new Float32Array(vector), { title: "hello" });
    const results = db.search(new Float32Array(queryVector), 10);

    for (const r of results) {
      console.log(r.id, r.distance, r.metadata);
      r.free();
    }
    ```

=== "Rust"

    ```rust
    use vctrs_core::db::Database;
    use vctrs_core::distance::Metric;

    let db = Database::open_or_create("./mydb", 384, Metric::Cosine)?;

    db.add("doc1", embedding, Some(json!({"title": "hello"})))?;
    let results = db.search(&query, 10, None, None, None)?;

    for r in &results {
        println!("{} {:.4}", r.id, r.distance);
    }
    ```

## Why vctrs

ChromaDB, Pinecone, and Qdrant are databases you run. vctrs is a library you embed.

- **Embedded** — no server, no config, no network round trips
- **Fast** — 6-7x faster than ChromaDB, matches raw numpy
- **Lightweight** — single native binary, no 200MB dependency tree
- **Crash-safe** — write-ahead log with CRC32 checksums
- **Instant loads** — memory-mapped vectors, zero-copy on startup

## Performance

10,000 vectors, 384 dimensions, cosine, Apple M-series:

| | vctrs | ChromaDB | numpy |
|---|---|---|---|
| Search k=10 | **0.14ms** | 0.93ms | 0.16ms |
| Insert 10k | **518ms** | 2367ms | — |
| Load from disk | **1.2ms** | 1.6ms | — |

## Filter Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` (default) | Equals | `{ field: "value" }` |
| `$ne` | Not equals | `{ field: { $ne: "value" } }` |
| `$in` | In list | `{ field: { $in: ["a", "b"] } }` |
| `$gt` | Greater than | `{ field: { $gt: 10 } }` |
| `$gte` | Greater or equal | `{ field: { $gte: 10 } }` |
| `$lt` | Less than | `{ field: { $lt: 20 } }` |
| `$lte` | Less or equal | `{ field: { $lte: 20 } }` |

Multiple keys in a filter are ANDed together.
