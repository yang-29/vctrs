# WebAssembly

vctrs compiles to WebAssembly, so you can run vector search directly in the browser. No server required.

The WASM build is ~220KB and runs entirely in-memory. It shares the same Rust core as the Python and Node.js bindings.

## Install

### npm

```bash
npm install @yang-29/vctrs-wasm
```

```javascript
import init, { VctrsDatabase } from "@yang-29/vctrs-wasm";

await init();
const db = new VctrsDatabase(384, "cosine");
```

### CDN (no bundler)

```html
<script type="module">
import init, { VctrsDatabase } from "https://unpkg.com/@yang-29/vctrs-wasm/vctrs_wasm.js";

await init();
const db = new VctrsDatabase(384, "cosine");
</script>
```

### Self-hosted

Build from source with [wasm-pack](https://rustwasm.github.io/wasm-pack/):

```bash
wasm-pack build wasm --target web --out-dir pkg
```

This produces a `wasm/pkg/` directory you can serve or bundle.

## Quick Start

```javascript
import init, { VctrsDatabase } from "@yang-29/vctrs-wasm";

await init();

// Create a database
const db = new VctrsDatabase(4, "cosine");

// Add vectors with metadata
db.add("cat", new Float32Array([1, 0, 0, 0]), { type: "animal" });
db.add("dog", new Float32Array([0.9, 0.1, 0, 0]), { type: "animal" });
db.add("car", new Float32Array([0, 0, 1, 0]), { type: "vehicle" });

// Search
const results = db.search(new Float32Array([1, 0, 0, 0]), 2);
for (const r of results) {
  console.log(r.id, r.distance, r.metadata);
  r.free(); // free WASM memory
}

// Clean up
db.free();
```

## API Reference

### `new VctrsDatabase(dim, metric)`

Create a new in-memory vector database.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim` | `number` | Vector dimensionality |
| `metric` | `string` | `"cosine"`, `"euclidean"` / `"l2"`, or `"dot"` / `"dot_product"` |

### `db.add(id, vector, metadata)`

Add a vector with a unique string ID.

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `string` | Unique identifier |
| `vector` | `Float32Array` | Vector of length `dim` |
| `metadata` | `object \| null` | Optional JSON metadata |

Throws if `id` already exists. Use `upsert` to insert-or-update.

### `db.upsert(id, vector, metadata)`

Insert a new vector or update an existing one.

### `db.search(vector, k, filter?, maxDistance?)`

Find the `k` nearest neighbors. Returns an array of `SearchResult` objects.

| Parameter | Type | Description |
|-----------|------|-------------|
| `vector` | `Float32Array` | Query vector of length `dim` |
| `k` | `number` | Number of results |
| `filter` | `object \| null` | Optional metadata filter (same syntax as Python/Node) |
| `maxDistance` | `number \| null` | Optional distance threshold |

### `db.addMany(ids, vectors, dim, metadatas?)`

Batch add multiple vectors. `vectors` is a flat `Float32Array` of length `ids.length * dim`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ids` | `string[]` | Array of unique IDs |
| `vectors` | `Float32Array` | Flat array of all vectors concatenated |
| `dim` | `number` | Vector dimensionality |
| `metadatas` | `object[] \| null` | Optional array of metadata objects |

### `db.get(id)`

Retrieve a vector and its metadata. Returns `{ vector: Float32Array, metadata: object | null }`.

### `db.ids()`

Return all vector IDs as an array of strings.

### `db.count(filter?)`

Count vectors, optionally with a metadata filter.

### `db.delete(id)`

Delete a vector by ID. Returns `true` if found and deleted.

### `db.contains(id)`

Check if a vector ID exists. Returns `boolean`.

### `db.length` (getter)

Number of vectors in the database.

### `db.dim` (getter)

Vector dimensionality.

### `db.export_json()`

Export all vectors and metadata as a JSON string.

```javascript
const json = db.export_json();
// Save to file, localStorage, IndexedDB, etc.
localStorage.setItem("vectors_backup", json);
```

### `db.import_json(json)`

Import vectors from a JSON string (upsert semantics).

```javascript
const json = localStorage.getItem("vectors_backup");
db.import_json(json);
```

### `SearchResult`

| Property | Type | Description |
|----------|------|-------------|
| `.id` | `string` | Vector ID |
| `.distance` | `number` | Distance to query (lower = more similar for cosine/euclidean) |
| `.metadata` | `object \| null` | Stored metadata |

!!! warning "Memory management"
    WASM objects are not garbage collected automatically. Call `.free()` on `SearchResult` objects and the `VctrsDatabase` when done to avoid memory leaks.

    ```javascript
    const results = db.search(query, 10);
    for (const r of results) {
      console.log(r.id);
      r.free();
    }
    db.free();
    ```

## Differences from Python / Node.js

The WASM build runs entirely in-memory. It does not support:

- **Persistence** — no `save()` / `load()`. Data is lost on page refresh. Use `export_json()` / `import_json()` with localStorage or IndexedDB for client-side persistence.
- **Async** — all operations are synchronous (but fast).
- **BLAS acceleration** — uses pure-Rust math (still fast for moderate dataset sizes).
- **Parallel search** — no `searchMany()` (WASM is single-threaded).

## Performance

Benchmarks from Chrome on Apple M-series (single-threaded WASM):

| Operation | Time |
|-----------|------|
| Insert 1,000 vectors (384d) | ~900ms |
| Search k=10 (1,000 vectors, 384d) | ~0.5ms |
| WASM binary size | 220KB |

For datasets under ~10,000 vectors, WASM performance is suitable for interactive use. For larger datasets, use the Python or Node.js bindings.

## Use Cases

- **Client-side semantic search** — search embeddings without a backend
- **Offline-first apps** — works without network after loading
- **Prototyping** — test vector search logic in the browser console
- **Edge/serverless** — run in Cloudflare Workers, Deno Deploy, etc.
