# @yang-29/vctrs-wasm

A fast vector database that runs in the browser via WebAssembly. ~220KB, zero dependencies.

Same Rust core as the [Python](https://pypi.org/project/vctrs/) and [Node.js](https://www.npmjs.com/package/@yang-29/vctrs) bindings.

## Install

```bash
npm install @yang-29/vctrs-wasm
```

## Usage

```javascript
import init, { VctrsDatabase } from "@yang-29/vctrs-wasm";

await init();

const db = new VctrsDatabase(384, "cosine");

db.add("doc1", new Float32Array(embedding), { title: "hello" });
db.add("doc2", new Float32Array(embedding2), { title: "world" });

const results = db.search(new Float32Array(query), 5);
for (const r of results) {
  console.log(r.id, r.distance, r.metadata);
  r.free();
}

db.free();
```

## CDN (no bundler)

```html
<script type="module">
import init, { VctrsDatabase } from "https://unpkg.com/@yang-29/vctrs-wasm/vctrs_wasm.js";
await init();
const db = new VctrsDatabase(4, "cosine");
</script>
```

## API

| Method | Description |
|--------|-------------|
| `new VctrsDatabase(dim, metric)` | Create database. Metrics: `"cosine"`, `"euclidean"`, `"dot"` |
| `db.add(id, vector, metadata)` | Add vector (Float32Array) with optional metadata |
| `db.upsert(id, vector, metadata)` | Insert or update |
| `db.search(vector, k)` | Find k nearest neighbors → `SearchResult[]` |
| `db.delete(id)` | Delete by ID → boolean |
| `db.contains(id)` | Check existence → boolean |
| `db.length` | Vector count |
| `db.dim` | Dimensionality |

`SearchResult` has `.id`, `.distance`, `.metadata` properties. Call `.free()` when done.

## Notes

- Runs entirely in-memory — no persistence across page reloads
- Call `.free()` on results and the database to avoid memory leaks
- Single-threaded (no SIMD/BLAS) — suitable for datasets under ~10k vectors
