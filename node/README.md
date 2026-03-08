# @yang-29/vctrs

A fast embedded vector database for Node.js. Native Rust core, no server required.

Same engine as the [Python](https://pypi.org/project/vctrs/) and [WASM](https://www.npmjs.com/package/@yang-29/vctrs-wasm) bindings.

## Install

```bash
npm install @yang-29/vctrs
```

## Usage

```typescript
import { VctrsDatabase } from "@yang-29/vctrs";

// Create or open a database
const db = new VctrsDatabase("./mydb", 384, "cosine");

// Add vectors
db.add("doc1", embedding, { title: "hello" });
db.add("doc2", embedding2, { title: "world" });

// Search
const results = db.search(queryVector, 10);
for (const r of results) {
  console.log(r.id, r.distance, r.metadata);
}

// Filtered search
const filtered = db.search(queryVector, 10, null, { category: "science" });

// Save to disk
db.save();
```

## Async

All methods have async variants that run on the libuv thread pool:

```typescript
await db.addAsync("doc1", vector, { title: "hello" });
const results = await db.searchAsync(queryVector, 10);
await db.saveAsync();
```

## API

| Method | Description |
|--------|-------------|
| `new VctrsDatabase(path, dim, metric)` | Create or open database |
| `new VctrsDatabase(path)` | Open existing (auto-detects dim/metric) |
| `db.add(id, vector, metadata?)` | Add vector |
| `db.addMany(ids, vectors, metadatas?)` | Batch add |
| `db.upsert(id, vector, metadata?)` | Insert or update |
| `db.upsertMany(ids, vectors, metadatas?)` | Batch upsert |
| `db.search(vector, k, efSearch?, where?, maxDistance?)` | Find k nearest neighbors |
| `db.searchMany(vectors, k, efSearch?, where?, maxDistance?)` | Batch search |
| `db.delete(id)` | Delete by ID |
| `db.deleteMany(ids)` | Batch delete |
| `db.get(id)` | Get vector and metadata |
| `db.contains(id)` | Check existence |
| `db.count(where?)` | Count vectors (with optional filter) |
| `db.save()` | Persist to disk |
| `db.stats()` | Index statistics |
| `db.length` | Vector count |
| `db.dim` | Dimensionality |
| `db.metric` | Distance metric |

## Filter Operators

```typescript
db.search(query, 10, null, {
  category: "science",           // $eq (default)
  year: { $gt: 2020 },           // $gt, $gte, $lt, $lte
  status: { $ne: "draft" },      // $ne
  tag: { $in: ["a", "b"] },      // $in
});
```

## Performance

10,000 vectors, 384 dimensions, cosine, Apple M-series:

| Operation | vctrs | ChromaDB |
|-----------|-------|----------|
| Search k=10 | **0.14ms** | 0.93ms |
| Insert 10k | **518ms** | 2367ms |
| Load from disk | **1.2ms** | 1.6ms |
