# Node.js — Getting Started

## Install

```bash
npm install @yang-29/vctrs
```

## Basic Usage

```typescript
import { VctrsDatabase } from "@yang-29/vctrs";

// Create a new database (or open existing)
const db = new VctrsDatabase("./mydb", 384, "cosine");

// Add vectors
db.add("doc1", [0.1, 0.2, ...], { title: "Hello" });

// Batch insert
db.addMany(ids, vectors, metadatas);

// Search
const results = db.search(queryVector, 10);
for (const r of results) {
  console.log(r.id, r.distance, r.metadata);
}

// Filtered search
const filtered = db.search(queryVec, 10, null, { category: "science" });

// Distance threshold
const close = db.search(queryVec, 10, null, null, 0.5);

// Save
db.save();
```

## Async Usage

All methods have async variants that run on the libuv thread pool:

```typescript
// Non-blocking — doesn't stall the event loop
await db.addAsync("doc1", vector, { title: "Hello" });
const results = await db.searchAsync(queryVector, 10);
await db.saveAsync();
```

## Opening Existing Databases

```typescript
// Auto-detects dim and metric from saved files
const db = new VctrsDatabase("./mydb");
console.log(db.dim, db.metric); // 384, "cosine"
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | required | Vector dimensionality |
| `metric` | `"cosine"` | `"cosine"`, `"euclidean"`/`"l2"`, `"dot"`/`"dot_product"` |
| `hnswM` | `16` | HNSW edges per node |
| `efConstruction` | `200` | Build-time search width |
| `quantize` | `false` | Enable SQ8 scalar quantization |

## Collections

Manage multiple isolated indexes under one directory:

```typescript
import { VctrsClient } from "@yang-29/vctrs";

const client = new VctrsClient("./data");

// Create collections with different configs
const movies = client.createCollection("movies", 384);
const docs = client.createCollection("docs", 768, "dot");

// Use them like regular databases
movies.add("m1", vector, { title: "Alien" });
movies.save();

// List, get, delete
client.listCollections();            // → ["docs", "movies"]
const db = client.getCollection("movies");
const items = client.getOrCreateCollection("items", 384);
client.deleteCollection("docs");
```

## Export / Import

Backup and restore databases as JSON:

```typescript
// Export to file
db.exportJson("backup.json");
db.exportJson("backup.json", true);  // pretty-print

// Import into existing database (upsert semantics)
db.importJson("backup.json");
```

The JSON format includes dimension, metric, and all vectors with metadata.
