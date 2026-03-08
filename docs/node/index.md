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
