[vctrs](README.md) / VctrsDatabase

Defined in: [index.d.ts:85](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L85)

A persistent vector database backed by an HNSW index.

Stores vectors with string IDs and optional JSON metadata.
Supports cosine, euclidean (L2), and dot-product distance metrics.
Uses memory-mapped I/O for fast loading and a write-ahead log for
crash safety.

## Example

```typescript
// Create a new database
const db = new VctrsDatabase("./my_db", 384, "cosine");
db.add("doc1", embedding, { title: "Hello" });
const results = db.search(queryVector, 5);
db.save();

// Open an existing database (auto-detects dim and metric)
const db2 = new VctrsDatabase("./my_db");
```

## Constructors

### Constructor

```ts
new VctrsDatabase(
   path, 
   dim?, 
   metric?, 
   hnswM?, 
   efConstruction?, 
   quantize?): VctrsDatabase;
```

Defined in: [index.d.ts:104](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L104)

Open or create a database.

When opening an existing database, `dim` and `metric` are
auto-detected — pass `null` or omit them.

#### Parameters

| Parameter | Type | Description |
| ------ | ------ | ------ |
| `path` | `string` | Directory path for the database files. |
| `dim?` | `number` | Vector dimensionality (e.g. 384). Required for new databases. |
| `metric?` | `string` | Distance metric: "cosine" (default), "euclidean"/"l2", or "dot"/"dot_product". |
| `hnswM?` | `number` | Max edges per HNSW node (default 16). Higher improves recall at the cost of memory. |
| `efConstruction?` | `number` | Build-time search width (default 200). Higher improves index quality at the cost of build time. |
| `quantize?` | `boolean` | Enable SQ8 scalar quantization for faster search. |

#### Returns

`VctrsDatabase`

#### Throws

If opening an existing database without `dim` and it doesn't exist,
  or if `metric` is invalid.

## Accessors

### deletedCount

#### Get Signature

```ts
get deletedCount(): number;
```

Defined in: [index.d.ts:300](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L300)

Number of deleted slots not yet reclaimed by `compact()`.

##### Returns

`number`

***

### dim

#### Get Signature

```ts
get dim(): number;
```

Defined in: [index.d.ts:309](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L309)

The vector dimensionality of this database.

##### Returns

`number`

***

### length

#### Get Signature

```ts
get length(): number;
```

Defined in: [index.d.ts:306](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L306)

Number of active vectors in the database.

##### Returns

`number`

***

### metric

#### Get Signature

```ts
get metric(): string;
```

Defined in: [index.d.ts:312](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L312)

The distance metric: "cosine", "euclidean", or "dot_product".

##### Returns

`string`

***

### quantizedSearch

#### Get Signature

```ts
get quantizedSearch(): boolean;
```

Defined in: [index.d.ts:297](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L297)

Whether quantized search is currently enabled.

##### Returns

`boolean`

***

### totalSlots

#### Get Signature

```ts
get totalSlots(): number;
```

Defined in: [index.d.ts:303](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L303)

Total allocated slots (active + deleted).

##### Returns

`number`

## Methods

### add()

```ts
add(
   id, 
   vector, 
   metadata?): void;
```

Defined in: [index.d.ts:123](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L123)

Add a vector with a unique string ID.

#### Parameters

| Parameter | Type | Description |
| ------ | ------ | ------ |
| `id` | `string` | Unique identifier. Throws if the ID already exists (use `upsert()` to insert-or-update). |
| `vector` | `number`\[\] | The embedding vector. Must match the database's dimensionality. |
| `metadata?` | `Record`\<`string`, `any`\> | Optional JSON-serializable metadata object. |

#### Returns

`void`

#### Throws

If the ID already exists or dimension doesn't match.

***

### addAsync()

```ts
addAsync(
   id, 
   vector, 
metadata?): Promise<void>;
```

Defined in: [index.d.ts:317](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L317)

Async version of `add()`.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `id` | `string` |
| `vector` | `number`\[\] |
| `metadata?` | `Record`\<`string`, `any`\> |

#### Returns

`Promise`\<`void`\>

***

### addMany()

```ts
addMany(
   ids, 
   vectors, 
   metadatas?): void;
```

Defined in: [index.d.ts:142](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L142)

Batch insert multiple vectors.

#### Parameters

| Parameter | Type | Description |
| ------ | ------ | ------ |
| `ids` | `string`\[\] | Array of unique identifiers. |
| `vectors` | `number`\[\]\[\] | Array of embedding vectors. |
| `metadatas?` | `Record`\<`string`, `any`\>\[\] | Optional array of metadata objects. |

#### Returns

`void`

#### Throws

If any ID already exists, lengths mismatch, or dimensions don't match.

***

### addManyAsync()

```ts
addManyAsync(
   ids, 
   vectors, 
metadatas?): Promise<void>;
```

Defined in: [index.d.ts:323](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L323)

Async version of `addMany()`.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `ids` | `string`\[\] |
| `vectors` | `number`\[\]\[\] |
| `metadatas?` | `Record`\<`string`, `any`\>\[\] |

#### Returns

`Promise`\<`void`\>

***

### close()

```ts
close(): void;
```

Defined in: [index.d.ts:271](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L271)

Close the database and save to disk. Alias for `save()`.

#### Returns

`void`

#### Throws

On I/O errors.

***

### compact()

```ts
compact(): void;
```

Defined in: [index.d.ts:282](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L282)

Rebuild the index with only live vectors, reclaiming deleted slots.
Call after many deletions to reduce memory and disk usage.

#### Returns

`void`

#### Throws

On internal errors.

***

### contains()

```ts
contains(id): boolean;
```

Defined in: [index.d.ts:251](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L251)

Check if a vector ID exists in the database.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `id` | `string` |

#### Returns

`boolean`

***

### count()

```ts
count(whereFilter?): number;
```

Defined in: [index.d.ts:216](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L216)

Count vectors matching a filter, or all vectors if no filter.

Uses the inverted metadata index for fast counting with equality
and `$in` filters.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `whereFilter?` | [`WhereFilter`](TypeAlias.WhereFilter.md) |

#### Returns

`number`

***

### delete()

```ts
delete(id): boolean;
```

Defined in: [index.d.ts:234](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L234)

Delete a vector by ID.

The slot is marked as deleted but not reclaimed until `compact()`
is called.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `id` | `string` |

#### Returns

`boolean`

`true` if found and deleted, `false` if not found.

#### Throws

On internal errors (e.g. WAL write failure).

***

### deleteAsync()

```ts
deleteAsync(id): Promise<boolean>;
```

Defined in: [index.d.ts:358](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L358)

Async version of `delete()`.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `id` | `string` |

#### Returns

`Promise`\<`boolean`\>

***

### disableQuantizedSearch()

```ts
disableQuantizedSearch(): void;
```

Defined in: [index.d.ts:294](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L294)

Disable quantized search and use full-precision vectors.

#### Returns

`void`

***

### enableQuantizedSearch()

```ts
enableQuantizedSearch(): void;
```

Defined in: [index.d.ts:291](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L291)

Enable SQ8 quantized search for faster HNSW traversal.

Quantized vectors use 4x less memory for distance comparisons
during graph traversal, with full-precision re-ranking of final
candidates.

#### Returns

`void`

***

### get()

```ts
get(id): GetResult;
```

Defined in: [index.d.ts:223](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L223)

Retrieve a vector and its metadata by ID.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `id` | `string` |

#### Returns

[`GetResult`](Interface.GetResult.md)

#### Throws

If the ID is not found.

***

### ids()

```ts
ids(): string\[\];
```

Defined in: [index.d.ts:254](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L254)

Return an array of all vector IDs in the database.

#### Returns

`string`\[\]

***

### save()

```ts
save(): void;
```

Defined in: [index.d.ts:264](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L264)

Persist the database to disk.

Writes the HNSW graph, metadata, and vectors, then truncates the
write-ahead log.

#### Returns

`void`

#### Throws

On I/O errors (e.g. disk full, permission denied).

***

### saveAsync()

```ts
saveAsync(): Promise<void>;
```

Defined in: [index.d.ts:355](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L355)

Async version of `save()`.

#### Returns

`Promise`\<`void`\>

***

### search()

```ts
search(
   vector, 
   k?, 
   efSearch?, 
   whereFilter?, 
   maxDistance?): SearchResult\[\];
```

Defined in: [index.d.ts:186](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L186)

Find the k nearest neighbors to a query vector.

#### Parameters

| Parameter | Type | Description |
| ------ | ------ | ------ |
| `vector` | `number`\[\] | The query embedding. |
| `k?` | `number` | Number of results to return (default 10). |
| `efSearch?` | `number` | HNSW search-time width. Higher improves recall. |
| `whereFilter?` | [`WhereFilter`](TypeAlias.WhereFilter.md) | Optional metadata filter. See [WhereFilter](TypeAlias.WhereFilter.md). |
| `maxDistance?` | `number` | Optional distance threshold — discard results beyond this distance. |

#### Returns

[`SearchResult`](Interface.SearchResult.md)\[\]

Array of SearchResult objects sorted by distance (ascending).

#### Throws

If the vector dimension doesn't match.

#### Example

```typescript
// Basic search
const results = db.search(queryVec, 5);

// With metadata filter
const results = db.search(queryVec, 10, null, { category: "science" });

// With distance threshold
const results = db.search(queryVec, 10, null, null, 0.5);
```

***

### searchAsync()

```ts
searchAsync(
   vector, 
   k?, 
   efSearch?, 
   whereFilter?, 
maxDistance?): Promise<SearchResult\[\]>;
```

Defined in: [index.d.ts:337](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L337)

Async version of `search()`.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `vector` | `number`\[\] |
| `k?` | `number` |
| `efSearch?` | `number` |
| `whereFilter?` | [`WhereFilter`](TypeAlias.WhereFilter.md) |
| `maxDistance?` | `number` |

#### Returns

`Promise`\<[`SearchResult`](Interface.SearchResult.md)[]\>

***

### searchMany()

```ts
searchMany(
   vectors, 
   k?, 
   efSearch?, 
   whereFilter?, 
   maxDistance?): SearchResult\[\]\[\];
```

Defined in: [index.d.ts:202](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L202)

Search multiple queries in parallel.

Significantly faster than calling `search()` in a loop.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `vectors` | `number`\[\]\[\] |
| `k?` | `number` |
| `efSearch?` | `number` |
| `whereFilter?` | [`WhereFilter`](TypeAlias.WhereFilter.md) |
| `maxDistance?` | `number` |

#### Returns

[`SearchResult`](Interface.SearchResult.md)\[\]\[\]

Array of result arrays, one per query.

#### Throws

If any vector dimension doesn't match.

***

### searchManyAsync()

```ts
searchManyAsync(
   vectors, 
   k?, 
   efSearch?, 
   whereFilter?, 
maxDistance?): Promise<SearchResult\[\]\[\]>;
```

Defined in: [index.d.ts:346](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L346)

Async version of `searchMany()`.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `vectors` | `number`\[\]\[\] |
| `k?` | `number` |
| `efSearch?` | `number` |
| `whereFilter?` | [`WhereFilter`](TypeAlias.WhereFilter.md) |
| `maxDistance?` | `number` |

#### Returns

`Promise`\<[`SearchResult`](Interface.SearchResult.md)\[\]\[\]\>

***

### stats()

```ts
stats(): VctrsStats;
```

Defined in: [index.d.ts:274](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L274)

Get graph-level statistics for diagnostics.

#### Returns

[`VctrsStats`](Interface.VctrsStats.md)

***

### update()

```ts
update(
   id, 
   vector?, 
   metadata?): void;
```

Defined in: [index.d.ts:244](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L244)

Update a vector's embedding and/or metadata in-place.

#### Parameters

| Parameter | Type | Description |
| ------ | ------ | ------ |
| `id` | `string` | The vector's unique identifier. |
| `vector?` | `number`\[\] | New embedding (or null to keep existing). |
| `metadata?` | `Record`\<`string`, `any`\> | New metadata (or null to keep existing). |

#### Returns

`void`

#### Throws

If the ID is not found or the vector dimension doesn't match.

***

### upsert()

```ts
upsert(
   id, 
   vector, 
   metadata?): void;
```

Defined in: [index.d.ts:132](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L132)

Insert a vector, or update it if the ID already exists.

Same as `add()` but overwrites existing entries instead of throwing.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `id` | `string` |
| `vector` | `number`\[\] |
| `metadata?` | `Record`\<`string`, `any`\> |

#### Returns

`void`

#### Throws

If the vector dimension doesn't match.

***

### upsertAsync()

```ts
upsertAsync(
   id, 
   vector, 
metadata?): Promise<void>;
```

Defined in: [index.d.ts:320](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L320)

Async version of `upsert()`.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `id` | `string` |
| `vector` | `number`\[\] |
| `metadata?` | `Record`\<`string`, `any`\> |

#### Returns

`Promise`\<`void`\>

***

### upsertMany()

```ts
upsertMany(
   ids, 
   vectors, 
   metadatas?): void;
```

Defined in: [index.d.ts:156](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L156)

Batch upsert — inserts new vectors, updates existing ones.

More efficient than calling `upsert()` in a loop because new
vectors are batch-inserted together.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `ids` | `string`\[\] |
| `vectors` | `number`\[\]\[\] |
| `metadatas?` | `Record`\<`string`, `any`\>\[\] |

#### Returns

`void`

#### Throws

If lengths mismatch or dimensions don't match.

***

### upsertManyAsync()

```ts
upsertManyAsync(
   ids, 
   vectors, 
metadatas?): Promise<void>;
```

Defined in: [index.d.ts:330](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L330)

Async version of `upsertMany()`.

#### Parameters

| Parameter | Type |
| ------ | ------ |
| `ids` | `string`\[\] |
| `vectors` | `number`\[\]\[\] |
| `metadatas?` | `Record`\<`string`, `any`\>\[\] |

#### Returns

`Promise`\<`void`\>
